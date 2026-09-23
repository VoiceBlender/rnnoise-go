//go:build arm64

package rnnoise

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestDotProdAvailable states plainly whether this machine can run the fast
// path, so a pass elsewhere in the suite cannot be mistaken for the assembly
// having been exercised.
func TestDotProdAvailable(t *testing.T) {
	t.Logf("HasASIMD=%v HasASIMDDP=%v", useASIMD, useDotProd)
	if !useDotProd {
		t.Skip("this core has no ARMv8.2 DotProd, so cgemvInt8 uses the generic path here; " +
			"the SDOT kernel is NOT covered by this run")
	}
}

// TestSDOTKernelDirect calls the assembly directly rather than through the
// dispatcher.
//
// Going through cgemvInt8 would let any guard -- a missing CPU feature, a shape
// check, an out-of-range activation -- silently route to the generic kernel, and
// the differential test would then compare generic against generic and pass
// without the assembly ever running. That is the failure mode that makes such a
// test worthless, so this one removes the possibility.
func TestSDOTKernelDirect(t *testing.T) {
	if !useDotProd {
		t.Skip("no DotProd on this core")
	}
	rng := rand.New(rand.NewSource(99))
	for _, sh := range []struct{ rows, cols int }{
		{384, 384},  // conv2
		{1152, 384}, // a GRU gate matrix
		{8, 4},      // the smallest legal shape
		{8, 8},
		{16, 12},
		{24, 20},
		{40, 4},
	} {
		for _, extreme := range []bool{false, true} {
			w := randWeights(rng, sh.rows, sh.cols, extreme)
			_, q := randActivations(rng, sh.cols, QuantSigned, extreme)
			scale := make([]float32, sh.rows)
			for i := range scale {
				scale[i] = float32(rng.Float64()*0.02 + 1e-4)
			}

			want := make([]float32, sh.rows)
			cgemvInt8Generic(want, w, scale, sh.rows, sh.cols, q)

			q8 := make([]int8, sh.cols)
			for i, v := range q {
				q8[i] = int8(v)
			}
			acc := make([]int32, sh.rows)
			cgemvInt8SDOT(acc, w, q8, sh.rows, sh.cols)
			got := make([]float32, sh.rows)
			for i := range got {
				got[i] = float32(acc[i]) * scale[i]
			}

			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("rows=%d cols=%d extreme=%v: out[%d] = %v, generic %v",
						sh.rows, sh.cols, extreme, i, got[i], want[i])
				}
			}
		}
	}
}

// TestSgemvFMADirect likewise calls the float kernel directly.
//
// It is compared against the generic Go path rather than against an unfused
// reference: on arm64 the compiler contracts the generic code into FMADDS too,
// so both sides fuse and the comparison is still exact. See exactarch.go.
func TestSgemvFMADirect(t *testing.T) {
	if !useASIMD {
		t.Skip("no ASIMD on this core")
	}
	rng := rand.New(rand.NewSource(1234))
	for _, sh := range []struct{ rows, cols int }{
		{128, 195}, {32, 1536}, {8, 1}, {8, 1536}, {24, 7}, {32, 3}, {40, 11}, {64, 5}, {256, 65},
	} {
		for _, stride := range []int{sh.rows, sh.rows + 3} {
			w := make([]float32, sh.cols*stride)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			x := make([]float32, sh.cols)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			want := make([]float32, sh.rows)
			got := make([]float32, sh.rows)
			sgemvGeneric(want, w, sh.rows, sh.cols, stride, x)
			sgemvFMA(got, w, sh.rows, sh.cols, stride, x)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("rows=%d cols=%d stride=%d: out[%d] = %v, generic %v",
						sh.rows, sh.cols, stride, i, got[i], want[i])
				}
			}
		}
	}
}

// TestSDOTEncoding re-derives every hand-encoded instruction in simd_arm64.s
// from the documented formula and checks the source against it.
//
// Go's assembler cannot validate a WORD, so a mistyped digit would assemble
// cleanly and then compute something else. This reads the assembly back and
// checks each literal against the encoding its own comment claims.
func TestSDOTEncoding(t *testing.T) {
	// SDOT Vd.4S, Vn.16B, Vm.16B, from `as -march=armv8.6-a` (see simd_arm64.go).
	sdot := func(rd, rn, rm uint32) uint32 { return 0x4E809400 | rm<<16 | rn<<5 | rd }

	f, err := os.Open("simd_arm64.s")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// WORD $0x4E829410 // SDOT V16.4S, V0.16B, V2.16B
	line := regexp.MustCompile(`WORD\s+\$0x([0-9A-Fa-f]+)\s*//\s*SDOT\s+V(\d+)\.4S,\s*V(\d+)\.16B,\s*V(\d+)\.16B`)
	found := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		txt := sc.Text()
		trimmed := strings.TrimSpace(txt)
		// Skip prose: the file's own comments discuss WORD encoding.
		if strings.HasPrefix(trimmed, "//") || !strings.HasPrefix(trimmed, "WORD") {
			continue
		}
		m := line.FindStringSubmatch(txt)
		if m == nil {
			t.Errorf("hand-encoded instruction is not in the checkable form "+
				"`WORD $0xXXXXXXXX // SDOT Vd.4S, Vn.16B, Vm.16B`: %s", strings.TrimSpace(txt))
			continue
		}
		got, err := strconv.ParseUint(m[1], 16, 32)
		if err != nil {
			t.Fatal(err)
		}
		rd, _ := strconv.Atoi(m[2])
		rn, _ := strconv.Atoi(m[3])
		rm, _ := strconv.Atoi(m[4])
		want := sdot(uint32(rd), uint32(rn), uint32(rm))
		if uint32(got) != want {
			t.Errorf("%s: encoded 0x%08X, but SDOT V%d.4S, V%d.16B, V%d.16B is 0x%08X",
				strings.TrimSpace(txt), got, rd, rn, rm, want)
		}
		found++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if found == 0 {
		t.Error("no hand-encoded SDOT found in simd_arm64.s; has the kernel changed?")
	}
	fmt.Fprintf(os.Stderr, "checked %d hand-encoded SDOT instructions\n", found)
}

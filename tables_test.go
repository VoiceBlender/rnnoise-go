package rnnoise

import (
	"bufio"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"
)

// upstreamTables holds the 48 kHz tables extracted verbatim from
// src/rnnoise_tables.c at 70f1d256, which upstream itself generated with
// dump_rnnoise_tables.c. Comparing against them needs no C toolchain and no
// model download, so the strongest correctness check in the package runs in
// plain CI.
type upstreamTables struct {
	nfft       int
	scale      float32
	shift      int
	factors    []int
	bitrev     []int32
	twiddles   []cpx
	halfWindow []float32
	dctTable   []float32
}

func loadUpstreamTables(t *testing.T) *upstreamTables {
	t.Helper()
	f, err := os.Open("testdata/upstream_tables_48k.txt")
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()

	ut := &upstreamTables{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<22)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
			continue
		}
		key, rest, _ := strings.Cut(line, " ")
		fields := strings.Fields(rest)
		switch key {
		case "nfft":
			ut.nfft = mustAtoi(t, fields[0])
		case "shift":
			ut.shift = mustAtoi(t, fields[0])
		case "scale":
			ut.scale = mustF32(t, fields[0])
		case "factors":
			for _, s := range fields {
				ut.factors = append(ut.factors, mustAtoi(t, s))
			}
		case "bitrev":
			for _, s := range fields {
				ut.bitrev = append(ut.bitrev, int32(mustAtoi(t, s)))
			}
		case "twiddles":
			for i := 0; i+1 < len(fields); i += 2 {
				ut.twiddles = append(ut.twiddles, cpx{mustF32(t, fields[i]), mustF32(t, fields[i+1])})
			}
		case "half_window":
			for _, s := range fields {
				ut.halfWindow = append(ut.halfWindow, mustF32(t, s))
			}
		case "dct_table":
			for _, s := range fields {
				ut.dctTable = append(ut.dctTable, mustF32(t, s))
			}
		default:
			t.Fatalf("unknown fixture key %q", key)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan fixture: %v", err)
	}
	return ut
}

func mustAtoi(t *testing.T, s string) int {
	t.Helper()
	v, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("atoi %q: %v", s, err)
	}
	return v
}

func mustF32(t *testing.T, s string) float32 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 32)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return float32(v)
}

// TestUpstreamTables48k is the null test for the whole table-generation path:
// everything we compute at runtime must reproduce upstream's committed 48 kHz
// tables with exact float32 equality, not merely to within a tolerance. It
// covers the FFT factorisation, the bit-reversal permutation, the twiddles
// (and with them Go's math.Sincos rounding against glibc's), the analysis
// window and the DCT matrix.
func TestUpstreamTables48k(t *testing.T) {
	ut := loadUpstreamTables(t)

	st, ok := newFFTState(refWindow)
	if !ok {
		t.Fatal("newFFTState(960) failed")
	}

	if st.nfft != ut.nfft {
		t.Errorf("nfft = %d, upstream %d", st.nfft, ut.nfft)
	}
	if st.scale != ut.scale {
		t.Errorf("scale = %v, upstream %v", st.scale, ut.scale)
	}
	// Upstream's committed shift is -1: the table was generated with
	// base == NULL, exactly as newFFTState builds it.
	if st.shift != ut.shift {
		t.Errorf("shift = %d, upstream %d", st.shift, ut.shift)
	}
	for i, want := range ut.factors {
		if got := st.factors[i]; got != want {
			t.Errorf("factors[%d] = %d, upstream %d", i, got, want)
		}
	}
	for i, want := range ut.bitrev {
		if got := st.bitrev[i]; got != want {
			t.Fatalf("bitrev[%d] = %d, upstream %d", i, got, want)
		}
	}
	var twDiff int
	for i, want := range ut.twiddles {
		if got := st.twiddles[i]; got != want {
			if twDiff++; twDiff <= 5 {
				t.Errorf("twiddles[%d] = {%v,%v}, upstream {%v,%v}", i, got.r, got.i, want.r, want.i)
			}
		}
	}
	if twDiff > 0 {
		t.Errorf("%d/%d twiddles differ from upstream", twDiff, len(ut.twiddles))
	}

	hw := makeHalfWindow(refFrame)
	for i, want := range ut.halfWindow {
		if got := hw[i]; got != want {
			t.Fatalf("half_window[%d] = %v, upstream %v", i, got, want)
		}
	}

	dct := makeDCTTable()
	for i, want := range ut.dctTable {
		if got := dct[i]; got != want {
			t.Fatalf("dct_table[%d] = %v, upstream %v", i, got, want)
		}
	}
}

// naiveDFT is the 1/N-scaled forward transform the butterflies must implement.
func naiveDFT(in []cpx) []cpx {
	n := len(in)
	out := make([]cpx, n)
	for k := 0; k < n; k++ {
		var acc complex128
		for j := 0; j < n; j++ {
			ang := -2 * math.Pi * float64(k) * float64(j) / float64(n)
			acc += complex(float64(in[j].r), float64(in[j].i)) * cmplx.Exp(complex(0, ang))
		}
		acc /= complex(float64(n), 0)
		out[k] = cpx{float32(real(acc)), float32(imag(acc))}
	}
	return out
}

// TestFFTAgainstDFT checks every radix path, including the generic butterfly
// that upstream has no equivalent for (7 for 44.1 kHz, 11 for 22.05 kHz) and
// the degenerate radix-2 stage that upstream compiles out.
func TestFFTAgainstDFT(t *testing.T) {
	sizes := []int{
		16, 20, 24, 40, 60, 120, // the plain 2/3/4/5 paths
		160, 240, 320, 480, 640, 960, // every supported window at rate%100==0
		882,  // 44.1 kHz: 2*3^2*7^2, exercises radix 7 and radix 2 with m==1
		440,  // 22.05 kHz: 2^3*5*11, exercises radix 11
		1920, // 96 kHz
	}
	rng := rand.New(rand.NewSource(1))
	for _, n := range sizes {
		st, ok := newFFTState(n)
		if !ok {
			t.Errorf("newFFTState(%d) failed", n)
			continue
		}
		in := make([]cpx, n)
		for i := range in {
			in[i] = cpx{float32(rng.NormFloat64()), float32(rng.NormFloat64())}
		}
		got := make([]cpx, n)
		st.fft(in, got)
		want := naiveDFT(in)

		var maxErr float64
		for i := range want {
			dr := float64(got[i].r - want[i].r)
			di := float64(got[i].i - want[i].i)
			if e := math.Hypot(dr, di); e > maxErr {
				maxErr = e
			}
		}
		// The reference accumulates in float64; float32 butterflies over n
		// points carry O(log n) relative error on a unit-RMS signal scaled by
		// 1/n, so this bound is loose enough to be stable and tight enough to
		// catch a wrong twiddle index or a mis-stridden butterfly.
		if maxErr > 1e-6 {
			t.Errorf("n=%d: max abs error %g vs float64 DFT", n, maxErr)
		}
	}
}

func TestFFTRejectsLargePrime(t *testing.T) {
	// 2*37: 37 exceeds maxGenericRadix, so the caller must fall back.
	if _, ok := newFFTState(74); ok {
		t.Error("newFFTState(74) unexpectedly succeeded; 37 > maxGenericRadix")
	}
}

// TestBluesteinAgainstDFT checks the fallback transform on the sizes that need
// it, and on sizes the mixed-radix path also handles so the two can be compared
// directly.
func TestBluesteinAgainstDFT(t *testing.T) {
	sizes := []int{
		164,    // 8200 Hz: 4*41, the smallest awkward window that is a real rate
		188,    // 9400 Hz: 4*47
		2 * 53, // a bare large prime doubled
		320,    // also reachable natively, for a cross-check
		960,    // the 48 kHz size, natively factorable
		882,    // 44.1 kHz
	}
	rng := rand.New(rand.NewSource(31))
	for _, n := range sizes {
		bs, ok := newBluesteinState(n)
		if !ok {
			t.Errorf("newBluesteinState(%d) failed", n)
			continue
		}
		if bs.size() != n {
			t.Errorf("n=%d: size() = %d", n, bs.size())
		}
		in := make([]cpx, n)
		for i := range in {
			in[i] = cpx{float32(rng.NormFloat64()), float32(rng.NormFloat64())}
		}
		got := make([]cpx, n)
		bs.fft(in, got)
		want := naiveDFT(in)

		var maxErr, refMax float64
		for i := range want {
			if m := math.Hypot(float64(want[i].r), float64(want[i].i)); m > refMax {
				refMax = m
			}
			dr := float64(got[i].r - want[i].r)
			di := float64(got[i].i - want[i].i)
			if e := math.Hypot(dr, di); e > maxErr {
				maxErr = e
			}
		}
		// Bluestein carries more rounding than a native transform: two
		// transforms of a longer size plus the chirp multiplies. The bound is
		// relative to the spectrum's own magnitude.
		if rel := maxErr / refMax; rel > 1e-5 {
			t.Errorf("n=%d: max error %g, %g relative to peak magnitude %g", n, maxErr, rel, refMax)
		}
	}
}

// TestBluesteinRatesResolve checks that a rate the mixed-radix path rejects is
// still accepted by New, which is the reason Bluestein exists.
func TestBluesteinRatesResolve(t *testing.T) {
	for _, rate := range []int{8200, 9400, 10600} {
		c, err := configForRate(rate)
		if err != nil {
			t.Errorf("rate %d: %v", rate, err)
			continue
		}
		if _, native := newFFTState(c.window); native {
			t.Logf("rate %d (window %d) is natively factorable after all", rate, c.window)
			continue
		}
		if _, isB := c.fft.(*bluesteinState); !isB {
			t.Errorf("rate %d (window %d) should have fallen back to Bluestein", rate, c.window)
		}
	}
}

// TestRefRateNeverUsesBluestein guards the property bit-exactness depends on:
// 48 kHz must always take the mixed-radix path, whose arithmetic order matches
// upstream's kiss_fft. Bluestein is a different algorithm and could not be
// bit-exact against C.
func TestRefRateNeverUsesBluestein(t *testing.T) {
	for _, rate := range testRates {
		c, err := configForRate(rate)
		if err != nil {
			t.Fatal(err)
		}
		if _, isB := c.fft.(*bluesteinState); isB {
			t.Errorf("rate %d unexpectedly uses Bluestein; every common rate must be "+
				"natively factorable", rate)
		}
	}
}

//go:build arm64

package rnnoise

import "golang.org/x/sys/cpu"

// Runtime dispatch for the arm64 kernels, mirroring the amd64 layout: simd.go
// holds the portable generic kernels and the fuzz-test oracle, this file gates
// the assembly on CPU features and on length, simd_arm64.s holds the assembly.
//
// Two notes specific to this architecture.
//
// First, Go's arm64 assembler has no vector integer multiply at all -- MUL and
// SMULL are scalar, VPMULL is polynomial, and neither SDOT nor SMLAL exists -- so
// the int8 dot product is emitted as a raw WORD. The encoding came from a real
// assembler rather than from the manual:
//
//	$ echo 'sdot v16.4s, v0.16b, v2.16b' | as -march=armv8.6-a -o t.o -
//	$ objdump -d t.o
//	  0: 4e829410  sdot v16.4s, v0.16b, v2.16b
//
// giving SDOT Vd.4S, Vn.16B, Vm.16B = 0x4E809400 | Rm<<16 | Rn<<5 | Rd.
// TestSDOTEncoding re-derives the constants in the assembly from that formula, so
// a mistyped digit fails a test rather than corrupting audio.
//
// Second, there is no SDOT fallback worth writing. Without ASIMDDP the only
// route is widening multiply-accumulate, which Go's assembler also lacks, so a
// pre-ARMv8.2 core keeps the generic Go path. ASIMDDP has been present on every
// Cortex-A since the A75 and on all of Apple's arm64 and AWS Graviton 2 onward.
var (
	useDotProd = cpu.ARM64.HasASIMDDP
	useASIMD   = cpu.ARM64.HasASIMD
)

// cgemvInt8SDOT is implemented in simd_arm64.s. It leaves the accumulator in
// int32 for the caller to scale.
//
//go:noescape
func cgemvInt8SDOT(acc []int32, w []int8, q8 []int8, rows int, cols int)

// sgemvFMA is implemented in simd_arm64.s. It requires rows to be a multiple
// of 8.
//
//go:noescape
func sgemvFMA(out []float32, weights []float32, rows int, cols int, colStride int, x []float32)

func cgemvInt8(out []float32, w []int8, scale []float32, rows, cols int, q []int16, acc []int32, q8 []int8) {
	if useDotProd && rows%8 == 0 && cols%4 == 0 && rows > 0 && cols > 0 &&
		len(acc) >= rows && len(q8) >= len(q) && int8Activations(q) {
		acc, q8 = acc[:rows], q8[:len(q)]
		// SDOT wants int8 activations; the shared quantiser widens to int16 for
		// amd64's VPMADDWD. Narrowing costs one pass over 384 values against
		// 442368 multiply-accumulates.
		for i, v := range q {
			q8[i] = int8(v)
		}
		cgemvInt8SDOT(acc, w, q8, rows, cols)
		for i := 0; i < rows; i++ {
			out[i] = float32(acc[i]) * scale[i]
		}
		return
	}
	cgemvInt8Generic(out, w, scale, rows, cols, q)
}

// int8Activations reports whether every quantised activation fits in an int8.
//
// The signed convention keeps them in [-127,127] and does; the unsigned one puts
// them in [0,254] and does not, so QuantUnsigned takes the generic path here. It
// exists only for the differential harness, and USDOT -- which would serve it --
// needs ARMv8.6 rather than 8.2.
func int8Activations(q []int16) bool {
	for _, v := range q {
		if v < -128 || v > 127 {
			return false
		}
	}
	return true
}

func sgemv(out, weights []float32, rows, cols, colStride int, x []float32) {
	if useASIMD && rows%8 == 0 && rows > 0 && cols > 0 {
		sgemvFMA(out, weights, rows, cols, colStride, x)
		return
	}
	sgemvGeneric(out, weights, rows, cols, colStride, x)
}

func sparseSgemv(out, w []float32, idx []int32, rows int, x []float32) {
	sparseSgemvGeneric(out, w, idx, rows, x)
}

func quantize(q []int16, in []float32, mode QuantMode) {
	quantizeGeneric(q, in, mode)
}

func sparseCgemvInt8(out []float32, w []int8, idx []int32, scale []float32, rows, cols int, q []int16) {
	sparseCgemvInt8Generic(out, w, idx, scale, rows, cols, q)
}

func vecTanh(y, x []float32) { vecTanhGeneric(y, x) }

func vecSigmoid(y, x []float32) { vecSigmoidGeneric(y, x) }

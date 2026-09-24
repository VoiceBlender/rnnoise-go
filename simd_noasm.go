//go:build !amd64 && !arm64

package rnnoise

// Portable dispatch: every kernel goes straight to its generic implementation.
// See simd_amd64.go for the vectorised paths.

func sgemv(out, weights []float32, rows, cols, colStride int, x []float32) {
	sgemvGeneric(out, weights, rows, cols, colStride, x)
}

func sparseSgemv(out, w []float32, idx []int32, rows int, x []float32) {
	sparseSgemvGeneric(out, w, idx, rows, x)
}

func quantize(q []int16, in []float32, mode QuantMode) {
	quantizeGeneric(q, in, mode)
}

func cgemvInt8(out []float32, w []int8, scale []float32, rows, cols int, q []int16, _ []int32, _ []int8) {
	cgemvInt8Generic(out, w, scale, rows, cols, q)
}

func sparseCgemvInt8(out []float32, w []int8, idx []int32, scale []float32, rows, cols int, q []int16) {
	sparseCgemvInt8Generic(out, w, idx, scale, rows, cols, q)
}

func vecTanh(y, x []float32) { vecTanhGeneric(y, x) }

func vecSigmoid(y, x []float32) { vecSigmoidGeneric(y, x) }

// No vectorised radix-5 butterfly here; the caller runs the scalar loop.
func bfly5Vec(f0, f1, f2, f3, f4, t1, t2, t3, t4 []cpx, y *[4]float32, blocks int) bool {
	return false
}

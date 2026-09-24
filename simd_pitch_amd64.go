//go:build amd64

package rnnoise

//go:noescape
func xcorrVecAVX2(x, y []float32, acc *[32]float32, length int) int

// xcorrVec fills acc with per-lane partial sums and reports how many elements
// it consumed. Zero means the caller does the whole thing scalar.
//
// The last vector load reads y[length-5 : length+3], so y needs length+3
// elements.
func xcorrVec(x, y []float32, acc *[32]float32, length int) int {
	if !useFMA || length < 8 || len(x) < length || len(y) < length+3 {
		return 0
	}
	return xcorrVecAVX2(x, y, acc, length)
}

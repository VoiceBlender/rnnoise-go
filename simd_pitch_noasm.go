//go:build !amd64

package rnnoise

// No vectorised pitch correlation off amd64; the caller runs the scalar loop.
func xcorrVec(x, y []float32, acc *[32]float32, length int) int { return 0 }

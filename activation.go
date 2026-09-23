package rnnoise

import "math"

// Activation kinds, from src/nnet.h.
const (
	actLinear  = 0
	actSigmoid = 1
	actTanh    = 2
	actRelu    = 3
	actSoftmax = 4
	actSwish   = 5
)

// floor32 is C's floor() applied to a float32, used by the activation
// quantisers. math.Floor takes float64, and the argument is exactly
// representable either way, so there is no double-rounding hazard.
func floor32(v float32) float32 {
	return float32(math.Floor(float64(v)))
}

// tanhApprox is upstream's rational tanh from src/vec.h. The constants are
// float32 in C (an `f` suffix), and Go converts the untyped constants to
// float32 in these expressions for the same reason, so the arithmetic matches.
//
// This is a real division, not a reciprocal approximation. Upstream's AVX path
// uses _mm256_rcp_ps here, whose result is microarchitecture-defined and
// differs between AMD and Intel; matching it is impossible, and matching the
// scalar build is both achievable and more accurate. The cost is that Go can
// never be bit-identical to a stock x86 librnnoise's activations -- only to its
// scalar build, which is the harness's oracle.
//
// Note for arm64: Go's backend may contract these multiply-adds into FMA, where
// upstream's `fmadd` macro expands to an unfused (a)*(b)+(c). Bit-exactness
// against C is therefore an amd64 property.
const (
	tanhN0 = 952.52801514
	tanhN1 = 96.39235687
	tanhN2 = 0.60863042
	tanhD0 = 952.72399902
	tanhD1 = 413.36801147
	tanhD2 = 11.88600922
)

// tanhCoefs holds the same coefficients in the order the assembly kernels
// expect. Sharing the constants with tanhApprox rather than re-encoding them as
// assembler literals is what guarantees the vector and scalar paths round the
// decimal source values to the identical float32.
var tanhCoefs = [8]float32{
	tanhN0, tanhN1, tanhN2,
	tanhD0, tanhD1, tanhD2,
	1, -1,
}

func tanhApprox(x float32) float32 {
	const (
		n0 = tanhN0
		n1 = tanhN1
		n2 = tanhN2
		d0 = tanhD0
		d1 = tanhD1
		d2 = tanhD2
	)
	x2 := x * x
	num := (n2*x2+n1)*x2 + n0
	den := (d2*x2+d1)*x2 + d0
	num = num * x / den
	return maxf(-1, minf(1, num))
}

func sigmoidApprox(x float32) float32 {
	return .5 + .5*tanhApprox(.5*x)
}

func vecTanhGeneric(y, x []float32) {
	for i, v := range x {
		y[i] = tanhApprox(v)
	}
}

func vecSigmoidGeneric(y, x []float32) {
	for i, v := range x {
		y[i] = sigmoidApprox(v)
	}
}

// computeActivation is upstream's compute_activation_c. Only sigmoid and tanh
// are reachable from RNNoise's topology; the rest are ported for completeness
// so a user-supplied model cannot silently get a linear layer where it asked
// for a ReLU.
//
// Upstream's SOFTMAX_HACK is reproduced: softmax is a plain copy, because the
// only caller that used it normalised afterwards.
func computeActivation(out, in []float32, activation int) {
	switch activation {
	case actSigmoid:
		vecSigmoid(out, in)
	case actTanh:
		vecTanh(out, in)
	case actSwish:
		for i, v := range in {
			out[i] = v * sigmoidApprox(v)
		}
	case actRelu:
		for i, v := range in {
			if v < 0 {
				out[i] = 0
			} else {
				out[i] = v
			}
		}
	case actSoftmax:
		copy(out, in)
	default: // actLinear
		if &out[0] != &in[0] {
			copy(out, in)
		}
	}
}

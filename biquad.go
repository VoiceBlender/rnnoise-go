package rnnoise

import "math"

// Upstream's fixed 48 kHz input high-pass, from src/denoise.c:
//
//	static const float a_hp[2] = {-1.99599, 0.99600};
//	static const float b_hp[2] = {-2, 1};
//
// H(z) = (1 - 2z^-1 + z^-2) / (1 - 1.99599 z^-1 + 0.99600 z^-2). The leading
// numerator coefficient is implicit in rnn_biquad. The discriminant
// a0^2 - 4*a1 is negative, so this is not a first-order DC blocker but a
// near-critically-damped second-order high-pass with a complex pole pair at
// ~18.75 Hz, Q ~= 0.61.
var (
	refHPa = [2]float32{-1.99599, 0.99600}
	refHPb = [2]float32{-2, 1}
)

// hpCoeffs returns the high-pass coefficients for a sample rate, preserving
// the 48 kHz filter's pole frequency (in hertz) and its Q by matched-z pole
// warping: r' = r^k, theta' = theta*k for k = 48000/rate.
//
// The double zero stays at DC at every rate, so b is unchanged. The
// Nyquist-asymptotic gain drifts slightly (+0.10 dB at 8 kHz); correcting it
// would break 48 kHz bit-exactness for a broadband gain the training data's
// level normalisation already makes irrelevant.
//
// rate == 48000 short-circuits to the literals so the reference path is
// bit-identical to C rather than merely equal to within float64 round-trip.
func hpCoeffs(rate int) (b, a [2]float32) {
	if rate == refRate {
		return refHPb, refHPa
	}
	return refHPb, warpHP(float64(refRate) / float64(rate))
}

// warpHP maps the reference denominator to a rate whose sample period is k
// times longer, preserving the pole pair's frequency in hertz and its Q.
//
// r and theta are recovered from the reference coefficients rather than
// hardcoded, so TestWarpHPIdentity genuinely exercises this derivation: at
// k == 1 it must reproduce the literals.
func warpHP(k float64) [2]float32 {
	a0 := float64(refHPa[0])
	a1 := float64(refHPa[1])
	r := math.Sqrt(a1)
	cosTheta := -a0 / (2 * r)
	if cosTheta > 1 {
		cosTheta = 1
	}
	theta := math.Acos(cosTheta)

	rk := math.Pow(r, k)
	tk := theta * k
	return [2]float32{
		float32(-2 * rk * math.Cos(tk)),
		float32(rk * rk),
	}
}

// biquad is upstream's rnn_biquad: a transposed direct-form II section with an
// implicit b0 == 1.
//
// The mixed float32/float64 arithmetic is deliberate and must not be
// "cleaned up". Upstream casts to double for the coefficient products while
// leaving the y[i] = x[i] + mem[0] add and the state itself in float. With a
// pole radius of ~0.998 the recursion is marginally stable and carries a large
// near-DC component against which the signal is a small difference, so the
// float64 intermediates keep that cancellation clean. Reproducing the exact
// rounding pattern is also what makes the stage bit-exact against C.
func biquad(y []float32, mem *[2]float32, x []float32, b, a *[2]float32) {
	m0, m1 := mem[0], mem[1]
	b0, b1 := float64(b[0]), float64(b[1])
	a0, a1 := float64(a[0]), float64(a[1])
	for i, xi := range x {
		yi := xi + m0
		xd, yd := float64(xi), float64(yi)
		m0 = float32(float64(m1) + (b0*xd - a0*yd))
		m1 = float32(b1*xd - a1*yd)
		y[i] = yi
	}
	mem[0], mem[1] = m0, m1
}

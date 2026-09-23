package rnnoise

import (
	"math"
	"math/cmplx"
	"testing"
)

// hpResponse evaluates |H(f)| in dB for the second-order high-pass with the
// implicit b0 == 1 that rnn_biquad realises.
func hpResponse(b, a [2]float32, rate int, f float64) float64 {
	w := 2 * math.Pi * f / float64(rate)
	z1 := cmplx.Exp(complex(0, -w))
	z2 := z1 * z1
	num := 1 + complex(float64(b[0]), 0)*z1 + complex(float64(b[1]), 0)*z2
	den := 1 + complex(float64(a[0]), 0)*z1 + complex(float64(a[1]), 0)*z2
	return 20 * math.Log10(cmplx.Abs(num/den))
}

// TestWarpHPIdentity checks the pole-warping derivation against the reference
// coefficients it was derived from. At k == 1 it must reproduce upstream's
// literals, which is a free self-check that r and theta were recovered
// correctly; hpCoeffs then short-circuits the reference rate so the 48 kHz
// path is bit-identical rather than merely close.
func TestWarpHPIdentity(t *testing.T) {
	got := warpHP(1)
	for i := range got {
		if d := math.Abs(float64(got[i] - refHPa[i])); d > 1e-6 {
			t.Errorf("warpHP(1)[%d] = %v, reference %v (diff %g)", i, got[i], refHPa[i], d)
		}
	}
	b, a := hpCoeffs(refRate)
	if a != refHPa || b != refHPb {
		t.Errorf("hpCoeffs(48000) = %v/%v, want the upstream literals %v/%v", b, a, refHPb, refHPa)
	}
}

// maxPassbandDrift bounds an accepted, deliberate deviation.
//
// Matched-z warping moves the poles but leaves the numerator's double zero at
// DC, so the Nyquist-asymptotic gain 4/(1-a0+a1) drifts with the rate: 1.0020
// at 48 kHz rising to 1.0121 at 8 kHz, a +0.09 dB broadband tilt at the lowest
// supported rate. Normalising it would make every rate disagree with the
// reference filter instead of just the low ones, and 0.1 dB of flat gain is
// far below what the training data's level randomisation (RMS normalised to
// ~3000, speech gain swept over tens of dB) already covers. So this is a
// regression guard on a known deviation, not an assertion that none exists.
const maxPassbandDrift = 0.12 // dB

// TestHPStableAndCalibrated checks that the warped filter stays stable and
// keeps the 48 kHz filter's shape in hertz at every rate. That matters because
// every training signal was high-passed with the 48 kHz filter: a fixed a_hp
// reused at 8 kHz would become a ~113 Hz high-pass and eat bands 0-2.
func TestHPStableAndCalibrated(t *testing.T) {
	want1k := hpResponse(refHPb, refHPa, refRate, 1000)
	want30 := hpResponse(refHPb, refHPa, refRate, 30)
	want100 := hpResponse(refHPb, refHPa, refRate, 100)

	for _, rate := range []int{48000, 44100, 32000, 24000, 16000, 12000, 8000} {
		b, a := hpCoeffs(rate)

		// Poles inside the unit circle: a1 is the squared pole radius.
		if r2 := float64(a[1]); r2 >= 1 {
			t.Errorf("rate %d: pole radius^2 = %v, filter is not stable", rate, r2)
		}

		// Passband: flat through the speech range, to within the known
		// drift below. Measured against the 48 kHz reference rather than
		// against 0 dB, because the reference itself is +0.017 dB there.
		if g := hpResponse(b, a, rate, 1000); math.Abs(g-want1k) > maxPassbandDrift {
			t.Errorf("rate %d: gain at 1 kHz = %.4f dB, 48 kHz reference %.4f dB", rate, g, want1k)
		}
		// Stopband shape preserved in hertz, not in normalised frequency.
		if g := hpResponse(b, a, rate, 30); math.Abs(g-want30) > 0.35 {
			t.Errorf("rate %d: gain at 30 Hz = %.3f dB, 48 kHz reference %.3f dB", rate, g, want30)
		}
		if g := hpResponse(b, a, rate, 100); math.Abs(g-want100) > 0.15 {
			t.Errorf("rate %d: gain at 100 Hz = %.3f dB, 48 kHz reference %.3f dB", rate, g, want100)
		}
		// DC must be fully rejected: the numerator has a double zero at z=1.
		if g := hpResponse(b, a, rate, 0); g > -200 {
			t.Errorf("rate %d: gain at DC = %.1f dB, want a true null", rate, g)
		}
	}
}

// TestBiquadBlocksDC exercises the time-domain implementation rather than the
// coefficients: a step input must settle to zero at every rate.
func TestBiquadBlocksDC(t *testing.T) {
	for _, rate := range []int{48000, 16000, 8000} {
		b, a := hpCoeffs(rate)
		var mem [2]float32
		x := make([]float32, rate) // one second
		y := make([]float32, len(x))
		for i := range x {
			x[i] = 1000
		}
		biquad(y, &mem, x, &b, &a)
		tail := y[len(y)-rate/10:]
		var peak float32
		for _, v := range tail {
			if av := float32(math.Abs(float64(v))); av > peak {
				peak = av
			}
		}
		if peak > 0.5 {
			t.Errorf("rate %d: DC not blocked, residual %.4f of 1000", rate, peak)
		}
	}
}

// TestBiquadPassesSpeechBand checks the passband end to end: a 1 kHz tone must
// come through at essentially unity gain at every rate.
func TestBiquadPassesSpeechBand(t *testing.T) {
	for _, rate := range []int{48000, 44100, 16000, 8000} {
		b, a := hpCoeffs(rate)
		var mem [2]float32
		n := rate
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(10000 * math.Sin(2*math.Pi*1000*float64(i)/float64(rate)))
		}
		biquad(y, &mem, x, &b, &a)
		// Skip the transient, then compare RMS over whole cycles.
		skip := rate / 10
		var sx, sy float64
		for i := skip; i < n; i++ {
			sx += float64(x[i]) * float64(x[i])
			sy += float64(y[i]) * float64(y[i])
		}
		gain := 20 * math.Log10(math.Sqrt(sy/sx))
		want := hpResponse(hpB(rate), hpA(rate), rate, 1000)
		if math.Abs(gain-want) > 0.01 {
			t.Errorf("rate %d: 1 kHz through-gain %.4f dB, analytic response %.4f dB",
				rate, gain, want)
		}
		// And that analytic response stays within the accepted drift.
		if math.Abs(want) > maxPassbandDrift {
			t.Errorf("rate %d: passband gain %.4f dB exceeds the accepted drift", rate, want)
		}
	}
}

func hpB(rate int) [2]float32 { b, _ := hpCoeffs(rate); return b }
func hpA(rate int) [2]float32 { _, a := hpCoeffs(rate); return a }

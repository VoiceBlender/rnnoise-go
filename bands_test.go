package rnnoise

import (
	"math"
	"math/rand"
	"testing"
)

var testRates = []int{8000, 12000, 16000, 24000, 32000, 44100, 48000}

// TestBandEnergyRateInvariance pins the property the whole native rate-scaled
// design rests on.
//
// kiss_fft is 1/N-scaled (src/kiss_fft.c: st->scale = 1.f/nfft), so by
// Parseval with the power-complementary window, sum(Ex) is the mean square of
// the windowed frame over 4 -- independent of the transform length, and
// therefore of the sample rate. That is what keeps every absolute-magnitude
// calibration in the feature path valid at other rates: the E<0.04 silence
// gate, log10(1e-2+Ex), the logMax-7 / follow-1.5 dynamic range follower, and
// the features[0]-=12 / features[1]-=4 offsets.
//
// Had the transform been unnormalised, all of those would need a per-rate
// correction and the port would be far more speculative. If this test ever
// fails, the multi-rate design is invalid, not just this function.
// toneFrame renders the same continuous-time signal at any sample rate: a sum
// of tones on the 50 Hz grid, all below 3.5 kHz so the signal fits under the
// Nyquist of every rate under test. Sampling the same function at different
// rates is what makes a cross-rate comparison meaningful; white noise at a
// fixed per-sample variance is a *different* physical signal at each rate,
// with 6x the spectral density at 8 kHz as at 48 kHz.
func toneFrame(rate, n, offset int) []float32 {
	x := make([]float32, n)
	for i := range x {
		t := float64(i+offset) / float64(rate)
		var v float64
		for k := 1; k <= 14; k++ {
			f := float64(k * 250) // 250 Hz .. 3.5 kHz, all multiples of 50
			amp := 1000.0 / float64(k)
			v += amp * math.Sin(2*math.Pi*f*t+float64(k))
		}
		x[i] = float32(v)
	}
	return x
}

// TestBandEnergyRateInvariance pins the property the whole native rate-scaled
// design rests on.
//
// Two facts combine. First, a 10 ms frame with a 20 ms window puts the FFT bin
// spacing at exactly 50 Hz at every supported rate, so eband20ms names the
// same absolute frequencies everywhere. Second, kiss_fft is 1/N-scaled
// (src/kiss_fft.c: st->scale = 1.f/nfft), so by Parseval the band energies
// measure signal power rather than a per-bin sum, and do not grow with the
// transform length.
//
// Together they mean: for a signal below every tested Nyquist, the per-band
// energies are the same numbers at 8 kHz as at 48 kHz. That is what keeps
// every absolute-magnitude calibration in the feature path valid at other
// rates -- the E<0.04 silence gate, log10(1e-2+Ex), the logMax-7 /
// follow-1.5 follower, and the features[0]-=12 / features[1]-=4 offsets.
//
// Had the transform been unnormalised, or the frame not been 10 ms, all of
// those would need a per-rate correction and the port would be speculative. If
// this test fails, the multi-rate design is wrong, not just this function.
func TestBandEnergyRateInvariance(t *testing.T) {
	// Every supported rate must give exactly 50 Hz bins for this to hold.
	for _, rate := range testRates {
		c, err := configForRate(rate)
		if err != nil {
			t.Fatalf("rate %d: %v", rate, err)
		}
		if c.binHz != 50 {
			t.Fatalf("rate %d: bin spacing %v, want exactly 50 Hz", rate, c.binHz)
		}
	}

	bandsAt := func(rate int) []float64 {
		c, _ := configForRate(rate)
		tr := newTransform(c)
		X := make([]cpx, c.freq)
		Ex := make([]float32, numBands)
		acc := make([]float64, numBands)
		const frames = 20
		for f := 0; f < frames; f++ {
			x := toneFrame(rate, c.window, f*c.frame)
			c.applyWindow(x)
			tr.forward(X, x)
			c.computeBandEnergy(Ex, X)
			for i, v := range Ex {
				acc[i] += float64(v) / frames
			}
		}
		return acc
	}

	ref := bandsAt(48000)
	var refTotal float64
	for _, v := range ref {
		refTotal += v
	}

	for _, rate := range testRates {
		if rate == 48000 {
			continue
		}
		c, _ := configForRate(rate)
		got := bandsAt(rate)

		// Per-band agreement over the bands this rate actually has. This is
		// the direct statement that the band table names the same
		// frequencies at both rates.
		for i := 0; i < c.activeBands; i++ {
			if ref[i] < refTotal*1e-9 {
				continue // band carries no tone; relative error is meaningless
			}
			if d := math.Abs(got[i]/ref[i] - 1); d > 0.02 {
				t.Errorf("rate %d: band %d (%.0f-%.0f Hz) energy %.4g, 48 kHz %.4g (%.2f%% off)",
					rate, i, float64(eband20ms[i])*50, float64(eband20ms[i+1])*50,
					got[i], ref[i], d*100)
			}
		}

		// Total energy agreement, which is what the silence gate and the
		// feature offsets are calibrated against.
		var total float64
		for _, v := range got {
			total += v
		}
		if d := math.Abs(total/refTotal - 1); d > 0.02 {
			t.Errorf("rate %d: sum(Ex) = %.6g, 48 kHz %.6g (%.2f%% off)", rate, total, refTotal, d*100)
		}
	}
}

// TestBandEnergyMatchesLowpassed48k states the native-rate equivalence
// directly: running at rate R must produce the same band energies as running
// the unmodified 48 kHz path on a signal whose FFT bins above R/2 were zeroed
// -- which is exactly how src/dump_features.c band-limits 47% of the training
// data (lowpass = FREQ_SIZE*3000/24000 * 50^u, then X[i] = 0 above it).
//
// This is the argument that reduced-rate operation is in-distribution rather
// than an approximation, so it is worth asserting rather than reasoning about.
func TestBandEnergyMatchesLowpassed48k(t *testing.T) {
	c48, _ := configForRate(48000)
	tr48 := newTransform(c48)

	for _, rate := range []int{8000, 16000, 24000} {
		c, _ := configForRate(rate)
		tr := newTransform(c)

		X48 := make([]cpx, c48.freq)
		Ex48 := make([]float32, numBands)
		X := make([]cpx, c.freq)
		Ex := make([]float32, numBands)

		x48 := toneFrame(48000, c48.window, 0)
		c48.applyWindow(x48)
		tr48.forward(X48, x48)
		// The band-limit training applies: zero every bin at or above the
		// reduced rate's Nyquist bin.
		for i := c.freq; i < c48.freq; i++ {
			X48[i] = cpx{}
		}
		c48.computeBandEnergy(Ex48, X48)

		x := toneFrame(rate, c.window, 0)
		c.applyWindow(x)
		tr.forward(X, x)
		c.computeBandEnergy(Ex, X)

		for i := 0; i < numBands; i++ {
			a, b := float64(Ex48[i]), float64(Ex[i])
			if a == 0 && b == 0 {
				continue
			}
			if a < 1e-6 {
				continue
			}
			if d := math.Abs(b/a - 1); d > 0.02 {
				t.Errorf("rate %d band %d: native %.6g, lowpassed-48k %.6g (%.2f%% off)",
					rate, i, b, a, d*100)
			}
		}
		// Bands entirely above Nyquist must be exactly zero in native mode,
		// which is what puts Ly at its log floor there, as in training.
		for i := c.activeBands; i < numBands; i++ {
			if Ex[i] != 0 {
				t.Errorf("rate %d: band %d is above Nyquist but has energy %v", rate, i, Ex[i])
			}
		}
	}
}

// TestActiveBandsMatchTrainingMask checks the claim that makes reduced rates
// safe: the bands interpBandGain actually reads for existing bins are exactly
// the bands the model has trained gains for.
//
// Bands at and above activeBands get untrained network output, because
// training masks their loss (dump_features.c: g[i] = -1 for i > band_lp, which
// train_rnnoise.py's mask() turns into a zero loss weight). If any real bin
// ever picked up one of those gains, reduced-rate output would be driven by
// garbage. Here we poison them and assert nothing downstream sees it.
func TestActiveBandsMatchTrainingMask(t *testing.T) {
	const poison = -1e30

	for _, rate := range testRates {
		c, err := configForRate(rate)
		if err != nil {
			t.Fatalf("rate %d: %v", rate, err)
		}
		bandE := make([]float32, numBands)
		for i := range bandE {
			if i < c.activeBands {
				bandE[i] = 0.5
			} else {
				bandE[i] = poison
			}
		}
		g := make([]float32, c.freq)
		c.interpBandGain(g, bandE)

		for i, v := range g {
			if v < 0 {
				t.Errorf("rate %d: bin %d (%.0f Hz) picked up an untrained band gain (%g)",
					rate, i, float64(i)*c.binHz, v)
				break
			}
		}

		// The converse: every band below activeBands must influence some bin,
		// otherwise activeBands is larger than it should be.
		for b := 0; b < c.activeBands; b++ {
			probe := make([]float32, numBands)
			probe[b] = 1
			c.interpBandGain(g, probe)
			var touched bool
			for _, v := range g {
				if v != 0 {
					touched = true
					break
				}
			}
			if !touched {
				t.Errorf("rate %d: band %d is counted active but reaches no bin", rate, b)
			}
		}
	}
}

// TestInterpBandGainNoLeak is a regression test for the reused-scratch hazard.
// Upstream gets bins at and above 400 zeroed for free because its callers pass
// fresh stack arrays; we reuse one buffer per stream, so interpBandGain has to
// clear that tail itself. If it stops doing so, the previous frame's gains
// survive above 20 kHz.
func TestInterpBandGainNoLeak(t *testing.T) {
	for _, rate := range []int{44100, 48000} {
		c, err := configForRate(rate)
		if err != nil {
			t.Fatal(err)
		}
		g := make([]float32, c.freq)
		hot := make([]float32, numBands)
		for i := range hot {
			hot[i] = 1
		}
		c.interpBandGain(g, hot)
		// Simulate a previous frame having written the whole buffer.
		for i := range g {
			g[i] = 1
		}
		cold := make([]float32, numBands)
		c.interpBandGain(g, cold)
		for i := eband20ms[numBands+1]; i < c.freq; i++ {
			if g[i] != 0 {
				t.Fatalf("rate %d: bin %d = %v, want 0 (gain leaked above 20 kHz)", rate, i, g[i])
			}
		}
	}
}

// TestTransformRoundTrip checks forward/inverse compose to identity, which is
// only true because the forward transform is 1/N-scaled and the inverse
// multiplies by N again.
func TestTransformRoundTrip(t *testing.T) {
	for _, rate := range testRates {
		c, _ := configForRate(rate)
		tr := newTransform(c)
		rng := rand.New(rand.NewSource(3))
		in := make([]float32, c.window)
		for i := range in {
			in[i] = float32(1000 * rng.NormFloat64())
		}
		X := make([]cpx, c.freq)
		tr.forward(X, in)
		out := make([]float32, c.window)
		tr.inverse(out, X)

		var maxErr float64
		for i := range in {
			if e := math.Abs(float64(out[i] - in[i])); e > maxErr {
				maxErr = e
			}
		}
		if rel := maxErr / 1000; rel > 1e-4 {
			t.Errorf("rate %d: forward/inverse round trip max error %g (relative %g)", rate, maxErr, rel)
		}
	}
}

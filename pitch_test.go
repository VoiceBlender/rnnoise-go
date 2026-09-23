package rnnoise

import (
	"math"
	"math/rand"
	"testing"
)

// naiveXCorr is the "simple version of the pitch correlation" upstream keeps
// under #if 0 in rnn_pitch_xcorr. It computes the same quantity as the
// unrolled kernel in a different summation order, so it validates the
// unrolled version's indexing without asserting bit equality.
func naiveXCorr(x, y []float32, out []float32, length, maxPitch int) {
	for i := 0; i < maxPitch; i++ {
		var sum float32
		for j := 0; j < length; j++ {
			sum += x[j] * y[i+j]
		}
		out[i] = sum
	}
}

// TestPitchXCorrMatchesNaive exercises the four-lag unrolled kernel including
// its non-unrolled tail, which reduced rates actually hit: maxPitch there is
// pitchMax-3*pitchMin, which need not be a multiple of 4.
func TestPitchXCorrMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, tc := range []struct{ length, maxPitch int }{
		{64, 32}, {64, 33}, {64, 34}, {64, 35}, // every tail residue
		{240, 147},             // 48 kHz coarse search: (768-180)/4
		{80, 49},               // 16 kHz coarse search: (256-60)/4
		{40, 23},               // 8 kHz coarse search: (128-30)/4
		{5, 1}, {6, 2}, {7, 3}, // minimum sizes
	} {
		x := make([]float32, tc.length)
		y := make([]float32, tc.length+tc.maxPitch+4)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		for i := range y {
			y[i] = float32(rng.NormFloat64())
		}
		got := make([]float32, tc.maxPitch)
		want := make([]float32, tc.maxPitch)
		pitchXCorr(x, y, got, tc.length, tc.maxPitch)
		naiveXCorr(x, y, want, tc.length, tc.maxPitch)
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-3*math.Abs(float64(want[i]))+1e-4 {
				t.Errorf("len=%d maxPitch=%d: xcorr[%d] = %v, naive %v",
					tc.length, tc.maxPitch, i, got[i], want[i])
				break
			}
		}
	}
}

// detectPitch runs the whole pitch chain exactly as
// rnn_compute_frame_features does, on a synthetic harmonic signal.
func detectPitch(c *rateConfig, f0 float64) (int, float32) {
	buf := make([]float32, c.pitchBuf)
	for i := range buf {
		tt := float64(i) / float64(c.sampleRate)
		var v float64
		for h := 1; h <= 12; h++ {
			if float64(h)*f0 > float64(c.sampleRate)/2*0.9 {
				break
			}
			v += math.Cos(2*math.Pi*float64(h)*f0*tt) / float64(h)
		}
		buf[i] = float32(1000 * v)
	}
	ps := newPitchState(c)
	ps.downsample(buf, ps.lp, c.pitchBuf)
	idx := ps.search(ps.lp[c.pitchMax>>1:], ps.lp, c.pitchFrame, c.pitchMax-3*c.pitchMin)
	idx = c.pitchMax - idx
	return ps.removeDoubling(ps.lp, 0, c.pitchMax, c.pitchMin, c.pitchFrame, idx, 0, 0)
}

// TestPitchAccuracySweep characterises the pitch search across its whole
// range at every rate, and pins the resolution trade that native rate-scaled
// operation makes.
//
// The finding, measured over 87 fundamentals from 80 to 400 Hz per rate: the
// absolute error stays within 2 native samples at every rate, because that is
// the granularity of the search's own +/-1 pseudo-interpolation on a 2x
// decimated grid. The *relative* error therefore scales with the sample
// period: about 0.95% of a 200 Hz period at 48 kHz against 5.98% at 8 kHz.
// Gross (octave) errors were 0/87 at 16 kHz and above, and 2/87 at 8 and
// 12 kHz.
//
// That is the honest cost of not resampling: a 48 kHz reference chain gets
// three times the lag resolution for the same voice. The design deliberately
// does not add fractional-lag interpolation to claw it back, because the model
// was trained on Exp computed from integer lags by exactly this code, so
// interpolating would move the features out of distribution for an unproven
// gain. Phase 8's quality metrics measure what it actually costs downstream.
func TestPitchAccuracySweep(t *testing.T) {
	// The search resolves to 2*T+offset on a half-rate grid, so 2 native
	// samples is the floor, not a tolerance to be tightened.
	const maxAbsErrSamples = 2.0
	// Gross errors are signal-dependent and not monotonic in the rate
	// (measured: 2/87 at 8 and 12 kHz, 1/87 at 32 kHz, 0 elsewhere).
	const maxGrossErrRate = 0.05

	for _, rate := range testRates {
		c, _ := configForRate(rate)
		var worst float64
		gross, n := 0, 0
		for f0 := 80.0; f0 <= 400.0; f0 += 3.7 {
			period := float64(rate) / f0
			if period < float64(c.pitchMin) || period > float64(c.pitchMax) {
				continue
			}
			got, _ := detectPitch(c, f0)
			n++
			if math.Abs(float64(got)/period-1) > 0.25 {
				gross++
				continue
			}
			if e := math.Abs(float64(got) - period); e > worst {
				worst = e
			}
		}
		if n == 0 {
			t.Fatalf("rate %d: no fundamentals in range", rate)
		}
		if worst > maxAbsErrSamples {
			t.Errorf("rate %d: worst period error %.2f native samples, limit %.1f",
				rate, worst, maxAbsErrSamples)
		}
		if r := float64(gross) / float64(n); r > maxGrossErrRate {
			t.Errorf("rate %d: %d/%d gross pitch errors (%.1f%%), limit %.0f%%",
				rate, gross, n, r*100, maxGrossErrRate*100)
		}
		t.Logf("rate %6d: worst error %.2f samples (%.2f%% of a 200 Hz period), %d/%d gross",
			rate, worst, 100*worst/(float64(rate)/200), gross, n)
	}
}

// TestPitchDetection drives the whole chain at every rate and checks the
// detected period is the true one, in native samples.
//
// This is what validates the rate-scaled pitch constants: pitchMin, pitchMax,
// the 2x downsample and the 4x/2x search decimation all have to line up, and
// pitchBuf has to be deep enough to hold a full-period lookback. It drives
// each rate at fundamentals whose period is a whole number of native samples,
// so the search grid can represent the answer exactly; TestPitchAccuracySweep
// covers the off-grid behaviour.
func TestPitchDetection(t *testing.T) {
	for _, rate := range testRates {
		c, err := configForRate(rate)
		if err != nil {
			t.Fatal(err)
		}
		// Periods of 80, 100 and 120 native samples sit inside
		// [pitchMin, pitchMax] at every supported rate.
		for _, period := range []int{80, 100, 120} {
			f0 := float64(rate) / float64(period)
			got, gain := detectPitch(c, f0)

			if got < c.pitchMin || got > c.pitchMax {
				t.Errorf("rate %d f0=%.1f: period %d outside [%d,%d]",
					rate, f0, got, c.pitchMin, c.pitchMax)
				continue
			}
			if d := abs(got - period); d > 1 {
				t.Errorf("rate %d f0=%.1f: detected period %d, true %d (%.1f Hz vs %.1f Hz)",
					rate, f0, got, period, float64(rate)/float64(got), f0)
			}
			if gain < 0.4 {
				t.Errorf("rate %d f0=%.1f: pitch gain %.3f too low for a periodic signal",
					rate, f0, gain)
			}
		}
	}
}

// TestPitchFeatureIsRateInvariant checks the features[64] rescaling: the
// network was trained on a pitch index counted in 48 kHz samples, so a native
// period has to be converted before it is fed in. Without the 48000/rate
// scaling a 100 Hz voice would present as 480 samples at 48 kHz but 160 at
// 16 kHz, and the model would read the latter as a ~300 Hz voice.
func TestPitchFeatureIsRateInvariant(t *testing.T) {
	// f0 = 100 Hz gives a whole-sample half-rate period at every rate listed.
	// 44.1 kHz is excluded because 44100/200 = 220.5 sits exactly half a
	// sample off the search grid; TestPitchAccuracySweep covers that.
	const f0 = 100.0
	want := 0.01 * (48000/f0 - 300)
	for _, rate := range []int{8000, 12000, 16000, 24000, 32000, 48000} {
		c, _ := configForRate(rate)
		got, _ := detectPitch(c, f0)
		feat := 0.01 * (float64(got)*c.pitchScale - 300)
		if d := math.Abs(feat - want); d > 0.02 {
			t.Errorf("rate %d: features[64] = %.4f, 48 kHz-equivalent %.4f (period %d)",
				rate, feat, want, got)
		}
	}
}

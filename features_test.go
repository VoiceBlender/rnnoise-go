package rnnoise

import (
	"math"
	"math/rand"
	"testing"
)

// bandLimitedSignal renders the same continuous-time voiced-speech-like
// signal at any sample rate: harmonics of f0 rolled off at 1/h and stopping
// below 3.5 kHz, so nothing lies above the Nyquist of any rate under test,
// plus a slow amplitude envelope so the frames are not all identical.
func bandLimitedSignal(rate, n int, f0 float64) []float32 {
	x := make([]float32, n)
	for i := range x {
		t := float64(i) / float64(rate)
		env := 0.6 + 0.4*math.Sin(2*math.Pi*3*t)
		var v float64
		for h := 1; h <= 25; h++ {
			f := f0 * float64(h)
			if f > 3500 {
				break
			}
			v += math.Cos(2*math.Pi*f*t+0.7*float64(h)) / float64(h)
		}
		x[i] = float32(4000 * env * v)
	}
	return x
}

// featuresAfter runs nFrames of the analysis pipeline and returns the last
// frame's feature vector, its silence flag and the detected pitch lag.
func featuresAfter(t *testing.T, rate, nFrames int, sig []float32) ([]float32, bool, int) {
	t.Helper()
	c, err := configForRate(rate)
	if err != nil {
		t.Fatal(err)
	}
	d := newDenoiser(c)
	silence := false
	for f := 0; f < nFrames; f++ {
		in := sig[f*c.frame : (f+1)*c.frame]
		silence = d.computeFrameFeatures(d.X, d.P, d.Ex, d.Ep, d.Exp, d.features, in)
	}
	out := make([]float32, numFeatures)
	copy(out, d.features)
	return out, silence, d.lastPeriod
}

// maxDelta returns the largest absolute difference over a[lo:hi] vs b[lo:hi].
func maxDelta(a, b []float32, lo, hi int) float64 {
	var m float64
	for i := lo; i < hi; i++ {
		if e := math.Abs(float64(a[i] - b[i])); e > m {
			m = e
		}
	}
	return m
}

// TestFeaturesMatchAcrossRates is the decisive test for the native
// rate-scaled design.
//
// For a signal containing nothing above 3.5 kHz, the 48 kHz analysis already
// sees zero energy in every band above that -- which is exactly the state a
// reduced rate is in, where those bands do not exist at all. So if the design
// is right, the whole 65-element vector the network sees must be the same
// numbers at 8 kHz as at 48 kHz, with no rescaling anywhere except
// features[64].
//
// That is far stronger than "the band energies agree": it exercises the
// log-domain dynamic-range follower walking the absent bands down to the
// floor, the DCT over all 32 bands including those, the features[0]-=12 /
// features[1]-=4 offsets, the pitch-correlation DCT and the features[64]
// rescaling, all at once. If it passes, a reduced rate really is presenting
// the model with an input from its training distribution rather than an
// approximation of one.
//
// The test runs two fundamentals to separate the design from its one genuine
// cost:
//
//   - 100 Hz has a whole-sample period at every rate (480 at 48 kHz, 80 at
//     8 kHz, 441 at 44.1 kHz), so the search grid can represent the lag
//     exactly. Measured: every rate recovers the identical 48 kHz-equivalent
//     lag of 480, the pitch-correlation features agree to 2e-5, and the
//     cepstral features to 2e-3 at 8 kHz.
//   - 140 Hz has a 342.86-sample period at 48 kHz, representable at no rate,
//     so each rate rounds to a different integer lag. The cepstral features
//     still agree to 9e-3, but the pitch-correlation features diverge by up
//     to 0.17 -- entirely from that +/-1 sample, as the 100 Hz case proves.
func TestFeaturesMatchAcrossRates(t *testing.T) {
	const frames = 30

	cases := []struct {
		f0               float64
		maxCeps, maxCorr float64
		maxFeat64        float64
		requireSameLag   bool
	}{
		// Whole-sample period: everything must line up almost exactly.
		{f0: 100, maxCeps: 0.005, maxCorr: 0.001, maxFeat64: 1e-6, requireSameLag: true},
		// Off-grid period: the cepstral half still tracks; the
		// pitch-correlation half carries the lag quantisation.
		{f0: 140, maxCeps: 0.010, maxCorr: 0.200, maxFeat64: 0.02, requireSameLag: false},
	}

	for _, tc := range cases {
		c48, _ := configForRate(48000)
		ref, silence, lag48 := featuresAfter(t, 48000, frames,
			bandLimitedSignal(48000, (frames+1)*c48.frame, tc.f0))
		if silence {
			t.Fatalf("f0=%.0f: 48 kHz reference frame was classified silent", tc.f0)
		}

		for _, rate := range testRates {
			if rate == 48000 {
				continue
			}
			c, _ := configForRate(rate)
			got, sil, lag := featuresAfter(t, rate, frames,
				bandLimitedSignal(rate, (frames+1)*c.frame, tc.f0))
			if sil {
				t.Errorf("f0=%.0f rate %d: frame classified silent but 48 kHz was not", tc.f0, rate)
				continue
			}

			ceps := maxDelta(got, ref, 0, numBands)
			corr := maxDelta(got, ref, numBands, 2*numBands)
			f64 := maxDelta(got, ref, numFeatures-1, numFeatures)
			lagEquiv := float64(lag) * c.pitchScale

			if ceps > tc.maxCeps {
				t.Errorf("f0=%.0f rate %d: cepstral features differ from 48 kHz by up to %.5f (limit %.4f)",
					tc.f0, rate, ceps, tc.maxCeps)
			}
			if corr > tc.maxCorr {
				t.Errorf("f0=%.0f rate %d: pitch-correlation features differ by up to %.5f (limit %.4f)",
					tc.f0, rate, corr, tc.maxCorr)
			}
			if f64 > tc.maxFeat64 {
				t.Errorf("f0=%.0f rate %d: features[64] = %.5f, 48 kHz %.5f",
					tc.f0, rate, got[numFeatures-1], ref[numFeatures-1])
			}
			if tc.requireSameLag && math.Abs(lagEquiv-float64(lag48)) > 0.5 {
				t.Errorf("f0=%.0f rate %d: lag %d is %.2f in 48 kHz samples, reference %d; "+
					"a whole-sample period must resolve identically at every rate",
					tc.f0, rate, lag, lagEquiv, lag48)
			}

			t.Logf("f0=%3.0f rate %6d: lag %3d (%.2f @48k, ref %d) ceps %.5f corr %.5f feat64 %.5f",
				tc.f0, rate, lag, lagEquiv, lag48, ceps, corr, f64)
		}
	}
}

// TestSilenceGate checks the gate trips on digital silence at every rate and,
// crucially, does NOT trip on genuinely quiet audio at the lowest rate.
//
// The gate compares sum(Ex) against an absolute 0.04. A reduced rate discards
// the bands above its Nyquist, so if the band energies had scaled with the
// transform length this threshold would drift and 8 kHz speech could be
// misread as silence, freezing the network. Rate-invariance is what prevents
// that; this test is the guard.
func TestSilenceGate(t *testing.T) {
	const frames = 6
	for _, rate := range testRates {
		c, _ := configForRate(rate)

		zero := make([]float32, (frames+1)*c.frame)
		if _, sil, _ := featuresAfter(t, rate, frames, zero); !sil {
			t.Errorf("rate %d: digital silence did not trip the silence gate", rate)
		}

		// -60 dBFS white noise: about 33 LSB RMS at the +/-32768 scale. Well
		// below anything audible as speech, but far above digital silence,
		// and the gate must let it through.
		rng := rand.New(rand.NewSource(5))
		quiet := make([]float32, (frames+1)*c.frame)
		for i := range quiet {
			quiet[i] = float32(32768 * 0.001 * rng.NormFloat64())
		}
		if _, sil, _ := featuresAfter(t, rate, frames, quiet); sil {
			t.Errorf("rate %d: -60 dBFS noise wrongly classified as silence", rate)
		}
	}
}

// TestFeaturesFinite guards against NaN or Inf reaching the network from any
// of the divisions and logarithms in the feature path, including the
// degenerate inputs where Ep or newE can be zero.
func TestFeaturesFinite(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for _, rate := range testRates {
		c, _ := configForRate(rate)
		n := 12 * c.frame
		cases := map[string][]float32{
			"silence":   make([]float32, n),
			"speechish": bandLimitedSignal(rate, n, 140),
			"fullscale": func() []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = 32767
					if i%2 == 0 {
						s[i] = -32768
					}
				}
				return s
			}(),
			"noise": func() []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(8000 * rng.NormFloat64())
				}
				return s
			}(),
			"dc": func() []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = 20000
				}
				return s
			}(),
		}
		for name, sig := range cases {
			d := newDenoiser(c)
			for f := 0; f < 11; f++ {
				in := sig[f*c.frame : (f+1)*c.frame]
				d.computeFrameFeatures(d.X, d.P, d.Ex, d.Ep, d.Exp, d.features, in)
				for i, v := range d.features {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("rate %d %s frame %d: features[%d] = %v", rate, name, f, i, v)
					}
				}
			}
		}
	}
}

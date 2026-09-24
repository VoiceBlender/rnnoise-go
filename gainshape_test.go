package rnnoise

import (
	"math"
	"math/rand"
	"testing"
)

// noiseAtten returns the energy ratio, in dB, between white noise in and the
// denoised output, past the frames the network needs to settle.
func noiseAtten(t *testing.T, o Options) float64 {
	t.Helper()
	o.SampleRate, o.Model = 16000, testModel(t)
	d, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	in, out := make([]float32, d.FrameSize()), make([]float32, d.FrameSize())
	var ein, eout float64
	for f := 0; f < 200; f++ {
		for i := range in {
			in[i] = float32(rng.NormFloat64() * 2000)
		}
		if _, err := d.Process(out, in); err != nil {
			t.Fatal(err)
		}
		if f < 50 {
			continue
		}
		for i := range in {
			ein += float64(in[i]) * float64(in[i])
			eout += float64(out[i]) * float64(out[i])
		}
	}
	return 10 * math.Log10(math.Max(eout, 1e-12)/ein)
}

func TestGainFloorCapsAttenuation(t *testing.T) {
	if base := noiseAtten(t, Options{}); base > -40 {
		t.Fatalf("unshaped attenuation %.1f dB, expected well past -40", base)
	}
	for _, floor := range []float64{-30, -20, -10, -6} {
		got := noiseAtten(t, Options{GainFloorDB: floor})
		if got < floor-1 || got > floor+1 {
			t.Errorf("floor %g dB: attenuation %.2f dB, want within 1 dB of the floor", floor, got)
		}
	}
}

func TestAggressivenessIsMonotonic(t *testing.T) {
	prev := math.Inf(1)
	for _, a := range []float64{0.25, 0.5, 1, 1.5, 3} {
		got := noiseAtten(t, Options{Aggressiveness: a})
		if got >= prev {
			t.Errorf("aggressiveness %g: attenuation %.2f dB, want below the previous %.2f", a, got, prev)
		}
		prev = got
	}
	if def, one := noiseAtten(t, Options{}), noiseAtten(t, Options{Aggressiveness: 1}); def != one {
		t.Errorf("Aggressiveness 1 gave %.4f dB, want the default %.4f", one, def)
	}
}

// The floor is applied after the exponent, so it bounds the result whatever
// the exponent asks for.
func TestGainFloorWinsOverAggressiveness(t *testing.T) {
	got := noiseAtten(t, Options{GainFloorDB: -20, Aggressiveness: 3})
	if got < -21 || got > -19 {
		t.Errorf("attenuation %.2f dB, want within 1 dB of -20", got)
	}
}

func TestGainShapeRejects(t *testing.T) {
	for _, o := range []Options{{GainFloorDB: 6}, {Aggressiveness: -1}} {
		o.SampleRate, o.Model = 16000, testModel(t)
		if _, err := New(o); err == nil {
			t.Errorf("New(%+v) succeeded, want an error", o)
		}
	}
}

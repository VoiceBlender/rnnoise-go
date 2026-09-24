package rnnoise

import (
	"math"
	"math/rand"
	"testing"
)

// The fast pitch correlation splits each lag across vector lanes and fuses the
// multiply-add, so it is not bit-exact. It must still agree to float32 noise.
func TestXcorrFastMatchesExact(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, length := range []int{8, 9, 15, 16, 64, 90, 180, 360, 361} {
		x := make([]float32, length)
		y := make([]float32, length+8)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		for i := range y {
			y[i] = float32(rng.NormFloat64())
		}
		var exact, fast [4]float32
		xcorrKernel(x, y, &exact, length)
		xcorrKernelFast(x, y, &fast, length)
		for k := range exact {
			rel := math.Abs(float64(exact[k]-fast[k])) / math.Max(math.Abs(float64(exact[k])), 1)
			if rel > 1e-5 {
				t.Errorf("length=%d lag=%d: exact %v, fast %v (relative %.2e)", length, k, exact[k], fast[k], rel)
			}
		}
	}
}

// voiced builds one frame of a harmonic stack in noise. snr scales the
// harmonics against a fixed noise floor; 0 is noise alone.
func voiced(buf []float32, rate, frame int, f0, snr float64, rng *rand.Rand) {
	for i := range buf {
		t := float64(frame*len(buf)+i) / float64(rate)
		var v float64
		for h := 1; h <= 6; h++ {
			v += math.Sin(2*math.Pi*f0*float64(h)*t) / float64(h)
		}
		buf[i] = float32(v*1500*snr + rng.NormFloat64()*1500)
	}
}

// Options.Fast must not change what the denoiser produces in any way a listener
// could notice. The correlation feeds only the period search, so in practice
// the output is identical; the bound here allows for an occasional near-tie
// resolving the other way without letting a broken kernel through.
func TestFastModeQuality(t *testing.T) {
	m := testModel(t)
	for _, rate := range []int{8000, 16000, 48000} {
		for _, snr := range []float64{10, 1, 0.3, 0} {
			exact, err := New(Options{SampleRate: rate, Model: m})
			if err != nil {
				t.Fatal(err)
			}
			fast, err := New(Options{SampleRate: rate, Model: m, Fast: true})
			if err != nil {
				t.Fatal(err)
			}
			n := exact.FrameSize()
			in := make([]float32, n)
			oe, of := make([]float32, n), make([]float32, n)
			rng := rand.New(rand.NewSource(11))
			var sig, dif float64
			var cnt int
			var maxVAD float64
			f0 := 90.0
			for f := 0; f < 400; f++ {
				if f0 += 0.35; f0 > 260 {
					f0 = 90
				}
				voiced(in, rate, f, f0, snr, rng)
				ve, err := exact.Process(oe, in)
				if err != nil {
					t.Fatal(err)
				}
				vf, err := fast.Process(of, in)
				if err != nil {
					t.Fatal(err)
				}
				if d := math.Abs(float64(ve - vf)); d > maxVAD {
					maxVAD = d
				}
				if f < 20 {
					continue // the network is still converging
				}
				for i := range oe {
					sig += float64(oe[i]) * float64(oe[i])
					d := float64(oe[i]) - float64(of[i])
					dif += d * d
					cnt++
				}
			}
			rel := math.Sqrt(dif/float64(cnt)) / math.Max(math.Sqrt(sig/float64(cnt)), 1e-9)
			if rel > 0.01 {
				t.Errorf("%d Hz snr=%g: fast output differs by %.2e of signal RMS, want below 1e-2", rate, snr, rel)
			}
			if maxVAD > 0.05 {
				t.Errorf("%d Hz snr=%g: VAD differs by %.3f, want below 0.05", rate, snr, maxVAD)
			}
		}
	}
}

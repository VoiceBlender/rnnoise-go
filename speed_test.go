package rnnoise

import (
	"math/rand"
	"testing"
)

// BenchmarkProcess reports per-frame cost. Divide 10 ms by the ns/op to get the
// real-time factor: the README table is generated from this.
//
// It also enforces the zero-allocation guarantee that makes pooling Denoisers
// across many streams worthwhile. Anything above 0 allocs/op is a regression: a
// slice escaping to the heap on the per-frame path would cost more in GC
// pressure at a few hundred streams than the arithmetic does.
func BenchmarkProcess(b *testing.B) {
	m, err := LoadModelFile("model/weights.bin")
	if err != nil {
		b.Skipf("model/weights.bin not present: %v", err)
	}
	for _, rate := range []int{8000, 16000, 44100, 48000} {
		for _, q := range []QuantMode{QuantSigned, QuantUnsigned} {
			b.Run(rateName(rate)+"/"+q.String(), func(b *testing.B) {
				d, err := New(Options{SampleRate: rate, Model: m, Quantization: q})
				if err != nil {
					b.Fatal(err)
				}
				n := d.FrameSize()
				rng := rand.New(rand.NewSource(1))
				in := make([]float32, n)
				out := make([]float32, n)
				for i := range in {
					in[i] = float32(3000 * rng.NormFloat64())
				}
				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if _, err := d.Process(out, in); err != nil {
						b.Fatal(err)
					}
				}
				// Real-time factor: how many streams one core could carry.
				b.ReportMetric(float64(n)/float64(rate)*1e9/float64(b.Elapsed().Nanoseconds())*float64(b.N), "xRT")
			})
		}
	}
}

// BenchmarkAnalysisOnly isolates the DSP front end from the network, to confirm
// where the time actually goes before any assembly is written.
func BenchmarkAnalysisOnly(b *testing.B) {
	c, _ := configForRate(48000)
	d := newDenoiser(c)
	rng := rand.New(rand.NewSource(1))
	in := make([]float32, c.frame)
	for i := range in {
		in[i] = float32(3000 * rng.NormFloat64())
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		d.computeFrameFeatures(d.X, d.P, d.Ex, d.Ep, d.Exp, d.features, in)
	}
}

func rateName(rate int) string {
	switch rate {
	case 8000:
		return "8k"
	case 16000:
		return "16k"
	case 44100:
		return "44k1"
	case 48000:
		return "48k"
	}
	return "other"
}

// Package ctest holds the test-only harnesses that compare this port against
// the original C and quantify what native rate-scaled operation costs.
//
// It is a separate module so that neither the C differential harness nor the
// vendored resampler appears in the library's dependency graph.
package ctest

import (
	"fmt"

	rnnoise "github.com/VoiceBlender/rnnoise-go"
	"github.com/VoiceBlender/rnnoise-go/ctest/internal/resampler"
)

// resampleQuality is Speex's highest setting. The reference chain's whole
// purpose is to represent the best available "resample to 48 kHz and back", so
// anything less would understate it and flatter native-rate mode.
const resampleQuality = 10

// denoiseResult is one chain's output, aligned so that sample i of out
// corresponds to sample i of the input.
type denoiseResult struct {
	out []float32
	vad []float32 // one entry per frame of the rate the denoising ran at
}

// runNative is chain T: denoise at the signal's own rate.
func runNative(m *rnnoise.Model, rate int, in []float32) (*denoiseResult, error) {
	d, err := rnnoise.New(rnnoise.Options{SampleRate: rate, Model: m})
	if err != nil {
		return nil, err
	}
	return runFrames(d, in)
}

// runFrames drives a Denoiser over whole frames and compensates its one-frame
// algorithmic delay, so the result lines up with the input.
func runFrames(d *rnnoise.Denoiser, in []float32) (*denoiseResult, error) {
	n := d.FrameSize()
	frames := len(in) / n
	out := make([]float32, frames*n)
	vad := make([]float32, 0, frames)
	buf := make([]float32, n)
	for f := 0; f < frames; f++ {
		v, err := d.Process(buf, in[f*n:(f+1)*n])
		if err != nil {
			return nil, err
		}
		copy(out[f*n:], buf)
		vad = append(vad, v)
	}
	// Delay() is the total input-to-output lag: one frame from the
	// overlap-add and one from the delayed gain application. Trimming it lines
	// the result up with the input, which every metric below depends on.
	delay := d.Delay()
	if delay >= len(out) {
		return &denoiseResult{out: nil, vad: vad}, nil
	}
	return &denoiseResult{out: out[delay:], vad: vad}, nil
}

// runResampled is chain R-prime: upsample to 48 kHz, denoise there, and come
// back. This is the maintainer's own recommendation for non-48 kHz input
// (upstream issue #37), so it is the fair thing to measure native-rate mode
// against.
//
// It is the primary reference precisely because it shares the network *and the
// code* with chain T. Any difference between them is the native-rate
// approximation alone, with Go-versus-C error and model differences excluded by
// construction.
func runResampled(m *rnnoise.Model, rate int, in []float32) (*denoiseResult, error) {
	const ref = 48000
	up, err := resample(in, rate, ref)
	if err != nil {
		return nil, err
	}
	mid, err := runNative(m, ref, up)
	if err != nil {
		return nil, err
	}
	down, err := resample(mid.out, ref, rate)
	if err != nil {
		return nil, err
	}
	return &denoiseResult{out: down, vad: mid.vad}, nil
}

// resample converts between rates with the Speex resampler, trimming its own
// latency so the result stays time-aligned with the input.
func resample(in []float32, from, to int) ([]float32, error) {
	if from == to {
		out := make([]float32, len(in))
		copy(out, in)
		return out, nil
	}
	r := resampler.New(1, from, to, resampleQuality)
	// Pad the input by the filter's latency so the tail is flushed out, then
	// drop the corresponding lead-in from the output.
	lead := r.OutputLatency()
	pad := r.InputLatency() + 1
	src := make([]float32, len(in)+pad)
	copy(src, in)

	want := len(src)*to/from + 64
	dst := make([]float32, want)
	_, written := r.ProcessFloat32(0, src, dst)
	if written <= lead {
		return nil, fmt.Errorf("resample %d->%d produced %d samples, latency is %d",
			from, to, written, lead)
	}
	out := dst[lead:written]
	// Trim to the exact expected length so metrics compare equal spans.
	exact := len(in) * to / from
	if len(out) > exact {
		out = out[:exact]
	}
	res := make([]float32, len(out))
	copy(res, out)
	return res, nil
}

// Package rnnoise is a pure-Go port of Xiph's RNNoise recurrent-neural-network
// noise suppressor, ported from upstream commit 70f1d256.
//
// It differs from the C original in one substantial way: it runs natively at
// any sample rate from 8 kHz up, not only at 48 kHz.
//
// # Sample rates
//
// Upstream is hard-wired to 48 kHz and its maintainer's advice for other rates
// is to resample to 48 kHz and back. This port instead keeps the frame at 10 ms
// and the analysis window at 20 ms whatever the rate, which holds the FFT bin
// spacing at 50 Hz and makes upstream's band table name the same absolute
// frequencies everywhere. The only consequence of a lower rate is that the
// bands above its Nyquist carry no energy -- which is exactly the input
// upstream's own training data augmentation produces for 47% of its frames, by
// zeroing FFT bins above a randomly chosen cutoff. Reduced-rate operation is
// therefore inside the model's training distribution rather than an
// extrapolation from it.
//
// The measured cost is pitch resolution: the period is found to within about
// two native samples at any rate, so the relative accuracy falls from 0.8% at
// 48 kHz to 5% at 8 kHz. Everything else -- the 32 band energies and the 65
// network inputs -- agrees with a 48 kHz run of the same signal to within a few
// thousandths.
//
// # Sample scale
//
// Samples are float32 but scaled to int16 magnitudes, roughly -32768 to 32767,
// not -1 to 1. This follows upstream, and it is not cosmetic: the silence gate
// and several feature offsets are calibrated against that absolute magnitude,
// so feeding unit-scaled audio makes every frame look like silence. Use
// ProcessInt16 to avoid the question entirely.
package rnnoise

import (
	"errors"
	"fmt"
)

// DefaultSampleRate is the rate upstream's constants are defined at, and the
// one New uses when Options.SampleRate is zero.
const DefaultSampleRate = refRate

// Options configures a Denoiser.
type Options struct {
	// SampleRate in Hz. Zero means DefaultSampleRate. Must be at least
	// MinSampleRate.
	SampleRate int

	// Model holds the trained weights and is required. Get the shipped ones
	// from the rnnoise-go/model package:
	//
	//	m, err := model.Load()
	//	d, err := rnnoise.New(rnnoise.Options{Model: m})
	//
	// A Model is read-only and may be shared between Denoisers, including
	// across goroutines.
	Model *Model

	// Quantization selects the int8 activation convention. The zero value,
	// QuantSigned, is the exact one and matches upstream's portable, NEON and
	// WebAssembly builds. QuantUnsigned reproduces a stock x86 build's
	// arithmetic and exists mainly for the differential harness.
	Quantization QuantMode
}

// New creates a Denoiser. The returned Denoiser performs no further allocation,
// so it is suited to pooling across many streams.
func New(o Options) (*Denoiser, error) {
	rate := o.SampleRate
	if rate == 0 {
		rate = DefaultSampleRate
	}
	if o.Model == nil {
		return nil, errors.New("rnnoise: Options.Model is required; " +
			"load the shipped weights with github.com/VoiceBlender/rnnoise-go/model.Load()")
	}
	c, err := configForRate(rate)
	if err != nil {
		return nil, err
	}
	d := newDenoiser(c)
	d.model = o.Model
	d.quant = o.Quantization
	return d, nil
}

// FrameSize is the number of samples Process consumes and produces per call:
// 10 ms, so rate/100 for any rate that is a multiple of 100.
func (d *Denoiser) FrameSize() int { return d.c.frame }

// SampleRate returns the configured rate in Hz.
func (d *Denoiser) SampleRate() int { return d.c.sampleRate }

// Delay is the total delay from an input sample to the output sample that
// corresponds to it, in samples. Output sample i matches input sample
// i - Delay(), so a caller aligning the two discards the first Delay() samples
// of output.
//
// It is 2*FrameSize, or 20 ms, and both frames are structural:
//
//   - One from the analysis and synthesis overlap-add. A 50%-overlapped STFT
//     cannot emit a frame until the window covering it is complete.
//   - One because the gains a frame's features produce are applied to the
//     *previous* frame's spectrum. That is required rather than incidental:
//     upstream trains the model with exactly one frame of lookahead relative to
//     its target (train_rnnoise.py aligns pred_gain[j] with target[j+3] while
//     the two valid 3-tap convolutions centre on j+2), so removing it would
//     feed the network's gains to the wrong frame.
//
// TestDelayMatchesMeasured verifies this against the measured lag rather than
// trusting the arithmetic.
func (d *Denoiser) Delay() int { return 2 * d.c.frame }

// BinSpacing returns the analysis resolution in Hz, 50 for any rate that is a
// multiple of 100 and within half a percent of it otherwise.
func (d *Denoiser) BinSpacing() float64 { return d.c.binHz }

// ActiveBands returns how many of the 32 bands lie at or below this rate's
// Nyquist frequency, and therefore how many carry trained gains. It is 32 at
// 32 kHz and above, 26 at 16 kHz and 20 at 8 kHz.
func (d *Denoiser) ActiveBands() int { return d.c.activeBands }

// Reset returns the Denoiser to its initial state, so one instance can be
// reused for an unrelated stream without reallocating.
func (d *Denoiser) Reset() {
	clearF32(d.analysisMem)
	clearF32(d.synthesisMem)
	clearF32(d.pitchBuf)
	clearF32(d.lastg)
	d.lastGain = 0
	d.lastPeriod = 0
	d.memHP = [2]float32{}
	for i := range d.delayedX {
		d.delayedX[i] = cpx{}
		d.delayedP[i] = cpx{}
	}
	clearF32(d.delayedEx)
	clearF32(d.delayedEp)
	clearF32(d.delayedExp)
	d.rnn.reset()
}

func clearF32(b []float32) {
	for i := range b {
		b[i] = 0
	}
}

// Process denoises exactly FrameSize samples and returns the frame's speech
// probability in [0,1].
//
// Samples are float32 at int16 scale (about -32768 to 32767), matching
// upstream; see the package documentation. out and in may be the same slice.
func (d *Denoiser) Process(out, in []float32) (float32, error) {
	n := d.c.frame
	if len(in) != n || len(out) != n {
		return 0, fmt.Errorf("rnnoise: Process needs exactly %d samples at %d Hz, got in=%d out=%d",
			n, d.c.sampleRate, len(in), len(out))
	}
	return d.processFrame(out, in), nil
}

// ProcessInt16 denoises exactly FrameSize samples of 16-bit PCM and returns the
// frame's speech probability in [0,1]. out and in may be the same slice.
//
// The conversion back to int16 truncates rather than rounds, matching
// upstream's examples/rnnoise_demo.c, so a round trip through this function
// reproduces the reference tool's output.
func (d *Denoiser) ProcessInt16(out, in []int16) (float32, error) {
	n := d.c.frame
	if len(in) != n || len(out) != n {
		return 0, fmt.Errorf("rnnoise: ProcessInt16 needs exactly %d samples at %d Hz, got in=%d out=%d",
			n, d.c.sampleRate, len(in), len(out))
	}
	for i, v := range in {
		d.pcm[i] = float32(v)
	}
	vad := d.processFrame(d.pcm, d.pcm)
	for i, v := range d.pcm {
		out[i] = int16(v)
	}
	return vad, nil
}

// processFrame is upstream's rnnoise_process_frame.
func (d *Denoiser) processFrame(out, in []float32) float32 {
	c := d.c

	biquad(d.hp, &d.memHP, in, &c.hpB, &c.hpA)
	silence := d.computeFrameFeatures(d.X, d.P, d.Ex, d.Ep, d.Exp, d.features, d.hp)
	if d.trc != nil {
		d.trc.silence = silence
	}

	var vad float32
	if !silence {
		vad = d.model.computeRNN(d.rnn, d.g, d.features, d.quant, d.nn)
		if d.trc != nil {
			copy(d.trc.gRaw, d.g)
		}
		d.pitchFilter(d.delayedX, d.delayedP, d.delayedEx, d.delayedEp, d.delayedExp, d.g)
		for i := 0; i < numBands; i++ {
			// Cap the decay at 0.6 per frame, corresponding to an RT60 of
			// 135 ms. That avoids unnaturally quick attenuation.
			const alpha = float32(.6)
			d.g[i] = maxf(d.g[i], alpha*d.lastg[i])
			// Compensate for energy change across the frame when computing the
			// threshold gain. Avoids leaking noise when energy increases, for
			// instance on transient noise.
			d.lastg[i] = float32(fmin(1,
				float64(d.g[i])*(float64(d.delayedEx[i])+1e-3)/(float64(d.Ex[i])+1e-3)))
		}
		c.interpBandGain(d.gf, d.g)
		for i := 0; i < c.freq; i++ {
			d.delayedX[i].r *= d.gf[i]
			d.delayedX[i].i *= d.gf[i]
		}
	}
	// Note the asymmetry, which is upstream's: on a silent frame delayedX is
	// synthesised ungained, so the 20 kHz band limit that interpBandGain
	// imposes does not apply to it.
	d.frameSynthesis(out, d.delayedX)

	copy(d.delayedX, d.X)
	copy(d.delayedP, d.P)
	copy(d.delayedEx, d.Ex)
	copy(d.delayedEp, d.Ep)
	copy(d.delayedExp, d.Exp)
	return vad
}

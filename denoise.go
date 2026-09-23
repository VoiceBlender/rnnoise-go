package rnnoise

import "math"

// Ported from upstream's src/denoise.c. Upstream's DenoiseState also carries
// `int memid` and `float pitch_enh_buf[PITCH_BUF_SIZE]`, which are declared but
// never read anywhere in the current source; they are left out here.

// Denoiser is a single mono noise-suppression stream. It is not safe for
// concurrent use. After construction it performs no allocation, so a pool of
// Denoisers can serve many streams without GC pressure; use Reset to recycle
// one.
type Denoiser struct {
	c     *rateConfig
	model *Model
	quant QuantMode

	tr    *transform
	pitch *pitchState
	rnn   *rnnState
	nn    *nnScratch

	// Persistent state, mirroring upstream's DenoiseState.
	analysisMem  []float32
	synthesisMem []float32
	pitchBuf     []float32
	lastGain     float32
	lastPeriod   int
	memHP        [2]float32
	lastg        []float32

	// The gains a frame's features produce are applied to the *previous*
	// frame's spectrum. This is not an implementation detail to optimise
	// away: train_rnnoise.py aligns pred_gain[j] with target[j+3] while the
	// two valid 3-tap convolutions centre on j+2, so the model is trained
	// with exactly one frame of lookahead relative to its target. Removing
	// the delay would feed it gains for the wrong frame.
	delayedX   []cpx
	delayedP   []cpx
	delayedEx  []float32
	delayedEp  []float32
	delayedExp []float32

	// Per-frame scratch, sized once at construction.
	x        []float32
	hp       []float32
	p        []float32
	synth    []float32
	X        []cpx
	P        []cpx
	Ex       []float32
	Ep       []float32
	Exp      []float32
	Ly       []float32
	features []float32
	g        []float32
	gf       []float32
	rf       []float32
	normf    []float32
	r        []float32
	newE     []float32
	norm     []float32
	pcm      []float32

	// trc is nil in production; see trace.go.
	trc *trace
}

func newDenoiser(c *rateConfig) *Denoiser {
	d := &Denoiser{
		c:     c,
		tr:    newTransform(c),
		pitch: newPitchState(c),
		rnn:   newRNNState(),
		nn:    newNNScratch(),

		analysisMem:  make([]float32, c.frame),
		synthesisMem: make([]float32, c.frame),
		pitchBuf:     make([]float32, c.pitchBuf),
		lastg:        make([]float32, numBands),

		delayedX:   make([]cpx, c.freq),
		delayedP:   make([]cpx, c.freq),
		delayedEx:  make([]float32, numBands),
		delayedEp:  make([]float32, numBands),
		delayedExp: make([]float32, numBands),

		x:        make([]float32, c.window),
		hp:       make([]float32, c.frame),
		p:        make([]float32, c.window),
		synth:    make([]float32, c.window),
		X:        make([]cpx, c.freq),
		P:        make([]cpx, c.freq),
		Ex:       make([]float32, numBands),
		Ep:       make([]float32, numBands),
		Exp:      make([]float32, numBands),
		Ly:       make([]float32, numBands),
		features: make([]float32, numFeatures),
		g:        make([]float32, numBands),
		gf:       make([]float32, c.freq),
		rf:       make([]float32, c.freq),
		normf:    make([]float32, c.freq),
		r:        make([]float32, numBands),
		newE:     make([]float32, numBands),
		norm:     make([]float32, numBands),
		pcm:      make([]float32, c.frame),
	}
	return d
}

// frameAnalysis is upstream's rnn_frame_analysis.
func (d *Denoiser) frameAnalysis(X []cpx, Ex []float32, in []float32) {
	c := d.c
	copy(d.x[:c.frame], d.analysisMem)
	copy(d.x[c.frame:], in[:c.frame])
	copy(d.analysisMem, in[:c.frame])
	c.applyWindow(d.x)
	d.tr.forward(X, d.x)
	c.computeBandEnergy(Ex, X)
}

// computeFrameFeatures is upstream's rnn_compute_frame_features. It reports
// whether the frame is silent, in which case the features are zeroed and the
// caller must skip the network and freeze its state.
func (d *Denoiser) computeFrameFeatures(X, P []cpx, Ex, Ep, Exp, features, in []float32) bool {
	c := d.c
	d.frameAnalysis(X, Ex, in)

	copy(d.pitchBuf, d.pitchBuf[c.frame:])
	copy(d.pitchBuf[c.pitchBuf-c.frame:], in[:c.frame])

	ps := d.pitch
	ps.downsample(d.pitchBuf, ps.lp, c.pitchBuf)
	idx := ps.search(ps.lp[c.pitchMax>>1:], ps.lp, c.pitchFrame, c.pitchMax-3*c.pitchMin)
	idx = c.pitchMax - idx
	idx, gain := ps.removeDoubling(ps.lp, 0, c.pitchMax, c.pitchMin, c.pitchFrame,
		idx, d.lastPeriod, d.lastGain)
	d.lastPeriod = idx
	d.lastGain = gain

	base := c.pitchBuf - c.window - idx
	for i := 0; i < c.window; i++ {
		d.p[i] = d.pitchBuf[base+i]
	}
	c.applyWindow(d.p)
	d.tr.forward(P, d.p)
	c.computeBandEnergy(Ep, P)
	c.computeBandCorr(Exp, X, P)
	for i := 0; i < numBands; i++ {
		// C promotes to double here: .001 is a double literal and sqrt
		// returns double, so the division rounds once on store.
		Exp[i] = float32(float64(Exp[i]) / math.Sqrt(.001+float64(Ex[i]*Ep[i])))
	}
	c.dct(features[numBands:], Exp)

	// The model was trained on a pitch index counted in 48 kHz samples, so a
	// native period has to be converted. pitchScale is exactly 1 at the
	// reference rate, and the value is left un-rounded: the native lag is
	// already coarser than the 48 kHz grid, so rounding to an integer 48 kHz
	// index would only throw away precision the model can use.
	if c.pitchScale == 1 {
		features[2*numBands] = float32(.01 * float64(idx-300))
	} else {
		features[2*numBands] = float32(.01 * (float64(idx)*c.pitchScale - 300))
	}

	logMax := float32(-2)
	follow := float32(-2)
	var E float32
	for i := 0; i < numBands; i++ {
		ly := float32(math.Log10(1e-2 + float64(Ex[i])))
		// Dynamic-range follower: clamp each band to within 1.5 decades of
		// the previous band and 7 of the running maximum. At reduced rates
		// the bands above Nyquist have Ex == 0, so this walks them down to
		// the floor -- which is exactly what upstream computes for the
		// band-limited 47% of its training data.
		ly = float32(fmax(float64(logMax)-7, fmax(float64(follow)-1.5, float64(ly))))
		logMax = float32(fmax(float64(logMax), float64(ly)))
		follow = float32(fmax(float64(follow)-1.5, float64(ly)))
		d.Ly[i] = ly
		E += Ex[i]
	}
	// Because kiss_fft is 1/N-scaled, sum(Ex) measures signal power and does
	// not grow with the transform length, so this absolute threshold stays
	// correct at every rate. It trips at an RMS of about 0.4 LSB at the
	// +/-32768 scale, i.e. digital silence.
	if float64(E) < 0.04 {
		// If there's no audio, avoid messing up the state.
		for i := range features[:numFeatures] {
			features[i] = 0
		}
		return true
	}
	c.dct(features, d.Ly)
	features[0] -= 12
	features[1] -= 4
	return false
}

// pitchFilter is upstream's rnn_pitch_filter: a comb filter that adds a
// scaled copy of the pitch-predicted spectrum back in, then renormalises each
// band to its original energy.
func (d *Denoiser) pitchFilter(X []cpx, P []cpx, Ex, Ep, Exp, g []float32) {
	c := d.c
	r := d.r
	for i := 0; i < numBands; i++ {
		if Exp[i] > g[i] {
			r[i] = 1
		} else {
			e2 := Exp[i] * Exp[i]
			g2 := g[i] * g[i]
			r[i] = float32(float64(e2*(1-g2)) / (.001 + float64(g2*(1-e2))))
		}
		r[i] = float32(math.Sqrt(float64(minf(1, maxf(0, r[i])))))
		r[i] = float32(float64(r[i]) * math.Sqrt(float64(Ex[i])/(1e-8+float64(Ep[i]))))
	}
	c.interpBandGain(d.rf, r)
	for i := 0; i < c.freq; i++ {
		X[i].r += d.rf[i] * P[i].r
		X[i].i += d.rf[i] * P[i].i
	}
	c.computeBandEnergy(d.newE, X)
	for i := 0; i < numBands; i++ {
		d.norm[i] = float32(math.Sqrt(float64(Ex[i]) / (1e-8 + float64(d.newE[i]))))
	}
	c.interpBandGain(d.normf, d.norm)
	for i := 0; i < c.freq; i++ {
		X[i].r *= d.normf[i]
		X[i].i *= d.normf[i]
	}
}

// frameSynthesis is upstream's frame_synthesis: inverse transform, window,
// overlap-add against the previous frame's tail.
func (d *Denoiser) frameSynthesis(out []float32, y []cpx) {
	c := d.c
	d.tr.inverse(d.synth, y)
	c.applyWindow(d.synth)
	for i := 0; i < c.frame; i++ {
		out[i] = d.synth[i] + d.synthesisMem[i]
	}
	copy(d.synthesisMem, d.synth[c.frame:])
}

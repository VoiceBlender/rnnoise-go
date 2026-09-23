package rnnoise

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"testing"
)

// Per-stage differential comparison against the C reference, following the
// goamr-nb/dsp_diff_test.go idiom: shell out to an externally built C binary
// and compare, skipping unless the harness is present.
//
//	make -C ctest/c        # builds both variants
//	make diff-vs-c         # runs this
//
// Two C variants exist because upstream is not one reference but three
// numerically distinct ones:
//
//   - the scalar build (RNNOISE_C_STAGEDUMP) quantises activations to signed
//     int8 with `bias` and evaluates tanh/sigmoid with a real divide. It is
//     bit-comparable, and it is what the Go port's default QuantSigned mode
//     targets.
//   - the AVX2 build (RNNOISE_C_STAGEDUMP_AVX2) quantises to unsigned with
//     `subias`, uses saturating int16 intermediates, and approximates the
//     activation denominator with _mm256_rcp_ps. That last one is
//     microarchitecture-defined -- AMD and Intel return different bits -- so
//     bit-exactness against it is impossible by construction, and this file
//     compares it under a tolerance instead.

const stageMagic = 0x47545352 // "RSTG"

// stageDims mirrors the geometry header the C dumper emits, so a mismatch in
// any compile-time constant is caught before a single value is compared.
type stageDims struct {
	frame, freq, bands, features, conv1Out, conv2Out, gruOut int32
}

// stageFrame is one frame's worth of reference intermediates. The field order
// is the record layout in ctest/c/stagedump.c.
type stageFrame struct {
	index      int32
	silence    int32
	pitchIndex int32
	pitchGain  float32
	vad        float32

	hp       []float32
	xr, xi   []float32
	ex       []float32
	pr, pi   []float32
	ep       []float32
	exp      []float32
	features []float32
	conv1Out []float32
	conv2Out []float32
	gru1     []float32
	gru2     []float32
	gru3     []float32
	gRaw     []float32
	gFinal   []float32
	gf       []float32
	out      []float32
}

type stageReader struct {
	r    *bufio.Reader
	dims stageDims
	buf  []byte
	f    stageFrame
}

func newStageReader(r io.Reader) (*stageReader, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	var hdr [32]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, fmt.Errorf("reading geometry header: %w", err)
	}
	if got := binary.LittleEndian.Uint32(hdr[0:]); got != stageMagic {
		return nil, fmt.Errorf("bad geometry magic 0x%08x", got)
	}
	sr := &stageReader{r: br}
	d := &sr.dims
	for i, p := range []*int32{&d.frame, &d.freq, &d.bands, &d.features, &d.conv1Out, &d.conv2Out, &d.gruOut} {
		*p = int32(binary.LittleEndian.Uint32(hdr[4+4*i:]))
	}
	f := &sr.f
	f.hp = make([]float32, d.frame)
	f.xr = make([]float32, d.freq)
	f.xi = make([]float32, d.freq)
	f.ex = make([]float32, d.bands)
	f.pr = make([]float32, d.freq)
	f.pi = make([]float32, d.freq)
	f.ep = make([]float32, d.bands)
	f.exp = make([]float32, d.bands)
	f.features = make([]float32, d.features)
	f.conv1Out = make([]float32, d.conv1Out)
	f.conv2Out = make([]float32, d.conv2Out)
	f.gru1 = make([]float32, d.gruOut)
	f.gru2 = make([]float32, d.gruOut)
	f.gru3 = make([]float32, d.gruOut)
	f.gRaw = make([]float32, d.bands)
	f.gFinal = make([]float32, d.bands)
	f.gf = make([]float32, d.freq)
	f.out = make([]float32, d.frame)

	n := 24
	for _, v := range sr.vectors() {
		n += 4 * len(v)
	}
	sr.buf = make([]byte, n)
	return sr, nil
}

// vectors lists the float slices in record order.
func (sr *stageReader) vectors() [][]float32 {
	f := &sr.f
	return [][]float32{
		f.hp, f.xr, f.xi, f.ex, f.pr, f.pi, f.ep, f.exp, f.features,
		f.conv1Out, f.conv2Out, f.gru1, f.gru2, f.gru3,
		f.gRaw, f.gFinal, f.gf, f.out,
	}
}

// next reads one frame, reusing the reader's buffers. Streaming rather than
// slurping matters: a 60 s comparison is 126 MB of records.
func (sr *stageReader) next() (*stageFrame, error) {
	if _, err := io.ReadFull(sr.r, sr.buf); err != nil {
		return nil, err
	}
	b := sr.buf
	if got := binary.LittleEndian.Uint32(b[0:]); got != stageMagic {
		return nil, fmt.Errorf("bad record magic 0x%08x (record desync)", got)
	}
	f := &sr.f
	f.index = int32(binary.LittleEndian.Uint32(b[4:]))
	f.silence = int32(binary.LittleEndian.Uint32(b[8:]))
	f.pitchIndex = int32(binary.LittleEndian.Uint32(b[12:]))
	f.pitchGain = math.Float32frombits(binary.LittleEndian.Uint32(b[16:]))
	f.vad = math.Float32frombits(binary.LittleEndian.Uint32(b[20:]))
	off := 24
	for _, v := range sr.vectors() {
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[off:]))
			off += 4
		}
	}
	return f, nil
}

// devs accumulates a stage's deviation statistics across all frames.
type devs struct {
	name     string
	maxAbs   float64
	maxRel   float64
	at       string
	nonZero  int
	compared int

	// Signal and error energy, for stages where a peak deviation says little.
	sigE, errE float64
}

// snr returns the reference-to-error ratio in dB, or +Inf if identical.
func (d *devs) snr() float64 {
	if d.errE == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(d.sigE/d.errE)
}

func (d *devs) add(frame, i int, got, want float32) {
	d.compared++
	d.sigE += float64(want) * float64(want)
	d.errE += (float64(got) - float64(want)) * (float64(got) - float64(want))
	if got == want {
		return
	}
	d.nonZero++
	a := math.Abs(float64(got) - float64(want))
	scale := math.Abs(float64(want))
	rel := a
	if scale > 1e-12 {
		rel = a / scale
	}
	if a > d.maxAbs {
		d.maxAbs = a
		d.at = fmt.Sprintf("frame %d index %d: got %v want %v", frame, i, got, want)
	}
	if rel > d.maxRel {
		d.maxRel = rel
	}
}

func (d *devs) addVec(frame int, got, want []float32) {
	n := len(want)
	if len(got) < n {
		n = len(got)
	}
	for i := 0; i < n; i++ {
		d.add(frame, i, got[i], want[i])
	}
}

// makeTestPCM produces a deterministic noisy 48 kHz signal covering the cases
// that matter: silence (to exercise the gate and the frozen-state path), voiced
// speech-like content, broadband noise, and a full-scale transient.
func makeTestPCM(frames int) []int16 {
	const rate = 48000
	n := frames * (rate / 100)
	out := make([]int16, n)
	rng := rand.New(rand.NewSource(20250923))
	for i := range out {
		t := float64(i) / rate
		var v float64
		switch {
		case i < 5*(rate/100):
			// Digital silence: the gate must trip and the state freeze.
			v = 0
		case i%(rate*3) < rate/2:
			// Noise only.
			v = 900 * rng.NormFloat64()
		default:
			env := math.Max(0, math.Sin(2*math.Pi*2.3*t))
			for h := 1; h <= 40; h++ {
				f := 135.0 * float64(h)
				if f > 20000 {
					break
				}
				v += 2500 * env / float64(h) * math.Cos(2*math.Pi*f*t+float64(h*h%11))
			}
			v += 700 * rng.NormFloat64()
		}
		// One full-scale transient, to exercise clipping behaviour.
		if i == 40*(rate/100) {
			v = 32767
		}
		if v > 32767 {
			v = 32767
		}
		if v < -32768 {
			v = -32768
		}
		out[i] = int16(v)
	}
	return out
}

// TestDiffVsC is the credibility gate for the whole port: a per-stage
// comparison against the C reference over a multi-second signal.
func TestDiffVsC(t *testing.T) {
	// The AVX2 comparison is gated on signal-to-error ratio against measured
	// floors, not on peak deviation, and the floors are much lower than they
	// first look like they should be. Three findings establish why, in order of
	// how easy each is to get wrong.
	//
	// 1. GCC FMA contraction, not SIMD dispatch, moves the DSP front end.
	//    Building the same C source with -mavx -mfma -mavx2 changes the FFT,
	//    band energies and features by 1-2 ulp even though upstream gives none
	//    of them a SIMD variant, because the compiler contracts a*b+c into FMA
	//    in plain C (helped along by nnet_arch.h's own
	//    #pragma GCC optimize("tree-vectorize")). `make -C ctest/c nofma`
	//    rebuilds that variant with -ffp-contract=off, and against it the whole
	//    front end is bit-identical to the scalar build. So this divergence is
	//    the compiler's, and no port can avoid it.
	//
	// 2. From conv1 onward the differences are real and unavoidable: unsigned
	//    activations with subias, saturating int16 intermediates via
	//    VPMADDUBSW, and _mm256_rcp_ps instead of a divide in the activations.
	//    The reciprocal approximation is microarchitecture-defined -- AMD and
	//    Intel return different bits -- so this can never close.
	//
	// 3. Three stacked recurrent layers amplify both, hard: 142 dB at the
	//    features becomes 60 dB after conv2 and 31 dB at the output.
	//
	// The floors below therefore come from a control experiment rather than
	// from taste. Comparing C-scalar against C-avx2 on this exact 600-frame
	// input -- two builds of the *same source*, so a lower bound on what any
	// implementation can achieve -- gives:
	//
	//	features 142.6   conv1_out 80.1   conv2_out 60.3   gru1 46.0
	//	gru2 37.9        gru3 36.4        g_final 29.3     out 30.5 dB
	//
	// Go's unsigned mode measures at or slightly better than that baseline on
	// every stage (out: 30.9 dB). The floors sit just under the baseline, so
	// this test fails if the port ever becomes meaningfully worse than a
	// differently-compiled build of the reference, which is the strongest claim
	// available once bit-exactness is off the table.
	cases := []struct {
		env   string
		quant QuantMode
		exact bool
		snr   map[string]float64 // min signal-to-error ratio in dB
	}{
		{
			env:   "RNNOISE_C_STAGEDUMP",
			quant: QuantSigned,
			exact: true,
		},
		{
			env:   "RNNOISE_C_STAGEDUMP_AVX2",
			quant: QuantUnsigned,
			exact: false,
			snr: map[string]float64{
				// Front end: FMA contraction only.
				"hp":   math.Inf(1),
				"X.re": 135, "X.im": 135, "P.re": 135, "P.im": 135,
				"Ex": 138, "Ep": 138, "Exp": 134, "features": 136,
				"pitch_gain": 128,
				// Network: quantisation convention plus rcpps, amplified.
				"conv1_out": 77, "conv2_out": 57,
				"gru1_state": 43, "gru2_state": 35, "gru3_state": 34,
				"g_raw": 27, "g_final": 27, "gf": 26,
				"vad": 32, "out": 28,
			},
		},
	}

	if !bitExactArch() {
		t.Skip(exactnessSkipReason)
	}
	ran := 0
	for _, tc := range cases {
		bin := os.Getenv(tc.env)
		if bin == "" {
			t.Logf("%s not set; skipping the %s comparison", tc.env, tc.quant)
			continue
		}
		if _, err := os.Stat(bin); err != nil {
			t.Fatalf("%s=%q: %v (run `make -C ctest/c`)", tc.env, bin, err)
		}
		ran++
		t.Run(tc.quant.String(), func(t *testing.T) {
			runDiff(t, bin, tc.quant, tc.exact, tc.snr)
		})
	}
	if ran == 0 {
		t.Skip("no C reference configured; set RNNOISE_C_STAGEDUMP (see make diff-vs-c)")
	}
}

func runDiff(t *testing.T, bin string, quant QuantMode, exact bool, snrLimits map[string]float64) {
	t.Helper()

	const frames = 600 // 6 s at 48 kHz
	pcm := makeTestPCM(frames)

	sr, cleanup, err := runStageDump(t, bin, frames)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	// The geometry header is the first thing that would catch a constant drift.
	c, err := configForRate(48000)
	if err != nil {
		t.Fatal(err)
	}
	if int(sr.dims.frame) != c.frame || int(sr.dims.freq) != c.freq ||
		int(sr.dims.bands) != numBands || int(sr.dims.features) != numFeatures ||
		int(sr.dims.conv1Out) != conv1OutSize || int(sr.dims.conv2Out) != conv2OutSize ||
		int(sr.dims.gruOut) != gruSize {
		t.Fatalf("geometry mismatch: C says frame=%d freq=%d bands=%d features=%d conv1=%d conv2=%d gru=%d",
			sr.dims.frame, sr.dims.freq, sr.dims.bands, sr.dims.features,
			sr.dims.conv1Out, sr.dims.conv2Out, sr.dims.gruOut)
	}

	m, err := LoadModelFile("model/weights.bin")
	if err != nil {
		t.Skipf("model/weights.bin not present: %v", err)
	}
	d, err := New(Options{SampleRate: 48000, Model: m, Quantization: quant})
	if err != nil {
		t.Fatal(err)
	}
	d.enableTrace()

	stats := map[string]*devs{}
	dev := func(name string) *devs {
		if s, ok := stats[name]; ok {
			return s
		}
		s := &devs{name: name}
		stats[name] = s
		return s
	}

	in := make([]float32, c.frame)
	out := make([]float32, c.frame)
	xr := make([]float32, c.freq)
	xi := make([]float32, c.freq)
	pr := make([]float32, c.freq)
	pi := make([]float32, c.freq)

	var silenceMismatch, pitchMismatch, nFrames int
	for f := 0; ; f++ {
		ref, err := sr.next()
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			t.Fatalf("frame %d: %v", f, err)
		}
		if int(ref.index) != f {
			t.Fatalf("frame index desync: record says %d, expected %d", ref.index, f)
		}
		base := f * c.frame
		if base+c.frame > len(pcm) {
			break
		}
		for i := range in {
			in[i] = float32(pcm[base+i])
		}
		vad, err := d.Process(out, in)
		if err != nil {
			t.Fatal(err)
		}
		nFrames++

		// Discrete stages first: a mismatch here invalidates everything after
		// it in the same frame, so report and keep going rather than flooding.
		silence := int32(0)
		if d.trc.silence {
			silence = 1
		}
		if silence != ref.silence {
			silenceMismatch++
			if silenceMismatch <= 3 {
				t.Errorf("frame %d: silence flag %d, C says %d", f, silence, ref.silence)
			}
		}
		if int32(d.lastPeriod) != ref.pitchIndex {
			pitchMismatch++
			if pitchMismatch <= 3 {
				t.Errorf("frame %d: pitch index %d, C says %d", f, d.lastPeriod, ref.pitchIndex)
			}
		}
		dev("pitch_gain").add(f, 0, d.lastGain, ref.pitchGain)
		dev("vad").add(f, 0, vad, ref.vad)

		for i := 0; i < c.freq; i++ {
			xr[i], xi[i] = d.X[i].r, d.X[i].i
			pr[i], pi[i] = d.P[i].r, d.P[i].i
		}
		dev("hp").addVec(f, d.hp, ref.hp)
		dev("X.re").addVec(f, xr, ref.xr)
		dev("X.im").addVec(f, xi, ref.xi)
		dev("Ex").addVec(f, d.Ex, ref.ex)
		dev("P.re").addVec(f, pr, ref.pr)
		dev("P.im").addVec(f, pi, ref.pi)
		dev("Ep").addVec(f, d.Ep, ref.ep)
		dev("Exp").addVec(f, d.Exp, ref.exp)
		dev("features").addVec(f, d.features, ref.features)
		dev("conv1_out").addVec(f, d.nn.tmp, ref.conv1Out)
		dev("conv2_out").addVec(f, d.nn.cat[:conv2OutSize], ref.conv2Out)
		dev("gru1_state").addVec(f, d.rnn.gru1State, ref.gru1)
		dev("gru2_state").addVec(f, d.rnn.gru2State, ref.gru2)
		dev("gru3_state").addVec(f, d.rnn.gru3State, ref.gru3)
		dev("g_raw").addVec(f, d.trc.gRaw, ref.gRaw)
		dev("g_final").addVec(f, d.g, ref.gFinal)
		// gf is only defined on frames the network ran on. The silence gate
		// skips interp_band_gain entirely, leaving C's gf at its stack
		// initialiser -- `float gf[FREQ_SIZE] = {1}`, so gf[0] == 1 -- and
		// leaving Go's reused buffer at whatever the previous frame left. The
		// value is never read on such a frame, and `out` matching exactly is
		// the proof of that, so comparing it would be comparing noise.
		if ref.silence == 0 {
			dev("gf").addVec(f, d.gf, ref.gf)
		}
		dev("out").addVec(f, out, ref.out)
	}

	if nFrames < frames/2 {
		t.Fatalf("only %d frames compared, expected about %d", nFrames, frames)
	}
	if silenceMismatch != 0 {
		t.Errorf("silence flag differed on %d/%d frames", silenceMismatch, nFrames)
	}
	if pitchMismatch != 0 {
		t.Errorf("pitch index differed on %d/%d frames", pitchMismatch, nFrames)
	}

	names := make([]string, 0, len(stats))
	for n := range stats {
		names = append(names, n)
	}
	sort.Strings(names)

	t.Logf("%d frames compared, %s quantisation", nFrames, quant)
	for _, n := range names {
		s := stats[n]
		verdict := "exact"
		if s.nonZero > 0 {
			verdict = fmt.Sprintf("maxabs %.3g  snr %5.1f dB  on %d/%d values",
				s.maxAbs, s.snr(), s.nonZero, s.compared)
		}
		t.Logf("  %-11s %s", n, verdict)

		if exact {
			if s.nonZero > 0 {
				t.Errorf("stage %s must match C exactly: %s (first worst: %s)", n, verdict, s.at)
			}
			continue
		}
		want, ok := snrLimits[n]
		if !ok {
			t.Errorf("stage %s has no SNR floor configured (deviation %s)", n, verdict)
			continue
		}
		if got := s.snr(); got < want {
			t.Errorf("stage %s: SNR %.1f dB against the reference, floor %.1f dB (worst: %s)",
				n, got, want, s.at)
		}
	}
}

// runStageDump starts the C reference on the fixture signal and returns a reader
// over its per-stage records.
//
// The input goes through a file rather than a pipe: the dumper writes about
// 21 KB per frame to stdout, so feeding its stdin concurrently would risk a
// deadlock for no benefit.
func runStageDump(t *testing.T, bin string, frames int) (*stageReader, func(), error) {
	t.Helper()
	pcm := makeTestPCM(frames)
	raw := make([]byte, 2*len(pcm))
	for i, v := range pcm {
		binary.LittleEndian.PutUint16(raw[2*i:], uint16(v))
	}
	inPath := filepath.Join(t.TempDir(), "in.pcm")
	if err := os.WriteFile(inPath, raw, 0o644); err != nil {
		return nil, nil, err
	}
	fin, err := os.Open(inPath)
	if err != nil {
		return nil, nil, err
	}
	cmd := exec.Command(bin)
	cmd.Stdin = fin
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fin.Close()
		return nil, nil, err
	}
	if err := cmd.Start(); err != nil {
		fin.Close()
		return nil, nil, err
	}
	cleanup := func() {
		stdout.Close()
		cmd.Wait()
		fin.Close()
	}
	sr, err := newStageReader(stdout)
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	return sr, cleanup, nil
}

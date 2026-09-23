package rnnoise

import (
	"math"
	"math/rand"
	"os"
	"sync"
	"testing"
)

// testModel loads the shipped weights by path rather than importing the model
// package, which would be an import cycle from inside package rnnoise.
func testModel(t *testing.T) *Model {
	t.Helper()
	m, err := LoadModelFile("model/weights.bin")
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("model/weights.bin not present; run `make model`")
		}
		t.Fatalf("loading model: %v", err)
	}
	return m
}

// speechLike renders a crude but voiced, formant-shaped signal: a 120 Hz
// fundamental with harmonics shaped by three formant resonances, amplitude
// modulated at syllable rate. It is not speech, but it has the harmonic
// structure and spectral tilt the network keys on.
func speechLike(rate, n int, seed int64) []float32 {
	x := make([]float32, n)
	formants := [3][2]float64{{700, 90}, {1220, 110}, {2600, 170}} // centre, bandwidth
	for i := range x {
		t := float64(i) / float64(rate)
		// Syllabic envelope with pauses, so the VAD has something to do.
		env := math.Max(0, math.Sin(2*math.Pi*2.5*t))
		var v float64
		for h := 1; h <= 60; h++ {
			f := 120.0 * float64(h)
			if f > float64(rate)/2*0.95 {
				break
			}
			// Sum of resonances evaluated at this harmonic.
			var g float64
			for _, fm := range formants {
				d := (f - fm[0]) / fm[1]
				g += 1 / (1 + d*d)
			}
			v += g / float64(h) * math.Cos(2*math.Pi*f*t+float64(h*h%7))
		}
		x[i] = float32(3000 * env * v)
	}
	return x
}

// TestEndToEnd is the Phase 4 milestone: the whole pipeline runs, stays finite,
// preserves the frame contract, and actually attenuates noise while passing
// speech.
func TestEndToEnd(t *testing.T) {
	m := testModel(t)

	for _, rate := range testRates {
		d, err := New(Options{SampleRate: rate, Model: m})
		if err != nil {
			t.Fatalf("rate %d: %v", rate, err)
		}
		if d.FrameSize() != rate/100 && rate%100 == 0 {
			t.Errorf("rate %d: FrameSize = %d", rate, d.FrameSize())
		}

		const seconds = 3
		n := seconds * rate
		n -= n % d.FrameSize()
		speech := speechLike(rate, n, 1)

		rng := rand.New(rand.NewSource(int64(rate)))
		noisy := make([]float32, n)
		noise := make([]float32, n)
		for i := range noisy {
			noise[i] = float32(600 * rng.NormFloat64())
			noisy[i] = speech[i] + noise[i]
		}

		out := make([]float32, n)
		vads := make([]float32, 0, n/d.FrameSize())
		for off := 0; off+d.FrameSize() <= n; off += d.FrameSize() {
			vad, err := d.Process(out[off:off+d.FrameSize()], noisy[off:off+d.FrameSize()])
			if err != nil {
				t.Fatalf("rate %d: %v", rate, err)
			}
			vads = append(vads, vad)
			if vad < 0 || vad > 1 || math.IsNaN(float64(vad)) {
				t.Fatalf("rate %d: vad = %v out of [0,1]", rate, vad)
			}
		}
		for i, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("rate %d: out[%d] = %v", rate, i, v)
			}
		}

		// Compare energy in the pauses (noise only) against the peaks
		// (speech + noise), skipping the first few frames of warm-up and
		// accounting for the one-frame algorithmic delay.
		skip := 10 * d.FrameSize()
		var quietIn, quietOut, loudIn, loudOut float64
		for i := skip; i < n-d.Delay(); i++ {
			env := math.Max(0, math.Sin(2*math.Pi*2.5*float64(i)/float64(rate)))
			o := float64(out[i+d.Delay()])
			in := float64(noisy[i])
			if env < 0.05 {
				quietIn += in * in
				quietOut += o * o
			} else if env > 0.7 {
				loudIn += in * in
				loudOut += o * o
			}
		}
		noiseRed := 10 * math.Log10(quietOut/quietIn)
		speechRed := 10 * math.Log10(loudOut/loudIn)

		// The network must suppress the noise-only stretches substantially
		// more than the speech stretches. Absolute figures depend on the
		// synthetic signal, so the test asserts the gap, which is the thing
		// that makes it a denoiser rather than an attenuator.
		if noiseRed > -6 {
			t.Errorf("rate %d: noise-only attenuation only %.1f dB, want at least 6 dB", rate, noiseRed)
		}
		if speechRed < -6 {
			t.Errorf("rate %d: speech attenuated by %.1f dB, too much", rate, speechRed)
		}
		if noiseRed > speechRed-6 {
			t.Errorf("rate %d: noise %.1f dB vs speech %.1f dB; gap too small", rate, noiseRed, speechRed)
		}

		// The VAD must discriminate: higher during speech than in pauses.
		var vSpeech, nSpeech, vPause, nPause float64
		for k, v := range vads {
			i := k * d.FrameSize()
			if i < skip {
				continue
			}
			env := math.Max(0, math.Sin(2*math.Pi*2.5*float64(i)/float64(rate)))
			if env < 0.05 {
				vPause += float64(v)
				nPause++
			} else if env > 0.7 {
				vSpeech += float64(v)
				nSpeech++
			}
		}
		if nSpeech > 0 && nPause > 0 {
			vSpeech /= nSpeech
			vPause /= nPause
			if vSpeech < vPause {
				t.Errorf("rate %d: mean VAD %.3f in speech vs %.3f in pauses", rate, vSpeech, vPause)
			}
		}

		t.Logf("rate %6d: noise %+.1f dB, speech %+.1f dB, VAD speech %.3f / pause %.3f, activeBands %d",
			rate, noiseRed, speechRed, vSpeech, vPause, d.ActiveBands())
	}
}

// TestProcessInPlace checks the documented aliasing guarantee.
func TestProcessInPlace(t *testing.T) {
	m := testModel(t)
	d1, _ := New(Options{Model: m})
	d2, _ := New(Options{Model: m})
	n := d1.FrameSize()

	rng := rand.New(rand.NewSource(4))
	sig := make([]float32, 40*n)
	for i := range sig {
		sig[i] = float32(3000 * rng.NormFloat64())
	}

	sep := make([]float32, n)
	inPlace := make([]float32, n)
	for off := 0; off+n <= len(sig); off += n {
		copy(inPlace, sig[off:off+n])
		v1, err := d1.Process(sep, sig[off:off+n])
		if err != nil {
			t.Fatal(err)
		}
		v2, err := d2.Process(inPlace, inPlace)
		if err != nil {
			t.Fatal(err)
		}
		if v1 != v2 {
			t.Fatalf("offset %d: vad %v in-place vs %v separate", off, v2, v1)
		}
		for i := range sep {
			if sep[i] != inPlace[i] {
				t.Fatalf("offset %d sample %d: %v in-place vs %v separate", off, i, inPlace[i], sep[i])
			}
		}
	}
}

// TestResetRestoresInitialState checks a pooled Denoiser is genuinely clean
// after Reset, by comparing a fresh instance against a reused one.
func TestResetRestoresInitialState(t *testing.T) {
	m := testModel(t)
	fresh, _ := New(Options{Model: m})
	reused, _ := New(Options{Model: m})
	n := fresh.FrameSize()

	rng := rand.New(rand.NewSource(6))
	sig := make([]float32, 30*n)
	for i := range sig {
		sig[i] = float32(2500 * rng.NormFloat64())
	}

	// Dirty the reused instance with unrelated audio, then reset it.
	scratch := make([]float32, n)
	for off := 0; off+n <= len(sig); off += n {
		copy(scratch, sig[off:off+n])
		for i := range scratch {
			scratch[i] = -scratch[i] * 3
		}
		if _, err := reused.Process(scratch, scratch); err != nil {
			t.Fatal(err)
		}
	}
	reused.Reset()

	a := make([]float32, n)
	b := make([]float32, n)
	for off := 0; off+n <= len(sig); off += n {
		va, err := fresh.Process(a, sig[off:off+n])
		if err != nil {
			t.Fatal(err)
		}
		vb, err := reused.Process(b, sig[off:off+n])
		if err != nil {
			t.Fatal(err)
		}
		if va != vb {
			t.Fatalf("offset %d: vad %v after Reset vs %v fresh", off, vb, va)
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("offset %d sample %d: %v after Reset vs %v fresh", off, i, b[i], a[i])
			}
		}
	}
}

// TestSilentInputStaysSilent checks digital silence in gives digital silence
// out, which the silence gate is supposed to guarantee.
func TestSilentInputStaysSilent(t *testing.T) {
	m := testModel(t)
	for _, rate := range testRates {
		d, _ := New(Options{SampleRate: rate, Model: m})
		n := d.FrameSize()
		in := make([]float32, n)
		out := make([]float32, n)
		for f := 0; f < 20; f++ {
			vad, err := d.Process(out, in)
			if err != nil {
				t.Fatal(err)
			}
			if vad != 0 {
				t.Errorf("rate %d frame %d: vad = %v on silence", rate, f, vad)
			}
			for i, v := range out {
				if v != 0 {
					t.Fatalf("rate %d frame %d: out[%d] = %v on silence", rate, f, i, v)
				}
			}
		}
	}
}

// TestNewRejects covers the argument validation.
func TestNewRejects(t *testing.T) {
	m := testModel(t)
	if _, err := New(Options{}); err == nil {
		t.Error("New with no Model succeeded")
	}
	if _, err := New(Options{Model: m, SampleRate: 4000}); err == nil {
		t.Error("New at 4 kHz succeeded")
	}
	d, err := New(Options{Model: m})
	if err != nil {
		t.Fatal(err)
	}
	if d.SampleRate() != DefaultSampleRate {
		t.Errorf("default rate = %d, want %d", d.SampleRate(), DefaultSampleRate)
	}
	short := make([]float32, d.FrameSize()-1)
	if _, err := d.Process(short, short); err == nil {
		t.Error("Process with a short frame succeeded")
	}
}

// TestConcurrentDenoisers checks the documented sharing contract: a Model is
// read-only and may back any number of Denoisers on any number of goroutines,
// even though a single Denoiser is not itself safe for concurrent use.
//
// Run under -race this also guards against a kernel reintroducing
// package-level mutable scratch, which would be invisible in single-threaded
// tests and corrupt audio at a few hundred concurrent streams.
func TestConcurrentDenoisers(t *testing.T) {
	m := testModel(t)

	const streams = 16
	const frames = 40

	// Reference results, computed serially.
	want := make([][]float32, streams)
	for s := 0; s < streams; s++ {
		d, err := New(Options{SampleRate: 48000, Model: m})
		if err != nil {
			t.Fatal(err)
		}
		want[s] = runStream(t, d, s, frames)
	}

	got := make([][]float32, streams)
	var wg sync.WaitGroup
	for s := 0; s < streams; s++ {
		wg.Add(1)
		go func(s int) {
			defer wg.Done()
			d, err := New(Options{SampleRate: 48000, Model: m})
			if err != nil {
				t.Error(err)
				return
			}
			got[s] = runStream(t, d, s, frames)
		}(s)
	}
	wg.Wait()

	for s := 0; s < streams; s++ {
		if len(got[s]) != len(want[s]) {
			t.Fatalf("stream %d: got %d samples, want %d", s, len(got[s]), len(want[s]))
		}
		for i := range want[s] {
			if got[s][i] != want[s][i] {
				t.Fatalf("stream %d sample %d: concurrent %v, serial %v",
					s, i, got[s][i], want[s][i])
			}
		}
	}
}

// runStream drives one Denoiser over a seed-dependent signal and returns the
// concatenated output.
func runStream(t *testing.T, d *Denoiser, seed, frames int) []float32 {
	t.Helper()
	n := d.FrameSize()
	rng := rand.New(rand.NewSource(int64(seed) + 1))
	in := make([]float32, n)
	out := make([]float32, n)
	all := make([]float32, 0, frames*n)
	for f := 0; f < frames; f++ {
		for i := range in {
			in[i] = float32(3000*rng.NormFloat64() + 2000*math.Sin(float64(f*n+i)*0.01))
		}
		if _, err := d.Process(out, in); err != nil {
			t.Fatal(err)
		}
		all = append(all, out...)
	}
	return all
}

// TestDelayMatchesMeasured verifies Delay() against the lag actually observed,
// rather than against the reasoning that produced the number.
//
// This exists because the value is easy to get wrong in a way nothing else
// catches: an off-by-one-frame error leaves the audio perfectly intelligible and
// only shows up when something tries to align input against output, at which
// point every objective measurement silently becomes meaningless. The
// native-rate quality harness was measuring -2.4 dB segmental SNR on clean
// speech for exactly this reason before the delay was corrected here.
func TestDelayMatchesMeasured(t *testing.T) {
	m := testModel(t)
	for _, rate := range testRates {
		d, err := New(Options{SampleRate: rate, Model: m})
		if err != nil {
			t.Fatal(err)
		}
		n := d.FrameSize()
		frames := 60
		in := speechLike(rate, frames*n, 1)
		out := make([]float32, frames*n)
		buf := make([]float32, n)
		for f := 0; f < frames; f++ {
			if _, err := d.Process(buf, in[f*n:(f+1)*n]); err != nil {
				t.Fatal(err)
			}
			copy(out[f*n:], buf)
		}

		// Find the shift that best correlates output with input.
		best, bestLag := -math.MaxFloat64, 0
		for lag := 0; lag <= 4*n; lag++ {
			var acc, ea, eb float64
			count := 0
			for i := 0; i+lag < len(out); i++ {
				a := float64(in[i])
				b := float64(out[i+lag])
				acc += a * b
				ea += a * a
				eb += b * b
				count++
			}
			if count == 0 || ea == 0 || eb == 0 {
				continue
			}
			if c := acc / math.Sqrt(ea*eb); c > best {
				best, bestLag = c, lag
			}
		}
		if bestLag != d.Delay() {
			t.Errorf("rate %d: measured delay %d samples (correlation %.3f), Delay() reports %d",
				rate, bestLag, best, d.Delay())
		}
		if best < 0.8 {
			t.Errorf("rate %d: best correlation only %.3f; the output does not track the input",
				rate, best)
		}
	}
}

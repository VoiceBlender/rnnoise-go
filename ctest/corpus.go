package ctest

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
)

// The corpus for the native-rate quality comparison, and an honest account of
// its limits.
//
// # What is available
//
// The workspace supplies real material: eight 8 kHz telephony recordings, two
// 16 kHz turn-detection clips, and about two hours of real 16 kHz noise
// (call-centre babble and street noise).
//
// # What that does and does not support
//
// Every one of those sources is band-limited at or below 8 kHz. So the real
// corpus can answer the question that matters for telephony and VoIP -- what
// does native 8 or 16 kHz operation cost against resampling to 48 kHz and back
// -- and it cannot say anything about genuinely wideband content, because there
// is none to test with. Feeding an 8 kHz recording to a 48 kHz chain leaves
// every band above 4 kHz empty in *both* chains, so they agree trivially and the
// comparison flatters native mode.
//
// Rather than present one number and bury that, the harness runs two corpora and
// reports them separately:
//
//   - "real": the recordings above. Conclusions are sound for rates up to
//     16 kHz, which is where the design's risk is concentrated anyway -- at
//     48 kHz the port is bit-exact against C, so there is nothing to measure.
//   - "synthetic": wideband formant-shaped speech in shaped noise, generated to
//     the full bandwidth of each rate. This exercises the upper bands that the
//     real material cannot, at the cost of not being speech.
//
// Replacing the synthetic set with real wideband speech would strengthen the
// high-rate rows. Nothing else here would change.

type corpusItem struct {
	name  string
	clean []float32 // at the item's own rate
	noise []float32
	rate  int
}

// speechPaths and noisePaths are relative to the workspace root.
var (
	workspaceRoot = "../.."
	speechGlobs   = []string{
		"jambonz-livekit-load-testing-benchmarks/fixtures/out/caller-*.wav",
		"smartturn-go/testdata/complete/*.wav",
		"smartturn-go/testdata/incomplete/*.wav",
	}
	noisePaths = []string{
		"noise/call_centre.wav",
		"noise/city.wav",
	}
)

// loadRealCorpus reads the workspace recordings. It reports what it could not
// find rather than silently thinning the corpus.
func loadRealCorpus(maxNoiseSeconds int) ([]corpusItem, []string, error) {
	var notes []string

	var speech []*wavFile
	var speechNames []string
	for _, g := range speechGlobs {
		matches, _ := filepath.Glob(filepath.Join(workspaceRoot, g))
		sort.Strings(matches)
		for _, p := range matches {
			w, err := readWAV(p)
			if err != nil {
				notes = append(notes, fmt.Sprintf("skipped %s: %v", filepath.Base(p), err))
				continue
			}
			speech = append(speech, w)
			speechNames = append(speechNames, filepath.Base(p))
		}
	}
	if len(speech) == 0 {
		return nil, notes, fmt.Errorf("no speech files found under %s", workspaceRoot)
	}

	var noises []*wavFile
	var noiseNames []string
	for _, rel := range noisePaths {
		p := filepath.Join(workspaceRoot, rel)
		if _, err := os.Stat(p); err != nil {
			notes = append(notes, fmt.Sprintf("noise file %s absent", rel))
			continue
		}
		w, err := readWAVPrefix(p, maxNoiseSeconds)
		if err != nil {
			notes = append(notes, fmt.Sprintf("skipped %s: %v", rel, err))
			continue
		}
		noises = append(noises, w)
		noiseNames = append(noiseNames, filepath.Base(rel))
	}
	if len(noises) == 0 {
		return nil, notes, fmt.Errorf("no noise files found under %s", workspaceRoot)
	}

	var items []corpusItem
	for i, sp := range speech {
		nz := noises[i%len(noises)]
		// Bring the noise to the speech's rate so a mixture is well defined.
		n, err := resample(nz.samples, nz.rate, sp.rate)
		if err != nil {
			return nil, notes, err
		}
		items = append(items, corpusItem{
			name:  fmt.Sprintf("%s+%s", speechNames[i], noiseNames[i%len(noiseNames)]),
			clean: sp.samples,
			noise: n,
			rate:  sp.rate,
		})
	}
	return items, notes, nil
}

// readWAVPrefix reads at most the first seconds of a file, so the two hours of
// noise does not have to be loaded whole.
func readWAVPrefix(path string, seconds int) (*wavFile, error) {
	w, err := readWAV(path)
	if err != nil {
		return nil, err
	}
	if n := seconds * w.rate; n > 0 && len(w.samples) > n {
		w.samples = w.samples[:n]
	}
	return w, nil
}

// syntheticItem builds wideband speech-like content and shaped noise directly at
// the requested rate, to exercise the bands the real corpus cannot reach.
func syntheticItem(rate int, seconds int, seed int64) corpusItem {
	n := seconds * rate
	clean := make([]float32, n)
	noise := make([]float32, n)

	// Voiced speech with a drifting fundamental, three formants, and syllabic
	// gating, band-limited only by the rate itself.
	formants := [3][2]float64{{700, 90}, {1220, 110}, {2600, 170}}
	rng := rand.New(rand.NewSource(seed))
	phase := 0.0
	for i := 0; i < n; i++ {
		t := float64(i) / float64(rate)
		f0 := 120 + 25*math.Sin(2*math.Pi*0.7*t)
		phase += 2 * math.Pi * f0 / float64(rate)
		env := math.Max(0, math.Sin(2*math.Pi*2.3*t))
		var v float64
		for h := 1; h <= 120; h++ {
			f := f0 * float64(h)
			if f > float64(rate)/2*0.95 {
				break
			}
			var g float64
			for _, fm := range formants {
				d := (f - fm[0]) / fm[1]
				g += 1 / (1 + d*d)
			}
			// -6 dB/octave source tilt on top of the formant shaping.
			v += g / float64(h) * math.Cos(phase*float64(h)+float64(h*h%13))
		}
		clean[i] = float32(2500 * env * v)
	}

	// Noise with a realistic -3 dB/octave tilt, made by filtering white noise
	// with a one-pole lowpass, plus a little broadband floor so the upper bands
	// are populated.
	var lp float64
	for i := 0; i < n; i++ {
		w := rng.NormFloat64()
		lp = 0.85*lp + 0.15*w
		noise[i] = float32(1500*lp + 250*w)
	}
	return corpusItem{
		name:  fmt.Sprintf("synthetic-%dHz", rate),
		clean: clean,
		noise: noise,
		rate:  rate,
	}
}

// mixAt returns clean and noisy signals at the requested rate and SNR. The
// clean signal is returned too, because every quality metric is measured
// against it rather than against another chain's output.
func mixAt(item corpusItem, rate int, snrDB float64) (clean, noisy []float32, err error) {
	clean, err = resample(item.clean, item.rate, rate)
	if err != nil {
		return nil, nil, err
	}
	nz, err := resample(item.noise, item.rate, rate)
	if err != nil {
		return nil, nil, err
	}
	if len(nz) == 0 || len(clean) == 0 {
		return nil, nil, fmt.Errorf("empty signal after resampling to %d", rate)
	}

	// Trim to whole 10 ms frames so every chain sees the same span.
	frame := rate / 100
	n := len(clean) / frame * frame
	if n == 0 {
		return nil, nil, fmt.Errorf("signal shorter than one frame at %d Hz", rate)
	}
	clean = clean[:n]

	// Tile the noise if it is shorter than the speech.
	tiled := make([]float32, n)
	for i := 0; i < n; i++ {
		tiled[i] = nz[i%len(nz)]
	}

	cr := rms(clean)
	nr := rms(tiled)
	if cr == 0 || nr == 0 {
		return nil, nil, fmt.Errorf("silent signal in %s", item.name)
	}
	// Scale the noise for the requested SNR against the speech.
	g := cr / nr / math.Pow(10, snrDB/20)
	noisy = make([]float32, n)
	for i := 0; i < n; i++ {
		v := float64(clean[i]) + g*float64(tiled[i])
		if v > 32767 {
			v = 32767
		}
		if v < -32768 {
			v = -32768
		}
		noisy[i] = float32(v)
	}
	return clean, noisy, nil
}

func rms(x []float32) float64 {
	if len(x) == 0 {
		return 0
	}
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(len(x)))
}

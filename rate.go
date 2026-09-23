package rnnoise

import (
	"fmt"
	"math"
	"sync"
)

// MinSampleRate is the lowest rate New accepts.
//
// The bound is not arbitrary. Upstream trains on band-limited data by zeroing
// FFT bins above a random cutoff drawn log-uniformly from [3006, 24050] Hz
// (src/dump_features.c), which is exactly the input a native rate-scaled core
// produces. 8 kHz puts the cutoff at 4 kHz, the 7th percentile of that
// distribution: supported, but in the tail, and measurably worse than 48 kHz.
// Below 8 kHz we would be extrapolating past what the model ever saw.
const MinSampleRate = 8000

// rateConfig holds everything derived from the sample rate. It is immutable
// once built and shared by every Denoiser at that rate, so the tables are
// computed once however many streams are running.
type rateConfig struct {
	sampleRate int

	frame  int // samples per Process call; FRAME_SIZE at 48 kHz
	window int // 2*frame; also PITCH_FRAME_SIZE
	freq   int // frame+1 usable FFT bins, 0..frame inclusive
	binHz  float64

	// activeBands is the number of bands the model has trained gains for at
	// this rate: the bands whose support lies at or below Nyquist.
	activeBands int

	pitchMin   int
	pitchMax   int
	pitchFrame int
	pitchBuf   int

	// pitchScale converts a detected period in native samples to the 48 kHz
	// sample count the model was trained to see in features[64].
	pitchScale float64

	hpB, hpA [2]float32

	halfWindow []float32
	dctTable   []float32
	fft        transformer
}

// transformer is the forward transform a rate uses: the mixed-radix path for
// every rate anyone is likely to ask for, or Bluestein for an awkward size.
// Both are 1/N-scaled, so the rest of the package cannot tell them apart.
type transformer interface {
	size() int
	fft(fin, fout []cpx)
}

var rateCache sync.Map // int (sampleRate) -> *rateConfig

func configForRate(rate int) (*rateConfig, error) {
	if c, ok := rateCache.Load(rate); ok {
		return c.(*rateConfig), nil
	}
	c, err := newRateConfig(rate)
	if err != nil {
		return nil, err
	}
	actual, _ := rateCache.LoadOrStore(rate, c)
	return actual.(*rateConfig), nil
}

func newRateConfig(rate int) (*rateConfig, error) {
	if rate < MinSampleRate {
		return nil, fmt.Errorf("rnnoise: sample rate %d below the minimum of %d Hz", rate, MinSampleRate)
	}

	// A 10 ms frame keeps the FFT bin spacing at 50 Hz, which is what makes
	// eband20ms name the same absolute frequencies at every rate. Rates that
	// are not a multiple of 100 get the nearest even frame size, which costs
	// at most a few tenths of a percent of bin-spacing error.
	var frame int
	if rate%100 == 0 {
		frame = rate / 100
	} else {
		frame = 2 * int(math.Round(float64(rate)/200))
	}
	binHz := float64(rate) / float64(2*frame)
	if err := checkBinSpacing(rate, frame, binHz); err != nil {
		return nil, err
	}

	c := &rateConfig{
		sampleRate: rate,
		frame:      frame,
		window:     2 * frame,
		freq:       frame + 1,
		binHz:      binHz,
		pitchScale: float64(refRate) / float64(rate),
	}

	// activeBands follows training's band_lp exactly: the first band whose
	// start bin is above the highest bin that carries energy. Bins 0..frame
	// exist, so the first absent bin is frame+1.
	c.activeBands = numBands
	for i := 0; i < numBands; i++ {
		if eband20ms[i] > c.freq {
			c.activeBands = i
			break
		}
	}

	// PITCH_FRAME_SIZE == WINDOW_SIZE and PITCH_BUF_SIZE == pitchMax+window
	// are hard invariants: rnn_compute_frame_features indexes the pitch
	// buffer at PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index, which must stay
	// non-negative for a period of pitchMax. pitchMax and pitchBuf must be
	// even because pitch_downsample and remove_doubling halve them.
	c.pitchFrame = c.window
	c.pitchMax = roundEven(1.6 * float64(frame))
	c.pitchMin = int(math.Round(float64(frame) / 8))
	c.pitchBuf = c.pitchMax + c.window
	if c.pitchMax%2 != 0 || c.pitchBuf%2 != 0 {
		return nil, fmt.Errorf("rnnoise: sample rate %d yields odd pitch constants (max=%d buf=%d)",
			rate, c.pitchMax, c.pitchBuf)
	}

	c.hpB, c.hpA = hpCoeffs(rate)
	c.halfWindow = makeHalfWindow(frame)
	c.dctTable = makeDCTTable()

	if fft, ok := newFFTState(c.window); ok {
		c.fft = fft
	} else if bs, ok := newBluesteinState(c.window); ok {
		// An awkward size: a prime factor above maxGenericRadix. Every rate in
		// common use avoids this, and 48 kHz in particular always takes the
		// mixed-radix path, so bit-exactness against the C reference is
		// unaffected.
		c.fft = bs
	} else {
		return nil, fmt.Errorf("rnnoise: sample rate %d needs a %d-point transform "+
			"that neither the mixed-radix nor the Bluestein path can build", rate, c.window)
	}

	return c, nil
}

// maxBinSpacingError bounds how far the bin spacing may drift from 50 Hz
// before the band table stops naming the frequencies the model was trained on.
const maxBinSpacingError = 0.005 // 0.5%

func checkBinSpacing(rate, frame int, binHz float64) error {
	if frame < MinSampleRate/100 {
		return fmt.Errorf("rnnoise: sample rate %d yields a %d-sample frame, too short", rate, frame)
	}
	if err := math.Abs(binHz/50 - 1); err > maxBinSpacingError {
		return fmt.Errorf("rnnoise: sample rate %d yields %.3f Hz bins, %.2f%% off the 50 Hz "+
			"the band table assumes", rate, binHz, err*100)
	}
	return nil
}

func roundEven(v float64) int {
	return 2 * int(math.Round(v/2))
}

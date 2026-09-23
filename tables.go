package rnnoise

import "math"

// Band edges in FFT bin indices, from upstream's eband20ms[NB_BANDS+2] in
// src/denoise.c. The commented frequencies assume 50 Hz per bin:
//
//	0 100 200 300 400 500 600 750 900 1.1k ... 14.1k 15.9k 17.8k 20.0k
//
// This table is the reason native rate-scaled processing works. It is
// expressed in *bins*, not hertz, and a 10 ms frame with a 20 ms window puts
// the bin spacing at rate/(2*rate/100) = 50 Hz at every sample rate. So the
// same table names the same absolute frequencies everywhere, and changing the
// sample rate only changes how many bins exist above it.
var eband20ms = [numBands + 2]int{
	0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77,
	87, 98, 110, 124, 140, 157, 176, 198, 223, 251, 282, 317, 356, 400,
}

const (
	numBands    = 32             // NB_BANDS
	numFeatures = 2*numBands + 1 // NB_FEATURES
	refRate     = 48000          // the rate upstream's constants are defined at
	refFrame    = refRate / 100  // FRAME_SIZE
	refWindow   = 2 * refFrame   // WINDOW_SIZE
	refFreq     = refFrame + 1   // FREQ_SIZE
)

// makeHalfWindow generates upstream's rnn_half_window for an overlap of n
// samples. Verbatim from src/dump_rnnoise_tables.c: the argument to the outer
// sine is squared, giving the power-complementary (Vorbis) window.
func makeHalfWindow(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		s := math.Sin(.5 * math.Pi * (float64(i) + .5) / float64(n))
		w[i] = float32(math.Sin(.5 * math.Pi * s * s))
	}
	return w
}

// makeDCTTable generates upstream's rnn_dct_table. Note it is indexed
// [i*numBands+j] with the j==0 column scaled by sqrt(1/2); the overall
// sqrt(2/22) normalisation lives in dct() itself, not here.
func makeDCTTable() []float32 {
	t := make([]float32, numBands*numBands)
	for i := 0; i < numBands; i++ {
		for j := 0; j < numBands; j++ {
			v := math.Cos((float64(i) + .5) * float64(j) * math.Pi / numBands)
			if j == 0 {
				v *= math.Sqrt(.5)
			}
			t[i*numBands+j] = float32(v)
		}
	}
	return t
}

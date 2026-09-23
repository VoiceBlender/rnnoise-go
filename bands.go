package rnnoise

import "math"

// dctScale is upstream's sqrt(2./22) in dct(). The 22 is a leftover from the
// old 22-band model and is NOT sqrt(2./NB_BANDS) -- the current model was
// trained with this exact constant, so it stays 22.
const dctScale = 2.0 / 22.0

// binsInBand returns how many of band i's bins actually exist at this rate.
//
// This is the whole of the native rate-scaling, and it is deliberately the
// only place the rate enters the band machinery. Every loop below still runs
// over all 33 intervals and all 32 bands exactly as upstream does; bins at or
// above Nyquist simply contribute nothing, which is bit-for-bit what upstream
// computes for a signal whose FFT bins were zeroed above the same cutoff --
// and that is precisely how src/dump_features.c band-limits 47% of the
// training data. Clipping the band loop instead, or moving the sum[1]/sum[32]
// end corrections to a new top band, would break that equivalence.
//
// Note frac below divides by the band's FULL width, not the truncated one,
// for the same reason: the absent bins are zero-valued, not absent.
func (c *rateConfig) binsInBand(i int) int {
	n := eband20ms[i+1] - eband20ms[i]
	if avail := c.freq - eband20ms[i]; avail < n {
		n = avail
	}
	if n < 0 {
		n = 0
	}
	return n
}

// computeBandEnergy is upstream's compute_band_energy: triangular, 50%
// overlapping bands, accumulated with a spill-over into the next band.
func (c *rateConfig) computeBandEnergy(bandE []float32, X []cpx) {
	var sum [numBands + 2]float32
	for i := 0; i < numBands+1; i++ {
		lo := eband20ms[i]
		bandSize := float32(eband20ms[i+1] - lo)
		for j := 0; j < c.binsInBand(i); j++ {
			frac := float32(j) / bandSize
			tmp := X[lo+j].r * X[lo+j].r
			tmp += X[lo+j].i * X[lo+j].i
			sum[i] += (1 - frac) * tmp
			sum[i+1] += frac * tmp
		}
	}
	sum[1] = (sum[0] + sum[1]) * 2 / 3
	sum[numBands] = (sum[numBands] + sum[numBands+1]) * 2 / 3
	for i := 0; i < numBands; i++ {
		bandE[i] = sum[i+1]
	}
}

// computeBandCorr is upstream's compute_band_corr: identical to
// computeBandEnergy but accumulating the X.P dot product.
func (c *rateConfig) computeBandCorr(bandE []float32, X, P []cpx) {
	var sum [numBands + 2]float32
	for i := 0; i < numBands+1; i++ {
		lo := eband20ms[i]
		bandSize := float32(eband20ms[i+1] - lo)
		for j := 0; j < c.binsInBand(i); j++ {
			frac := float32(j) / bandSize
			tmp := X[lo+j].r * P[lo+j].r
			tmp += X[lo+j].i * P[lo+j].i
			sum[i] += (1 - frac) * tmp
			sum[i+1] += frac * tmp
		}
	}
	sum[1] = (sum[0] + sum[1]) * 2 / 3
	sum[numBands] = (sum[numBands] + sum[numBands+1]) * 2 / 3
	for i := 0; i < numBands; i++ {
		bandE[i] = sum[i+1]
	}
}

// interpBandGain is upstream's interp_band_gain: linear interpolation of a
// per-band value across the FFT bins.
//
// Two upstream quirks are load-bearing and reproduced here.
//
// First, the loops only ever write bins below eband20ms[NB_BANDS+1] == 400.
// Upstream's callers pass fresh zero-initialised stack arrays, so bins 400 and
// above stay zero -- that, not the memset, is what hard-limits the output to
// 20 kHz. (Upstream's memset(g, 0, FREQ_SIZE) is a byte count where a float
// count was meant, clearing only ~120 floats, but the loops overwrite those
// anyway, so it is benign.) We reuse scratch buffers to stay allocation-free,
// so that tail has to be zeroed explicitly or the previous frame's gains leak
// into the top of the spectrum. Below 40 kHz the tail is empty; at 44.1 kHz
// and 48 kHz it is real and audible.
//
// Second, bands at and above activeBands hold untrained network output at
// reduced rates. They are unreachable here because every bin they cover is
// above Nyquist -- see TestActiveBandsMatchTrainingMask.
func (c *rateConfig) interpBandGain(g []float32, bandE []float32) {
	top := eband20ms[numBands+1]
	if top > c.freq {
		top = c.freq
	}
	for i := top; i < c.freq; i++ {
		g[i] = 0
	}

	for i := 1; i < numBands; i++ {
		lo := eband20ms[i]
		bandSize := float32(eband20ms[i+1] - lo)
		for j := 0; j < c.binsInBand(i); j++ {
			frac := float32(j) / bandSize
			g[lo+j] = (1-frac)*bandE[i-1] + frac*bandE[i]
		}
	}
	for j := 0; j < eband20ms[1] && j < c.freq; j++ {
		g[j] = bandE[0]
	}
	for j := eband20ms[numBands]; j < eband20ms[numBands+1] && j < c.freq; j++ {
		g[j] = bandE[numBands-1]
	}
}

// dct is upstream's dct(). Note the accumulation is float32 but the final
// scaling happens in float64, because C promotes sum*sqrt(2./22) to double
// before the store rounds it back. Folding the scale into the table or doing
// it in float32 would change the last bit.
func (c *rateConfig) dct(out, in []float32) {
	tbl := c.dctTable
	for i := 0; i < numBands; i++ {
		var sum float32
		for j := 0; j < numBands; j++ {
			sum += in[j] * tbl[j*numBands+i]
		}
		out[i] = float32(float64(sum) * math.Sqrt(dctScale))
	}
}

// fmax reproduces C's MAX16 ternary rather than math.Max, whose NaN and
// signed-zero behaviour differs.
func fmax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func fmin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

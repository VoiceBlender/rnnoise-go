package ctest

import (
	"math"
	"math/cmplx"
	"sort"
)

// Objective metrics for the native-rate comparison.
//
// Everything here works from signals through the public API, never from the
// library's internals. That is partly to keep the engine's dependency graph
// clean, and partly because the effective band gain measured from the input and
// output is more informative than the network's raw gain output: it includes the
// pitch comb filter and the per-frame decay cap, which is what a listener
// actually hears.

// bandEdgeHz mirrors the engine's band layout, expressed in hertz so the metrics
// do not depend on any particular transform length. These are upstream's
// eband20ms indices at 50 Hz per bin.
var bandEdgeHz = []float64{
	0, 100, 200, 300, 400, 500, 600, 750, 900, 1050, 1200, 1400, 1600, 1800,
	2050, 2350, 2650, 3000, 3400, 3850, 4350, 4900, 5500, 6200, 7000, 7850,
	8800, 9900, 11150, 12550, 14100, 15850, 17800, 20000,
}

// fftRadix2 is an in-place iterative radix-2 FFT, used only by the metrics.
// n must be a power of two.
func fftRadix2(a []complex128) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := -2 * math.Pi / float64(length)
		wl := cmplx.Exp(complex(0, ang))
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			for k := 0; k < length/2; k++ {
				u := a[i+k]
				v := a[i+k+length/2] * w
				a[i+k] = u + v
				a[i+k+length/2] = u - v
				w *= wl
			}
		}
	}
}

// bandPower returns per-frame band powers for a signal, on 20 ms frames hopped
// by 10 ms with a Hann window. Bands whose lower edge is at or above Nyquist are
// reported as zero.
func bandPower(x []float32, rate int) [][]float64 {
	win := rate / 50 // 20 ms
	hop := rate / 100
	if win < 16 || len(x) < win {
		return nil
	}
	n := 1
	for n < win {
		n <<= 1
	}
	hann := make([]float64, win)
	for i := range hann {
		hann[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(win))
	}
	binHz := float64(rate) / float64(n)
	nyq := float64(rate) / 2

	nBands := len(bandEdgeHz) - 1
	var out [][]float64
	buf := make([]complex128, n)
	for off := 0; off+win <= len(x); off += hop {
		for i := 0; i < win; i++ {
			buf[i] = complex(float64(x[off+i])*hann[i], 0)
		}
		for i := win; i < n; i++ {
			buf[i] = 0
		}
		fftRadix2(buf)

		row := make([]float64, nBands)
		for b := 0; b < nBands; b++ {
			lo, hi := bandEdgeHz[b], bandEdgeHz[b+1]
			if lo >= nyq {
				continue
			}
			if hi > nyq {
				hi = nyq
			}
			k0 := int(math.Ceil(lo / binHz))
			k1 := int(math.Floor(hi / binHz))
			if k1 >= n/2 {
				k1 = n/2 - 1
			}
			var p float64
			for k := k0; k <= k1; k++ {
				m := cmplx.Abs(buf[k])
				p += m * m
			}
			row[b] = p
		}
		out = append(out, row)
	}
	return out
}

// activeMask marks the (frame, band) pairs where a reference signal carries
// enough energy for a ratio or a log-ratio to mean anything.
//
// Without this every band-domain metric is dominated by noise. The corpus is
// band-limited -- the real recordings are 8 or 16 kHz material -- so when it is
// resampled up to 44.1 kHz, most bands hold nothing but resampler leakage
// 80-plus dB down. Dividing by those, or taking their log ratio, produces
// numbers in the tens of dB that describe the floor rather than the denoiser.
//
// The threshold is relative to the reference's own peak band power, so it
// adapts to the signal level instead of assuming one.
func activeMask(ref [][]float64, relDB float64) ([][]bool, int) {
	var peak float64
	for _, row := range ref {
		for _, v := range row {
			if v > peak {
				peak = v
			}
		}
	}
	thresh := peak * math.Pow(10, relDB/10)
	mask := make([][]bool, len(ref))
	n := 0
	for f, row := range ref {
		mask[f] = make([]bool, len(row))
		for b, v := range row {
			if v > thresh {
				mask[f][b] = true
				n++
			}
		}
	}
	return mask, n
}

// activeRelDB is how far below a signal's peak band power a band may sit and
// still be measured: 40 dB keeps real spectral detail while excluding empty
// bands and inter-word silence.
const activeRelDB = -40

// alignLag finds the integer lag (in samples) that maximises the correlation of
// b against a, searched over +/-maxLag.
//
// The harness fails on any non-zero residual lag rather than compensating for
// it. A misaligned comparison silently destroys every metric below while still
// producing plausible-looking numbers, which is the worst possible failure mode
// for a quality report.
func alignLag(a, b []float32, maxLag int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	best, bestLag := math.Inf(-1), 0
	for lag := -maxLag; lag <= maxLag; lag++ {
		var acc float64
		count := 0
		for i := 0; i < n; i++ {
			j := i + lag
			if j < 0 || j >= len(b) {
				continue
			}
			acc += float64(a[i]) * float64(b[j])
			count++
		}
		if count == 0 {
			continue
		}
		acc /= float64(count)
		if acc > best {
			best, bestLag = acc, lag
		}
	}
	return bestLag
}

// snrDB is the reference-to-error ratio of test against ref, in dB.
func snrDB(ref, test []float32) float64 {
	n := len(ref)
	if len(test) < n {
		n = len(test)
	}
	var se, ee float64
	for i := 0; i < n; i++ {
		r := float64(ref[i])
		e := r - float64(test[i])
		se += r * r
		ee += e * e
	}
	if ee == 0 {
		return math.Inf(1)
	}
	if se == 0 {
		return math.Inf(-1)
	}
	return 10 * math.Log10(se/ee)
}

// segSNRdB is the segmental SNR of test against clean: the mean over 20 ms
// segments of the per-segment SNR, clamped to [-10, 35] dB as is conventional,
// skipping segments where the clean signal carries no energy.
func segSNRdB(clean, test []float32, rate int) float64 {
	seg := rate / 50
	n := len(clean)
	if len(test) < n {
		n = len(test)
	}
	var sum float64
	var count int
	// A segment counts as active if its clean RMS is above this, on the
	// +/-32768 scale. Below it the ratio is dominated by the floor.
	const activeRMS = 30.0
	for off := 0; off+seg <= n; off += seg {
		var se, ee float64
		for i := off; i < off+seg; i++ {
			c := float64(clean[i])
			e := c - float64(test[i])
			se += c * c
			ee += e * e
		}
		if math.Sqrt(se/float64(seg)) < activeRMS {
			continue
		}
		var v float64
		switch {
		case ee == 0:
			v = 35
		default:
			v = 10 * math.Log10(se/ee)
		}
		if v > 35 {
			v = 35
		}
		if v < -10 {
			v = -10
		}
		sum += v
		count++
	}
	if count == 0 {
		return math.NaN()
	}
	return sum / float64(count)
}

// lsdDB is the log-spectral distance between test and clean, in dB, over the
// bands that exist at this rate.
func lsdDB(clean, test []float32, rate int) float64 {
	pc := bandPower(clean, rate)
	pt := bandPower(test, rate)
	if len(pc) == 0 || len(pt) == 0 {
		return math.NaN()
	}
	frames := len(pc)
	if len(pt) < frames {
		frames = len(pt)
	}
	mask, active := activeMask(pc[:frames], activeRelDB)
	if active == 0 {
		return math.NaN()
	}
	// A floor 80 dB below the clean peak keeps a heavily suppressed band from
	// contributing an unbounded log ratio.
	var peak float64
	for _, row := range pc[:frames] {
		for _, v := range row {
			if v > peak {
				peak = v
			}
		}
	}
	floor := peak * 1e-8

	var sum float64
	var count int
	for f := 0; f < frames; f++ {
		for b := range pc[f] {
			if !mask[f][b] {
				continue
			}
			d := 10 * math.Log10((pt[f][b]+floor)/(pc[f][b]+floor))
			sum += d * d
			count++
		}
	}
	if count == 0 {
		return math.NaN()
	}
	return math.Sqrt(sum / float64(count))
}

// effectiveGainStats compares the per-band gain each chain actually applied,
// measured as sqrt(outputPower/inputPower) per band per frame.
//
// This is a spectral measurement rather than a read of the network's gain
// output, so it also captures the pitch comb filter and the decay cap, and it
// needs nothing beyond the public API.
func effectiveGainStats(in, a, b []float32, rate int) (mean, p95, max float64) {
	pi := bandPower(in, rate)
	pa := bandPower(a, rate)
	pb := bandPower(b, rate)
	if len(pi) == 0 || len(pa) == 0 || len(pb) == 0 {
		return math.NaN(), math.NaN(), math.NaN()
	}
	frames := min3(len(pi), len(pa), len(pb))
	mask, active := activeMask(pi[:frames], activeRelDB)
	if active == 0 {
		return math.NaN(), math.NaN(), math.NaN()
	}

	var diffs []float64
	for f := 0; f < frames; f++ {
		for bnd := range pi[f] {
			if !mask[f][bnd] {
				continue
			}
			ga := math.Sqrt(pa[f][bnd] / pi[f][bnd])
			gb := math.Sqrt(pb[f][bnd] / pi[f][bnd])
			diffs = append(diffs, math.Abs(ga-gb))
		}
	}
	if len(diffs) == 0 {
		return math.NaN(), math.NaN(), math.NaN()
	}
	sortFloats(diffs)
	var sum float64
	for _, d := range diffs {
		sum += d
	}
	mean = sum / float64(len(diffs))
	p95 = diffs[int(0.95*float64(len(diffs)-1))]
	max = diffs[len(diffs)-1]
	return mean, p95, max
}

// gainFlutter is a musical-noise proxy: the temporal standard deviation of the
// effective per-band gain, measured only on frames where the clean signal is
// silent, so it reflects how steadily the suppressor holds noise down rather
// than how it tracks speech.
func gainFlutter(clean, in, out []float32, rate int) float64 {
	pi := bandPower(in, rate)
	po := bandPower(out, rate)
	pc := bandPower(clean, rate)
	if len(pi) == 0 || len(po) == 0 || len(pc) == 0 {
		return math.NaN()
	}
	frames := min3(len(pi), len(po), len(pc))
	mask, active := activeMask(pi[:frames], activeRelDB)
	if active == 0 {
		return math.NaN()
	}

	var cleanTotal float64
	for f := 0; f < frames; f++ {
		for _, v := range pc[f] {
			cleanTotal += v
		}
	}
	quietThresh := cleanTotal / float64(frames) * 0.01

	nBands := len(pi[0])
	var acc float64
	var used int
	for b := 0; b < nBands; b++ {
		var vals []float64
		for f := 0; f < frames; f++ {
			var ce float64
			for _, v := range pc[f] {
				ce += v
			}
			if ce > quietThresh || !mask[f][b] {
				continue
			}
			vals = append(vals, math.Sqrt(po[f][b]/pi[f][b]))
		}
		if len(vals) < 8 {
			continue
		}
		var m float64
		for _, v := range vals {
			m += v
		}
		m /= float64(len(vals))
		var s float64
		for _, v := range vals {
			s += (v - m) * (v - m)
		}
		acc += math.Sqrt(s / float64(len(vals)))
		used++
	}
	if used == 0 {
		return math.NaN()
	}
	return acc / float64(used)
}

// vadStats compares two chains' speech probabilities.
func vadStats(a, b []float32) (meanAbs, agreement float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return math.NaN(), math.NaN()
	}
	var sum float64
	var agree int
	for i := 0; i < n; i++ {
		sum += math.Abs(float64(a[i]) - float64(b[i]))
		if (a[i] >= 0.5) == (b[i] >= 0.5) {
			agree++
		}
	}
	return sum / float64(n), float64(agree) / float64(n)
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

func sortFloats(v []float64) { sort.Float64s(v) }

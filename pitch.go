package rnnoise

import "math"

// Ported from upstream's src/pitch.c and src/pitch.h (CELT's pitch analysis),
// float branch. Fixed-size C arrays become per-stream scratch in pitchState so
// the hot path stays allocation-free, and the PITCH_* constants become the
// rate-scaled values from rateConfig.
//
// None of this is vectorised, deliberately. xcorrKernel's rotating four-lag
// accumulation defines a specific float summation order; changing it can flip
// a near-tie in findBestPitch, which changes the reported period, which
// changes features[64] *and* the whole P/Ep/Exp half of the feature vector.
// It is about 2% of a frame, so there is nothing to win and a cascading
// divergence to lose.

// pitchState holds the scratch buffers the pitch search needs.
type pitchState struct {
	// fast selects the vectorised, non-bit-exact correlation.
	fast bool

	c *rateConfig

	lp    []float32 // half-rate signal, pitchBuf/2
	xLP4  []float32 // quarter-rate analysis frame, pitchFrame/4
	yLP4  []float32 // quarter-rate search buffer, (pitchFrame+pitchMax)/4
	xcorr []float32 // pitchMax/2
	yy    []float32 // yy_lookup, pitchMax+1
	lpc   [4]float32
	lpc2  [5]float32
	ac    [5]float32
	mem   [5]float32
}

func newPitchState(c *rateConfig) *pitchState {
	return &pitchState{
		c:     c,
		lp:    make([]float32, c.pitchBuf>>1),
		xLP4:  make([]float32, c.pitchFrame>>2),
		yLP4:  make([]float32, (c.pitchFrame+c.pitchMax)>>2),
		xcorr: make([]float32, c.pitchMax>>1),
		yy:    make([]float32, c.pitchMax+1),
	}
}

// xcorrKernel is the four-lag unrolled correlation from src/pitch.h. The
// rotating y_0..y_3 registers are what make the summation order specific.
func xcorrKernel(x, y []float32, sum *[4]float32, length int) {
	var xi, yi int
	y0, y1, y2, y3 := y[yi], y[yi+1], y[yi+2], float32(0)
	yi += 3

	j := 0
	for ; j < length-3; j += 4 {
		tmp := x[xi]
		xi++
		y3 = y[yi]
		yi++
		sum[0] += tmp * y0
		sum[1] += tmp * y1
		sum[2] += tmp * y2
		sum[3] += tmp * y3

		tmp = x[xi]
		xi++
		y0 = y[yi]
		yi++
		sum[0] += tmp * y1
		sum[1] += tmp * y2
		sum[2] += tmp * y3
		sum[3] += tmp * y0

		tmp = x[xi]
		xi++
		y1 = y[yi]
		yi++
		sum[0] += tmp * y2
		sum[1] += tmp * y3
		sum[2] += tmp * y0
		sum[3] += tmp * y1

		tmp = x[xi]
		xi++
		y2 = y[yi]
		yi++
		sum[0] += tmp * y3
		sum[1] += tmp * y0
		sum[2] += tmp * y1
		sum[3] += tmp * y2
	}
	if j < length {
		j++
		tmp := x[xi]
		xi++
		y3 = y[yi]
		yi++
		sum[0] += tmp * y0
		sum[1] += tmp * y1
		sum[2] += tmp * y2
		sum[3] += tmp * y3
	}
	if j < length {
		j++
		tmp := x[xi]
		xi++
		y0 = y[yi]
		yi++
		sum[0] += tmp * y1
		sum[1] += tmp * y2
		sum[2] += tmp * y3
		sum[3] += tmp * y0
	}
	if j < length {
		tmp := x[xi]
		y1 = y[yi]
		sum[0] += tmp * y2
		sum[1] += tmp * y3
		sum[2] += tmp * y0
		sum[3] += tmp * y1
	}
}

// xcorrKernelFast is xcorrKernel with each lag's sum split across vector lanes
// and fused multiply-add. Not bit-exact against the C scalar build; selected by
// Options.Fast.
func xcorrKernelFast(x, y []float32, sum *[4]float32, length int) {
	var acc [32]float32
	n := xcorrVec(x, y, &acc, length)
	if n == 0 {
		xcorrKernel(x, y, sum, length)
		return
	}
	for k := 0; k < 4; k++ {
		a := acc[k*8 : k*8+8 : k*8+8]
		sum[k] += ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
	}
	for j := n; j < length; j++ {
		xj := x[j]
		sum[0] += xj * y[j]
		sum[1] += xj * y[j+1]
		sum[2] += xj * y[j+2]
		sum[3] += xj * y[j+3]
	}
}

// innerProd is celt_inner_prod.
func innerProd(x, y []float32, n int) float32 {
	var xy float32
	for i := 0; i < n; i++ {
		xy += x[i] * y[i]
	}
	return xy
}

// dualInnerProd is dual_inner_prod: two correlations sharing one pass over x.
func dualInnerProd(x, y01, y02 []float32, n int) (float32, float32) {
	var xy01, xy02 float32
	for i := 0; i < n; i++ {
		xy01 += x[i] * y01[i]
		xy02 += x[i] * y02[i]
	}
	return xy01, xy02
}

// pitchXCorr is rnn_pitch_xcorr: the unrolled version, including its
// non-unrolled tail for a maxPitch that is not a multiple of 4. That tail
// matters at reduced rates, where maxPitch-3*minPitch need not be divisible
// by 4.
func pitchXCorr(x, y []float32, xcorr []float32, length, maxPitch int, fast bool) {
	i := 0
	for ; i < maxPitch-3; i += 4 {
		sum := [4]float32{}
		if fast {
			xcorrKernelFast(x, y[i:], &sum, length)
		} else {
			xcorrKernel(x, y[i:], &sum, length)
		}
		xcorr[i] = sum[0]
		xcorr[i+1] = sum[1]
		xcorr[i+2] = sum[2]
		xcorr[i+3] = sum[3]
	}
	for ; i < maxPitch; i++ {
		xcorr[i] = innerProd(x, y[i:], length)
	}
}

// findBestPitch is find_best_pitch: pick the two lags with the best
// normalised correlation. The 1e-12 scaling of xcorr16 is upstream's guard
// against overflow when squaring.
func findBestPitch(xcorr []float32, y []float32, length, maxPitch int, bestPitch *[2]int) {
	var bestNum [2]float32
	var bestDen [2]float32
	Syy := float32(1)

	bestNum[0] = -1
	bestNum[1] = -1
	bestDen[0] = 0
	bestDen[1] = 0
	bestPitch[0] = 0
	bestPitch[1] = 1
	for j := 0; j < length; j++ {
		Syy += y[j] * y[j]
	}
	for i := 0; i < maxPitch; i++ {
		if xcorr[i] > 0 {
			xcorr16 := xcorr[i]
			// Considering the range of xcorr16, this should avoid both
			// underflows and overflows (inf) when squaring xcorr16.
			xcorr16 *= 1e-12
			num := xcorr16 * xcorr16
			if num*bestDen[1] > bestNum[1]*Syy {
				if num*bestDen[0] > bestNum[0]*Syy {
					bestNum[1] = bestNum[0]
					bestDen[1] = bestDen[0]
					bestPitch[1] = bestPitch[0]
					bestNum[0] = num
					bestDen[0] = Syy
					bestPitch[0] = i
				} else {
					bestNum[1] = num
					bestDen[1] = Syy
					bestPitch[1] = i
				}
			}
		}
		Syy += y[i+length]*y[i+length] - y[i]*y[i]
		Syy = maxf(1, Syy)
	}
}

// downsample is rnn_pitch_downsample for the single-channel case: decimate by
// two, then flatten with a short whitening filter.
//
// Two of its constants do not scale with the sample rate, and both are kept
// verbatim. ac[0] *= 1.0001 is a -40 dB noise floor, which is scale-free. The
// lag window ac[i] -= ac[i]*(.008*i)^2 is in normalised frequency, so its
// bandwidth in hertz shrinks with the rate and the whitener is smoothed about
// six times harder at 8 kHz than at 48 kHz. Scaling the .008 would depart from
// the reference for no validated benefit.
func (p *pitchState) downsample(x []float32, xLP []float32, length int) {
	for i := 1; i < length>>1; i++ {
		xLP[i] = .5 * (.5*(x[2*i-1]+x[2*i+1]) + x[2*i])
	}
	xLP[0] = .5 * (.5*x[1] + x[0])

	ac := p.ac[:]
	celtAutocorr(xLP, ac, 4, length>>1)

	// Noise floor -40 dB.
	ac[0] *= 1.0001
	// Lag windowing.
	for i := 1; i <= 4; i++ {
		ac[i] -= ac[i] * (.008 * float32(i)) * (.008 * float32(i))
	}

	lpc := p.lpc[:]
	celtLPC(lpc, ac, 4)
	tmp := float32(1)
	for i := 0; i < 4; i++ {
		tmp = .9 * tmp
		lpc[i] = lpc[i] * tmp
	}
	// Add a zero.
	const c1 = float32(.8)
	lpc2 := &p.lpc2
	lpc2[0] = lpc[0] + .8
	lpc2[1] = lpc[1] + c1*lpc[0]
	lpc2[2] = lpc[2] + c1*lpc[1]
	lpc2[3] = lpc[3] + c1*lpc[2]
	lpc2[4] = c1 * lpc[3]
	p.mem = [5]float32{}
	celtFIR5(xLP, lpc2[:], xLP, length>>1, &p.mem)
}

// search is rnn_pitch_search: a coarse 4x-decimated correlation, a finer
// 2x-decimated pass around the two best coarse lags, then pseudo-interpolation.
func (p *pitchState) search(xLP, y []float32, length, maxPitch int) int {
	var bestPitch [2]int
	lag := length + maxPitch

	// Downsample by 2 again.
	for j := 0; j < length>>2; j++ {
		p.xLP4[j] = xLP[2*j]
	}
	for j := 0; j < lag>>2; j++ {
		p.yLP4[j] = y[2*j]
	}

	// Coarse search with 4x decimation.
	pitchXCorr(p.xLP4, p.yLP4, p.xcorr, length>>2, maxPitch>>2, p.fast)
	findBestPitch(p.xcorr, p.yLP4, length>>2, maxPitch>>2, &bestPitch)

	// Finer search with 2x decimation.
	for i := 0; i < maxPitch>>1; i++ {
		p.xcorr[i] = 0
		if abs(i-2*bestPitch[0]) > 2 && abs(i-2*bestPitch[1]) > 2 {
			continue
		}
		sum := innerProd(xLP, y[i:], length>>1)
		p.xcorr[i] = maxf(-1, sum)
	}
	findBestPitch(p.xcorr, y, length>>1, maxPitch>>1, &bestPitch)

	// Refine by pseudo-interpolation.
	offset := 0
	if bestPitch[0] > 0 && bestPitch[0] < (maxPitch>>1)-1 {
		a := p.xcorr[bestPitch[0]-1]
		b := p.xcorr[bestPitch[0]]
		cc := p.xcorr[bestPitch[0]+1]
		if (cc - a) > .7*(b-a) {
			offset = 1
		} else if (a - cc) > .7*(b-cc) {
			offset = -1
		}
	}
	return 2*bestPitch[0] - offset
}

// computePitchGain is compute_pitch_gain. sqrt() is double in C, so the
// division happens in float64 and rounds once on return.
func computePitchGain(xy, xx, yy float32) float32 {
	return float32(float64(xy) / math.Sqrt(1+float64(xx*yy)))
}

var secondCheck = [16]int{0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2}

// removeDoubling is rnn_remove_doubling: reject period multiples by looking
// for a stronger correlation at T/k.
//
// x is indexed from the caller's base; xOff is where the analysis window
// starts, standing in for C's "x += maxperiod".
//
// Every fractional heuristic here is scale-free (expressed via minperiod, T0
// and g0), so the only rate adaptation needed is in the arguments. One
// constant does not scale: the abs(T1-prev_period) <= 1 and <= 2 continuity
// tolerances are absolute, in half-rate samples, so at 8 kHz they span about
// 5% of a typical period against 0.26% at 48 kHz. The effect is a mild extra
// bias toward retaining the previous period, i.e. slightly smoother F0
// tracking at low rates. Scaling them by rate/48000 would clamp to 1 for every
// rate at or below 48 kHz anyway, so there is nothing to do.
func (p *pitchState) removeDoubling(x []float32, xOff, maxperiod, minperiod, n int, t0In, prevPeriod int, prevGain float32) (int, float32) {
	minperiod0 := minperiod
	maxperiod /= 2
	minperiod /= 2
	t0 := t0In / 2
	prevPeriod /= 2
	n /= 2
	xOff += maxperiod
	if t0 >= maxperiod {
		t0 = maxperiod - 1
	}

	T := t0
	xs := x[xOff:]
	xx, xy := dualInnerProd(xs, xs, x[xOff-t0:], n)
	yyLookup := p.yy
	yyLookup[0] = xx
	yy := xx
	for i := 1; i <= maxperiod; i++ {
		yy = yy + x[xOff-i]*x[xOff-i] - x[xOff+n-i]*x[xOff+n-i]
		yyLookup[i] = maxf(0, yy)
	}
	yy = yyLookup[t0]
	bestXY := xy
	bestYY := yy
	g := computePitchGain(xy, xx, yy)
	g0 := g

	// Look for any pitch at T/k.
	for k := 2; k <= 15; k++ {
		T1 := (2*t0 + k) / (2 * k)
		if T1 < minperiod {
			break
		}
		// Look for another strong correlation at T1b.
		var T1b int
		if k == 2 {
			if T1+t0 > maxperiod {
				T1b = t0
			} else {
				T1b = t0 + T1
			}
		} else {
			T1b = (2*secondCheck[k]*t0 + k) / (2 * k)
		}
		var xy2 float32
		xy, xy2 = dualInnerProd(xs, x[xOff-T1:], x[xOff-T1b:], n)
		xy = .5 * (xy + xy2)
		yy = .5 * (yyLookup[T1] + yyLookup[T1b])
		g1 := computePitchGain(xy, xx, yy)

		var cont float32
		if abs(T1-prevPeriod) <= 1 {
			cont = prevGain
		} else if abs(T1-prevPeriod) <= 2 && 5*k*k < t0 {
			cont = .5 * prevGain
		}
		thresh := maxf(.3, .7*g0-cont)
		// Bias against very high pitch (very short period) to avoid
		// false-positives due to short-term correlation.
		if T1 < 3*minperiod {
			thresh = maxf(.4, .85*g0-cont)
		} else if T1 < 2*minperiod {
			thresh = maxf(.5, .9*g0-cont)
		}
		if g1 > thresh {
			bestXY = xy
			bestYY = yy
			T = T1
			g = g1
		}
	}
	bestXY = maxf(0, bestXY)
	var pg float32
	if bestYY <= bestXY {
		pg = 1
	} else {
		pg = bestXY / (bestYY + 1)
	}

	var xcorr [3]float32
	for k := 0; k < 3; k++ {
		xcorr[k] = innerProd(xs, x[xOff-(T+k-1):], n)
	}
	offset := 0
	if (xcorr[2] - xcorr[0]) > .7*(xcorr[1]-xcorr[0]) {
		offset = 1
	} else if (xcorr[0] - xcorr[2]) > .7*(xcorr[1]-xcorr[2]) {
		offset = -1
	}
	if pg > g {
		pg = g
	}
	t0Out := 2*T + offset
	if t0Out < minperiod0 {
		t0Out = minperiod0
	}
	return t0Out, pg
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

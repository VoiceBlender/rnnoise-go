package rnnoise

import "math"

// Mixed-radix complex FFT, ported structurally from upstream's src/kiss_fft.c
// (CELT's variant of KISS FFT). The butterflies keep upstream's exact
// fstride/m/N/mm conventions and, critically, its exact arithmetic order: that
// is what lets the 48 kHz path reproduce librnnoise bit-for-bit. Do not
// "simplify" an expression here, even when it is algebraically equivalent.
//
// Two deliberate extensions over upstream:
//
//   - kf_factor accepts primes above 5 and dispatches them to a generic
//     radix-p butterfly. Upstream rejects them outright, which would make
//     44.1 kHz (window 882 = 2*3^2*7^2) unreachable. 48 kHz factors as
//     5,4,4,4,3 and never reaches the generic path, so this cannot affect it.
//   - kfBfly2 keeps the degenerate m==1 branch that upstream compiles out
//     unless CUSTOM_MODES is defined. Upstream's 960-point transform never
//     calls radix 2 at all; 882 ends on a radix-2 stage with m==1.
//
// Note on FMA: upstream compiles these expressions to unfused SSE2 on amd64,
// and Go's amd64 backend likewise does not contract float32 a*b+c into FMA, so
// the two agree. On arm64 Go *does* contract, so exact equality with the C
// scalar build is an amd64 property. See TestFFTMatchesUpstreamTables.

const maxFactors = 8

// cpx mirrors kiss_fft_cpx. A struct of two float32 rather than complex64:
// Go's complex multiply does not decompose into the same operation order as
// upstream's C_MUL macro, and that order is load-bearing here.
type cpx struct{ r, i float32 }

// cmul is C_MUL from _kiss_fft_guts.h (float branch).
func cmul(a, b cpx) cpx {
	return cpx{
		r: a.r*b.r - a.i*b.i,
		i: a.r*b.i + a.i*b.r,
	}
}

// fftState mirrors kiss_fft_state. Immutable once built, so one instance is
// shared by every denoiser running at the same sample rate.
type fftState struct {
	nfft     int
	scale    float32
	shift    int
	factors  [2 * maxFactors]int
	bitrev   []int32
	twiddles []cpx
}

// maxGenericRadix bounds the stack-resident scratch in kfBflyGeneric, and with
// it the largest prime factor a transform may contain. Sizes needing a larger
// prime fall back to Bluestein.
const maxGenericRadix = 32

// kfFactor is upstream's kf_factor, with the p>5 rejection relaxed: any prime
// up to maxGenericRadix is accepted and handled by kfBflyGeneric. The radix-4
// promotion, the stage reversal and the facbuf[2i+1] fill are unchanged.
func kfFactor(n int, facbuf *[2 * maxFactors]int) bool {
	p := 4
	stages := 0
	nbak := n

	// Factor out powers of 4, powers of 2, then any remaining primes.
	for {
		for n%p != 0 {
			switch p {
			case 4:
				p = 2
			case 2:
				p = 3
			default:
				p += 2
			}
			if p > 32000 || p*p > n {
				p = n // No more factors, skip to end.
			}
		}
		n /= p
		if p > maxGenericRadix {
			return false
		}
		if stages >= maxFactors {
			return false
		}
		facbuf[2*stages] = p
		if p == 2 && stages > 1 {
			facbuf[2*stages] = 4
			facbuf[2] = 2
		}
		stages++
		if n <= 1 {
			break
		}
	}
	n = nbak
	// Reverse the order to get the radix 4 at the end, so we can use the fast
	// degenerate case. Reversing also improves the noise behaviour.
	for i := 0; i < stages/2; i++ {
		facbuf[2*i], facbuf[2*(stages-i-1)] = facbuf[2*(stages-i-1)], facbuf[2*i]
	}
	for i := 0; i < stages; i++ {
		n /= facbuf[2*i]
		facbuf[2*i+1] = n
	}
	return true
}

// computeBitrevTable is upstream's compute_bitrev_table. f is the destination
// slice, fpos the write cursor within it.
func computeBitrevTable(fout int, f []int32, fpos, fstride, inStride int, factors []int) {
	p := factors[0] // the radix
	m := factors[1] // stage's fft length/p
	if m == 1 {
		for j := 0; j < p; j++ {
			f[fpos] = int32(fout + j)
			fpos += fstride * inStride
		}
		return
	}
	for j := 0; j < p; j++ {
		computeBitrevTable(fout, f, fpos, fstride*p, inStride, factors[2:])
		fpos += fstride * inStride
		fout += m
	}
}

// computeTwiddles is upstream's compute_twiddles: the phase is accumulated in
// float64 and rounded once to float32, matching kf_cexp's (float)cos/(float)sin.
func computeTwiddles(tw []cpx, nfft int) {
	for i := range tw {
		const pi = 3.14159265358979323846264338327
		phase := (-2 * pi / float64(nfft)) * float64(i)
		tw[i] = cpx{r: float32(math.Cos(phase)), i: float32(math.Sin(phase))}
	}
}

// newFFTState is upstream's rnn_fft_alloc_twiddles with base == NULL. It
// reports false for sizes kfFactor cannot handle, leaving the caller to pick a
// Bluestein transform instead.
func newFFTState(nfft int) (*fftState, bool) {
	st := &fftState{
		nfft:  nfft,
		scale: 1 / float32(nfft),
		shift: -1,
	}
	st.twiddles = make([]cpx, nfft)
	computeTwiddles(st.twiddles, nfft)
	if !kfFactor(nfft, &st.factors) {
		return nil, false
	}
	st.bitrev = make([]int32, nfft)
	computeBitrevTable(0, st.bitrev, 0, 1, 1, st.factors[:])
	return st, true
}

func kfBfly2(fout []cpx, m, n int) {
	if m == 1 {
		// Degenerate case where all the twiddles are 1. Upstream guards this
		// with CUSTOM_MODES; reachable here for sizes ending on a radix-2
		// stage, such as 882.
		p := 0
		for i := 0; i < n; i++ {
			t := fout[p+1]
			fout[p+1] = cpx{fout[p].r - t.r, fout[p].i - t.i}
			fout[p].r += t.r
			fout[p].i += t.i
			p += 2
		}
		return
	}
	// We know that m==4 here because the radix-2 is just after a radix-4.
	const tw float32 = 0.7071067812
	p := 0
	for i := 0; i < n; i++ {
		q := p + 4
		var t cpx

		t = fout[q]
		fout[q] = cpx{fout[p].r - t.r, fout[p].i - t.i}
		fout[p].r += t.r
		fout[p].i += t.i

		t.r = (fout[q+1].r + fout[q+1].i) * tw
		t.i = (fout[q+1].i - fout[q+1].r) * tw
		fout[q+1] = cpx{fout[p+1].r - t.r, fout[p+1].i - t.i}
		fout[p+1].r += t.r
		fout[p+1].i += t.i

		t.r = fout[q+2].i
		t.i = -fout[q+2].r
		fout[q+2] = cpx{fout[p+2].r - t.r, fout[p+2].i - t.i}
		fout[p+2].r += t.r
		fout[p+2].i += t.i

		t.r = (fout[q+3].i - fout[q+3].r) * tw
		t.i = -(fout[q+3].i + fout[q+3].r) * tw
		fout[q+3] = cpx{fout[p+3].r - t.r, fout[p+3].i - t.i}
		fout[p+3].r += t.r
		fout[p+3].i += t.i

		p += 8
	}
}

func kfBfly4(fout []cpx, fstride int, st *fftState, m, n, mm int) {
	if m == 1 {
		// Degenerate case where all the twiddles are 1.
		p := 0
		for i := 0; i < n; i++ {
			var scratch0, scratch1 cpx

			scratch0 = cpx{fout[p].r - fout[p+2].r, fout[p].i - fout[p+2].i}
			fout[p].r += fout[p+2].r
			fout[p].i += fout[p+2].i
			scratch1 = cpx{fout[p+1].r + fout[p+3].r, fout[p+1].i + fout[p+3].i}
			fout[p+2] = cpx{fout[p].r - scratch1.r, fout[p].i - scratch1.i}
			fout[p].r += scratch1.r
			fout[p].i += scratch1.i
			scratch1 = cpx{fout[p+1].r - fout[p+3].r, fout[p+1].i - fout[p+3].i}

			fout[p+1].r = scratch0.r + scratch1.i
			fout[p+1].i = scratch0.i - scratch1.r
			fout[p+3].r = scratch0.r - scratch1.i
			fout[p+3].i = scratch0.i + scratch1.r
			p += 4
		}
		return
	}
	var scratch [6]cpx
	m2 := 2 * m
	m3 := 3 * m
	for i := 0; i < n; i++ {
		f := i * mm
		tw1, tw2, tw3 := 0, 0, 0
		// m is guaranteed to be a multiple of 4.
		for j := 0; j < m; j++ {
			scratch[0] = cmul(fout[f+m], st.twiddles[tw1])
			scratch[1] = cmul(fout[f+m2], st.twiddles[tw2])
			scratch[2] = cmul(fout[f+m3], st.twiddles[tw3])

			scratch[5] = cpx{fout[f].r - scratch[1].r, fout[f].i - scratch[1].i}
			fout[f].r += scratch[1].r
			fout[f].i += scratch[1].i
			scratch[3] = cpx{scratch[0].r + scratch[2].r, scratch[0].i + scratch[2].i}
			scratch[4] = cpx{scratch[0].r - scratch[2].r, scratch[0].i - scratch[2].i}
			fout[f+m2] = cpx{fout[f].r - scratch[3].r, fout[f].i - scratch[3].i}
			tw1 += fstride
			tw2 += fstride * 2
			tw3 += fstride * 3
			fout[f].r += scratch[3].r
			fout[f].i += scratch[3].i

			fout[f+m].r = scratch[5].r + scratch[4].i
			fout[f+m].i = scratch[5].i - scratch[4].r
			fout[f+m3].r = scratch[5].r - scratch[4].i
			fout[f+m3].i = scratch[5].i + scratch[4].r
			f++
		}
	}
}

func kfBfly3(fout []cpx, fstride int, st *fftState, m, n, mm int) {
	m2 := 2 * m
	var scratch [5]cpx
	epi3 := st.twiddles[fstride*m]
	for i := 0; i < n; i++ {
		f := i * mm
		tw1, tw2 := 0, 0
		for k := 0; k < m; k++ {
			scratch[1] = cmul(fout[f+m], st.twiddles[tw1])
			scratch[2] = cmul(fout[f+m2], st.twiddles[tw2])

			scratch[3] = cpx{scratch[1].r + scratch[2].r, scratch[1].i + scratch[2].i}
			scratch[0] = cpx{scratch[1].r - scratch[2].r, scratch[1].i - scratch[2].i}
			tw1 += fstride
			tw2 += fstride * 2

			fout[f+m].r = fout[f].r - scratch[3].r*0.5
			fout[f+m].i = fout[f].i - scratch[3].i*0.5

			scratch[0].r *= epi3.i
			scratch[0].i *= epi3.i

			fout[f].r += scratch[3].r
			fout[f].i += scratch[3].i

			fout[f+m2].r = fout[f+m].r + scratch[0].i
			fout[f+m2].i = fout[f+m].i - scratch[0].r

			fout[f+m].r = fout[f+m].r - scratch[0].i
			fout[f+m].i = fout[f+m].i + scratch[0].r

			f++
		}
	}
}

func kfBfly5(fout []cpx, fstride int, st *fftState, m, n, mm int) {
	var scratch [13]cpx
	tw := st.twiddles
	ya := tw[fstride*m]
	yb := tw[fstride*2*m]
	for i := 0; i < n; i++ {
		f0 := i * mm
		f1 := f0 + m
		f2 := f0 + 2*m
		f3 := f0 + 3*m
		f4 := f0 + 4*m

		for u := 0; u < m; u++ {
			scratch[0] = fout[f0]

			scratch[1] = cmul(fout[f1], tw[u*fstride])
			scratch[2] = cmul(fout[f2], tw[2*u*fstride])
			scratch[3] = cmul(fout[f3], tw[3*u*fstride])
			scratch[4] = cmul(fout[f4], tw[4*u*fstride])

			scratch[7] = cpx{scratch[1].r + scratch[4].r, scratch[1].i + scratch[4].i}
			scratch[10] = cpx{scratch[1].r - scratch[4].r, scratch[1].i - scratch[4].i}
			scratch[8] = cpx{scratch[2].r + scratch[3].r, scratch[2].i + scratch[3].i}
			scratch[9] = cpx{scratch[2].r - scratch[3].r, scratch[2].i - scratch[3].i}

			fout[f0].r = fout[f0].r + (scratch[7].r + scratch[8].r)
			fout[f0].i = fout[f0].i + (scratch[7].i + scratch[8].i)

			scratch[5].r = scratch[0].r + (scratch[7].r*ya.r + scratch[8].r*yb.r)
			scratch[5].i = scratch[0].i + (scratch[7].i*ya.r + scratch[8].i*yb.r)

			scratch[6].r = scratch[10].i*ya.i + scratch[9].i*yb.i
			scratch[6].i = -(scratch[10].r*ya.i + scratch[9].r*yb.i)

			fout[f1] = cpx{scratch[5].r - scratch[6].r, scratch[5].i - scratch[6].i}
			fout[f4] = cpx{scratch[5].r + scratch[6].r, scratch[5].i + scratch[6].i}

			scratch[11].r = scratch[0].r + (scratch[7].r*yb.r + scratch[8].r*ya.r)
			scratch[11].i = scratch[0].i + (scratch[7].i*yb.r + scratch[8].i*ya.r)
			scratch[12].r = scratch[9].i*ya.i - scratch[10].i*yb.i
			scratch[12].i = scratch[10].r*yb.i - scratch[9].r*ya.i

			fout[f2] = cpx{scratch[11].r + scratch[12].r, scratch[11].i + scratch[12].i}
			fout[f3] = cpx{scratch[11].r - scratch[12].r, scratch[11].i - scratch[12].i}

			f0++
			f1++
			f2++
			f3++
			f4++
		}
	}
}

// kfBfly7 is a radix-7 butterfly in the same idiom as kf_bfly3/5, which
// upstream does not have: its kf_factor rejects any prime above 5.
//
// 44.1 kHz needs it. That window is 882 = 2*3^2*7^2, so without this the two
// radix-7 stages fall to kfBflyGeneric's O(p^2) inner DFT, and 44.1 kHz ends up
// *slower* than 48 kHz despite having fewer samples -- measured 148 us against
// 108 us per frame before this existed.
//
// The saving comes from folding the seven-point DFT over conjugate pairs. With
// W = exp(-2i*pi/7) and the pairs (1,6), (2,5), (3,4),
//
//	X[k] = x0 + sum_j [ (x_j + x_{7-j})*cos(2*pi*j*k/7)
//	                  - i*(x_j - x_{7-j})*sin(2*pi*j*k/7) ]
//
// and X[7-k] is the same with the sine term added rather than subtracted, since
// cos is even and sin odd about 2*pi*j. So three sums, three differences and
// three real-scalar passes replace 49 complex multiplies -- the same trick
// kf_bfly5 uses for its two pairs.
//
// The cosines and sines come from the twiddle table rather than from literals:
// at a radix-p stage fstride*m == nfft/p, so twiddles[fstride*m*j] is exactly
// W^j. That keeps them consistent with every other stage's rounding.
func kfBfly7(fout []cpx, fstride int, st *fftState, m, n, mm int) {
	tw := st.twiddles
	ya := tw[fstride*m]   // W^1
	yb := tw[fstride*2*m] // W^2
	yc := tw[fstride*3*m] // W^3
	// The table holds exp(-i*theta), so the imaginary parts are negated sines.
	s1, s2, s3 := -ya.i, -yb.i, -yc.i

	for i := 0; i < n; i++ {
		f := i * mm
		for u := 0; u < m; u++ {
			x0 := fout[f]
			x1 := cmul(fout[f+m], tw[u*fstride])
			x2 := cmul(fout[f+2*m], tw[2*u*fstride])
			x3 := cmul(fout[f+3*m], tw[3*u*fstride])
			x4 := cmul(fout[f+4*m], tw[4*u*fstride])
			x5 := cmul(fout[f+5*m], tw[5*u*fstride])
			x6 := cmul(fout[f+6*m], tw[6*u*fstride])

			p1 := cpx{x1.r + x6.r, x1.i + x6.i}
			q1 := cpx{x1.r - x6.r, x1.i - x6.i}
			p2 := cpx{x2.r + x5.r, x2.i + x5.i}
			q2 := cpx{x2.r - x5.r, x2.i - x5.i}
			p3 := cpx{x3.r + x4.r, x3.i + x4.i}
			q3 := cpx{x3.r - x4.r, x3.i - x4.i}

			fout[f] = cpx{
				x0.r + p1.r + p2.r + p3.r,
				x0.i + p1.i + p2.i + p3.i,
			}

			// k = 1: cosines (W1,W2,W3), sines (s1,s2,s3).
			ar := x0.r + p1.r*ya.r + p2.r*yb.r + p3.r*yc.r
			ai := x0.i + p1.i*ya.r + p2.i*yb.r + p3.i*yc.r
			br := q1.r*s1 + q2.r*s2 + q3.r*s3
			bi := q1.i*s1 + q2.i*s2 + q3.i*s3
			fout[f+m] = cpx{ar + bi, ai - br}
			fout[f+6*m] = cpx{ar - bi, ai + br}

			// k = 2: j*k mod 7 gives cosines (W2,W3,W1) and sines
			// (s2,-s3,-s1), the signs following sin's oddness.
			ar = x0.r + p1.r*yb.r + p2.r*yc.r + p3.r*ya.r
			ai = x0.i + p1.i*yb.r + p2.i*yc.r + p3.i*ya.r
			br = q1.r*s2 - q2.r*s3 - q3.r*s1
			bi = q1.i*s2 - q2.i*s3 - q3.i*s1
			fout[f+2*m] = cpx{ar + bi, ai - br}
			fout[f+5*m] = cpx{ar - bi, ai + br}

			// k = 3: cosines (W3,W1,W2), sines (s3,-s1,s2).
			ar = x0.r + p1.r*yc.r + p2.r*ya.r + p3.r*yb.r
			ai = x0.i + p1.i*yc.r + p2.i*ya.r + p3.i*yb.r
			br = q1.r*s3 - q2.r*s1 + q3.r*s2
			bi = q1.i*s3 - q2.i*s1 + q3.i*s2
			fout[f+3*m] = cpx{ar + bi, ai - br}
			fout[f+4*m] = cpx{ar - bi, ai + br}

			f++
		}
	}
}

// kfBflyGeneric is KISS FFT's kf_bfly_generic adapted to CELT's (N, mm)
// calling convention. It handles any radix the specialised butterflies do not,
// which for the rates we support means 7 (44.1 kHz) and 11 (22.05 kHz).
// Upstream has no equivalent, so there is no arithmetic order to match.
func kfBflyGeneric(fout []cpx, fstride int, st *fftState, m, n, mm, p int) {
	var scratch [maxGenericRadix]cpx
	norig := st.nfft
	for i := 0; i < n; i++ {
		base := i * mm
		for u := 0; u < m; u++ {
			k := u
			for q1 := 0; q1 < p; q1++ {
				scratch[q1] = fout[base+k]
				k += m
			}
			k = u
			for q1 := 0; q1 < p; q1++ {
				twidx := 0
				fout[base+k] = scratch[0]
				for q := 1; q < p; q++ {
					twidx += fstride * k
					if twidx >= norig {
						twidx -= norig
					}
					t := cmul(scratch[q], st.twiddles[twidx])
					fout[base+k].r += t.r
					fout[base+k].i += t.i
				}
				k += m
			}
		}
	}
}

// fftImpl is upstream's rnn_fft_impl: the in-place butterfly cascade.
func (st *fftState) fftImpl(fout []cpx) {
	var fstride [maxFactors + 1]int
	shift := st.shift
	if shift <= 0 {
		shift = 0
	}

	fstride[0] = 1
	l := 0
	var m int
	for {
		p := st.factors[2*l]
		m = st.factors[2*l+1]
		fstride[l+1] = fstride[l] * p
		l++
		if m == 1 {
			break
		}
	}
	m = st.factors[2*l-1]
	for i := l - 1; i >= 0; i-- {
		m2 := 1
		if i != 0 {
			m2 = st.factors[2*i-1]
		}
		switch p := st.factors[2*i]; p {
		case 2:
			kfBfly2(fout, m, fstride[i])
		case 4:
			kfBfly4(fout, fstride[i]<<shift, st, m, fstride[i], m2)
		case 3:
			kfBfly3(fout, fstride[i]<<shift, st, m, fstride[i], m2)
		case 5:
			kfBfly5(fout, fstride[i]<<shift, st, m, fstride[i], m2)
		case 7:
			kfBfly7(fout, fstride[i]<<shift, st, m, fstride[i], m2)
		default:
			kfBflyGeneric(fout, fstride[i]<<shift, st, m, fstride[i], m2, p)
		}
		m = m2
	}
}

func (st *fftState) size() int { return st.nfft }

// fft is upstream's rnn_fft_c: scale, bit-reverse, then run the cascade. It is
// the 1/nfft-scaled forward transform. In-place is not supported (fin and fout
// must not alias), matching upstream.
func (st *fftState) fft(fin, fout []cpx) {
	scale := st.scale
	for i := 0; i < st.nfft; i++ {
		x := fin[i]
		fout[st.bitrev[i]] = cpx{scale * x.r, scale * x.i}
	}
	st.fftImpl(fout)
}

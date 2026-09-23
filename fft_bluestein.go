package rnnoise

import "math"

// Bluestein's algorithm (the chirp-z transform), the fallback for transform
// sizes the mixed-radix path cannot factor.
//
// kfFactor handles any size whose largest prime factor is at most
// maxGenericRadix, which covers every sample rate anyone is likely to ask for:
// all of 8/12/16/24/32/44.1/48 kHz factor into 2, 3, 5 and 7, and 22.05 kHz
// needs 11. But a rate like 8200 Hz gives a 164-sample window = 4*41, and 41
// exceeds what a stack-resident generic butterfly should attempt. Rather than
// reject such rates, this expresses the awkward transform as two power-of-two
// transforms, so New succeeds for any integer rate at or above MinSampleRate.
//
// The identity is nk = (n^2 + k^2 - (k-n)^2)/2, which turns the DFT into a
// convolution:
//
//	X[k] = conj(w[k]) * sum_n (x[n]*conj(w[n])) * w[k-n],   w[m] = exp(i*pi*m^2/N)
//
// The convolution is then evaluated with a power-of-two FFT of length
// M >= 2N-1. Cost is roughly five times a native transform of the same size,
// which is irrelevant for a rate nobody uses in anger.
//
// Accuracy is lower than a native transform -- the chirp multiplies add
// rounding, and M is larger -- so this is never used for a rate the mixed-radix
// path can handle, and in particular never at 48 kHz, where bit-exactness
// against the C reference is a requirement.
type bluesteinState struct {
	n     int
	m     int
	inner *fftState

	// chirp holds conj(w[k]) for k < n, applied before and after the
	// convolution.
	chirp []cpx
	// kernel is the transformed, pre-scaled convolution kernel.
	kernel []cpx

	// Scratch, so a transform allocates nothing.
	a  []cpx
	fa []cpx
	pc []cpx
	cc []cpx
}

func (b *bluesteinState) size() int { return b.n }

func newBluesteinState(n int) (*bluesteinState, bool) {
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}
	inner, ok := newFFTState(m)
	if !ok {
		return nil, false
	}

	b := &bluesteinState{
		n:      n,
		m:      m,
		inner:  inner,
		chirp:  make([]cpx, n),
		kernel: make([]cpx, m),
		a:      make([]cpx, m),
		fa:     make([]cpx, m),
		pc:     make([]cpx, m),
		cc:     make([]cpx, m),
	}

	// w[j] = exp(i*pi*j^2/n). The exponent is reduced modulo 2n before the
	// trig call: j^2 grows as n^2, and for n in the thousands that would cost
	// most of float64's precision before math.Sincos ever sees it.
	w := func(j int) (float64, float64) {
		jj := (j * j) % (2 * n)
		ang := math.Pi * float64(jj) / float64(n)
		s, c := math.Sincos(ang)
		return c, s
	}

	for k := 0; k < n; k++ {
		c, s := w(k)
		b.chirp[k] = cpx{r: float32(c), i: float32(-s)} // conj(w[k])
	}

	// The kernel is w[j] for j in [0,n), wrapped so that index m-j carries
	// w[j] for the negative lags, which is what makes the length-m circular
	// convolution agree with the linear one over the range that matters.
	kern := make([]cpx, m)
	for j := 0; j < n; j++ {
		c, s := w(j)
		v := cpx{r: float32(c), i: float32(s)}
		kern[j] = v
		if j != 0 {
			kern[m-j] = v
		}
	}
	// Fold the convolution's normalisation and this package's 1/n forward
	// scaling into the kernel, so the hot path has no extra pass.
	//
	// The inner transform is itself 1/m-scaled, so recovering an unnormalised
	// circular convolution costs a factor of m*m; the forward transform this
	// implements is 1/n-scaled, hence m*m/n.
	b.inner.fft(kern, b.kernel)
	s := float32(float64(m) * float64(m) / float64(n))
	for i := range b.kernel {
		b.kernel[i].r *= s
		b.kernel[i].i *= s
	}
	return b, true
}

// fft computes the 1/n-scaled forward transform, matching fftState.fft.
func (b *bluesteinState) fft(fin, fout []cpx) {
	n, m := b.n, b.m

	for i := 0; i < n; i++ {
		b.a[i] = cmul(fin[i], b.chirp[i])
	}
	for i := n; i < m; i++ {
		b.a[i] = cpx{}
	}
	b.inner.fft(b.a, b.fa)

	// Multiply by the kernel, then conjugate: the inverse transform is
	// conj(fft(conj(.))) because fft is itself 1/m-scaled.
	for i := 0; i < m; i++ {
		p := cmul(b.fa[i], b.kernel[i])
		b.pc[i] = cpx{r: p.r, i: -p.i}
	}
	b.inner.fft(b.pc, b.cc)

	for k := 0; k < n; k++ {
		c := cpx{r: b.cc[k].r, i: -b.cc[k].i}
		fout[k] = cmul(c, b.chirp[k])
	}
}

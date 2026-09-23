package rnnoise

// transform owns the scratch buffers the FFT wrappers need, so the per-frame
// path never allocates.
type transform struct {
	c *rateConfig
	x []cpx
	y []cpx
}

func newTransform(c *rateConfig) *transform {
	return &transform{c: c, x: make([]cpx, c.window), y: make([]cpx, c.window)}
}

// applyWindow is upstream's apply_window: the power-complementary half-window
// applied symmetrically over the 2-frame analysis buffer.
func (c *rateConfig) applyWindow(x []float32) {
	w := c.halfWindow
	for i := 0; i < c.frame; i++ {
		x[i] *= w[i]
		x[c.window-1-i] *= w[i]
	}
}

// forward is upstream's forward_transform: a full complex FFT of a real
// signal with the imaginary part zeroed, keeping the first freq bins.
//
// Upstream leaves the obvious 2x real-input optimisation on the table, and so
// do we. The three transforms are about 4% of a frame, so halving them buys
// ~2% end to end, and it would cost the exact arithmetic order that makes the
// 48 kHz path bit-comparable against C.
func (t *transform) forward(out []cpx, in []float32) {
	for i := 0; i < t.c.window; i++ {
		t.x[i] = cpx{r: in[i], i: 0}
	}
	t.c.fft.fft(t.x, t.y)
	copy(out[:t.c.freq], t.y[:t.c.freq])
}

// inverse is upstream's inverse_transform. It hermitian-extends the half
// spectrum, runs the *forward* transform again, and reads the result in
// reverse order scaled by the window size. Since the forward transform is
// 1/N-scaled, that composition is the unnormalised inverse, and the pair
// round-trips to identity.
func (t *transform) inverse(out []float32, in []cpx) {
	c := t.c
	copy(t.x[:c.freq], in[:c.freq])
	for i := c.freq; i < c.window; i++ {
		t.x[i] = cpx{r: t.x[c.window-i].r, i: -t.x[c.window-i].i}
	}
	c.fft.fft(t.x, t.y)
	// Output in reverse order for the IFFT.
	n := float32(c.window)
	out[0] = n * t.y[0].r
	for i := 1; i < c.window; i++ {
		out[i] = n * t.y[c.window-i].r
	}
}

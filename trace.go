package rnnoise

// trace captures the one per-frame intermediate that processFrame would
// otherwise overwrite before a test could read it.
//
// Every other stage the differential harness compares is still live in the
// Denoiser when the frame returns: d.hp, d.X, d.Ex, d.P, d.Ep, d.Exp,
// d.features, d.gf, d.rnn's GRU states, and -- because computeRNN's skip
// connections only write cat[conv2OutSize:] -- d.nn.tmp still holds conv1's
// output and d.nn.cat[:conv2OutSize] still holds conv2's. So the production
// path needs no instrumentation beyond this, and the harness reads the real
// buffers rather than copies of them.
type trace struct {
	// gRaw is the network's gain output before the 0.6-per-frame decay cap
	// rewrites d.g in place.
	gRaw []float32

	// silence is the last frame's silence-gate decision. The gate zeroes the
	// features and skips the network, so it cannot be inferred reliably from
	// the outputs alone -- a genuinely zero feature vector is possible without
	// the gate firing.
	silence bool
}

// enableTrace turns on intermediate capture. Unexported: the harness lives in
// this package, and exposing a probe API would enlarge the public surface for
// no caller's benefit.
func (d *Denoiser) enableTrace() {
	d.trc = &trace{gRaw: make([]float32, numBands)}
}

package rnnoise

// Layer drivers, ported from upstream's src/nnet.c (the compute_generic_*
// family) and src/nnet_arch.h (compute_linear_c / compute_activation_c).

// nnScratch holds the per-stream temporaries the network needs, so a frame
// costs no allocation. Sized from the model's largest layer.
type nnScratch struct {
	q       []int16 // quantised activations, widened; see simd.go
	q8      []int8  // the same narrowed to int8, for arm64's SDOT kernel
	acc     []int32 // raw int32 accumulator, likewise
	zrh     []float32
	recur   []float32
	convTmp []float32
	tmp     []float32
	cat     []float32
	vad     [vadSize]float32
}

func newNNScratch() *nnScratch {
	return &nnScratch{
		q:       make([]int16, catSize),
		q8:      make([]int8, catSize),
		acc:     make([]int32, gruGates),
		zrh:     make([]float32, gruGates),
		recur:   make([]float32, gruGates),
		convTmp: make([]float32, conv2Kernel*conv2InSize),
		tmp:     make([]float32, conv1OutSize),
		cat:     make([]float32, catSize),
	}
}

// rnnState is upstream's RNNState: the recurrent and convolution history.
type rnnState struct {
	conv1State []float32
	conv2State []float32
	gru1State  []float32
	gru2State  []float32
	gru3State  []float32
}

func newRNNState() *rnnState {
	return &rnnState{
		// A causal kernel of width k keeps (k-1) frames of input history.
		conv1State: make([]float32, conv1InSize*(conv1Kernel-1)),
		conv2State: make([]float32, conv2InSize*(conv2Kernel-1)),
		gru1State:  make([]float32, gruSize),
		gru2State:  make([]float32, gruSize),
		gru3State:  make([]float32, gruSize),
	}
}

func (s *rnnState) reset() {
	for _, b := range [][]float32{s.conv1State, s.conv2State, s.gru1State, s.gru2State, s.gru3State} {
		for i := range b {
			b[i] = 0
		}
	}
}

// compute is upstream's compute_linear_c.
func (l *linearLayer) compute(out, in []float32, q QuantMode, sc *nnScratch) {
	m, n := l.nbInputs, l.nbOutputs
	bias := l.bias

	switch {
	case l.floatWeights != nil:
		if l.weightsIdx != nil {
			sparseSgemv(out, l.floatWeights, l.weightsIdx, n, in)
		} else {
			sgemv(out, l.floatWeights, n, m, n, in)
		}
	case l.weights != nil:
		qa := sc.q[:m]
		quantize(qa, in[:m], q)
		if l.weightsIdx != nil {
			sparseCgemvInt8(out, l.weights, l.weightsIdx, l.scale, n, m, qa)
		} else {
			cgemvInt8(out, l.weights, l.scale, n, m, qa, sc.acc[:n], sc.q8[:m])
		}
		// Only use SU biases for integer matrices on SU archs.
		if q == QuantUnsigned {
			bias = l.subias
		}
	default:
		for i := 0; i < n; i++ {
			out[i] = 0
		}
	}

	if bias != nil {
		for i := 0; i < n; i++ {
			out[i] += bias[i]
		}
	}
	if l.diag != nil {
		// Diag is only used for GRU recurrent weights, where 3*M == N.
		for i := 0; i < m; i++ {
			out[i] += l.diag[i] * in[i]
			out[i+m] += l.diag[i+m] * in[i]
			out[i+2*m] += l.diag[i+2*m] * in[i]
		}
	}
}

// computeDense is upstream's compute_generic_dense.
func computeDense(l *linearLayer, out, in []float32, activation int, q QuantMode, sc *nnScratch) {
	l.compute(out, in, q, sc)
	computeActivation(out[:l.nbOutputs], out[:l.nbOutputs], activation)
}

// computeConv1d is upstream's compute_generic_conv1d: a causal 1-D convolution
// expressed as a linear layer over a flattened k-frame window, with mem
// carrying the (k-1) previous frames. There is no lookahead.
func computeConv1d(l *linearLayer, out, mem, in []float32, inputSize, activation int, q QuantMode, sc *nnScratch) {
	tmp := sc.convTmp[:l.nbInputs]
	hist := l.nbInputs - inputSize
	if hist != 0 {
		copy(tmp[:hist], mem[:hist])
	}
	copy(tmp[hist:], in[:inputSize])
	l.compute(out, tmp, q, sc)
	computeActivation(out[:l.nbOutputs], out[:l.nbOutputs], activation)
	if hist != 0 {
		copy(mem[:hist], tmp[inputSize:])
	}
}

// computeGRU is upstream's compute_generic_gru.
//
// Two details are easy to get wrong and are load-bearing. The gate order in the
// weight layout is z, r, h -- not PyTorch's r, z, n; upstream's exporter
// reorders on the way out. And z gates the *old* state: the update is
// h = z*state + (1-z)*h, which is inverted relative to the textbook
// h = (1-z)*state + z*h.
func computeGRU(iw, rw *linearLayer, state, in []float32, q QuantMode, sc *nnScratch) {
	n := rw.nbInputs
	zrh := sc.zrh[:3*n]
	recur := sc.recur[:3*n]

	iw.compute(zrh, in, q, sc)
	rw.compute(recur, state, q, sc)

	for i := 0; i < 2*n; i++ {
		zrh[i] += recur[i]
	}
	computeActivation(zrh[:2*n], zrh[:2*n], actSigmoid)

	z := zrh[0:n]
	r := zrh[n : 2*n]
	h := zrh[2*n : 3*n]

	for i := 0; i < n; i++ {
		h[i] += recur[2*n+i] * r[i]
	}
	computeActivation(h, h, actTanh)
	for i := 0; i < n; i++ {
		h[i] = z[i]*state[i] + (1-z[i])*h[i]
	}
	copy(state[:n], h)
}

// computeRNN is upstream's compute_rnn: two causal convolutions feeding three
// stacked GRUs, with the conv2 output and all three GRU states concatenated
// into the two output heads.
func (m *Model) computeRNN(st *rnnState, gains []float32, features []float32, q QuantMode, sc *nnScratch) float32 {
	tmp := sc.tmp[:conv1OutSize]
	cat := sc.cat[:catSize]

	computeConv1d(&m.conv1, tmp, st.conv1State, features, conv1InSize, actTanh, q, sc)
	computeConv1d(&m.conv2, cat, st.conv2State, tmp, conv2InSize, actTanh, q, sc)

	computeGRU(&m.gru1Input, &m.gru1Recurrent, st.gru1State, cat, q, sc)
	computeGRU(&m.gru2Input, &m.gru2Recurrent, st.gru2State, st.gru1State, q, sc)
	computeGRU(&m.gru3Input, &m.gru3Recurrent, st.gru3State, st.gru2State, q, sc)

	// Skip connections: the heads see the conv2 output alongside every GRU.
	copy(cat[conv2OutSize:], st.gru1State)
	copy(cat[conv2OutSize+gruSize:], st.gru2State)
	copy(cat[conv2OutSize+2*gruSize:], st.gru3State)

	computeDense(&m.denseOut, gains, cat, actSigmoid, q, sc)
	computeDense(&m.vadDense, sc.vad[:], cat, actSigmoid, q, sc)
	return sc.vad[0]
}

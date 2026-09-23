package rnnoise

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Weight blob format, from src/nnet.h and src/write_weights.c: a sequence of
// 64-byte WeightHead records, each followed by its payload zero-padded up to a
// multiple of 64 bytes.
//
// Upstream writes these native-endian with no byte-order marker, which only
// works because it generates and consumes them on one machine. We define the
// format as little-endian; tools/blobgen writes that, and the reader below
// enforces the magic and version so a native-endian blob from upstream's own
// dump_weights_blob on a big-endian host is rejected rather than misread.
const (
	weightBlobVersion = 0
	weightBlockSize   = 64
	weightNameMax     = 44
	sparseBlockSize   = 32 // 8 rows x 4 columns
)

const (
	weightTypeFloat   = 0
	weightTypeInt     = 1
	weightTypeQWeight = 2
	weightTypeInt8    = 3
)

// Layer dimensions, mirroring the generated src/rnnoise_data.h. They are
// asserted against the blob at load time so a mismatched model fails loudly
// instead of producing noise.
const (
	conv1InSize  = numFeatures // 65
	conv1OutSize = 128
	conv1Kernel  = 3
	conv2InSize  = conv1OutSize // 128
	conv2OutSize = 384
	conv2Kernel  = 3
	gruSize      = 384
	gruGates     = 3 * gruSize // 1152: the z, r and h gates
	catSize      = conv2OutSize + 3*gruSize
	gainsSize    = numBands
	vadSize      = 1
)

// QuantMode selects the int8 activation convention for quantised layers.
type QuantMode int

const (
	// QuantSigned quantises activations to int8 as round(127*x) and uses the
	// layer's `bias`. This is the exact, non-saturating convention, and it is
	// what upstream's portable scalar build, its ARM NEON build and the
	// rnnoise.wasm build all use. It is the default because it is the only one
	// for which a bit-exact C oracle can be built.
	QuantSigned QuantMode = iota

	// QuantUnsigned quantises to 127+round(127*x) and uses `subias`. This is
	// upstream's USE_SU_BIAS path, which every stock x86 build takes because
	// vec.h selects vec_avx.h whenever __SSE2__ is defined. It exists so the
	// differential harness can compare like-for-like against such a build.
	QuantUnsigned
)

func (q QuantMode) String() string {
	if q == QuantUnsigned {
		return "unsigned"
	}
	return "signed"
}

// linearLayer mirrors upstream's LinearLayer: a possibly-quantised,
// possibly-sparse affine transform.
type linearLayer struct {
	bias         []float32
	subias       []float32
	weights      []int8
	floatWeights []float32
	weightsIdx   []int32
	diag         []float32
	scale        []float32
	nbInputs     int
	nbOutputs    int
}

// Model holds the trained weights. It is read-only after loading, so one Model
// may be shared by any number of Denoisers, including concurrently.
type Model struct {
	conv1 linearLayer
	conv2 linearLayer

	gru1Input     linearLayer
	gru1Recurrent linearLayer
	gru2Input     linearLayer
	gru2Recurrent linearLayer
	gru3Input     linearLayer
	gru3Recurrent linearLayer

	denseOut linearLayer
	vadDense linearLayer
}

type weightArray struct {
	typ   int
	f32   []float32
	i32   []int32
	i8    []int8
	bytes int
}

// LoadModel reads a weight blob.
func LoadModel(r io.Reader) (*Model, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("rnnoise: reading model: %w", err)
	}
	return LoadModelBytes(b)
}

// LoadModelFile reads a weight blob from disk.
func LoadModelFile(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("rnnoise: opening model: %w", err)
	}
	defer f.Close()
	return LoadModel(f)
}

// LoadModelBytes parses a weight blob held in memory. b is decoded, not
// retained, so the caller may reuse or free it afterwards.
func LoadModelBytes(b []byte) (*Model, error) {
	arrays, err := parseWeights(b)
	if err != nil {
		return nil, err
	}
	m := &Model{}
	type spec struct {
		layer                                    *linearLayer
		name                                     string
		bias, subias, weights, floatWeights, idx string
		diag, scale                              string
		nbInputs, nbOutputs                      int
	}
	// This table mirrors init_rnnoise() in the generated rnnoise_data.c.
	specs := []spec{
		{&m.conv1, "conv1", "conv1_bias", "", "", "conv1_weights_float", "", "", "", conv1Kernel * conv1InSize, conv1OutSize},
		{&m.conv2, "conv2", "conv2_bias", "conv2_subias", "conv2_weights_int8", "conv2_weights_float", "", "", "conv2_scale", conv2Kernel * conv2InSize, conv2OutSize},
		{&m.gru1Input, "gru1_input", "gru1_input_bias", "gru1_input_subias", "gru1_input_weights_int8", "gru1_input_weights_float", "gru1_input_weights_idx", "", "gru1_input_scale", gruSize, gruGates},
		{&m.gru1Recurrent, "gru1_recurrent", "gru1_recurrent_bias", "gru1_recurrent_subias", "gru1_recurrent_weights_int8", "gru1_recurrent_weights_float", "gru1_recurrent_weights_idx", "gru1_recurrent_weights_diag", "gru1_recurrent_scale", gruSize, gruGates},
		{&m.gru2Input, "gru2_input", "gru2_input_bias", "gru2_input_subias", "gru2_input_weights_int8", "gru2_input_weights_float", "gru2_input_weights_idx", "", "gru2_input_scale", gruSize, gruGates},
		{&m.gru2Recurrent, "gru2_recurrent", "gru2_recurrent_bias", "gru2_recurrent_subias", "gru2_recurrent_weights_int8", "gru2_recurrent_weights_float", "gru2_recurrent_weights_idx", "gru2_recurrent_weights_diag", "gru2_recurrent_scale", gruSize, gruGates},
		{&m.gru3Input, "gru3_input", "gru3_input_bias", "gru3_input_subias", "gru3_input_weights_int8", "gru3_input_weights_float", "gru3_input_weights_idx", "", "gru3_input_scale", gruSize, gruGates},
		{&m.gru3Recurrent, "gru3_recurrent", "gru3_recurrent_bias", "gru3_recurrent_subias", "gru3_recurrent_weights_int8", "gru3_recurrent_weights_float", "gru3_recurrent_weights_idx", "gru3_recurrent_weights_diag", "gru3_recurrent_scale", gruSize, gruGates},
		{&m.denseOut, "dense_out", "dense_out_bias", "", "", "dense_out_weights_float", "", "", "", catSize, gainsSize},
		{&m.vadDense, "vad_dense", "vad_dense_bias", "", "", "vad_dense_weights_float", "", "", "", catSize, vadSize},
	}
	for _, s := range specs {
		if err := linearInit(s.layer, arrays, s.bias, s.subias, s.weights, s.floatWeights,
			s.idx, s.diag, s.scale, s.nbInputs, s.nbOutputs); err != nil {
			return nil, fmt.Errorf("rnnoise: model layer %s: %w", s.name, err)
		}
	}
	return m, nil
}

func parseWeights(b []byte) (map[string]*weightArray, error) {
	arrays := map[string]*weightArray{}
	for off := 0; off < len(b); {
		if len(b)-off < weightBlockSize {
			return nil, errors.New("rnnoise: model truncated in a record header")
		}
		head := b[off : off+weightBlockSize]
		if !bytes.Equal(head[0:4], []byte("DNNw")) {
			return nil, fmt.Errorf("rnnoise: bad record magic %q at offset %d; "+
				"is this a DNNw weight blob?", head[0:4], off)
		}
		version := int32(binary.LittleEndian.Uint32(head[4:]))
		typ := int32(binary.LittleEndian.Uint32(head[8:]))
		size := int32(binary.LittleEndian.Uint32(head[12:]))
		blockSize := int32(binary.LittleEndian.Uint32(head[16:]))
		if version != weightBlobVersion {
			return nil, fmt.Errorf("rnnoise: model blob version %d, want %d", version, weightBlobVersion)
		}
		if size < 0 || blockSize < size {
			return nil, fmt.Errorf("rnnoise: record at offset %d has size %d, block %d", off, size, blockSize)
		}
		nameField := head[20 : 20+weightNameMax]
		if nameField[weightNameMax-1] != 0 {
			return nil, fmt.Errorf("rnnoise: unterminated record name at offset %d", off)
		}
		name := string(nameField[:bytes.IndexByte(nameField, 0)])

		payloadOff := off + weightBlockSize
		if int(blockSize) > len(b)-payloadOff {
			return nil, fmt.Errorf("rnnoise: record %q claims %d bytes but only %d remain",
				name, blockSize, len(b)-payloadOff)
		}
		payload := b[payloadOff : payloadOff+int(size)]

		a := &weightArray{typ: int(typ), bytes: int(size)}
		switch typ {
		case weightTypeFloat:
			if size%4 != 0 {
				return nil, fmt.Errorf("rnnoise: float record %q has %d bytes", name, size)
			}
			a.f32 = make([]float32, size/4)
			for i := range a.f32 {
				a.f32[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[4*i:]))
			}
		case weightTypeInt:
			if size%4 != 0 {
				return nil, fmt.Errorf("rnnoise: int record %q has %d bytes", name, size)
			}
			a.i32 = make([]int32, size/4)
			for i := range a.i32 {
				a.i32[i] = int32(binary.LittleEndian.Uint32(payload[4*i:]))
			}
		case weightTypeInt8, weightTypeQWeight:
			a.i8 = make([]int8, size)
			for i, v := range payload {
				a.i8[i] = int8(v)
			}
		default:
			return nil, fmt.Errorf("rnnoise: record %q has unknown type %d", name, typ)
		}
		if _, dup := arrays[name]; dup {
			return nil, fmt.Errorf("rnnoise: duplicate record %q", name)
		}
		arrays[name] = a
		off = payloadOff + int(blockSize)
	}
	if len(arrays) == 0 {
		return nil, errors.New("rnnoise: model blob contains no records")
	}
	return arrays, nil
}

// linearInit mirrors upstream's linear_init, including its size checks. Those
// checks are the only thing standing between a wrong model file and silent
// garbage, so they are kept strict.
func linearInit(l *linearLayer, arrays map[string]*weightArray,
	bias, subias, weights, floatWeights, idx, diag, scale string,
	nbInputs, nbOutputs int) error {

	*l = linearLayer{nbInputs: nbInputs, nbOutputs: nbOutputs}

	getF32 := func(name string, n int) ([]float32, error) {
		a := arrays[name]
		if a == nil {
			return nil, fmt.Errorf("missing array %q", name)
		}
		if a.typ != weightTypeFloat {
			return nil, fmt.Errorf("array %q has type %d, want float", name, a.typ)
		}
		if len(a.f32) != n {
			return nil, fmt.Errorf("array %q has %d floats, want %d", name, len(a.f32), n)
		}
		return a.f32, nil
	}

	var err error
	if bias != "" {
		if l.bias, err = getF32(bias, nbOutputs); err != nil {
			return err
		}
	}
	if subias != "" {
		if l.subias, err = getF32(subias, nbOutputs); err != nil {
			return err
		}
	}

	totalBlocks := 0
	if idx != "" {
		a := arrays[idx]
		if a == nil {
			return fmt.Errorf("missing index array %q", idx)
		}
		if a.typ != weightTypeInt {
			return fmt.Errorf("index array %q has type %d, want int", idx, a.typ)
		}
		if totalBlocks, err = checkIdx(a.i32, nbInputs, nbOutputs); err != nil {
			return fmt.Errorf("index array %q: %w", idx, err)
		}
		l.weightsIdx = a.i32
	}

	wantWeights := nbInputs * nbOutputs
	if idx != "" {
		wantWeights = sparseBlockSize * totalBlocks
	}
	if weights != "" {
		if a := arrays[weights]; a != nil {
			if a.typ != weightTypeInt8 && a.typ != weightTypeQWeight {
				return fmt.Errorf("array %q has type %d, want int8", weights, a.typ)
			}
			if len(a.i8) != wantWeights {
				return fmt.Errorf("array %q has %d int8, want %d", weights, len(a.i8), wantWeights)
			}
			l.weights = a.i8
		}
	}
	// floatWeights is optional: upstream emits it only in a debug build, and
	// compute_linear *prefers* it over the int8 weights when both are present,
	// so a blob carrying both would silently run unquantised. tools/blobgen
	// omits it, matching upstream's -DDISABLE_DEBUG_FLOAT default.
	if floatWeights != "" {
		if a := arrays[floatWeights]; a != nil {
			if a.typ != weightTypeFloat {
				return fmt.Errorf("array %q has type %d, want float", floatWeights, a.typ)
			}
			if len(a.f32) != wantWeights {
				return fmt.Errorf("array %q has %d floats, want %d", floatWeights, len(a.f32), wantWeights)
			}
			l.floatWeights = a.f32
		}
	}
	if l.weights == nil && l.floatWeights == nil {
		return fmt.Errorf("no weights found (looked for %q and %q)", weights, floatWeights)
	}
	if diag != "" {
		if l.diag, err = getF32(diag, nbOutputs); err != nil {
			return err
		}
	}
	if l.weights != nil {
		if l.scale, err = getF32(scale, nbOutputs); err != nil {
			return err
		}
	}

	// A fully dense index visits exactly the same (weight block, activation
	// quad) pairs in the same order as the dense kernel, so dropping it is
	// bit-identical and lets the hot path skip an indirection. The shipped
	// model is dense in all six GRU matrices.
	if l.weightsIdx != nil && idxIsDense(l.weightsIdx, nbInputs) {
		l.weightsIdx = nil
	}
	return nil
}

// checkIdx mirrors upstream's find_idx_check and returns the total block count.
func checkIdx(idx []int32, nbIn, nbOut int) (int, error) {
	total := 0
	remain := len(idx)
	p := 0
	for remain > 0 {
		nbBlocks := int(idx[p])
		p++
		remain--
		if nbBlocks < 0 || remain < nbBlocks {
			return 0, fmt.Errorf("truncated block group")
		}
		for i := 0; i < nbBlocks; i++ {
			pos := int(idx[p])
			p++
			if pos+3 >= nbIn || pos&3 != 0 {
				return 0, fmt.Errorf("column position %d invalid for %d inputs", pos, nbIn)
			}
		}
		remain -= nbBlocks
		nbOut -= 8
		total += nbBlocks
	}
	if nbOut != 0 {
		return 0, fmt.Errorf("index covers the wrong number of output rows (off by %d)", nbOut)
	}
	return total, nil
}

// idxIsDense reports whether every row-group lists all 4-column blocks of the
// input in ascending order, which makes the index a no-op.
func idxIsDense(idx []int32, nbIn int) bool {
	want := nbIn / 4
	p := 0
	for p < len(idx) {
		nb := int(idx[p])
		p++
		if nb != want {
			return false
		}
		for k := 0; k < nb; k++ {
			if int(idx[p+k]) != 4*k {
				return false
			}
		}
		p += nb
	}
	return true
}

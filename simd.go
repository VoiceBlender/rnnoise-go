package rnnoise

// Portable kernels for the neural network's hot loops, ported from upstream's
// src/vec.h (the scalar branch: the one selected when neither __SSE2__ nor
// __ARM_NEON is defined).
//
// These are the portable fallback and the oracle the assembly is fuzz-tested
// against, in the manner of the sibling goamr-nb and goamr-wb modules.
//
// # Why exact equality is achievable
//
// Upstream's scalar int8 kernels accumulate int8 x int8 products into a
// *float32*, not an int32. That is exact here, and exactly equal to integer
// accumulation, because every partial sum is a whole number well inside
// float32's 2^24 exact-integer range: the worst case is 96 column-blocks of
// 4 taps at 127 x 254 = 12,387,072 for the unsigned convention, and half that
// for the signed one.
//
// So an assembly kernel may accumulate in int32 lanes, in any order, and still
// be bit-identical to this code. That is what lets simd_test.go demand exact
// equality rather than pick a tolerance, and it is the property that makes the
// assembly trustworthy.
//
// # Why there is one int8 kernel and not four
//
// Activations are widened to int16 by the quantiser rather than kept as int8.
// Upstream instead keeps them narrow so its AVX2 path can use VPMADDUBSW's
// unsigned-by-signed form, which is why it needs the unsigned
// (USE_SU_BIAS) convention at all. Widening costs one instruction per four
// activations and buys two things: the signed and unsigned conventions become
// the same kernel, differing only in what the quantiser writes, and the int16
// products never saturate the way VPMADDUBSW's do.
//
// The sparse variant stays scalar. The shipped model's index is fully dense, so
// loadModel drops it (see idxIsDense) and the dense kernel handles every layer;
// the sparse path exists only for a hypothetical sparsified user model, where
// correctness matters and speed does not.

// sgemvGeneric is upstream's sgemv: out[i] = sum_j weights[j*colStride+i]*x[j].
//
// Upstream dispatches to sgemv16x1/sgemv8x1 when rows is a multiple of 16 or 8,
// but those differ from this loop only in memory access order -- for a fixed
// output i they still accumulate over j in ascending order -- so one
// implementation reproduces all three bit for bit.
func sgemvGeneric(out, weights []float32, rows, cols, colStride int, x []float32) {
	// vad_dense has one output row, and the general loop below would rebuild a
	// slice header per column for a single multiply-add. Same accumulation
	// order over j, so this stays bit-identical.
	if rows == 1 {
		var acc float32
		for j := 0; j < cols; j++ {
			acc += weights[j*colStride] * x[j]
		}
		out[0] = acc
		return
	}
	for i := 0; i < rows; i++ {
		out[i] = 0
	}
	for j := 0; j < cols; j++ {
		xj := x[j]
		w := weights[j*colStride:]
		for i := 0; i < rows; i++ {
			out[i] += w[i] * xj
		}
	}
}

// sparseSgemvGeneric is upstream's sparse_sgemv8x4, for float weights carrying
// a block index. No layer of the shipped model uses it -- the index only
// appears on int8 layers -- but a user-supplied model could.
func sparseSgemvGeneric(out, w []float32, idx []int32, rows int, x []float32) {
	for i := 0; i < rows; i++ {
		out[i] = 0
	}
	p, wp := 0, 0
	for i := 0; i < rows; i += 8 {
		cols := int(idx[p])
		p++
		for j := 0; j < cols; j++ {
			pos := int(idx[p])
			p++
			y := out[i:]
			for q := 0; q < 4; q++ {
				xq := x[pos+q]
				for k := 0; k < 8; k++ {
					y[k] += w[wp+8*q+k] * xq
				}
			}
			wp += 32
		}
	}
}

// quantizeGeneric is the activation quantisation of upstream's cgemv8x4,
// widened to int16. QuantSigned gives floor(.5 + 127*x) and QuantUnsigned
// gives 127 + floor(.5 + 127*x), matching the two branches of vec.h.
//
// Upstream performs the +.5 in double, because .5 is a double literal and the
// result feeds a C cast to int. This does it in float32, which agrees for every
// |127*x| below 2^22 -- and every activation reaching a quantised layer is a
// tanh output in [-1,1], so 127*x never leaves [-127,127]. Keeping it in
// float32 is what lets the vectorised quantiser produce identical results
// without a float64 path.
func quantizeGeneric(q []int16, in []float32, mode QuantMode) {
	bias := int32(0)
	if mode == QuantUnsigned {
		bias = 127
	}
	for i, v := range in {
		q[i] = int16(bias + int32(floor32(.5+127*v)))
	}
}

// cgemvInt8Generic is upstream's dense cgemv8x4: an 8-row by 4-column blocked
// int8 matrix-vector product, scaled per output row. q holds the activations
// widened to int16 by quantizeGeneric.
func cgemvInt8Generic(out []float32, w []int8, scale []float32, rows, cols int, q []int16) {
	for i := 0; i < rows; i++ {
		out[i] = 0
	}
	wp := 0
	for i := 0; i < rows; i += 8 {
		for j := 0; j < cols; j += 4 {
			x0 := int32(q[j])
			x1 := int32(q[j+1])
			x2 := int32(q[j+2])
			x3 := int32(q[j+3])
			for k := 0; k < 8; k++ {
				acc := int32(w[wp+4*k+0])*x0 + int32(w[wp+4*k+1])*x1 +
					int32(w[wp+4*k+2])*x2 + int32(w[wp+4*k+3])*x3
				out[i+k] += float32(acc)
			}
			wp += 32
		}
	}
	for i := 0; i < rows; i++ {
		out[i] *= scale[i]
	}
}

// sparseCgemvInt8Generic is upstream's sparse_cgemv8x4: the same kernel driven
// by a block index.
//
// For a fully dense index this visits the same (weight block, activation quad)
// pairs in the same order as cgemvInt8Generic and so produces bit-identical
// results -- which is why loadModel can drop such an index.
func sparseCgemvInt8Generic(out []float32, w []int8, idx []int32, scale []float32, rows, cols int, q []int16) {
	for i := 0; i < rows; i++ {
		out[i] = 0
	}
	p, wp := 0, 0
	for i := 0; i < rows; i += 8 {
		colblocks := int(idx[p])
		p++
		for j := 0; j < colblocks; j++ {
			pos := int(idx[p])
			p++
			x0 := int32(q[pos])
			x1 := int32(q[pos+1])
			x2 := int32(q[pos+2])
			x3 := int32(q[pos+3])
			for k := 0; k < 8; k++ {
				acc := int32(w[wp+4*k+0])*x0 + int32(w[wp+4*k+1])*x1 +
					int32(w[wp+4*k+2])*x2 + int32(w[wp+4*k+3])*x3
				out[i+k] += float32(acc)
			}
			wp += 32
		}
	}
	for i := 0; i < rows; i++ {
		out[i] *= scale[i]
	}
}

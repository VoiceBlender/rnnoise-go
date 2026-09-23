package rnnoise

import (
	"math/rand"
	"testing"
)

// Differential tests for the assembly kernels, in the manner of
// goamr-nb/simd_test.go: every kernel is checked against its generic
// counterpart in simd.go with EXACT equality, not a tolerance.
//
// Exactness is available because the int8 accumulation is over whole numbers
// bounded well inside float32's 2^24 exact-integer range, so lane order cannot
// change the result. See the comment at the top of simd.go. If a kernel ever
// needs a tolerance here, the premise has broken and the kernel is wrong.

// randWeights fills an 8x4-blocked int8 weight matrix.
func randWeights(rng *rand.Rand, rows, cols int, extreme bool) []int8 {
	w := make([]int8, rows*cols)
	for i := range w {
		if extreme {
			// Worst case for the accumulator bound: every weight saturated.
			if rng.Intn(2) == 0 {
				w[i] = 127
			} else {
				w[i] = -128
			}
		} else {
			w[i] = int8(rng.Intn(256) - 128)
		}
	}
	return w
}

func randActivations(rng *rand.Rand, n int, mode QuantMode, extreme bool) ([]float32, []int16) {
	in := make([]float32, n)
	for i := range in {
		if extreme {
			// Saturated activations, the other half of the worst case.
			if rng.Intn(2) == 0 {
				in[i] = 1
			} else {
				in[i] = -1
			}
		} else {
			in[i] = float32(rng.Float64()*2 - 1)
		}
	}
	q := make([]int16, n)
	quantizeGeneric(q, in, mode)
	return in, q
}

// TestCgemvInt8MatchesGeneric fuzzes the vectorised int8 GEMV against the
// portable one over every shape the model uses plus the tail and boundary cases,
// in both quantisation conventions, including the saturated-magnitude worst case
// that stresses the accumulator bound.
func TestCgemvInt8MatchesGeneric(t *testing.T) {
	shapes := []struct{ rows, cols int }{
		{384, 384},   // conv2
		{1152, 384},  // every GRU gate matrix
		{8, 4},       // one block: the smallest legal shape
		{8, 8},       // exercises the column loop with no unroll
		{16, 12},     // more rows than one block, non-unrollable columns
		{24, 20},     // unroll of 4 plus a one-block tail
		{32, 36},     // two unrolls plus a tail
		{8, 16},      // exactly one unroll, no tail
		{40, 4},      // many row-blocks, single column block
		{1152, 1536}, // wider than anything in the model
	}
	rng := rand.New(rand.NewSource(0xC0FFEE))

	for _, mode := range []QuantMode{QuantSigned, QuantUnsigned} {
		for _, sh := range shapes {
			for _, extreme := range []bool{false, true} {
				w := randWeights(rng, sh.rows, sh.cols, extreme)
				_, q := randActivations(rng, sh.cols, mode, extreme)
				scale := make([]float32, sh.rows)
				for i := range scale {
					scale[i] = float32(rng.Float64()*0.02 + 1e-4)
				}

				want := make([]float32, sh.rows)
				got := make([]float32, sh.rows)
				acc := make([]int32, sh.rows)
				q8 := make([]int8, sh.cols)
				cgemvInt8Generic(want, w, scale, sh.rows, sh.cols, q)
				cgemvInt8(got, w, scale, sh.rows, sh.cols, q, acc, q8)

				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("mode=%s rows=%d cols=%d extreme=%v: out[%d] = %v, generic %v",
							mode, sh.rows, sh.cols, extreme, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestCgemvInt8AccumulatorBound is the guard on the premise that makes exact
// equality possible. It drives the worst case the model can produce -- every
// weight and every activation saturated -- and checks the exact integer result
// is still inside float32's exactly-representable integer range.
//
// If a future model widened a quantised layer past this bound, the float32
// running sum in the generic kernel would start rounding while an int32
// accumulator would not, and the two would diverge. This test says so directly
// instead of leaving it to be discovered as a mysterious inexactness.
func TestCgemvInt8AccumulatorBound(t *testing.T) {
	const float32ExactInt = 1 << 24

	for _, mode := range []QuantMode{QuantSigned, QuantUnsigned} {
		// The largest activation magnitude the quantiser can emit.
		in := []float32{1, -1}
		q := make([]int16, 2)
		quantizeGeneric(q, in, mode)
		maxAct := int32(0)
		for _, v := range q {
			if a := int32(v); a > maxAct {
				maxAct = a
			} else if -a > maxAct {
				maxAct = -a
			}
		}
		// The widest quantised layer in the model: 384 inputs.
		worst := int64(maxAct) * 128 * int64(gruSize)
		if worst >= float32ExactInt {
			t.Errorf("mode=%s: worst-case accumulator %d reaches float32's exact-integer "+
				"limit %d; the generic kernel's float32 sum would round and the assembly's "+
				"int32 sum would not", mode, worst, float32ExactInt)
		}
		t.Logf("mode=%-8s max |activation| = %3d, worst-case accumulator = %8d (%.1f%% of 2^24)",
			mode, maxAct, worst, 100*float64(worst)/float32ExactInt)
	}
}

// TestQuantizeRange checks the quantiser against the domain its float32
// arithmetic is valid over, and against upstream's formula.
func TestQuantizeRange(t *testing.T) {
	cases := []struct {
		in               float32
		signed, unsigned int16
	}{
		{0, 0, 127},
		{1, 127, 254},
		{-1, -127, 0},
		{0.5, 64, 191},  // floor(.5 + 63.5) = 64
		{-0.5, -63, 64}, // floor(.5 - 63.5) = -63
		{1e-9, 0, 127},
		{-1e-9, 0, 127},
	}
	for _, tc := range cases {
		for _, m := range []QuantMode{QuantSigned, QuantUnsigned} {
			q := make([]int16, 1)
			quantize(q, []float32{tc.in}, m)
			want := tc.signed
			if m == QuantUnsigned {
				want = tc.unsigned
			}
			if q[0] != want {
				t.Errorf("quantize(%v, %s) = %d, want %d", tc.in, m, q[0], want)
			}
		}
	}
}

// TestSparseDenseEquivalence checks the claim loadModel relies on: a fully
// dense block index visits the same (weight block, activation quad) pairs in the
// same order as the dense kernel, so dropping the index is bit-identical rather
// than merely equivalent.
func TestSparseDenseEquivalence(t *testing.T) {
	const rows, cols = 1152, 384
	rng := rand.New(rand.NewSource(7))
	w := randWeights(rng, rows, cols, false)
	_, q := randActivations(rng, cols, QuantSigned, false)
	scale := make([]float32, rows)
	for i := range scale {
		scale[i] = float32(rng.Float64())
	}

	// Build the dense index upstream would emit: per 8-row group, all cols/4
	// column blocks in ascending order.
	var idx []int32
	for i := 0; i < rows; i += 8 {
		idx = append(idx, int32(cols/4))
		for c := 0; c < cols; c += 4 {
			idx = append(idx, int32(c))
		}
	}
	if !idxIsDense(idx, cols) {
		t.Fatal("constructed index should be recognised as dense")
	}

	dense := make([]float32, rows)
	sparse := make([]float32, rows)
	cgemvInt8Generic(dense, w, scale, rows, cols, q)
	sparseCgemvInt8Generic(sparse, w, idx, scale, rows, cols, q)
	for i := range dense {
		if dense[i] != sparse[i] {
			t.Fatalf("out[%d]: dense %v, sparse-with-dense-index %v", i, dense[i], sparse[i])
		}
	}
}

func BenchmarkCgemvInt8(b *testing.B) {
	const rows, cols = 1152, 384 // one GRU gate matrix
	rng := rand.New(rand.NewSource(1))
	w := randWeights(rng, rows, cols, false)
	_, q := randActivations(rng, cols, QuantSigned, false)
	scale := make([]float32, rows)
	out := make([]float32, rows)

	b.Run("generic", func(b *testing.B) {
		b.SetBytes(int64(rows * cols))
		for i := 0; i < b.N; i++ {
			cgemvInt8Generic(out, w, scale, rows, cols, q)
		}
	})
	b.Run("dispatch", func(b *testing.B) {
		acc := make([]int32, rows)
		q8 := make([]int8, cols)
		b.SetBytes(int64(rows * cols))
		for i := 0; i < b.N; i++ {
			cgemvInt8(out, w, scale, rows, cols, q, acc, q8)
		}
	})
}

// TestSgemvMatchesGeneric fuzzes the vectorised float GEMV against the portable
// one with exact equality. Exactness holds because the kernel vectorises across
// the output index, leaving each output's accumulation order over j untouched,
// and because it uses unfused multiply-add -- see the note in simd_amd64.s.
func TestSgemvMatchesGeneric(t *testing.T) {
	shapes := []struct{ rows, cols int }{
		{128, 195}, // conv1
		{32, 1536}, // dense_out
		{8, 1},     // smallest vectorised shape
		{8, 1536},  // one accumulator, long inner loop
		{24, 7},    // three 8-row blocks, odd column count
		{32, 3},    // exactly one 32-row block
		{40, 11},   // one 32-row block plus an 8-row remainder
		{64, 5},    // two 32-row blocks
		{1, 1536},  // vad_dense: falls back to generic
		{7, 9},     // not a multiple of 8: falls back
		{256, 65},  // wider than anything in the model
	}
	rng := rand.New(rand.NewSource(0x5EED))
	for _, sh := range shapes {
		// colStride is the output row count in every call the model makes,
		// but exercise a padded stride too.
		for _, stride := range []int{sh.rows, sh.rows + 3} {
			w := make([]float32, sh.cols*stride)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			x := make([]float32, sh.cols)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			want := make([]float32, sh.rows)
			got := make([]float32, sh.rows)
			sgemvGeneric(want, w, sh.rows, sh.cols, stride, x)
			sgemv(got, w, sh.rows, sh.cols, stride, x)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("rows=%d cols=%d stride=%d: out[%d] = %v, generic %v",
						sh.rows, sh.cols, stride, i, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkSgemv(b *testing.B) {
	const rows, cols = 32, 1536 // dense_out
	rng := rand.New(rand.NewSource(2))
	w := make([]float32, cols*rows)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	x := make([]float32, cols)
	out := make([]float32, rows)
	b.Run("generic", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sgemvGeneric(out, w, rows, cols, rows, x)
		}
	})
	b.Run("dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sgemv(out, w, rows, cols, rows, x)
		}
	})
}

// TestVecTanhMatchesGeneric and TestVecSigmoidMatchesGeneric fuzz the
// activation kernels against the scalar ones with exact equality, over the
// reachable input domain.
//
// "Reachable" is load-bearing. For |x| large enough that x*x overflows to +Inf,
// the scalar path yields NaN while MINPS returns its second operand and the
// vector path yields 1.0. Every value reaching these functions is a bounded
// pre-activation of a layer whose weights and inputs are both bounded, so the
// case cannot occur; TestFeaturesFinite guards the pipeline against NaN
// independently. The domain here spans well past anything the network produces.
func TestVecTanhMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(1205))
	for _, n := range []int{1, 7, 8, 9, 15, 16, 31, 33, 128, 384, 768, 1536} {
		x := make([]float32, n)
		for i := range x {
			switch i % 4 {
			case 0:
				x[i] = float32(rng.NormFloat64() * 4)
			case 1:
				x[i] = float32(rng.NormFloat64() * 0.01)
			case 2:
				x[i] = float32(rng.NormFloat64() * 1000)
			default:
				x[i] = float32(rng.NormFloat64())
			}
		}
		// Include the exact boundaries and both zeros.
		if n >= 4 {
			x[0], x[1], x[2], x[3] = 0, -0, 1e6, -1e6
		}
		for _, fn := range []struct {
			name     string
			vec, gen func([]float32, []float32)
		}{
			{"tanh", vecTanh, vecTanhGeneric},
			{"sigmoid", vecSigmoid, vecSigmoidGeneric},
		} {
			want := make([]float32, n)
			got := make([]float32, n)
			fn.gen(want, x)
			fn.vec(got, x)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s n=%d: y[%d] = %v, generic %v (x = %v)",
						fn.name, n, i, got[i], want[i], x[i])
				}
			}
		}
	}
}

// TestQuantizeMatchesGeneric fuzzes the vectorised quantiser, including the
// 16-lane tail.
func TestQuantizeMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9A))
	for _, mode := range []QuantMode{QuantSigned, QuantUnsigned} {
		for _, n := range []int{1, 15, 16, 17, 31, 32, 33, 128, 384, 1536} {
			in := make([]float32, n)
			for i := range in {
				switch i % 5 {
				case 0:
					in[i] = 1
				case 1:
					in[i] = -1
				case 2:
					in[i] = 0
				case 3:
					in[i] = float32(rng.Float64()*2 - 1)
				default:
					// Exercise the half-way rounding boundary.
					in[i] = float32(rng.Intn(255)-127) / 127 / 2
				}
			}
			want := make([]int16, n)
			got := make([]int16, n)
			quantizeGeneric(want, in, mode)
			quantize(got, in, mode)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("mode=%s n=%d: q[%d] = %d, generic %d (in = %v)",
						mode, n, i, got[i], want[i], in[i])
				}
			}
		}
	}
}

package rnnoise

import (
	"math"
	"math/rand"
	"testing"
)

// The vectorised radix-5 butterfly must equal the scalar one bit for bit: it
// performs the same multiplies and adds on the same operands, in the same
// order, with no fused multiply-add.
func TestBfly5VecBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	ran := 0
	for nfft := 160; nfft <= 1920; nfft += 20 {
		st, ok := newFFTState(nfft)
		if !ok || st.b5m == 0 {
			continue // no radix-5 stage in this factorisation
		}
		n, mm := nfft/(5*st.b5m), 5*st.b5m

		vec := make([]cpx, nfft)
		for i := range vec {
			vec[i] = cpx{float32(rng.NormFloat64()), float32(rng.NormFloat64())}
		}
		scalar := append([]cpx(nil), vec...)

		kfBfly5(vec, st.b5fstride, st, st.b5m, n, mm)
		saved := st.b5tw
		st.b5tw = [4][]cpx{} // force the scalar path
		kfBfly5(scalar, st.b5fstride, st, st.b5m, n, mm)
		st.b5tw = saved

		for i := range vec {
			if vec[i] != scalar[i] {
				t.Fatalf("nfft=%d index %d: vector %v, scalar %v", nfft, i, vec[i], scalar[i])
			}
		}
		ran++
	}
	if ran == 0 {
		t.Fatal("no transform size exercised the radix-5 stage")
	}
	t.Logf("bit-identical across %d transform sizes", ran)
}

// Silence drives every intermediate to zero, where a - b and -(b - a) differ in
// sign bit. The kernel blends rather than negates for exactly this reason.
func TestBfly5VecSignedZero(t *testing.T) {
	st, ok := newFFTState(960)
	if !ok || st.b5m == 0 {
		t.Skip("no radix-5 stage")
	}
	n, mm := 960/(5*st.b5m), 5*st.b5m
	for _, fill := range []cpx{{0, 0}, {-0, -0}, {0, -0}, {-0, 0}} {
		vec := make([]cpx, 960)
		for i := range vec {
			vec[i] = fill
		}
		scalar := append([]cpx(nil), vec...)

		kfBfly5(vec, st.b5fstride, st, st.b5m, n, mm)
		saved := st.b5tw
		st.b5tw = [4][]cpx{}
		kfBfly5(scalar, st.b5fstride, st, st.b5m, n, mm)
		st.b5tw = saved

		for i := range vec {
			if vec[i] != scalar[i] || signbits(vec[i]) != signbits(scalar[i]) {
				t.Fatalf("fill %v index %d: vector %v, scalar %v", fill, i, vec[i], scalar[i])
			}
		}
	}
}

func signbits(c cpx) [2]bool {
	return [2]bool{math.Signbit(float64(c.r)), math.Signbit(float64(c.i))}
}

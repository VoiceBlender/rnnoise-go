// Command avogen generates ../../simd_amd64.s.
//
// Only the dense int8 GEMV lives here. It is 2.80 M of the 2.88 M
// multiply-accumulates in a frame, and it is the one kernel where hand-written
// Plan 9 assembly is genuinely error-prone: eight row accumulators, a
// configurable column unroll, and a cross-lane reduction at the end. Every
// other kernel in the package is small enough to hand-write in the style of the
// sibling goamr-nb and goamr-wb modules, and avo stays out of the library's
// dependency graph by living in this nested module.
//
//	go run ./tools/avogen -out simd_amd64.s   (see `make gen-asm`)
package main

import (
	"github.com/mmcloughlin/avo/attr"
	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

// colUnroll is how many 4-wide column blocks the inner loop consumes per
// iteration. Each block is 7 instructions for 32 multiply-accumulates, so the
// loop is front-end bound rather than multiplier bound; unrolling amortises the
// pointer bump and the branch, and gives the out-of-order engine more
// independent loads to hide L3 latency behind.
const colUnroll = 4

func main() {
	Package("github.com/VoiceBlender/rnnoise-go")
	ConstraintExpr("amd64")
	genCgemvInt8AVX2()
	genKfBfly5AVX2()
	Generate()
}

// permIdx reorders the result of the final VPHADDD back into row order. See the
// derivation in genCgemvInt8AVX2.
var permIdx = GLOBL("cgemvPermIdx", attr.RODATA|attr.NOPTR)

func init() {
	for i, v := range []int{0, 1, 4, 5, 2, 3, 6, 7} {
		DATA(4*i, operand.U32(uint32(v)))
	}
}

func genCgemvInt8AVX2() {
	TEXT("cgemvInt8AVX2", NOSPLIT,
		"func(out []float32, w []int8, scale []float32, q []int16, rows int, cols int)")

	Doc(
		"cgemvInt8AVX2 computes out[i] = scale[i] * sum_j w[i,j]*q[j] for an",
		"8-row by 4-column blocked int8 weight matrix, with the activations q",
		"already quantised and widened to int16.",
		"",
		"The accumulation is exact. Every partial sum is a whole number bounded",
		"by 96 blocks * 4 taps * 127 * 254 = 12,387,072, comfortably inside both",
		"int32 and float32's exact-integer range, so accumulating in int32 lanes",
		"in this order gives bit-identical results to cgemvInt8Generic's float32",
		"running sum. simd_test.go asserts exactly that.",
		"",
		"Requires rows % 8 == 0 and cols % 4 == 0.",
	)

	outPtr := Load(Param("out").Base(), GP64())
	wPtr := Load(Param("w").Base(), GP64())
	scalePtr := Load(Param("scale").Base(), GP64())
	qPtr := Load(Param("q").Base(), GP64())
	rows := Load(Param("rows"), GP64())
	cols := Load(Param("cols"), GP64())

	perm := YMM()
	VMOVDQU(permIdx.Offset(0), perm)

	// colBytes is the weight stride for one row-block: cols/4 blocks of 32 B.
	colBytes := GP64()
	MOVQ(cols, colBytes)
	SHLQ(operand.U8(3), colBytes) // cols/4 * 32 == cols * 8

	i := GP64()
	XORQ(i, i)

	Label("rowblock")
	CMPQ(i, rows)
	JGE(operand.LabelRef("done"))

	// Two accumulators, each holding two partial sums per row:
	//   acc0 = [r0a r0b r1a r1b r2a r2b r3a r3b]   (rows i+0..i+3)
	//   acc1 = [r4a r4b r5a r5b r6a r6b r7a r7b]   (rows i+4..i+7)
	// where 'a' covers columns 0,1 of each 4-wide block and 'b' columns 2,3.
	// Splitting each row's sum this way is free: integer addition is
	// associative, so the halves can be combined once at the end.
	acc0, acc1 := YMM(), YMM()
	VPXOR(acc0, acc0, acc0)
	VPXOR(acc1, acc1, acc1)

	// wRow walks the weight blocks for this row-block; qCur walks activations.
	wRow := GP64()
	MOVQ(wPtr, wRow)
	qCur := GP64()
	MOVQ(qPtr, qCur)

	j := GP64()
	XORQ(j, j)

	emitBlock := func(k int) {
		wlo, whi, xb := YMM(), YMM(), YMM()
		// One VPMOVSXBW loads 16 bytes and sign-extends to 16 int16, so a
		// 32-byte block needs no separate load: rows 0-3 of the block come
		// from the low half, rows 4-7 from the high half.
		VPMOVSXBW(operand.Mem{Base: wRow, Disp: 32 * k}, wlo)
		VPMOVSXBW(operand.Mem{Base: wRow, Disp: 32*k + 16}, whi)
		// The four activations of this block, replicated across all four
		// 64-bit halves: [x0 x1 x2 x3] x4 as 16 int16.
		VPBROADCASTQ(operand.Mem{Base: qCur, Disp: 8 * k}, xb)
		// VPMADDWD multiplies adjacent int16 pairs and sums them, so each
		// int32 lane lands exactly one (row, column-pair) partial sum.
		p0, p1 := YMM(), YMM()
		VPMADDWD(wlo, xb, p0)
		VPMADDWD(whi, xb, p1)
		VPADDD(p0, acc0, acc0)
		VPADDD(p1, acc1, acc1)
	}

	// Unrolled body, entered only while a full unroll remains.
	unrollCols := GP64()
	MOVQ(cols, unrollCols)
	SUBQ(operand.Imm(4*colUnroll-1), unrollCols)

	Label("colblock_unrolled")
	CMPQ(j, unrollCols)
	JGE(operand.LabelRef("colblock_tail"))
	for k := 0; k < colUnroll; k++ {
		emitBlock(k)
	}
	ADDQ(operand.Imm(32*colUnroll), wRow)
	ADDQ(operand.Imm(8*colUnroll), qCur)
	ADDQ(operand.Imm(4*colUnroll), j)
	JMP(operand.LabelRef("colblock_unrolled"))

	// Remainder, one block at a time.
	Label("colblock_tail")
	CMPQ(j, cols)
	JGE(operand.LabelRef("reduce"))
	emitBlock(0)
	ADDQ(operand.Imm(32), wRow)
	ADDQ(operand.Imm(8), qCur)
	ADDQ(operand.Imm(4), j)
	JMP(operand.LabelRef("colblock_tail"))

	Label("reduce")
	// VPHADDD folds adjacent int32 pairs within each 128-bit lane, taking the
	// low lane from src1 and src2's low halves and the high lane from their
	// high halves. With src1 = acc0 and src2 = acc1 that yields
	//   [r0 r1 r4 r5 r2 r3 r6 r7]
	// which cgemvPermIdx = [0 1 4 5 2 3 6 7] restores to row order.
	sum := YMM()
	VPHADDD(acc1, acc0, sum)
	ordered := YMM()
	VPERMD(sum, perm, ordered)

	// Scale and store the eight rows.
	fl, sc := YMM(), YMM()
	VCVTDQ2PS(ordered, fl)
	VMOVUPS(operand.Mem{Base: scalePtr}, sc)
	VMULPS(sc, fl, fl)
	VMOVUPS(fl, operand.Mem{Base: outPtr})

	// Advance to the next row-block: 8 rows of output and scale, and one
	// row-block's worth of weights.
	ADDQ(operand.Imm(32), outPtr)
	ADDQ(operand.Imm(32), scalePtr)
	ADDQ(colBytes, wPtr)
	ADDQ(operand.Imm(8), i)
	JMP(operand.LabelRef("rowblock"))

	Label("done")
	VZEROUPPER()
	RET()
}

// keep the reg import used even if a future edit drops an explicit register.
var _ = reg.RAX

// cmulAVX2 is cmul on four interleaved complex values at once.
//
// It computes exactly a.r*b.r - a.i*b.i and a.r*b.i + a.i*b.r, with separate
// multiplies and no fused multiply-add, so it is bit-identical to cmul. Float
// addition is commutative, so the imaginary lane's operand order is immaterial.
func cmulAVX2(a, b reg.VecVirtual) reg.VecVirtual {
	br, bi, as := YMM(), YMM(), YMM()
	VMOVSLDUP(b, br) // [b.r b.r ...]
	VMOVSHDUP(b, bi) // [b.i b.i ...]
	VPERMILPS(operand.U8(0xB1), a, as)
	p, q, out := YMM(), YMM(), YMM()
	VMULPS(br, a, p)  // [a.r*b.r  a.i*b.r]
	VMULPS(bi, as, q) // [a.i*b.i  a.r*b.i]
	VADDSUBPS(q, p, out)
	return out
}

func genKfBfly5AVX2() {
	// Declared here rather than in an init: DATA applies to the most recently
	// declared GLOBL, so two package-level globals would interleave.
	//
	// signOdd flips the sign of the imaginary lanes, for the places where the
	// scalar code negates a whole expression.
	signOdd := GLOBL("bfly5SignOdd", attr.RODATA|attr.NOPTR)
	for i := 0; i < 8; i++ {
		v := uint32(0)
		if i%2 == 1 {
			v = 0x80000000
		}
		DATA(4*i, operand.U32(v))
	}

	TEXT("kfBfly5AVX2", NOSPLIT,
		"func(f0, f1, f2, f3, f4, t1, t2, t3, t4 []cpx, y *[4]float32, blocks int)")

	Doc(
		"kfBfly5AVX2 runs the radix-5 butterfly over blocks*4 consecutive u, on",
		"four interleaved complex values at a time.",
		"",
		"It is bit-identical to kfBfly5: every multiply and add is the same",
		"operation on the same pair of operands, in the same order, with no",
		"fused multiply-add. The twiddles must already be gathered into the",
		"dense t1..t4 arrays, since the scalar code reads them at strides of",
		"1, 2, 3 and 4.",
	)

	p0, p1, p2, p3, p4 := GP64(), GP64(), GP64(), GP64(), GP64()
	q1, q2, q3, q4 := GP64(), GP64(), GP64(), GP64()
	Load(Param("f0").Base(), p0)
	Load(Param("f1").Base(), p1)
	Load(Param("f2").Base(), p2)
	Load(Param("f3").Base(), p3)
	Load(Param("f4").Base(), p4)
	Load(Param("t1").Base(), q1)
	Load(Param("t2").Base(), q2)
	Load(Param("t3").Base(), q3)
	Load(Param("t4").Base(), q4)

	yp := GP64()
	Load(Param("y"), yp)
	yar, yai, ybr, ybi := YMM(), YMM(), YMM(), YMM()
	VBROADCASTSS(operand.Mem{Base: yp, Disp: 0}, yar)
	VBROADCASTSS(operand.Mem{Base: yp, Disp: 4}, yai)
	VBROADCASTSS(operand.Mem{Base: yp, Disp: 8}, ybr)
	VBROADCASTSS(operand.Mem{Base: yp, Disp: 12}, ybi)

	n := Load(Param("blocks"), GP64())
	TESTQ(n, n)
	JZ(operand.LabelRef("done"))

	Label("loop")
	a0, a1, a2, a3, a4 := YMM(), YMM(), YMM(), YMM(), YMM()
	VMOVUPS(operand.Mem{Base: p0}, a0)
	VMOVUPS(operand.Mem{Base: p1}, a1)
	VMOVUPS(operand.Mem{Base: p2}, a2)
	VMOVUPS(operand.Mem{Base: p3}, a3)
	VMOVUPS(operand.Mem{Base: p4}, a4)
	tv1, tv2, tv3, tv4 := YMM(), YMM(), YMM(), YMM()
	VMOVUPS(operand.Mem{Base: q1}, tv1)
	VMOVUPS(operand.Mem{Base: q2}, tv2)
	VMOVUPS(operand.Mem{Base: q3}, tv3)
	VMOVUPS(operand.Mem{Base: q4}, tv4)

	s1 := cmulAVX2(a1, tv1)
	s2 := cmulAVX2(a2, tv2)
	s3 := cmulAVX2(a3, tv3)
	s4 := cmulAVX2(a4, tv4)

	s7, s10, s8, s9 := YMM(), YMM(), YMM(), YMM()
	VADDPS(s4, s1, s7)
	VSUBPS(s4, s1, s10)
	VADDPS(s3, s2, s8)
	VSUBPS(s3, s2, s9)

	// fout[f0] = fout[f0] + (s7 + s8)
	t, o0 := YMM(), YMM()
	VADDPS(s8, s7, t)
	VADDPS(t, a0, o0)
	VMOVUPS(o0, operand.Mem{Base: p0})

	// s5 = s0 + (s7*ya.r + s8*yb.r); both components use the same scalars.
	pa, pb, s5 := YMM(), YMM(), YMM()
	VMULPS(yar, s7, pa)
	VMULPS(ybr, s8, pb)
	VADDPS(pb, pa, s5)
	VADDPS(s5, a0, s5)

	// s6 swaps components: s6.r uses the imaginary parts, s6.i the real ones,
	// and the scalar code negates the whole imaginary expression.
	s10s, s9s := YMM(), YMM()
	VPERMILPS(operand.U8(0xB1), s10, s10s)
	VPERMILPS(operand.U8(0xB1), s9, s9s)
	s6 := YMM()
	VMULPS(yai, s10s, pa)
	VMULPS(ybi, s9s, pb)
	VADDPS(pb, pa, s6)
	VXORPS(signOdd, s6, s6)

	o1, o4 := YMM(), YMM()
	VSUBPS(s6, s5, o1)
	VADDPS(s6, s5, o4)
	VMOVUPS(o1, operand.Mem{Base: p1})
	VMOVUPS(o4, operand.Mem{Base: p4})

	// s11 = s0 + (s7*yb.r + s8*ya.r)
	s11 := YMM()
	VMULPS(ybr, s7, pa)
	VMULPS(yar, s8, pb)
	VADDPS(pb, pa, s11)
	VADDPS(s11, a0, s11)

	// s12.r = s9.i*ya.i - s10.i*yb.i, s12.i = s10.r*yb.i - s9.r*ya.i. The two
	// lanes subtract in opposite orders, so blend rather than negate: a-b and
	// -(b-a) differ for signed zero, which silence produces.
	e, od, s12 := YMM(), YMM(), YMM()
	VMULPS(yai, s9s, pa)
	VMULPS(ybi, s10s, pb)
	VSUBPS(pb, pa, e)  // s9s*ya.i - s10s*yb.i
	VSUBPS(pa, pb, od) // s10s*yb.i - s9s*ya.i
	VBLENDPS(operand.U8(0xAA), od, e, s12)

	o2, o3 := YMM(), YMM()
	VADDPS(s12, s11, o2)
	VSUBPS(s12, s11, o3)
	VMOVUPS(o2, operand.Mem{Base: p2})
	VMOVUPS(o3, operand.Mem{Base: p3})

	for _, r := range []reg.GPVirtual{p0, p1, p2, p3, p4, q1, q2, q3, q4} {
		ADDQ(operand.Imm(32), r)
	}
	DECQ(n)
	JNZ(operand.LabelRef("loop"))

	Label("done")
	VZEROUPPER()
	RET()
}

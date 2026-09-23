// Hand-written AVX2 kernels, in the style of the sibling goamr-nb and
// goamr-wb modules. The generated int8 GEMV lives in simd_gemv_amd64.s.
//
// NOTE ON FMA: these kernels deliberately use separate VMULPS and VADDPS rather
// than VFMADD*. Upstream's reference computes an unfused `y[k] += w[k]*xj`, so
// fusing here would round differently and forfeit bit-exactness against the C
// scalar build -- the property simd_test.go asserts and cdiff_test.go depends
// on. The multiply-add pairs issue on separate ports anyway, so on Zen 4 the
// cost is small compared to what is gained.

#include "textflag.h"

// func sgemvAVX2(out []float32, weights []float32, rows int, cols int, colStride int, x []float32)
//
// out[i] = sum_j weights[j*colStride + i] * x[j], for rows a multiple of 8.
//
// Vectorising across i (the output index) rather than across j is what keeps
// this bit-exact: for any single output, the accumulation over j still runs in
// ascending order with the same rounding at every step, exactly as
// sgemvGeneric and upstream's sgemv16x1 do. Vectorising across j instead would
// reassociate the sum and change the result.
//
// Registers: AX out, CX weights, DX x, SI rows, DI cols, R8 colStride in bytes,
// R9 i, R10 j, R11 weight cursor, R12 scratch, R13 activation cursor.
// R14 (goroutine) and R15 (GOT) are left alone.
TEXT ·sgemvAVX2(SB), NOSPLIT, $0-96
	MOVQ out_base+0(FP), AX
	MOVQ weights_base+24(FP), CX
	MOVQ rows+48(FP), SI
	MOVQ cols+56(FP), DI
	MOVQ colStride+64(FP), R8
	MOVQ x_base+72(FP), DX

	SHLQ $2, R8  // colStride: floats -> bytes
	XORQ R9, R9  // i = 0

	// Main block: 32 rows at a time, held in four accumulators across the
	// whole j loop, so the weights are streamed once per 32 rows and the
	// activations are broadcast once per j.
rowblock32:
	MOVQ SI, R12
	SUBQ $32, R12
	CMPQ R9, R12
	JG   rowblock8

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	MOVQ CX, R11
	LEAQ (R11)(R9*4), R11  // &weights[i]
	MOVQ DX, R13           // &x[0]
	XORQ R10, R10          // j = 0

col32:
	CMPQ R10, DI
	JGE  store32

	VBROADCASTSS (R13), Y4

	VMULPS 0(R11), Y4, Y5
	VMULPS 32(R11), Y4, Y6
	VMULPS 64(R11), Y4, Y7
	VMULPS 96(R11), Y4, Y8

	VADDPS Y5, Y0, Y0
	VADDPS Y6, Y1, Y1
	VADDPS Y7, Y2, Y2
	VADDPS Y8, Y3, Y3

	ADDQ $4, R13
	ADDQ R8, R11
	INCQ R10
	JMP  col32

store32:
	MOVQ    AX, R12
	LEAQ    (R12)(R9*4), R12
	VMOVUPS Y0, 0(R12)
	VMOVUPS Y1, 32(R12)
	VMOVUPS Y2, 64(R12)
	VMOVUPS Y3, 96(R12)
	ADDQ    $32, R9
	JMP     rowblock32

	// Remainder: 8 rows at a time.
rowblock8:
	CMPQ R9, SI
	JGE  done

	VXORPS Y0, Y0, Y0
	MOVQ   CX, R11
	LEAQ   (R11)(R9*4), R11
	MOVQ   DX, R13
	XORQ   R10, R10

col8:
	CMPQ         R10, DI
	JGE          store8
	VBROADCASTSS (R13), Y4
	VMULPS       0(R11), Y4, Y5
	VADDPS       Y5, Y0, Y0
	ADDQ         $4, R13
	ADDQ         R8, R11
	INCQ         R10
	JMP          col8

store8:
	MOVQ    AX, R12
	LEAQ    (R12)(R9*4), R12
	VMOVUPS Y0, 0(R12)
	ADDQ    $8, R9
	JMP     rowblock8

done:
	VZEROUPPER
	RET

// func vecTanhAVX2(y []float32, x []float32, coef *[8]float32)
//
// Upstream's rational tanh, eight lanes at a time. The coefficients come in as
// a pointer rather than as assembler literals so the vector and scalar paths
// provably share the same float32 values (see tanhCoefs).
//
// VDIVPS, not VRCPPS: upstream's AVX path approximates the denominator with
// _mm256_rcp_ps, whose result is microarchitecture-defined, so matching it is
// impossible. A real divide matches upstream's scalar build exactly and is more
// accurate besides.
//
// One documented divergence, on unreachable input: for x large enough that x*x
// overflows to +Inf, the generic code yields NaN while MINPS returns its second
// operand and this yields 1.0. Every value reaching tanh here is a bounded
// pre-activation, so the case cannot arise; TestVecTanhMatchesGeneric fuzzes the
// reachable domain.
TEXT ·vecTanhAVX2(SB), NOSPLIT, $0-56
	MOVQ y_base+0(FP), AX
	MOVQ x_base+24(FP), CX
	MOVQ x_len+32(FP), DX
	MOVQ coef+48(FP), BX

	VBROADCASTSS 0(BX), Y8   // n0
	VBROADCASTSS 4(BX), Y9   // n1
	VBROADCASTSS 8(BX), Y10  // n2
	VBROADCASTSS 12(BX), Y11 // d0
	VBROADCASTSS 16(BX), Y12 // d1
	VBROADCASTSS 20(BX), Y13 // d2
	VBROADCASTSS 24(BX), Y14 // +1
	VBROADCASTSS 28(BX), Y15 // -1

	XORQ SI, SI
	MOVQ DX, DI
	ANDQ $-8, DI // whole vectors only; the caller handles the tail

tanhloop:
	CMPQ    SI, DI
	JGE     tanhdone
	VMOVUPS (CX)(SI*4), Y0

	VMULPS Y0, Y0, Y1 // x2

	VMULPS Y1, Y10, Y2 // n2*x2
	VADDPS Y9, Y2, Y2  // + n1
	VMULPS Y1, Y2, Y2  // * x2
	VADDPS Y8, Y2, Y2  // + n0  -> num

	VMULPS Y1, Y13, Y3 // d2*x2
	VADDPS Y12, Y3, Y3 // + d1
	VMULPS Y1, Y3, Y3  // * x2
	VADDPS Y11, Y3, Y3 // + d0  -> den

	VMULPS Y0, Y2, Y2 // num*x
	VDIVPS Y3, Y2, Y2 // / den

	VMINPS Y14, Y2, Y2 // min(1, .)
	VMAXPS Y15, Y2, Y2 // max(-1, .)

	VMOVUPS Y2, (AX)(SI*4)
	ADDQ    $8, SI
	JMP     tanhloop

tanhdone:
	VZEROUPPER
	RET

// func quantizeAVX2(q []int16, in []float32, bias int32)
//
// q[i] = bias + floor(0.5 + 127*in[i]), sixteen lanes at a time. bias is 0 for
// the signed convention and 127 for the unsigned one, which is the only
// difference between them once activations are widened to int16.
//
// The arithmetic is float32 throughout, matching quantizeGeneric. Upstream adds
// the 0.5 in double because .5 is a double literal there, which agrees for every
// |127*x| below 2^22; every activation reaching a quantised layer is a tanh
// output in [-1,1].
TEXT ·quantizeAVX2(SB), NOSPLIT, $0-52
	MOVQ  q_base+0(FP), AX
	MOVQ  in_base+24(FP), CX
	MOVQ  in_len+32(FP), DX
	MOVL  bias+48(FP), BX

	VBROADCASTSS f127<>(SB), Y8
	VBROADCASTSS fhalf<>(SB), Y9
	MOVL         BX, X10
	VPBROADCASTD X10, Y10

	XORQ SI, SI
	MOVQ DX, DI
	ANDQ $-16, DI // whole 16-lane groups only

quantloop:
	CMPQ    SI, DI
	JGE     quantdone
	VMOVUPS (CX)(SI*4), Y0
	VMOVUPS 32(CX)(SI*4), Y1

	VMULPS    Y8, Y0, Y0
	VADDPS    Y9, Y0, Y0
	VROUNDPS  $1, Y0, Y0 // floor
	VCVTPS2DQ Y0, Y0
	VPADDD    Y10, Y0, Y0

	VMULPS    Y8, Y1, Y1
	VADDPS    Y9, Y1, Y1
	VROUNDPS  $1, Y1, Y1
	VCVTPS2DQ Y1, Y1
	VPADDD    Y10, Y1, Y1

	// Pack to int16. VPACKSSDW works per 128-bit lane, giving
	// [a0-3 b0-3 a4-7 b4-7]; VPERMQ with 0xD8 selects quadwords 0,2,1,3 to
	// restore [a0-7 b0-7]. Values lie in [-127,254], so the saturating pack
	// never clamps.
	VPACKSSDW Y1, Y0, Y2
	VPERMQ    $0xD8, Y2, Y2
	VMOVDQU   Y2, (AX)(SI*2)

	ADDQ $16, SI
	JMP  quantloop

quantdone:
	VZEROUPPER
	RET

// func vecSigmoidAVX2(y []float32, x []float32, coef *[8]float32)
//
// sigmoid(x) = .5 + .5*tanh_approx(.5*x), upstream's own definition, so this is
// the tanh kernel with a pre-scale and a post-scale. Keeping it as its own
// kernel rather than staging through a shared scratch buffer avoids introducing
// package-level mutable state, which would race between Denoisers running on
// different goroutines.
TEXT ·vecSigmoidAVX2(SB), NOSPLIT, $0-56
	MOVQ y_base+0(FP), AX
	MOVQ x_base+24(FP), CX
	MOVQ x_len+32(FP), DX
	MOVQ coef+48(FP), BX

	VBROADCASTSS 0(BX), Y8
	VBROADCASTSS 4(BX), Y9
	VBROADCASTSS 8(BX), Y10
	VBROADCASTSS 12(BX), Y11
	VBROADCASTSS 16(BX), Y12
	VBROADCASTSS 20(BX), Y13
	VBROADCASTSS 24(BX), Y14
	VBROADCASTSS 28(BX), Y15
	VBROADCASTSS fhalf<>(SB), Y7

	XORQ SI, SI
	MOVQ DX, DI
	ANDQ $-8, DI

sigloop:
	CMPQ    SI, DI
	JGE     sigdone
	VMOVUPS (CX)(SI*4), Y0

	VMULPS Y7, Y0, Y0 // .5*x

	VMULPS Y0, Y0, Y1 // x2

	VMULPS Y1, Y10, Y2
	VADDPS Y9, Y2, Y2
	VMULPS Y1, Y2, Y2
	VADDPS Y8, Y2, Y2

	VMULPS Y1, Y13, Y3
	VADDPS Y12, Y3, Y3
	VMULPS Y1, Y3, Y3
	VADDPS Y11, Y3, Y3

	VMULPS Y0, Y2, Y2
	VDIVPS Y3, Y2, Y2

	VMINPS Y14, Y2, Y2
	VMAXPS Y15, Y2, Y2

	VMULPS Y7, Y2, Y2 // .5*tanh
	VADDPS Y7, Y2, Y2 // .5 + .
	VMOVUPS Y2, (AX)(SI*4)

	ADDQ $8, SI
	JMP  sigloop

sigdone:
	VZEROUPPER
	RET

DATA f127<>+0(SB)/4, $0x42fe0000 // 127.0
GLOBL f127<>(SB), RODATA|NOPTR, $4

DATA fhalf<>+0(SB)/4, $0x3f000000 // 0.5
GLOBL fhalf<>(SB), RODATA|NOPTR, $4

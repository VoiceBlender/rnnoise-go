// AVX2 four-lag correlation for the pitch search.
//
// Unlike the kernels in simd_amd64.s this one DOES use VFMADD231PS and splits
// each lag's sum across eight lanes, so it is not bit-exact against the C
// scalar build. It runs only when Options.Fast is set; see simd_pitch_amd64.go.

#include "textflag.h"

// func xcorrVecAVX2(x, y []float32, acc *[32]float32, length int) int
//
// acc[8*k+l] receives lane l of lag k's partial sum, for k in 0..3. Returns the
// number of elements consumed, always a multiple of 8; the caller handles the
// remainder and the horizontal sums.
TEXT ·xcorrVecAVX2(SB), NOSPLIT, $0-72
	MOVQ x_base+0(FP), SI
	MOVQ y_base+24(FP), DI
	MOVQ acc+48(FP), DX
	MOVQ length+56(FP), CX

	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	XORQ AX, AX
	MOVQ CX, R8
	SHRQ $3, R8         // whole 8-element blocks
	TESTQ R8, R8
	JZ   done

loop:
	VMOVUPS (SI)(AX*4), Y0    // x[j..j+7]
	VMOVUPS (DI)(AX*4), Y8    // y[j+0..]
	VMOVUPS 4(DI)(AX*4), Y9   // y[j+1..]
	VMOVUPS 8(DI)(AX*4), Y10  // y[j+2..]
	VMOVUPS 12(DI)(AX*4), Y11 // y[j+3..]
	VFMADD231PS Y8, Y0, Y4
	VFMADD231PS Y9, Y0, Y5
	VFMADD231PS Y10, Y0, Y6
	VFMADD231PS Y11, Y0, Y7
	ADDQ $8, AX
	DECQ R8
	JNZ  loop

done:
	VMOVUPS Y4, 0(DX)
	VMOVUPS Y5, 32(DX)
	VMOVUPS Y6, 64(DX)
	VMOVUPS Y7, 96(DX)
	VZEROUPPER
	MOVQ AX, ret+64(FP)
	RET

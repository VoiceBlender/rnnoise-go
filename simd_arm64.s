// ARM64 NEON kernels.
//
// Go's arm64 assembler has no vector integer multiply of any kind -- MUL and
// SMULL are scalar, VPMULL is polynomial, and there is no SDOT or SMLAL -- so the
// dot product is emitted as a raw WORD. Only that one instruction is encoded by
// hand; everything else uses named instructions the assembler checks.
//
// The encoding was taken from a real assembler rather than derived from the
// manual (see the note in simd_arm64.go), and TestSDOTEncoding re-derives it
// from the same table so a typo cannot pass silently.

#include "textflag.h"

// func cgemvInt8SDOT(acc []int32, w []int8, q8 []int8, rows int, cols int)
//
// acc[i] = sum_j w[i,j]*q8[j] for an 8-row by 4-column blocked int8 matrix,
// left as raw int32. The caller applies the per-row scale in Go: that is
// 1152 multiplies against 442368 multiply-accumulates, so it costs nothing and
// it keeps two more instructions (SCVTF and a vector FMUL, neither of which Go's
// assembler has) out of hand-encoded form.
//
// SDOT is exact -- int8 x int8 products accumulated into int32, no saturation --
// so this is bit-identical to cgemvInt8Generic, which simd_test.go asserts.
//
// The weight layout suits SDOT exactly. A 32-byte block holds w[4*k+q] for row k
// and tap q, so bytes 0-15 are rows 0-3 with their four taps adjacent, which is
// precisely the grouping SDOT reduces: one instruction produces four rows'
// four-tap dot products.
//
// Requires rows % 8 == 0, cols % 4 == 0 and ASIMDDP.
TEXT ·cgemvInt8SDOT(SB), NOSPLIT, $0-88
	MOVD acc_base+0(FP), R0
	MOVD w_base+24(FP), R1
	MOVD q8_base+48(FP), R2
	MOVD rows+72(FP), R3
	MOVD cols+80(FP), R4

	MOVD $0, R5 // i, the row-block cursor

rowblock:
	CMP  R3, R5
	BGE  done
	VEOR V16.B16, V16.B16, V16.B16 // rows i+0..i+3
	VEOR V17.B16, V17.B16, V17.B16 // rows i+4..i+7
	MOVD R2, R6                    // activation cursor
	MOVD $0, R7                    // j, the column cursor

colloop:
	CMP   R4, R7
	BGE   store
	VLD1  (R1), [V0.B16, V1.B16] // one 8x4 block: rows 0-3, then 4-7
	VLD1R (R6), [V2.S4]          // four taps, replicated across all four lanes
	WORD  $0x4E829410            // SDOT V16.4S, V0.16B, V2.16B
	WORD  $0x4E829431            // SDOT V17.4S, V1.16B, V2.16B
	ADD   $32, R1
	ADD   $4, R6
	ADD   $4, R7
	B     colloop

store:
	VST1 [V16.S4, V17.S4], (R0)
	ADD  $32, R0
	ADD  $8, R5
	B    rowblock

done:
	RET

// func sgemvFMA(out []float32, weights []float32, rows int, cols int, colStride int, x []float32)
//
// out[i] = sum_j weights[j*colStride + i] * x[j], for rows a multiple of 8.
//
// Unlike the amd64 kernel this one *does* fuse, with VFMLA. On arm64 Go's own
// compiler already contracts the equivalent Go source into FMADDS, so bit
// exactness against the unfused C reference is unavailable on this architecture
// whatever the assembly does -- see exactarch.go. Given that, fusing is the
// right choice: it is faster and more accurate, and it matches what the generic
// Go path on this architecture already produces.
TEXT ·sgemvFMA(SB), NOSPLIT, $0-96
	MOVD out_base+0(FP), R0
	MOVD weights_base+24(FP), R1
	MOVD rows+48(FP), R2
	MOVD cols+56(FP), R3
	MOVD colStride+64(FP), R4
	MOVD x_base+72(FP), R5

	LSL  $2, R4, R4 // colStride: floats -> bytes
	MOVD $0, R6     // i

	// 32 rows at a time, held in four accumulators across the whole j loop.
rowblock32:
	MOVD R2, R7
	SUB  $32, R7, R7
	CMP  R7, R6
	BGT  rowblock8

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16

	MOVD R1, R8
	ADD  R6<<2, R8, R8 // &weights[i]
	MOVD R5, R9        // &x[0]
	MOVD $0, R10       // j

col32:
	CMP   R3, R10
	BGE   store32
	VLD1R (R9), [V8.S4] // x[j], broadcast
	VLD1  (R8), [V9.S4, V10.S4, V11.S4, V12.S4]
	ADD   $64, R8, R11
	VLD1  (R11), [V13.S4, V14.S4, V15.S4, V16.S4]
	VFMLA V8.S4, V9.S4, V0.S4
	VFMLA V8.S4, V10.S4, V1.S4
	VFMLA V8.S4, V11.S4, V2.S4
	VFMLA V8.S4, V12.S4, V3.S4
	VFMLA V8.S4, V13.S4, V4.S4
	VFMLA V8.S4, V14.S4, V5.S4
	VFMLA V8.S4, V15.S4, V6.S4
	VFMLA V8.S4, V16.S4, V7.S4
	ADD   $4, R9
	ADD   R4, R8
	ADD   $1, R10
	B     col32

store32:
	MOVD R0, R11
	ADD  R6<<2, R11, R11
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R11)
	ADD  $64, R11, R12
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R12)
	ADD  $32, R6
	B    rowblock32

	// Remainder: 8 rows at a time.
rowblock8:
	CMP  R2, R6
	BGE  fmadone

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	MOVD R1, R8
	ADD  R6<<2, R8, R8
	MOVD R5, R9
	MOVD $0, R10

col8:
	CMP   R3, R10
	BGE   store8
	VLD1R (R9), [V8.S4]
	VLD1  (R8), [V9.S4, V10.S4]
	VFMLA V8.S4, V9.S4, V0.S4
	VFMLA V8.S4, V10.S4, V1.S4
	ADD   $4, R9
	ADD   R4, R8
	ADD   $1, R10
	B     col8

store8:
	MOVD R0, R11
	ADD  R6<<2, R11, R11
	VST1 [V0.S4, V1.S4], (R11)
	ADD  $8, R6
	B    rowblock8

fmadone:
	RET

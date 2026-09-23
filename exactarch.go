package rnnoise

import "runtime"

// bitExactArch reports whether this architecture reproduces upstream's C scalar
// build bit for bit.
//
// It is amd64 only, and the reason is the compiler rather than this package.
// Upstream computes unfused expressions -- `y[k] += w[k]*xj`, and a rational
// tanh built from separate multiplies and adds -- and Go's amd64 backend emits
// MULSS/ADDSS, matching it. Go's arm64 backend contracts the same source into
// FMADDS, so every float multiply-add in the port rounds once instead of twice
// and the results differ in the last bits.
//
// Verify with:
//
//	GOARCH=arm64 go build -gcflags=-S ./ 2>&1 | grep FMADDS
//
// This is not a defect on either side. A fused multiply-add is *more* accurate,
// and the assembly kernels deliberately avoid it only so that bit-exactness
// against the reference stays available as a verification tool. What arm64 gives
// up is that tool, not correctness: it runs the same Go source, validated on
// amd64, and its outputs agree with the amd64 build to within float rounding.
//
// Go offers no -ffp-contract=off. Fusion can be suppressed per expression by
// writing float32(a*b) + c, since an explicit conversion to the same type is
// specified to round, but doing that at every multiply-add site across the FFT,
// the band energies, the pitch search and the network would trade arm64
// accuracy and speed for a property only the test suite consumes.
func bitExactArch() bool { return runtime.GOARCH == "amd64" }

// exactnessSkipReason explains a skip in terms a reader can act on.
const exactnessSkipReason = "bit-exactness against the C reference is an amd64 property: " +
	"Go's arm64 backend contracts float multiply-adds into FMADDS, where upstream and " +
	"Go/amd64 both round twice. See bitExactArch in exactarch.go. The tolerance-based " +
	"tests still run here and cover this architecture."

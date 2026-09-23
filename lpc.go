package rnnoise

// Ported from upstream's src/celt_lpc.c (CELT's LPC helpers), float branch:
// SHR32/SHL32/EXTEND32/ROUND16 are identities and MULT32_32_Q31(a,b) is a*b,
// so the shifts in the C source vanish here.

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// celtLPC is rnn_lpc: Levinson-Durbin recursion from autocorrelation to LPC
// coefficients, bailing out once it has 30 dB of prediction gain.
func celtLPC(lpc []float32, ac []float32, p int) {
	for i := 0; i < p; i++ {
		lpc[i] = 0
	}
	if ac[0] == 0 {
		return
	}
	err := ac[0]
	for i := 0; i < p; i++ {
		// Sum up this iteration's reflection coefficient.
		var rr float32
		for j := 0; j < i; j++ {
			rr += lpc[j] * ac[i-j]
		}
		rr += ac[i+1]
		r := -rr / err
		// Update LPC coefficients and total error.
		lpc[i] = r
		for j := 0; j < (i+1)>>1; j++ {
			tmp1 := lpc[j]
			tmp2 := lpc[i-1-j]
			lpc[j] = tmp1 + r*tmp2
			lpc[i-1-j] = tmp2 + r*tmp1
		}
		err = err - (r*r)*err
		// Bail out once we get 30 dB gain.
		if err < .001*ac[0] {
			break
		}
	}
}

// celtAutocorr is rnn_autocorr restricted to the overlap == 0 case, which is
// the only way rnn_pitch_downsample calls it (window NULL, overlap 0). The
// windowing branch and its PITCH_BUF_SIZE/2 scratch array are therefore not
// ported.
func celtAutocorr(x []float32, ac []float32, lag, n int) {
	fastN := n - lag
	pitchXCorr(x, x, ac, fastN, lag+1)
	for k := 0; k <= lag; k++ {
		var d float32
		for i := k + fastN; i < n; i++ {
			d += x[i] * x[i-k]
		}
		ac[k] += d
	}
}

// celtFIR5 is celt_fir5 from src/pitch.c: a fixed order-5 FIR run in place
// over the half-rate signal. For the float build SHL32(EXTEND32(x),SIG_SHIFT)
// and ROUND16(sum,SIG_SHIFT) are both identities.
func celtFIR5(x []float32, num []float32, y []float32, n int, mem *[5]float32) {
	num0, num1, num2, num3, num4 := num[0], num[1], num[2], num[3], num[4]
	mem0, mem1, mem2, mem3, mem4 := mem[0], mem[1], mem[2], mem[3], mem[4]
	for i := 0; i < n; i++ {
		sum := x[i]
		sum += num0 * mem0
		sum += num1 * mem1
		sum += num2 * mem2
		sum += num3 * mem3
		sum += num4 * mem4
		mem4 = mem3
		mem3 = mem2
		mem2 = mem1
		mem1 = mem0
		mem0 = x[i]
		y[i] = sum
	}
	mem[0], mem[1], mem[2], mem[3], mem[4] = mem0, mem1, mem2, mem3, mem4
}

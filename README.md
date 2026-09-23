# rnnoise-go

A pure-Go port of [Xiph's RNNoise](https://gitlab.xiph.org/xiph/rnnoise) recurrent
neural-network noise suppressor: 32 bands, two causal convolutions feeding three
GRUs with skip connections, int8-quantised weights.

No cgo. No wasm runtime. One dependency, `golang.org/x/sys`, for CPU feature
detection.

**It runs natively at any sample rate from 8 kHz up**, which the C original does
not: upstream is hard-wired to 48 kHz and its maintainer's advice for anything
else is to resample to 48 kHz and back.

```go
import (
    rnnoise "github.com/VoiceBlender/rnnoise-go"
    "github.com/VoiceBlender/rnnoise-go/model"
)

d, err := rnnoise.New(rnnoise.Options{SampleRate: 16000, Model: model.MustLoad()})
// d.FrameSize() == 160 samples == 10 ms
for /* each frame */ {
    vad, err := d.ProcessInt16(out, in) // vad is the frame's speech probability
}
```

Samples are int16-scaled (about ±32768), not ±1.0, matching upstream. This is
not cosmetic: the silence gate and several feature offsets are calibrated
against that absolute magnitude. `ProcessInt16` sidesteps the question.

`model/` is a separate module, so importing the engine does not pull in the
3.4 MB weight blob; a program that loads weights from disk pays nothing for it.

## Why native multi-rate works

Upstream's band table, `eband20ms`, is expressed in **FFT bin indices at 50 Hz
per bin**, not in hertz. Keep the frame at 10 ms and the analysis window at
20 ms, and the bin spacing stays exactly 50 Hz at every sample rate — so the
same table names the same absolute frequencies everywhere. The only consequence
of a lower rate is that bins above its Nyquist do not exist.

That is precisely the input upstream trains on. `dump_features.c` band-limits
its training data by zeroing FFT bins above a cutoff drawn log-uniformly from
[3.0, 24.1] kHz, which happens for **47% of training frames** — 25% at or below
8 kHz. Reduced-rate operation is inside the model's training distribution, not
an extrapolation from it.

Two further facts make it exact rather than approximate. kiss_fft is 1/N-scaled,
so by Parseval the band energies measure signal *power* and do not grow with the
transform length — which keeps the absolute-magnitude calibrations valid. And
the pitch index is rescaled to 48 kHz samples before it reaches the network,
since that is the unit it was trained on.

Measured, for a signal below every rate's Nyquist (`features_test.go`):

| | 8 kHz | 16 kHz | 44.1 kHz |
|---|---|---|---|
| Max deviation of the 32 cepstral features from a 48 kHz run | 0.0021 | 0.00084 | 0.00003 |
| Pitch features, whole-sample period | 2e-5 | 1e-5 | <1e-5 |

The one real cost is pitch lag resolution. The period is found to within about
two native samples at any rate, so relative accuracy falls from 0.79% at 48 kHz
to 4.98% at 8 kHz, and gross (octave) errors rise from 0/87 at 16 kHz and above
to 2/87 at 8–12 kHz. Fractional-lag interpolation is deliberately *not* used:
the model was trained on `Exp` computed from integer lags by exactly this code.

End-to-end on synthetic voiced speech in white noise (`TestEndToEnd`):

| rate | noise-only attenuation | speech attenuation | VAD speech / pause | active bands |
|---|---|---|---|---|
| 8000 | −25.9 dB | −5.5 dB | 0.96 / 0.60 | 20 |
| 16000 | −25.5 dB | −4.8 dB | 0.99 / 0.59 | 26 |
| 48000 | −28.7 dB | −4.4 dB | 1.00 / 0.39 | 32 |

## Verified against the C reference

At 48 kHz the port is **bit-exact** against upstream's C, every stage, over
600 frames of silence, voiced speech, noise and a full-scale transient — the
FFT, band energies, features, all three GRU states, gains, output, VAD, pitch
gain and index.

`make diff-vs-c` reproduces it. The harness builds the C library via
`ctest/c/Makefile` and dumps per-stage intermediates from a shim that
`#include`s `denoise.c`, so upstream is compared unpatched and the shim contains
wiring only, never arithmetic. The bit-exact oracle is the **scalar** C build
(`-U__SSE2__ -U__SSE__ -U__AVX__`), which selects `vec.h`'s portable branch:
signed int8 activations with `bias`, and a real divide in tanh/sigmoid. That is
also what upstream's ARM NEON build uses, and it is this port's default
`QuantSigned` mode.

A **stock AVX2** C build cannot be matched bit-for-bit, for two reasons:

- It uses `_mm256_rcp_ps` for the activation denominator. That result is
  microarchitecture-defined — AMD and Intel differ — so no implementation can
  match it.
- More surprisingly, **GCC's FMA contraction moves the DSP front end**, which
  has no SIMD variant upstream at all. Rebuilding the same AVX2 variant with
  `-ffp-contract=off` (`make -C ctest/c nofma`) makes the FFT, band energies and
  features bit-identical to the scalar build again. That divergence belongs to
  the compiler, not to any port.

Three stacked recurrent layers amplify both: 142 dB at the features becomes
60 dB after conv2 and 31 dB at the output. The harness therefore measures Go's
`QuantUnsigned` mode against a control — C-scalar versus C-AVX2, two builds of
the *same source* — and Go lands at or slightly better than that lower bound on
every stage (output 30.9 dB against the control's 30.5 dB).

`TestGoldenStages` keeps all of this under guard in plain CI with no C
toolchain, by checking per-stage sha256 digests **of the C reference's own
output**, committed in `testdata/cgolden_48k.json`.

## What native rate-scaling costs

`make nativerate-report` measures it against the fair comparison: resampling to
48 kHz and back. Both chains are this Go code and this model, so the only
difference between them is the native-rate approximation.

Averaged over 0/10/20 dB SNR mixtures. ΔsegSNRi measures quality against the
*clean* signal, so native mode is allowed to come out ahead, and positive values
mean it did.

| corpus | rate | ΔsegSNRi | output SNR | VAD agreement | LSD native / ref |
|---|---|---|---|---|---|
| real | 8000 | −0.09 dB | 18.2 dB | 96.3% | 5.09 / 5.37 |
| real | 16000 | −0.05 dB | 19.9 dB | 98.1% | 5.09 / 5.34 |
| real | 44100 | **+0.11 dB** | 25.0 dB | 99.4% | 5.37 / 5.36 |
| synthetic | 8000 | −0.76 dB | 18.3 dB | 97.8% | 2.97 / 2.84 |
| synthetic | 16000 | −0.32 dB | 20.7 dB | 97.8% | 2.67 / 2.64 |
| synthetic | 44100 | **+0.19 dB** | 24.1 dB | 98.5% | 2.87 / 2.87 |

**On real speech, native mode is indistinguishable from resampling** — within
0.1 dB at every rate, and it reaches the clean signal slightly more closely
(lower LSD) at 8 and 16 kHz, having avoided two resampling passes.

**The cost is concentrated in pitch accuracy, and shows up on strongly periodic
signals.** The synthetic corpus is a harmonic stack with a drifting fundamental,
more strictly periodic than any real voice, and deliberately so: it is a stress
test for the one thing native mode genuinely gives up.

Practical guidance:

- **At 16 kHz and above, use native mode.** The measured cost is at or below
  0.3 dB even on the stress corpus, and nil on speech.
- **At 8–12 kHz, native mode is fine for speech** — within 0.1 dB on real
  telephony recordings, which is what those rates carry in practice. If your
  material is unusually periodic (music, tones, synthetic voices) and the last
  half-decibel matters, resample to 48 kHz instead.
- Either way, native mode saves two resampling passes and their latency.

The real corpus is eight 8 kHz telephony recordings, two 16 kHz clips, and real
call-centre and street noise at 16 kHz. **Every source is band-limited at or
below 8 kHz**, so the "real" rows above 16 kHz test upsampled narrowband
material in both chains, which makes them agree more easily than genuinely
wideband audio would. That is why the synthetic corpus is reported separately
rather than averaged in.

## Performance

One core of a Ryzen 9 7900, AVX2. Real-time factor is how many streams one core
could carry. Sessions per CPU is the number to plan against: it is measured with
every hardware thread busy, where per-thread capacity is about 2.4× lower than a
single-threaded benchmark suggests. Method and full detail are in
[BENCHMARKS.md](BENCHMARKS.md); `make bench-report` reproduces them and
`testdata/capacity.json` is the committed baseline `make bench-capacity` checks.

| rate | µs/frame | ×RT | sessions/CPU |
|---|---|---|---|
| 8 kHz | 58.8 | 170 | **1637** |
| 16 kHz | 67.3 | 149 | **1547** |
| 32 kHz | 86.6 | 116 | **1227** |
| 44.1 kHz | 105.7 | 95 | **1044** |
| 48 kHz | 108.4 | 92 | **1022** |
| (reference) C with AVX2, 48 kHz | 65.2 | 153 | 1440–1522 |

At 16 kHz the port is **1.6× cheaper than native C**, because a 48 kHz-only
denoiser pays full 48 kHz cost whatever the input rate and then pays for
resampling on top.

Allocation-free after construction (0 B/op, 0 allocs/op), so Denoisers pool
cleanly across streams, and a `Model` is read-only so one copy of the weights
backs all of them. `TestConcurrentDenoisers` checks that under `-race`.

Individual AVX2 kernels, against the portable Go they replace:

| kernel | generic | AVX2 | |
|---|---|---|---|
| int8 GEMV, one GRU gate matrix | 256 µs | **5.7 µs** | 45× — 77 GB/s, at the memory bandwidth limit |
| float GEMV, `dense_out` | 21.6 µs | **1.2 µs** | 18× |

Every one is **bit-identical** to its portable counterpart, not merely close,
and `simd_test.go` asserts exact equality. That follows from a property of the
arithmetic rather than luck: the int8 partial sums are whole numbers bounded by
96 blocks × 4 taps × 127 × 254 = 12,387,072, comfortably inside float32's
exact-integer range, so accumulating in int32 lanes in any order gives the same
answer as upstream's float32 running sum. `TestCgemvInt8AccumulatorBound`
measures that headroom (74% of 2²⁴ in the worst case) and fails if a future
model erodes it.

Two deliberate refusals, both costing speed to keep exactness: **no FMA**, since
upstream computes an unfused `y[k] += w[k]*xj` and fusing would round
differently; and **a real divide in the activations**, not `VRCPPS`.

## ARM64

NEON kernels for the int8 GEMV (`SDOT`) and the float GEMV (`VFMLA`), validated
by execution under qemu — `make test-arm64` cross-compiles the test binary and
runs the whole suite in an arm64 container. The SDOT kernel is **bit-identical**
to the portable one, asserted over every shape the model uses plus
saturated-magnitude worst cases.

Go's arm64 assembler has no vector integer multiply, so the dot product is
emitted as a raw `WORD`. `TestSDOTEncoding` reads the assembly back and
re-derives every literal from the encoding formula, so a mistyped digit fails a
test instead of quietly computing something else. `QuantUnsigned` takes the
generic path, since its activations do not fit int8 and `USDOT` needs ARMv8.6
rather than 8.2; a core without ARMv8.2 DotProd also falls back — present on
every Cortex-A since the A75, all of Apple's arm64, and Graviton 2 onward.

**Performance on ARM64 is unmeasured.** Emulation validates correctness but its
timings are meaningless; real figures need real hardware.

**Bit-exactness against the C reference is an amd64 property.** Go's arm64
backend contracts float multiply-adds into `FMADDS`, where upstream's C and
Go/amd64 both round twice. That is not a defect on either side — a fused
multiply-add is *more* accurate, and the assembly kernels avoid it only to keep
bit-exactness usable as a verification tool. ARM64 gives up the tool, not
correctness: it runs the same Go source, validated on amd64, and agrees with the
amd64 build to within float rounding. `TestGoldenStages` and `TestDiffVsC`
therefore skip there with that explanation rather than failing and looking like
a broken port; every tolerance-based test still runs. See `exactarch.go`.

## Licence and provenance

BSD-3-Clause, the same terms as upstream — see `LICENSE`.

The algorithm, the network topology and the trained weights are the work of
Jean-Marc Valin and contributors, with portions deriving from Opus/CELT.
Upstream's notice is reproduced verbatim in `COPYING.rnnoise`, and `NOTICE`
records what derives from where. The weights' origin and pinned hash are in
`model/LICENSE.weights`; `make verify-model` checks the committed blob.

# Benchmarks

Reproduce with `make bench-report`. The machine-readable baseline is
`testdata/capacity.json`, which `make bench-capacity` checks for regressions and
`make update-capacity` re-records.

All figures are from one host:

| | |
|---|---|
| CPU | AMD Ryzen 9 7900, 12 cores / 24 threads, boost 5.48 GHz |
| Memory | 384 KiB L1d, 12 MiB L2, 64 MiB L3 (2 instances) |
| Go | 1.26.3, `CGO_ENABLED=0` |
| C reference | gcc 13.3, `-O2 -mavx -mfma -mavx2` |

## Reading these numbers

A frame is always 10 ms of audio whatever the sample rate, so 10 ms divided by
the per-frame cost is the real-time factor, which is also sessions per core at
100% CPU.

Two figures are given, and the gap between them is the point. Single-thread is
one stream with nothing else running. Saturated is one stream per hardware
thread, all busy.

Saturated capacity per thread runs about 2.4× lower than the single-thread
figure, for three reasons: the single-thread run has the turbo clock to itself
(about 5.5 GHz against 4.2 all-core), 24 hardware threads share 12 physical
cores, and every stream pulls 2.8 MB of weights per frame through a shared L3,
so one read-only copy of the model saves memory but not bandwidth.

Plan against the saturated column. The single-thread figure overstates a real
server by more than double.

The signal is synthetic voiced speech that never trips the silence gate, so
these are a worst case; real audio with pauses is cheaper, since a gated frame
skips the network. Measurements take the best of several runs, which suppresses
thermal and scheduler noise. Go figures then repeat to within about 1.5%
between independent runs, C to within about 6%.

## By sample rate

| rate | µs/frame | ×RT = sessions/core | sessions/CPU (saturated) | sessions/thread |
|---|---|---|---|---|
| 8000 | 58.8 | 170 | 1637 | 68.2 |
| 12000 | 62.7 | 159 | 1611 | 67.1 |
| 16000 | 67.3 | 149 | 1547 | 64.5 |
| 24000 | 77.5 | 129 | 1380 | 57.5 |
| 32000 | 86.6 | 116 | 1227 | 51.1 |
| 44100 | 105.7 | 95 | 1044 | 43.5 |
| 48000 | 108.4 | 92 | 1022 | 42.6 |

Cost falls with the rate because the DSP front end shrinks with the frame while
the network's cost is fixed: the network is roughly 60% of a frame at 48 kHz and
most of the rest at 8 kHz.

## Against the C original, 48 kHz

| | µs/frame | ×RT | sessions/CPU (saturated) |
|---|---|---|---|
| C, AVX2 | 65.2 | 153 | 1440 to 1522 |
| Go | 108.4 | 92 | 1022 |
| C, scalar | 767.6 | 13 | |

Go is about 1.4× behind C at 48 kHz. The gap is the price of choices made to
keep the port bit-exact against the C scalar build: a non-saturating integer
multiply-accumulate that does half the work per instruction of the saturating
one C uses, a real divide in the activations rather than a reciprocal
approximation, and no fused multiply-add in the float path. C also gets an
autovectorised FFT from GCC where the Go one is scalar.

## At other rates, the port is cheaper than C

The C original is 48 kHz only, so serving another rate means resampling both
ways. Measured with the Speex resampler, per 10 ms of audio, round trip:

| quality | 8 kHz | 16 kHz |
|---|---|---|
| 3 | 24.8 µs | 24.0 µs |
| 5 | 42.1 µs | 39.9 µs |
| 10 | 136.8 µs | 135.9 µs |

At 16 kHz per frame, taking quality 5:

| | µs/frame | sessions/core |
|---|---|---|
| Go, native 16 kHz | 67.3 | 149 |
| C plus resampling | 105.1 | 95 |

Native rate-scaling makes the port 1.6× cheaper than C at 16 kHz, because a
48 kHz-only denoiser pays full 48 kHz cost whatever the input rate and then pays
for resampling on top. At quality 10 the resampling alone costs more than the
entire Go denoiser at 16 kHz.

## Kernel level

Against the portable Go each kernel replaces, and bit-identical to it:

| kernel | generic | AVX2 | |
|---|---|---|---|
| int8 matrix-vector, one GRU gate matrix | 256 µs | 5.7 µs | 45×, memory-bandwidth bound at 77 GB/s |
| float matrix-vector, output layer | 21.6 µs | 1.2 µs | 18× |

# Benchmarks

Reproduce with `make bench-report`. The machine-readable Go baseline is
`testdata/capacity.json`, which `make bench-capacity` checks for regressions and
`make update-capacity` re-records.

All figures below are from one host:

| | |
|---|---|
| CPU | AMD Ryzen 9 7900, 12 cores / 24 threads, boost 5.48 GHz |
| Memory hierarchy | 384 KiB L1d, 12 MiB L2, 64 MiB L3 (2 instances) |
| Go | 1.26.3, `CGO_ENABLED=0` |
| C reference | gcc 13.3, `-O2 -mavx -mfma -mavx2`, upstream at `70f1d256` |

## Reading these numbers

A frame is always 10 ms of audio, whatever the sample rate, so **10 ms divided by
the per-frame cost is the real-time factor, which is also sessions per core at
100% CPU**.

Two figures are given, and the gap between them is the point:

- **Single-thread** — one stream, nothing else running.
- **Saturated** — one stream per hardware thread, all busy.

Saturated capacity per thread is about 2.4× lower than the single-thread figure.
Three things account for it: the single-thread run has the turbo clock to itself
(~5.5 GHz against ~4.2 all-core), 24 hardware threads share 12 physical cores,
and every stream pulls 2.8 MB of weights per frame through a shared L3 — sharing
one read-only copy of the model saves memory, not bandwidth.

**Plan against the saturated column.** Quoting the single-thread figure would
overstate a real server by more than double.

The signal is synthetic voiced speech that never trips the silence gate, so these
are a worst case; real audio with pauses is cheaper, because a gated frame skips
the network entirely.

Measurements take the best of several runs, which suppresses thermal and
scheduler noise. With that, the Go figures repeat to within about 1.5% between
independent runs and the C ones to within about 6%.

## Sessions per CPU, by sample rate (Go port)

| rate | µs/frame | ×RT = sessions/core | **sessions/CPU (saturated)** | sessions/thread |
|---|---|---|---|---|
| 8000 | 58.8 | 170 | **1637** | 68.2 |
| 12000 | 62.7 | 159 | **1611** | 67.1 |
| 16000 | 67.3 | 149 | **1547** | 64.5 |
| 24000 | 77.5 | 129 | **1380** | 57.5 |
| 32000 | 86.6 | 116 | **1227** | 51.1 |
| 44100 | 105.7 | 95 | **1044** | 43.5 |
| 48000 | 108.4 | 92 | **1022** | 42.6 |

Cost falls with the rate because the DSP front end shrinks with the frame while
the network's cost is fixed — the network is roughly 60% of a frame at 48 kHz and
most of the rest at 8 kHz.

44.1 kHz was once an outlier at 148 µs, slower than 48 kHz despite having fewer
samples, because its 882-point window is 2·3²·7² and the radix-7 stages fell to a
generic O(p²) butterfly. `kfBfly7` in `fft.go` fixed that: 148 → 106 µs, and
capacity 643 → 1044 sessions/CPU.

## Against the C original and the WebAssembly build (48 kHz)

| | µs/frame | ×RT | **sessions/CPU (saturated)** |
|---|---|---|---|
| native C, AVX2 | 65.2 | 153 | **1440–1522** |
| **Go (this port)** | **108.4** | **92** | **1022** |
| WebAssembly under wazero, 8 streams/instance | 212.6 | 47 | **558** |
| WebAssembly under wazero, 24/instance (old default) | 212.6 | 47 | **400** |
| native C, scalar (no SIMD) | 767.6 | 13 | — |

Go is about **1.4× behind native C** and **2.6× ahead of the WebAssembly build**
as it was configured.

The gap to C is the price of choices made deliberately, each recorded where it is
made: `VPMADDWD` (8 MAC/instruction) instead of C's saturating `VPMADDUBSW` (16),
a real divide instead of `VRCPPS`, and no FMA in the float GEMV — all so the port
stays bit-exact against the C scalar build — plus C's GCC-autovectorised FFT
against a scalar Go one.

The WebAssembly rows are measured in VoiceBlender's own `denoise` package
(`go test -bench BenchmarkConcurrentStreams ./internal/audiofilter/denoise`),
because that is where that backend lives. Its capacity depends on how many
streams share a module instance: wazero cannot take concurrent calls into one
module, so streams sharing an instance serialise behind its lock. 8 per instance
measured best; the 24 that package defaulted to costs it 28%.

## At other rates, the Go port beats native C

The C original and the WebAssembly build are 48 kHz only, so serving another rate
means resampling both ways. Measured with the Speex resampler this workspace
uses, per 10 ms of audio, round trip:

| quality | 8 kHz | 16 kHz |
|---|---|---|
| 3 | 24.8 µs | 24.0 µs |
| 5 | 42.1 µs | 39.9 µs |
| 10 | 136.8 µs | 135.9 µs |

At 16 kHz, per frame, taking quality 5:

| | µs/frame | sessions/core |
|---|---|---|
| **Go, native 16 kHz** | **67.3** | **149** |
| native C + resample | 105.1 | 95 |
| WebAssembly + resample | 252.5 | 40 |

So native rate-scaling makes this port **1.6× cheaper than native C** at 16 kHz,
and 3.7× cheaper than the WebAssembly path — because a 48 kHz-only denoiser pays
full 48 kHz cost whatever the input rate, and then pays for resampling on top. At
quality 10 the resampling alone costs more than the entire Go denoiser at 16 kHz.

## Kernel-level

Against the portable Go each kernel replaces, and bit-identical to it:

| kernel | generic | AVX2 | |
|---|---|---|---|
| int8 GEMV, one GRU gate matrix | 256 µs | 5.7 µs | 45×, 77 GB/s — at the memory bandwidth limit |
| float GEMV, `dense_out` | 21.6 µs | 1.2 µs | 18× |

Progress at 48 kHz across the SIMD work: 1450 µs scalar → 170 (int8 GEMV) → 116
(float GEMV) → 108 (activations and quantiser).

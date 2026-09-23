# rnnoise-go

Pure-Go port of [Xiph's RNNoise](https://gitlab.xiph.org/xiph/rnnoise): 32 bands,
two causal convolutions feeding three GRUs with skip connections, int8-quantised
weights. No cgo, no wasm runtime, one dependency (`golang.org/x/sys`, for CPU
feature detection).

The C original is fixed at 48 kHz. This runs natively at any rate from 8 kHz up.

## Install

```
go get github.com/VoiceBlender/rnnoise-go
go get github.com/VoiceBlender/rnnoise-go/model
```

`model/` is a separate module, so the 3.4 MB weight blob is fetched only if you
import it. Load weights from disk with `rnnoise.LoadModelFile` instead and you
do not need it.

## Usage

```go
import (
    rnnoise "github.com/VoiceBlender/rnnoise-go"
    "github.com/VoiceBlender/rnnoise-go/model"
)

d, err := rnnoise.New(rnnoise.Options{SampleRate: 16000, Model: model.MustLoad()})
// d.FrameSize() == 160 samples == 10 ms
for /* each frame */ {
    vad, err := d.ProcessInt16(out, in) // speech probability for the frame
}
```

Frames are 10 ms; `FrameSize()` gives the sample count for the configured rate.
Samples are int16-scaled (about ±32768), not ±1.0, since the silence gate and
several feature offsets are calibrated against that absolute magnitude.

Allocation-free after construction. A `Model` is read-only and backs any number
of Denoisers concurrently; each stream needs its own `Denoiser`.

## Sample rates

Any rate from 8 kHz up, with no resampling. On real speech, native operation
measures within 0.1 dB of resampling to 48 kHz and back at every rate. The cost
falls on pitch resolution and shows up only on strongly periodic material, where
below 16 kHz resampling to 48 kHz remains slightly better.
`make nativerate-report` reproduces the comparison.

## Accuracy

At 48 kHz the port is bit-exact against upstream C at every stage, over silence,
voiced speech, noise and a full-scale transient. `make diff-vs-c` reproduces the
comparison against a scalar C build; `TestGoldenStages` keeps it under CI with
no C toolchain.

## Performance

One core of a Ryzen 9 7900, AVX2. Sessions per CPU is measured with every
hardware thread busy.

| rate | µs/frame | ×RT | sessions/CPU |
|---|---|---|---|
| 8 kHz | 58.8 | 170 | 1637 |
| 16 kHz | 67.3 | 149 | 1547 |
| 32 kHz | 86.6 | 116 | 1227 |
| 44.1 kHz | 105.7 | 95 | 1044 |
| 48 kHz | 108.4 | 92 | 1022 |
| C with AVX2, 48 kHz | 65.2 | 153 | 1440 to 1522 |

At 16 kHz the port is 1.6× cheaper than native C, since a 48 kHz-only denoiser
pays full 48 kHz cost at any input rate plus resampling. Method and full detail
in [BENCHMARKS.md](BENCHMARKS.md).

## ARM64

NEON kernels for the int8 and float matrix-vector paths, validated under
emulation; throughput on real hardware is unmeasured. Bit-exactness against C is
an amd64 property, since the arm64 compiler fuses float multiply-adds, so the
exactness tests skip there and the tolerance-based tests run.

## Licence

BSD-3-Clause, the same terms as upstream; see `LICENSE`. The algorithm, network
topology and trained weights are the work of Jean-Marc Valin and contributors,
with portions from Opus/CELT. `NOTICE` records provenance; the weights' origin
and pinned hash are in `model/LICENSE.weights`, checked by `make verify-model`.

// Command rnnoise denoises a 16-bit PCM WAV file.
//
// It runs at the file's own sample rate, so no resampling is involved for any
// rate from 8 kHz up. Each channel is denoised independently. The 20 ms
// algorithmic delay is compensated, so the output is the same length as the
// input and aligned with it.
//
// Usage:
//
//	rnnoise [flags] input.wav output.wav
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	rnnoise "github.com/VoiceBlender/rnnoise-go"
	"github.com/VoiceBlender/rnnoise-go/model"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("rnnoise: ")

	modelPath := flag.String("model", "", "read weights from this file instead of the embedded ones")
	quiet := flag.Bool("q", false, "suppress the summary")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: rnnoise [flags] input.wav output.wav\n\n"+
			"Denoises 16-bit PCM WAV at its native sample rate (8 kHz and up).\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() != 2 {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(flag.Arg(0), flag.Arg(1), *modelPath, *quiet); err != nil {
		// The library prefixes its own errors, and so does the logger.
		log.Fatal(strings.TrimPrefix(err.Error(), "rnnoise: "))
	}
}

func run(inPath, outPath, modelPath string, quiet bool) error {
	w, err := readWAV(inPath)
	if err != nil {
		return err
	}
	if w.frames() == 0 {
		return fmt.Errorf("%s: no audio samples", inPath)
	}

	m, err := loadModel(modelPath)
	if err != nil {
		return err
	}

	start := time.Now()
	stats, err := denoise(w, m)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	if err := writeWAV(outPath, w); err != nil {
		return err
	}
	if quiet {
		return nil
	}

	audio := time.Duration(float64(w.frames()) / float64(w.rate) * float64(time.Second))
	fmt.Printf("%s: %d Hz, %s, %s\n", inPath, w.rate, plural(w.channels, "channel"), audio.Round(time.Millisecond))
	fmt.Printf("%d of 32 bands active, %d-sample frames\n", stats.activeBands, stats.frameSize)
	fmt.Printf("mean speech probability %.2f\n", stats.meanVAD)
	fmt.Printf("denoised in %s (%.0f× real time)\n", elapsed.Round(time.Millisecond), audio.Seconds()/elapsed.Seconds())
	fmt.Printf("wrote %s\n", outPath)
	return nil
}

func loadModel(path string) (*rnnoise.Model, error) {
	if path == "" {
		return model.Load()
	}
	return rnnoise.LoadModelFile(path)
}

type stats struct {
	frameSize   int
	activeBands int
	meanVAD     float64
}

// denoise replaces w.samples with the denoised signal, one Denoiser pass per
// channel. Output sample i corresponds to input sample i-Delay(), so each
// channel is run for Delay() samples past the end of the input and the first
// Delay() output samples are discarded, which realigns the two.
func denoise(w *wavFile, m *rnnoise.Model) (stats, error) {
	d, err := rnnoise.New(rnnoise.Options{SampleRate: w.rate, Model: m})
	if err != nil {
		return stats{}, err
	}

	frame, delay := d.FrameSize(), d.Delay()
	n := w.frames()
	nframes := (n + delay + frame - 1) / frame

	out := make([]int16, len(w.samples))
	inBuf := make([]int16, frame)
	outBuf := make([]int16, frame)

	var vadSum float64
	var vadCount int

	for ch := 0; ch < w.channels; ch++ {
		if ch > 0 {
			d.Reset()
		}
		for f := 0; f < nframes; f++ {
			base := f * frame
			for i := range inBuf {
				if src := base + i; src < n {
					inBuf[i] = w.samples[src*w.channels+ch]
				} else {
					inBuf[i] = 0 // pad past the end so the tail flushes
				}
			}
			vad, err := d.ProcessInt16(outBuf, inBuf)
			if err != nil {
				return stats{}, err
			}
			if base < n { // only frames holding real input count towards the mean
				vadSum += float64(vad)
				vadCount++
			}
			for i, v := range outBuf {
				dst := base + i - delay
				if dst < 0 {
					continue
				}
				if dst >= n {
					break
				}
				out[dst*w.channels+ch] = v
			}
		}
	}

	w.samples = out
	return stats{
		frameSize:   frame,
		activeBands: d.ActiveBands(),
		meanVAD:     vadSum / float64(vadCount),
	}, nil
}

func plural(n int, unit string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, unit)
	}
	return fmt.Sprintf("%d %ss", n, unit)
}

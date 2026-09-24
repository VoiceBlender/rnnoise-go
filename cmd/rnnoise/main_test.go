package main

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	rnnoise "github.com/VoiceBlender/rnnoise-go"
	"github.com/VoiceBlender/rnnoise-go/model"
)

// tone builds one second of test audio: noise alone for the first half, then a
// harmonic stack in the same noise. Denoising should flatten the first half and
// leave the second largely intact.
func tone(rate, channels int) *wavFile {
	n := rate
	rng := rand.New(rand.NewSource(7))
	s := make([]int16, n*channels)
	for i := 0; i < n; i++ {
		t := float64(i) / float64(rate)
		var v float64
		if t >= 0.5 {
			for k := 1; k <= 4; k++ {
				v += math.Sin(2*math.Pi*140*float64(k)*t) / float64(k)
			}
			v *= 4000
		}
		for ch := 0; ch < channels; ch++ {
			x := v + rng.NormFloat64()*1500
			s[i*channels+ch] = int16(math.Max(-32768, math.Min(32767, x)))
		}
	}
	return &wavFile{rate: rate, channels: channels, samples: s}
}

func rms(s []int16, channels, ch int, lo, hi int) float64 {
	var acc float64
	var n int
	for i := lo; i < hi; i++ {
		v := float64(s[i*channels+ch])
		acc += v * v
		n++
	}
	return math.Sqrt(acc / float64(n))
}

func TestDenoise(t *testing.T) {
	m, err := model.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ rate, channels int }{
		{8000, 1}, {16000, 1}, {44100, 1}, {48000, 2},
	} {
		w := tone(tc.rate, tc.channels)
		in := append([]int16(nil), w.samples...)
		want := len(in)

		st, err := denoise(w, rnnoise.Options{Model: m})
		if err != nil {
			t.Fatalf("%d Hz: %v", tc.rate, err)
		}
		if len(w.samples) != want {
			t.Errorf("%d Hz: length %d, want %d", tc.rate, len(w.samples), want)
		}
		if st.frameSize != tc.rate/100 {
			t.Errorf("%d Hz: frame size %d, want %d", tc.rate, st.frameSize, tc.rate/100)
		}

		for ch := 0; ch < tc.channels; ch++ {
			// Skip the first 100 ms, where the network is still converging.
			lo, hi := tc.rate/10, tc.rate/2
			noiseIn := rms(in, tc.channels, ch, lo, hi)
			noiseOut := rms(w.samples, tc.channels, ch, lo, hi)
			if atten := 20 * math.Log10(math.Max(noiseOut, 1e-9)/noiseIn); atten > -20 {
				t.Errorf("%d Hz ch%d: noise attenuated only %.1f dB, want at least 20", tc.rate, ch, -atten)
			}

			lo, hi = tc.rate/2+tc.rate/10, tc.rate
			speechIn := rms(in, tc.channels, ch, lo, hi)
			speechOut := rms(w.samples, tc.channels, ch, lo, hi)
			if loss := 20 * math.Log10(math.Max(speechOut, 1e-9)/speechIn); loss < -6 {
				t.Errorf("%d Hz ch%d: speech lost %.1f dB, want at most 6", tc.rate, ch, -loss)
			}
		}
	}
}

// The denoiser has a 20 ms algorithmic delay; denoise compensates for it, so
// the speech onset must land in the same place it did on input.
func TestDenoiseAlignment(t *testing.T) {
	m, err := model.Load()
	if err != nil {
		t.Fatal(err)
	}
	const rate = 16000
	w := tone(rate, 1)
	if _, err := denoise(w, rnnoise.Options{Model: m}); err != nil {
		t.Fatal(err)
	}

	onset := func(s []int16) int {
		win := rate / 100 // 10 ms
		for i := 0; i+win <= len(s); i += win {
			if rms(s, 1, 0, i, i+win) > 2000 {
				return i
			}
		}
		return -1
	}
	got := onset(w.samples)
	want := rate / 2
	if got < 0 {
		t.Fatal("no speech onset found in output")
	}
	if d := got - want; d < -rate/100 || d > rate/100 {
		t.Errorf("speech onset at sample %d, want within 10 ms of %d", got, want)
	}
}

func TestWAVRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rt.wav")
	want := tone(16000, 2)
	if err := writeWAV(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := readWAV(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.rate != want.rate || got.channels != want.channels {
		t.Fatalf("got %d Hz %d ch, want %d Hz %d ch", got.rate, got.channels, want.rate, want.channels)
	}
	if len(got.samples) != len(want.samples) {
		t.Fatalf("got %d samples, want %d", len(got.samples), len(want.samples))
	}
	for i := range got.samples {
		if got.samples[i] != want.samples[i] {
			t.Fatalf("sample %d: got %d, want %d", i, got.samples[i], want.samples[i])
		}
	}
}

func TestReadWAVRejects(t *testing.T) {
	dir := t.TempDir()
	// An 8-bit PCM header, which the tool does not accept.
	b := make([]byte, 44)
	copy(b[0:4], "RIFF")
	binary.LittleEndian.PutUint32(b[4:8], 36)
	copy(b[8:12], "WAVE")
	copy(b[12:16], "fmt ")
	binary.LittleEndian.PutUint32(b[16:20], 16)
	binary.LittleEndian.PutUint16(b[20:22], 1)
	binary.LittleEndian.PutUint16(b[22:24], 1)
	binary.LittleEndian.PutUint32(b[24:28], 16000)
	binary.LittleEndian.PutUint16(b[34:36], 8)
	copy(b[36:40], "data")

	for _, tc := range []struct {
		name string
		body []byte
	}{
		{"not-riff", []byte("this is not a wav file at all")},
		{"eight-bit", b},
	} {
		path := filepath.Join(dir, tc.name+".wav")
		if err := os.WriteFile(path, tc.body, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := readWAV(path); err == nil {
			t.Errorf("%s: readWAV succeeded, want an error", tc.name)
		}
	}
}

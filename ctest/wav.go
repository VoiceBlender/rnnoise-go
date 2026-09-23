package ctest

import (
	"encoding/binary"
	"fmt"
	"os"
)

// Minimal reader for the only format the corpus uses: RIFF/WAVE, 16-bit PCM,
// mono. Anything else is rejected rather than silently misread, because a
// mis-parsed header would show up as a plausible-looking quality regression.

type wavFile struct {
	rate    int
	samples []float32 // int16 scale, matching rnnoise's convention
}

func readWAV(path string) (*wavFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 44 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%s: not a RIFF/WAVE file", path)
	}

	var rate, channels, bits int
	var data []byte
	for off := 12; off+8 <= len(b); {
		id := string(b[off : off+4])
		sz := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		body := off + 8
		if sz < 0 || body+sz > len(b) {
			sz = len(b) - body // tolerate a truncated final chunk
		}
		switch id {
		case "fmt ":
			if sz < 16 {
				return nil, fmt.Errorf("%s: short fmt chunk", path)
			}
			format := int(binary.LittleEndian.Uint16(b[body : body+2]))
			channels = int(binary.LittleEndian.Uint16(b[body+2 : body+4]))
			rate = int(binary.LittleEndian.Uint32(b[body+4 : body+8]))
			bits = int(binary.LittleEndian.Uint16(b[body+14 : body+16]))
			if format != 1 && format != 0xFFFE {
				return nil, fmt.Errorf("%s: WAVE format %d is not PCM", path, format)
			}
		case "data":
			data = b[body : body+sz]
		}
		off = body + sz
		if sz%2 == 1 {
			off++ // chunks are word-aligned
		}
	}
	if data == nil {
		return nil, fmt.Errorf("%s: no data chunk", path)
	}
	if bits != 16 {
		return nil, fmt.Errorf("%s: %d-bit samples, only 16-bit is supported", path, bits)
	}
	if channels != 1 {
		return nil, fmt.Errorf("%s: %d channels, only mono is supported", path, channels)
	}

	n := len(data) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:])))
	}
	return &wavFile{rate: rate, samples: out}, nil
}

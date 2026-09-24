package main

import (
	"encoding/binary"
	"fmt"
	"os"
)

// Minimal RIFF/WAVE support for 16-bit PCM, the only format this tool accepts.
// Anything else is rejected rather than silently misread, since a mis-parsed
// header would sound like a denoising failure.

type wavFile struct {
	rate     int
	channels int
	samples  []int16 // interleaved
}

// frames is the number of sample frames, i.e. samples per channel.
func (w *wavFile) frames() int { return len(w.samples) / w.channels }

func readWAV(path string) (*wavFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%s: not a RIFF/WAVE file", path)
	}

	var rate, channels, bits int
	var data []byte
	var haveFmt bool
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
			haveFmt = true
		case "data":
			data = b[body : body+sz]
		}
		off = body + sz
		if sz%2 == 1 {
			off++ // chunks are word-aligned
		}
	}

	switch {
	case !haveFmt:
		return nil, fmt.Errorf("%s: no fmt chunk", path)
	case data == nil:
		return nil, fmt.Errorf("%s: no data chunk", path)
	case bits != 16:
		return nil, fmt.Errorf("%s: %d-bit samples, only 16-bit PCM is supported", path, bits)
	case channels < 1:
		return nil, fmt.Errorf("%s: %d channels", path, channels)
	}

	n := len(data) / 2
	n -= n % channels // drop a trailing partial sample frame
	s := make([]int16, n)
	for i := range s {
		s[i] = int16(binary.LittleEndian.Uint16(data[2*i:]))
	}
	return &wavFile{rate: rate, channels: channels, samples: s}, nil
}

func writeWAV(path string, w *wavFile) error {
	const headerSize = 44
	dataSize := 2 * len(w.samples)
	blockAlign := 2 * w.channels

	b := make([]byte, headerSize+dataSize)
	copy(b[0:4], "RIFF")
	binary.LittleEndian.PutUint32(b[4:8], uint32(headerSize-8+dataSize))
	copy(b[8:12], "WAVE")
	copy(b[12:16], "fmt ")
	binary.LittleEndian.PutUint32(b[16:20], 16) // PCM fmt chunk size
	binary.LittleEndian.PutUint16(b[20:22], 1)  // PCM
	binary.LittleEndian.PutUint16(b[22:24], uint16(w.channels))
	binary.LittleEndian.PutUint32(b[24:28], uint32(w.rate))
	binary.LittleEndian.PutUint32(b[28:32], uint32(w.rate*blockAlign))
	binary.LittleEndian.PutUint16(b[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(b[34:36], 16) // bits per sample
	copy(b[36:40], "data")
	binary.LittleEndian.PutUint32(b[40:44], uint32(dataSize))
	for i, v := range w.samples {
		binary.LittleEndian.PutUint16(b[headerSize+2*i:], uint16(v))
	}
	return os.WriteFile(path, b, 0o644)
}

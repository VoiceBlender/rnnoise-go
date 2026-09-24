// Package model provides RNNoise's trained weights, embedded in the binary.
//
// It is a separate package so that the weights are linked into a binary only
// if it imports this package: a program that loads a model from disk with
// rnnoise.LoadModelFile does not carry them.
//
// The weights are upstream's, repacked by tools/blobgen. Their provenance,
// pinned hash and licence are recorded in LICENSE.weights.
package model

import (
	_ "embed"
	"sync"

	rnnoise "github.com/VoiceBlender/rnnoise-go"
)

//go:embed weights.bin
var blob []byte

var (
	once   sync.Once
	cached *rnnoise.Model
	cerr   error
)

// Load parses the embedded weights. The result is cached, so repeated calls are
// cheap and every caller shares one read-only Model.
func Load() (*rnnoise.Model, error) {
	once.Do(func() {
		cached, cerr = rnnoise.LoadModelBytes(blob)
	})
	return cached, cerr
}

// MustLoad is Load but panics on failure. The weights are embedded and
// validated at build time, so a failure here means the binary itself is
// corrupt rather than anything a caller can handle.
func MustLoad() *rnnoise.Model {
	m, err := Load()
	if err != nil {
		panic("rnnoise/model: " + err.Error())
	}
	return m
}

// Size reports the embedded blob's size in bytes.
func Size() int { return len(blob) }

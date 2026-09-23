package rnnoise

import (
	"math"
	"testing"
)

// TestRateConfigs pins the derived constants at every rate we care about. The
// 48 kHz row must reproduce upstream's denoise.h verbatim; the others are the
// rate-scaled invariants (pitchFrame == window, pitchBuf == pitchMax+window,
// both even).
func TestRateConfigs(t *testing.T) {
	cases := []struct {
		rate                                int
		frame, pitchMin, pitchMax, pitchBuf int
		activeBands                         int
	}{
		{48000, 480, 60, 768, 1728, 32},
		{44100, 441, 55, 706, 1588, 32},
		{32000, 320, 40, 512, 1152, 32},
		{24000, 240, 30, 384, 864, 29},
		{16000, 160, 20, 256, 576, 26},
		{12000, 120, 15, 192, 432, 23},
		{8000, 80, 10, 128, 288, 20},
	}
	for _, tc := range cases {
		c, err := configForRate(tc.rate)
		if err != nil {
			t.Errorf("rate %d: %v", tc.rate, err)
			continue
		}
		if c.frame != tc.frame {
			t.Errorf("rate %d: frame = %d, want %d", tc.rate, c.frame, tc.frame)
		}
		if c.pitchMin != tc.pitchMin || c.pitchMax != tc.pitchMax || c.pitchBuf != tc.pitchBuf {
			t.Errorf("rate %d: pitch min/max/buf = %d/%d/%d, want %d/%d/%d",
				tc.rate, c.pitchMin, c.pitchMax, c.pitchBuf, tc.pitchMin, tc.pitchMax, tc.pitchBuf)
		}
		if c.activeBands != tc.activeBands {
			t.Errorf("rate %d: activeBands = %d, want %d", tc.rate, c.activeBands, tc.activeBands)
		}

		// Invariants rnn_compute_frame_features depends on.
		if c.pitchFrame != c.window {
			t.Errorf("rate %d: pitchFrame %d != window %d", tc.rate, c.pitchFrame, c.window)
		}
		if c.pitchBuf != c.pitchMax+c.window {
			t.Errorf("rate %d: pitchBuf %d != pitchMax+window %d", tc.rate, c.pitchBuf, c.pitchMax+c.window)
		}
		if c.pitchBuf-c.window-c.pitchMax < 0 {
			t.Errorf("rate %d: pitch buffer cannot hold a full-period lookback", tc.rate)
		}
		if c.pitchMax%2 != 0 || c.pitchBuf%2 != 0 {
			t.Errorf("rate %d: pitchMax/pitchBuf must be even, got %d/%d", tc.rate, c.pitchMax, c.pitchBuf)
		}
		if c.freq != c.frame+1 || c.window != 2*c.frame {
			t.Errorf("rate %d: window/freq inconsistent", tc.rate)
		}

		// Bin spacing is what makes eband20ms rate-invariant.
		if math.Abs(c.binHz-50) > 50*maxBinSpacingError {
			t.Errorf("rate %d: bin spacing %.4f Hz too far from 50", tc.rate, c.binHz)
		}

		// features[64] must land in the trained range at every rate: the
		// reference maps [pitchMin, pitchMax] to [60, 768] by construction.
		lo := 0.01 * (float64(c.pitchMin)*c.pitchScale - 300)
		hi := 0.01 * (float64(c.pitchMax)*c.pitchScale - 300)
		if lo < -2.5 || hi > 4.8 {
			t.Errorf("rate %d: features[64] range [%.3f, %.3f] outside the trained [-2.4, 4.68]",
				tc.rate, lo, hi)
		}
	}
}

// TestRateConfig48kMatchesUpstream states the reference row against the
// literals in upstream's src/denoise.h, so a change to the derivation that
// happens to keep the table self-consistent still fails.
func TestRateConfig48kMatchesUpstream(t *testing.T) {
	c, err := configForRate(48000)
	if err != nil {
		t.Fatal(err)
	}
	const (
		frameSize      = 480  // FRAME_SIZE
		windowSize     = 960  // WINDOW_SIZE
		freqSize       = 481  // FREQ_SIZE
		pitchMinPeriod = 60   // PITCH_MIN_PERIOD
		pitchMaxPeriod = 768  // PITCH_MAX_PERIOD
		pitchFrameSize = 960  // PITCH_FRAME_SIZE
		pitchBufSize   = 1728 // PITCH_BUF_SIZE
	)
	if c.frame != frameSize || c.window != windowSize || c.freq != freqSize {
		t.Errorf("frame/window/freq = %d/%d/%d, upstream %d/%d/%d",
			c.frame, c.window, c.freq, frameSize, windowSize, freqSize)
	}
	if c.pitchMin != pitchMinPeriod || c.pitchMax != pitchMaxPeriod ||
		c.pitchFrame != pitchFrameSize || c.pitchBuf != pitchBufSize {
		t.Errorf("pitch constants = %d/%d/%d/%d, upstream %d/%d/%d/%d",
			c.pitchMin, c.pitchMax, c.pitchFrame, c.pitchBuf,
			pitchMinPeriod, pitchMaxPeriod, pitchFrameSize, pitchBufSize)
	}
	if c.pitchScale != 1 {
		t.Errorf("pitchScale = %v, want exactly 1 at the reference rate", c.pitchScale)
	}
}

func TestRateConfigRejects(t *testing.T) {
	for _, rate := range []int{0, 4000, 7999} {
		if _, err := configForRate(rate); err == nil {
			t.Errorf("configForRate(%d) succeeded, want an error", rate)
		}
	}
}

// TestRateConfigShared checks the cache hands out one immutable config per
// rate, so N concurrent streams do not each rebuild ~12 KB of tables.
func TestRateConfigShared(t *testing.T) {
	a, err := configForRate(16000)
	if err != nil {
		t.Fatal(err)
	}
	b, err := configForRate(16000)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Error("configForRate returned distinct configs for the same rate")
	}
}

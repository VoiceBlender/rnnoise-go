package rnnoise

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
)

// The golden fixture locks in bit-exactness against the C reference without
// needing a C toolchain, the 58 MB model tarball, or gcc at test time.
//
// Its authority comes from how it is produced: `make golden` runs the C scalar
// reference and digests *its* per-stage output, so the committed hashes are the
// C reference's, not the Go port's. TestGoldenStages then checks the Go port
// still reproduces them. The C differential test (TestDiffVsC) remains the
// primary gate; this is what keeps the property under guard in plain CI.

const goldenPath = "testdata/cgolden_48k.json"

// goldenFrames is the length of the checked signal. 600 frames is 6 s at
// 48 kHz, enough to cover the silence gate, voiced content, noise-only
// stretches and a full-scale transient.
const goldenFrames = 600

// stageNames is the canonical order the digests are computed in. Changing it,
// or any stage's contents, invalidates the fixture.
var stageNames = []string{
	"hp", "X.re", "X.im", "Ex", "P.re", "P.im", "Ep", "Exp", "features",
	"conv1_out", "conv2_out", "gru1_state", "gru2_state", "gru3_state",
	"g_raw", "g_final", "gf", "out", "vad", "pitch_gain", "pitch_index", "silence",
}

type goldenFile struct {
	Comment   string            `json:"comment"`
	Upstream  string            `json:"upstream_commit"`
	Variant   string            `json:"c_variant"`
	Rate      int               `json:"sample_rate"`
	Frames    int               `json:"frames"`
	Quantiser string            `json:"quantisation"`
	Digests   map[string]string `json:"stage_sha256"`
}

// stageHasher digests a set of named stages frame by frame, in a fixed order,
// so the C-driven generator and the Go checker agree byte for byte.
type stageHasher struct {
	h   map[string]*digest
	buf []byte
}

type digest struct {
	s interface {
		Write([]byte) (int, error)
		Sum([]byte) []byte
	}
}

func newStageHasher() *stageHasher {
	sh := &stageHasher{h: map[string]*digest{}, buf: make([]byte, 4)}
	for _, n := range stageNames {
		sh.h[n] = &digest{s: sha256.New()}
	}
	return sh
}

func (sh *stageHasher) addVec(name string, v []float32) {
	d := sh.h[name]
	for _, x := range v {
		binary.LittleEndian.PutUint32(sh.buf, math.Float32bits(x))
		d.s.Write(sh.buf)
	}
}

func (sh *stageHasher) addF32(name string, x float32) {
	sh.addVec(name, []float32{x})
}

func (sh *stageHasher) addI32(name string, x int32) {
	d := sh.h[name]
	binary.LittleEndian.PutUint32(sh.buf, uint32(x))
	d.s.Write(sh.buf)
}

func (sh *stageHasher) digests() map[string]string {
	out := map[string]string{}
	for n, d := range sh.h {
		out[n] = hex.EncodeToString(d.s.Sum(nil))
	}
	return out
}

// TestGoldenStages runs the Go port over the fixture signal and checks every
// stage digest against the committed C reference values. Pure Go: no C, no
// model download beyond the committed blob.
func TestGoldenStages(t *testing.T) {
	if !bitExactArch() {
		t.Skip(exactnessSkipReason)
	}
	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skipf("%s not present; generate it with `make golden` (needs the C reference)", goldenPath)
	}
	var want goldenFile
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatalf("parsing %s: %v", goldenPath, err)
	}
	if want.Frames != goldenFrames || want.Rate != 48000 {
		t.Fatalf("%s is for %d frames at %d Hz, this test expects %d at 48000",
			goldenPath, want.Frames, want.Rate, goldenFrames)
	}

	m, err := LoadModelFile("model/weights.bin")
	if err != nil {
		t.Skipf("model/weights.bin not present: %v", err)
	}
	got, err := hashGoStages(m, want.Quantiser)
	if err != nil {
		t.Fatal(err)
	}

	names := append([]string(nil), stageNames...)
	sort.Strings(names)
	var bad []string
	for _, n := range names {
		w, ok := want.Digests[n]
		if !ok {
			t.Errorf("stage %s missing from %s", n, goldenPath)
			continue
		}
		if got[n] != w {
			bad = append(bad, n)
		}
	}
	if len(bad) > 0 {
		t.Errorf("these stages no longer match the committed C reference digests: %v\n"+
			"The port has diverged from upstream %s (%s build). Run `make diff-vs-c` for a\n"+
			"per-value report; regenerate the fixture only if the change was intended.",
			bad, want.Upstream, want.Variant)
	}
}

// hashGoStages runs the Go pipeline over the fixture signal and digests each
// stage. Shared with the generator so both sides hash identically.
func hashGoStages(m *Model, quantiser string) (map[string]string, error) {
	quant := QuantSigned
	if quantiser == "unsigned" {
		quant = QuantUnsigned
	}
	c, err := configForRate(48000)
	if err != nil {
		return nil, err
	}
	d, err := New(Options{SampleRate: 48000, Model: m, Quantization: quant})
	if err != nil {
		return nil, err
	}
	d.enableTrace()

	pcm := makeTestPCM(goldenFrames)
	in := make([]float32, c.frame)
	out := make([]float32, c.frame)
	xr := make([]float32, c.freq)
	xi := make([]float32, c.freq)
	pr := make([]float32, c.freq)
	pi := make([]float32, c.freq)

	sh := newStageHasher()
	for f := 0; f < goldenFrames; f++ {
		base := f * c.frame
		for i := range in {
			in[i] = float32(pcm[base+i])
		}
		vad, err := d.Process(out, in)
		if err != nil {
			return nil, err
		}
		for i := 0; i < c.freq; i++ {
			xr[i], xi[i] = d.X[i].r, d.X[i].i
			pr[i], pi[i] = d.P[i].r, d.P[i].i
		}
		sh.addVec("hp", d.hp)
		sh.addVec("X.re", xr)
		sh.addVec("X.im", xi)
		sh.addVec("Ex", d.Ex)
		sh.addVec("P.re", pr)
		sh.addVec("P.im", pi)
		sh.addVec("Ep", d.Ep)
		sh.addVec("Exp", d.Exp)
		sh.addVec("features", d.features)
		sh.addVec("conv1_out", d.nn.tmp)
		sh.addVec("conv2_out", d.nn.cat[:conv2OutSize])
		sh.addVec("gru1_state", d.rnn.gru1State)
		sh.addVec("gru2_state", d.rnn.gru2State)
		sh.addVec("gru3_state", d.rnn.gru3State)
		sh.addVec("g_raw", d.trc.gRaw)
		sh.addVec("g_final", d.g)
		// gf is undefined on silence frames -- upstream leaves it at a stack
		// initialiser there and never reads it -- so it is only hashed when the
		// network ran. See the note in cdiff_test.go.
		if !d.trc.silence {
			sh.addVec("gf", d.gf)
		}
		sh.addVec("out", out)
		sh.addF32("vad", vad)
		sh.addF32("pitch_gain", d.lastGain)
		sh.addI32("pitch_index", int32(d.lastPeriod))
		silence := int32(0)
		if d.trc.silence {
			silence = 1
		}
		sh.addI32("silence", silence)
	}
	return sh.digests(), nil
}

// hashRefStages digests the same stages from the C reference's records, which is
// what makes the committed fixture the C reference's own output rather than a
// snapshot of the Go port.
func hashRefStages(sr *stageReader) (map[string]string, int, error) {
	sh := newStageHasher()
	n := 0
	for ; n < goldenFrames; n++ {
		ref, err := sr.next()
		if err != nil {
			return nil, n, fmt.Errorf("frame %d: %w", n, err)
		}
		sh.addVec("hp", ref.hp)
		sh.addVec("X.re", ref.xr)
		sh.addVec("X.im", ref.xi)
		sh.addVec("Ex", ref.ex)
		sh.addVec("P.re", ref.pr)
		sh.addVec("P.im", ref.pi)
		sh.addVec("Ep", ref.ep)
		sh.addVec("Exp", ref.exp)
		sh.addVec("features", ref.features)
		sh.addVec("conv1_out", ref.conv1Out)
		sh.addVec("conv2_out", ref.conv2Out)
		sh.addVec("gru1_state", ref.gru1)
		sh.addVec("gru2_state", ref.gru2)
		sh.addVec("gru3_state", ref.gru3)
		sh.addVec("g_raw", ref.gRaw)
		sh.addVec("g_final", ref.gFinal)
		if ref.silence == 0 {
			sh.addVec("gf", ref.gf)
		}
		sh.addVec("out", ref.out)
		sh.addF32("vad", ref.vad)
		sh.addF32("pitch_gain", ref.pitchGain)
		sh.addI32("pitch_index", ref.pitchIndex)
		sh.addI32("silence", ref.silence)
	}
	return sh.digests(), n, nil
}

// TestUpdateGolden regenerates testdata/cgolden_48k.json from the C scalar
// reference. It is inert unless both RNNOISE_UPDATE_GOLDEN and
// RNNOISE_C_STAGEDUMP are set, so a normal run can never silently rewrite the
// fixture it is meant to be checked against.
//
//	make golden
func TestUpdateGolden(t *testing.T) {
	if os.Getenv("RNNOISE_UPDATE_GOLDEN") == "" {
		t.Skip("set RNNOISE_UPDATE_GOLDEN=1 (see make golden)")
	}
	bin := os.Getenv("RNNOISE_C_STAGEDUMP")
	if bin == "" {
		t.Fatal("RNNOISE_C_STAGEDUMP must point at the C scalar reference; " +
			"the fixture is only meaningful if it comes from C")
	}

	sr, cleanup, err := runStageDump(t, bin, goldenFrames)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	digests, n, err := hashRefStages(sr)
	if err != nil {
		t.Fatal(err)
	}
	if n != goldenFrames {
		t.Fatalf("C reference produced %d frames, wanted %d", n, goldenFrames)
	}

	g := goldenFile{
		Comment: "Per-stage sha256 digests of the C reference's own output over " +
			"makeTestPCM(600), used by TestGoldenStages so bit-exactness stays " +
			"under guard without a C toolchain. Regenerate with `make golden`.",
		Upstream:  "70f1d256acd4b34a572f999a05c87bf00b67730d",
		Variant:   "scalar (-U__SSE2__ -U__SSE__ -U__AVX__), signed activations with bias",
		Rate:      48000,
		Frames:    goldenFrames,
		Quantiser: "signed",
		Digests:   digests,
	}
	buf, err := json.MarshalIndent(g, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(goldenPath, append(buf, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s from %s (%d frames, %d stages)", goldenPath, bin, n, len(digests))
}

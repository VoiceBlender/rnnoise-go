// Command blobgen converts upstream's generated src/rnnoise_data.c into the
// little-endian DNNw weight blob this module embeds.
//
// It is a pure-Go lexer of the C source, deliberately: the alternative is to
// compile upstream's write_weights.c, which needs a C toolchain and a 78 MB
// translation unit. Keeping it in Go means `make model` works anywhere, and the
// env-gated TestBlobMatchesUpstreamDumper still cross-checks the output against
// the C dumper byte for byte when a compiler is available.
//
// Two deviations from upstream's blob, both documented in model/LICENSE.weights:
//
//   - The blob is always little-endian. Upstream writes native-endian records
//     with no byte-order marker, which is only safe because it generates and
//     consumes them on the same machine.
//   - The *_weights_float debug duplicates of quantised layers are omitted, as
//     upstream's own -DDISABLE_DEBUG_FLOAT build does. This matters for
//     correctness, not just size: compute_linear prefers float_weights over the
//     int8 weights when both are present, so shipping them would silently
//     select the unquantised path.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

const (
	weightBlobVersion = 0
	weightBlockSize   = 64 // == sizeof(WeightHead)
	sparseBlockSize   = 32 // 8 rows x 4 cols
)

// Weight record types, from src/nnet.h.
const (
	typeFloat   = 0
	typeInt     = 1
	typeQWeight = 2
	typeInt8    = 3
)

func main() {
	in := flag.String("in", "", "path to upstream src/rnnoise_data.c")
	out := flag.String("out", "", "output path")
	tables := flag.String("tables", "", "instead, convert upstream src/rnnoise_tables.c into the test fixture")
	flag.Parse()

	if *out == "" {
		fatal("-out is required")
	}
	if *tables != "" {
		if err := convertTables(*tables, *out); err != nil {
			fatal("%v", err)
		}
		return
	}
	if *in == "" {
		fatal("-in is required")
	}
	if err := convertWeights(*in, *out); err != nil {
		fatal("%v", err)
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "blobgen: "+format+"\n", args...)
	os.Exit(1)
}

var (
	// static const float conv1_weights_float[24960] = { ... };
	reArrayDef = regexp.MustCompile(`static const (float|opus_int8|int) ([A-Za-z0-9_]+)\[(\d+)\] = \{`)
	// #define WEIGHTS_conv1_weights_float_TYPE WEIGHT_TYPE_float
	reTypeDef = regexp.MustCompile(`#define WEIGHTS_([A-Za-z0-9_]+)_TYPE WEIGHT_TYPE_([a-z0-9]+)`)
	// {"conv1_weights_float",  WEIGHTS_..._TYPE, sizeof(...), ...},
	reArrayEntry = regexp.MustCompile(`\{"([A-Za-z0-9_]+)",\s+WEIGHTS_`)
)

type array struct {
	name   string
	ctype  string
	count  int
	typ    int
	offset int // byte offset of the '{' in the source, for the guard check
	body   string
}

func convertWeights(inPath, outPath string) error {
	src, err := os.ReadFile(inPath)
	if err != nil {
		return err
	}
	text := string(src)

	// Regions guarded by #ifndef DISABLE_DEBUG_FLOAT are the float duplicates
	// of quantised layers; upstream's default build omits them and so do we.
	guards := debugFloatRegions(text)

	types := map[string]int{}
	for _, m := range reTypeDef.FindAllStringSubmatch(text, -1) {
		switch m[2] {
		case "float":
			types[m[1]] = typeFloat
		case "int":
			types[m[1]] = typeInt
		case "int8":
			types[m[1]] = typeInt8
		case "qweight":
			types[m[1]] = typeQWeight
		default:
			return fmt.Errorf("unknown weight type %q for %s", m[2], m[1])
		}
	}
	if len(types) == 0 {
		return fmt.Errorf("no WEIGHTS_*_TYPE defines found in %s; is this rnnoise_data.c?", inPath)
	}

	// Collect every array definition, skipping the debug-float ones.
	arrays := map[string]*array{}
	for _, loc := range reArrayDef.FindAllStringSubmatchIndex(text, -1) {
		m := text[loc[0]:loc[1]]
		sub := reArrayDef.FindStringSubmatch(m)
		open := loc[1] - 1 // index of '{'
		if inAnyRegion(guards, open) {
			continue
		}
		end := strings.Index(text[open:], "};")
		if end < 0 {
			return fmt.Errorf("unterminated initialiser for %s", sub[2])
		}
		n, _ := strconv.Atoi(sub[3])
		arrays[sub[2]] = &array{
			name:   sub[2],
			ctype:  sub[1],
			count:  n,
			typ:    types[sub[2]],
			offset: open,
			body:   text[open+1 : open+end],
		}
	}

	// Emit in the order of upstream's rnnoise_arrays[], so a byte-for-byte
	// comparison against the C dumper is meaningful.
	order, err := arrayOrder(text)
	if err != nil {
		return err
	}

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	var emitted int
	for _, name := range order {
		a := arrays[name]
		if a == nil {
			// Either guarded out (a debug-float duplicate) or genuinely absent.
			continue
		}
		payload, err := encode(a)
		if err != nil {
			return err
		}
		if err := writeRecord(w, a.name, a.typ, payload); err != nil {
			return err
		}
		emitted++
		if strings.HasSuffix(a.name, "_weights_idx") {
			if err := checkIdx(a); err != nil {
				return fmt.Errorf("%s: %w", a.name, err)
			}
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}
	st, _ := f.Stat()
	fmt.Printf("blobgen: wrote %d records, %d bytes to %s\n", emitted, st.Size(), outPath)
	return nil
}

func debugFloatRegions(text string) [][2]int {
	var out [][2]int
	const openTok = "#ifndef DISABLE_DEBUG_FLOAT"
	const closeTok = "#endif /*DISABLE_DEBUG_FLOAT*/"
	pos := 0
	for {
		i := strings.Index(text[pos:], openTok)
		if i < 0 {
			return out
		}
		i += pos
		j := strings.Index(text[i:], closeTok)
		if j < 0 {
			return out
		}
		j += i + len(closeTok)
		out = append(out, [2]int{i, j})
		pos = j
	}
}

func inAnyRegion(regions [][2]int, off int) bool {
	for _, r := range regions {
		if off >= r[0] && off < r[1] {
			return true
		}
	}
	return false
}

func arrayOrder(text string) ([]string, error) {
	i := strings.Index(text, "const WeightArray rnnoise_arrays[]")
	if i < 0 {
		return nil, fmt.Errorf("rnnoise_arrays[] not found")
	}
	j := strings.Index(text[i:], "};")
	if j < 0 {
		return nil, fmt.Errorf("rnnoise_arrays[] unterminated")
	}
	block := text[i : i+j]
	var order []string
	for _, m := range reArrayEntry.FindAllStringSubmatch(block, -1) {
		order = append(order, m[1])
	}
	if len(order) == 0 {
		return nil, fmt.Errorf("rnnoise_arrays[] listed no arrays")
	}
	return order, nil
}

// encode converts an initialiser body to its little-endian payload. Floats are
// parsed at float64 then rounded once to float32, exactly as the C compiler
// rounds these decimal literals when storing them into a `static const float`.
func encode(a *array) ([]byte, error) {
	fields := splitValues(a.body)
	if len(fields) != a.count {
		return nil, fmt.Errorf("%s: parsed %d values, declared %d", a.name, len(fields), a.count)
	}
	switch a.ctype {
	case "float":
		buf := make([]byte, 4*len(fields))
		for i, s := range fields {
			v, err := strconv.ParseFloat(s, 32)
			if err != nil {
				return nil, fmt.Errorf("%s[%d]: %w", a.name, i, err)
			}
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(float32(v)))
		}
		return buf, nil
	case "int":
		buf := make([]byte, 4*len(fields))
		for i, s := range fields {
			v, err := strconv.Atoi(s)
			if err != nil {
				return nil, fmt.Errorf("%s[%d]: %w", a.name, i, err)
			}
			binary.LittleEndian.PutUint32(buf[4*i:], uint32(int32(v)))
		}
		return buf, nil
	case "opus_int8":
		buf := make([]byte, len(fields))
		for i, s := range fields {
			v, err := strconv.Atoi(s)
			if err != nil {
				return nil, fmt.Errorf("%s[%d]: %w", a.name, i, err)
			}
			if v < -128 || v > 127 {
				return nil, fmt.Errorf("%s[%d]: %d out of int8 range", a.name, i, v)
			}
			buf[i] = byte(int8(v))
		}
		return buf, nil
	}
	return nil, fmt.Errorf("%s: unhandled C type %q", a.name, a.ctype)
}

func splitValues(body string) []string {
	fields := strings.FieldsFunc(body, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\n' || r == '\t' || r == '\r'
	})
	out := fields[:0]
	for _, f := range fields {
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}

// checkIdx validates a sparse index array and reports whether it is in fact
// fully dense.
//
// The format is, per group of 8 output rows: a block count, then that many
// 4-aligned column positions. The shipped model turns out to be 100% dense and
// strictly monotonic in every group, which is why the dense cgemv8x4 kernel can
// be used unconditionally for it. If a future model is genuinely sparse this
// prints so, and the sparse kernel is required.
func checkIdx(a *array) error {
	vals := splitValues(a.body)
	idx := make([]int, len(vals))
	for i, s := range vals {
		v, err := strconv.Atoi(s)
		if err != nil {
			return err
		}
		idx[i] = v
	}
	pos, groups, blocks := 0, 0, 0
	dense, monotonic := true, true
	for pos < len(idx) {
		nb := idx[pos]
		pos++
		if pos+nb > len(idx) {
			return fmt.Errorf("truncated group at %d", pos)
		}
		prev := -1
		for k := 0; k < nb; k++ {
			c := idx[pos+k]
			if c%4 != 0 {
				return fmt.Errorf("column %d is not 4-aligned", c)
			}
			if c <= prev {
				monotonic = false
			}
			// A fully dense group lists every 4-column block from 0 upward.
			if c != 4*k {
				dense = false
			}
			prev = c
		}
		pos += nb
		groups++
		blocks += nb
	}
	fmt.Printf("blobgen: %s: %d row-groups, %d blocks (%d weights), dense=%v monotonic=%v\n",
		a.name, groups, blocks, blocks*sparseBlockSize, dense, monotonic)
	return nil
}

func writeRecord(w *bufio.Writer, name string, typ int, payload []byte) error {
	if len(name) >= 43 {
		return fmt.Errorf("name %q is too long for the 44-byte field", name)
	}
	var head [weightBlockSize]byte
	copy(head[0:4], "DNNw")
	binary.LittleEndian.PutUint32(head[4:], weightBlobVersion)
	binary.LittleEndian.PutUint32(head[8:], uint32(typ))
	binary.LittleEndian.PutUint32(head[12:], uint32(len(payload)))
	blockSize := (len(payload) + weightBlockSize - 1) / weightBlockSize * weightBlockSize
	binary.LittleEndian.PutUint32(head[16:], uint32(blockSize))
	copy(head[20:], name)
	if _, err := w.Write(head[:]); err != nil {
		return err
	}
	if _, err := w.Write(payload); err != nil {
		return err
	}
	pad := make([]byte, blockSize-len(payload))
	_, err := w.Write(pad)
	return err
}

// convertTables regenerates testdata/upstream_tables_48k.txt from upstream's
// committed src/rnnoise_tables.c. The fixture is checked in so the strongest
// test in the package -- exact float32 equality of every generated table
// against upstream's own -- runs without a C toolchain or a model download.
func convertTables(inPath, outPath string) error {
	src, err := os.ReadFile(inPath)
	if err != nil {
		return err
	}
	text := string(src)

	grab := func(pattern string) (string, error) {
		re := regexp.MustCompile(pattern)
		m := re.FindStringSubmatch(text)
		if m == nil {
			return "", fmt.Errorf("pattern %q not found in %s", pattern, inPath)
		}
		return m[1], nil
	}

	var b strings.Builder
	b.WriteString("# Extracted verbatim from xiph/rnnoise src/rnnoise_tables.c, which upstream\n")
	b.WriteString("# itself generated with dump_rnnoise_tables.c.\n")
	b.WriteString("# Regenerate with: make tables-fixture. Do not edit by hand.\n")

	kfft, err := grab(`(?s)rnn_kfft\s*=\s*\{(.*?)\};`)
	if err != nil {
		return err
	}
	for _, f := range []struct{ key, pat string }{
		{"nfft", `(\d+), /\* nfft \*/`},
		{"scale", `([0-9.eE+-]+)f, /\* scale \*/`},
		{"shift", `(-?\d+), /\* shift \*/`},
	} {
		m := regexp.MustCompile(f.pat).FindStringSubmatch(kfft)
		if m == nil {
			return fmt.Errorf("%s not found in rnn_kfft", f.key)
		}
		fmt.Fprintf(&b, "%s %s\n", f.key, m[1])
	}
	facs := regexp.MustCompile(`\{([0-9,\s]+)\},\s*/\* factors \*/`).FindStringSubmatch(kfft)
	if facs == nil {
		return fmt.Errorf("factors not found in rnn_kfft")
	}
	fmt.Fprintf(&b, "factors %s\n", strings.Join(splitValues(facs[1]), " "))

	for _, s := range []struct{ key, pat string }{
		{"bitrev", `(?s)fft_bitrev\[\d+\]\s*=\s*\{(.*?)\};`},
		{"half_window", `(?s)rnn_half_window\[\]\s*=\s*\{(.*?)\};`},
		{"dct_table", `(?s)rnn_dct_table\[\]\s*=\s*\{(.*?)\};`},
	} {
		body, err := grab(s.pat)
		if err != nil {
			return err
		}
		vals := splitValues(strings.ReplaceAll(body, "f", ""))
		fmt.Fprintf(&b, "%s %s\n", s.key, strings.Join(vals, " "))
	}

	twBody, err := grab(`(?s)fft_twiddles\[\d+\]\s*=\s*\{(.*?)\};`)
	if err != nil {
		return err
	}
	tw := regexp.MustCompile(`\{\s*(-?[0-9.eE+-]+)f,\s*(-?[0-9.eE+-]+)f\s*\}`).FindAllStringSubmatch(twBody, -1)
	if len(tw) == 0 {
		return fmt.Errorf("no twiddles parsed")
	}
	parts := make([]string, 0, 2*len(tw))
	for _, m := range tw {
		parts = append(parts, m[1], m[2])
	}
	fmt.Fprintf(&b, "twiddles %s\n", strings.Join(parts, " "))

	return os.WriteFile(outPath, []byte(b.String()), 0o644)
}

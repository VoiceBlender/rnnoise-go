package rnnoise

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Capacity measurement: how many concurrent streams one machine can carry, per
// sample rate, recorded so later changes can be compared against it.
//
// Two figures are reported per rate, and the difference between them matters.
//
//   - Single-thread cost. One stream, nothing else running. A frame is always
//     10 ms of audio, so 10 ms divided by this is the real-time factor, which is
//     also sessions per core at 100% CPU.
//   - Saturated throughput. One stream per hardware thread, all busy. This is
//     what a loaded server actually gets, and it is far lower per thread: the
//     single-thread figure rides the turbo clock and has the core to itself,
//     while under load the clock drops, SMT siblings share execution units, and
//     every stream is pulling 2.8 MB of weights per frame through a shared L3.
//
// Quoting the single-thread number as capacity overstates a real server by more
// than a factor of two, so plan against the saturated column.
//
// The test is slow and machine-dependent, so it is skipped unless
// RNNOISE_CAPACITY=1. The stored baseline records the CPU it was taken on and
// the regression check is skipped on a different machine rather than producing
// meaningless failures.
const capacityPath = "testdata/capacity.json"

type capacityRow struct {
	Rate           int     `json:"rate"`
	SingleUS       float64 `json:"single_us_per_frame"`
	SingleXRT      float64 `json:"single_xrt"`
	SaturatedFPS   float64 `json:"saturated_frames_per_sec"`
	SessionsPerCPU float64 `json:"sessions_per_cpu"`
	SessionsPerHW  float64 `json:"sessions_per_hw_thread"`
}

type capacityFile struct {
	Comment string        `json:"comment"`
	CPU     string        `json:"cpu"`
	Threads int           `json:"hw_threads"`
	GoVer   string        `json:"go_version"`
	Rows    []capacityRow `json:"rows"`
}

// machineID fingerprints the host so a baseline taken elsewhere is reported
// rather than enforced.
func machineID() (cpu string, threads int) {
	threads = runtime.NumCPU()
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "unknown", threads
	}
	for _, line := range strings.Split(string(b), "\n") {
		if name, val, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(name) == "model name" {
			return strings.TrimSpace(val), threads
		}
	}
	return "unknown", threads
}

func capacityRates() []int { return []int{8000, 12000, 16000, 24000, 32000, 44100, 48000} }

func TestCapacity(t *testing.T) {
	if os.Getenv("RNNOISE_CAPACITY") == "" {
		t.Skip("set RNNOISE_CAPACITY=1 (see make bench-capacity); slow and machine-dependent")
	}
	m, err := LoadModelFile("model/weights.bin")
	if err != nil {
		t.Skipf("model unavailable: %v", err)
	}
	cpu, threads := machineID()
	t.Logf("%s, %d hardware threads, %s", cpu, threads, runtime.Version())

	var rows []capacityRow
	for _, rate := range capacityRates() {
		row := capacityRow{Rate: rate}
		row.SingleUS = measureSingle(t, m, rate)
		row.SingleXRT = 0.010 / (row.SingleUS * 1e-6)
		row.SaturatedFPS = measureSaturated(t, m, rate, threads)
		// A live stream needs 100 frames per second of wall clock.
		row.SessionsPerCPU = row.SaturatedFPS / 100
		row.SessionsPerHW = row.SessionsPerCPU / float64(threads)
		rows = append(rows, row)
	}

	t.Log("")
	t.Logf("%-7s | %10s %9s | %12s %11s %12s", "rate",
		"us/frame", "xRT", "sat frames/s", "sess/CPU", "sess/thread")
	for _, r := range rows {
		t.Logf("%-7d | %10.1f %9.1f | %12.0f %11.0f %12.1f",
			r.Rate, r.SingleUS, r.SingleXRT, r.SaturatedFPS, r.SessionsPerCPU, r.SessionsPerHW)
	}
	t.Log("")

	if os.Getenv("RNNOISE_UPDATE_CAPACITY") != "" {
		writeCapacity(t, cpu, threads, rows)
		return
	}
	compareCapacity(t, cpu, threads, rows)
}

// measureSingle takes the best of several runs, which suppresses scheduler and
// clock noise better than an average does.
func measureSingle(t *testing.T, m *Model, rate int) float64 {
	t.Helper()
	d, err := New(Options{SampleRate: rate, Model: m})
	if err != nil {
		t.Fatal(err)
	}
	n := d.FrameSize()
	src := capacitySignal(rate, n)
	buf := make([]float32, n)
	for i := 0; i < 500; i++ {
		copy(buf, src)
		d.Process(buf, buf)
	}
	best := math.Inf(1)
	const iters = 20000
	for rep := 0; rep < 5; rep++ {
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			copy(buf, src)
			d.Process(buf, buf)
		}
		if dt := time.Since(t0).Seconds() / iters; dt < best {
			best = dt
		}
	}
	return best * 1e6
}

// measureSaturated takes the best of several runs. Throughput on a shared
// machine drifts with thermals and with whatever else the OS is doing, and a
// single run varied by over 20% between attempts -- too noisy for a regression
// gate to mean anything. Best-of-N rejects transient interference the same way
// it does for the single-thread figure.
func measureSaturated(t *testing.T, m *Model, rate, threads int) float64 {
	t.Helper()
	best := 0.0
	for rep := 0; rep < 3; rep++ {
		if fps := saturatedRun(t, m, rate, threads); fps > best {
			best = fps
		}
	}
	return best
}

func saturatedRun(t *testing.T, m *Model, rate, threads int) float64 {
	t.Helper()
	var total int64
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			d, err := New(Options{SampleRate: rate, Model: m})
			if err != nil {
				return
			}
			n := d.FrameSize()
			src := capacitySignal(rate, n)
			buf := make([]float32, n)
			var local int64
			for {
				select {
				case <-stop:
					atomic.AddInt64(&total, local)
					return
				default:
				}
				for i := 0; i < 200; i++ {
					copy(buf, src)
					d.Process(buf, buf)
				}
				local += 200
			}
		}()
	}
	const dur = 2 * time.Second
	time.Sleep(dur)
	close(stop)
	wg.Wait()
	return float64(total) / dur.Seconds()
}

// capacitySignal is voiced-speech-like content that keeps the silence gate open,
// so the measurement reflects the network actually running. Real audio with
// pauses is cheaper, because a gated frame skips the network entirely -- these
// figures are a worst case.
func capacitySignal(rate, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		t := float64(i) / float64(rate)
		var v float64
		for h := 1; h <= 40; h++ {
			f := 140.0 * float64(h)
			if f > float64(rate)/2*0.95 {
				break
			}
			v += 2500 / float64(h) * math.Cos(2*math.Pi*f*t+float64(h))
		}
		x[i] = float32(v)
	}
	return x
}

func writeCapacity(t *testing.T, cpu string, threads int, rows []capacityRow) {
	t.Helper()
	f := capacityFile{
		Comment: "Concurrent-stream capacity per sample rate. Plan against " +
			"sessions_per_cpu, which is measured with every hardware thread busy; " +
			"single_xrt is a single-thread figure that rides the turbo clock and " +
			"overstates a loaded server by more than 2x. Signal is synthetic voiced " +
			"speech that never trips the silence gate, so these are a worst case. " +
			"Regenerate with `make update-capacity`.",
		CPU:     cpu,
		Threads: threads,
		GoVer:   runtime.Version(),
		Rows:    rows,
	}
	buf, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(capacityPath, append(buf, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", capacityPath)
}

func compareCapacity(t *testing.T, cpu string, threads int, rows []capacityRow) {
	t.Helper()
	raw, err := os.ReadFile(capacityPath)
	if err != nil {
		t.Logf("%s not present; run `make update-capacity` to record a baseline", capacityPath)
		return
	}
	var want capacityFile
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatalf("parsing %s: %v", capacityPath, err)
	}
	if want.CPU != cpu || want.Threads != threads {
		t.Logf("baseline was taken on %q with %d threads, this is %q with %d; "+
			"reporting without enforcing", want.CPU, want.Threads, cpu, threads)
		return
	}

	prev := map[int]capacityRow{}
	for _, r := range want.Rows {
		prev[r.Rate] = r
	}
	// With best-of-3 saturated runs the measurement repeats to within a few
	// percent, so 10% catches a real regression without firing on noise.
	const margin = 0.10
	for _, r := range rows {
		p, ok := prev[r.Rate]
		if !ok {
			t.Logf("no baseline for %d Hz", r.Rate)
			continue
		}
		if r.SessionsPerCPU < p.SessionsPerCPU*(1-margin) {
			t.Errorf("%d Hz: capacity regressed from %.0f to %.0f sessions/CPU (%.1f%% down)",
				r.Rate, p.SessionsPerCPU, r.SessionsPerCPU,
				100*(1-r.SessionsPerCPU/p.SessionsPerCPU))
		}
		if r.SingleUS > p.SingleUS*(1+margin) {
			t.Errorf("%d Hz: per-frame cost regressed from %.1f to %.1f us",
				r.Rate, p.SingleUS, r.SingleUS)
		}
		if r.SessionsPerCPU > p.SessionsPerCPU*(1+margin) {
			t.Logf("%d Hz: capacity IMPROVED from %.0f to %.0f sessions/CPU (%+.1f%%); "+
				"re-record with `make update-capacity`",
				r.Rate, p.SessionsPerCPU, r.SessionsPerCPU,
				100*(r.SessionsPerCPU/p.SessionsPerCPU-1))
		}
	}
	fmt.Fprintln(os.Stderr)
}

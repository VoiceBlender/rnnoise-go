SHELL := /bin/bash

.PHONY: tools test bench vet fmt model verify-model tables-fixture ctest-build diff-vs-c golden nativerate-report update-metrics bench-capacity update-capacity bench-c bench-report gen-asm check-asm test-arm64 all

MODEL_SHA  := $(shell cat MODEL_VERSION)
MODEL_URL  := https://media.xiph.org/rnnoise/models/rnnoise_data-$(MODEL_SHA).tar.gz
UPSTREAM   := https://github.com/xiph/rnnoise
UPSTREAM_REV := 70f1d256acd4b34a572f999a05c87bf00b67730d

# Where the C reference is checked out and built. Test-only; gitignored.
VENDOR := ctest/c/vendor

all: test

# --- Toolchain ---------------------------------------------------------------
# Nothing here is `go install`-able: the generators are workspace modules run
# with `go run`, and everything else is a system package. So this warms the
# module cache -- avo's tree is what makes gen-asm fail offline -- and reports
# which system tools are missing and what they gate.
tools:
	@for m in . ctest tools/avogen tools/blobgen; do \
	  (cd $$m && go mod download) || exit 1; \
	done
	@miss=0; for t in \
	  "git|C reference checkout" \
	  "curl|model tarball download" \
	  "tar|model tarball" \
	  "sha256sum|model and blob verification" \
	  "gcc|C reference builds: ctest-build, diff-vs-c, golden, bench-c" \
	  "docker|test-arm64, plus one-time: docker run --privileged --rm tonistiigi/binfmt --install arm64"; \
	do \
	  bin=$${t%%|*}; why=$${t#*|}; \
	  if command -v $$bin >/dev/null 2>&1; then printf '  %-9s ok\n' "$$bin"; \
	  else printf '  %-9s MISSING -- %s\n' "$$bin" "$$why"; miss=1; fi; \
	done; \
	if [ $$miss -ne 0 ]; then \
	  echo; echo "install the missing packages with the distro package manager"; \
	fi

# Unit tests (pure Go, no C toolchain, no model download).
test:
	go test ./...

bench:
	go test -bench . -benchmem .

# --- Capacity and cross-implementation comparison -----------------------------
# How many concurrent streams a machine carries, per sample rate. Slow (about a
# minute) and machine-dependent, so it is gated behind an env var and its stored
# baseline records the CPU it was taken on; a run on different hardware reports
# without enforcing.
#
# Plan against sessions_per_cpu, which is measured with every hardware thread
# busy. The single-thread xRT figure rides the turbo clock and overstates a
# loaded server by more than 2x.
bench-capacity:
	RNNOISE_CAPACITY=1 go test -count=1 -run TestCapacity -v -timeout 30m .

# Re-record testdata/capacity.json. Only when a change in these numbers is meant.
update-capacity:
	RNNOISE_CAPACITY=1 RNNOISE_UPDATE_CAPACITY=1 \
		go test -count=1 -run TestCapacity -v -timeout 30m .

# The C reference's own cost, for the comparison. Both it and the Go harness take
# the best of several runs, so the two are measured the same way.
bench-c: $(VENDOR)/.stamp
	$(MAKE) -C ctest/c bench
	@echo "native C, 48 kHz, $$(nproc) hardware threads:"
	@printf '  avx2   single:     '; ./ctest/c/build/bench_avx2 20000
	@printf '  scalar single:     '; ./ctest/c/build/bench_scalar 5000
	@printf '  avx2   saturated:  '; ./ctest/c/build/benchmt_avx2

# Everything, into BENCHMARKS.md. The WebAssembly column comes from
# VoiceBlender's own denoise package and is recorded there; see BENCHMARKS.md.
bench-report: bench-capacity bench-c
	@echo
	@echo "Recorded results are in BENCHMARKS.md and testdata/capacity.json."

vet:
	go vet ./...
	GOARCH=arm64 go vet ./...

# ARM64 correctness under emulation. The test binary is cross-compiled here and
# executed in an arm64 container, so no arm64 Go toolchain is needed.
#
# Requires Docker with a qemu binfmt handler registered, once per host:
#
#	docker run --privileged --rm tonistiigi/binfmt --install arm64
#
# This validates correctness, including the hand-encoded SDOT kernel, but NOT
# performance: emulation is an order of magnitude slow and its timings mean
# nothing. Benchmarks need real hardware.
test-arm64:
	GOARCH=arm64 CGO_ENABLED=0 go test -c -o rnnoise.arm64.test .
	docker run --rm --platform linux/arm64 -v "$(CURDIR)":/w -w /w alpine:latest \
		./rnnoise.arm64.test -test.v
	rm -f rnnoise.arm64.test

fmt:
	gofmt -l .

# --- Model weights -----------------------------------------------------------
# Fetches the pinned model tarball and regenerates model/weights.bin. The
# shipped blob is committed, so this is only needed to bump MODEL_VERSION.
model: $(VENDOR)/.stamp
	go run ./tools/blobgen -in $(VENDOR)/src/rnnoise_data.c -out model/weights.bin
	sha256sum model/weights.bin | cut -d' ' -f1 > model/weights.sha256

verify-model:
	@echo "$$(cat model/weights.sha256)  model/weights.bin" | sha256sum -c -

# --- C reference -------------------------------------------------------------
# Upstream's autogen.sh needs libtoolize, which is not assumed present, and
# upstream ships no CMakeLists.txt, so ctest/c/Makefile drives gcc directly.
$(VENDOR)/.stamp:
	mkdir -p $(VENDOR)
	git clone --quiet $(UPSTREAM) $(VENDOR) 2>/dev/null || true
	cd $(VENDOR) && git checkout --quiet $(UPSTREAM_REV)
	curl -sSL -o $(VENDOR)/model.tar.gz "$(MODEL_URL)"
	@echo "$(MODEL_SHA)  $(VENDOR)/model.tar.gz" | sha256sum -c -
	tar xzf $(VENDOR)/model.tar.gz -C $(VENDOR) --strip-components=0
	touch $@

ctest-build: $(VENDOR)/.stamp
	$(MAKE) -C ctest/c

# Per-stage differential comparison against the C reference. Builds two C
# variants: a scalar one (the bit-exact oracle for the default QuantSigned
# build) and an AVX2 one (tolerance reference, and the oracle for
# QuantUnsigned).
diff-vs-c: ctest-build
	RNNOISE_C_STAGEDUMP=$(CURDIR)/ctest/c/build/stagedump_scalar \
	RNNOISE_C_STAGEDUMP_AVX2=$(CURDIR)/ctest/c/build/stagedump_avx2 \
		go test -run TestDiffVsC -v .

# Regenerates testdata/cgolden_48k.json: per-stage sha256 digests of the C
# reference's OWN output. TestGoldenStages then guards bit-exactness in plain CI
# with no C toolchain. Only run this when a divergence from upstream is intended.
golden: ctest-build
	RNNOISE_UPDATE_GOLDEN=1 \
	RNNOISE_C_STAGEDUMP=$(CURDIR)/ctest/c/build/stagedump_scalar \
		go test -run TestUpdateGolden -v .

# Native-rate quality report: native versus resample-to-48k-and-back, with the
# metric table. Needs no C toolchain -- both chains are Go, which is the point:
# any difference between them is the native-rate approximation alone.
nativerate-report:
	cd ctest && go test -run TestNativeRateQuality -v -timeout 25m .

# Re-record the quality baseline. Only when a change in these numbers is meant.
update-metrics:
	cd ctest && RNNOISE_UPDATE_METRICS=1 go test -run TestNativeRateQuality -v -timeout 25m .

# --- Generated assembly ------------------------------------------------------
# avo generates the int8 GEMV family only; the other kernels are hand-written
# in the style of the sibling goamr-nb/goamr-wb modules.
gen-asm:
	go run ./tools/avogen -out simd_gen_amd64.s
	gofmt -l .

# CI guard: regenerating must not change the committed assembly. avo records its
# own -out path in the first line of the file, so that line is excluded from the
# comparison; everything below it is what actually matters.
check-asm:
	@tmp=$$(mktemp -d); \
	go run ./tools/avogen -out $$tmp/gen.s; \
	if diff -u <(tail -n +2 simd_gen_amd64.s) <(tail -n +2 $$tmp/gen.s); then \
	  echo "generated assembly is up to date"; rm -rf $$tmp; \
	else \
	  echo "simd_gen_amd64.s is stale; run make gen-asm"; rm -rf $$tmp; exit 1; \
	fi

# Regenerates testdata/upstream_tables_48k.txt from upstream's committed
# src/rnnoise_tables.c. The fixture is committed so plain CI needs no C.
tables-fixture: $(VENDOR)/.stamp
	go run ./tools/blobgen -tables $(VENDOR)/src/rnnoise_tables.c \
		-out testdata/upstream_tables_48k.txt

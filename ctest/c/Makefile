# Builds the C reference in two numerically distinct variants, plus a per-stage
# dumper for each.
#
# Upstream's autogen.sh needs libtoolize, which is not assumed present, and
# upstream ships no CMakeLists.txt, so this drives gcc directly. The source list
# is Makefile.am's librnnoise_la_SOURCES. Deliberately absent:
#
#   -DHAVE_CONFIG_H       so no generated config.h is needed
#   -DRNN_ENABLE_X86_RTCD so x86cpu.c and its cpuid asm stay out, and the
#                         per-object SSE4.1/AVX2 flag dance is unnecessary
#
# -DDISABLE_DEBUG_FLOAT matches upstream's default and is required for
# correctness here, not just size: compute_linear prefers float_weights over the
# int8 weights when both exist, so leaving the debug copies in would silently
# run the quantised layers unquantised.
#
# The two variants:
#
#   scalar  Selects vec.h's portable branch by undefining the x86 SIMD feature
#           macros. That branch does NOT define USE_SU_BIAS, so it quantises
#           activations to signed int8 and uses `bias`, and it evaluates
#           tanh/sigmoid with a real divide. That makes it the bit-exact oracle
#           for the Go port's default QuantSigned mode. GCC still emits SSE2
#           instructions (the x86-64 ABI requires it); only the source path
#           changes. This works because vec.h dispatches on
#           __AVX__ || __SSE2__ and the only x86 header it pulls,
#           x86/x86_arch_macros.h, is entirely #ifdef _MSC_VER.
#
#   avx2    The stock configuration almost everyone actually runs. Defines
#           USE_SU_BIAS, so it quantises to unsigned with `subias`, uses
#           saturating int16 intermediates via VPMADDUBSW, and approximates the
#           activation denominator with _mm256_rcp_ps -- whose result is
#           microarchitecture-defined and differs between AMD and Intel. Bit
#           exactness against it is therefore impossible by construction; it is
#           a tolerance reference, and the oracle for QuantUnsigned.

VENDOR  ?= vendor
BUILD   ?= build
CC      ?= gcc

SRCS := denoise.c rnn.c pitch.c kiss_fft.c celt_lpc.c nnet.c nnet_default.c \
        parse_lpcnet_weights.c rnnoise_data.c rnnoise_tables.c

BASE_CFLAGS := -O2 -I$(VENDOR)/include -I$(VENDOR)/src -I$(VENDOR) \
               -DDISABLE_DEBUG_FLOAT -Wall -Wno-cpp

SCALAR_CFLAGS := $(BASE_CFLAGS) -U__SSE2__ -U__SSE__ -U__AVX__ -DDISABLE_NEON
AVX2_CFLAGS   := $(BASE_CFLAGS) -mavx -mfma -mavx2

SCALAR_OBJS := $(addprefix $(BUILD)/scalar/,$(SRCS:.c=.o))
AVX2_OBJS   := $(addprefix $(BUILD)/avx2/,$(SRCS:.c=.o))

# stagedump replaces denoise.o and rnn.o, which it #includes.
SCALAR_LINK := $(filter-out $(BUILD)/scalar/denoise.o $(BUILD)/scalar/rnn.o,$(SCALAR_OBJS))
AVX2_LINK   := $(filter-out $(BUILD)/avx2/denoise.o $(BUILD)/avx2/rnn.o,$(AVX2_OBJS))

#   avx2_nofma  The same as avx2 but with -ffp-contract=off. Not used by the
#           tests; it exists as the evidence for an attribution that is
#           otherwise easy to get wrong. Comparing scalar against avx2 shows the
#           FFT, band energies and features differing by 1-2 ulp even though
#           none of them has a SIMD variant upstream. Comparing scalar against
#           avx2_nofma shows them bit-identical. So that divergence is GCC
#           contracting multiply-adds into FMA in plain C code -- amplified by
#           nnet_arch.h's own #pragma GCC optimize("tree-vectorize") -- and not
#           a property of the algorithm or of any port of it.

NOFMA_CFLAGS := $(AVX2_CFLAGS) -ffp-contract=off
NOFMA_OBJS   := $(addprefix $(BUILD)/nofma/,$(SRCS:.c=.o))
NOFMA_LINK   := $(filter-out $(BUILD)/nofma/denoise.o $(BUILD)/nofma/rnn.o,$(NOFMA_OBJS))

# bench / benchmt measure the C reference's per-frame cost and its saturated
# throughput, for the sessions-per-CPU comparison against the Go port. They link
# the library proper rather than the stagedump shim, so nothing instruments the
# measured path. denoise.o and rnn.o are only built for these: the stagedump
# targets replace both.
BENCH_OBJS_avx2   := $(AVX2_OBJS)
BENCH_OBJS_scalar := $(SCALAR_OBJS)

bench: $(BUILD)/bench_avx2 $(BUILD)/bench_scalar $(BUILD)/benchmt_avx2

$(BUILD)/bench_avx2: bench.c $(AVX2_OBJS)
	$(CC) $(AVX2_CFLAGS) bench.c $(AVX2_OBJS) -lm -o $@

$(BUILD)/bench_scalar: bench.c $(SCALAR_OBJS)
	$(CC) $(SCALAR_CFLAGS) bench.c $(SCALAR_OBJS) -lm -o $@

$(BUILD)/benchmt_avx2: benchmt.c $(AVX2_OBJS)
	$(CC) $(AVX2_CFLAGS) benchmt.c $(AVX2_OBJS) -lm -lpthread -o $@

.PHONY: all clean probe nofma bench
all: $(BUILD)/stagedump_scalar $(BUILD)/stagedump_avx2
nofma: $(BUILD)/stagedump_avx2_nofma

$(BUILD)/scalar/%.o: $(VENDOR)/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(SCALAR_CFLAGS) -c $< -o $@

$(BUILD)/avx2/%.o: $(VENDOR)/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(AVX2_CFLAGS) -c $< -o $@

$(BUILD)/stagedump_scalar: stagedump.c $(SCALAR_LINK)
	@mkdir -p $(dir $@)
	$(CC) $(SCALAR_CFLAGS) stagedump.c $(SCALAR_LINK) -lm -o $@

$(BUILD)/stagedump_avx2: stagedump.c $(AVX2_LINK)
	@mkdir -p $(dir $@)
	$(CC) $(AVX2_CFLAGS) stagedump.c $(AVX2_LINK) -lm -o $@

$(BUILD)/nofma/%.o: $(VENDOR)/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(NOFMA_CFLAGS) -c $< -o $@

$(BUILD)/stagedump_avx2_nofma: stagedump.c $(NOFMA_LINK)
	@mkdir -p $(dir $@)
	$(CC) $(NOFMA_CFLAGS) stagedump.c $(NOFMA_LINK) -lm -o $@

# probe prints which vec.h branch and which quantisation convention each set of
# flags selects, so a silent change in upstream's dispatch is caught rather than
# quietly invalidating the "bit-exact oracle" claim.
probe:
	@for v in scalar avx2; do \
	  if [ $$v = scalar ]; then FLAGS="$(SCALAR_CFLAGS)"; else FLAGS="$(AVX2_CFLAGS)"; fi; \
	  printf '%-7s ' $$v; \
	  echo '#include "vec.h"' > $(BUILD)/probe.c; \
	  echo '#ifdef USE_SU_BIAS' >> $(BUILD)/probe.c; \
	  echo 'const char *q = "PROBE=unsigned+subias"' >> $(BUILD)/probe.c; \
	  echo '#else' >> $(BUILD)/probe.c; \
	  echo 'const char *q = "PROBE=signed+bias"' >> $(BUILD)/probe.c; \
	  echo '#endif' >> $(BUILD)/probe.c; \
	  echo '#if defined(__AVX2__)' >> $(BUILD)/probe.c; \
	  echo 'const char *p = "PROBE=vec_avx.h/AVX2"' >> $(BUILD)/probe.c; \
	  echo '#elif defined(__SSE2__)' >> $(BUILD)/probe.c; \
	  echo 'const char *p = "PROBE=vec_avx.h/SSE2"' >> $(BUILD)/probe.c; \
	  echo '#else' >> $(BUILD)/probe.c; \
	  echo 'const char *p = "PROBE=vec.h/scalar"' >> $(BUILD)/probe.c; \
	  echo '#endif' >> $(BUILD)/probe.c; \
	  $(CC) $$FLAGS -E $(BUILD)/probe.c 2>/dev/null | grep -o 'PROBE=[^;]*' | tr '\n' ' '; \
	  echo; \
	done

clean:
	rm -rf $(BUILD)

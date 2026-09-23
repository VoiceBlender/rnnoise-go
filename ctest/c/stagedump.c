/* Per-stage reference dumper for the Go port's differential tests.
 *
 * This is the only C we write, and it deliberately contains no algorithm: it
 * includes upstream's denoise.c and rnn.c directly, which makes their statics
 * and the DenoiseState layout visible, then reimplements only the *wiring* of
 * rnnoise_process_frame and compute_rnn so intermediates can be tapped. Every
 * computation still runs upstream's own functions, so a divergence found here
 * is a divergence in the Go port, not in this file.
 *
 * Link it against every other object of the library EXCEPT denoise.o and
 * rnn.o, which it replaces.
 *
 * Usage: stagedump <in.pcm >stages.bin
 *   in.pcm is raw 16-bit host-endian mono PCM at 48 kHz, as
 *   examples/rnnoise_demo.c consumes.
 *
 * The record layout is mirrored by traceRecord in cdiff_test.go. All values are
 * little-endian; the magic doubles as an endianness check.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "denoise.c"
#include "rnn.c"

#define STAGE_MAGIC 0x47545352 /* "RSTG" */

/* Instrumented copy of compute_rnn's wiring. Calls the same primitives in the
 * same order; exists only to expose tmp (conv1 output) and cat (conv2 output).
 */
static void compute_rnn_tapped(const RNNoise *model, RNNState *rnn,
                               float *gains, float *vad, const float *input,
                               int arch, float *conv1_out, float *conv2_out) {
  float tmp[MAX_NEURONS];
  float cat[CONV2_OUT_SIZE + GRU1_OUT_SIZE + GRU2_OUT_SIZE + GRU3_OUT_SIZE];
  compute_generic_conv1d(&model->conv1, tmp, rnn->conv1_state, input,
                         CONV1_IN_SIZE, ACTIVATION_TANH, arch);
  memcpy(conv1_out, tmp, CONV1_OUT_SIZE * sizeof(float));
  compute_generic_conv1d(&model->conv2, cat, rnn->conv2_state, tmp,
                         CONV2_IN_SIZE, ACTIVATION_TANH, arch);
  memcpy(conv2_out, cat, CONV2_OUT_SIZE * sizeof(float));
  compute_generic_gru(&model->gru1_input, &model->gru1_recurrent, rnn->gru1_state, cat, arch);
  compute_generic_gru(&model->gru2_input, &model->gru2_recurrent, rnn->gru2_state, rnn->gru1_state, arch);
  compute_generic_gru(&model->gru3_input, &model->gru3_recurrent, rnn->gru3_state, rnn->gru2_state, arch);
  RNN_COPY(&cat[CONV2_OUT_SIZE], rnn->gru1_state, GRU1_OUT_SIZE);
  RNN_COPY(&cat[CONV2_OUT_SIZE + GRU1_OUT_SIZE], rnn->gru2_state, GRU2_OUT_SIZE);
  RNN_COPY(&cat[CONV2_OUT_SIZE + GRU1_OUT_SIZE + GRU2_OUT_SIZE], rnn->gru3_state, GRU3_OUT_SIZE);
  compute_generic_dense(&model->dense_out, gains, cat, ACTIVATION_SIGMOID, arch);
  compute_generic_dense(&model->vad_dense, vad, cat, ACTIVATION_SIGMOID, arch);
}

static void put_u32(FILE *f, unsigned v) {
  unsigned char b[4] = {v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff};
  fwrite(b, 1, 4, f);
}

static void put_i32(FILE *f, int v) { put_u32(f, (unsigned)v); }

static void put_f32(FILE *f, float v) {
  unsigned u;
  memcpy(&u, &v, 4);
  put_u32(f, u);
}

static void put_f32v(FILE *f, const float *v, int n) {
  int i;
  for (i = 0; i < n; i++) put_f32(f, v[i]);
}

static void put_cpxv(FILE *f, const kiss_fft_cpx *v, int n) {
  int i;
  for (i = 0; i < n; i++) put_f32(f, v[i].r);
  for (i = 0; i < n; i++) put_f32(f, v[i].i);
}

/* Instrumented copy of rnnoise_process_frame's wiring. */
static float process_frame_tapped(DenoiseState *st, float *out, const float *in, FILE *f,
                                  int frame_index) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[FREQ_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float g_raw[NB_BANDS];
  float gf[FREQ_SIZE] = {1};
  float vad_prob = 0;
  float conv1_out[CONV1_OUT_SIZE];
  float conv2_out[CONV2_OUT_SIZE];
  int silence;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};

  rnn_biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = rnn_compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

  RNN_CLEAR(g, NB_BANDS);
  RNN_CLEAR(g_raw, NB_BANDS);
  RNN_CLEAR(conv1_out, CONV1_OUT_SIZE);
  RNN_CLEAR(conv2_out, CONV2_OUT_SIZE);

  if (!silence) {
    compute_rnn_tapped(&st->model, &st->rnn, g, &vad_prob, features, st->arch,
                       conv1_out, conv2_out);
    RNN_COPY(g_raw, g, NB_BANDS);
    rnn_pitch_filter(st->delayed_X, st->delayed_P, st->delayed_Ex, st->delayed_Ep,
                     st->delayed_Exp, g);
    for (i = 0; i < NB_BANDS; i++) {
      float alpha = .6f;
      g[i] = MAX16(g[i], alpha * st->lastg[i]);
      st->lastg[i] = MIN16(1.f, g[i] * (st->delayed_Ex[i] + 1e-3) / (Ex[i] + 1e-3));
    }
    interp_band_gain(gf, g);
    for (i = 0; i < FREQ_SIZE; i++) {
      st->delayed_X[i].r *= gf[i];
      st->delayed_X[i].i *= gf[i];
    }
  }
  frame_synthesis(st, out, st->delayed_X);

  put_u32(f, STAGE_MAGIC);
  put_i32(f, frame_index);
  put_i32(f, silence);
  put_i32(f, st->last_period);
  put_f32(f, st->last_gain);
  put_f32(f, vad_prob);
  put_f32v(f, x, FRAME_SIZE);
  put_cpxv(f, X, FREQ_SIZE);
  put_f32v(f, Ex, NB_BANDS);
  put_cpxv(f, P, FREQ_SIZE);
  put_f32v(f, Ep, NB_BANDS);
  put_f32v(f, Exp, NB_BANDS);
  put_f32v(f, features, NB_FEATURES);
  put_f32v(f, conv1_out, CONV1_OUT_SIZE);
  put_f32v(f, conv2_out, CONV2_OUT_SIZE);
  put_f32v(f, st->rnn.gru1_state, GRU1_OUT_SIZE);
  put_f32v(f, st->rnn.gru2_state, GRU2_OUT_SIZE);
  put_f32v(f, st->rnn.gru3_state, GRU3_OUT_SIZE);
  put_f32v(f, g_raw, NB_BANDS);
  put_f32v(f, g, NB_BANDS);
  put_f32v(f, gf, FREQ_SIZE);
  put_f32v(f, out, FRAME_SIZE);

  RNN_COPY(st->delayed_X, X, FREQ_SIZE);
  RNN_COPY(st->delayed_P, P, FREQ_SIZE);
  RNN_COPY(st->delayed_Ex, Ex, NB_BANDS);
  RNN_COPY(st->delayed_Ep, Ep, NB_BANDS);
  RNN_COPY(st->delayed_Exp, Exp, NB_BANDS);
  return vad_prob;
}

int main(int argc, char **argv) {
  DenoiseState *st;
  short tmp[FRAME_SIZE];
  float x[FRAME_SIZE];
  int i, frame = 0;
  long max_frames = 0; /* 0 = unlimited */

  if (argc > 1) max_frames = atol(argv[1]);

  st = rnnoise_create(NULL);
  if (st == NULL) {
    fprintf(stderr, "stagedump: rnnoise_create failed\n");
    return 1;
  }

  /* Announce the geometry so the reader can verify it agrees. */
  put_u32(stdout, STAGE_MAGIC);
  put_i32(stdout, FRAME_SIZE);
  put_i32(stdout, FREQ_SIZE);
  put_i32(stdout, NB_BANDS);
  put_i32(stdout, NB_FEATURES);
  put_i32(stdout, CONV1_OUT_SIZE);
  put_i32(stdout, CONV2_OUT_SIZE);
  put_i32(stdout, GRU1_OUT_SIZE);

  while (fread(tmp, sizeof(short), FRAME_SIZE, stdin) == FRAME_SIZE) {
    for (i = 0; i < FRAME_SIZE; i++) x[i] = tmp[i];
    process_frame_tapped(st, x, x, stdout, frame);
    frame++;
    if (max_frames && frame >= max_frames) break;
  }
  rnnoise_destroy(st);
  fflush(stdout);
  fprintf(stderr, "stagedump: %d frames\n", frame);
  return 0;
}

/* Per-frame cost of the C reference, for the sessions-per-core comparison.
 *
 * Links the library proper (not the stagedump shim) so it measures
 * rnnoise_process_frame exactly as a real caller would, with no instrumentation
 * in the path.
 *
 * Usage: bench [frames]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "rnnoise.h"

#define FRAME 480

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
  long frames = 20000;
  if (argc > 1) frames = atol(argv[1]);

  DenoiseState *st = rnnoise_create(NULL);
  if (!st) { fprintf(stderr, "rnnoise_create failed\n"); return 1; }

  float x[FRAME];
  /* Speech-like content, so the silence gate never short-circuits the network
     and the measurement reflects real work. */
  for (int i = 0; i < FRAME; i++) {
    double t = i / 48000.0, v = 0;
    for (int h = 1; h <= 40; h++) v += 2500.0 / h * cos(2 * M_PI * 140 * h * t + h);
    x[i] = (float)v;
  }

  /* Warm up: converge the recurrent state and page everything in. */
  for (int i = 0; i < 500; i++) { float y[FRAME]; memcpy(y, x, sizeof x); rnnoise_process_frame(st, y, y); }

  double best = 1e30;
  for (int rep = 0; rep < 5; rep++) {
    double t0 = now_sec();
    for (long i = 0; i < frames; i++) {
      float y[FRAME];
      memcpy(y, x, sizeof x);
      rnnoise_process_frame(st, y, y);
    }
    double dt = (now_sec() - t0) / frames;
    if (dt < best) best = dt;
  }
  rnnoise_destroy(st);

  printf("%.1f us/frame   %.1f xRT\n", best * 1e6, 0.010 / best);
  return 0;
}

/* Saturated throughput of the C reference: one denoiser per thread, every
 * hardware thread busy, which is what a loaded server actually sees. A
 * single-threaded figure rides the turbo clock and overstates capacity. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include "rnnoise.h"
#define FRAME 480

static volatile int running = 1;
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+1e-9*ts.tv_nsec;}

typedef struct { long frames; } worker;

static void *run(void *arg) {
  worker *w = arg;
  DenoiseState *st = rnnoise_create(NULL);
  float x[FRAME];
  for (int i = 0; i < FRAME; i++) {
    double t = i/48000.0, v = 0;
    for (int h = 1; h <= 40; h++) v += 2500.0/h*cos(2*M_PI*140*h*t+h);
    x[i] = (float)v;
  }
  float y[FRAME];
  long n = 0;
  while (running) {
    for (int i = 0; i < 200; i++) { memcpy(y, x, sizeof x); rnnoise_process_frame(st, y, y); }
    n += 200;
  }
  w->frames = n;
  rnnoise_destroy(st);
  return NULL;
}

/* Best of several runs, matching the Go harness's method. Saturated throughput
   on a shared machine drifts with thermals and with whatever else is running --
   a single pass varied by 12% between attempts -- and comparing a best-of-N
   figure against a single-pass one would make the comparison meaningless in
   whichever direction the luck fell. */
static double one_run(int threads) {
  pthread_t *tid = calloc(threads, sizeof *tid);
  worker *ws = calloc(threads, sizeof *ws);
  running = 1;
  double t0 = now_sec();
  for (int i = 0; i < threads; i++) pthread_create(&tid[i], NULL, run, &ws[i]);
  sleep(2);
  running = 0;
  long total = 0;
  for (int i = 0; i < threads; i++) { pthread_join(tid[i], NULL); total += ws[i].frames; }
  double dt = now_sec() - t0;
  free(tid); free(ws);
  return total / dt;
}

int main(int argc, char **argv) {
  int threads = (argc > 1) ? atoi(argv[1]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
  double fps = 0;
  for (int rep = 0; rep < 3; rep++) {
    double v = one_run(threads);
    if (v > fps) fps = v;
  }
  printf("%d threads: %.0f frames/s  %.1f us/frame/thread  %.0f sess/chip  %.1f sess/thread\n",
         threads, fps, 1e6/fps*threads, fps/100, fps/100/threads);
  return 0;
}

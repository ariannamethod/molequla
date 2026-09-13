/* mel.c — see mel.h. */

#include "mel.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_FFT       400
#define HOP         160
#define SAMPLE_RATE 16000
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* One table of 400 sin/cos values shared by every transform length, indexed at
 * stride 400/N. The recursion below only ever visits N that divide 400, so the
 * index is exact; a table per length would be the same numbers computed twice. */
static float g_sin[N_FFT], g_cos[N_FFT], g_hann[N_FFT];
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void mel_tables_init(void) {
    for (int i = 0; i < N_FFT; i++) {
        double theta = (2.0 * M_PI * i) / N_FFT;
        g_sin[i] = sinf((float) theta);
        g_cos[i] = cosf((float) theta);
        /* periodic Hann: divisor is the length, not length-1 */
        g_hann[i] = (float) (0.5 * (1.0 - cosf((float) ((2.0 * M_PI * i) / N_FFT))));
    }
}

/* Naive DFT of a real input, used at the odd length the recursion bottoms out on
 * (400 -> 200 -> 100 -> 50 -> 25). Output is interleaved re/im. */
static void dft(const float *in, int N, float *out) {
    const int step = N_FFT / N;
    for (int k = 0; k < N; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            int idx = (k * n * step) % N_FFT;
            re += in[n] * g_cos[idx];
            im -= in[n] * g_sin[idx];
        }
        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

/* Radix-2 Cooley-Tukey over a real input, scratch taken from the tail of `in`
 * (which is why the caller sizes fft_in at 2*N) and of `out`. */
static void fft_rec(float *in, int N, float *out) {
    if (N == 1) { out[0] = in[0]; out[1] = 0; return; }
    const int half = N / 2;
    if (N - half * 2 == 1) { dft(in, N, out); return; }

    float *even = in + N;
    for (int i = 0; i < half; i++) even[i] = in[2 * i];
    float *even_fft = out + 2 * N;
    fft_rec(even, half, even_fft);

    float *odd = even;
    for (int i = 0; i < half; i++) odd[i] = in[2 * i + 1];
    float *odd_fft = even_fft + N;
    fft_rec(odd, half, odd_fft);

    const int step = N_FFT / N;
    for (int k = 0; k < half; k++) {
        int idx = k * step;
        float re = g_cos[idx], im = -g_sin[idx];
        float re_odd = odd_fft[2 * k + 0], im_odd = odd_fft[2 * k + 1];
        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

typedef struct {
    int          ith, n_threads;
    const float *padded;     /* signal with both pads applied */
    int          n_valid;    /* samples that may be non-zero: n_samples + n_fft/2 */
    const float *filters;
    int          n_fft_bins;
    ears_mel    *mel;
} mel_job;

static void *mel_worker(void *arg) {
    mel_job *j = (mel_job *) arg;
    ears_mel *mel = j->mel;
    const int n_bins = j->n_fft_bins;

    float *fft_in  = (float *) calloc((size_t) N_FFT * 2, sizeof(float));
    float *fft_out = (float *) calloc((size_t) N_FFT * 2 * 2 * 2, sizeof(float));
    if (!fft_in || !fft_out) { free(fft_in); free(fft_out); return (void *) -1; }

    int i = j->ith;
    int last = j->n_valid / HOP + 1;
    if (last > mel->n_len) last = mel->n_len;

    for (; i < last; i += j->n_threads) {
        const int offset = i * HOP;
        int n = N_FFT;
        if (j->n_valid - offset < n) n = j->n_valid - offset;
        for (int t = 0; t < n; t++) fft_in[t] = g_hann[t] * j->padded[offset + t];
        for (int t = n; t < N_FFT; t++) fft_in[t] = 0.0f;

        fft_rec(fft_in, N_FFT, fft_out);

        /* |X_k|^2 written back over the first half of fft_out, in place */
        for (int k = 0; k < n_bins; k++)
            fft_out[k] = fft_out[2 * k + 0] * fft_out[2 * k + 0]
                       + fft_out[2 * k + 1] * fft_out[2 * k + 1];

        for (int b = 0; b < mel->n_mel; b++) {
            const float *fr = j->filters + (size_t) b * n_bins;
            double sum = 0.0;
            int k = 0;
            for (; k < n_bins - 3; k += 4)
                sum += fft_out[k + 0] * fr[k + 0] + fft_out[k + 1] * fr[k + 1]
                     + fft_out[k + 2] * fr[k + 2] + fft_out[k + 3] * fr[k + 3];
            for (; k < n_bins; k++) sum += fft_out[k] * fr[k];
            if (sum < 1e-10) sum = 1e-10;
            mel->data[(size_t) b * mel->n_len + i] = (float) log10(sum);
        }
    }

    /* Frames past the audio are all-zero input: log10(1e-10) without the transform. */
    const float floor_v = (float) log10(1e-10);
    for (; i < mel->n_len; i += j->n_threads)
        for (int b = 0; b < mel->n_mel; b++)
            mel->data[(size_t) b * mel->n_len + i] = floor_v;

    free(fft_in);
    free(fft_out);
    return NULL;
}

void ears_mel_free(ears_mel *m) {
    if (!m) return;
    free(m->data);
    m->data = NULL;
    m->n_mel = m->n_len = m->n_len_org = 0;
}

int ears_mel_compute(const float *pcm, int n_samples,
                     const float *filters, int n_mel, int n_fft_bins,
                     int n_threads, ears_mel *out) {
    if (!pcm || !filters || !out || n_samples < 0 || n_mel <= 0) return -1;
    if (n_fft_bins != 1 + N_FFT / 2) return -1;   /* filters must be bin_0..nyquist */
    if (n_threads < 1) n_threads = 1;

    pthread_once(&g_once, mel_tables_init);

    /* Padding, in the order whisper.cpp applies it: 200 zeros then the signal then
     * 30 s of zeros then 200 more, and the leading 200 overwritten by a reflection
     * of the signal's start. The reflection count is clamped to n_samples-1 so a
     * clip shorter than the pad does not read past its own end. */
    const int pad_half = N_FFT / 2;                  /* 200 */
    const int pad_tail = SAMPLE_RATE * 30;           /* 480000 */
    const size_t n_padded = (size_t) n_samples + pad_tail + 2 * (size_t) pad_half;

    float *padded = (float *) calloc(n_padded, sizeof(float));
    if (!padded) return -1;
    memcpy(padded + pad_half, pcm, (size_t) n_samples * sizeof(float));

    int n_reflect = n_samples - 1;
    if (n_reflect > pad_half) n_reflect = pad_half;
    if (n_reflect < 0) n_reflect = 0;
    for (int i = 0; i < n_reflect; i++)
        padded[pad_half - n_reflect + i] = pcm[n_reflect - i];

    memset(out, 0, sizeof(*out));
    out->n_mel     = n_mel;
    out->n_len     = (int) ((n_padded - N_FFT) / HOP);
    out->n_len_org = 1 + (n_samples + pad_half - N_FFT) / HOP;
    out->data = (float *) malloc((size_t) n_mel * out->n_len * sizeof(float));
    if (!out->data) { free(padded); return -1; }

    const int n_valid = n_samples + pad_half;

    mel_job *jobs = (mel_job *) calloc((size_t) n_threads, sizeof(mel_job));
    pthread_t *th = (pthread_t *) calloc((size_t) n_threads, sizeof(pthread_t));
    if (!jobs || !th) { free(jobs); free(th); free(padded); ears_mel_free(out); return -1; }

    int launched = 0;
    for (int t = 0; t < n_threads; t++) {
        jobs[t] = (mel_job) { t, n_threads, padded, n_valid, filters, n_fft_bins, out };
        if (t == 0) continue;
        /* A stripe whose thread will not start is run below on this thread rather
         * than dropped: every stripe must be written or the mel has holes. */
        if (pthread_create(&th[t], NULL, mel_worker, &jobs[t]) != 0) break;
        launched = t;
    }
    void *rc0 = mel_worker(&jobs[0]);
    for (int t = 1; t <= launched; t++) pthread_join(th[t], NULL);
    /* stripes whose thread never started */
    for (int t = launched + 1; t < n_threads; t++) mel_worker(&jobs[t]);

    free(jobs); free(th); free(padded);
    if (rc0) { ears_mel_free(out); return -1; }

    /* Clamp and normalise against the global maximum, which is taken over every
     * frame including the 30 s tail — those sit at log10(1e-10) and never win. */
    const size_t n = (size_t) n_mel * out->n_len;
    double mmax = -1e20;
    for (size_t i = 0; i < n; i++) if (out->data[i] > mmax) mmax = out->data[i];
    mmax -= 8.0;
    for (size_t i = 0; i < n; i++) {
        double v = out->data[i];
        if (v < mmax) v = mmax;
        out->data[i] = (float) ((v + 4.0) / 4.0);
    }
    return 0;
}

/* ── audio in ─────────────────────────────────────────────────────────────── */

static uint32_t le32(const unsigned char *p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

int ears_read_wav(const char *path, float **pcm, int *n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ears: cannot open wav '%s'\n", path); return -1; }
    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fprintf(stderr, "ears: '%s' is not RIFF/WAVE\n", path); fclose(f); return -1;
    }
    int channels = 0, rate = 0, bits = 0;
    unsigned char ch[8];
    while (fread(ch, 1, 8, f) == 8) {
        const uint32_t sz = le32(ch + 4);
        if (!memcmp(ch, "fmt ", 4)) {
            unsigned char b[40];
            const uint32_t want = sz < sizeof(b) ? sz : (uint32_t) sizeof(b);
            if (fread(b, 1, want, f) != want) break;
            if (sz > want) fseek(f, (long) (sz - want), SEEK_CUR);
            if (sz & 1) fseek(f, 1, SEEK_CUR);
            channels = b[2] | (b[3] << 8);
            rate     = (int) le32(b + 4);
            bits     = b[14] | (b[15] << 8);
        } else if (!memcmp(ch, "data", 4)) {
            if (rate != 16000 || bits != 16 || channels < 1) {
                fprintf(stderr, "ears: '%s' is %d Hz %d ch %d bit; v1 needs 16 kHz 16-bit mono\n",
                        path, rate, channels, bits);
                fclose(f); return -1;
            }
            const size_t n16 = sz / 2;
            int16_t *raw = (int16_t *) malloc(n16 * sizeof(int16_t));
            if (!raw) { fclose(f); return -1; }
            if (fread(raw, 2, n16, f) != n16) { free(raw); fclose(f); return -1; }
            const size_t n = n16 / (size_t) channels;
            float *out = (float *) malloc(n * sizeof(float));
            if (!out) { free(raw); fclose(f); return -1; }
            /* whisper.cpp's reader averages nothing: for multi-channel input it
             * takes channel 0, and every wav here is mono anyway. */
            for (size_t i = 0; i < n; i++) out[i] = raw[i * (size_t) channels] / 32768.0f;
            free(raw); fclose(f);
            *pcm = out;
            *n_samples = (int) n;
            return 0;
        } else {
            fseek(f, (long) (sz + (sz & 1)), SEEK_CUR);
        }
    }
    fprintf(stderr, "ears: '%s' has no data chunk\n", path);
    fclose(f);
    return -1;
}

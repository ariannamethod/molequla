/* mel.c — see mel.h. */

#include "mel.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_FFT       400
#define HOP         160
#define SAMPLE_RATE 16000

/* whisper pads the signal out to a fixed 30 s window before counting frames, and
 * every one of those constants belongs to the model rather than to the transform,
 * which is why they are applied here and not inside nt_logmel. */
int ears_mel_compute(const float *pcm, int n_samples,
                     const float *filters, int n_mel, int n_fft_bins,
                     int n_threads, ears_mel *out) {
    return nt_logmel(out, pcm, n_samples, filters, n_mel, n_fft_bins,
                     N_FFT, HOP, SAMPLE_RATE * 30, n_threads);
}

void ears_mel_free(ears_mel *m) { nt_mel_free(m); }

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

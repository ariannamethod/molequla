/* mel.h — log-mel spectrogram front end.
 *
 * The one block of the forward path notorch does not cover: there is no fft, stft,
 * hann or mel symbol in notorch.h or notorch_vision.h, so this is written from
 * scratch. It decides parity — every later stage consumes its output — so the
 * arithmetic here follows `whisper.cpp:3046-3240` step for step, including the
 * choice of a real-input Cooley-Tukey recursion falling back to a naive DFT at odd
 * lengths and the 400-entry sin/cos table both of them index.
 *
 * 16 kHz mono, n_fft 400, hop 160, periodic Hann, 80 mel bins from the filter bank
 * stored in the model file, log10 with a 1e-10 floor, clamp to (global max - 8),
 * then (x + 4) / 4.
 *
 * Layout is mel-major, data[j * n_len + i] for bin j and frame i — the layout
 * whisper.cpp's encoder reads, so a window is a contiguous stride per bin.
 */

#ifndef EARS_MEL_H
#define EARS_MEL_H

typedef struct {
    int    n_mel;      /* 80 */
    int    n_len;      /* frames over the 30 s-padded signal */
    int    n_len_org;  /* frames that carry real audio */
    float *data;       /* [n_mel][n_len] */
} ears_mel;

/* pcm: n_samples mono f32 in [-1, 1] at 16 kHz.
 * filters: [n_mel][n_fft_bins] from the model file (80 x 201).
 * n_threads splits the frame loop; results do not depend on it.
 * Returns 0 and fills `out` (caller calls ears_mel_free), -1 on failure. */
int  ears_mel_compute(const float *pcm, int n_samples,
                      const float *filters, int n_mel, int n_fft_bins,
                      int n_threads, ears_mel *out);

void ears_mel_free(ears_mel *m);

/* 16 kHz mono s16 RIFF -> f32 in [-1, 1]. Lives beside the mel because it is the
 * same stage: the only audio ingestion the organ has. Caller frees *pcm. */
int ears_read_wav(const char *path, float **pcm, int *n_samples);

#endif /* EARS_MEL_H */

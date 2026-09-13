/* mel.h — log-mel spectrogram front end.
 *
 * The transform is no longer here. Window, real FFT, power spectrum, filter-bank
 * matmul, log10, clamp and normalise now live in notorch as `nt_logmel`
 * (notorch.h, AUDIO OPS) — they were written in this file first because notorch
 * had no fft, stft, hann or mel symbol at all, and that was the only block of the
 * forward path it did not cover. They were moved upstream unchanged, so what used
 * to be measured here is now measured there: bit-identical to
 * `whisper_pcm_to_mel`, max|d| = 0.
 *
 * What stays is this file's own business. `ears_mel_compute` fixes whisper's
 * constants — 16 kHz, n_fft 400, hop 160, a 30 s padded window — because those
 * describe this model and not a spectrogram. `ears_read_wav` stays because notorch
 * has no audio I/O and should not grow one to read a RIFF header.
 *
 * Layout is mel-major, data[j * n_len + i] for bin j and frame i — the layout
 * whisper.cpp's encoder reads, so a window is a contiguous stride per bin.
 */

#ifndef EARS_MEL_H
#define EARS_MEL_H

#include <ariannamethod/notorch.h>

/* n_mel, n_len, n_len_org, data — notorch's, under this engine's name. */
typedef nt_mel ears_mel;

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

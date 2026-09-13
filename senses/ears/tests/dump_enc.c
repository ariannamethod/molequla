/* dump_enc — ears' encoder output for the first 30 s window, in the oracle's format.
 *
 *   dump_enc <model.bin> <wav> <out.f32> [threads]
 *
 * out.f32: i32 n_ctx, i32 n_state, then n_ctx*n_state f32, one row per audio frame.
 *
 * EARS_ENC_BREAK skips the stored positional embedding, which is the specific
 * mistake PORT_NOTES warns about (`encoder.positional_embedding` is a weight, not
 * a sinusoid to synthesise). That is how the encoder gate is shown going red.
 */

#include "../ears.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: dump_enc <model.bin> <wav> <out.f32> [threads]\n");
        return 2;
    }
    const int n_threads = (argc == 5) ? atoi(argv[4]) : 4;

    ears_model *m = ears_model_load(argv[1]);
    if (!m) return 1;

    float *pcm = NULL; int n = 0;
    if (ears_read_wav(argv[2], &pcm, &n) != 0) { ears_model_free(m); return 1; }

    ears_mel mel;
    if (ears_mel_compute(pcm, n, m->f->filters, m->f->n_mel, m->f->n_fft, n_threads, &mel) != 0) {
        fprintf(stderr, "dump_enc: mel failed\n"); free(pcm); ears_model_free(m); return 1;
    }
    free(pcm);

    if (getenv("EARS_ENC_BREAK")) {
        fprintf(stderr, "dump_enc: EARS_ENC_BREAK — positional embedding zeroed on purpose\n");
        for (long i = 0; i < (long) m->h.n_audio_ctx * m->h.n_audio_state; i++) m->e_pe[i] = 0.0f;
    }

    const int NC = m->h.n_audio_ctx, NS = m->h.n_audio_state;
    float *enc = (float *) malloc((size_t) NC * NS * sizeof(float));
    if (!enc || ears_encode(m, &mel, 0, n_threads, enc) != 0) {
        fprintf(stderr, "dump_enc: encode failed\n");
        free(enc); ears_mel_free(&mel); ears_model_free(m); return 1;
    }

    FILE *o = fopen(argv[3], "wb");
    if (!o) { fprintf(stderr, "dump_enc: cannot write %s\n", argv[3]); free(enc); ears_mel_free(&mel); ears_model_free(m); return 1; }
    fwrite(&NC, sizeof(int), 1, o);
    fwrite(&NS, sizeof(int), 1, o);
    fwrite(enc, sizeof(float), (size_t) NC * NS, o);
    fclose(o);
    fprintf(stderr, "dump_enc: n_ctx=%d n_state=%d -> %s\n", NC, NS, argv[3]);

    free(enc); ears_mel_free(&mel); ears_model_free(m);
    return 0;
}

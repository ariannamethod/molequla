/* dump_mel — ears' own log-mel for a wav, in the oracle's dump format.
 *
 *   dump_mel <model.bin> <wav> <out.f32> [threads]
 *
 * out.f32: i32 n_mel, i32 n_len, i32 n_len_org, then n_mel*n_len f32, mel-major —
 * byte for byte what harness/oracle_dump writes, so cmp_f32 can diff them
 * directly and a shape disagreement fails on the header rather than on the data.
 *
 * EARS_MEL_BREAK is read here and nowhere else: set it to a frequency-bin index
 * and that bin's power is zeroed before the filter bank, which is how the mel gate
 * is shown going red. A gate never seen red is decoration.
 */

#include "../ears.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: dump_mel <model.bin> <wav> <out.f32> [threads]\n");
        return 2;
    }
    const int n_threads = (argc == 5) ? atoi(argv[4]) : 4;

    gb_file *f = gb_open(argv[1]);
    if (!f) return 1;

    float *pcm = NULL; int n = 0;
    if (ears_read_wav(argv[2], &pcm, &n) != 0) { gb_close(f); return 1; }

    const float *filters = f->filters;
    float *broken = NULL;
    const char *brk = getenv("EARS_MEL_BREAK");
    if (brk) {
        /* Zero one column of the filter bank: a deliberate, localised corruption of
         * the one block the gate is there to protect. */
        const int bin = atoi(brk);
        broken = (float *) malloc((size_t) f->n_mel * f->n_fft * sizeof(float));
        if (!broken) { free(pcm); gb_close(f); return 1; }
        memcpy(broken, filters, (size_t) f->n_mel * f->n_fft * sizeof(float));
        if (bin >= 0 && bin < f->n_fft)
            for (int b = 0; b < f->n_mel; b++) broken[(size_t) b * f->n_fft + bin] = 0.0f;
        fprintf(stderr, "dump_mel: EARS_MEL_BREAK=%d — filter bin zeroed on purpose\n", bin);
        filters = broken;
    }

    ears_mel mel;
    if (ears_mel_compute(pcm, n, filters, f->n_mel, f->n_fft, n_threads, &mel) != 0) {
        fprintf(stderr, "dump_mel: mel failed\n");
        free(pcm); free(broken); gb_close(f); return 1;
    }
    free(pcm); free(broken);

    FILE *o = fopen(argv[3], "wb");
    if (!o) { fprintf(stderr, "dump_mel: cannot write %s\n", argv[3]); ears_mel_free(&mel); gb_close(f); return 1; }
    fwrite(&mel.n_mel, sizeof(int), 1, o);
    fwrite(&mel.n_len, sizeof(int), 1, o);
    fwrite(&mel.n_len_org, sizeof(int), 1, o);
    fwrite(mel.data, sizeof(float), (size_t) mel.n_mel * mel.n_len, o);
    fclose(o);
    fprintf(stderr, "dump_mel: n_mel=%d n_len=%d n_len_org=%d -> %s\n",
            mel.n_mel, mel.n_len, mel.n_len_org, argv[3]);

    ears_mel_free(&mel);
    gb_close(f);
    return 0;
}

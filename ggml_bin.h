/* ggml_bin.h — reader for the legacy flat ggml `.bin` Whisper model file.
 *
 * Whisper models are not GGUF. `src/whisper.cpp:1507` checks GGML_FILE_MAGIC
 * (0x67676d6c, ascii "ggml") and walks a flat little-endian stream:
 *
 *   u32  magic
 *   i32  n_vocab n_audio_ctx n_audio_state n_audio_head n_audio_layer
 *   i32  n_text_ctx n_text_state n_text_head n_text_layer n_mels ftype
 *   i32  n_mel, i32 n_fft, f32 filters[n_mel * n_fft]
 *   i32  n_vocab_tok, then per token: i32 len, char bytes[len]
 *   tensors to EOF: i32 n_dims, i32 name_len, i32 ttype,
 *                   i32 ne[n_dims], char name[name_len], payload
 *
 * ttype 0 = f32, 1 = f16. ne[0] varies fastest, so a weight listed [in, out] is
 * stored row-major as [out][in] — a PyTorch Linear weight.
 *
 * The whole file is read once into one heap block and every tensor is a pointer
 * into it: 77 MB for tiny, 148 MB for base, no second copy and no per-tensor
 * allocation to leak.
 */

#ifndef EARS_GGML_BIN_H
#define EARS_GGML_BIN_H

#include <stdint.h>
#include <stddef.h>

#define GGML_BIN_MAGIC 0x67676d6cu
#define GGML_BIN_F32   0
#define GGML_BIN_F16   1

typedef struct {
    char      name[64];
    int       ttype;            /* GGML_BIN_F32 | GGML_BIN_F16 */
    int       ndim;
    int       ne[4];            /* ne[0] fastest-varying */
    int64_t   nelem;
    const void *data;           /* into gb_file::blob */
} gb_tensor;

typedef struct {
    int n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer;
    int n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype;
} gb_hparams;

typedef struct {
    unsigned char *blob;        /* the file, verbatim */
    size_t         blob_len;

    gb_hparams hparams;

    int          n_mel, n_fft;  /* mel filter bank geometry (80 x 201) */
    const float *filters;       /* [n_mel][n_fft] */

    int          n_tok;         /* tokens stored in the file (50257 here) */
    const char **tok;           /* not NUL-terminated: use tok_len */
    int         *tok_len;

    gb_tensor *tensors;
    int        n_tensors;
} gb_file;

/* Read `path` whole. Returns NULL and prints why on failure. */
gb_file *gb_open(const char *path);
void     gb_close(gb_file *f);

/* Exact-name lookup. Returns NULL if absent. */
const gb_tensor *gb_find(const gb_file *f, const char *name);

/* Lookup that also checks the element count, so a shape mistake fails at load
 * rather than as silent garbage 40 layers later. Returns NULL on either miss. */
const gb_tensor *gb_need(const gb_file *f, const char *name, int64_t nelem);

/* f16 -> f32 for `n` values. IEEE half, including subnormals and inf/nan. */
void gb_f16_to_f32(const uint16_t *src, float *dst, int64_t n);

/* A tensor as f32, allocated with malloc(): f32 tensors are copied, f16 ones
 * converted. Caller frees. NULL on OOM or a NULL tensor. */
float *gb_dequant(const gb_tensor *t);

#endif /* EARS_GGML_BIN_H */

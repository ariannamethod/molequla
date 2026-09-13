/* ears.h — the organ: one wav in, one transcript out.
 *
 * Whisper on notorch. The model file is the legacy flat ggml `.bin` (ggml_bin.h),
 * the front end is mel.h, and everything after it rides notorch primitives:
 * nt_qmatmul / nt_qmatvec for the f16 weight matrices, nt_attention for both
 * self- and cross-attention, nt_blas_mm / nt_blas_mmT for the f32 products.
 *
 * Multilingual tiny and base. EOT 50257, SOT 50258, the timestamp and task tokens
 * shifted by n_langs - 98 = 1 (`src/whisper.cpp:1632`).
 */

#ifndef EARS_H
#define EARS_H

#include "ggml_bin.h"
#include "mel.h"

/* ── weights ──────────────────────────────────────────────────────────────── */

/* A matrix kept in the file's own precision. f16 stays packed and goes through
 * notorch's packed kernels; the few genuinely-f32 matrices take the f32 path. */
typedef struct {
    const void *data;
    int         f16;      /* 1 = packed f16 (GGUF dtype code 1), 0 = f32 */
    int         out, in;  /* logical [out, in], row-major */
} ears_mat;

typedef struct {
    float   *attn_ln_w, *attn_ln_b;
    ears_mat q_w, k_w, v_w, o_w;      /* attn.key carries no bias */
    float   *q_b, *v_b, *o_b;
    float   *mlp_ln_w, *mlp_ln_b;
    ears_mat fc1_w, fc2_w;
    float   *fc1_b, *fc2_b;
} ears_enc_block;

typedef struct {
    float   *attn_ln_w, *attn_ln_b;
    ears_mat q_w, k_w, v_w, o_w;
    float   *q_b, *v_b, *o_b;
    float   *cross_ln_w, *cross_ln_b;
    ears_mat cq_w, ck_w, cv_w, co_w;
    float   *cq_b, *cv_b, *co_b;
    float   *mlp_ln_w, *mlp_ln_b;
    ears_mat fc1_w, fc2_w;
    float   *fc1_b, *fc2_b;
} ears_dec_block;

/* ── vocabulary ───────────────────────────────────────────────────────────── */

/* whisper.cpp stores each token as the bytes it decodes to, so turning ids back
 * into UTF-8 is concatenation — the byte-level BPE table was already applied when
 * the model was converted. Ids past the stored table are the special tokens the
 * loader synthesises. */
typedef struct {
    int          n_vocab;      /* 51865 */
    int          n_tok;        /* stored entries, 50257 */
    const char **tok;
    int         *tok_len;
    int eot, sot, translate, transcribe, solm, prev, nosp, not_ts, beg;
    int n_langs;
} ears_vocab;

typedef struct ears_model ears_model;

struct ears_model {
    gb_file   *f;
    gb_hparams h;
    ears_vocab vocab;

    /* encoder */
    ears_mat  conv1_w, conv2_w;       /* [Cout, Cin*3] */
    float    *conv1_b, *conv2_b;
    float    *e_pe;                   /* [n_audio_ctx][n_audio_state] */
    ears_enc_block *enc;
    float    *e_ln_w, *e_ln_b;

    /* decoder */
    ears_mat  d_te;                   /* [n_vocab, n_text_state], tied to logits */
    float    *d_pe;                   /* [n_text_ctx][n_text_state] */
    ears_dec_block *dec;
    float    *d_ln_w, *d_ln_b;
};

ears_model *ears_model_load(const char *path);
void        ears_model_free(ears_model *m);

/* Language id for an ISO code, or -1. `ears_lang_str(id)` is the inverse. */
int         ears_lang_id(const char *code);
const char *ears_lang_str(int id);
int         ears_n_langs(void);

/* Append token `id` as bytes to `dst` (at most `cap`); returns bytes written. */
int         ears_token_append(const ears_vocab *v, int id, char *dst, size_t cap);

/* ── encoder ──────────────────────────────────────────────────────────────── */

/* One 30 s window starting at mel frame `mel_offset`. `out` must hold
 * n_audio_ctx * n_audio_state floats, row per audio frame. */
int  ears_encode(const ears_model *m, const ears_mel *mel, int mel_offset,
                 int n_threads, float *out);

/* ── decoder ──────────────────────────────────────────────────────────────── */

typedef struct ears_dec_state ears_dec_state;

/* Cross-attention K/V are a function of the encoder output alone, so they are
 * computed once per window and reused for every decoder step. */
ears_dec_state *ears_dec_begin(const ears_model *m, const float *embd_enc, int n_threads);
void            ears_dec_end(ears_dec_state *s);

/* Append `n` tokens at position `n_past` and return logits for the last one
 * (n_vocab floats, owned by the state). NULL on failure. */
const float *ears_decode(ears_dec_state *s, const int *tokens, int n, int n_past);

#endif /* EARS_H */

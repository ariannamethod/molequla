/* encoder.c — model load plus the audio encoder.
 *
 * conv1 (k3 s1) -> GELU -> conv2 (k3 s2) -> GELU -> + stored positional embedding
 * -> N pre-LN blocks of non-causal self-attention and MLP -> ln_post.
 *
 * `encoder.positional_embedding` is a weight in the file, not a sinusoid to be
 * synthesised, and `attn.key` has no bias in either model — both confirmed against
 * ggml-tiny.layout.txt / ggml-base.layout.txt, which list no `key.bias` at all.
 */

#include "ears.h"
#include "nnops.h"

#include <ariannamethod/notorch.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_CONV_K 3

/* ── loading ──────────────────────────────────────────────────────────────── */

static int mat_of(ears_mat *w, const gb_file *f, const char *name, int out, int in) {
    const gb_tensor *t = gb_need(f, name, (int64_t) out * in);
    if (!t) return -1;
    w->data = t->data;
    w->f16  = (t->ttype == GGML_BIN_F16);
    w->out  = out;
    w->in   = in;
    return 0;
}

static float *vec_of(const gb_file *f, const char *name, int n) {
    const gb_tensor *t = gb_need(f, name, n);
    return t ? gb_dequant(t) : NULL;
}

void ears_model_free(ears_model *m) {
    if (!m) return;
    free(m->conv1_b); free(m->conv2_b); free(m->e_pe); free(m->e_ln_w); free(m->e_ln_b);
    free(m->d_pe); free(m->d_ln_w); free(m->d_ln_b);
    if (m->enc)
        for (int i = 0; i < m->h.n_audio_layer; i++) {
            ears_enc_block *b = &m->enc[i];
            free(b->attn_ln_w); free(b->attn_ln_b);
            free(b->q_b); free(b->v_b); free(b->o_b);
            free(b->mlp_ln_w); free(b->mlp_ln_b); free(b->fc1_b); free(b->fc2_b);
        }
    if (m->dec)
        for (int i = 0; i < m->h.n_text_layer; i++) {
            ears_dec_block *b = &m->dec[i];
            free(b->attn_ln_w); free(b->attn_ln_b);
            free(b->q_b); free(b->v_b); free(b->o_b);
            free(b->cross_ln_w); free(b->cross_ln_b);
            free(b->cq_b); free(b->cv_b); free(b->co_b);
            free(b->mlp_ln_w); free(b->mlp_ln_b); free(b->fc1_b); free(b->fc2_b);
        }
    free(m->enc); free(m->dec);
    gb_close(m->f);
    free(m);
}

/* Every load step funnels through these so one missing tensor aborts the load
 * instead of leaving a NULL pointer for the forward pass to find. */
#define WANT(expr) do { if (!(expr)) goto fail; } while (0)
#define MAT(dst, name, o, i) do { if (mat_of(&(dst), f, (name), (o), (i)) != 0) goto fail; } while (0)

ears_model *ears_model_load(const char *path) {
    gb_file *f = gb_open(path);
    if (!f) return NULL;

    ears_model *m = (ears_model *) calloc(1, sizeof(*m));
    if (!m) { gb_close(f); return NULL; }
    m->f = f;
    m->h = f->hparams;

    const gb_hparams *h = &m->h;
    if (h->n_mels != f->n_mel || f->n_fft != 1 + 400 / 2) {
        fprintf(stderr, "ears: unexpected mel geometry n_mels=%d filters=%dx%d\n",
                h->n_mels, f->n_mel, f->n_fft);
        goto fail;
    }

    /* Special token ids. The base numbers are whisper.cpp:443-453; every
     * multilingual model takes the branch at :1632 that bumps EOT and SOT by one
     * and shifts the rest by n_langs - 98. Here n_langs = 99, so the shift is 1
     * and EOT = 50257, SOT = 50258, token_beg = 50364. */
    ears_vocab *v = &m->vocab;
    v->n_vocab = h->n_vocab;
    v->n_tok   = f->n_tok;
    v->tok     = f->tok;
    v->tok_len = f->tok_len;
    if (h->n_vocab < 51865) {
        fprintf(stderr, "ears: n_vocab %d is not a multilingual model; v1 needs one\n", h->n_vocab);
        goto fail;
    }
    v->n_langs = h->n_vocab - 51765 - 1;
    {
        const int dt = v->n_langs - 98;
        v->eot = 50257; v->sot = 50258;
        v->translate = 50357 + dt; v->transcribe = 50358 + dt;
        v->solm = 50359 + dt; v->prev = 50360 + dt; v->nosp = 50361 + dt;
        v->not_ts = 50362 + dt; v->beg = 50363 + dt;
    }

    const int NS = h->n_audio_state, NL = h->n_audio_layer;
    const int TS = h->n_text_state,  TL = h->n_text_layer;
    const int FF_A = 4 * NS, FF_T = 4 * TS;

    MAT(m->conv1_w, "encoder.conv1.weight", NS, h->n_mels * N_CONV_K);
    MAT(m->conv2_w, "encoder.conv2.weight", NS, NS * N_CONV_K);
    WANT(m->conv1_b = vec_of(f, "encoder.conv1.bias", NS));
    WANT(m->conv2_b = vec_of(f, "encoder.conv2.bias", NS));
    WANT(m->e_pe    = vec_of(f, "encoder.positional_embedding", (int64_t) h->n_audio_ctx * NS));
    WANT(m->e_ln_w  = vec_of(f, "encoder.ln_post.weight", NS));
    WANT(m->e_ln_b  = vec_of(f, "encoder.ln_post.bias", NS));

    WANT(m->enc = (ears_enc_block *) calloc((size_t) NL, sizeof(ears_enc_block)));
    for (int i = 0; i < NL; i++) {
        ears_enc_block *b = &m->enc[i];
        char n[96];
#define EN(suffix) (snprintf(n, sizeof(n), "encoder.blocks.%d." suffix, i), n)
        WANT(b->attn_ln_w = vec_of(f, EN("attn_ln.weight"), NS));
        WANT(b->attn_ln_b = vec_of(f, EN("attn_ln.bias"), NS));
        MAT(b->q_w, EN("attn.query.weight"), NS, NS);
        MAT(b->k_w, EN("attn.key.weight"),   NS, NS);
        MAT(b->v_w, EN("attn.value.weight"), NS, NS);
        MAT(b->o_w, EN("attn.out.weight"),   NS, NS);
        WANT(b->q_b = vec_of(f, EN("attn.query.bias"), NS));
        WANT(b->v_b = vec_of(f, EN("attn.value.bias"), NS));
        WANT(b->o_b = vec_of(f, EN("attn.out.bias"), NS));
        WANT(b->mlp_ln_w = vec_of(f, EN("mlp_ln.weight"), NS));
        WANT(b->mlp_ln_b = vec_of(f, EN("mlp_ln.bias"), NS));
        MAT(b->fc1_w, EN("mlp.0.weight"), FF_A, NS);
        MAT(b->fc2_w, EN("mlp.2.weight"), NS, FF_A);
        WANT(b->fc1_b = vec_of(f, EN("mlp.0.bias"), FF_A));
        WANT(b->fc2_b = vec_of(f, EN("mlp.2.bias"), NS));
#undef EN
    }

    MAT(m->d_te, "decoder.token_embedding.weight", h->n_vocab, TS);
    WANT(m->d_pe   = vec_of(f, "decoder.positional_embedding", (int64_t) h->n_text_ctx * TS));
    WANT(m->d_ln_w = vec_of(f, "decoder.ln.weight", TS));
    WANT(m->d_ln_b = vec_of(f, "decoder.ln.bias", TS));

    WANT(m->dec = (ears_dec_block *) calloc((size_t) TL, sizeof(ears_dec_block)));
    for (int i = 0; i < TL; i++) {
        ears_dec_block *b = &m->dec[i];
        char n[96];
#define DE(suffix) (snprintf(n, sizeof(n), "decoder.blocks.%d." suffix, i), n)
        WANT(b->attn_ln_w = vec_of(f, DE("attn_ln.weight"), TS));
        WANT(b->attn_ln_b = vec_of(f, DE("attn_ln.bias"), TS));
        MAT(b->q_w, DE("attn.query.weight"), TS, TS);
        MAT(b->k_w, DE("attn.key.weight"),   TS, TS);
        MAT(b->v_w, DE("attn.value.weight"), TS, TS);
        MAT(b->o_w, DE("attn.out.weight"),   TS, TS);
        WANT(b->q_b = vec_of(f, DE("attn.query.bias"), TS));
        WANT(b->v_b = vec_of(f, DE("attn.value.bias"), TS));
        WANT(b->o_b = vec_of(f, DE("attn.out.bias"), TS));
        WANT(b->cross_ln_w = vec_of(f, DE("cross_attn_ln.weight"), TS));
        WANT(b->cross_ln_b = vec_of(f, DE("cross_attn_ln.bias"), TS));
        MAT(b->cq_w, DE("cross_attn.query.weight"), TS, TS);
        MAT(b->ck_w, DE("cross_attn.key.weight"),   TS, TS);
        MAT(b->cv_w, DE("cross_attn.value.weight"), TS, TS);
        MAT(b->co_w, DE("cross_attn.out.weight"),   TS, TS);
        WANT(b->cq_b = vec_of(f, DE("cross_attn.query.bias"), TS));
        WANT(b->cv_b = vec_of(f, DE("cross_attn.value.bias"), TS));
        WANT(b->co_b = vec_of(f, DE("cross_attn.out.bias"), TS));
        WANT(b->mlp_ln_w = vec_of(f, DE("mlp_ln.weight"), TS));
        WANT(b->mlp_ln_b = vec_of(f, DE("mlp_ln.bias"), TS));
        MAT(b->fc1_w, DE("mlp.0.weight"), FF_T, TS);
        MAT(b->fc2_w, DE("mlp.2.weight"), TS, FF_T);
        WANT(b->fc1_b = vec_of(f, DE("mlp.0.bias"), FF_T));
        WANT(b->fc2_b = vec_of(f, DE("mlp.2.bias"), TS));
#undef DE
    }

    return m;

fail:
    ears_model_free(m);
    return NULL;
}

/* ── convolution ──────────────────────────────────────────────────────────── */

/* One k=3 pad-1 convolution over a [Cin][Lin] channel-major signal.
 *
 * The convolution itself is notorch's. ggml lowers conv_1d to im2col followed by
 * mul_mat and builds the im2col tensor as F16 (ggml.c, ggml_conv_1d passes
 * GGML_TYPE_F16), so both operands of that product are half precision — which is
 * why the call is nt_conv1d_f16cols and not nt_conv1d: the columns round where
 * ggml rounds and the dot accumulates in f32 on both sides.
 *
 * What is left in this function is the model file's storage format. A whisper .bin
 * may hold the conv weight as F16 and it has to be f32 before the product; that is
 * a ggml_bin concern, and notorch's op takes the weight it is given. */
static int conv1d(float *out, const float *in, int Cin, int Lin, int Lout,
                  const ears_mat *W, const float *bias, int stride) {
    const int K = N_CONV_K, Cout = W->out;
    /* The geometry the caller asked for must be the one pad-1 actually produces,
     * or the buffers downstream are the wrong length. */
    if ((Lin + 2 - K) / stride + 1 != Lout) return -1;

    float *wf = NULL;
    if (W->f16) {
        wf = (float *) malloc((size_t) Cout * Cin * K * sizeof(float));
        if (!wf) return -1;
        gb_f16_to_f32((const uint16_t *) W->data, wf, (int64_t) Cout * Cin * K);
    }
    const float *wsrc = W->f16 ? wf : (const float *) W->data;

    const int rc = nt_conv1d_f16cols(out, in, wsrc, bias, Cin, Lin, Cout, K, stride, 1);
    free(wf);
    return rc;
}

/* ── encoder ──────────────────────────────────────────────────────────────── */

/* Stage dump, for pinning a parity failure to a layer instead of guessing.
 * EARS_DUMP_STAGE selects it: "conv" writes the post-convolution activations in
 * the [n_state][n_len] layout whisper.cpp's embd_conv uses; an integer i writes
 * the [n_ctx][n_state] activations after block i (0-based), and -1 writes them
 * straight after the positional embedding. EARS_DUMP_PATH is the file. Both unset
 * is the normal path and costs one getenv per window. */
static void dump_stage(const char *stage, const float *x, int d0, int d1) {
    const char *want = getenv("EARS_DUMP_STAGE");
    const char *path = getenv("EARS_DUMP_PATH");
    if (!want || !path || strcmp(want, stage)) return;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(&d0, sizeof(int), 1, f);
    fwrite(&d1, sizeof(int), 1, f);
    fwrite(x, sizeof(float), (size_t) d0 * d1, f);
    fclose(f);
    fprintf(stderr, "ears: stage '%s' [%d %d] -> %s\n", stage, d0, d1, path);
}

int ears_encode(const ears_model *m, const ears_mel *mel, int mel_offset,
                int n_threads, float *out) {
    (void) n_threads;   /* notorch fans the matmuls out itself */

    const int NC = m->h.n_audio_ctx;          /* 1500 */
    const int NS = m->h.n_audio_state;
    const int NH = m->h.n_audio_head;
    const int HD = NS / NH;
    const int FF = 4 * NS;
    const int L0 = 2 * NC;                    /* 3000 mel frames per window */
    const int NM = m->h.n_mels;

    int rc = -1;
    float *win  = (float *) calloc((size_t) NM * L0, sizeof(float));
    float *c1   = (float *) malloc((size_t) NS * L0 * sizeof(float));
    float *c2   = (float *) malloc((size_t) NS * NC * sizeof(float));
    float *tmp  = (float *) malloc((size_t) NC * NS * sizeof(float));
    float *q    = (float *) malloc((size_t) NC * NS * sizeof(float));
    float *k    = (float *) malloc((size_t) NC * NS * sizeof(float));
    float *v    = (float *) malloc((size_t) NC * NS * sizeof(float));
    float *att  = (float *) malloc((size_t) NC * NS * sizeof(float));
    float *proj = (float *) malloc((size_t) NC * NS * sizeof(float));
    float *up   = (float *) malloc((size_t) NC * FF * sizeof(float));
    float *head = (float *) malloc((size_t) 4 * NC * HD * sizeof(float));
    if (!win || !c1 || !c2 || !tmp || !q || !k || !v || !att || !proj || !up || !head) goto done;

    /* Window of the mel, zero-padded past the end of the audio. */
    {
        int i0 = mel_offset < mel->n_len ? mel_offset : mel->n_len;
        int i1 = mel_offset + L0 < mel->n_len ? mel_offset + L0 : mel->n_len;
        for (int b = 0; b < NM; b++)
            for (int i = i0; i < i1; i++)
                win[(size_t) b * L0 + (i - i0)] = mel->data[(size_t) b * mel->n_len + i];
    }

    if (conv1d(c1, win, NM, L0, L0, &m->conv1_w, m->conv1_b, 1) != 0) goto done;
    ears_gelu(c1, (long) NS * L0);
    if (conv1d(c2, c1, NS, L0, NC, &m->conv2_w, m->conv2_b, 2) != 0) goto done;
    ears_gelu(c2, (long) NS * NC);
    dump_stage("conv", c2, NC, NS);

    /* [NS][NC] -> [NC][NS] and add the stored positional embedding. */
    for (int t = 0; t < NC; t++) {
        float *dst = out + (size_t) t * NS;
        const float *pe = m->e_pe + (size_t) t * NS;
        for (int d = 0; d < NS; d++) dst[d] = c2[(size_t) d * NC + t] + pe[d];
    }
    dump_stage("-1", out, NC, NS);

    for (int l = 0; l < m->h.n_audio_layer; l++) {
        const ears_enc_block *b = &m->enc[l];

        memcpy(tmp, out, (size_t) NC * NS * sizeof(float));
        ears_layernorm(tmp, b->attn_ln_w, b->attn_ln_b, NC, NS);
        if (ears_linear(q, tmp, &b->q_w, b->q_b, NC) != 0) goto done;
        if (ears_linear(k, tmp, &b->k_w, NULL,  NC) != 0) goto done;   /* no key bias */
        if (ears_linear(v, tmp, &b->v_w, b->v_b, NC) != 0) goto done;
        if (ears_mha(att, q, k, v, NC, NC, NH, HD, head) != 0) goto done;
        if (ears_linear(proj, att, &b->o_w, b->o_b, NC) != 0) goto done;
        for (size_t i = 0; i < (size_t) NC * NS; i++) out[i] += proj[i];

        memcpy(tmp, out, (size_t) NC * NS * sizeof(float));
        ears_layernorm(tmp, b->mlp_ln_w, b->mlp_ln_b, NC, NS);
        if (ears_linear(up, tmp, &b->fc1_w, b->fc1_b, NC) != 0) goto done;
        ears_gelu(up, (long) NC * FF);
        if (ears_linear(proj, up, &b->fc2_w, b->fc2_b, NC) != 0) goto done;
        for (size_t i = 0; i < (size_t) NC * NS; i++) out[i] += proj[i];

        { char s[16]; snprintf(s, sizeof(s), "%d", l); dump_stage(s, out, NC, NS); }
    }

    ears_layernorm(out, m->e_ln_w, m->e_ln_b, NC, NS);
    rc = 0;

done:
    free(win); free(c1); free(c2); free(tmp); free(q); free(k); free(v);
    free(att); free(proj); free(up); free(head);
    return rc;
}

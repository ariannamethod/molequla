/* decoder.c — text decoder with a KV cache, over one encoded window.
 *
 * Per block: pre-LN causal self-attention against the cache, pre-LN cross-attention
 * against the window's K/V, MLP. Then decoder.ln and logits tied to the transpose
 * of the token embedding.
 *
 * Tokens are fed one at a time even when a whole prompt arrives at once: with a
 * single query row the causal mask is exactly "every key written so far", so
 * nt_attention with T = 1 and S = n_past + 1 is the causal attention, and there is
 * no mask to get wrong. A 4-token prompt does not make that a cost.
 *
 * The cross K/V depend only on the encoder output, so they are computed once in
 * ears_dec_begin and reused for every step of the window; the self-attention
 * cache is addressed by n_past, so re-decoding from position 0 overwrites it and
 * needs no separate reset.
 */

#include "ears.h"
#include "nnops.h"

#include <ariannamethod/notorch.h>

#include <stdlib.h>
#include <string.h>

struct ears_dec_state {
    const ears_model *m;
    int NC, TS, TH, HD, FF, TL, NCTX;

    float **kc, **vc;        /* per layer, [NC][TS] cross K/V */
    float **kself, **vself;  /* per layer, [NCTX][TS] self cache */

    float *x, *tmp, *q, *att, *proj, *up, *head;
    float *logits;
};

void ears_dec_end(ears_dec_state *s) {
    if (!s) return;
    for (int l = 0; l < s->TL; l++) {
        if (s->kc)    free(s->kc[l]);
        if (s->vc)    free(s->vc[l]);
        if (s->kself) free(s->kself[l]);
        if (s->vself) free(s->vself[l]);
    }
    free(s->kc); free(s->vc); free(s->kself); free(s->vself);
    free(s->x); free(s->tmp); free(s->q); free(s->att); free(s->proj);
    free(s->up); free(s->head); free(s->logits);
    free(s);
}

ears_dec_state *ears_dec_begin(const ears_model *m, const float *embd_enc, int n_threads) {
    (void) n_threads;
    ears_dec_state *s = (ears_dec_state *) calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m    = m;
    s->NC   = m->h.n_audio_ctx;
    s->TS   = m->h.n_text_state;
    s->TH   = m->h.n_text_head;
    s->HD   = s->TS / s->TH;
    s->FF   = 4 * s->TS;
    s->TL   = m->h.n_text_layer;
    s->NCTX = m->h.n_text_ctx;

    s->kc    = (float **) calloc((size_t) s->TL, sizeof(float *));
    s->vc    = (float **) calloc((size_t) s->TL, sizeof(float *));
    s->kself = (float **) calloc((size_t) s->TL, sizeof(float *));
    s->vself = (float **) calloc((size_t) s->TL, sizeof(float *));
    s->x     = (float *) malloc((size_t) s->TS * sizeof(float));
    s->tmp   = (float *) malloc((size_t) s->TS * sizeof(float));
    s->q     = (float *) malloc((size_t) s->TS * sizeof(float));
    s->att   = (float *) malloc((size_t) s->TS * sizeof(float));
    s->proj  = (float *) malloc((size_t) s->TS * sizeof(float));
    s->up    = (float *) malloc((size_t) s->FF * sizeof(float));
    s->head  = (float *) malloc((size_t) (2 + 2 * s->NC) * s->HD * sizeof(float));
    s->logits= (float *) malloc((size_t) m->h.n_vocab * sizeof(float));
    if (!s->kc || !s->vc || !s->kself || !s->vself || !s->x || !s->tmp || !s->q ||
        !s->att || !s->proj || !s->up || !s->head || !s->logits) { ears_dec_end(s); return NULL; }

    for (int l = 0; l < s->TL; l++) {
        const ears_dec_block *b = &m->dec[l];
        s->kc[l]    = (float *) malloc((size_t) s->NC * s->TS * sizeof(float));
        s->vc[l]    = (float *) malloc((size_t) s->NC * s->TS * sizeof(float));
        s->kself[l] = (float *) malloc((size_t) s->NCTX * s->TS * sizeof(float));
        s->vself[l] = (float *) malloc((size_t) s->NCTX * s->TS * sizeof(float));
        if (!s->kc[l] || !s->vc[l] || !s->kself[l] || !s->vself[l]) { ears_dec_end(s); return NULL; }
        /* cross_attn.key has no bias, like attn.key */
        if (ears_linear(s->kc[l], embd_enc, &b->ck_w, NULL,   s->NC) != 0 ||
            ears_linear(s->vc[l], embd_enc, &b->cv_w, b->cv_b, s->NC) != 0) {
            ears_dec_end(s); return NULL;
        }
    }
    return s;
}

const float *ears_decode(ears_dec_state *s, const int *tokens, int n, int n_past) {
    const ears_model *m = s->m;
    const int TS = s->TS, TH = s->TH, HD = s->HD, FF = s->FF;

    if (n < 1 || n_past < 0 || n_past + n > s->NCTX) return NULL;

    for (int i = 0; i < n; i++) {
        const int tok = tokens[i];
        const int pos = n_past + i;
        if (tok < 0 || tok >= m->h.n_vocab) return NULL;

        /* token embedding + learned positional embedding */
        {
            const float *pe = m->d_pe + (size_t) pos * TS;
            if (m->d_te.f16) {
                gb_f16_to_f32((const uint16_t *) m->d_te.data + (size_t) tok * TS, s->x, TS);
            } else {
                memcpy(s->x, (const float *) m->d_te.data + (size_t) tok * TS, TS * sizeof(float));
            }
            for (int d = 0; d < TS; d++) s->x[d] += pe[d];
        }

        for (int l = 0; l < s->TL; l++) {
            const ears_dec_block *b = &m->dec[l];
            float *krow = s->kself[l] + (size_t) pos * TS;
            float *vrow = s->vself[l] + (size_t) pos * TS;

            memcpy(s->tmp, s->x, TS * sizeof(float));
            ears_layernorm(s->tmp, b->attn_ln_w, b->attn_ln_b, 1, TS);
            if (ears_linear(s->q,  s->tmp, &b->q_w, b->q_b, 1) != 0) return NULL;
            if (ears_linear(krow,  s->tmp, &b->k_w, NULL,   1) != 0) return NULL;
            if (ears_linear(vrow,  s->tmp, &b->v_w, b->v_b, 1) != 0) return NULL;
            if (ears_mha(s->att, s->q, s->kself[l], s->vself[l], 1, pos + 1, TH, HD, s->head) != 0)
                return NULL;
            if (ears_linear(s->proj, s->att, &b->o_w, b->o_b, 1) != 0) return NULL;
            for (int d = 0; d < TS; d++) s->x[d] += s->proj[d];

            memcpy(s->tmp, s->x, TS * sizeof(float));
            ears_layernorm(s->tmp, b->cross_ln_w, b->cross_ln_b, 1, TS);
            if (ears_linear(s->q, s->tmp, &b->cq_w, b->cq_b, 1) != 0) return NULL;
            if (ears_mha(s->att, s->q, s->kc[l], s->vc[l], 1, s->NC, TH, HD, s->head) != 0)
                return NULL;
            if (ears_linear(s->proj, s->att, &b->co_w, b->co_b, 1) != 0) return NULL;
            for (int d = 0; d < TS; d++) s->x[d] += s->proj[d];

            memcpy(s->tmp, s->x, TS * sizeof(float));
            ears_layernorm(s->tmp, b->mlp_ln_w, b->mlp_ln_b, 1, TS);
            if (ears_linear(s->up, s->tmp, &b->fc1_w, b->fc1_b, 1) != 0) return NULL;
            ears_gelu(s->up, FF);
            if (ears_linear(s->proj, s->up, &b->fc2_w, b->fc2_b, 1) != 0) return NULL;
            for (int d = 0; d < TS; d++) s->x[d] += s->proj[d];
        }
    }

    ears_layernorm(s->x, m->d_ln_w, m->d_ln_b, 1, TS);
    if (ears_linear(s->logits, s->x, &m->d_te, NULL, 1) != 0) return NULL;
    return s->logits;
}

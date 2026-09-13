/* nnops.c — see nnops.h. */

#include "nnops.h"

#include <ariannamethod/notorch.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

float ears_f16(float v) { return (float) (__fp16) v; }

int ears_linear(float *y, const float *x, const ears_mat *W, const float *b, int T) {
    const int m = W->out, k = W->in;
    if (W->f16) {
        /* GGUF dtype code 1 is F16; nt_qmatmul keeps the weight packed and
         * dequantises a block at a time, so base's 53 MB embedding never
         * materialises as f32. It delegates to nt_qmatvec at T == 1. */
        if (nt_qmatmul(y, (const uint8_t *) W->data, 1, x, m, k, T) != 0) return -1;
    } else {
        nt_blas_mmT(y, x, (const float *) W->data, T, k, m);
    }
    if (b)
        for (int t = 0; t < T; t++) {
            float *r = y + (size_t) t * m;
            for (int i = 0; i < m; i++) r[i] += b[i];
        }
    return 0;
}

void ears_layernorm(float *x, const float *w, const float *b, int T, int D) {
    for (int t = 0; t < T; t++) {
        float *r = x + (size_t) t * D;
        float sum = 0.0f;
        for (int i = 0; i < D; i++) sum += r[i];
        const float mean = sum / D;
        float var = 0.0f;
        for (int i = 0; i < D; i++) { float d = r[i] - mean; r[i] = d; var += d * d; }
        var /= D;
        const float scale = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < D; i++) r[i] = r[i] * scale * w[i] + b[i];
    }
}

void ears_gelu(float *x, long n) {
    const float c = 0.79788456080286535588f, a = 0.044715f;
    for (long i = 0; i < n; i++) {
        float v = x[i];
        if (v <= -10.0f) { x[i] = 0.0f; continue; }
        if (v >=  10.0f) { continue; }
        const float xr = ears_f16(v);                       /* table index */
        const float g  = 0.5f * xr * (1.0f + tanhf(c * xr * (1.0f + a * xr * xr)));
        x[i] = ears_f16(g);                                 /* table payload */
    }
}

int ears_mha(float *out, const float *Q, const float *K, const float *V,
             int T, int S, int n_head, int head_dim, float *head) {
    const int D = n_head * head_dim;
    float *qh = head;
    float *kh = qh + (size_t) T * head_dim;
    float *vh = kh + (size_t) S * head_dim;
    float *oh = vh + (size_t) S * head_dim;
    /* Q, K and V are rounded to f16 on the way into the product because that is
     * what they are in whisper.cpp: the encoder casts K and V to the model's
     * intermediate type (F16) before the attention, and mul_mat with an F16 src0
     * converts the f32 side to f16 as well. Rounding here costs one instruction
     * per gathered value and removes the largest remaining term in the encoder
     * parity diff. The accumulation stays f32 — ggml's flash path sums into an
     * f16 register, which is a backend artifact, not the model. */
    for (int h = 0; h < n_head; h++) {
        const int off = h * head_dim;
        for (int t = 0; t < T; t++) {
            const float *src = Q + (size_t) t * D + off;
            float *dst = qh + (size_t) t * head_dim;
            for (int d = 0; d < head_dim; d++) dst[d] = ears_f16(src[d]);
        }
        for (int s = 0; s < S; s++) {
            const float *ks = K + (size_t) s * D + off, *vs = V + (size_t) s * D + off;
            float *kd = kh + (size_t) s * head_dim, *vd = vh + (size_t) s * head_dim;
            for (int d = 0; d < head_dim; d++) { kd[d] = ears_f16(ks[d]); vd[d] = ears_f16(vs[d]); }
        }
        /* nt_attention carries the 1/sqrt(head_dim) scale. whisper splits the same
         * factor as head_dim^-0.25 on Q and on K; the product is identical, so Q
         * and K reach here unscaled. */
        if (nt_attention(oh, qh, kh, vh, T, S, head_dim) != 0) return -1;
        for (int t = 0; t < T; t++)
            memcpy(out + (size_t) t * D + off, oh + (size_t) t * head_dim, head_dim * sizeof(float));
    }
    return 0;
}

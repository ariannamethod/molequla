/* nnops.h — the arithmetic the encoder and decoder share.
 *
 * Thin wrappers over notorch, plus the two element-wise blocks whose exact form
 * decides whether greedy decoding lands on the same token as the oracle:
 *
 *  - LayerNorm is ggml's NORM_TYPE_NORMAL: mean and variance over the row in f32,
 *    eps 1e-5 inside the square root, then the per-channel affine.
 *  - GELU is ggml's f16 lookup table (ggml/src/ggml-cpu/vec.h:46 defines
 *    GGML_GELU_FP16), not the exact f32 tanh form. The table rounds the input to
 *    f16, evaluates tanh-GELU there, and stores the result as f16, which shifts
 *    values by ~1e-3 — enough to flip a near-tied argmax, so it is reproduced
 *    rather than approximated.
 *
 * Matrix products go to notorch: nt_qmatmul for a batch of rows against packed
 * f16 weights, nt_qmatvec for the single-row decoder case, nt_blas_mm / mmT for
 * f32 operands. Nothing here hand-rolls a matvec.
 */

#ifndef EARS_NNOPS_H
#define EARS_NNOPS_H

#include "ears.h"

/* y[T][out] = x[T][in] @ W^T + b (b may be NULL). Uses nt_qmatmul for T > 1 and
 * nt_qmatvec for T == 1 — the same kernel family either way. */
int  ears_linear(float *y, const float *x, const ears_mat *W, const float *b, int T);

/* In-place LayerNorm of T rows of D, then * w + b. */
void ears_layernorm(float *x, const float *w, const float *b, int T, int D);

/* In-place GELU over n values, through ggml's f16 table. */
void ears_gelu(float *x, long n);

/* Multi-head attention over contiguous [T, n_head*head_dim] buffers, via
 * nt_attention per head (which applies the 1/sqrt(head_dim) scale itself).
 * S is the key/value length; T == S is self-attention, S == n_audio_ctx is cross.
 * Scratch `head` must hold (2*T + 2*S) * head_dim floats. */
int  ears_mha(float *out, const float *Q, const float *K, const float *V,
              int T, int S, int n_head, int head_dim, float *head);

/* Round to f16 and back. ggml's im2col emits F16, so the convolution inputs pass
 * through this before the dot; skipping it is a ~1e-3 drift at the very first
 * layer, which the 4-6 blocks after it amplify. */
float ears_f16(float v);

#endif /* EARS_NNOPS_H */

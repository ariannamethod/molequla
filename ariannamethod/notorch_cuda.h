// ariannamethod_cuda.h — CUDA/cuBLAS backend for AML
// Zero PyTorch. Zero Python. Pure CUDA C.
//
// Compile: nvcc -c ariannamethod_cuda.cu -lcublas
// Link:    cc ... ariannamethod_cuda.o -lcublas -lcudart -L/usr/local/cuda/lib64
//
// "GPU goes brrrr. No Python required."

#ifndef ARIANNAMETHOD_CUDA_H
#define ARIANNAMETHOD_CUDA_H

#ifdef USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

// ── Init / Shutdown ────────────────────────────────────────────────
int   gpu_init(void);           // Returns 0 on success
void  gpu_shutdown(void);

// ── Memory management ──────────────────────────────────────────────
// GPU memory pool: weights stay on GPU, activations are transient
float* gpu_alloc(int n);                          // Alloc n floats on GPU
void   gpu_free(float* d_ptr);                    // Free GPU memory
void   gpu_upload(float* d_dst, const float* h_src, int n);   // CPU→GPU
void   gpu_download(float* h_dst, const float* d_src, int n); // GPU→CPU
void   gpu_zero(float* d_ptr, int n);             // memset to 0

// ── GEMM operations (the money shots) ──────────────────────────────
// All are thin wrappers around cublasSgemm.

// Y(M,N) = A(M,K) × B^T(N,K)  — used by seq_matvec forward
//   A = X(T, in_dim),  B = W(out_dim, in_dim),  Y = out(T, out_dim)
void gpu_sgemm_nt(int M, int N, int K,
                  const float* d_A, const float* d_B, float* d_C);

// C(M,N) = A(M,K) × B(K,N)  — general matmul, no transpose
void gpu_sgemm_nn(int M, int N, int K,
                  const float* d_A, const float* d_B, float* d_C);

// C(M,N) = A^T(K,M) × B(K,N) — used by backward dW
void gpu_sgemm_tn(int M, int N, int K,
                  const float* d_A, const float* d_B, float* d_C);

// ── Elementwise kernels ────────────────────────────────────────────
// These run on GPU to avoid CPU↔GPU transfers between ops

void gpu_add(float* d_out, const float* d_a, const float* d_b, int n);
void gpu_mul(float* d_out, const float* d_a, const float* d_b, int n);
/* y[i] += alpha * x[i] — wraps cublasSaxpy. */
void gpu_axpy(float* d_y, const float* d_x, int n, float alpha);
void gpu_silu(float* d_out, const float* d_in, int n);
void gpu_rmsnorm(float* d_out, const float* d_in, int T, int D);

// ── Backward kernels ──────────────────────────────────────────────
void gpu_silu_backward(float* d_grad_in, const float* d_grad_out,
                       const float* d_input, int n);
void gpu_add_backward(float* d_ga, float* d_gb, const float* d_grad, int n);
void gpu_mul_backward(float* d_ga, float* d_gb,
                      const float* d_grad, const float* d_a, const float* d_b, int n);
void gpu_rmsnorm_backward(float* d_gx, const float* d_grad,
                          const float* d_x, int T, int D);

// ── Weight cache ──────────────────────────────────────────────────
// Upload weights once, reuse across forward/backward passes
typedef struct {
    const char* name;
    float* d_data;
    int    len;
    int    dirty;  // 1 = needs re-upload after adam step
} GPU_WeightSlot;

#define GPU_MAX_WEIGHTS 256

int    gpu_cache_weight(const char* name, const float* h_data, int len);
float* gpu_get_weight(const char* name, int* len);
void   gpu_mark_all_dirty(void);  // After adam step, mark for re-upload
void   gpu_sync_dirty_weights(void); // Re-upload only changed weights
float* gpu_scratch(int slot, int n_floats);


// ── Attention kernel ──────────────────────────────────────────────
void gpu_multi_head_attention(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_out, float* d_scores,
    int T, int D, int n_heads);

void gpu_multi_head_attention_backward(
    const float* d_Q, const float* d_K, const float* d_V,
    const float* d_scores,
    const float* d_dout,
    float* d_dQ, float* d_dK, float* d_dV,
    float* d_scratch_TT, float* d_scratch_TT2,
    int T, int D, int n_heads);

// ── Cross-entropy kernel ──────────────────────────────────────────
float gpu_cross_entropy(const float* d_logits, const float* d_targets,
                        float* d_losses, int T, int V);

void gpu_cross_entropy_backward(float* d_grad_logits,
                                const float* d_logits,
                                const float* d_targets,
                                int T, int V);

// ── Chuck inner-loop kernel ────────────────────────────────────────
// Per-element: m, v EMA; m_hat, v_hat bias-corrected; param -= eff_lr * m_hat/(√v_hat + eps).
// Grad is uploaded; m, v are persistent on GPU; param is read+write.
void gpu_chuck_inner(float* d_param, float* d_m, float* d_v, const float* d_grad,
                     int n, float beta1, float beta2, float bc1, float bc2,
                     float eff_lr, float eps);

// ── RRPRAM low-rank attention (forward + backward) ────────────────
// Wr_combined layout: [Wr_a flat | Wr_b flat]
//   Wr_a: H*E*R floats — head h offset = h*E*R, indexed [d, r] = h*E*R + d*R + r
//   Wr_b: H*R*T_r floats — head h offset = H*E*R + h*R*T_r, indexed [r, j]
//   Total length = H*R*(E + T_r), assumes T_r == T
// V: [T, H*hd] — V_h at offset h*hd with stride H*hd
// Out: [T, H*hd]
// d_scores_out (scratch): [H, T, T] — softmaxed scores, persisted for backward
// d_U (scratch): [H, T, R] — U buffer for backward reuse
void gpu_rrpram_lr_forward(
    const float* d_X, const float* d_Wr_combined, const float* d_V,
    float* d_out,
    float* d_U,           /* [H, T, R] — persisted for backward */
    float* d_scores,      /* [H, T, T] — softmaxed, persisted for backward */
    int T, int E, int H, int R, int hd);

void gpu_rrpram_lr_backward(
    const float* d_X, const float* d_Wr_combined, const float* d_V,
    const float* d_U, const float* d_scores,   /* from forward */
    const float* d_dout,
    float* d_dWr_combined, float* d_dX, float* d_dV,
    float* d_d_attn, float* d_d_score,         /* scratch [H,T,T] */
    int T, int E, int H, int R, int hd);

// ── SEQ-RMSNORM (with optional gamma) ────────────────────────────
// y = (x / rms) * gamma; if d_gamma == NULL → just x / rms.
void gpu_seq_rmsnorm_gamma(float* d_out, const float* d_in,
                            const float* d_gamma, int T, int D);
// d_gx = grad wrt x, d_gg = grad wrt gamma (NULL if no gamma).
void gpu_seq_rmsnorm_backward(float* d_gx, float* d_gg,
                               const float* d_grad, const float* d_x,
                               const float* d_gamma, int T, int D);

// ── Reductions (cuBLAS) ──────────────────────────────────────────
// Returns ||x||_2 (Euclidean norm) of a GPU-resident buffer.
float gpu_nrm2(const float* d_x, int n);
// In-place scale: x *= alpha (cuBLAS Sscal).
void gpu_sscal(float* d_x, int n, float alpha);

// ── SwiGLU ────────────────────────────────────────────────────────
void gpu_swiglu(float* d_out, const float* d_g, const float* d_u, int n);
void gpu_swiglu_backward(float* d_dg, float* d_du,
                          const float* d_dout, const float* d_g,
                          const float* d_u, int n);

// ── RoPE (multi-head) ─────────────────────────────────────────────
// Applies rotation per (t, head, even-odd pair within head_dim).
void gpu_rope_forward(float* d_out, const float* d_in,
                      int T, int D, int n_heads, int head_dim, float fb);
void gpu_rope_backward(float* d_gx, const float* d_gout,
                       int T, int D, int n_heads, int head_dim, float fb);

// ── Scale ─────────────────────────────────────────────────────────
void gpu_scale(float* d_out, const float* d_in, int n, float s);

// ── Sequential embedding lookup ───────────────────────────────────
void gpu_seq_embedding_forward(float* d_out, const float* d_wte,
                                const float* d_tokens,
                                int T, int D, int wte_rows);
void gpu_seq_embedding_backward(float* d_dwte, const float* d_dout,
                                 const float* d_tokens,
                                 int T, int D, int wte_rows);

// ── Sequential cross-entropy (mean over valid positions, with ignore_index) ─
// Returns mean loss; populates d_valid (int per t, 0 or 1).
float gpu_seq_cross_entropy(const float* d_logits, const float* d_tokens,
                             float* d_losses, int* d_valid,
                             int T, int V, int ignore);
void gpu_seq_cross_entropy_backward(float* d_grad_logits,
                                     const float* d_logits,
                                     const float* d_tokens,
                                     int T, int V, int ignore,
                                     int n_valid);

#ifdef __cplusplus
}
#endif

#endif // USE_CUDA
#endif // ARIANNAMETHOD_CUDA_H

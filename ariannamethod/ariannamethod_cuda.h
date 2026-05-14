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
#ifdef __cplusplus
}
#endif

#endif // USE_CUDA
#endif // ARIANNAMETHOD_CUDA_H

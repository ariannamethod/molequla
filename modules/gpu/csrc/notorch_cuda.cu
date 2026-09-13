// ariannamethod_cuda.cu — CUDA/cuBLAS backend for AML
// Pure CUDA C. No PyTorch. No Python. No bullshit.
//
// Compile:
//   nvcc -c ariannamethod_cuda.cu -lcublas -O3
//
// "A100 goes brrrr. 50-100x over CPU."

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "notorch_cuda.h"

// ═══════════════════════════════════════════════════════════════════
// Globals
// ═══════════════════════════════════════════════════════════════════

static cublasHandle_t g_cublas = NULL;
static int g_gpu_ready = 0;
static GPU_WeightSlot g_wcache[GPU_MAX_WEIGHTS];
static int g_wcache_count = 0;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(call, ret) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return ret; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS ERROR] %s:%d: status %d\n", __FILE__, __LINE__, st); \
    } \
} while(0)

// ═══════════════════════════════════════════════════════════════════
// Init / Shutdown
// ═══════════════════════════════════════════════════════════════════

extern "C" int gpu_init(void) {
    if (g_gpu_ready) return 0;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "[GPU] No CUDA devices found\n");
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s — %.0f MB, compute %d.%d\n",
           prop.name, prop.totalGlobalMem / 1e6, prop.major, prop.minor);

    cublasStatus_t st = cublasCreate(&g_cublas);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[GPU] cuBLAS init failed: %d\n", st);
        return -1;
    }

    // Use TF32 for A100 — 8x faster than FP32, negligible accuracy loss
    cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    g_gpu_ready = 1;
    memset(g_wcache, 0, sizeof(g_wcache));
    g_wcache_count = 0;

    printf("[GPU] cuBLAS ready (TF32 enabled)\n");
    return 0;
}

extern "C" void gpu_shutdown(void) {
    if (!g_gpu_ready) return;
    // Free weight cache
    for (int i = 0; i < g_wcache_count; i++) {
        if (g_wcache[i].d_data) cudaFree(g_wcache[i].d_data);
    }
    g_wcache_count = 0;
    if (g_cublas) cublasDestroy(g_cublas);
    g_cublas = NULL;
    g_gpu_ready = 0;
    printf("[GPU] shutdown\n");
}

// ═══════════════════════════════════════════════════════════════════
// Memory management
// ═══════════════════════════════════════════════════════════════════

/* Free-list cache: avoid cudaMalloc/cudaFree on every tape clear.
 * Bucketize by next-power-of-two size class. Track each cached buffer's
 * size via a parallel array so gpu_free can find the bucket from the ptr
 * (linear scan, but bounded by cache capacity ~448 entries).
 */
#define GPU_CACHE_BUCKETS 28        /* up to 2^27 floats = 512 MB single tensor */
#define GPU_CACHE_PER_BUCKET 32
typedef struct {
    float* slots[GPU_CACHE_PER_BUCKET];
    int    count;
} gpu_cache_bucket;
static gpu_cache_bucket g_alloc_cache[GPU_CACHE_BUCKETS];

/* Side table: ptr → bucket. Bounded fixed-size open-addressed hash table.
 * 4096 slots is far more than realistic concurrent live alloc count. */
#define GPU_PTR_MAP_SIZE 65536
typedef struct {
    float* ptr;
    int    bucket;
} gpu_ptr_entry;
static gpu_ptr_entry g_ptr_map[GPU_PTR_MAP_SIZE];

static unsigned gpu_ptr_hash(float* p) {
    unsigned long long u = (unsigned long long)p;
    u = (u >> 7) * 11400714819323198485ULL;
    return (unsigned)(u >> 32) & (GPU_PTR_MAP_SIZE - 1);
}

static void gpu_ptr_map_set(float* p, int bucket) {
    unsigned h = gpu_ptr_hash(p);
    for (int i = 0; i < GPU_PTR_MAP_SIZE; i++) {
        unsigned idx = (h + i) & (GPU_PTR_MAP_SIZE - 1);
        if (g_ptr_map[idx].ptr == NULL || g_ptr_map[idx].ptr == p) {
            g_ptr_map[idx].ptr = p;
            g_ptr_map[idx].bucket = bucket;
            return;
        }
    }
    fprintf(stderr, "[GPU] ptr_map full — buffer leak\n");
}

static int gpu_ptr_map_get_and_clear(float* p) {
    unsigned h = gpu_ptr_hash(p);
    for (int i = 0; i < GPU_PTR_MAP_SIZE; i++) {
        unsigned idx = (h + i) & (GPU_PTR_MAP_SIZE - 1);
        if (g_ptr_map[idx].ptr == p) {
            int b = g_ptr_map[idx].bucket;
            g_ptr_map[idx].ptr = NULL;
            g_ptr_map[idx].bucket = -1;
            return b;
        }
        if (g_ptr_map[idx].ptr == NULL) return -1;
    }
    return -1;
}

static int gpu_cache_bucket_for(int n) {
    int b = 0;
    int v = 1;
    while (v < n && b < GPU_CACHE_BUCKETS - 1) { v <<= 1; b++; }
    return b;
}

extern "C" float* gpu_alloc(int n) {
    int b = gpu_cache_bucket_for(n);
    if (b < GPU_CACHE_BUCKETS && g_alloc_cache[b].count > 0) {
        float* p = g_alloc_cache[b].slots[--g_alloc_cache[b].count];
        gpu_ptr_map_set(p, b);
        return p;
    }
    /* Round up alloc to next pow2 so any future alloc with same bucket fits. */
    int alloc_n = 1; while (alloc_n < n) alloc_n <<= 1;
    if (alloc_n < n) alloc_n = n;
    float* d_ptr = NULL;
    cudaError_t err = cudaMalloc(&d_ptr, (size_t)alloc_n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU] alloc failed: %s (%d floats = %.1f MB)\n",
                cudaGetErrorString(err), n, n * 4.0f / 1e6);
        return NULL;
    }
    gpu_ptr_map_set(d_ptr, b);
    return d_ptr;
}

extern "C" void gpu_free(float* d_ptr) {
    if (!d_ptr) return;
    int bucket = gpu_ptr_map_get_and_clear(d_ptr);
    if (bucket < 0 || bucket >= GPU_CACHE_BUCKETS) {
        cudaFree(d_ptr);
        return;
    }
    if (g_alloc_cache[bucket].count < GPU_CACHE_PER_BUCKET) {
        g_alloc_cache[bucket].slots[g_alloc_cache[bucket].count++] = d_ptr;
        return;
    }
    cudaFree(d_ptr);
}

extern "C" void gpu_alloc_cache_clear(void) {
    for (int b = 0; b < GPU_CACHE_BUCKETS; b++) {
        for (int i = 0; i < g_alloc_cache[b].count; i++)
            cudaFree(g_alloc_cache[b].slots[i]);
        g_alloc_cache[b].count = 0;
    }
}

extern "C" void gpu_upload(float* d_dst, const float* h_src, int n) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void gpu_download(float* h_dst, const float* d_src, int n) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C" void gpu_zero(float* d_ptr, int n) {
    CUDA_CHECK(cudaMemset(d_ptr, 0, n * sizeof(float)));
}

// ═══════════════════════════════════════════════════════════════════
// GEMM wrappers — the core of GPU acceleration
// ═══════════════════════════════════════════════════════════════════
//
// cuBLAS is column-major. We store row-major.
// Trick: to compute C = A × B^T in row-major,
//   call cublasSgemm with: C^T = B × A^T in col-major
//   i.e., cublasSgemm(N, T, K, N, ... B, N, A, K, ... C, N)
//
// Row-major C(M,N) = A(M,K) × B^T(N,K):
//   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//               N, M, K, &alpha, d_B, K, d_A, K, &beta, d_C, N)

extern "C" void gpu_sgemm_nt(int M, int N, int K,
                              const float* d_A, const float* d_B, float* d_C) {
    // C(M,N) = A(M,K) × B^T(N,K)   [row-major]
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, K,    // B(N,K) row-major → col-major: B^T, ld=K
        d_A, K,    // A(M,K) row-major → col-major: A^T, ld=K
        &beta,
        d_C, N));  // C(M,N) row-major → col-major: C^T, ld=N
}

extern "C" void gpu_sgemm_nn(int M, int N, int K,
                              const float* d_A, const float* d_B, float* d_C) {
    // C(M,N) = A(M,K) × B(K,N)   [row-major]
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,    // B(K,N) row-major → col-major, ld=N
        d_A, K,    // A(M,K) row-major → col-major, ld=K
        &beta,
        d_C, N));  // C(M,N) row-major → col-major, ld=N
}

extern "C" void gpu_sgemm_tn(int M, int N, int K,
                              const float* d_A, const float* d_B, float* d_C) {
    // C(M,N) = A^T(K,M) × B(K,N)   [row-major]
    // A stored as (K,M), B as (K,N), C as (M,N)
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        d_B, N,    // B(K,N)
        d_A, M,    // A(K,M) — we want A^T so in col-major this becomes OP_T
        &beta,
        d_C, N));
}

// ═══════════════════════════════════════════════════════════════════
// Elementwise CUDA kernels
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void kernel_mul(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void kernel_silu(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

__global__ void kernel_rmsnorm(float* out, const float* in, int T, int D) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* x = in + t * D;
    float* y = out + t * D;

    // Compute RMS using shared memory reduction
    extern __shared__ float sdata[];
    float local_sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local_sum += x[d] * x[d];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / D + 1e-6f);
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        y[d] = x[d] / rms;
}

static int gpu_blocks(int n, int threads) { return (n + threads - 1) / threads; }

extern "C" void gpu_add(float* d_out, const float* d_a, const float* d_b, int n) {
    kernel_add<<<gpu_blocks(n, 256), 256>>>(d_out, d_a, d_b, n);
}

extern "C" void gpu_mul(float* d_out, const float* d_a, const float* d_b, int n) {
    kernel_mul<<<gpu_blocks(n, 256), 256>>>(d_out, d_a, d_b, n);
}

extern "C" void gpu_silu(float* d_out, const float* d_in, int n) {
    kernel_silu<<<gpu_blocks(n, 256), 256>>>(d_out, d_in, n);
}

extern "C" void gpu_axpy(float* d_y, const float* d_x, int n, float alpha) {
    if (!g_cublas) return;
    CUBLAS_CHECK(cublasSaxpy(g_cublas, n, &alpha, d_x, 1, d_y, 1));
}

extern "C" float gpu_nrm2(const float* d_x, int n) {
    if (!g_cublas || !d_x || n <= 0) return 0.0f;
    float result = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(g_cublas, n, d_x, 1, &result));
    return result;
}

extern "C" void gpu_sscal(float* d_x, int n, float alpha) {
    if (!g_cublas || !d_x || n <= 0) return;
    CUBLAS_CHECK(cublasSscal(g_cublas, n, &alpha, d_x, 1));
}

extern "C" void gpu_rmsnorm(float* d_out, const float* d_in, int T, int D) {
    int threads = D < 256 ? D : 256;
    kernel_rmsnorm<<<T, threads, threads * sizeof(float)>>>(d_out, d_in, T, D);
}

// ═══════════════════════════════════════════════════════════════════
// Backward kernels
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_silu_backward(float* grad_in, const float* grad_out,
                                     const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float sig = 1.0f / (1.0f + expf(-x));
        float silu_val = x * sig;
        // d(silu)/dx = sig + x * sig * (1 - sig) = sig * (1 + x * (1 - sig))
        grad_in[i] = grad_out[i] * (sig + silu_val * (1.0f - sig));
    }
}

__global__ void kernel_add_backward(float* ga, float* gb, const float* grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { ga[i] = grad[i]; gb[i] = grad[i]; }
}

__global__ void kernel_mul_backward(float* ga, float* gb,
                                    const float* grad, const float* a,
                                    const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { ga[i] = grad[i] * b[i]; gb[i] = grad[i] * a[i]; }
}

__global__ void kernel_rmsnorm_backward(float* gx, const float* grad,
                                        const float* x, int T, int D) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* x_t = x + t * D;
    const float* dout_t = grad + t * D;
    float* gx_t = gx + t * D;

    extern __shared__ float sdata[];

    // Compute ss = sum(x^2)
    float local_ss = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local_ss += x_t[d] * x_t[d];
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / D + 1e-6f);
    float rms3 = rms * rms * rms;

    // Compute sum_dx = sum(dout * x)
    float local_sd = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local_sd += dout_t[d] * x_t[d];
    sdata[threadIdx.x] = local_sd;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum_dx = sdata[0];

    for (int d = threadIdx.x; d < D; d += blockDim.x)
        gx_t[d] = (dout_t[d] / rms) - (x_t[d] * sum_dx / (D * rms3));
}

extern "C" void gpu_silu_backward(float* d_grad_in, const float* d_grad_out,
                                   const float* d_input, int n) {
    kernel_silu_backward<<<gpu_blocks(n, 256), 256>>>(d_grad_in, d_grad_out, d_input, n);
}

extern "C" void gpu_add_backward(float* d_ga, float* d_gb, const float* d_grad, int n) {
    kernel_add_backward<<<gpu_blocks(n, 256), 256>>>(d_ga, d_gb, d_grad, n);
}

extern "C" void gpu_mul_backward(float* d_ga, float* d_gb,
                                  const float* d_grad, const float* d_a,
                                  const float* d_b, int n) {
    kernel_mul_backward<<<gpu_blocks(n, 256), 256>>>(d_ga, d_gb, d_grad, d_a, d_b, n);
}

extern "C" void gpu_rmsnorm_backward(float* d_gx, const float* d_grad,
                                      const float* d_x, int T, int D) {
    int threads = D < 256 ? D : 256;
    kernel_rmsnorm_backward<<<T, threads, threads * sizeof(float)>>>(d_gx, d_grad, d_x, T, D);
}

// ═══════════════════════════════════════════════════════════════════
// Weight cache — upload once, reuse
// ═══════════════════════════════════════════════════════════════════

static int wcache_find(const char* name) {
    for (int i = 0; i < g_wcache_count; i++)
        if (g_wcache[i].name && strcmp(g_wcache[i].name, name) == 0)
            return i;
    return -1;
}

extern "C" int gpu_cache_weight(const char* name, const float* h_data, int len) {
    int idx = wcache_find(name);
    if (idx >= 0) {
        // Re-upload if size changed or dirty
        if (g_wcache[idx].len != len) {
            cudaFree(g_wcache[idx].d_data);
            g_wcache[idx].d_data = gpu_alloc(len);
            g_wcache[idx].len = len;
        }
        gpu_upload(g_wcache[idx].d_data, h_data, len);
        g_wcache[idx].dirty = 0;
        return idx;
    }
    if (g_wcache_count >= GPU_MAX_WEIGHTS) {
        fprintf(stderr, "[GPU] weight cache full (%d slots)\n", GPU_MAX_WEIGHTS);
        return -1;
    }
    idx = g_wcache_count++;
    g_wcache[idx].name = strdup(name);
    g_wcache[idx].d_data = gpu_alloc(len);
    g_wcache[idx].len = len;
    g_wcache[idx].dirty = 0;
    if (g_wcache[idx].d_data)
        gpu_upload(g_wcache[idx].d_data, h_data, len);
    return idx;
}

extern "C" float* gpu_get_weight(const char* name, int* len) {
    int idx = wcache_find(name);
    if (idx < 0) { if (len) *len = 0; return NULL; }
    if (len) *len = g_wcache[idx].len;
    return g_wcache[idx].d_data;
}

extern "C" void gpu_mark_all_dirty(void) {
    for (int i = 0; i < g_wcache_count; i++)
        g_wcache[i].dirty = 1;
}

extern "C" void gpu_sync_dirty_weights(void) {
    // This is called after adam step: download updated weights from CPU
    // In a full GPU pipeline, adam would run on GPU too.
    // For now, we re-upload from CPU after adam updates.
}


#define GPU_SCRATCH_SLOTS 16
static float* g_scratch_buf[GPU_SCRATCH_SLOTS];
static size_t g_scratch_sz[GPU_SCRATCH_SLOTS];

extern "C" float* gpu_scratch(int slot, int n_floats) {
    if (slot < 0 || slot >= GPU_SCRATCH_SLOTS) return NULL;
    size_t bytes = (size_t)n_floats * sizeof(float);
    if (bytes > g_scratch_sz[slot]) {
        if (g_scratch_buf[slot]) cudaFree(g_scratch_buf[slot]);
        cudaMalloc((void**)&g_scratch_buf[slot], bytes);
        g_scratch_sz[slot] = bytes;
    }
    return g_scratch_buf[slot];
}


// ═══════════════════════════════════════════════════════════════════
// Multi-head causal attention — GPU kernel
// ═══════════════════════════════════════════════════════════════════
//
// Q,K,V: [T, D],  D = n_heads * head_dim
// Output: [T, D]
// Uses cublasSgemm per head for QK^T and attn*V
// Custom kernel for causal softmax

__global__ void kernel_causal_softmax(float* scores, int T, int n_heads) {
    // scores[h * T * T + i * T + j]
    // Apply causal mask (j > i -> -inf) then softmax per row
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= n_heads || i >= T) return;

    float* row = scores + h * T * T + i * T;

    // Causal mask
    for (int j = i + 1; j < T; j++)
        row[j] = -1e10f;

    // Find max
    float mx = row[0];
    for (int j = 1; j <= i; j++)
        if (row[j] > mx) mx = row[j];

    // Exp and sum
    float sum = 0;
    for (int j = 0; j <= i; j++) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int j = 0; j < T; j++)
        row[j] = (j <= i) ? row[j] * inv_sum : 0.0f;
}

extern "C" void gpu_multi_head_attention(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_out, float* d_scores,
    int T, int D, int n_heads)
{
    if (!g_cublas) return;
    int head_dim = D / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    float beta = 0.0f;

    // Batched QK^T: one cuBLAS call replaces n_heads launches.
    // Per-head stride for Q/K is head_dim (col offset in row-major [T, D]).
    // Per-head stride for scores is T*T (separate [T, T] slabs).
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        T, T, head_dim,
        &scale,
        d_K, D, head_dim,                  /* K_h: ld=D, stride=head_dim */
        d_Q, D, head_dim,
        &beta,
        d_scores, T, (long long)T * T,
        n_heads));

    // Causal softmax
    dim3 grid(n_heads, T);
    kernel_causal_softmax<<<grid, 1>>>(d_scores, T, n_heads);

    // Batched attn * V
    float alpha_v = 1.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, T, T,
        &alpha_v,
        d_V, D, head_dim,
        d_scores, T, (long long)T * T,
        &beta,
        d_out, D, head_dim,
        n_heads));
}

// ═══════════════════════════════════════════════════════════════════
// Attention backward
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_softmax_backward(float* d_grad_scores,
                                         const float* d_scores,
                                         const float* d_grad_out_scores,
                                         int T, int n_heads) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= n_heads || i >= T) return;

    const float* attn_row = d_scores + h * T * T + i * T;
    const float* dout_row = d_grad_out_scores + h * T * T + i * T;
    float* grad_row = d_grad_scores + h * T * T + i * T;

    float dot = 0;
    for (int j = 0; j <= i; j++)
        dot += attn_row[j] * dout_row[j];

    for (int j = 0; j < T; j++)
        grad_row[j] = (j <= i) ? attn_row[j] * (dout_row[j] - dot) : 0.0f;
}

extern "C" void gpu_multi_head_attention_backward(
    const float* d_Q, const float* d_K, const float* d_V,
    const float* d_scores,
    const float* d_dout,
    float* d_dQ, float* d_dK, float* d_dV,
    float* d_scratch_TT,
    float* d_scratch_TT2,
    int T, int D, int n_heads)
{
    if (!g_cublas) return;
    int head_dim = D / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    float alpha = 1.0f, beta = 0.0f;
    long long S_TT = (long long)T * T;

    // Step 1: d_attn_weights[h](T,T) = dout_h(T,hd) * V_h(T,hd)^T  (batched)
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        T, T, head_dim,
        &alpha,
        d_V,    D, head_dim,
        d_dout, D, head_dim,
        &beta,
        d_scratch_TT2, T, S_TT,
        n_heads));

    // Step 2: softmax backward
    dim3 grid(n_heads, T);
    kernel_softmax_backward<<<grid, 1>>>(d_scratch_TT, d_scores, d_scratch_TT2, T, n_heads);

    // Step 3: dV_h(T,hd) = scores_h^T(T,T) * dout_h(T,hd)  (batched)
    gpu_zero(d_dV, T * D);
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        head_dim, T, T,
        &alpha,
        d_dout,   D, head_dim,
        d_scores, T, S_TT,
        &beta,
        d_dV,     D, head_dim,
        n_heads));

    // Step 4: dQ_h(T,hd) = grad_scores_h(T,T) * K_h(T,hd) * scale  (batched)
    gpu_zero(d_dQ, T * D);
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, T, T,
        &scale,
        d_K,           D, head_dim,
        d_scratch_TT,  T, S_TT,
        &beta,
        d_dQ,          D, head_dim,
        n_heads));

    // Step 5: dK_h(T,hd) = grad_scores_h^T(T,T) * Q_h(T,hd) * scale  (batched)
    gpu_zero(d_dK, T * D);
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        head_dim, T, T,
        &scale,
        d_Q,          D, head_dim,
        d_scratch_TT, T, S_TT,
        &beta,
        d_dK,         D, head_dim,
        n_heads));
}

// ═══════════════════════════════════════════════════════════════════
// Cross-entropy — GPU kernel
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_cross_entropy_forward(const float* logits, const float* targets,
                                              float* losses, int T, int V) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float* l = logits + t * V;
    int target = (int)targets[t];
    if (target < 0 || target >= V) target = 0;

    float mx = l[0];
    for (int j = 1; j < V; j++)
        if (l[j] > mx) mx = l[j];

    float sum = 0;
    for (int j = 0; j < V; j++)
        sum += expf(l[j] - mx);

    losses[t] = -((l[target] - mx) - logf(sum + 1e-10f));
}

__global__ void kernel_cross_entropy_backward(float* grad_logits,
                                               const float* logits,
                                               const float* targets,
                                               int T, int V, float scale) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float* l = logits + t * V;
    float* gl = grad_logits + t * V;
    int target = (int)targets[t];
    if (target < 0 || target >= V) target = 0;

    float mx = l[0];
    for (int j = 1; j < V; j++)
        if (l[j] > mx) mx = l[j];

    float sum = 0;
    for (int j = 0; j < V; j++)
        sum += expf(l[j] - mx);

    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int j = 0; j < V; j++) {
        float prob = expf(l[j] - mx) * inv_sum;
        gl[j] = scale * (prob - (j == target ? 1.0f : 0.0f));
    }
}

extern "C" float gpu_cross_entropy(const float* d_logits, const float* d_targets,
                                    float* d_losses, int T, int V) {
    kernel_cross_entropy_forward<<<T, 1>>>(d_logits, d_targets, d_losses, T, V);
    /* Reduce on GPU via cuBLAS Sasum (Σ |x|; losses are ≥ 0 so this is a sum). */
    float total = 0.0f;
    if (g_cublas) {
        CUBLAS_CHECK(cublasSasum(g_cublas, T, d_losses, 1, &total));
    } else {
        float* h_losses = (float*)malloc(T * sizeof(float));
        gpu_download(h_losses, d_losses, T);
        for (int t = 0; t < T; t++) total += h_losses[t];
        free(h_losses);
    }
    return total / T;
}

extern "C" void gpu_cross_entropy_backward(float* d_grad_logits,
                                            const float* d_logits,
                                            const float* d_targets,
                                            int T, int V) {
    float scale = 1.0f / T;
    kernel_cross_entropy_backward<<<T, 1>>>(d_grad_logits, d_logits, d_targets, T, V, scale);
}

// ═══════════════════════════════════════════════════════════════════
// Chuck inner loop — m, v EMA + bias-correct + param update.
// Per-element: trivially parallel.
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_chuck_inner(float* p, float* m, float* v, const float* g,
                                    int n, float beta1, float beta2,
                                    float bc1, float bc2, float eff_lr, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    p[i] -= eff_lr * m_hat / (sqrtf(v_hat) + eps);
}

extern "C" void gpu_chuck_inner(float* d_param, float* d_m, float* d_v, const float* d_grad,
                                 int n, float beta1, float beta2, float bc1, float bc2,
                                 float eff_lr, float eps) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_chuck_inner<<<blocks, threads>>>(d_param, d_m, d_v, d_grad, n, beta1, beta2, bc1, bc2, eff_lr, eps);
}

// ═══════════════════════════════════════════════════════════════════
// RRPRAM low-rank attention (forward + backward) — single GPU port
// Per head h:
//   U_h[T,R]  = X[T,E] @ Wra_h[E,R]
//   S_h[T,T]  = U_h[T,R] @ Wrb_h[R,T]   (causal softmax applied)
//   Out_h[T,hd] = A_h[T,T] @ V_h[T,hd]   (V_h has stride out_dim = H*hd)
//
// Backward (per head h):
//   d_attn[T,T]   = dout_h[T,hd] @ V_h^T[hd,T]
//   d_V_h[T,hd]  += A_h^T[T,T] @ dout_h[T,hd]
//   d_score      = softmax_bwd(d_attn, A)
//   d_U_h[T,R]   = d_score @ Wrb_h^T[T,R]
//   d_Wrb_h[R,T]+= U_h^T[R,T] @ d_score   (causal lower-triangular)
//   d_X[T,E]    += d_U_h @ Wra_h^T[R,E]
//   d_Wra_h[E,R]+= X^T[E,T] @ d_U_h
// ═══════════════════════════════════════════════════════════════════

extern "C" void gpu_rrpram_lr_forward(
    const float* d_X, const float* d_Wr_combined, const float* d_V,
    float* d_out, float* d_U, float* d_scores,
    int T, int E, int H, int R, int hd)
{
    if (!g_cublas) return;
    long wra_total = (long)H * E * R;
    int  out_dim = H * hd;
    float alpha = 1.0f, beta = 0.0f;

    for (int h = 0; h < H; h++) {
        const float* Wra_h = d_Wr_combined + (long)h * E * R;          /* [E,R] row-major */
        const float* Wrb_h = d_Wr_combined + wra_total + (long)h * R * T;/* [R,T] row-major */
        float* U_h     = d_U + (long)h * T * R;                         /* [T,R] row-major */
        float* S_h     = d_scores + (long)h * T * T;                    /* [T,T] row-major */

        /* U_h[T,R] = X[T,E] @ Wra_h[E,R] — NN gemm */
        gpu_sgemm_nn(T, R, E, d_X, Wra_h, U_h);
        /* S_h[T,T] = U_h[T,R] @ Wrb_h[R,T] — NN gemm */
        gpu_sgemm_nn(T, T, R, U_h, Wrb_h, S_h);
    }

    /* Causal softmax in-place over [H, T, T]. */
    dim3 grid(H, T);
    kernel_causal_softmax<<<grid, 1>>>(d_scores, T, H);

    /* Out_h[T,hd] = A_h[T,T] @ V_h[T,hd]; V_h has stride out_dim = H*hd.
     * V_h is a sub-tensor of V[T, H*hd] starting at column h*hd, ld=H*hd
     * (row-major). Use cublasSgemm directly with strided V.
     * Col-major view: Out_h^T(hd,T) = V_h^T(hd,T) × A_h^T(T,T). */
    for (int h = 0; h < H; h++) {
        const float* S_h = d_scores + (long)h * T * T;
        const float* Vh  = d_V   + h * hd;
        float*       Oh  = d_out + h * hd;
        CUBLAS_CHECK(cublasSgemm(g_cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hd, T, T,
            &alpha,
            Vh,  out_dim,
            S_h, T,
            &beta,
            Oh,  out_dim));
    }
}

/* Softmax backward kernel (Jacobian-vector product) for general H heads.
 * Reuses kernel_softmax_backward from MH path — same layout. */

/* Helper: row-major C(M,N) = A(M,K) × B(K,N) with beta=1 (accumulate). */
static void gpu_sgemm_nn_acc(int M, int N, int K,
                             const float* d_A, const float* d_B, float* d_C) {
    float alpha = 1.0f, beta = 1.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N));
}

/* Helper: row-major C(M,N) = A(M,K) × B^T(N,K) with beta. */
static void gpu_sgemm_nt_beta(int M, int N, int K,
                              const float* d_A, const float* d_B, float* d_C, float beta) {
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, K,
        d_A, K,
        &beta,
        d_C, N));
}

/* Helper: row-major C(M,N) = A^T(K,M) × B(K,N) with beta. */
static void gpu_sgemm_tn_beta(int M, int N, int K,
                              const float* d_A, const float* d_B, float* d_C, float beta) {
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, M,
        &beta,
        d_C, N));
}

extern "C" void gpu_rrpram_lr_backward(
    const float* d_X, const float* d_Wr_combined, const float* d_V,
    const float* d_U, const float* d_scores,
    const float* d_dout,
    float* d_dWr_combined, float* d_dX, float* d_dV,
    float* d_d_attn, float* d_d_score,
    int T, int E, int H, int R, int hd)
{
    if (!g_cublas) return;
    long wra_total = (long)H * E * R;
    int  out_dim = H * hd;
    float alpha = 1.0f, beta_acc = 1.0f, beta_zero = 0.0f;

    /* zero global accumulators */
    gpu_zero(d_dX,  (long)T * E);
    gpu_zero(d_dV,  (long)T * out_dim);
    gpu_zero(d_dWr_combined, wra_total + (long)H * R * T);

    /* Phase 1: per-head, compute d_attn[H,T,T] (no V_h gradient yet — accumulate later)
     * and d_V partial via softmaxed scores. */
    for (int h = 0; h < H; h++) {
        const float* dout_h= d_dout + h * hd;
        const float* V_h   = d_V    + h * hd;
        const float* S_h   = d_scores + (long)h * T * T;
        float* d_attn_h    = d_d_attn  + (long)h * T * T;

        /* d_attn_h[T,T] = dout_h[T,hd] @ V_h^T[hd,T] — row-major NT gemm with strided V_h.
         * V_h^T is V_h transposed; V_h is [T,hd] inside V[T,H*hd] with col-stride out_dim.
         * Direct cublas: col-major view → C^T(T,T) = V_h(T,hd) viewed col-major → V_h is
         * column-major [hd,T] with ld=out_dim → CUBLAS_OP_N. dout_h same. */
        CUBLAS_CHECK(cublasSgemm(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, hd,
            &alpha,
            V_h,    out_dim,
            dout_h, out_dim,
            &beta_zero,
            d_attn_h, T));

        /* d_V_h[T,hd] += A_h^T[T,T] × dout_h[T,hd] — strided row-major TN gemm. */
        CUBLAS_CHECK(cublasSgemm(g_cublas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            hd, T, T,
            &alpha,
            dout_h, out_dim,
            S_h,    T,
            &beta_acc,
            d_dV + h * hd, out_dim));
    }

    /* Causal softmax backward across all heads. */
    dim3 grid(H, T);
    kernel_softmax_backward<<<grid, 1>>>(d_d_score, d_scores, d_d_attn, T, H);

    /* Phase 2: per-head compute d_U_h, then dWrb_h, dWra_h, accumulate into dX.
     * Reuse d_d_attn buffer for d_U scratch (no longer needed). */
    for (int h = 0; h < H; h++) {
        const float* Wra_h = d_Wr_combined + (long)h * E * R;
        const float* Wrb_h = d_Wr_combined + wra_total + (long)h * R * T;
        const float* U_h   = d_U      + (long)h * T * R;
        const float* d_score_h = d_d_score + (long)h * T * T;
        float* dWra_h = d_dWr_combined + (long)h * E * R;
        float* dWrb_h = d_dWr_combined + wra_total + (long)h * R * T;

        /* Reuse d_d_attn[h*T*T...] as d_U_h scratch (size T*R ≤ T*T). */
        float* d_U_h_buf = d_d_attn + (long)h * T * T;

        /* d_U_h[T,R] = d_score[T,T] @ Wrb_h^T[T,R] — NT gemm, beta=0 */
        gpu_sgemm_nt_beta(T, R, T, d_score_h, Wrb_h, d_U_h_buf, 0.0f);

        /* d_Wrb_h[R,T] += U_h^T[R,T] @ d_score[T,T] — TN gemm, beta=1 */
        gpu_sgemm_tn_beta(R, T, T, U_h, d_score_h, dWrb_h, 1.0f);

        /* d_X[T,E] += d_U_h[T,R] @ Wra_h^T[R,E] — NT gemm, beta=1 */
        gpu_sgemm_nt_beta(T, E, R, d_U_h_buf, Wra_h, d_dX, 1.0f);

        /* d_Wra_h[E,R] += X^T[E,T] @ d_U_h[T,R] — TN gemm, beta=1 */
        gpu_sgemm_tn_beta(E, R, T, d_X, d_U_h_buf, dWra_h, 1.0f);
    }
}

// ═══════════════════════════════════════════════════════════════════
// SEQ-RMSNORM with optional gamma — forward + backward
// ═══════════════════════════════════════════════════════════════════
//
// y[t,d] = (x[t,d] / rms[t]) * gamma[d]   (gamma optional)
// Reuses kernel_rmsnorm for the no-gamma path. With gamma we apply it
// as a separate per-element step.

__global__ void kernel_apply_gamma(float* y, const float* gamma, int T, int D) {
    int t = blockIdx.x;
    int d_start = threadIdx.x;
    if (t >= T) return;
    for (int d = d_start; d < D; d += blockDim.x)
        y[t * D + d] *= gamma[d];
}

extern "C" void gpu_seq_rmsnorm_gamma(float* d_out, const float* d_in,
                                       const float* d_gamma, int T, int D) {
    /* y = x / rms */
    int threads = D < 256 ? D : 256;
    kernel_rmsnorm<<<T, threads, threads * sizeof(float)>>>(d_out, d_in, T, D);
    if (d_gamma) {
        kernel_apply_gamma<<<T, threads>>>(d_out, d_gamma, T, D);
    }
}

/* Backward: same as kernel_rmsnorm_backward but with gamma support.
 *   has_gamma=0: gx[t,d] = (dout/rms) - x*sum(dout*x)/(D*rms^3)
 *   has_gamma=1: dout_eff = dout * gamma; gx as above with dout_eff;
 *                gg[d] += sum_t dout[t,d] * (x[t,d] / rms[t]).
 */
__global__ void kernel_seq_rmsnorm_backward(float* gx, const float* dout,
                                            const float* x,
                                            const float* gamma,
                                            int T, int D, int has_gamma) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* x_t = x + t * D;
    const float* dout_t = dout + t * D;
    float* gx_t = gx + t * D;

    extern __shared__ float sdata[];
    float local_ss = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local_ss += x_t[d] * x_t[d];
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / D + 1e-6f);
    float rms3 = rms * rms * rms;

    float local_sd = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float de = has_gamma ? dout_t[d] * gamma[d] : dout_t[d];
        local_sd += de * x_t[d];
    }
    sdata[threadIdx.x] = local_sd;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum_dx = sdata[0];

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float de = has_gamma ? dout_t[d] * gamma[d] : dout_t[d];
        gx_t[d] = (de / rms) - (x_t[d] * sum_dx / (D * rms3));
    }
}

/* gamma gradient kernel: gg[d] = Σ_t dout[t,d] * x[t,d] / rms[t]
 * Each block handles one d: T-reduction.
 */
__global__ void kernel_seq_rmsnorm_gamma_grad(float* gg, const float* dout,
                                              const float* x,
                                              int T, int D) {
    int d = blockIdx.x;
    if (d >= D) return;
    extern __shared__ float sdata[];
    float* rms_buf = sdata;          /* T floats */
    /* Compute rms[t] first — but we need x[t,*]. Cooperative across threads in block.
     * Simpler: each thread computes for a t-stripe and accumulates.
     * Re-derive rms[t] inline (cost: T·D adds; D blocks → total T·D^2 — only OK for tiny D).
     * For our case D ≤ 768, T ≤ 256 → 50M ops, fine.
     */
    float local = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const float* x_t = x + t * D;
        float ss = 0;
        for (int dd = 0; dd < D; dd++) ss += x_t[dd] * x_t[dd];
        float rms = sqrtf(ss / D + 1e-6f);
        local += dout[t * D + d] * (x_t[d] / rms);
    }
    rms_buf[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) rms_buf[threadIdx.x] += rms_buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) gg[d] = rms_buf[0];
}

extern "C" void gpu_seq_rmsnorm_backward(float* d_gx, float* d_gg,
                                          const float* d_grad, const float* d_x,
                                          const float* d_gamma, int T, int D) {
    int threads = D < 256 ? D : 256;
    int has_gamma = d_gamma ? 1 : 0;
    kernel_seq_rmsnorm_backward<<<T, threads, threads * sizeof(float)>>>(
        d_gx, d_grad, d_x, d_gamma, T, D, has_gamma);
    if (d_gg && d_gamma) {
        int gthreads = T < 128 ? T : 128;
        kernel_seq_rmsnorm_gamma_grad<<<D, gthreads, gthreads * sizeof(float)>>>(
            d_gg, d_grad, d_x, T, D);
    }
}

// ═══════════════════════════════════════════════════════════════════
// SwiGLU forward + backward
// y[i] = silu(g[i]) * u[i] = g * sigmoid(g) * u
// dgate = dout * u * silu'(g);  silu'(g) = sig + g*sig*(1-sig)
// dup   = dout * silu(g)
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_swiglu(float* out, const float* g, const float* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = g[i];
        float sig = 1.0f / (1.0f + expf(-gv));
        out[i] = gv * sig * u[i];
    }
}

__global__ void kernel_swiglu_backward(float* dg, float* du,
                                       const float* dout, const float* g,
                                       const float* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = g[i];
        float uv = u[i];
        float sig = 1.0f / (1.0f + expf(-gv));
        float silu = gv * sig;
        float dsilu_dg = sig * (1.0f + gv * (1.0f - sig));
        dg[i] = dout[i] * uv * dsilu_dg;
        du[i] = dout[i] * silu;
    }
}

extern "C" void gpu_swiglu(float* d_out, const float* d_g, const float* d_u, int n) {
    kernel_swiglu<<<gpu_blocks(n, 256), 256>>>(d_out, d_g, d_u, n);
}

extern "C" void gpu_swiglu_backward(float* d_dg, float* d_du,
                                     const float* d_dout, const float* d_g,
                                     const float* d_u, int n) {
    kernel_swiglu_backward<<<gpu_blocks(n, 256), 256>>>(d_dg, d_du, d_dout, d_g, d_u, n);
}

// ═══════════════════════════════════════════════════════════════════
// RoPE forward + backward
// Per (t, head, i in head_dim/2):
//   freq  = 1 / fb^(2i/head_dim)
//   angle = t * freq
//   x' = x*cos - y*sin;  y' = x*sin + y*cos
// Backward (transpose of orthogonal rotation):
//   dx = dx'*cos + dy'*sin;  dy = -dx'*sin + dy'*cos
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_rope_forward(float* out, const float* in,
                                    int T, int D, int n_heads, int head_dim, float fb) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    if (t >= T || h >= n_heads || i >= head_dim / 2) return;
    int base = t * D + h * head_dim;
    float freq = 1.0f / powf(fb, 2.0f * i / head_dim);
    float angle = t * freq;
    float c = cosf(angle), s = sinf(angle);
    float x = in[base + 2 * i];
    float y = in[base + 2 * i + 1];
    out[base + 2 * i]     = x * c - y * s;
    out[base + 2 * i + 1] = x * s + y * c;
}

__global__ void kernel_rope_backward(float* gx, const float* gout,
                                     int T, int D, int n_heads, int head_dim, float fb) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    if (t >= T || h >= n_heads || i >= head_dim / 2) return;
    int base = t * D + h * head_dim;
    float freq = 1.0f / powf(fb, 2.0f * i / head_dim);
    float angle = t * freq;
    float c = cosf(angle), s = sinf(angle);
    float dx0 = gout[base + 2 * i];
    float dx1 = gout[base + 2 * i + 1];
    gx[base + 2 * i]     =  dx0 * c + dx1 * s;
    gx[base + 2 * i + 1] = -dx0 * s + dx1 * c;
}

extern "C" void gpu_rope_forward(float* d_out, const float* d_in,
                                  int T, int D, int n_heads, int head_dim, float fb) {
    int half = head_dim / 2;
    if (half <= 0) return;
    dim3 grid(T, n_heads);
    int threads = half;
    kernel_rope_forward<<<grid, threads>>>(d_out, d_in, T, D, n_heads, head_dim, fb);
}

extern "C" void gpu_rope_backward(float* d_gx, const float* d_gout,
                                   int T, int D, int n_heads, int head_dim, float fb) {
    int half = head_dim / 2;
    if (half <= 0) return;
    dim3 grid(T, n_heads);
    int threads = half;
    kernel_rope_backward<<<grid, threads>>>(d_gx, d_gout, T, D, n_heads, head_dim, fb);
}

// ═══════════════════════════════════════════════════════════════════
// Scale (uniform multiply by scalar) — forward + backward
//   out[i] = scale * in[i]
//   gin[i] = scale * gout[i]  (same kernel)
// Reuses cublasSaxpy + zero-then-axpy.
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_scale(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = s * in[i];
}

extern "C" void gpu_scale(float* d_out, const float* d_in, int n, float s) {
    kernel_scale<<<gpu_blocks(n, 256), 256>>>(d_out, d_in, n, s);
}

// ═══════════════════════════════════════════════════════════════════
// Sequential embedding lookup (forward) + scatter-add (backward)
// Forward: y[t, d] = wte[tokens[t], d]
// Backward: dwte[tokens[t], d] += dout[t, d]
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_seq_embed_forward(float* out, const float* wte,
                                         const float* tokens,
                                         int T, int D, int wte_rows) {
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    int tok = (int)tokens[t];
    if (tok < 0) tok = 0;
    if (tok >= wte_rows) tok = wte_rows - 1;
    out[t * D + d] = wte[tok * D + d];
}

__global__ void kernel_seq_embed_backward(float* dwte, const float* dout,
                                          const float* tokens,
                                          int T, int D, int wte_rows) {
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    int tok = (int)tokens[t];
    if (tok < 0) tok = 0;
    if (tok >= wte_rows) tok = wte_rows - 1;
    atomicAdd(&dwte[tok * D + d], dout[t * D + d]);
}

extern "C" void gpu_seq_embedding_forward(float* d_out, const float* d_wte,
                                           const float* d_tokens,
                                           int T, int D, int wte_rows) {
    int threads = 256;
    int dblocks = gpu_blocks(D, threads);
    dim3 grid(T, dblocks);
    kernel_seq_embed_forward<<<grid, threads>>>(d_out, d_wte, d_tokens, T, D, wte_rows);
}

extern "C" void gpu_seq_embedding_backward(float* d_dwte, const float* d_dout,
                                            const float* d_tokens,
                                            int T, int D, int wte_rows) {
    int threads = 256;
    int dblocks = gpu_blocks(D, threads);
    dim3 grid(T, dblocks);
    kernel_seq_embed_backward<<<grid, threads>>>(d_dwte, d_dout, d_tokens, T, D, wte_rows);
}

// ═══════════════════════════════════════════════════════════════════
// Sequential cross-entropy (token-level, masked) — forward + backward
// Forward: per t in [0,T): pick target = (int)tokens[t]; if target == ignore, skip.
//   Compute log-softmax over V; loss[t] = -log p[target]; mean = Σ valid / N_valid.
// Backward: dlogits[t,j] = (softmax_j - delta_{j,target}) * (1/N_valid * dout)
//   Skipped positions: dlogits[t,j] = 0.
// ═══════════════════════════════════════════════════════════════════

__global__ void kernel_seq_cross_entropy_forward(const float* logits,
                                                  const float* tokens,
                                                  float* losses, int* valid_flags,
                                                  int T, int V, int ignore) {
    int t = blockIdx.x;
    if (t >= T) return;
    int target = (int)tokens[t];
    if (target == ignore || target < 0 || target >= V) {
        losses[t] = 0.0f;
        valid_flags[t] = 0;
        return;
    }
    valid_flags[t] = 1;
    const float* l = logits + t * V;
    float mx = l[0];
    for (int j = 1; j < V; j++) if (l[j] > mx) mx = l[j];
    float sum = 0;
    for (int j = 0; j < V; j++) sum += expf(l[j] - mx);
    losses[t] = -((l[target] - mx) - logf(sum + 1e-10f));
}

__global__ void kernel_seq_cross_entropy_backward(float* grad_logits,
                                                   const float* logits,
                                                   const float* tokens,
                                                   int T, int V, int ignore,
                                                   float scale) {
    int t = blockIdx.x;
    if (t >= T) return;
    float* gl = grad_logits + t * V;
    int target = (int)tokens[t];
    if (target == ignore || target < 0 || target >= V) {
        for (int j = 0; j < V; j++) gl[j] = 0.0f;
        return;
    }
    const float* l = logits + t * V;
    float mx = l[0];
    for (int j = 1; j < V; j++) if (l[j] > mx) mx = l[j];
    float sum = 0;
    for (int j = 0; j < V; j++) sum += expf(l[j] - mx);
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int j = 0; j < V; j++) {
        float prob = expf(l[j] - mx) * inv_sum;
        gl[j] = scale * (prob - (j == target ? 1.0f : 0.0f));
    }
}

extern "C" float gpu_seq_cross_entropy(const float* d_logits, const float* d_tokens,
                                        float* d_losses, int* d_valid,
                                        int T, int V, int ignore) {
    kernel_seq_cross_entropy_forward<<<T, 1>>>(d_logits, d_tokens, d_losses, d_valid, T, V, ignore);
    float* h_losses = (float*)malloc(T * sizeof(float));
    int* h_valid = (int*)malloc(T * sizeof(int));
    cudaMemcpy(h_losses, d_losses, T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid, d_valid, T * sizeof(int), cudaMemcpyDeviceToHost);
    float total = 0;
    int n_valid = 0;
    for (int t = 0; t < T; t++) { total += h_losses[t]; n_valid += h_valid[t]; }
    free(h_losses); free(h_valid);
    return n_valid > 0 ? total / n_valid : 0.0f;
}

extern "C" void gpu_seq_cross_entropy_backward(float* d_grad_logits,
                                                const float* d_logits,
                                                const float* d_tokens,
                                                int T, int V, int ignore,
                                                int n_valid) {
    float scale = n_valid > 0 ? 1.0f / n_valid : 0.0f;
    kernel_seq_cross_entropy_backward<<<T, 1>>>(d_grad_logits, d_logits, d_tokens,
                                                T, V, ignore, scale);
}

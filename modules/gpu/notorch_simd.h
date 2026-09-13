// notorch_simd.h — in-house AVX2 + FMA matmul replacing cblas
//
// Drop-in CBLAS shim under -DUSE_SIMD (mutually exclusive with -DUSE_BLAS).
// Zero external dependencies: only <immintrin.h> + <pthread.h> (libc).
//
// Targets x86_64 with AVX2 + FMA (Intel Haswell 2013+, AMD Excavator 2015+).
// 256-bit YMM registers, 8 float32 lanes per register.
//
// Design:
//   - 6×16 register-blocked micro-kernel (12 YMM accumulators, fits Skylake's 16-reg file)
//   - Outer triple-loop with cache blocking (Mc=64, Kc=128, Nc=512) sized for 32KB L1d
//   - Pack A and B into contiguous panels for streaming through micro-kernel
//   - Pthread row-partitioning across the M dimension
//
// Compile: cc -mavx2 -mfma -O2 ... -DUSE_SIMD -lpthread
//
// "GPU goes brrrr — but so does AVX2 if you know how to ask."

#ifndef NOTORCH_SIMD_H
#define NOTORCH_SIMD_H

#ifdef USE_SIMD

#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>

// ── CBLAS API surface ─────────────────────────────────────────────────────
// Mirrors OpenBLAS / Apple Accelerate cblas.h enums and signatures so the
// existing cblas_sgemm / sgemv / sger call sites in notorch.c work unchanged.

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;

// ── Tuning constants ──────────────────────────────────────────────────────
// Block sizes for cache blocking. Sized for i5-8500T (32KB L1d, 256KB L2,
// 9MB shared L3). Mc × Kc panel of A ≈ 32KB, Kc × Nc panel of B ≈ 256KB.
#define NT_SIMD_MR 6     // micro-kernel row block (must match unrolled kernel)
#define NT_SIMD_NR 16    // micro-kernel col block (= 2 × YMM width of 8)
#define NT_SIMD_MC 96    // outer M block — multiple of MR
#define NT_SIMD_KC 256   // outer K block
#define NT_SIMD_NC 1024  // outer N block — multiple of NR

// Default thread count: capped by hw, env-overridable via NT_SIMD_THREADS.
#ifndef NT_SIMD_MAX_THREADS
#define NT_SIMD_MAX_THREADS 16
#endif

// ── 6×16 AVX2+FMA micro-kernel ────────────────────────────────────────────
// Computes C[6, 16] += A[6, k] @ B[k, 16] in registers.
// A is row-packed (stride k between rows). B is col-packed in 16-wide strips.
//
// 12 YMM accumulators (6 rows × 2 ymm-cols of 8 floats each).
// Per k-step: 2 B loads, 6 A broadcasts, 12 FMAs.
//
// `accumulate` controls whether C is read first (true) or zeroed (false).

static inline void nt_simd_micro_6x16(
    const float* __restrict A,  // [6, k] row-packed, stride=k
    const float* __restrict B,  // [k, 16] each row contiguous 16 floats
    float* __restrict C,        // [6, ldc]
    int k,
    int ldc,
    int accumulate)
{
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;

    if (accumulate) {
        c00 = _mm256_loadu_ps(C + 0*ldc + 0);  c01 = _mm256_loadu_ps(C + 0*ldc + 8);
        c10 = _mm256_loadu_ps(C + 1*ldc + 0);  c11 = _mm256_loadu_ps(C + 1*ldc + 8);
        c20 = _mm256_loadu_ps(C + 2*ldc + 0);  c21 = _mm256_loadu_ps(C + 2*ldc + 8);
        c30 = _mm256_loadu_ps(C + 3*ldc + 0);  c31 = _mm256_loadu_ps(C + 3*ldc + 8);
        c40 = _mm256_loadu_ps(C + 4*ldc + 0);  c41 = _mm256_loadu_ps(C + 4*ldc + 8);
        c50 = _mm256_loadu_ps(C + 5*ldc + 0);  c51 = _mm256_loadu_ps(C + 5*ldc + 8);
    } else {
        c00 = _mm256_setzero_ps();  c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();  c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();  c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();  c31 = _mm256_setzero_ps();
        c40 = _mm256_setzero_ps();  c41 = _mm256_setzero_ps();
        c50 = _mm256_setzero_ps();  c51 = _mm256_setzero_ps();
    }

    // Prefetch first cache lines of B
    _mm_prefetch((const char*)(B + 0), _MM_HINT_T0);
    _mm_prefetch((const char*)(B + 16), _MM_HINT_T0);

    for (int p = 0; p < k; p++) {
        __m256 b0 = _mm256_loadu_ps(B + p*16 + 0);
        __m256 b1 = _mm256_loadu_ps(B + p*16 + 8);

        // Prefetch B 8 iterations ahead (B is contiguous 16 floats per row)
        if (p + 8 < k) {
            _mm_prefetch((const char*)(B + (p+8)*16), _MM_HINT_T0);
            _mm_prefetch((const char*)(B + (p+8)*16 + 8), _MM_HINT_T0);
        }

        __m256 a;
        a = _mm256_broadcast_ss(A + 0*k + p);
        c00 = _mm256_fmadd_ps(a, b0, c00);  c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A + 1*k + p);
        c10 = _mm256_fmadd_ps(a, b0, c10);  c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A + 2*k + p);
        c20 = _mm256_fmadd_ps(a, b0, c20);  c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A + 3*k + p);
        c30 = _mm256_fmadd_ps(a, b0, c30);  c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A + 4*k + p);
        c40 = _mm256_fmadd_ps(a, b0, c40);  c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A + 5*k + p);
        c50 = _mm256_fmadd_ps(a, b0, c50);  c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(C + 0*ldc + 0, c00);  _mm256_storeu_ps(C + 0*ldc + 8, c01);
    _mm256_storeu_ps(C + 1*ldc + 0, c10);  _mm256_storeu_ps(C + 1*ldc + 8, c11);
    _mm256_storeu_ps(C + 2*ldc + 0, c20);  _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc + 0, c30);  _mm256_storeu_ps(C + 3*ldc + 8, c31);
    _mm256_storeu_ps(C + 4*ldc + 0, c40);  _mm256_storeu_ps(C + 4*ldc + 8, c41);
    _mm256_storeu_ps(C + 5*ldc + 0, c50);  _mm256_storeu_ps(C + 5*ldc + 8, c51);
}

// ── Edge fallback: scalar C += A @ B for arbitrary [m, k] @ [k, n] ────────
// Used for tail blocks where m < MR or n < NR.

static inline void nt_simd_edge_scalar(
    const float* A, int A_row_stride, int A_col_stride,
    const float* B, int B_row_stride, int B_col_stride,
    float* C, int C_row_stride,
    int m, int n, int k,
    int accumulate)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = accumulate ? C[i*C_row_stride + j] : 0.0f;
            for (int p = 0; p < k; p++) {
                s += A[i*A_row_stride + p*A_col_stride] *
                     B[p*B_row_stride + j*B_col_stride];
            }
            C[i*C_row_stride + j] = s;
        }
    }
}

// ── Pack A panel [Mc, Kc] into row-major contig buffer for kernel ─────────
// A_strided is the source. row_stride / col_stride are its access strides.
// Output: row-major [m, k] where each row of MR contiguous elements is what
// the micro-kernel reads (it reads A + r*k + p).

static inline void nt_simd_pack_A(
    const float* A_src, int row_stride, int col_stride,
    float* A_pack, int m, int k)
{
    if (col_stride == 1) {
        // Fast path: copy contiguous rows via AVX2.
        for (int i = 0; i < m; i++) {
            const float* src = A_src + i*row_stride;
            float* dst = A_pack + i*k;
            int p = 0;
            for (; p + 8 <= k; p += 8) {
                _mm256_storeu_ps(dst + p, _mm256_loadu_ps(src + p));
            }
            for (; p < k; p++) dst[p] = src[p];
        }
    } else {
        // Strided fallback (Trans-A: A is accessed column-major).
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                A_pack[i*k + p] = A_src[i*row_stride + p*col_stride];
            }
        }
    }
}

// ── Pack B panel [Kc, Nc] in NR-wide strips for kernel ────────────────────
// B_strided is the source; row_stride / col_stride are its access strides.
// Output: layout where each NR=16-wide column strip is contiguous,
// kernel reads B_pack + (strip_idx * k * 16) + p*16 + 0..15.

static inline void nt_simd_pack_B(
    const float* B_src, int row_stride, int col_stride,
    float* B_pack, int k, int n)
{
    int strip_count = (n + NT_SIMD_NR - 1) / NT_SIMD_NR;
    for (int s = 0; s < strip_count; s++) {
        int j_base = s * NT_SIMD_NR;
        int j_end  = j_base + NT_SIMD_NR;
        int short_strip = (j_end > n);
        if (short_strip) j_end = n;
        float* dst = B_pack + s * k * NT_SIMD_NR;

        if (col_stride == 1 && !short_strip) {
            // Vectorized fast path: 16 contiguous floats per row of B,
            // copied via 2× 256-bit AVX2 loads/stores. Dominant case
            // (NN forward, NT input-grad).
            for (int p = 0; p < k; p++) {
                __m256 v0 = _mm256_loadu_ps(B_src + p*row_stride + j_base + 0);
                __m256 v1 = _mm256_loadu_ps(B_src + p*row_stride + j_base + 8);
                _mm256_storeu_ps(dst + p*NT_SIMD_NR + 0, v0);
                _mm256_storeu_ps(dst + p*NT_SIMD_NR + 8, v1);
            }
        } else {
            // Strided / edge fallback (TN paths, ragged tail).
            for (int p = 0; p < k; p++) {
                int j;
                for (j = j_base; j < j_end; j++) {
                    dst[p*NT_SIMD_NR + (j - j_base)] = B_src[p*row_stride + j*col_stride];
                }
                for (; j < j_base + NT_SIMD_NR; j++) {
                    dst[p*NT_SIMD_NR + (j - j_base)] = 0.0f;
                }
            }
        }
    }
}

// ── Single-thread GEMM core: C[m,n] = A[m,k] @ B[k,n] ─────────────────────
// A_row_stride / A_col_stride = how A is laid out (handles transpose by stride swap).
// Similarly for B. C is always row-major with C_row_stride = ldc.
// Caller handles alpha/beta — this kernel computes C += A@B (or C = A@B if
// initial_zero). Alpha=1 path optimised; alpha != 1 handled in cblas wrapper.

static void nt_simd_sgemm_block(
    const float* A_src, int A_row_stride, int A_col_stride,
    const float* B_src, int B_row_stride, int B_col_stride,
    float* C, int ldc,
    int m, int n, int k,
    int initial_zero)
{
    // Allocate pack buffers (heap; for production tile through stack via alloca
    // or per-thread arena. Heap is fine for first cut.)
    float* A_pack = (float*)aligned_alloc(64, NT_SIMD_MC * NT_SIMD_KC * sizeof(float));
    float* B_pack = (float*)aligned_alloc(64,
        ((NT_SIMD_NC + NT_SIMD_NR - 1) / NT_SIMD_NR) * NT_SIMD_KC * NT_SIMD_NR * sizeof(float));
    if (!A_pack || !B_pack) {
        // OOM fallback: scalar
        nt_simd_edge_scalar(A_src, A_row_stride, A_col_stride,
                            B_src, B_row_stride, B_col_stride,
                            C, ldc, m, n, k, !initial_zero);
        free(A_pack); free(B_pack);
        return;
    }

    if (initial_zero) {
        for (int i = 0; i < m; i++) memset(C + i*ldc, 0, n * sizeof(float));
    }

    for (int kc = 0; kc < k; kc += NT_SIMD_KC) {
        int kc_size = (k - kc < NT_SIMD_KC) ? (k - kc) : NT_SIMD_KC;

        for (int nc = 0; nc < n; nc += NT_SIMD_NC) {
            int nc_size = (n - nc < NT_SIMD_NC) ? (n - nc) : NT_SIMD_NC;

            // Pack B[kc:kc+kc_size, nc:nc+nc_size] into B_pack
            nt_simd_pack_B(B_src + kc*B_row_stride + nc*B_col_stride,
                           B_row_stride, B_col_stride,
                           B_pack, kc_size, nc_size);

            for (int mc = 0; mc < m; mc += NT_SIMD_MC) {
                int mc_size = (m - mc < NT_SIMD_MC) ? (m - mc) : NT_SIMD_MC;

                // Pack A[mc:mc+mc_size, kc:kc+kc_size]
                nt_simd_pack_A(A_src + mc*A_row_stride + kc*A_col_stride,
                               A_row_stride, A_col_stride,
                               A_pack, mc_size, kc_size);

                // Iterate MR×NR micro-kernel tiles within (mc_size × nc_size)
                int strip_count = (nc_size + NT_SIMD_NR - 1) / NT_SIMD_NR;
                for (int s = 0; s < strip_count; s++) {
                    int j_base = s * NT_SIMD_NR;
                    int j_size = (nc_size - j_base < NT_SIMD_NR) ?
                                 (nc_size - j_base) : NT_SIMD_NR;
                    const float* B_strip = B_pack + s * kc_size * NT_SIMD_NR;

                    for (int i = 0; i < mc_size; i += NT_SIMD_MR) {
                        int i_size = (mc_size - i < NT_SIMD_MR) ?
                                     (mc_size - i) : NT_SIMD_MR;
                        const float* A_block = A_pack + i * kc_size;
                        float* C_block = C + (mc + i) * ldc + (nc + j_base);

                        if (i_size == NT_SIMD_MR && j_size == NT_SIMD_NR) {
                            // Full 6×16 tile — fast path
                            nt_simd_micro_6x16(A_block, B_strip, C_block,
                                               kc_size, ldc, 1);
                        } else {
                            // Edge tile — scalar fallback (rare)
                            // C_block has stride ldc, access C_block[ii*ldc + jj]
                            for (int ii = 0; ii < i_size; ii++) {
                                for (int jj = 0; jj < j_size; jj++) {
                                    float s = C_block[ii*ldc + jj];
                                    for (int p = 0; p < kc_size; p++) {
                                        s += A_block[ii*kc_size + p] *
                                             B_strip[p*NT_SIMD_NR + jj];
                                    }
                                    C_block[ii*ldc + jj] = s;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(A_pack);
    free(B_pack);
}

// ── Persistent thread pool ────────────────────────────────────────────────
// Avoids pthread_create / pthread_join overhead (~10-50µs per call) by
// keeping workers alive on a condvar. Critical for small matmuls (T=128
// shapes) where create/join was dominating wall time.

typedef struct {
    const float* A_src; int A_row_stride; int A_col_stride;
    const float* B_src; int B_row_stride; int B_col_stride;
    float* C; int ldc;
    int m_start; int m_end; int n; int k;
    int initial_zero;
} nt_simd_job;

typedef struct {
    pthread_t thread;
    nt_simd_job* job;
    pthread_mutex_t mu;
    pthread_cond_t  cv_work;       // worker waits here for new job
    pthread_cond_t  cv_done;       // master waits here for completion
    atomic_int      state;         // 0=idle, 1=working, 2=done, 3=shutdown
} nt_simd_worker_state;

#define NT_SIMD_POOL_SIZE NT_SIMD_MAX_THREADS

static nt_simd_worker_state g_pool[NT_SIMD_POOL_SIZE];
static int                  g_pool_n = 0;
static pthread_once_t       g_pool_once = PTHREAD_ONCE_INIT;

static void nt_simd_sgemm_block(
    const float*, int, int, const float*, int, int, float*, int,
    int, int, int, int);   // forward decl

static void* nt_simd_worker_loop(void* arg) {
    nt_simd_worker_state* w = (nt_simd_worker_state*)arg;
    // State machine:
    //   0 = initial (pool just created)
    //   1 = work pending (master set, worker should pick up)
    //   2 = work done  (master will clear before next dispatch)
    //   3 = shutdown
    // Worker only wakes for state == 1 or 3. Master signals cv_work after
    // setting state=1. Worker signals cv_done after setting state=2.
    for (;;) {
        pthread_mutex_lock(&w->mu);
        while (atomic_load(&w->state) != 1 && atomic_load(&w->state) != 3) {
            pthread_cond_wait(&w->cv_work, &w->mu);
        }
        int s = atomic_load(&w->state);
        nt_simd_job* j = w->job;
        pthread_mutex_unlock(&w->mu);

        if (s == 3) break;  // shutdown

        if (s == 1 && j) {
            int m_local = j->m_end - j->m_start;
            if (m_local > 0) {
                nt_simd_sgemm_block(
                    j->A_src + j->m_start * j->A_row_stride, j->A_row_stride, j->A_col_stride,
                    j->B_src, j->B_row_stride, j->B_col_stride,
                    j->C + j->m_start * j->ldc, j->ldc,
                    m_local, j->n, j->k,
                    j->initial_zero);
            }
        }

        pthread_mutex_lock(&w->mu);
        atomic_store(&w->state, 2);  // done; master will reset to 0 (idle proxy) before next dispatch
        pthread_cond_signal(&w->cv_done);
        pthread_mutex_unlock(&w->mu);
    }
    return NULL;
}

static void nt_simd_pool_shutdown(void) {
    for (int i = 0; i < g_pool_n; i++) {
        pthread_mutex_lock(&g_pool[i].mu);
        atomic_store(&g_pool[i].state, 3);
        pthread_cond_signal(&g_pool[i].cv_work);
        pthread_mutex_unlock(&g_pool[i].mu);
    }
    for (int i = 0; i < g_pool_n; i++) {
        pthread_join(g_pool[i].thread, NULL);
        pthread_mutex_destroy(&g_pool[i].mu);
        pthread_cond_destroy(&g_pool[i].cv_work);
        pthread_cond_destroy(&g_pool[i].cv_done);
    }
    g_pool_n = 0;
}

static int nt_simd_thread_count(void) {
    const char* env = getenv("NT_SIMD_THREADS");
    if (env) {
        int n = atoi(env);
        if (n > 0 && n <= NT_SIMD_MAX_THREADS) return n;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > NT_SIMD_MAX_THREADS) n = NT_SIMD_MAX_THREADS;
    return (int)n;
}

static void nt_simd_pool_init_impl(void) {
    g_pool_n = nt_simd_thread_count();
    for (int i = 0; i < g_pool_n; i++) {
        pthread_mutex_init(&g_pool[i].mu, NULL);
        pthread_cond_init(&g_pool[i].cv_work, NULL);
        pthread_cond_init(&g_pool[i].cv_done, NULL);
        atomic_init(&g_pool[i].state, 0);
        g_pool[i].job = NULL;
        pthread_create(&g_pool[i].thread, NULL, nt_simd_worker_loop, &g_pool[i]);
    }
    atexit(nt_simd_pool_shutdown);
}

static inline void nt_simd_pool_ensure(void) {
    pthread_once(&g_pool_once, nt_simd_pool_init_impl);
}

// Dispatch m-rows partition to pool, wait for completion.
static void nt_simd_pool_dispatch(nt_simd_job* jobs, int n_jobs) {
    nt_simd_pool_ensure();
    if (n_jobs > g_pool_n) n_jobs = g_pool_n;

    for (int i = 0; i < n_jobs; i++) {
        pthread_mutex_lock(&g_pool[i].mu);
        g_pool[i].job = &jobs[i];
        atomic_store(&g_pool[i].state, 1);  // work pending
        pthread_cond_signal(&g_pool[i].cv_work);
        pthread_mutex_unlock(&g_pool[i].mu);
    }
    for (int i = 0; i < n_jobs; i++) {
        pthread_mutex_lock(&g_pool[i].mu);
        // Wait until worker sets state to 2 (done). Worker stays asleep
        // after that; we leave state=2 between dispatches. No reset.
        while (atomic_load(&g_pool[i].state) == 1) {
            pthread_cond_wait(&g_pool[i].cv_done, &g_pool[i].mu);
        }
        g_pool[i].job = NULL;
        pthread_mutex_unlock(&g_pool[i].mu);
    }
}

// ── Public CBLAS-shim entry points ────────────────────────────────────────

// C[m,n] = alpha * op(A) @ op(B) + beta * C
// op(A) is m×k, op(B) is k×n.
// For RowMajor: A is [M, lda] if NoTrans (so A[i*lda + p]), else [K, lda] (A[p*lda + i]).
// Identical convention for B vs ldb.
static inline void cblas_sgemm(
    CBLAS_ORDER order,
    CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc)
{
    (void)order; // RowMajor only
    int A_row_stride = (TransA == CblasNoTrans) ? lda : 1;
    int A_col_stride = (TransA == CblasNoTrans) ? 1 : lda;
    int B_row_stride = (TransB == CblasNoTrans) ? ldb : 1;
    int B_col_stride = (TransB == CblasNoTrans) ? 1 : ldb;

    /* CBLAS contract: C ← β·C + α·A@B.
     * Previous wrapper did C := β·C, then C += A@B, then C *= α — which
     * yields α·β·C_orig + α·A@B (wrong whenever β ≠ 0 and α ≠ 1).
     * Fix 2026-05-14 (Arianna Method, neo node): fold α into A via a small
     * scratch buffer so the kernel-accumulated product is already α·A@B.
     * α = 1 fast path stays allocation-free. */
    const float* A_use   = A;
    int A_row_stride_use = A_row_stride;
    int A_col_stride_use = A_col_stride;
    float* alpha_scratch = NULL;
    if (alpha != 1.0f) {
        alpha_scratch = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
        if (alpha_scratch) {
            for (int i = 0; i < M; i++) {
                const float* a_row = A + (size_t)i * A_row_stride;
                float*       s_row = alpha_scratch + (size_t)i * K;
                for (int p = 0; p < K; p++) {
                    s_row[p] = alpha * a_row[(size_t)p * A_col_stride];
                }
            }
            A_use            = alpha_scratch;
            A_row_stride_use = K;
            A_col_stride_use = 1;
        } else {
            fprintf(stderr,
                    "[notorch_simd] cblas_sgemm: malloc(%zu B) for alpha "
                    "scratch failed; alpha=%g lost — result will be incorrect.\n",
                    (size_t)M * (size_t)K * sizeof(float), alpha);
        }
    }

    // Apply beta to C (and zero-init if beta == 0)
    int initial_zero = (beta == 0.0f) ? 1 : 0;
    if (!initial_zero && beta != 1.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i*ldc + j] *= beta;
    }

    // Single-thread fast path for small matmuls — threading overhead would
    // dominate even with persistent pool (signal latency ~5-10µs).
    // Threshold ~256K mul-adds = ~2 GFLOP at 0.1ms gives meaningful work
    // for cross-thread sync to amortize.
    long mnk = (long)M * (long)N * (long)K;
    int nthreads = nt_simd_thread_count();
    if (M < 2 * NT_SIMD_MR || nthreads < 2 || mnk < 256L*1024L) {
        nt_simd_sgemm_block(A_use, A_row_stride_use, A_col_stride_use,
                            B, B_row_stride, B_col_stride,
                            C, ldc,
                            M, N, K,
                            initial_zero);
    } else {
        // Persistent thread pool dispatch — no per-call create/join.
        nt_simd_job jobs[NT_SIMD_MAX_THREADS];
        int rows_per_thread = (M + nthreads - 1) / nthreads;
        rows_per_thread = ((rows_per_thread + NT_SIMD_MR - 1) / NT_SIMD_MR) * NT_SIMD_MR;

        int actual_jobs = 0;
        for (int t = 0; t < nthreads; t++) {
            int m_start = t * rows_per_thread;
            int m_end = m_start + rows_per_thread;
            if (m_start >= M) break;
            if (m_end > M) m_end = M;
            jobs[actual_jobs++] = (nt_simd_job){
                A_use, A_row_stride_use, A_col_stride_use,
                B, B_row_stride, B_col_stride,
                C, ldc,
                m_start, m_end, N, K,
                initial_zero
            };
        }
        nt_simd_pool_dispatch(jobs, actual_jobs);
    }

    free(alpha_scratch);
}

// y[m] = alpha * op(A) @ x + beta * y
// op(A) is m×n if NoTrans, n×m if Trans.
static inline void cblas_sgemv(
    CBLAS_ORDER order,
    CBLAS_TRANSPOSE Trans,
    int M, int N,
    float alpha,
    const float* A, int lda,
    const float* X, int incX,
    float beta,
    float* Y, int incY)
{
    (void)order;
    int out_dim = (Trans == CblasNoTrans) ? M : N;
    int in_dim  = (Trans == CblasNoTrans) ? N : M;

    for (int i = 0; i < out_dim; i++) {
        // Read A row/col according to trans
        const float* A_row = (Trans == CblasNoTrans) ? (A + i*lda) : (A + i);
        int A_step = (Trans == CblasNoTrans) ? 1 : lda;

        // AVX2 dot: 8 floats per FMA
        __m256 acc = _mm256_setzero_ps();
        int p = 0;
        if (A_step == 1 && incX == 1) {
            for (; p + 8 <= in_dim; p += 8) {
                __m256 av = _mm256_loadu_ps(A_row + p);
                __m256 xv = _mm256_loadu_ps(X + p);
                acc = _mm256_fmadd_ps(av, xv, acc);
            }
        }
        // Horizontal sum
        float buf[8];
        _mm256_storeu_ps(buf, acc);
        float dot = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
        // Tail (and the strided fallback)
        for (; p < in_dim; p++) {
            dot += A_row[p*A_step] * X[p*incX];
        }
        if (A_step != 1 || incX != 1) {
            // Re-do as scalar with strides (vector path didn't fire above)
            dot = 0;
            for (int q = 0; q < in_dim; q++)
                dot += A_row[q*A_step] * X[q*incX];
        }
        float old = (beta == 0.0f) ? 0.0f : (beta * Y[i*incY]);
        Y[i*incY] = old + alpha * dot;
    }
}

// A[m,n] += alpha * X[m] @ Y[n]^T  (rank-1 update)
static inline void cblas_sger(
    CBLAS_ORDER order,
    int M, int N,
    float alpha,
    const float* X, int incX,
    const float* Y, int incY,
    float* A, int lda)
{
    (void)order;
    for (int i = 0; i < M; i++) {
        float ax = alpha * X[i*incX];
        if (ax == 0.0f) continue;
        __m256 axv = _mm256_set1_ps(ax);
        float* A_row = A + i*lda;
        int j = 0;
        if (incY == 1) {
            for (; j + 8 <= N; j += 8) {
                __m256 yv = _mm256_loadu_ps(Y + j);
                __m256 av = _mm256_loadu_ps(A_row + j);
                av = _mm256_fmadd_ps(axv, yv, av);
                _mm256_storeu_ps(A_row + j, av);
            }
        }
        for (; j < N; j++) {
            A_row[j] += ax * Y[j*incY];
        }
    }
}

#endif // USE_SIMD
#endif // NOTORCH_SIMD_H

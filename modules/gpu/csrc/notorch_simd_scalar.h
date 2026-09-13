// notorch_simd_scalar.h — DEBUG variant: cblas shim that uses ONLY pure scalar.
// Same API as notorch_simd.h but pure C scalar matmul. Lets us isolate whether
// the nanollama loss-bug is in (a) my AVX kernel, (b) my pthread pool,
// (c) some pre-existing notorch.c path that USE_BLAS=1 activates.
//
// Build: cc -O2 -DUSE_SIMD -I. ... (instead of including notorch_simd.h)

#ifndef NOTORCH_SIMD_SCALAR_H
#define NOTORCH_SIMD_SCALAR_H

#ifdef USE_SIMD

#include <stdlib.h>
#include <string.h>

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;

static inline void cblas_sgemm(
    CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    int M, int N, int K, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc)
{
    (void)order;
    int A_row_stride = (TransA == CblasNoTrans) ? lda : 1;
    int A_col_stride = (TransA == CblasNoTrans) ? 1 : lda;
    int B_row_stride = (TransB == CblasNoTrans) ? ldb : 1;
    int B_col_stride = (TransB == CblasNoTrans) ? 1 : ldb;

    if (beta == 0.0f) {
        for (int i = 0; i < M; i++)
            memset(C + i*ldc, 0, N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i*ldc + j] *= beta;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int p = 0; p < K; p++) {
                s += A[i*A_row_stride + p*A_col_stride] *
                     B[p*B_row_stride + j*B_col_stride];
            }
            C[i*ldc + j] += alpha * s;
        }
    }
}

static inline void cblas_sgemv(
    CBLAS_ORDER order, CBLAS_TRANSPOSE Trans,
    int M, int N, float alpha,
    const float* A, int lda,
    const float* X, int incX,
    float beta, float* Y, int incY)
{
    (void)order;
    int out_dim = (Trans == CblasNoTrans) ? M : N;
    int in_dim  = (Trans == CblasNoTrans) ? N : M;
    int A_row_stride = (Trans == CblasNoTrans) ? lda : 1;
    int A_col_stride = (Trans == CblasNoTrans) ? 1 : lda;

    for (int i = 0; i < out_dim; i++) {
        float s = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            s += A[i*A_row_stride + j*A_col_stride] * X[j*incX];
        }
        float old = (beta == 0.0f) ? 0.0f : (beta * Y[i*incY]);
        Y[i*incY] = old + alpha * s;
    }
}

static inline void cblas_sger(
    CBLAS_ORDER order, int M, int N, float alpha,
    const float* X, int incX, const float* Y, int incY,
    float* A, int lda)
{
    (void)order;
    for (int i = 0; i < M; i++) {
        float ax = alpha * X[i*incX];
        for (int j = 0; j < N; j++) {
            A[i*lda + j] += ax * Y[j*incY];
        }
    }
}

#endif // USE_SIMD
#endif
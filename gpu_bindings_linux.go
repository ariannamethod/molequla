//go:build linux

package main

/*
#include "ariannamethod_cuda.h"
#include <stdlib.h>
*/
import "C"

import (
	"sync/atomic"
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════════════════
// GPU CGO bindings — Linux only.
//
// Wraps the gpu_* primitives from ariannamethod_cuda.h (linked via cgo_aml.go's
// `-DUSE_CUDA -lcudart -lcublas` directives). On macOS / non-linux these
// symbols are not available; gpu_bindings_stub.go provides matching no-op
// signatures so the rest of molequla compiles identically on any host.
//
// Memory model: gpu_alloc returns a device pointer (opaque to Go). Treat as
// unsafe.Pointer. The pointer is valid until gpu_free or gpu_shutdown.
// gpu_cache_weight uploads and caches per-name; gpu_get_weight returns the
// cached device pointer (do not free).
// ═══════════════════════════════════════════════════════════════════════════════

// gpuInitialized tracks whether gpu_init() returned 0 successfully. Avoids
// repeated init attempts. Set atomically; gpuReady reads it.
var gpuInitialized atomic.Bool

// gpuInit initializes the CUDA runtime + cuBLAS handle. Returns 0 on success,
// non-zero if no CUDA hardware / driver mismatch / cuBLAS create failed.
// Idempotent: subsequent calls are no-ops once successful.
func gpuInit() int {
	if gpuInitialized.Load() {
		return 0
	}
	rc := int(C.gpu_init())
	if rc == 0 {
		gpuInitialized.Store(true)
	}
	return rc
}

// gpuShutdown frees the weight cache and destroys the cuBLAS handle. Safe to
// call even if init never succeeded.
func gpuShutdown() {
	if gpuInitialized.Load() {
		C.gpu_shutdown()
		gpuInitialized.Store(false)
	}
}

// gpuReady reports whether the CUDA backend is live and usable.
func gpuReady() bool {
	return gpuInitialized.Load()
}

// gpuAlloc allocates n float32 elements on the GPU. Returns an opaque device
// pointer (nil on failure). Caller must gpuFree to release.
func gpuAlloc(n int) unsafe.Pointer {
	if n <= 0 {
		return nil
	}
	return unsafe.Pointer(C.gpu_alloc(C.int(n)))
}

// gpuFree releases a device pointer obtained from gpuAlloc. Safe on nil.
func gpuFree(p unsafe.Pointer) {
	if p != nil {
		C.gpu_free((*C.float)(p))
	}
}

// gpuUpload copies host data to a device pointer. len(src) must match the n
// elements originally allocated for dst. No-op if either is nil/empty.
func gpuUpload(dst unsafe.Pointer, src []float32) {
	if dst == nil || len(src) == 0 {
		return
	}
	C.gpu_upload((*C.float)(dst), (*C.float)(unsafe.Pointer(&src[0])), C.int(len(src)))
}

// gpuDownload copies n elements from a device pointer to host slice. dst must
// have len(dst) >= n.
func gpuDownload(dst []float32, src unsafe.Pointer, n int) {
	if src == nil || n <= 0 || len(dst) < n {
		return
	}
	C.gpu_download((*C.float)(unsafe.Pointer(&dst[0])), (*C.float)(src), C.int(n))
}

// gpuZero memsets a device buffer to 0.
func gpuZero(p unsafe.Pointer, n int) {
	if p == nil || n <= 0 {
		return
	}
	C.gpu_zero((*C.float)(p), C.int(n))
}

// gpuSgemmNT computes C(M,N) = A(M,K) × B^T(N,K) on cuBLAS. The canonical
// matvec path for our generation hot loop is M=1, N=Nout, K=Nin — i.e.
// out[Nout] = wte_row[Nin] @ W[Nout×Nin]^T.
func gpuSgemmNT(M, N, K int, dA, dB, dC unsafe.Pointer) {
	C.gpu_sgemm_nt(C.int(M), C.int(N), C.int(K), (*C.float)(dA), (*C.float)(dB), (*C.float)(dC))
}

// gpuSgemmNN computes C(M,N) = A(M,K) × B(K,N) (no transpose).
func gpuSgemmNN(M, N, K int, dA, dB, dC unsafe.Pointer) {
	C.gpu_sgemm_nn(C.int(M), C.int(N), C.int(K), (*C.float)(dA), (*C.float)(dB), (*C.float)(dC))
}

// gpuSgemmTN computes C(M,N) = A^T(K,M) × B(K,N) — used by backward dW.
func gpuSgemmTN(M, N, K int, dA, dB, dC unsafe.Pointer) {
	C.gpu_sgemm_tn(C.int(M), C.int(N), C.int(K), (*C.float)(dA), (*C.float)(dB), (*C.float)(dC))
}

// gpuAdd: dOut[i] = dA[i] + dB[i]. Elementwise.
func gpuAdd(dOut, dA, dB unsafe.Pointer, n int) {
	C.gpu_add((*C.float)(dOut), (*C.float)(dA), (*C.float)(dB), C.int(n))
}

// gpuMul: dOut[i] = dA[i] * dB[i].
func gpuMul(dOut, dA, dB unsafe.Pointer, n int) {
	C.gpu_mul((*C.float)(dOut), (*C.float)(dA), (*C.float)(dB), C.int(n))
}

// gpuSiLU: dOut[i] = dIn[i] / (1 + exp(-dIn[i])).
func gpuSiLU(dOut, dIn unsafe.Pointer, n int) {
	C.gpu_silu((*C.float)(dOut), (*C.float)(dIn), C.int(n))
}

// gpuRMSNorm normalises T row vectors of dimension D in place per row. Use
// T=1 for the per-token forward step.
func gpuRMSNorm(dOut, dIn unsafe.Pointer, T, D int) {
	C.gpu_rmsnorm((*C.float)(dOut), (*C.float)(dIn), C.int(T), C.int(D))
}

// gpuCacheWeight uploads h_data under `name` and returns the cache slot index
// (or -1 on failure). Subsequent gpuGetWeight(name) returns the same device
// pointer until gpuMarkAllDirty + re-upload.
func gpuCacheWeight(name string, h []float32) int {
	if !gpuInitialized.Load() || len(h) == 0 {
		return -1
	}
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	return int(C.gpu_cache_weight(cn, (*C.float)(unsafe.Pointer(&h[0])), C.int(len(h))))
}

// gpuGetWeight returns (device pointer, length) for a previously cached
// weight name. Returns (nil, 0) if not cached.
func gpuGetWeight(name string) (unsafe.Pointer, int) {
	if !gpuInitialized.Load() {
		return nil, 0
	}
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	var clen C.int
	p := C.gpu_get_weight(cn, &clen)
	return unsafe.Pointer(p), int(clen)
}

// gpuMarkAllDirty flags every cached weight slot as needing re-upload after
// the next adam / chuck training step on host. The next forward will see
// fresh weights once gpu_sync_dirty_weights() runs (called inside ariannamethod
// hot-path AML scripts).
func gpuMarkAllDirty() {
	if gpuInitialized.Load() {
		C.gpu_mark_all_dirty()
	}
}

// gpuScratch returns a per-slot scratch device buffer of at least n_floats
// capacity. 16 slots available (per notorch_cuda.cu:505). Reuse across
// per-token forward steps to avoid alloc churn.
func gpuScratch(slot, n int) unsafe.Pointer {
	if !gpuInitialized.Load() {
		return nil
	}
	return unsafe.Pointer(C.gpu_scratch(C.int(slot), C.int(n)))
}

// gpuMultiHeadAttention runs full causal multi-head attention on GPU. Q/K/V
// laid out [T × D] row-major; D = n_heads * head_dim. d_scores is a [T*T*nh]
// scratch buffer. Output [T × D].
func gpuMultiHeadAttention(dQ, dK, dV, dOut, dScores unsafe.Pointer, T, D, nHeads int) {
	C.gpu_multi_head_attention(
		(*C.float)(dQ), (*C.float)(dK), (*C.float)(dV),
		(*C.float)(dOut), (*C.float)(dScores),
		C.int(T), C.int(D), C.int(nHeads),
	)
}

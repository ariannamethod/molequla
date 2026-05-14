//go:build !linux

package main

import "unsafe"

// ═══════════════════════════════════════════════════════════════════════════════
// GPU stubs — non-linux builds (darwin / windows / etc).
//
// CUDA toolchain is only wired through cgo_aml.go on Linux. Other platforms
// get pure-Go stubs so the rest of molequla compiles unchanged. gpuReady()
// always returns false → ForwardStep dispatcher routes to the existing CPU
// path everywhere. Signatures mirror gpu_bindings_linux.go exactly.
// ═══════════════════════════════════════════════════════════════════════════════

func gpuInit() int                                                              { return -1 }
func gpuShutdown()                                                              {}
func gpuReady() bool                                                            { return false }
func gpuAlloc(n int) unsafe.Pointer                                             { return nil }
func gpuFree(p unsafe.Pointer)                                                  {}
func gpuUpload(dst unsafe.Pointer, src []float32)                               {}
func gpuDownload(dst []float32, src unsafe.Pointer, n int)                      {}
func gpuZero(p unsafe.Pointer, n int)                                           {}
func gpuSgemmNT(M, N, K int, dA, dB, dC unsafe.Pointer)                         {}
func gpuSgemmNN(M, N, K int, dA, dB, dC unsafe.Pointer)                         {}
func gpuSgemmTN(M, N, K int, dA, dB, dC unsafe.Pointer)                         {}
func gpuAdd(dOut, dA, dB unsafe.Pointer, n int)                                 {}
func gpuMul(dOut, dA, dB unsafe.Pointer, n int)                                 {}
func gpuSiLU(dOut, dIn unsafe.Pointer, n int)                                   {}
func gpuRMSNorm(dOut, dIn unsafe.Pointer, T, D int)                             {}
func gpuCacheWeight(name string, h []float32) int                               { return -1 }
func gpuGetWeight(name string) (unsafe.Pointer, int)                            { return nil, 0 }
func gpuMarkAllDirty()                                                          {}
func gpuScratch(slot, n int) unsafe.Pointer                                     { return nil }
func gpuMultiHeadAttention(dQ, dK, dV, dOut, dScores unsafe.Pointer, T, D, nHeads int) {
}

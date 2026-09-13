//go:build cuda

// CUDA-build linkage for the notorch bridge — links libnotorch_gpu.a
// (notorch.c compiled -DUSE_CUDA, plus notorch_cuda.o) and cuBLAS/cudart
// instead of the plain libnotorch.a the !cuda build takes
// (cgo_notorch_cpu.go). Linkage only: the GPU entry points themselves live
// in modules/gpu, and this package reaches them through gpu_bridge.go.
//
// -DUSE_CUDA is not optional here even though no GPU call is made from this
// file. It changes the shape of nt_tensor in notorch.h (the d_data / gpu_valid
// / cpu_dirty mirror, notorch.h:34) and opens the ariannamethod_cuda.h include
// inside ariannamethod.c (ariannamethod.c:85) that cgo_aml.go compiles — hence
// the -I into modules/gpu/csrc, where that header lives.

package main

/*
#cgo linux CFLAGS: -DUSE_CUDA -I/usr/local/cuda/include -I${SRCDIR}/modules/gpu/csrc
#cgo linux LDFLAGS: -L/usr/local/lib -lnotorch_gpu -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++ -lm
#cgo linux pkg-config: openblas
*/
import "C"

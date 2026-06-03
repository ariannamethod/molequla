//go:build cuda

// CUDA-build linkage + GPU enable for the notorch trainer. Links
// libnotorch_gpu.a (notorch.c compiled -DUSE_CUDA, plus notorch_cuda.o)
// and cuBLAS/cudart, so the notorch training tape dispatches matvecs to
// the device. ntGPUEnable() is the automatic GPU/CPU rule (06_PLAN §8);
// the !cuda counterpart is the no-op stub in gpu_notorch_stub.go.

package main

/*
#cgo linux CFLAGS: -DUSE_CUDA -I/usr/local/cuda/include
#cgo linux LDFLAGS: -L/usr/local/lib -lnotorch_gpu -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++ -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lm
#include <notorch.h>

// notorch_cuda.h pulls in the CUDA runtime headers, which the cgo C
// parser stumbles on. Forward-declare the four GPU entry points the
// trainer needs — the symbols resolve from libnotorch_gpu.a at link time.
int       gpu_init(void);
void      gpu_shutdown(void);
long long nt_gpu_dispatch_count(void);
void      nt_gpu_dispatch_reset(void);
*/
import "C"

// ntGPUEnable runs gpu_init(); on success it switches the notorch tape to
// GPU dispatch (nt_set_gpu_mode) and clears the dispatch counter. On
// failure the trainer stays on CPU/BLAS. Automatic, no flag (06_PLAN §8).
func ntGPUEnable() (bool, string) {
	if C.gpu_init() != 0 {
		return false, "gpu_init() failed — notorch trainer on CPU/BLAS"
	}
	C.nt_set_gpu_mode(1)
	C.nt_gpu_dispatch_reset()
	return true, "notorch trainer on GPU — tape dispatching to cuBLAS"
}

// ntSetGPUForStage keeps the notorch tape on GPU for ALL stages on the CUDA
// build. A stage-gate to CPU for small models was tried (2026-06-03) and
// backfired: the CPU path inside the USE_CUDA build runs ~0.3 steps/s on the
// pod (naive/unthreaded matmul) — ~27x slower than GPU even at child. The fast
// CPU substrate is polygon's !cuda build (proper OpenBLAS), where this is a
// no-op stub. So on the device build, GPU is the only viable path.
func ntSetGPUForStage(stage int) {
	_ = stage
	C.nt_set_gpu_mode(1)
}

// ntGPUDispatchCount returns the cuBLAS dispatch count — criterion-4 proof
// that the training tape's matvecs reached the device.
func ntGPUDispatchCount() int64 { return int64(C.nt_gpu_dispatch_count()) }

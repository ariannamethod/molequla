//go:build !cuda

// notorch-trainer GPU enable — CPU-build stub. The default `go build` has
// no CUDA toolchain: the notorch trainer runs on CPU/BLAS. The real bodies
// (gpu_init / nt_set_gpu_mode) live in notorch_cuda.go, compiled only with
// `-tags cuda`.

package gpu

// NotorchEnable attempts to put the notorch trainer on the GPU. On a non-CUDA
// build it always reports CPU/BLAS — there is nothing to enable.
func NotorchEnable() (bool, string) {
	return false, "trainer on CPU/BLAS (built without -tags cuda)"
}

// NotorchDispatchCount — cuBLAS dispatch count. Always 0 on a non-CUDA build.
func NotorchDispatchCount() int64 { return 0 }

// NotorchSetStage — no-op on a non-CUDA build (trainer always CPU/BLAS).
func NotorchSetStage(stage int) {}

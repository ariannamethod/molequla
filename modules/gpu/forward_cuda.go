//go:build linux && cuda

package gpu

// ═══════════════════════════════════════════════════════════════════════════════
// GPU forward path — Linux only. Inference matvec via cuBLAS sgemm.
//
// Strategy: dispatch only the matvec primitive to the GPU. The surrounding
// transformer logic in ForwardStep (RMSNorm, RoPE, attention scoring, SiLU,
// SwiGLU gating, residual additions) stays on CPU/Go because it operates on
// short vectors where CPU↔GPU transfer cost dominates per-op kernel time.
// Matvec is the only operation in the forward path where weight×activation
// dimensions justify the transfer.
//
// For each MatrixParam tagged with a non-empty gpuKey (set by gpuRefreshWeights
// at generation start) the dispatcher in Matvec calls this package's Matvec
// instead of the BLAS path. Activations arrive as float32, are uploaded to a
// scratch device buffer, sgemm-multiplied against the cached weight and
// downloaded. The whole hop is ~3µs on PCIe 4 plus ~1µs of cuBLAS time for
// typical molequla shapes (NEmbd 16-384, V 643-50K).
//
// Training stays on CPU: gradEnabled.Load() guards the caller in molequla.go
// because the autograd graph requires host-side parent references and there is
// no GPU backward wired here.
// ═══════════════════════════════════════════════════════════════════════════════

// gpuScratchSlots — fixed slot allocation strategy so scratch buffers per
// Matvec call reuse the same device memory across the generation loop and
// allocation churn drops to one-time-only.
const (
	gpuScratchX   = 0 // input activation (float32, max size = max(NEmbd, 4*NEmbd))
	gpuScratchOut = 1 // matvec output  (float32, max size = max vocab/4*NEmbd)
)

// Matvec computes out = W(key) @ x on the GPU and returns the nout results.
// Returns nil on any failure — the caller must fall back to CPU.
//
// Preconditions:
//   - key is non-empty and cached (CacheWeight already ran)
//   - len(x) is the weight's input dimension, nout its output dimension
//   - Ready() is true
func Matvec(key string, x []float32, nout int) []float32 {
	nin := len(x)
	if nin <= 0 || nout <= 0 {
		return nil
	}

	dW, wLen := gpuGetWeight(key)
	if dW == nil || wLen != nout*nin {
		// Weight not cached (or vocab grew since last refresh). Caller falls
		// back to CPU and the next gpuRefreshWeights will fix the cache.
		return nil
	}

	// Activation buffer — reuse slot every step.
	dX := gpuScratch(gpuScratchX, nin)
	dOut := gpuScratch(gpuScratchOut, nout)
	if dX == nil || dOut == nil {
		return nil
	}

	gpuUpload(dX, x)

	// out[Nout] = X[Nin] @ W[Nout × Nin]^T  (M=1 matvec via NT form).
	gpuSgemmNT(1, nout, nin, dX, dW, dOut)

	out := make([]float32, nout)
	gpuDownload(out, dOut, nout)
	return out
}

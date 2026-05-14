//go:build linux

package main

import (
	"fmt"
	"os"
)

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
// at generation start) the dispatcher in Matvec calls MatvecGPU instead of the
// BLAS path. Activations are converted to float32, uploaded to a scratch
// device buffer, sgemm-multiplied against the cached weight, downloaded and
// converted back. The whole hop is ~3µs on PCIe 4 plus ~1µs of cuBLAS time
// for typical molequla shapes (NEmbd 16-384, V 643-50K).
//
// Training stays on CPU: gradEnabled.Load() guards the dispatcher because the
// autograd graph requires host-side parent references and there is no GPU
// backward wired here.
// ═══════════════════════════════════════════════════════════════════════════════

// gpuScratchSlots — fixed slot allocation strategy so scratch buffers per
// Matvec call reuse the same device memory across the generation loop and
// allocation churn drops to one-time-only.
const (
	gpuScratchX   = 0 // input activation (float32, max size = max(NEmbd, 4*NEmbd))
	gpuScratchOut = 1 // matvec output  (float32, max size = max vocab/4*NEmbd)
)

// MatvecGPU computes out = m @ x on the GPU. Returns *Vec with .Data filled
// and no autograd parent. Returns nil on failure (caller must fall back).
//
// Preconditions:
//   - m.gpuKey is non-empty and cached (gpu_cache_weight already ran)
//   - len(x.Data) == m.Nin
//   - gpuReady() is true
//
// Postconditions: returned Vec has len(.Data) == m.Nout, no .children, no
// .backFn (autograd-free).
func (m *MatrixParam) MatvecGPU(x *Vec) *Vec {
	nin := len(x.Data)
	nout := m.Nout
	if nin != m.Nin || nin <= 0 || nout <= 0 {
		return nil
	}

	dW, wLen := gpuGetWeight(m.gpuKey)
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

	// float64 → float32 (cuBLAS sgemm is float32-only). Per-call alloc is
	// small (NEmbd or 4*NEmbd) and tracked by Go GC; if profiling shows hot
	// spot here, hoist into a per-organism scratch slice.
	xF32 := make([]float32, nin)
	for i, v := range x.Data {
		xF32[i] = float32(v)
	}
	gpuUpload(dX, xF32)

	// out[Nout] = X[Nin] @ W[Nout × Nin]^T  (M=1 matvec via NT form).
	gpuSgemmNT(1, nout, nin, dX, dW, dOut)

	// Download and convert back.
	outF32 := make([]float32, nout)
	gpuDownload(outF32, dOut, nout)
	outF64 := make([]float64, nout)
	for i, v := range outF32 {
		outF64[i] = float64(v)
	}

	return NewVec(outF64)
}

// gpuRefreshWeights uploads every entry in gpt.Base into the GPU weight cache
// under its map key (`wte`, `wpe`, `lm_head`, `l{li}.wq`, etc.). Idempotent —
// `gpu_cache_weight` overwrites an existing slot of the same name. Call once
// at the top of GenerateResonant (before the for-step loop) so the cache
// reflects any host-side weight mutations from intervening micro-train bursts.
//
// O(total_weight_elements) per call. For embryo (NEmbd=16, V~600, ~50 named
// weights) this is ~30K floats = sub-millisecond. For adult (NEmbd=384,
// V=50K) it is ~30M floats = tens of milliseconds — still cheap relative to
// a 180-token generation loop that would otherwise spend 5-10ms per token on
// CPU matvec.
func gpuRefreshWeights(gpt *GPT) {
	if !gpuReady() || gpt == nil {
		return
	}
	uploaded := 0
	for name, m := range gpt.Base {
		if m == nil || m.Nout <= 0 || m.Nin <= 0 || len(m.Rows) != m.Nout {
			continue
		}
		// Flatten rows × cols into contiguous float32 buffer.
		flat := make([]float32, m.Nout*m.Nin)
		for i := 0; i < m.Nout; i++ {
			row := m.Rows[i].Data
			base := i * m.Nin
			for j := 0; j < m.Nin && j < len(row); j++ {
				flat[base+j] = float32(row[j])
			}
		}
		if slot := gpuCacheWeight(name, flat); slot >= 0 {
			m.gpuKey = name
			uploaded++
		}
	}
	if os.Getenv("MOLEQULA_GPU_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[gpu] refreshed %d weight slots\n", uploaded)
	}
}

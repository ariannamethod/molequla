package main

import (
	"fmt"
	"os"

	"github.com/ariannamethod/molequla/modules/gpu"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Bridge to modules/gpu — the only GPU code left in the organism package.
//
// Everything that talks to cuBLAS lives in modules/gpu behind `linux && cuda`;
// this file holds what Go forces to stay here, and nothing else: the two
// functions that touch molequla's own types (MatrixParam / Vec / GPT) and the
// thin names the call sites already use. It carries no build tag because the
// package it calls is a no-op on every non-CUDA build — gpu.Ready() is false,
// so both bodies return before they do any work, and the default build compiles
// not one line of cgo from modules/gpu.
// ═══════════════════════════════════════════════════════════════════════════════

func gpuInit() int                { return gpu.Init() }
func gpuReady() bool              { return gpu.Ready() }
func ntGPUEnable() (bool, string) { return gpu.NotorchEnable() }
func ntGPUDispatchCount() int64   { return gpu.NotorchDispatchCount() }
func ntSetGPUForStage(stage int)  { gpu.NotorchSetStage(stage) }

// MatvecGPU computes out = m @ x on the GPU. Returns *Vec with .Data filled
// and no autograd parent. Returns nil on failure (caller must fall back).
//
// Preconditions:
//   - m.gpuKey is non-empty and cached (gpuRefreshWeights already ran)
//   - len(x.Data) == m.Nin
//   - gpuReady() is true
//
// Postconditions: returned Vec has len(.Data) == m.Nout, no .children, no
// .backFn (autograd-free). The float64↔float32 conversion is here because
// cuBLAS sgemm is float32-only and molequla's activations are float64; the
// per-call allocation is small (NEmbd or 4*NEmbd) and tracked by Go GC.
func (m *MatrixParam) MatvecGPU(x *Vec) *Vec {
	nin := len(x.Data)
	nout := m.Nout
	if nin != m.Nin || nin <= 0 || nout <= 0 {
		return nil
	}

	xF32 := make([]float32, nin)
	for i, v := range x.Data {
		xF32[i] = float32(v)
	}

	outF32 := gpu.Matvec(m.gpuKey, xF32, nout)
	if outF32 == nil {
		return nil
	}

	outF64 := make([]float64, nout)
	for i, v := range outF32 {
		outF64[i] = float64(v)
	}
	return NewVec(outF64)
}

// gpuRefreshWeights uploads every entry in gpt.Base into the GPU weight cache
// under its map key (`wte`, `wpe`, `lm_head`, `l{li}.wq`, etc.). Idempotent —
// gpu.CacheWeight overwrites an existing slot of the same name. Call once at
// the top of GenerateResonant (before the for-step loop) so the cache reflects
// any host-side weight mutations from intervening micro-train bursts.
//
// O(total_weight_elements) per call. For embryo (NEmbd=16, V~600, ~50 named
// weights) this is ~30K floats = sub-millisecond. For adult (NEmbd=384,
// V=50K) it is ~30M floats = tens of milliseconds — still cheap relative to
// a 180-token generation loop that would otherwise spend 5-10ms per token on
// CPU matvec.
func gpuRefreshWeights(gpt *GPT) {
	if !gpu.Ready() || gpt == nil {
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
		if gpu.CacheWeight(name, flat) {
			m.gpuKey = name
			uploaded++
		}
	}
	if os.Getenv("MOLEQULA_GPU_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[gpu] refreshed %d weight slots\n", uploaded)
	}
}

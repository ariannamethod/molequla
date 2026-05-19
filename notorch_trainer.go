package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// notorch trainer — molequla's content transformer trained on notorch's C tape.
//
// Replaces the AML-interpreter path (aml_trainer.go): instead of running the
// transformer as a re-parsed AML script per step on the AML core's CPU
// autograd, the model is built once in notorch ops and trained on notorch's
// compiled tape (BLAS, optional CUDA), Chuck optimizer. See
// 06_PLAN_gpu_training.md, Increment 1.
//
// Content model only — RoPE + MHA + SwiGLU, non-parametric RMSNorm. molequla's
// dormant RRPRAM / hybrid heads are Increment 2. model.Base stays the canonical
// float64 weight store; per burst it is mirrored into notorch tensors and back.
//
// Naming: every symbol here is nt-prefixed — molequla already has a (disabled)
// `notorchTrainSteps` Hebbian stub; these must not collide with it.
// ═══════════════════════════════════════════════════════════════════════════════

// ntTapeNeedsReset is set after a growth event (Net2Net changed dims) so the
// next burst wipes the positional Chuck moment slots before training — old
// slots are meaningless once the param set changes (06_PLAN §6, audit S1).
var ntTapeNeedsReset bool

// ntOnGrowth signals the notorch trainer to reset its tape state before the
// next burst. Call whenever MaybeGrowArchitecture has grown the model.
func ntOnGrowth() { ntTapeNeedsReset = true }

// ntOrderedParam pairs a model.Base weight with its key. The slice order is
// fixed and deterministic — Chuck moment slots are positional (keyed by
// registration order), so registration MUST be byte-identical every burst.
// Never derive this from a Go map range (map iteration is randomized).
type ntOrderedParam struct {
	name string
	mp   *MatrixParam
}

// ntContentParams returns the content-transformer weights in a fixed order:
// wte, then per layer {wq,wk,wv,wo,fc_g,fc_v,fc2}, then lm_head. wpe is omitted
// — the trainer uses RoPE for position (06_PLAN §6, audit #3).
func ntContentParams(model *GPT) []ntOrderedParam {
	out := make([]ntOrderedParam, 0, 2+7*model.NLayer)
	out = append(out, ntOrderedParam{"wte", model.Base["wte"]})
	for l := 0; l < model.NLayer; l++ {
		pfx := fmt.Sprintf("l%d.", l)
		for _, suf := range []string{"wq", "wk", "wv", "wo", "fc_g", "fc_v", "fc2"} {
			out = append(out, ntOrderedParam{pfx + suf, model.Base[pfx+suf]})
		}
	}
	out = append(out, ntOrderedParam{"lm_head", model.Base["lm_head"]})
	return out
}

// ntFlattenMatrix copies a MatrixParam (Nout×Nin float64) into a row-major
// float32 slice — the layout notorch tensors expect.
func ntFlattenMatrix(mp *MatrixParam) []float32 {
	flat := make([]float32, mp.Nout*mp.Nin)
	for i := 0; i < mp.Nout && i < len(mp.Rows); i++ {
		if mp.Rows[i] == nil {
			continue
		}
		row := mp.Rows[i].Data
		for j := 0; j < mp.Nin && j < len(row); j++ {
			flat[i*mp.Nin+j] = float32(row[j])
		}
	}
	return flat
}

// ntUnflattenMatrix writes a row-major float32 slice back into a MatrixParam.
func ntUnflattenMatrix(mp *MatrixParam, flat []float32) {
	for i := 0; i < mp.Nout && i < len(mp.Rows); i++ {
		if mp.Rows[i] == nil {
			continue
		}
		row := mp.Rows[i].Data
		for j := 0; j < mp.Nin && j < len(row); j++ {
			if i*mp.Nin+j < len(flat) {
				row[j] = float64(flat[i*mp.Nin+j])
			}
		}
	}
}

// ntBuildForward builds molequla's content transformer on the active notorch
// tape and returns the cross-entropy loss tape index. pIdx holds the param
// tape indices in ntContentParams order; tokIdx/tgtIdx are input tape indices.
func ntBuildForward(model *GPT, pIdx []int, tokIdx, tgtIdx, T, vocab int) int {
	D := model.NEmbd
	headDim := D / model.NHead
	wte := pIdx[0]
	lmHead := pIdx[len(pIdx)-1]

	h := ntSeqEmbedding(wte, -1, tokIdx, T, D) // WTE only — RoPE handles position
	for l := 0; l < model.NLayer; l++ {
		b := 1 + l*7
		wq, wk, wv, wo := pIdx[b], pIdx[b+1], pIdx[b+2], pIdx[b+3]
		fcG, fcV, fc2 := pIdx[b+4], pIdx[b+5], pIdx[b+6]

		hn := ntSeqRMSNorm(h, -1, T, D) // gamma -1 → non-parametric (matches molequla)
		q := ntRope(ntSeqLinear(wq, hn, T), T, headDim)
		k := ntRope(ntSeqLinear(wk, hn, T), T, headDim)
		v := ntSeqLinear(wv, hn, T)
		attn := ntMHCausalAttention(q, k, v, T, headDim)
		h = ntAdd(h, ntSeqLinear(wo, attn, T))

		hn = ntSeqRMSNorm(h, -1, T, D)
		gate := ntSilu(ntSeqLinear(fcG, hn, T))
		up := ntSeqLinear(fcV, hn, T)
		h = ntAdd(h, ntSeqLinear(fc2, ntMul(gate, up), T))
	}
	hf := ntSeqRMSNorm(h, -1, T, D)
	logits := ntSeqLinear(lmHead, hf, T)
	return ntSeqCrossEntropy(logits, tgtIdx, T, vocab)
}

// ntTrainCore runs `steps` training steps of molequla's content model on
// notorch. lrFor(step) supplies the per-step learning rate. Caller holds
// model.mu. Returns (avg loss, counted steps).
// Returns (avg loss, counted steps, step-loop wall ms) — the wall time is the
// pure training cost, criterion-2 metric (06_PLAN §11.2), measured over the
// step loop only, excluding the per-burst weight mirror in/out.
func ntTrainCore(model *GPT, tok *EvolvingTokenizer, docs []string, steps, seqLen int, lrFor func(int) float64) (float64, int, float64) {
	if len(docs) == 0 || steps <= 0 {
		return 0, 0, 0
	}
	vocab := tok.VocabSize
	params := ntContentParams(model)

	// Mirror model.Base weights into notorch tensors (created once per burst).
	tensors := make([]ntTensor, len(params))
	for i, p := range params {
		t := ntTensorNew2D(p.mp.Nout, p.mp.Nin)
		ntTensorSet(t, ntFlattenMatrix(p.mp))
		tensors[i] = t
	}
	defer func() {
		for _, t := range tensors {
			ntTensorFree(t)
		}
	}()

	// Post-growth: wipe positional Chuck slots before the first step (S1).
	if ntTapeNeedsReset {
		ntTapeDestroy()
		ntTapeNeedsReset = false
	}

	guard := newNTNanGuard()
	tokBuf := make([]float32, seqLen)
	tgtBuf := make([]float32, seqLen)
	var lossSum float64
	var lossN int

	t0 := time.Now()
	for step := 0; step < steps; step++ {
		ids := tok.Encode(docs[rand.Intn(len(docs))])
		if len(ids) < 2 {
			continue
		}
		start := 0
		if len(ids) > seqLen+1 {
			start = rand.Intn(len(ids) - seqLen - 1)
		}
		for i := 0; i < seqLen; i++ {
			idx := start + i
			if idx < len(ids) {
				tokBuf[i] = float32(ids[idx])
			} else {
				tokBuf[i] = 0
			}
			if idx+1 < len(ids) {
				tgtBuf[i] = float32(ids[idx+1])
			} else {
				tgtBuf[i] = 0
			}
		}

		ntTapeStart()
		// Register params in the fixed ntContentParams order (B1).
		pIdx := make([]int, len(tensors))
		for i, t := range tensors {
			pIdx[i] = ntTapeParam(t)
		}
		ntTapeNoDecay(pIdx[0]) // wte — no weight decay on embeddings

		tokT := ntTensorNew(seqLen)
		ntTensorSet(tokT, tokBuf)
		tgtT := ntTensorNew(seqLen)
		ntTensorSet(tgtT, tgtBuf)
		tokIdx := ntTapeInput(tokT)
		tgtIdx := ntTapeInput(tgtT)
		ntTensorFree(tokT)
		ntTensorFree(tgtT)

		lossIdx := ntBuildForward(model, pIdx, tokIdx, tgtIdx, seqLen, vocab)
		loss := ntEntryScalar(lossIdx)
		ntTapeBackward(lossIdx)
		if guard.check() {
			ntTapeClipGrads(1.0)
			ntTapeChuckStep(lrFor(step), loss)
		}
		ntTapeClear()

		if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
			lossSum += loss
			lossN++
		}
		model.globalStep++
	}
	elapsedMs := float64(time.Since(t0).Microseconds()) / 1000.0

	// Mirror trained weights back into the canonical model.Base store.
	for i, p := range params {
		ntUnflattenMatrix(p.mp, ntTensorGet(tensors[i], p.mp.Nout*p.mp.Nin))
	}
	if lossN > 0 {
		return lossSum / float64(lossN), lossN, elapsedMs
	}
	return 0, 0, elapsedMs
}

// ntBurstTrain — ecology micro-burst on the notorch path. Mirrors amlBurstTrain
// (aml_trainer.go:252): fixed burst LR scaled by embryo/current embd.
func ntBurstTrain(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, burstLR float64) {
	if CFG.Trainer == "aml" {
		amlBurstTrain(model, tok, docs, steps, burstLR)
		return
	}
	if len(docs) == 0 || steps <= 0 {
		return
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	embryoEmbd := CFG.GrowthStages[0][1]
	lr := burstLR * float64(embryoEmbd) / float64(model.NEmbd)
	avg, n, ms := ntTrainCore(model, tok, docs, steps, model.BlockSize, func(int) float64 { return lr })
	if model.growthFreezeRemaining > 0 {
		model.growthFreezeRemaining -= steps
		if model.growthFreezeRemaining < 0 {
			model.growthFreezeRemaining = 0
		}
	}
	if n > 0 {
		fmt.Printf("[notorch] burst complete: %d steps, avg loss %.4f | %.0fms %.1f steps/s | gpu-dispatch=%d\n",
			steps, avg, ms, ntStepsPerSec(n, ms), ntGPUDispatchCount())
	}
}

// ntStepsPerSec — steps/sec from a counted-step total and wall ms (criterion 2).
func ntStepsPerSec(n int, ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	return float64(n) / (ms / 1000.0)
}

// ntWarmupTrain — per-stage warmup on the notorch path. Mirrors amlTrainSteps
// (aml_trainer.go:139): cosine LR driven by molequla's cosineLR (so the
// post-growth Chuck-state reset, S1, costs no LR-schedule continuity — the
// schedule lives in cosineLR, not in Chuck's internal macro counter).
func ntWarmupTrain(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, overrides ...int) {
	if CFG.Trainer == "aml" {
		amlTrainSteps(model, tok, docs, steps, overrides...)
		return
	}
	if len(docs) == 0 || steps <= 0 {
		return
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	seqLen := model.BlockSize
	if len(overrides) > 0 && overrides[0] > 0 && overrides[0] < seqLen {
		seqLen = overrides[0]
	}
	embryoEmbd := CFG.GrowthStages[0][1]
	g0 := model.globalStep
	lrFor := func(step int) float64 {
		gs := g0 + step
		lr := cosineLR(gs, gs-model.growthStepOffset)
		lr *= float64(embryoEmbd) / float64(model.NEmbd)
		if model.growthFreezeRemaining > 0 {
			lr *= CFG.PostGrowthLRScale
		}
		return lr
	}
	avg, n, ms := ntTrainCore(model, tok, docs, steps, seqLen, lrFor)
	if model.growthFreezeRemaining > 0 {
		model.growthFreezeRemaining -= steps
		if model.growthFreezeRemaining < 0 {
			model.growthFreezeRemaining = 0
		}
	}
	if n > 0 {
		fmt.Printf("[notorch] warmup complete: %d steps, avg loss %.4f | %.0fms %.1f steps/s | gpu-dispatch=%d\n",
			steps, avg, ms, ntStepsPerSec(n, ms), ntGPUDispatchCount())
	}
}

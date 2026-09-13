package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// notorch trainer — molequla's transformer trained on notorch's C tape.
//
// Replaces the AML-interpreter path (aml_trainer.go): instead of running the
// transformer as a re-parsed AML script per step on the AML core's CPU
// autograd, the model is built once in notorch ops and trained on notorch's
// compiled tape (BLAS, optional CUDA), Chuck optimizer. See
// 06_PLAN_gpu_training.md, Increment 1.
//
// The tape computes the function inference runs (repair 2b, MOLEQULALOG2.md
// 2026-09-13): wte[tok] + wpe[pos], RoPE + MHA (+ the op-33 RRPRAM blend on
// hybrid heads), SwiGLU, non-parametric RMSNorm, both residual branches scaled
// by residualAlpha = 1/sqrt(NLayer), every linear map applied with its delta
// adapters (base·x + Σ α_i·A_i·B_i·x), and a cross-entropy masked to the
// positions a document actually has. parity_test.go holds the gate: the tape
// loss and Go's LossOnSequence agree on one sequence.
//
// model.Base and model.Deltas stay the canonical float64 weight store; per
// burst they are mirrored into notorch tensors and back.
//
// Naming: every symbol here is nt-prefixed — molequla already has a (disabled)
// `notorchTrainStep` Hebbian stub; these must not collide with it.
// ═══════════════════════════════════════════════════════════════════════════════

// ntTapeNeedsReset is set after a growth event (Net2Net changed dims, or a new
// delta module was appended) so the next burst wipes the positional Chuck
// moment slots before training — old slots are meaningless once the param set
// changes (06_PLAN §6, audit S1).
var ntTapeNeedsReset bool

// ntOnGrowth signals the notorch trainer to reset its tape state before the
// next burst. Call whenever MaybeGrowArchitecture has grown the model or a
// delta module has been appended.
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
// wte, wpe, then per layer {wq,wk,wv,wo,fc_g,fc_v,fc2}, then lm_head. wpe is a
// trained parameter because inference adds it to every token (ForwardStep,
// wte[tok] + wpe[pos]) on top of RoPE, and the AML trainer trains it too; a
// trainer that omits it leaves the mouth running on a random positional table.
func ntContentParams(model *GPT) []ntOrderedParam {
	out := make([]ntOrderedParam, 0, 3+7*model.NLayer)
	out = append(out, ntOrderedParam{"wte", model.Base["wte"]})
	out = append(out, ntOrderedParam{"wpe", model.Base["wpe"]})
	for l := 0; l < model.NLayer; l++ {
		pfx := fmt.Sprintf("l%d.", l)
		for _, suf := range []string{"wq", "wk", "wv", "wo", "fc_g", "fc_v", "fc2"} {
			out = append(out, ntOrderedParam{pfx + suf, model.Base[pfx+suf]})
		}
	}
	out = append(out, ntOrderedParam{"lm_head", model.Base["lm_head"]})
	return out
}

// ntDeltaNames lists the delta-adapter keys inference applies, in a fixed
// order: per layer {wq,wk,wv,wo,fc_g,fc_v,fc2}, then lm_head. The per-head
// `l%d.h%d.w_pattern` adapters that AddDeltaModule also allocates are not
// applied by applyWithDeltas (w_pattern is retired) and are not trained.
func ntDeltaNames(model *GPT) []string {
	out := make([]string, 0, 1+7*model.NLayer)
	for l := 0; l < model.NLayer; l++ {
		pfx := fmt.Sprintf("l%d.", l)
		for _, suf := range []string{"wq", "wk", "wv", "wo", "fc_g", "fc_v", "fc2"} {
			out = append(out, pfx+suf)
		}
	}
	return append(out, "lm_head")
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

// ntTensorFromMatrix mirrors a MatrixParam into a fresh 2-D notorch tensor.
func ntTensorFromMatrix(mp *MatrixParam) ntTensor {
	t := ntTensorNew2D(mp.Nout, mp.Nin)
	ntTensorSet(t, ntFlattenMatrix(mp))
	return t
}

// ntDeltaMirror is one delta adapter (module i, weight name) mirrored into
// notorch tensors, with its effective alpha and, after register(), its tape
// indices.
type ntDeltaMirror struct {
	module int
	name   string
	da     *DeltaAdapter
	a, b   ntTensor
	alpha  float64 // ActiveAlpha[i] · deltaAlphaScale at mirror time
	aIdx   int
	bIdx   int
}

// ntMirror holds one burst's notorch-side copy of the model: content weights,
// delta adapters, and (on hybrid models) the packed RRPRAM factors with their
// frozen gate vectors. Registration order on the tape is fixed — content, then
// deltas, then RRPRAM — because Chuck's moment slots are positional.
type ntMirror struct {
	model   *GPT
	seqLen  int
	params  []ntOrderedParam
	tensors []ntTensor
	deltas  []ntDeltaMirror
	hasRR   bool
	wr      []ntTensor
	gateC   []ntTensor
	gateR   []ntTensor
	// tape indices, valid between register() and the next ntTapeClear
	pIdx     []int
	wrIdx    []int
	gateCIdx []int
	gateRIdx []int
}

// ntNewMirror mirrors the model for a burst at sequence length seqLen.
func ntNewMirror(model *GPT, seqLen int) *ntMirror {
	m := &ntMirror{model: model, seqLen: seqLen, params: ntContentParams(model)}
	m.tensors = make([]ntTensor, len(m.params))
	for i, p := range m.params {
		m.tensors[i] = ntTensorFromMatrix(p.mp)
	}
	for i, mod := range model.Deltas {
		alpha := 1.0
		if i < len(model.ActiveAlpha) {
			alpha = model.ActiveAlpha[i]
		}
		alpha *= model.deltaAlphaScale
		for _, name := range ntDeltaNames(model) {
			da, ok := mod[name]
			if !ok || da == nil || da.A == nil || da.B == nil {
				continue
			}
			m.deltas = append(m.deltas, ntDeltaMirror{
				module: i, name: name, da: da, alpha: alpha,
				a: ntTensorFromMatrix(da.A), b: ntTensorFromMatrix(da.B),
			})
		}
	}
	m.hasRR = layerHasHybrid()
	m.wr = make([]ntTensor, model.NLayer)
	m.gateC = make([]ntTensor, model.NLayer)
	m.gateR = make([]ntTensor, model.NLayer)
	if m.hasRR {
		for l := 0; l < model.NLayer; l++ {
			combined := ntPackWr(model, l)
			if combined == nil {
				continue
			}
			wt := ntTensorNew(len(combined))
			ntTensorSet(wt, combined)
			m.wr[l] = wt
			gc, gr := ntBuildGateVectors(model, l, seqLen)
			gct := ntTensorNew(len(gc))
			ntTensorSet(gct, gc)
			grt := ntTensorNew(len(gr))
			ntTensorSet(grt, gr)
			m.gateC[l], m.gateR[l] = gct, grt
		}
	}
	return m
}

// register puts every mirrored tensor on the active tape in the fixed order
// and records the tape indices. Call once per step after ntTapeStart().
func (m *ntMirror) register() {
	m.pIdx = make([]int, len(m.tensors))
	for i, t := range m.tensors {
		m.pIdx[i] = ntTapeParam(t)
	}
	ntTapeNoDecay(m.pIdx[0]) // wte — no weight decay on embeddings
	ntTapeNoDecay(m.pIdx[1]) // wpe — same rule for the positional table
	for i := range m.deltas {
		d := &m.deltas[i]
		d.aIdx = ntTapeParam(d.a)
		ntTapeNoDecay(d.aIdx) // low-rank factors: no weight decay (double-shrink)
		d.bIdx = ntTapeParam(d.b)
		ntTapeNoDecay(d.bIdx)
	}
	m.wrIdx = make([]int, m.model.NLayer)
	m.gateCIdx = make([]int, m.model.NLayer)
	m.gateRIdx = make([]int, m.model.NLayer)
	for l := 0; l < m.model.NLayer; l++ {
		m.wrIdx[l] = -1
		if m.wr[l] == nil {
			continue
		}
		m.wrIdx[l] = ntTapeParam(m.wr[l])
		ntTapeNoDecay(m.wrIdx[l])
		m.gateCIdx[l] = ntTapeParamFrozen(m.gateC[l]) // frozen → no optimizer slot
		m.gateRIdx[l] = ntTapeParamFrozen(m.gateR[l])
	}
}

// linear applies weight `name` (tape index wIdx) to xIdx and adds every delta
// adapter inference would add: base·x + Σ_i α_i·A_i·(B_i·x).
func (m *ntMirror) linear(name string, wIdx, xIdx, T int) int {
	y := ntSeqLinear(wIdx, xIdx, T)
	for i := range m.deltas {
		d := &m.deltas[i]
		if d.name != name || d.alpha == 0 {
			continue
		}
		bx := ntSeqLinear(d.bIdx, xIdx, T)
		y = ntAdd(y, ntScale(ntSeqLinear(d.aIdx, bx, T), d.alpha))
	}
	return y
}

// pullBack copies every trained tensor into the canonical Go-side store.
func (m *ntMirror) pullBack() {
	for i, p := range m.params {
		ntTensorSyncCPU(m.tensors[i])
		ntUnflattenMatrix(p.mp, ntTensorGet(m.tensors[i], p.mp.Nout*p.mp.Nin))
	}
	for i := range m.deltas {
		d := &m.deltas[i]
		ntTensorSyncCPU(d.a)
		ntUnflattenMatrix(d.da.A, ntTensorGet(d.a, d.da.A.Nout*d.da.A.Nin))
		ntTensorSyncCPU(d.b)
		ntUnflattenMatrix(d.da.B, ntTensorGet(d.b, d.da.B.Nout*d.da.B.Nin))
	}
	if m.hasRR {
		for l := 0; l < m.model.NLayer; l++ {
			if m.wr[l] == nil {
				continue
			}
			a := m.model.Base[fmt.Sprintf("l%d.wr_a", l)]
			b := m.model.Base[fmt.Sprintf("l%d.wr_b", l)]
			if a == nil || b == nil {
				continue
			}
			ntTensorSyncCPU(m.wr[l])
			ntUnpackWr(m.model, l, ntTensorGet(m.wr[l], a.Nout*a.Nin+b.Nout*b.Nin))
		}
	}
}

// free releases every mirrored tensor.
func (m *ntMirror) free() {
	for _, t := range m.tensors {
		ntTensorFree(t)
	}
	for i := range m.deltas {
		ntTensorFree(m.deltas[i].a)
		ntTensorFree(m.deltas[i].b)
	}
	for l := range m.wr {
		for _, t := range []ntTensor{m.wr[l], m.gateC[l], m.gateR[l]} {
			if t != nil {
				ntTensorFree(t)
			}
		}
	}
}

// ntBuildForward builds molequla's transformer on the active notorch tape from
// a registered mirror and returns (loss tape index, logits tape index). The
// graph is the function ForwardStep runs: wte[tok] + wpe[pos]; per layer
// pre-norm RMSNorm, q/k/v with deltas, RoPE on q and k, causal MHA (blended
// with the op-33 RRPRAM head on hybrid layers), wo with deltas, residual scaled
// by residualAlpha; pre-norm, SwiGLU with deltas, fc2 with deltas, residual
// scaled by residualAlpha; final RMSNorm, lm_head with deltas; cross-entropy
// masked to the positions in maskIdx (mean over the unmasked count).
func ntBuildForward(m *ntMirror, tokIdx, tgtIdx, maskIdx, T, vocab int) (int, int) {
	model := m.model
	D := model.NEmbd
	headDim := D / model.NHead
	alpha := model.residualAlpha
	wte := m.pIdx[0]
	wpe := m.pIdx[1]
	lmHead := m.pIdx[len(m.pIdx)-1]

	h := ntSeqEmbedding(wte, wpe, tokIdx, T, D) // wte[tok] + wpe[pos], as inference does; RoPE on q/k below
	for l := 0; l < model.NLayer; l++ {
		b := 2 + l*7
		pfx := fmt.Sprintf("l%d.", l)
		wq, wk, wv, wo := m.pIdx[b], m.pIdx[b+1], m.pIdx[b+2], m.pIdx[b+3]
		fcG, fcV, fc2 := m.pIdx[b+4], m.pIdx[b+5], m.pIdx[b+6]

		hn := ntSeqRMSNorm(h, -1, T, D) // gamma -1 → non-parametric (matches molequla)
		q := ntRope(m.linear(pfx+"wq", wq, hn, T), T, headDim)
		k := ntRope(m.linear(pfx+"wk", wk, hn, T), T, headDim)
		v := m.linear(pfx+"wv", wv, hn, T)
		attn := ntMHCausalAttention(q, k, v, T, headDim)

		// Inc2: low-rank RRPRAM head (Resonance form, op 33), output-level blend.
		// rrpram_out = (xn @ Wr_a) @ Wr_b → causal softmax → @ v, packed over all
		// heads (full D input, same v as content). Per-head frozen gate masks
		// content-only heads (gateR=0) and weights hybrid heads by sigmoid(alpha):
		//   out = gateC ⊙ content_out + gateR ⊙ rrpram_out
		if l < len(m.wrIdx) && m.wrIdx[l] >= 0 {
			rAttn := ntRrpramLowrankAttention(m.wrIdx[l], hn, v, T, D, model.NHead, headDim)
			attn = ntAdd(ntMul(attn, m.gateCIdx[l]), ntMul(rAttn, m.gateRIdx[l]))
		}
		h = ntAdd(h, ntScale(m.linear(pfx+"wo", wo, attn, T), alpha))

		hn = ntSeqRMSNorm(h, -1, T, D)
		gate := ntSilu(m.linear(pfx+"fc_g", fcG, hn, T))
		up := m.linear(pfx+"fc_v", fcV, hn, T)
		h = ntAdd(h, ntScale(m.linear(pfx+"fc2", fc2, ntMul(gate, up), T), alpha))
	}
	hf := ntSeqRMSNorm(h, -1, T, D)
	logits := m.linear("lm_head", lmHead, hf, T)
	return ntSeqCrossEntropyMasked(logits, tgtIdx, maskIdx, T, vocab), logits
}

// ntPackWr flattens the per-layer factors wr_a [NHead·NEmbd × R] then
// wr_b [NHead·R × BlockSize] into the single combined buffer notorch op-33 reads
// (all Wr_a then all Wr_b, row-major). Returns nil if the layer has no factors.
func ntPackWr(model *GPT, l int) []float32 {
	a := model.Base[fmt.Sprintf("l%d.wr_a", l)]
	b := model.Base[fmt.Sprintf("l%d.wr_b", l)]
	if a == nil || b == nil {
		return nil
	}
	out := make([]float32, 0, a.Nout*a.Nin+b.Nout*b.Nin)
	out = append(out, ntFlattenMatrix(a)...)
	out = append(out, ntFlattenMatrix(b)...)
	return out
}

// ntUnpackWr splits a trained combined buffer back into the wr_a / wr_b Base
// matrices (the canonical Go-side store), mirroring ntPackWr.
func ntUnpackWr(model *GPT, l int, combined []float32) {
	a := model.Base[fmt.Sprintf("l%d.wr_a", l)]
	b := model.Base[fmt.Sprintf("l%d.wr_b", l)]
	if a == nil || b == nil {
		return
	}
	aLen := a.Nout * a.Nin
	if aLen > len(combined) {
		return
	}
	ntUnflattenMatrix(a, combined[:aLen])
	ntUnflattenMatrix(b, combined[aLen:])
}

// ntBuildGateVectors returns the per-head output-level blend masks for layer l,
// each of length T·NEmbd. content-out weight gateC = 1 - g_h, rrpram-out weight
// gateR = g_h, where g_h = sigmoid(alpha_{l,h}) for a hybrid head and 0 for a
// content head (so content heads stay pure content and their factors get zero
// gradient). The gate is FROZEN this increment — sigmoid is precomputed Go-side,
// keeping nt_sigmoid/nt_scale_by_t off the tape (notorch GPU-sync bug class).
func ntBuildGateVectors(model *GPT, l, T int) (gateC, gateR []float32) {
	D := model.NEmbd
	hd := model.HeadDim
	htypes := headTypesForNHead(model.NHead)
	hg := make([]float32, model.NHead)
	for h := 0; h < model.NHead; h++ {
		g := float32(0.0) // content head → pure content (gateR = 0)
		if h < len(htypes) && (htypes[h] == "hybrid" || htypes[h] == "rrpram") {
			if mp := model.Base[fmt.Sprintf("l%d.h%d.alpha", l, h)]; mp != nil &&
				len(mp.Rows) > 0 && len(mp.Rows[0].Data) > 0 {
				g = float32(1.0 / (1.0 + math.Exp(-mp.Rows[0].Data[0]))) // sigmoid(alpha)
			} else {
				g = float32(1.0 / (1.0 + math.Exp(-CFG.HybridAlphaInit)))
			}
		}
		hg[h] = g
	}
	gateC = make([]float32, T*D)
	gateR = make([]float32, T*D)
	for t := 0; t < T; t++ {
		for h := 0; h < model.NHead; h++ {
			for d := 0; d < hd; d++ {
				idx := t*D + h*hd + d
				gateR[idx] = hg[h]
				gateC[idx] = 1.0 - hg[h]
			}
		}
	}
	return gateC, gateR
}

// ntWindow fills one training window from ids starting at `start`: tokens,
// next-token targets, and a mask that is 1 only where a real target exists.
// Positions past the document are token 0 with mask 0, so they price nothing
// and train nothing.
func ntWindow(ids []int, start, T int, tok, tgt, mask []float32) {
	for i := 0; i < T; i++ {
		idx := start + i
		tok[i], tgt[i], mask[i] = 0, 0, 0
		if idx < len(ids) {
			tok[i] = float32(ids[idx])
		}
		if idx+1 < len(ids) {
			tgt[i] = float32(ids[idx+1])
			mask[i] = 1
		}
	}
}

// ntPushInputs records tokens/targets/mask as tape inputs and returns their
// indices; the tensors are released (the tape holds its own references).
func ntPushInputs(tok, tgt, mask []float32) (tokIdx, tgtIdx, maskIdx int) {
	tokT := ntTensorNew(len(tok))
	ntTensorSet(tokT, tok)
	tgtT := ntTensorNew(len(tgt))
	ntTensorSet(tgtT, tgt)
	maskT := ntTensorNew(len(mask))
	ntTensorSet(maskT, mask)
	tokIdx = ntTapeInput(tokT)
	tgtIdx = ntTapeInput(tgtT)
	maskIdx = ntTapeInput(maskT)
	ntTensorFree(tokT)
	ntTensorFree(tgtT)
	ntTensorFree(maskT)
	return
}

// ntTrainCore runs `steps` training steps of molequla's model on notorch.
// lrFor(step) supplies the per-step learning rate. Caller holds model.mu.
// Returns (avg loss, counted steps, step-loop wall ms) — the wall time is the
// pure training cost, criterion-2 metric (06_PLAN §11.2), measured over the
// step loop only, excluding the per-burst weight mirror in/out.
func ntTrainCore(model *GPT, tok *EvolvingTokenizer, docs []string, steps, seqLen int, lrFor func(int) float64) (float64, int, float64) {
	if len(docs) == 0 || steps <= 0 {
		return 0, 0, 0
	}
	vocab := tok.VocabSize
	// Inc2 (B3): op-33 assumes T_r == T and the combined Wr is packed at width
	// BlockSize, so RRPRAM-bearing bursts MUST run at T = BlockSize (this also
	// satisfies the documented gpu_rrpram_lr T-vs-T_max stride workaround). Pin it.
	if layerHasHybrid() {
		seqLen = model.BlockSize
	}
	m := ntNewMirror(model, seqLen)
	defer m.free()

	// Post-growth: wipe positional Chuck slots before the first step (S1).
	if ntTapeNeedsReset {
		ntTapeDestroy()
		ntTapeNeedsReset = false
	}

	guard := newNTNanGuard()
	tokBuf := make([]float32, seqLen)
	tgtBuf := make([]float32, seqLen)
	maskBuf := make([]float32, seqLen)
	var lossSum float64
	var lossN int

	t0 := time.Now()
	for step := 0; step < steps; step++ {
		if trainAborting() {
			// Shutting down: leave the loop at a step boundary. pullBack below
			// still mirrors everything trained so far into model.Base, and the
			// caller releases model.mu so the exit path can save it.
			break
		}
		ids := tok.Encode(docs[rand.Intn(len(docs))])
		if len(ids) < 2 {
			continue
		}
		start := 0
		if len(ids) > seqLen+1 {
			start = rand.Intn(len(ids) - seqLen - 1)
		}
		ntWindow(ids, start, seqLen, tokBuf, tgtBuf, maskBuf)

		ntTapeStart()
		m.register() // fixed order every step (B1): content, deltas, RRPRAM
		tokIdx, tgtIdx, maskIdx := ntPushInputs(tokBuf, tgtBuf, maskBuf)

		lossIdx, _ := ntBuildForward(m, tokIdx, tgtIdx, maskIdx, seqLen, vocab)
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

	// Mirror trained weights back into the canonical model.Base / Deltas store.
	m.pullBack()
	if lossN > 0 {
		return lossSum / float64(lossN), lossN, elapsedMs
	}
	return 0, 0, elapsedMs
}

// ntSequenceLoss builds the trainer's forward graph on the tape for one token
// sequence (no optimizer step) and returns the mean cross-entropy over the
// n = len(ids)-1 predicted positions — the same quantity Go inference computes
// in LossOnSequence — plus the logits of the last predicted position, for a
// component-wise parity check. Diagnostic: the tape is destroyed afterwards
// so Chuck slots of a live burst are never disturbed. Caller holds model.mu
// (or owns the model, as tests do).
func ntSequenceLoss(model *GPT, tok *EvolvingTokenizer, ids []int) (float64, []float32) {
	n := len(ids) - 1
	if n > model.BlockSize {
		n = model.BlockSize
	}
	if n <= 0 {
		return 0, nil
	}
	T := n
	if layerHasHybrid() {
		T = model.BlockSize // op-33 packing pins T to BlockSize; the mask prices n
	}
	m := ntNewMirror(model, T)
	defer m.free()
	tokBuf := make([]float32, T)
	tgtBuf := make([]float32, T)
	maskBuf := make([]float32, T)
	ntWindow(ids[:n+1], 0, T, tokBuf, tgtBuf, maskBuf)

	ntTapeStart()
	m.register()
	tokIdx, tgtIdx, maskIdx := ntPushInputs(tokBuf, tgtBuf, maskBuf)
	lossIdx, logitsIdx := ntBuildForward(m, tokIdx, tgtIdx, maskIdx, T, tok.VocabSize)
	loss := ntEntryScalar(lossIdx)
	V := tok.VocabSize
	all := ntEntryData(logitsIdx, T*V)
	var last []float32
	if len(all) >= n*V {
		last = append([]float32(nil), all[(n-1)*V:n*V]...)
	}
	ntTapeDestroy()
	return loss, last
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
		// Report the steps that ran, not the steps that were asked for: a warmup
		// cut short by a shutdown used to announce the full 1600 it never did.
		ran := model.globalStep - g0
		note := ""
		if ran < steps {
			note = fmt.Sprintf(" (stopped early, %d requested)", steps)
		}
		fmt.Printf("[notorch] warmup complete: %d steps%s, avg loss %.4f | %.0fms %.1f steps/s | gpu-dispatch=%d\n",
			ran, note, avg, ms, ntStepsPerSec(n, ms), ntGPUDispatchCount())
	}
}

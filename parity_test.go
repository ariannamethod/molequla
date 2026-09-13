package main

import (
	"math"
	"testing"
)

// "train ≡ infer": the function the notorch tape optimises must be the function
// Go inference runs. Repair 2b closes four gaps the audits found — the residual
// scale 1/sqrt(NLayer) inference applies (molequla.go:2975, :2992), the RMSNorm
// epsilon (1e-5 in Go, 1e-6 in notorch), padding a short document with token 0
// under an unmasked cross-entropy, and the delta adapters inference applies from
// their random init while the trainer never saw them. The gate: LossOnSequence
// (Go, float64) and the tape loss (notorch, float32) on one sequence agree.

func parityModel(t *testing.T, nLayer int) (*GPT, *EvolvingTokenizer, []int) {
	t.Helper()
	CFG.NEmbd = 16
	CFG.NLayer = nLayer
	CFG.NHead = 2
	CFG.BlockSize = 24
	CFG.HeadTypes = []string{"content", "content"}
	CFG.RRPRAMRank = 8
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.Trainer = "notorch"
	docs := []string{
		"the trainer and the mouth must agree on the function they compute",
		"a residual scaled in one place and not the other is two organisms",
		"an epsilon is small until the embryo's activations are smaller",
	}
	tok := NewEvolvingTokenizer(docs)
	model := NewGPT(tok)
	ids := tok.Encode(docs[0] + " " + docs[1])
	if len(ids) < CFG.BlockSize+2 {
		t.Fatalf("test sequence too short: %d tokens", len(ids))
	}
	return model, tok, ids
}

// goLastLogits runs Go inference over seq and returns the logits at the last
// predicted position (the same position the tape's last row prices).
func goLastLogits(model *GPT, seq []int) []float64 {
	n := len(seq) - 1
	keys := make([][]*Vec, model.NLayer)
	values := make([][]*Vec, model.NLayer)
	var logits *Vec
	for pos := 0; pos < n; pos++ {
		logits = model.ForwardStep(seq[pos], pos, keys, values)
	}
	return logits.Data
}

// The tape must match Go inference on a full window, layers > 1 so that
// residualAlpha = 1/sqrt(NLayer) is not 1. Forty training steps first, so the
// residual branches carry real magnitude: at init 0.08 they are too small for
// a scale of 1/sqrt(2) to move the loss, and a gate that cannot fail is
// decoration.
func TestTrainForwardMatchesInferenceLoss(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	model, tok, ids := parityModel(t, 2)
	docs := []string{"the trainer and the mouth must agree on the function they compute",
		"a residual scaled in one place and not the other is two organisms"}
	if avg, n, _ := ntTrainCore(model, tok, docs, 40, model.BlockSize, func(int) float64 { return 3e-3 }); n == 0 || math.IsNaN(avg) {
		t.Fatalf("warm-up training did not run: avg=%v n=%d", avg, n)
	}
	seq := ids[:CFG.BlockSize+1]

	goLoss := model.LossOnSequence(seq).Data
	tapeLoss, tapeLast := ntSequenceLoss(model, tok, seq)
	if math.IsNaN(goLoss) || math.IsNaN(tapeLoss) || goLoss <= 0 || tapeLoss <= 0 {
		t.Fatalf("losses not finite/positive: go=%v tape=%v", goLoss, tapeLoss)
	}
	// float32 tape vs float64 inference over 24 positions: 2e-3 absolute is
	// far above the accumulation noise and far below any of the four gaps.
	if d := math.Abs(goLoss - tapeLoss); d > 2e-3 {
		t.Fatalf("train forward diverges from inference: go=%.6f tape=%.6f |Δ|=%.2e — the tape is not the function the mouth runs", goLoss, tapeLoss, d)
	}
	// Component-wise on the last position: relative to the logit range.
	goLast := goLastLogits(model, seq)
	if len(goLast) != len(tapeLast) {
		t.Fatalf("logit widths differ: go=%d tape=%d", len(goLast), len(tapeLast))
	}
	var span, worst float64
	for i := range goLast {
		if a := math.Abs(goLast[i]); a > span {
			span = a
		}
		if d := math.Abs(goLast[i] - float64(tapeLast[i])); d > worst {
			worst = d
		}
	}
	if span == 0 || worst/span > 1e-3 {
		t.Fatalf("last-position logits diverge: worst |Δ|=%.3e over span %.3e (%.2e relative)", worst, span, worst/span)
	}
	t.Logf("parity: go=%.6f tape=%.6f |Δloss|=%.2e, last logits worst |Δ|=%.2e / span %.2e", goLoss, tapeLoss, math.Abs(goLoss-tapeLoss), worst, span)
}

// A document shorter than the window must be priced on its real positions
// only: no padding token trains, and the mean is over n, not BlockSize.
func TestTrainForwardMatchesInferenceOnShortSequence(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	model, tok, ids := parityModel(t, 2)
	seq := ids[:9] // n = 8 predicted positions, well under BlockSize 24

	goLoss := model.LossOnSequence(seq).Data
	tapeLoss, _ := ntSequenceLoss(model, tok, seq)
	if d := math.Abs(goLoss - tapeLoss); d > 2e-3 {
		t.Fatalf("short-sequence loss diverges: go=%.6f tape=%.6f |Δ|=%.2e — padding is being priced", goLoss, tapeLoss, d)
	}
	t.Logf("short sequence n=%d: go=%.6f tape=%.6f |Δ|=%.2e", len(seq)-1, goLoss, tapeLoss, math.Abs(goLoss-tapeLoss))
}

// The delta adapters are part of θ = ε + γ + αδ at inference from birth, so
// the default trainer must train them, not leave them at their random init.
func TestNotorchTrainerTrainsDeltas(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	model, tok, _ := parityModel(t, 1)
	if len(model.Deltas) == 0 {
		t.Fatal("a newborn organism has no delta module")
	}
	da := model.Deltas[0]["l0.wq"]
	if da == nil {
		t.Fatal("delta module has no l0.wq adapter")
	}
	before := make([][]float64, len(da.B.Rows))
	for i, r := range da.B.Rows {
		before[i] = append([]float64(nil), r.Data...)
	}
	docs := []string{"deltas are appended souls and a soul that never learns is furniture"}
	avg, n, _ := ntTrainCore(model, tok, docs, 30, model.BlockSize, func(int) float64 { return 1e-3 })
	if n == 0 || math.IsNaN(avg) {
		t.Fatalf("trainer did not run: avg=%v n=%d", avg, n)
	}
	var moved float64
	for i := range before {
		for j := range before[i] {
			if d := math.Abs(before[i][j] - da.B.Rows[i].Data[j]); d > moved {
				moved = d
			}
		}
	}
	if moved < 1e-5 {
		t.Fatalf("delta B barely moved (%.2e) — the trainer does not train the adapters inference applies", moved)
	}
	t.Logf("delta l0.wq B moved max |Δ|=%.2e over %d steps, avg loss %.4f", moved, n, avg)
}

package main

import (
	"math"
	"testing"
)

// TestNotorchTrainerTrainsWpe guards "the trained model is the model that runs":
// inference adds the learned positional embedding wpe to every token
// (ForwardStep), so the notorch trainer must register wpe on the tape and
// train it, exactly as the AML trainer does. Before the repair the trainer
// omitted wpe and every organism ran inference on a positional table that
// never left its random init.
func TestNotorchTrainerTrainsWpe(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.RRPRAMRank = 8
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.Trainer = "notorch"

	docs := []string{
		"the position of a token in the window is a learned table in this organism",
		"the trainer and the mouth must agree on what an embedding is made of",
	}
	tok := NewEvolvingTokenizer(docs)
	model := NewGPT(tok)

	wpe := model.Base["wpe"]
	wte := model.Base["wte"]
	if wpe == nil || wte == nil {
		t.Fatal("model has no wpe/wte")
	}
	wpeBefore := make([][]float64, len(wpe.Rows))
	for i, r := range wpe.Rows {
		wpeBefore[i] = append([]float64(nil), r.Data...)
	}
	wteBefore := make([][]float64, len(wte.Rows))
	for i, r := range wte.Rows {
		wteBefore[i] = append([]float64(nil), r.Data...)
	}

	avg, n, _ := ntTrainCore(model, tok, docs, 30, model.BlockSize, func(int) float64 { return 1e-3 })
	if n == 0 || math.IsNaN(avg) || math.IsInf(avg, 0) || avg <= 0 {
		t.Fatalf("trainer did not run cleanly: avg=%v n=%d", avg, n)
	}

	maxDelta := func(before [][]float64, mp *MatrixParam) float64 {
		var m float64
		for i := range before {
			for j := range before[i] {
				if d := math.Abs(before[i][j] - mp.Rows[i].Data[j]); d > m {
					m = d
				}
			}
		}
		return m
	}
	// 1e-5 separates a real gradient step from the float64↔float32 mirror
	// quantization (~1e-9 at these magnitudes), same bar as the RRPRAM test.
	if d := maxDelta(wteBefore, wte); d < 1e-5 {
		t.Fatalf("wte barely moved (%.2e) — the trainer is not training at all", d)
	}
	if d := maxDelta(wpeBefore, wpe); d < 1e-5 {
		t.Fatalf("wpe barely moved (%.2e) — the trainer omits the positional table that inference adds", d)
	}
}

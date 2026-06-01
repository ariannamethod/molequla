package main

import (
	"math"
	"testing"
)

// TestRRPRAMForward exercises the Increment-2 notorch trainer end to end on a
// small hybrid model: it must build the op-33 low-rank RRPRAM head + frozen-gate
// output-level blend, descend to a finite loss, and actually train the factors.
func TestRRPRAMForward(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.NEmbd = 32
	CFG.NLayer = 1
	CFG.NHead = 2
	CFG.BlockSize = 96
	CFG.HeadTypes = headTypesForNHead(2) // ["content","hybrid"]
	CFG.HybridAlphaInit = 0.5
	CFG.RRPRAMRank = 8
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.Trainer = "notorch"

	docs := []string{
		"the quick brown fox jumps over the lazy dog again and again",
		"resonance is the unbroken field of attention and memory across time",
		"molequla grows from embryo to adult through slow ontogenesis",
		"low rank attention factors wr a and wr b train on the notorch tape",
	}
	tok := NewEvolvingTokenizer(docs)
	model := NewGPT(tok)

	// Factors must exist with the exact op-33 packing dims.
	a := model.Base["l0.wr_a"]
	b := model.Base["l0.wr_b"]
	if a == nil || b == nil {
		t.Fatal("RRPRAM factors not allocated for a hybrid model")
	}
	if a.Nout != CFG.NHead*CFG.NEmbd || a.Nin != CFG.RRPRAMRank {
		t.Fatalf("wr_a dims = %dx%d, want %dx%d", a.Nout, a.Nin, CFG.NHead*CFG.NEmbd, CFG.RRPRAMRank)
	}
	if b.Nout != CFG.NHead*CFG.RRPRAMRank || b.Nin != CFG.BlockSize {
		t.Fatalf("wr_b dims = %dx%d, want %dx%d", b.Nout, b.Nin, CFG.NHead*CFG.RRPRAMRank, CFG.BlockSize)
	}

	// Snapshot wr_a per row. Rows [h·NEmbd : (h+1)·NEmbd) are head h's factor
	// block (op-33 packing). Head 0 is content (gate masks it → 0 grad), head 1
	// is hybrid (must train).
	before := make([][]float64, a.Nout)
	for i, r := range a.Rows {
		before[i] = append([]float64(nil), r.Data...)
	}

	avg, n, _ := ntTrainCore(model, tok, docs, 40, model.BlockSize, func(int) float64 { return 1e-3 })
	if n == 0 {
		t.Fatal("no training steps counted")
	}
	if math.IsNaN(avg) || math.IsInf(avg, 0) {
		t.Fatalf("loss is NaN/Inf: %v", avg)
	}
	if avg <= 0 {
		t.Fatalf("loss should be positive, got %v", avg)
	}

	// Distinguish a real gradient from the float64↔float32 mirror quantization
	// (~1e-9 for these magnitudes) by max-abs change per head block. The hybrid
	// head must move much more than the gate-masked content head.
	var hybridMax, contentMax float64
	for i := 0; i < a.Nout; i++ {
		head := i / CFG.NEmbd
		for j := range before[i] {
			d := math.Abs(before[i][j] - a.Rows[i].Data[j])
			if head == 0 {
				if d > contentMax {
					contentMax = d
				}
			} else if d > hybridMax {
				hybridMax = d
			}
		}
	}
	if hybridMax < 1e-5 {
		t.Fatalf("hybrid head wr_a barely moved (%.2e) — op-33 backward did not reach the factors", hybridMax)
	}
	if contentMax > 1e-6 {
		t.Fatalf("content head wr_a moved %.2e (> quantization) — the gate mask should give it zero gradient", contentMax)
	}
	t.Logf("RRPRAM trainer OK: avg loss %.4f over %d steps; hybrid Δ=%.2e (trained), content Δ=%.2e (masked)", avg, n, hybridMax, contentMax)
}

// TestRRPRAMContentParityNoHybrid guards B1: with no hybrid head the param list
// and forward must be byte-identical to Inc1 (content-only) — no RRPRAM splice.
func TestRRPRAMContentParityNoHybrid(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 96
	CFG.HeadTypes = []string{"content"}
	CFG.RRPRAMRank = 8
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.Trainer = "notorch"

	if layerHasHybrid() {
		t.Fatal("content-only topology must not report hybrid")
	}
	docs := []string{"a small content only organism with no rrpram heads at all"}
	tok := NewEvolvingTokenizer(docs)
	model := NewGPT(tok)
	if _, ok := model.Base["l0.wr_a"]; ok {
		t.Fatal("content-only model must not allocate RRPRAM factors")
	}
	avg, n, _ := ntTrainCore(model, tok, docs, 20, model.BlockSize, func(int) float64 { return 1e-3 })
	if n == 0 || math.IsNaN(avg) || math.IsInf(avg, 0) || avg <= 0 {
		t.Fatalf("content-only trainer regressed: avg=%v n=%d", avg, n)
	}
}

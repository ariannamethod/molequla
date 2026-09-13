package main

import "testing"

// The DNA emission line reports two numbers taken at the first generated step:
// the transformer magnitude of the raw logits and the overlay weight that
// magnitude buys. An embryo speaks with the overlay at full authority; an adult
// speaks with it gone, and the line must be able to tell the two apart. The
// gate goes red if the fields stay at their zero value or if the fade stops
// tracking the magnitude.
func TestDNAEmissionFieldsFollowTheOverlayFade(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	model, tok, field, docs := grazeTestModel(t)
	CFG.CorpusLogitOverlay = true
	CFG.MaxGenTokens = 4

	GenerateResonant(model, tok, field, "the river", docs, true)
	if model.lastGenMag <= 0 {
		t.Fatalf("lastGenMag = %v after a generation; the first step never measured the raw logits", model.lastGenMag)
	}
	if model.lastOverlayWeight < 0 || model.lastOverlayWeight > 1 {
		t.Fatalf("lastOverlayWeight = %v, want a weight in [0,1]", model.lastOverlayWeight)
	}
	embryoMag, embryoWeight := model.lastGenMag, model.lastOverlayWeight
	if embryoWeight <= 0 {
		t.Fatalf("lastOverlayWeight = %v on an embryo at mean |logit| %.3f; the overlay is what speaks here",
			embryoWeight, embryoMag)
	}
	t.Logf("embryo: mag %.3f weight %.3f (fade %.2f)", embryoMag, embryoWeight, 1-embryoWeight)

	// Warm the organism the way TestCrossGrazeReachesSamplingWhenOverlayOnAndWarmed
	// does: scale lm_head so mean |logit| lands far past the fade band.
	for _, row := range model.Base["lm_head"].Rows {
		for j := range row.Data {
			row.Data[j] *= 400
		}
	}
	GenerateResonant(model, tok, field, "the river", docs, true)
	if model.lastGenMag <= 2 {
		t.Fatalf("lastGenMag = %v on a warmed organism (embryo %v), want past the fade band", model.lastGenMag, embryoMag)
	}
	if model.lastOverlayWeight != 0 {
		t.Fatalf("lastOverlayWeight = %v on a warmed organism (embryo %v), want 0: the overlay is gone and fade prints 1.00",
			model.lastOverlayWeight, embryoWeight)
	}
}

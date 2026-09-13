package main

import (
	"math"
	"testing"
)

// Sampling gates against the real SoftmaxProbs and TopKTopPSample. They
// lived in tests/molequla_test.go as package `tests`, exercising a copy of
// both functions pasted into the test file — a change to the real ones could
// not turn that suite red (audit C, C-TST-01; feedback_circular_tests_lying).
// Moved here on 2026-09-13, bodies unchanged.

func TestTopKTopPSample(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.05, 0.65}
	for i := 0; i < 10; i++ {
		if result := TopKTopPSample(probs, 1, 1.0, 0.0, 1.0); result != 3 {
			t.Errorf("With k=1, expected index 3, got %d", result)
		}
	}
}

func TestTopKTopPSampleTopKLimits(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.3, 0.4}
	for i := 0; i < 100; i++ {
		if result := TopKTopPSample(probs, 2, 1.0, 0.0, 1.0); result != 2 && result != 3 {
			t.Errorf("With k=2, expected index 2 or 3, got %d", result)
		}
	}
}

func TestMinPFiltersLowProbs(t *testing.T) {
	probs := []float64{0.01, 0.02, 0.07, 0.9} // max = 0.9; min_p=0.1 → threshold 0.09
	for i := 0; i < 20; i++ {
		if result := TopKTopPSample(probs, 0, 1.0, 0.1, 1.0); result != 3 {
			t.Errorf("With min_p=0.1, expected index 3, got %d", result)
		}
	}
}

func TestMinPKeepsProportional(t *testing.T) {
	probs := []float64{0.05, 0.15, 0.30, 0.50} // max = 0.5; min_p=0.2 → threshold 0.1
	for i := 0; i < 100; i++ {
		if result := TopKTopPSample(probs, 0, 1.0, 0.2, 1.0); result == 0 {
			t.Errorf("With min_p=0.2, index 0 should never be sampled, but got it")
		}
	}
}

func TestTypicalPPrefersTypical(t *testing.T) {
	probs := []float64{0.25, 0.25, 0.25, 0.25}
	seen := make(map[int]bool)
	for i := 0; i < 100; i++ {
		seen[TopKTopPSample(probs, 0, 1.0, 0.0, 0.9)] = true
	}
	if len(seen) < 3 {
		t.Errorf("With uniform probs and typical_p=0.9, expected at least 3 different indices, got %d", len(seen))
	}
}

func TestTypicalPWithVariedProbs(t *testing.T) {
	probs := []float64{0.01, 0.09, 0.30, 0.60}
	for i := 0; i < 50; i++ {
		if result := TopKTopPSample(probs, 0, 1.0, 0.0, 0.8); result < 0 || result >= len(probs) {
			t.Errorf("Sample out of range: %d", result)
		}
	}
}

func TestCombinedMinPTypicalP(t *testing.T) {
	probs := []float64{0.02, 0.08, 0.20, 0.70} // min_p=0.1 with max 0.7 → threshold 0.07
	for i := 0; i < 50; i++ {
		if result := TopKTopPSample(probs, 0, 1.0, 0.1, 0.9); result == 0 {
			t.Errorf("With min_p=0.1, index 0 should never be sampled")
		}
	}
}

func TestSoftmaxProbs(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0}
	probs := SoftmaxProbs(logits)
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax should sum to 1, got %f", sum)
	}
	for i, p := range probs {
		if p < 0 {
			t.Errorf("Softmax[%d] should be positive, got %f", i, p)
		}
	}
	for i := 0; i < len(probs)-1; i++ {
		if probs[i] >= probs[i+1] {
			t.Errorf("Higher logit should give higher prob: probs[%d]=%f >= probs[%d]=%f", i, probs[i], i+1, probs[i+1])
		}
	}
}

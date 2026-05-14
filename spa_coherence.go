package main

import "math"

// ═══════════════════════════════════════════════════════════════════════════════
// SPA — Sentence Phonon Attention — coherence-without-training pass.
//
// Pure-Go port of canonical AML's spa_embed + spa_connectedness ops
// (~/arianna/ariannamethod.ai/core/ariannamethod.c, commit ef52cde).
// SPA is forward-only by design — Q's third pillar from
// ~/arianna/q/README.md:177-179. Used to score per-sentence
// connectedness so that weak sentences in a multi-sentence generation
// can be flagged or reseeded with neighbor context.
//
// The math is duplicated here (instead of routed through AML/CGO) to
// keep the post-generation pass cheap: one Go call, no script string
// building, no CGO crossing per sentence. The vendored AML still
// carries the same ops for AML-script consumers — see B1 step 1
// in molequla/PROJECT_LOG.md.
//
// Math (verbatim from canonical AML):
//
//   spa_embed:
//     e[d]      = sum_i (alpha^(n-1-i) * W[ids[i]][d]) / sum_i alpha^(n-1-i)
//     e         = e / ||e||
//
//   spa_connectedness:
//     scores[i] = sum_{j != i} exp(E_i · E_j / sqrt(D) + bias[|i-j|])
//
// Q's reseed gate: sentence i is "weak" iff scores[i] < 0.6 * mean(scores).
// ═══════════════════════════════════════════════════════════════════════════════

const SPAWeakThresholdRatio = 0.6 // Q's default; weak ↔ score < 0.6 × mean.

// SPACoherenceScores returns per-sentence connectedness scores for the
// given S sentences. W is the flat row-major embedding matrix [V*D]:
// pass gpt.Base["wte"] flat to use learned embeddings, or a separately
// initialised random matrix for Q-style weightless mode (recommended at
// embryo/infant stages when wte is not yet meaningful).
//
// sentenceTokens: token-id sequence per sentence; bad ids (negative or
// out of range) are skipped within a sentence (does not propagate to
// score validity for that sentence overall).
//
// D: embedding dim, must match W column count (len(W) must be divisible
// by D).
//
// alpha: exponential decay for spa_embed; Q default 0.85.
//
// Returns nil on invalid args. Returns a length-S slice otherwise; even
// if a sentence ends up with zero accumulated weight it returns score 0
// for that index (the cross-attention term degrades cleanly to
// exp(0 / sqrt(D)) contributions from siblings).
func SPACoherenceScores(W []float32, sentenceTokens [][]int, D int, alpha float32) []float32 {
	S := len(sentenceTokens)
	if S == 0 || D <= 0 || len(W) == 0 || len(W)%D != 0 {
		return nil
	}
	V := len(W) / D

	// --- spa_embed: build E[S*D] flat row-major. ---
	E := make([]float32, S*D)
	for i, toks := range sentenceTokens {
		n := len(toks)
		base := i * D
		var totalW float32 = 0
		for k, t := range toks {
			if t < 0 || t >= V {
				continue
			}
			w := float32(math.Pow(float64(alpha), float64(n-1-k)))
			row := W[t*D : (t+1)*D]
			for d := 0; d < D; d++ {
				E[base+d] += w * row[d]
			}
			totalW += w
		}
		if totalW > 0 {
			for d := 0; d < D; d++ {
				E[base+d] /= totalW
			}
		}
		// L2 normalise (matches canonical AML — `+ 1e-8f` smoother).
		var norm float32 = 0
		for d := 0; d < D; d++ {
			norm += E[base+d] * E[base+d]
		}
		norm = 1.0 / float32(math.Sqrt(float64(norm)+1e-8))
		for d := 0; d < D; d++ {
			E[base+d] *= norm
		}
	}

	// --- spa_connectedness: bidirectional cross-attention score. ---
	scores := make([]float32, S)
	invSD := 1.0 / float32(math.Sqrt(float64(D)))
	for i := 0; i < S; i++ {
		var total float32 = 0
		for j := 0; j < S; j++ {
			if i == j {
				continue
			}
			var dot float32 = 0
			for d := 0; d < D; d++ {
				dot += E[i*D+d] * E[j*D+d]
			}
			dot *= invSD
			total += float32(math.Exp(float64(dot)))
		}
		scores[i] = total
	}
	return scores
}

// SPAWeakSentences returns the indices of sentences whose connectedness
// score is below SPAWeakThresholdRatio (0.6) times the mean — Q's reseed
// criterion. Empty result means everything passed the gate.
func SPAWeakSentences(scores []float32) []int {
	if len(scores) < 2 {
		return nil
	}
	var sum float32 = 0
	for _, s := range scores {
		sum += s
	}
	mean := sum / float32(len(scores))
	threshold := SPAWeakThresholdRatio * mean
	var weak []int
	for i, s := range scores {
		if s < threshold {
			weak = append(weak, i)
		}
	}
	return weak
}

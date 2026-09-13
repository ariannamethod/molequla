package main

import "math"

// ═══════════════════════════════════════════════════════════════════════════════
// Q-style metaweights overlay — raw-probability, dynamic-gate, coherence-without-training
//
// Ported from ~/arianna/q/postgpt_q.c:1305-1395, the reference implementation
// of «coherence emerges from corpus statistics» pattern. The mechanism mixes
// the model's untrained logits with five statistical signals (Hebbian / prophecy
// / destiny / bigram / trigram) added as RAW probability values, not log-probs.
//
// Coefficients are dynamic — magnitude-detector picks weightless or trained
// bundle based on average |logit|. While the transformer's weights are weak,
// the overlay dominates and produces coherent output. As weights strengthen,
// overlay fades, ceding to model voice. Q's auto-curriculum.
//
// Scratch reuse: a single OverlayScratch is constructed once per
// GenerateResonant call and threaded into every per-step overlay invocation.
// Per-step allocations dropped from six [V]float64 slices to zero — material
// over a multi-hour ecology run on a CPU pod (codex P1, 2026-05-14).
// ═══════════════════════════════════════════════════════════════════════════════

// MetaCoeffs holds the five Dario-field overlay coefficients for one regime.
type MetaCoeffs struct {
	Heb, Pro, Ds, Bg, Tg float64
}

// Reference values verbatim from postgpt_q.c:1358-1359.
var (
	metaCoeffsWeightless = MetaCoeffs{Heb: 1.0, Pro: 0.7, Ds: 0.15, Bg: 15.0, Tg: 10.0}
	metaCoeffsTrained    = MetaCoeffs{Heb: 0.6, Pro: 0.4, Ds: 0.3, Bg: 5.0, Tg: 3.0}
)

// Transformer-magnitude gate threshold for switching weightless → trained
// coefficient bundles. Postgpt_q.c:1356 uses 0.1 (raw untrained wte). Molequla
// seeds wte from metaweights (postgpt.c:541-574) which lifts mean|logit| to
// ~0.25 even at zero training, so 0.1 is too low — keep weightless coeffs in
// force until real training raises mag past 1.0 (smooth gate at tg=0.5).
const metaTFGateThreshold = 1.0

// metaFadeWidth — how far above metaTFGateThreshold, in mean |logit|, the
// overlay takes to fade out. At the threshold the weightless bundle applies
// in full; at threshold + width the overlay is gone and the transformer
// speaks alone. Before repair 6 (MOLEQULALOG2.md, 2026-09-13) the whole
// additive stack vanished in one step at the threshold.
const metaFadeWidth = 1.0

// overlayFadeProgress is 0 while the transformer is untrained (tmag ≤
// threshold), 1 once the overlay has faded out (tmag ≥ threshold +
// metaFadeWidth), linear in between. The caller's overlay weight is 1 - this,
// and the coefficient bundle slides along the same ramp, so nothing steps.
func overlayFadeProgress(tmag float64) float64 {
	s := (tmag - metaTFGateThreshold) / metaFadeWidth
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}

// lerpCoeffs interpolates two coefficient bundles: s = 0 → a, s = 1 → b.
func lerpCoeffs(a, b MetaCoeffs, s float64) MetaCoeffs {
	return MetaCoeffs{
		Heb: a.Heb + s*(b.Heb-a.Heb),
		Pro: a.Pro + s*(b.Pro-a.Pro),
		Ds:  a.Ds + s*(b.Ds-a.Ds),
		Bg:  a.Bg + s*(b.Bg-a.Bg),
		Tg:  a.Tg + s*(b.Tg-a.Tg),
	}
}

// OverlayScratch holds reusable [V]float64 buffers + cached static terms
// (destiny, unigram) for one GenerateResonant call. Per-step bigram/trigram/
// hebbian buffers are reset only at indices that were touched the previous
// step; destiny + unigram are computed once at construction and not touched
// per step because they do not change while weights are locked and the field
// is not mutated during generation.
//
// Construct with NewOverlayScratch(V) at the start of generation and call
// PrepareStatic(model, field) once before the per-token loop. Pass into
// MetaweightsOverlay each step.
type OverlayScratch struct {
	V int

	// Per-step buffers — reset by clearTouched at the start of each call.
	Bigram   []float64
	Trigram  []float64
	Hebbian  []float64
	Prophecy []float64

	// Static-per-generation buffers — filled once by PrepareStatic.
	Unigram   []float64
	Destiny   []float64
	staticSet bool

	// Index trackers for sparse reset. Per-step we only zero positions we
	// wrote into; full make() is gone.
	touchedBigram   []int
	touchedTrigram  []int
	touchedHebbian  []int
	touchedProphecy []int
}

// NewOverlayScratch allocates a scratch struct for vocab size V.
func NewOverlayScratch(V int) *OverlayScratch {
	if V <= 0 {
		return &OverlayScratch{V: 0}
	}
	return &OverlayScratch{
		V:        V,
		Bigram:   make([]float64, V),
		Trigram:  make([]float64, V),
		Hebbian:  make([]float64, V),
		Prophecy: make([]float64, V),
		Unigram:  make([]float64, V),
		Destiny:  make([]float64, V),
	}
}

// PrepareStatic fills Unigram and Destiny once for this generation call.
// Caller holds model.mu so wte is stable; field is read-locked locally only
// for the unigram snapshot.
func (s *OverlayScratch) PrepareStatic(model *GPT, field *CooccurField) {
	if s == nil || s.V == 0 {
		return
	}
	V := s.V

	// Unigram — snapshot under short read lock, then unlock for the heavy
	// destiny work. The lock window is now O(|field.Unigram|) instead of
	// covering the per-token call's full body.
	for i := range s.Unigram {
		s.Unigram[i] = 0
	}
	if field != nil {
		field.mu.RLock()
		var uniTotal float64
		for _, v := range field.Unigram {
			uniTotal += v
		}
		if uniTotal > 0 {
			for tid, v := range field.Unigram {
				if tid < V {
					s.Unigram[tid] = v / uniTotal
				}
			}
		}
		field.mu.RUnlock()
	}

	// Destiny — cosine(wte[i], gammaDir). Weights are stable for the
	// generation call, so we compute once instead of every token.
	for i := range s.Destiny {
		s.Destiny[i] = 0
	}
	if model != nil {
		if gammaDir, mag := model.GammaContrastiveProjection(); mag > 0 && len(gammaDir) > 0 {
			if wte := model.Base["wte"]; wte != nil {
				D := wte.Nin
				if D > 0 && D <= len(gammaDir) {
					for v := 0; v < V && v < wte.Nout; v++ {
						row := wte.Rows[v].Data
						var dot, en float64
						for d := 0; d < D; d++ {
							dot += gammaDir[d] * row[d]
							en += row[d] * row[d]
						}
						en = math.Sqrt(en + 1e-10)
						if en > 1e-8 {
							s.Destiny[v] = dot / en
						}
					}
				}
			}
		}
	}
	s.staticSet = true
}

// clearTouched zeroes only the indices the previous step wrote into and
// resets touched lists. Sparse reset — vocab-wide loops gone.
func (s *OverlayScratch) clearTouched() {
	for _, i := range s.touchedBigram {
		s.Bigram[i] = 0
	}
	s.touchedBigram = s.touchedBigram[:0]
	for _, i := range s.touchedTrigram {
		s.Trigram[i] = 0
	}
	s.touchedTrigram = s.touchedTrigram[:0]
	for _, i := range s.touchedHebbian {
		s.Hebbian[i] = 0
	}
	s.touchedHebbian = s.touchedHebbian[:0]
	for _, i := range s.touchedProphecy {
		s.Prophecy[i] = 0
	}
	s.touchedProphecy = s.touchedProphecy[:0]
}

// MetaweightsOverlay applies the Q-style dynamic logit overlay in place on
// `logits` (model's raw pre-temperature logits over the vocabulary).
//
//   - `ids` — the running token context for this generation step.
//   - `field` — the organism's CooccurField (must not be nil).
//   - `model` — used for destiny term (cached in scratch).
//   - `prophecyField` — persistent expectation field; nil means «not yet seeded».
//   - `scratch` — reusable buffers (created once per GenerateResonant call).
//   - Returns the modified logits slice + the (possibly seeded) prophecy field.
//
// Mechanism mirrors postgpt_q.c:1354-1395.
func MetaweightsOverlay(
	logits []float64,
	ids []int,
	field *CooccurField,
	model *GPT,
	prophecyField []float64,
	scratch *OverlayScratch,
) ([]float64, []float64) {
	V := len(logits)
	if V == 0 || field == nil || len(ids) < 1 {
		return logits, prophecyField
	}

	// Use a fallback scratch if the caller forgot to pass one. Allocation
	// only on the cold path — production callers always pass scratch.
	if scratch == nil || scratch.V != V {
		scratch = NewOverlayScratch(V)
		scratch.PrepareStatic(model, field)
	} else {
		scratch.clearTouched()
	}

	// 1. Magnitude gate — pick coefficient bundle.
	var tmag float64
	for _, v := range logits {
		if v < 0 {
			tmag -= v
		} else {
			tmag += v
		}
	}
	tmag /= float64(V)
	// The bundle slides from the weightless reference values to the trained
	// ones across the fade band above the threshold (repair 6); below it the
	// weightless bundle applies unchanged.
	coeffs := lerpCoeffs(metaCoeffsWeightless, metaCoeffsTrained, overlayFadeProgress(tmag))

	// 1.5. Transformer gate — silence untrained transformer logits before
	// overlay (pitomadom.c:583-586 inspiration). Only kicks in below the
	// coefficient-switch threshold — once warmup is real (`tmag` > 1.0)
	// the transformer leads and overlay is just a bias layer. The original
	// formulation `tg = clamp((tmag-0.5)/1.5, 0, 1)` over-silenced warmed
	// 16-dim embryo models (mag~1.0 → tg≈0.33) and let overlay's bigram
	// chain dominate on the BPE subword level, producing subword salad
	// instead of words (sweep cell 2 v2 regression, fix 2026-05-14 PM).
	if tmag <= metaTFGateThreshold {
		tg := tmag / metaTFGateThreshold
		if tg < 0 {
			tg = 0
		} else if tg > 1 {
			tg = 1
		}
		for i := range logits {
			logits[i] *= tg
		}
	}

	// 2. Bigram + trigram per-context — normalised probabilities written
	// sparsely into scratch.Bigram / scratch.Trigram. Only positions with
	// non-zero context get touched (then zeroed again at the next step).
	field.mu.RLock()
	prev := ids[len(ids)-1]
	if ctx, ok := field.BigramByFirst[prev]; ok {
		var total float64
		for _, v := range ctx {
			total += v
		}
		if total > 0 {
			for tid, v := range ctx {
				if tid < V {
					scratch.Bigram[tid] = v / total
					scratch.touchedBigram = append(scratch.touchedBigram, tid)
				}
			}
		}
	}
	if len(ids) >= 2 {
		a, b := ids[len(ids)-2], ids[len(ids)-1]
		if ctx, ok := field.TrigramByContext[[2]int{a, b}]; ok {
			var total float64
			for _, v := range ctx {
				total += v
			}
			if total > 0 {
				for tid, v := range ctx {
					if tid < V {
						scratch.Trigram[tid] = v / total
						scratch.touchedTrigram = append(scratch.touchedTrigram, tid)
					}
				}
			}
		}
	}

	// 3. Hebbian — window-walked co-occurrence, max-normalised.
	windowSize := CFG.CooccurWindowSize
	if windowSize <= 0 || windowSize > len(ids) {
		windowSize = len(ids)
	}
	var hebMax float64
	for j := len(ids) - windowSize; j < len(ids); j++ {
		if neighbors, ok := field.CooccurWindow[ids[j]]; ok {
			for tid, cnt := range neighbors {
				if tid < V {
					if scratch.Hebbian[tid] == 0 {
						scratch.touchedHebbian = append(scratch.touchedHebbian, tid)
					}
					scratch.Hebbian[tid] += cnt
					if scratch.Hebbian[tid] > hebMax {
						hebMax = scratch.Hebbian[tid]
					}
				}
			}
		}
	}
	field.mu.RUnlock()
	if hebMax > 0 {
		for _, i := range scratch.touchedHebbian {
			scratch.Hebbian[i] /= hebMax
		}
	}

	// 4. Prophecy — seed once, then age per step. Normalised values land
	// in scratch.Prophecy (zeroed via touchedProphecy each step).
	if prophecyField == nil {
		prophecyField = make([]float64, V)
		// Seed from trigram (primary) + half-weight bigram fallback.
		for _, i := range scratch.touchedTrigram {
			prophecyField[i] += scratch.Trigram[i]
		}
		for _, i := range scratch.touchedBigram {
			prophecyField[i] += 0.5 * scratch.Bigram[i]
		}
		var pt float64
		for _, v := range prophecyField {
			pt += v
		}
		if pt > 0 {
			for i := range prophecyField {
				prophecyField[i] /= pt
			}
		} else {
			prophecyField = nil
		}
	} else {
		decay := CFG.MetaProphecyDecay
		if decay <= 0 || decay > 1 {
			decay = 0.95
		}
		for i := range prophecyField {
			prophecyField[i] *= decay
		}
	}
	if prophecyField != nil {
		var pt float64
		for _, v := range prophecyField {
			pt += v
		}
		if pt > 0 {
			inv := 1.0 / pt
			for i := 0; i < V && i < len(prophecyField); i++ {
				p := prophecyField[i] * inv
				if p != 0 {
					scratch.Prophecy[i] = p
					scratch.touchedProphecy = append(scratch.touchedProphecy, i)
				}
			}
		}
	}

	// 5. Static terms are already in scratch.Destiny + scratch.Unigram
	// (filled once by PrepareStatic). Fall-through computation if caller
	// gave a fresh scratch without PrepareStatic — handled at top of fn.
	if !scratch.staticSet {
		scratch.PrepareStatic(model, field)
	}

	// 6. The actual overlay — Q's line 1383, raw values:
	//    raw[i] += c_heb*heb[i] + c_pro*pro[i] + c_ds*ds + c_bg*bg + c_tg*tg
	for i := 0; i < V; i++ {
		logits[i] += coeffs.Heb*scratch.Hebbian[i] +
			coeffs.Pro*scratch.Prophecy[i] +
			coeffs.Ds*scratch.Destiny[i] +
			coeffs.Bg*scratch.Bigram[i] +
			coeffs.Tg*scratch.Trigram[i]

		// Unigram damping (postgpt_q.c:1393-1394) — suppress noise tokens.
		u := scratch.Unigram[i]
		if u < 1e-6 {
			logits[i] -= 2.0
		} else if u > 0.01 {
			logits[i] -= 0.3 * (u - 0.01) * 100.0
		}
	}

	return logits, prophecyField
}

// MetaweightsOverlayCollapse zeroes the prophecy slot for a token that was
// just sampled — Q's «collapse on fulfilment» pattern. nil-safe.
func MetaweightsOverlayCollapse(prophecyField []float64, sampledID int) {
	if prophecyField != nil && sampledID >= 0 && sampledID < len(prophecyField) {
		prophecyField[sampledID] = 0
	}
}

// MetaweightsRepetitionPenalty applies postgpt's uniform repetition penalty
// + bigram blocking on raw logits AFTER the metaweights overlay. Mirror of
// postgpt.c:960-967 and postgpt_q.c:1407-1408.
//
//   - Distinct tokens in the last 12 ids: logits[t] *= 0.5 (uniform).
//   - Bigram blocking (cl >= 2): for every position where ctx[i]==ctx[cl-2],
//     penalise ctx[i+1] by 0.2. Kills two-token cycles.
//
// Dedupe uses a fixed [12]int linear scan — zero heap alloc per step
// (codex P3, 2026-05-14).
func MetaweightsRepetitionPenalty(logits []float64, ids []int) {
	V := len(logits)
	cl := len(ids)
	if V == 0 || cl == 0 {
		return
	}
	start := cl - 12
	if start < 0 {
		start = 0
	}
	var seen [12]int
	seenN := 0
	for ri := cl - 1; ri >= start; ri-- {
		t := ids[ri]
		if t < 0 || t >= V {
			continue
		}
		dup := false
		for i := 0; i < seenN; i++ {
			if seen[i] == t {
				dup = true
				break
			}
		}
		if dup {
			continue
		}
		// Halve a positive logit; leave a negative one where it is. The
		// reference multiplies unconditionally, and after the overlay's
		// unigram damping (-2.0 on rare tokens) a repeated token with a
		// negative logit was moved toward zero — up the ranking — by ×0.5:
		// the penalty rewarded repetition where the overlay had declared the
		// token unlikely (audit C, C-OVL-04). Repair 6.
		if logits[t] > 0 {
			logits[t] *= 0.5
		}
		if seenN < 12 {
			seen[seenN] = t
			seenN++
		}
	}
	if cl >= 2 {
		last := ids[cl-2]
		for ri := 0; ri < cl-1; ri++ {
			if ids[ri] == last && ids[ri+1] >= 0 && ids[ri+1] < V && logits[ids[ri+1]] > 0 {
				logits[ids[ri+1]] *= 0.2 // same sign rule as above (C-OVL-04)
			}
		}
	}
}

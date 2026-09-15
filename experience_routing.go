package main

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"sort"
	"strings"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE CAFETERIA — experience is distributed, not broadcast (new logic §12, §14)
//
// Until this file, every organism ate every fragment: dnaSources(element) named
// the same directories for all four, launch.sh passed the same extra sources to
// all four, and dnaRead appended the same bytes to each corpus. The only thing
// that differed was arrival time. §12 asks for the opposite default — one
// organism receives X, another X + Y, overlap fine, byte-identical broadcast
// never the default — and it asks that the organism's *current state* decide,
// because the elemental corpus is a birth condition and not a profession.
//
// Nothing here coordinates between processes. The four organisms are separate
// processes that share only the fragment's file name and its bytes, so both
// halves of the decision are computed from those two things plus the reader's
// own corpus field:
//
//   owner      — hash(src/name) mod (the elements that may read this source)
//                picks exactly one guaranteed eater per fragment. Content-blind
//                on purpose: it is a slot lottery over file names, not a rule of
//                the form "flowers go to Earth". It is what makes "every
//                fragment reaches at least one organism" true without anybody
//                asking anybody.
//   resonance  — the fragment is already in the reader's language: the share of
//                its token bigrams that its own CooccurField has seen is at or
//                above the CFG.ExperienceResonanceQuantile quantile of the
//                coverages this organism has recently measured.
//   novelty    — the fragment is outside the reader's language: that same share
//                is at or below the CFG.ExperienceNoveltyQuantile quantile of
//                the same ring. Without this branch an organism is sealed inside
//                what it already knows, and the senses — the only food that is
//                new by construction — would never be eaten by anybody but
//                their hash owner.
//
// The band in between is refused. Coverage is state: it rises as the organism
// eats, so the same fragment file routes differently in week two than in week
// one, which is the drift §12 is after.
//
// The bar is state too, and that is what this file changed on 2026-09-15.
// It carried two absolute numbers — 0.965 and 0.620, the medians of two
// populations measured that morning on stage-3 corpora. Coverage grows with the
// corpus, so an absolute bar cannot follow the organism: the same air
// checkpoint declined 8 of 9 fragments against the 1498-line seed corpus and 0
// of 12 against its grown 9586-line reservoir, where 141 of 145 live fragments
// sit above 0.965 and the cafeteria had become a broadcast again
// (MOLEQULALOG2.md, 2026-09-15). Each organism now carries a ring of its own
// last CFG.ExperienceCoverageWindow measured coverages, persisted beside its
// dna_cursor.json, and the two bars are quantiles of that ring. The
// distribution moves with the corpus; the shares do not.
//
// Until the ring holds CFG.ExperienceCoverageWarm samples the organism has no
// distribution to quantile, so only the owner rule runs and everything else is
// declined as "warming" — which the [cafeteria] line names, so a session can
// tell a warming organism from one that is judging. A newborn without a field
// at all is a different case and is upstream of this one: it fails the
// ExperienceMinPairs test and eats everything as "unmeasured".
// ═══════════════════════════════════════════════════════════════════════════════

// experienceMeal is one fragment this organism accepted, kept so that what it
// says next is shaped by what it just ate (§14). `extra` marks food that came
// from a sense rather than from a sibling: raw outside experience, which may
// seed a probe but must never be padded verbatim into collective DNA.
type experienceMeal struct {
	src string
	// The corpus lines this fragment was cut into by splitCorpusLine and
	// appended as (routing repair 3), not the fragment they came from: these
	// are what loadCorpusLines hands back, so these are what the padding has to
	// be able to recognise and what a probe is taken from.
	lines []string
	extra bool
}

// experienceState is this process's view of itself: the field and tokenizer the
// training loop is currently using (published by dnaWrite, which already
// receives both), the last CFG.ExperienceMealMemory fragments eaten, and the
// last CFG.ExperienceCoverageWindow coverages it measured — the distribution
// its own two bars are quantiles of.
type experienceState struct {
	mu    sync.Mutex
	field *CooccurField
	tok   *EvolvingTokenizer
	meals []experienceMeal

	cov      []float64 // oldest first; at most CFG.ExperienceCoverageWindow
	covPath  string    // where it is persisted; empty = in memory only (tests)
	covDirty bool      // a measurement has arrived since the last write
}

// experienceRouting is the per-process singleton. One organism, one process,
// one state.
var experienceRouting = &experienceState{}

// observe publishes the live field and tokenizer. dnaWrite is called one line
// before dnaRead on every tick and already holds both, so the allocator reads
// the same field generation the organism is speaking with.
func (s *experienceState) observe(field *CooccurField, tok *EvolvingTokenizer) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.field, s.tok = field, tok
	s.mu.Unlock()
}

// experienceSample cuts `text` down to about `budget` bytes as evenly spaced
// windows across the whole fragment, never as a head. The head of a DNA
// fragment is the writer's generated answer, whose bigrams every organism
// already has: sampling the head made 240-byte coverage saturate at 1.000 for
// sibling fragments and destroyed the signal (measured 2026-09-15). Encoding is
// O(bytes x merges) and cost 143 ms on a whole 5 KB fragment against 22 ms at
// budget 480, with the quartiles of the sibling distribution preserved to
// within 0.006.
func experienceSample(text string, budget int) string {
	const win = 120
	if budget <= 0 || len(text) <= budget || len(text) <= win {
		return text
	}
	k := budget / win
	if k < 1 {
		k = 1
	}
	var b strings.Builder
	b.Grow(k * (win + 1))
	span := len(text) - win
	for i := 0; i < k; i++ {
		off := span * i / k
		b.WriteString(text[off : off+win])
		b.WriteByte(' ')
	}
	return b.String()
}

// experienceCoverage is the share of `text`'s adjacent token pairs that the
// organism's own bigram table has already seen, and the number of pairs the
// share was taken over. This is the CooccurField the organism already builds
// from its corpus every thirty ticks (molequla.go:4227) — no second index and
// no second tokenizer, as §13 asks.
func experienceCoverage(field *CooccurField, tok *EvolvingTokenizer, text string) (float64, int) {
	if field == nil || tok == nil {
		return 0, 0
	}
	ids := tok.Encode(experienceSample(text, CFG.ExperienceCoverageSampleBytes))
	if len(ids) < 2 {
		return 0, 0
	}
	field.mu.RLock()
	defer field.mu.RUnlock()
	if !field.Built {
		return 0, 0
	}
	seen := 0
	for i := 0; i+1 < len(ids); i++ {
		if m := field.BigramByFirst[ids[i]]; m != nil && m[ids[i+1]] > 0 {
			seen++
		}
	}
	return float64(seen) / float64(len(ids)-1), len(ids) - 1
}

// experienceReaders lists the elements allowed to read `src`, in dnaElements
// order: everybody but the writer. Every process computes the same list from
// the source name alone.
func experienceReaders(src string) []string {
	out := make([]string, 0, len(dnaElements))
	for _, e := range dnaElements {
		if e != src {
			out = append(out, e)
		}
	}
	return out
}

// experienceOwner names the one element guaranteed to eat this fragment.
// Keyed on the file name, so it is identical in all four processes and blind to
// what the fragment says. A mitosis child carries its parent's --element
// (molequla.go:5985-5988), so the slot survives division: both eat what the
// slot owns.
func experienceOwner(src, name string) string {
	readers := experienceReaders(src)
	if len(readers) == 0 {
		return ""
	}
	h := fnv.New64a()
	h.Write([]byte(src))
	h.Write([]byte{'/'})
	h.Write([]byte(name))
	return readers[int(h.Sum64()%uint64(len(readers)))]
}

// experienceCoverageFile is the ring's file, written in the organism's working
// directory beside dna_cursor.json (dna_field.go). The ring is a life's worth
// of measurement, not a cache: a restart that reset it would put the organism
// back into warming and hand the owner rule the whole colony for a few ticks.
const experienceCoverageFile = "experience_coverage.json"

// experienceQuantile is the q-quantile of an ascending slice by linear
// interpolation between the two neighbouring order statistics — the definition
// R's type 7 and numpy's default use, chosen because it is the one that returns
// the sample itself at q = 0 and q = 1 and moves continuously in between.
func experienceQuantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
	}
	if q <= 0 {
		return sorted[0]
	}
	if q >= 1 {
		return sorted[n-1]
	}
	pos := q * float64(n-1)
	lo := int(pos)
	frac := pos - float64(lo)
	if lo+1 >= n {
		return sorted[n-1]
	}
	return sorted[lo] + (sorted[lo+1]-sorted[lo])*frac
}

// recordCoverage appends one measurement to the ring and returns the two bars
// it defines, and whether the ring is warm enough to have defined them.
func (s *experienceState) recordCoverage(cov float64) (hi, lo float64, warm bool) {
	window := CFG.ExperienceCoverageWindow
	if window < 1 {
		window = 1
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cov = append(s.cov, cov)
	if len(s.cov) > window {
		s.cov = s.cov[len(s.cov)-window:]
	}
	s.covDirty = true
	if len(s.cov) < CFG.ExperienceCoverageWarm {
		return 0, 0, false
	}
	sorted := make([]float64, len(s.cov))
	copy(sorted, s.cov)
	sort.Float64s(sorted)
	return experienceQuantile(sorted, CFG.ExperienceResonanceQuantile),
		experienceQuantile(sorted, CFG.ExperienceNoveltyQuantile), true
}

// loadCoverage points the ring at its file and reads what is there. Called once
// at boot, beside loadDNACursor.
func (s *experienceState) loadCoverage(path string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.covPath = path
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var on struct {
		Coverage []float64 `json:"coverage"`
	}
	if json.Unmarshal(data, &on) != nil || on.Coverage == nil {
		return
	}
	window := CFG.ExperienceCoverageWindow
	if window > 0 && len(on.Coverage) > window {
		on.Coverage = on.Coverage[len(on.Coverage)-window:]
	}
	s.cov = on.Coverage
	s.covDirty = false
}

// saveCoverage writes the ring atomically, and only when a measurement has
// arrived since the last write. Called once per dnaRead pass, not once per
// fragment: at the read budgets that would be up to twelve writes a tick, the
// write storm the checkpoint debounce exists against.
func (s *experienceState) saveCoverage() {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.covPath == "" || !s.covDirty {
		s.mu.Unlock()
		return
	}
	path := s.covPath
	out := make([]float64, len(s.cov))
	copy(out, s.cov)
	s.covDirty = false
	s.mu.Unlock()
	data, err := json.Marshal(struct {
		Coverage []float64 `json:"coverage"`
	}{out})
	if err != nil {
		return
	}
	tmp := path + ".tmp"
	if os.WriteFile(tmp, data, 0644) == nil {
		os.Rename(tmp, path)
	}
}

// admits is the cafeteria decision for one fragment. It returns whether this
// organism eats it, why, and the coverage the decision was taken on (-1 when
// coverage was not consulted).
func (s *experienceState) admits(element, src, name, text string) (bool, string, float64) {
	if !CFG.ExperienceRouting {
		return true, "broadcast", -1
	}
	if experienceOwner(src, name) == element {
		return true, "owner", -1
	}
	s.mu.Lock()
	field, tok := s.field, s.tok
	s.mu.Unlock()
	cov, pairs := experienceCoverage(field, tok, text)
	if pairs < CFG.ExperienceMinPairs {
		// No field yet (a newborn before its first corpus build) or a fragment
		// too short to measure. A starving allocator is the failure mode §12's
		// second gate exists to catch, so an unmeasurable fragment is food.
		return true, "unmeasured", -1
	}
	// The measurement joins the organism's own distribution before it is judged
	// against it: the ring is what this organism has recently been offered, and
	// this fragment is part of that whether it is eaten or not.
	hi, lo, warm := s.recordCoverage(cov)
	if !warm {
		// No distribution yet, so no quantile and nothing to compare against.
		// The owner rule above has already guaranteed this fragment an eater,
		// so declining here starves nobody — and admitting instead would be the
		// byte-identical broadcast §12 refuses, handed out by default to every
		// organism that has just restarted.
		return false, "warming", cov
	}
	if cov >= hi {
		return true, "resonance", cov
	}
	if cov <= lo {
		return true, "novelty", cov
	}
	return false, "refused", cov
}

// ═══════════════════════════════════════════════════════════════════════════════
// WHAT A PASS DECIDED, IN ONE LINE
//
// admits is taken up to twelve times a tick and until this tally left no trace:
// after the 2026-09-15T12:00Z session `grep -ci declin molequla-run/*/*.stdout`
// found nothing, so whether the cafeteria had refused anything at all could not
// be read out of a session (MOLEQULALOG2.md, 2026-09-15, "the second session").
// A routing change that cannot be counted per session cannot be judged.
//
// It is its own line and not a widening of `[dna] … consumed` because that line
// is printed only when bytes were added (molequla.go:6680), and a pass that
// declines everything it was offered adds none — which is exactly the pass this
// tally exists to show. Making `consumed` print on those passes would put a
// `consumed 0 bytes from 0 files: []` line on the tick and change what the byte
// and event totals daily.sh already sums out of it mean. One line, one grep
// target, printed only when at least one fragment was judged.
//
// Aggregates only. At the read budgets (8 sibling + 4 extra) a line per
// fragment is up to twelve lines a tick, and there is no debug flag in this
// binary for the per-fragment source and coverage to hide behind — the only
// os.Getenv switch on this path is MOLEQULA_GPU_DEBUG, which is the bridge's —
// so the per-fragment detail is not printed at all rather than printed always.
// ═══════════════════════════════════════════════════════════════════════════════

// cafeteriaTally is what one dnaRead pass decided, counted by the branch of
// admits that decided it. "broadcast" is absent on purpose: with
// CFG.ExperienceRouting off there is no cafeteria, every fragment is food, and
// the caller prints nothing.
type cafeteriaTally struct {
	owner, resonance, novelty, unmeasured int // admitted, per branch
	band                                  int // declined: coverage inside the organism's own middle half
	warming                               int // declined: the ring is not yet long enough to be quantiled
	measured                              int // coverage measurements the pass spent (the ExperienceMaxMeasuredPerTick budget)
}

// record charges one fragment's decision to the tally.
func (t *cafeteriaTally) record(why string) {
	switch why {
	case "owner":
		t.owner++
	case "resonance":
		t.resonance++
	case "novelty":
		t.novelty++
	case "unmeasured":
		t.unmeasured++
	case "refused":
		t.band++
	case "warming":
		t.warming++
	}
}

func (t *cafeteriaTally) admitted() int  { return t.owner + t.resonance + t.novelty + t.unmeasured }
func (t *cafeteriaTally) declined() int  { return t.band + t.warming }
func (t *cafeteriaTally) decisions() int { return t.admitted() + t.declined() }

// line is the pass's line, newline included. admitted is the sum of the four
// bracketed reasons and declined the sum of its two, so the line can be checked
// against itself by whoever reads it. warming is its own reason and not folded
// into band because the two mean opposite things about the organism: band is a
// judgement, warming is the absence of one, and a session that cannot tell them
// apart cannot tell a restarted organism from a selective one.
func (t *cafeteriaTally) line(element string) string {
	return fmt.Sprintf("[cafeteria] %s admitted=%d (owner=%d resonance=%d novelty=%d unmeasured=%d) declined=%d (band=%d warming=%d) measured=%d\n",
		element, t.admitted(), t.owner, t.resonance, t.novelty, t.unmeasured,
		t.declined(), t.band, t.warming, t.measured)
}

// remember keeps an accepted fragment in the meal ring as the corpus lines it
// was appended as. dnaRead cuts a fragment with splitCorpusLine before the
// append, so these lines and not the fragment are what loadCorpusLines will
// hand back to dnaWrite as `docs`.
func (s *experienceState) remember(src string, lines []string, extra bool) {
	if s == nil || CFG.ExperienceMealMemory <= 0 {
		return
	}
	kept := make([]string, 0, len(lines))
	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if ln != "" {
			kept = append(kept, ln)
		}
	}
	if len(kept) == 0 {
		return
	}
	s.mu.Lock()
	s.meals = append(s.meals, experienceMeal{src: src, lines: kept, extra: extra})
	if len(s.meals) > CFG.ExperienceMealMemory {
		s.meals = s.meals[len(s.meals)-CFG.ExperienceMealMemory:]
	}
	s.mu.Unlock()
}

// experienceIsExtraSource reports whether a source directory is a sense rather
// than a sibling organism.
func experienceIsExtraSource(src string) bool {
	for _, e := range dnaElements {
		if e == src {
			return false
		}
	}
	return true
}

// experienceFirstSentence takes the leading sentence of a line, bounded, for use as a
// probe. Sentence enough: the first terminator, else the whole bounded line.
func experienceFirstSentence(s string, max int) string {
	s = strings.TrimSpace(strings.Join(strings.Fields(s), " "))
	if s == "" {
		return ""
	}
	if i := strings.IndexAny(s, ".!?"); i > 0 && i+1 < len(s) {
		s = s[:i+1]
	}
	if len(s) > max {
		s = strings.TrimSpace(s[:max])
	}
	return s
}

// probe returns the question this organism is asked next. §14: the world must
// be metabolized by somebody before it becomes culture, and the probe is where
// that happens — the organism is asked about what it just ate and answers in
// its own voice, instead of the fixed six-question round robin that made the
// emission independent of the meal. Sense food is preferred over sibling food
// because it is the only food that has not been metabolized by anybody yet.
// `step` rotates over the ring so consecutive ticks do not repeat one line.
// Returns "" when there is nothing eaten to ask about; the caller keeps its
// round robin then.
func (s *experienceState) probe(step int) string {
	if s == nil || !CFG.ExperienceProbeFromMeals {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.meals) == 0 {
		return ""
	}
	if step < 0 {
		step = -step
	}
	pick := func(extra bool) string {
		var cands []experienceMeal
		for _, m := range s.meals {
			if m.extra == extra {
				cands = append(cands, m)
			}
		}
		// A degenerate fragment must not yield a degenerate probe — the risk
		// the §18 order names against this step. An embryo's emitted fragment
		// opens with two or three bytes of its own speech and a full stop, so
		// its leading "sentence" can be "A." or "is a."; walk back through the
		// ring until a probe with substance appears, and hand the round robin
		// back if none has.
		for i := 0; i < len(cands); i++ {
			m := cands[len(cands)-1-(step+i)%len(cands)]
			p := experienceFirstSentence(m.lines[0], CFG.ExperienceProbeMaxChars)
			if len(p) >= CFG.ExperienceProbeMinChars && len(strings.Fields(p)) >= CFG.ExperienceProbeMinWords {
				return p
			}
		}
		return ""
	}
	if p := pick(true); p != "" {
		return p
	}
	return pick(false)
}

// recentPadding returns up to `n` recently eaten SIBLING lines, newest first,
// for the padding of an emitted fragment. Sense lines are deliberately absent:
// §14 forbids raw outside experience from entering collective DNA without
// passing through an organism, and padding is a byte copy, not a passage.
func (s *experienceState) recentPadding(n int) []string {
	if s == nil || n <= 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, n)
	for i := len(s.meals) - 1; i >= 0 && len(out) < n; i-- {
		if s.meals[i].extra {
			continue
		}
		for _, ln := range s.meals[i].lines {
			if len(out) >= n {
				break
			}
			out = append(out, ln)
		}
	}
	return out
}

// isRawExperience reports whether a corpus line is a sense fragment this
// organism ate verbatim. dnaRead appends a fragment to the corpus as one line
// and loadCorpusLines hands that line back truncated to CFG.MaxLineChars, so a
// random padding draw can re-emit outside experience byte for byte. This is the
// predicate that stops it.
func (s *experienceState) isRawExperience(line string) bool {
	if s == nil {
		return false
	}
	line = strings.TrimSpace(line)
	if line == "" {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, m := range s.meals {
		if !m.extra {
			continue
		}
		for _, ln := range m.lines {
			if line == ln || strings.HasPrefix(ln, line) || strings.HasPrefix(line, ln) {
				return true
			}
		}
	}
	return false
}

// ═══════════════════════════════════════════════════════════════════════════════
// THE §13 GATE — eligibility read from the voice, not from the label
//
// §13: "Eligibility for this shouldn't be tied mechanically to the label adult.
// It should depend on demonstrated coherence." The two numbers that say whether
// a fragment is the organism speaking or the corpus speaking through it already
// exist in the process and are already printed: model.lastGenMag, the mean
// |logit| of the raw output at the first step of the last generation, and
// fade = 1 - model.lastOverlayWeight (molequla.go:1843-1844, 6083-6091).
//
// Defaults from the live colony, 130 `[dna] wrote` lines under
// molequla-run/*/*.stdout, session ending 2026-09-13T20:29:30Z:
//
//   fade  = 1.00 on all 130 lines — the overlay is gone everywhere, so fade is
//           a necessary condition that refuses nobody in this colony today. It
//           still binds whenever the overlay is running below its own fade
//           width (overlayFadeProgress, metaweights_overlay.go:53).
//   mag   = 2.82 .. 16.35, median 8.66. Below 6.00 every one of 10 observed
//           generations produced gen=0 — the organism emitted no text at all.
//           At or above 6.00, 92 of 120 produced text. 6.00 is that floor.
//
// All four organisms were at stage=3 across that whole range, which is why the
// gate must not read the stage: one label covers organisms that spoke 197 bytes
// and organisms that spoke none.
//
// This function only decides. The injection itself is not implemented here; the
// decision is printed as eligible=0|1 on the `[dna] wrote` line so the next
// session can read it out of the logs.
// ═══════════════════════════════════════════════════════════════════════════════

// injectionEligible reports whether this organism's voice is its own enough to
// receive sentence-boundary knowledge injection, and why not when it is not.
func injectionEligible(fade, mag float64) (bool, string) {
	if fade < CFG.InjectionFadeMin {
		return false, "overlay"
	}
	if mag < CFG.InjectionMagMin {
		return false, "mute"
	}
	return true, "voice"
}

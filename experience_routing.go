package main

import (
	"fmt"
	"hash/fnv"
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
//                above CFG.ExperienceResonanceHigh.
//   novelty    — the fragment is outside the reader's language: that same share
//                is at or below CFG.ExperienceNoveltyLow. Without this branch an
//                organism is sealed inside what it already knows, and the senses
//                — the only food that is new by construction — would never be
//                eaten by anybody but their hash owner. See the measured
//                distribution in MOLEQULALOG2.md, 2026-09-15.
//
// The band in between is refused. Coverage is state: it rises as the organism
// eats, so the same fragment file routes differently in week two than in week
// one, which is the drift §12 is after.
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
// receives both), and the last CFG.ExperienceMealMemory fragments eaten.
type experienceState struct {
	mu    sync.Mutex
	field *CooccurField
	tok   *EvolvingTokenizer
	meals []experienceMeal
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
	if cov >= CFG.ExperienceResonanceHigh {
		return true, "resonance", cov
	}
	if cov <= CFG.ExperienceNoveltyLow {
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
	band                                  int // declined: coverage between the two thresholds
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
	}
}

func (t *cafeteriaTally) admitted() int  { return t.owner + t.resonance + t.novelty + t.unmeasured }
func (t *cafeteriaTally) decisions() int { return t.admitted() + t.band }

// line is the pass's line, newline included. admitted is the sum of the four
// bracketed reasons and declined the sum of the one, so the line can be checked
// against itself by whoever reads it.
func (t *cafeteriaTally) line(element string) string {
	return fmt.Sprintf("[cafeteria] %s admitted=%d (owner=%d resonance=%d novelty=%d unmeasured=%d) declined=%d (band=%d) measured=%d\n",
		element, t.admitted(), t.owner, t.resonance, t.novelty, t.unmeasured, t.band, t.band, t.measured)
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

package main

import (
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

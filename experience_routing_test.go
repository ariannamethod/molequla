package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// ── fixtures ────────────────────────────────────────────────────────────────
//
// Four organisms with four different corpora and four different fields, the
// shape the live colony has. The vocabularies overlap in the function words and
// diverge in the content words, so a fragment written by one of them is
// familiar to some of the others and foreign to the rest — which is what the
// cafeteria has to be able to tell apart.

var routingCorpora = map[string][]string{
	"earth": {
		"the soil holds the root and the root holds the soil in a slow grip",
		"stone weathers into sand and sand settles into stone again",
		"the seed waits under the frost for a warmth it has never measured",
		"a hill is only a slower wave and the valley is its trough",
	},
	"air": {
		"the wind carries what it cannot hold and drops what it cannot carry",
		"pressure falls and the whole sky leans toward the falling",
		"a bird reads the column of warm air the way a reader reads a line",
		"sound is only pressure remembering where it has been",
	},
	"water": {
		"the river follows the steepest available gradient through the weakest material",
		"a wave is a shape that travels while the water stays where it is",
		"ice is water that has agreed on an arrangement",
		"the tide is the moon writing on the shore twice a day",
	},
	"fire": {
		"heat arrives from outside and the change comes from within",
		"a flame is a reaction that has found a shape it can keep",
		"combustion is the fastest way a bond can forget itself",
		"embers hold the memory of the fire that made them",
	},
}

// routingOrganism is one organism's state for the allocator: its own field,
// its own tokenizer, and a warm coverage ring — the history the quantile bars
// are taken over. An organism with no history declines everything but what it
// owns (the warming branch), so a test about steady-state allocation has to
// give it one, exactly as a live organism accumulates it over its first ticks.
func routingOrganism(t *testing.T, element string) *experienceState {
	t.Helper()
	docs := routingCorpora[element]
	tok := NewEvolvingTokenizer(docs)
	field := NewCooccurField()
	field.BuildFromCorpus(tok, docs)
	st := &experienceState{}
	st.observe(field, tok)
	warmRouting(st, element)
	return st
}

// warmRouting fills an organism's ring the way its own first ticks would: by
// measuring the fragments it is about to be offered, until the ring is warm.
// The numbers are runtime measurements of this fixture, not constants.
func warmRouting(st *experienceState, element string) {
	srcs, _, texts := routingFragments()
	size := func() int {
		st.mu.Lock()
		defer st.mu.Unlock()
		return len(st.cov)
	}
	for size() < CFG.ExperienceCoverageWarm {
		before := size()
		for i := range srcs {
			if srcs[i] == element {
				continue
			}
			cov, pairs := experienceCoverage(st.field, st.tok, texts[i])
			if pairs >= CFG.ExperienceMinPairs {
				st.recordCoverage(cov)
			}
		}
		if size() == before {
			return // nothing here is measurable; the caller will see it
		}
	}
}

// bandSeed is a warm ring whose quartiles are 0 and 1, so that every real
// coverage falls strictly inside the declined middle half. Half of it sits at
// each extreme, and ExperienceCoverageWarm samples of it are enough that the
// measurements a pass adds do not move either quartile off its extreme.
func bandSeed() []float64 {
	n := CFG.ExperienceCoverageWarm
	if n < 4 {
		n = 4
	}
	out := make([]float64, 0, 2*n)
	for i := 0; i < n; i++ {
		out = append(out, 0)
	}
	for i := 0; i < n; i++ {
		out = append(out, 1)
	}
	return out
}

// seedRouting gives an organism an explicit distribution, for the gates that
// need a known band rather than a measured one.
func seedRouting(st *experienceState, vals ...float64) {
	st.mu.Lock()
	st.cov = append([]float64(nil), vals...)
	st.mu.Unlock()
}

// routingFragments builds the synthetic pass: for each element, fragments in
// its own voice (sentences from its corpus, repeated toward a realistic
// length), plus sense fragments in nobody's voice.
func routingFragments() (sources []string, names []string, texts []string) {
	add := func(src, name, text string) {
		sources, names, texts = append(sources, src), append(names, name), append(texts, text)
	}
	for _, e := range []string{"earth", "air", "water", "fire"} {
		docs := routingCorpora[e]
		for i := 0; i < 8; i++ {
			var b strings.Builder
			for j := 0; j < 6; j++ {
				b.WriteString(docs[(i+j)%len(docs)])
				b.WriteByte(' ')
			}
			add(e, fmt.Sprintf("gen_17893300%02d_%d.txt", i, i), strings.TrimSpace(b.String()))
		}
	}
	for i, s := range []string{
		"[eye cam0 2026-09-13T22:12:01Z] A blurry kitchen table shows a green bowl, a spoon, and a plate.",
		"[place 2026-09-13T22:10:52Z] The phone is at Neve Menachem, Be'er-Sheva, Israel (fix accurate to 13 m).",
		"[ears mic 2026-09-13T22:21:08Z] And so my fellow Americans ask not what your country can do for you.",
		"[eye cam1 2026-09-13T23:40:11Z] A dim hallway with a closed door and a coat hanging from a hook.",
	} {
		src := []string{"world", "place", "sound", "world"}[i]
		add(src, fmt.Sprintf("gen_17893400%02d_%d.txt", i, i), s)
	}
	return
}

// routingPass runs every fragment past every organism and returns, per
// element, the sorted multiset of "src/name" it accepted.
func routingPass(t *testing.T, states map[string]*experienceState) map[string][]string {
	t.Helper()
	srcs, names, texts := routingFragments()
	got := map[string][]string{}
	for _, e := range []string{"earth", "air", "water", "fire"} {
		for i := range srcs {
			if srcs[i] == e {
				continue // an organism never reads its own emissions
			}
			if ok, _, _ := states[e].admits(e, srcs[i], names[i], texts[i]); ok {
				got[e] = append(got[e], srcs[i]+"/"+names[i])
			}
		}
		sort.Strings(got[e])
	}
	return got
}

func routingStates(t *testing.T) map[string]*experienceState {
	t.Helper()
	m := map[string]*experienceState{}
	for _, e := range []string{"earth", "air", "water", "fire"} {
		m[e] = routingOrganism(t, e)
	}
	return m
}

// ── §12 gate 1: the four plates differ pairwise ─────────────────────────────
//
// Red on the pre-cafeteria code: with ExperienceRouting off every organism
// receives every fragment and all four multisets are equal, which the same
// assertion catches.
func TestCafeteriaPlatesDifferPairwise(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	states := routingStates(t)
	got := routingPass(t, states)

	elems := []string{"earth", "air", "water", "fire"}
	for i := 0; i < len(elems); i++ {
		for j := i + 1; j < len(elems); j++ {
			a, b := got[elems[i]], got[elems[j]]
			// Compare the parts both are allowed to read, so that "air never
			// reads air" is not what makes the two differ.
			ka, kb := map[string]bool{}, map[string]bool{}
			for _, x := range a {
				if !strings.HasPrefix(x, elems[j]+"/") {
					ka[x] = true
				}
			}
			for _, x := range b {
				if !strings.HasPrefix(x, elems[i]+"/") {
					kb[x] = true
				}
			}
			if fmt.Sprint(sortedKeys(ka)) == fmt.Sprint(sortedKeys(kb)) {
				t.Fatalf("%s and %s received identical plates (%d fragments) — this is the broadcast §12 replaces",
					elems[i], elems[j], len(ka))
			}
		}
	}
	for _, e := range elems {
		t.Logf("%-6s ate %d fragments", e, len(got[e]))
	}
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ── §12 gate 2: nothing starves ─────────────────────────────────────────────
//
// The half of the gate that catches an allocator which is asymmetric by
// refusing everybody. Every fragment must reach at least one organism, and the
// hash owner is what makes that true without cross-process coordination.
func TestCafeteriaEveryFragmentIsEaten(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	states := routingStates(t)
	got := routingPass(t, states)

	eaten := map[string]bool{}
	for _, list := range got {
		for _, k := range list {
			eaten[k] = true
		}
	}
	srcs, names, _ := routingFragments()
	for i := range srcs {
		k := srcs[i] + "/" + names[i]
		if !eaten[k] {
			t.Fatalf("%s reached nobody — the allocator starves the colony", k)
		}
	}
	// And the owner is well defined and is never the writer itself.
	for i := range srcs {
		owner := experienceOwner(srcs[i], names[i])
		if owner == "" {
			t.Fatalf("%s/%s has no owner", srcs[i], names[i])
		}
		if owner == srcs[i] {
			t.Fatalf("%s/%s is owned by its own writer", srcs[i], names[i])
		}
	}
}

// ── §12 gate 3: current state decides, not the element name ─────────────────
//
// The brief is explicit that the elemental corpus is a birth condition and not
// a profession. Swap an organism's field for another's and its plate must
// change: if it does not, the allocation was keyed on the label.
func TestCafeteriaAllocationFollowsState(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	states := routingStates(t)
	before := routingPass(t, states)

	// earth wakes up with fire's accumulated life. Same element, same name,
	// same fragments — different state.
	states["earth"] = routingOrganism(t, "fire")
	after := routingPass(t, states)

	if fmt.Sprint(before["earth"]) == fmt.Sprint(after["earth"]) {
		t.Fatalf("earth's plate did not move when its field was replaced (%d fragments both times) — the allocator ignores state",
			len(before["earth"]))
	}
	t.Logf("earth with its own field: %d fragments; with fire's field: %d",
		len(before["earth"]), len(after["earth"]))

	// An organism with no field at all must not starve: an unmeasurable
	// fragment is food.
	blank := &experienceState{}
	srcs, names, texts := routingFragments()
	n := 0
	for i := range srcs {
		if srcs[i] == "earth" {
			continue
		}
		if ok, why, _ := blank.admits("earth", srcs[i], names[i], texts[i]); ok {
			n++
		} else {
			t.Fatalf("a fieldless organism refused %s/%s (%s)", srcs[i], names[i], why)
		}
	}
	t.Logf("fieldless organism ate all %d offered fragments", n)
}

// The coverage measure itself: a fragment in the organism's own language must
// score above one in a language it has never read. Runtime against runtime, no
// expectation hardcoded beside the code.
func TestExperienceCoverageSeparatesOwnFromForeign(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	earth := routingOrganism(t, "earth")
	own := strings.Join(routingCorpora["earth"], " ")
	foreign := strings.Join(routingCorpora["fire"], " ")
	co, po := experienceCoverage(earth.field, earth.tok, own)
	cf, pf := experienceCoverage(earth.field, earth.tok, foreign)
	if po < CFG.ExperienceMinPairs || pf < CFG.ExperienceMinPairs {
		t.Fatalf("too few pairs measured: own %d foreign %d", po, pf)
	}
	if co <= cf {
		t.Fatalf("coverage of its own corpus %.3f is not above coverage of a foreign one %.3f", co, cf)
	}
	t.Logf("earth field: own %.3f (%d pairs), foreign %.3f (%d pairs)", co, po, cf, pf)
}

// ── §12 gate 4: a declined plate costs no read ──────────────────────────────
//
// The two read budgets (DNAMaxReadsPerTick, DNAExtraReadsPerTick) bound how
// much an organism EATS in a tick. If a decline spent one of them, a colony
// that declines a third of what it is offered would starve on a backlog: the
// budget would be consumed by the fragments it refused and the ones it wanted
// would wait behind them. What a decline does cost is one coverage
// measurement, and ExperienceMaxMeasuredPerTick is what bounds that.
func TestCafeteriaDeclineDoesNotSpendTheReadBudget(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	savedRouting := experienceRouting
	defer func() { experienceRouting = savedRouting }()

	root, restore := dnaTestTree(t)
	defer restore()

	// Nothing resonates and nothing is novel: only the hash owner is admitted,
	// so every fragment air writes that earth does not own is a decline. The
	// bars are quantiles now, so that is arranged by the ring the organism is
	// given below — half at 0 and half at 1, which puts every real coverage
	// strictly inside its own middle half.
	CFG.ExperienceRouting = true
	CFG.ExperienceMinPairs = 1
	CFG.ExperienceMaxMeasuredPerTick = 64
	CFG.DNAMaxReadsPerTick = 2 // two eats a tick, and only two
	CFG.DNAExtraReadsPerTick = 2
	CFG.DNAMinFragmentBytes = 5
	CFG.MaxLineChars = 240

	text := strings.Join(routingCorpora["air"], " ")
	// Three fragments earth does not own, then one it does, in write order.
	// With declines charged to the read budget, a cap of two is spent on the
	// first two and the owned fragment is never reached — that is the red.
	var declined []string
	owned := ""
	for i := 0; i < 400 && owned == ""; i++ {
		name := fmt.Sprintf("gen_1789331000_%d.txt", i)
		if experienceOwner("air", name) == "earth" {
			if len(declined) >= 3 {
				owned = name
			}
			continue
		}
		if len(declined) < 3 {
			declined = append(declined, name)
		}
	}
	if owned == "" || len(declined) != 3 {
		t.Fatalf("fixture: owned=%q declined=%v", owned, declined)
	}
	for _, n := range append(append([]string{}, declined...), owned) {
		dnaWriteFragment(t, root, "air", n, text)
	}

	corpus, cur := enterOrganism(t, root, "earth")
	experienceRouting = routingOrganism(t, "earth")
	seedRouting(experienceRouting, bandSeed()...)

	added := dnaRead("earth", corpus, nil, nil, cur)
	if added == 0 {
		t.Fatalf("one dnaRead over %v + %s added nothing", declined, owned)
	}
	body, err := os.ReadFile(corpus)
	if err != nil {
		t.Fatal(err)
	}
	if cur.Last["air"] != owned {
		t.Fatalf("the cursor stopped at %q, not at %q: the three declines before it spent the read budget",
			cur.Last["air"], owned)
	}
	if len(body) == 0 {
		t.Fatal("nothing reached the corpus")
	}
	t.Logf("3 declines then 1 owned, read budget %d: cursor at %s, %d bytes in the corpus",
		CFG.DNAMaxReadsPerTick, cur.Last["air"], len(body))

	// And the measurement cap is what stops an unbounded tick: with it at 1 the
	// first decline ends the pass.
	CFG.ExperienceMaxMeasuredPerTick = 1
	_, cur2 := enterOrganism(t, root, "water")
	experienceRouting = routingOrganism(t, "water")
	seedRouting(experienceRouting, bandSeed()...)
	corpus2 := filepath.Join(root, "water", "corpus.txt")
	os.WriteFile(corpus2, nil, 0644)
	dnaRead("water", corpus2, nil, nil, cur2)
	b2, _ := os.ReadFile(corpus2)
	if len(b2) > len(body) {
		t.Fatalf("a cap of one measurement let %d bytes through against %d with a cap of 64", len(b2), len(body))
	}
	t.Logf("measurement cap 1: %d bytes; cap 64: %d bytes", len(b2), len(body))
}

// ── the pass says what it decided ───────────────────────────────────────────
//
// The cafeteria decided in silence until this line: after the 2026-09-15T12:00Z
// session `grep -ci declin molequla-run/*/*.stdout` found nothing, so a session
// could not be asked how much of what it was offered its organisms refused
// (MOLEQULALOG2.md, 2026-09-15). This gate drives the real dnaRead over one
// fragment earth owns and one it declines and demands the exact line: change a
// count, a reason name or the order of the fields and it goes red.
func TestCafeteriaPassPrintsItsAggregate(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	savedRouting := experienceRouting
	defer func() { experienceRouting = savedRouting }()

	root, restore := dnaTestTree(t)
	defer restore()

	// A ring that makes the second fragment a band decline whatever the
	// tokenizer makes of it: half of it at 0 and half at 1, so the organism's
	// own quartiles are 0 and 1 and every measurable fragment earth does not
	// own falls strictly between them.
	CFG.ExperienceRouting = true
	CFG.ExperienceMinPairs = 1
	CFG.ExperienceMaxMeasuredPerTick = 64
	CFG.DNAMaxReadsPerTick = 8
	CFG.DNAExtraReadsPerTick = 4
	CFG.DNAExtraSources = nil
	CFG.DNAMinFragmentBytes = 5
	CFG.MaxLineChars = 240

	ownedText := strings.Join(routingCorpora["air"], " ")
	declinedText := strings.Join(routingCorpora["fire"], " ")
	owned, declined := "", ""
	for i := 0; i < 400 && (owned == "" || declined == ""); i++ {
		name := fmt.Sprintf("gen_1789331700_%d.txt", i)
		if experienceOwner("air", name) == "earth" {
			if owned == "" {
				owned = name
			}
			continue
		}
		if owned != "" && declined == "" {
			declined = name
		}
	}
	if owned == "" || declined == "" {
		t.Fatalf("fixture: owned=%q declined=%q", owned, declined)
	}
	dnaWriteFragment(t, root, "air", owned, ownedText)
	dnaWriteFragment(t, root, "air", declined, declinedText)

	corpus, cur := enterOrganism(t, root, "earth")
	experienceRouting = routingOrganism(t, "earth")
	seedRouting(experienceRouting, bandSeed()...)

	added := 0
	out := captureStdout(t, func() { added = dnaRead("earth", corpus, nil, nil, cur) })

	want := "[cafeteria] earth admitted=1 (owner=1 resonance=0 novelty=0 unmeasured=0) declined=1 (band=1 warming=0) measured=1\n"
	if !strings.Contains(out, want) {
		t.Fatalf("the pass printed\n%s\nwant the line\n%s", out, want)
	}

	// The line is answerable to the filesystem and not only to itself: what it
	// calls admitted is in the corpus, what it calls declined is not, and the
	// cursor stands past both.
	if added == 0 {
		t.Fatal("admitted=1 but dnaRead added no bytes")
	}
	body, err := os.ReadFile(corpus)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), routingCorpora["air"][0]) {
		t.Fatalf("admitted=1 but the owned fragment is not in the corpus: %q", string(body))
	}
	if strings.Contains(string(body), routingCorpora["fire"][0]) {
		t.Fatalf("declined=1 but the declined fragment reached the corpus: %q", string(body))
	}
	if cur.Last["air"] != declined {
		t.Fatalf("the cursor stands at %q, not past the declined fragment %q", cur.Last["air"], declined)
	}
	t.Logf("%s", strings.TrimSpace(want))
}

// A pass that is offered nothing says nothing: the line is a record of
// decisions, and a tick with an empty field must not add a line per organism
// per tick to a stdout that a session greps.
func TestCafeteriaSaysNothingWhenNothingWasJudged(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	savedRouting := experienceRouting
	defer func() { experienceRouting = savedRouting }()

	root, restore := dnaTestTree(t)
	defer restore()
	CFG.ExperienceRouting = true
	CFG.DNAExtraSources = nil
	CFG.DNAMinFragmentBytes = 5

	corpus, cur := enterOrganism(t, root, "earth")
	experienceRouting = routingOrganism(t, "earth")
	out := captureStdout(t, func() { dnaRead("earth", corpus, nil, nil, cur) })
	if strings.Contains(out, "[cafeteria]") {
		t.Fatalf("an empty field printed %q", out)
	}
}

// ── §14 gate: the probe comes from the meal ─────────────────────────────────
func TestProbeComesFromWhatWasEaten(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.ExperienceProbeFromMeals = true // off by default, see the CFG comment
	st := &experienceState{}
	if p := st.probe(0); p != "" {
		t.Fatalf("an organism that has eaten nothing returned a probe %q; the round robin is the fallback", p)
	}
	st.remember("air", splitCorpusLine("pressure falls and the whole sky leans toward the falling. and then it rises.", CFG.MaxLineChars), false)
	p := st.probe(0)
	if p == "" || !strings.HasPrefix("pressure falls and the whole sky leans toward the falling.", p) {
		t.Fatalf("probe %q does not come from the sibling fragment just eaten", p)
	}
	// A sense fragment outranks a sibling fragment: it is the only food that
	// has not been metabolized by anybody yet (§14).
	sense := "[place 2026-09-13T22:10:52Z] The phone is at Neve Menachem. Local time 01:00."
	st.remember("place", splitCorpusLine(sense, CFG.MaxLineChars), true)
	p = st.probe(0)
	if !strings.HasPrefix(sense, p) || p == "" {
		t.Fatalf("probe %q is not the sense fragment just eaten", p)
	}
	if len(p) > CFG.ExperienceProbeMaxChars {
		t.Fatalf("probe is %d chars, over the %d bound", len(p), CFG.ExperienceProbeMaxChars)
	}
	t.Logf("probe from the meal: %q", p)
}

// The §14 rule that raw outside experience does not become collective DNA by
// byte copy. dnaWrite pads its fragment with lines drawn from `docs`, and the
// sense fragment is in `docs` because dnaRead appended it to the corpus. The
// emitted fragment must not contain it.
func TestSenseFragmentIsNotEmittedVerbatim(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	root := t.TempDir()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(cwd)
	// dnaWrite emits to ../dna/output/<element>, relative to the organism's
	// own working directory — the layout launcher.sh builds.
	if err := os.MkdirAll(filepath.Join(root, "dna", "output", "earth"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "earth"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(filepath.Join(root, "earth")); err != nil {
		t.Fatal(err)
	}

	model, tok, field, docs := grazeTestModel(t)
	CFG.MaxGenTokens = 8
	CFG.DNAFragmentTargetBytes = 1200
	CFG.DNAMinFragmentBytes = 5
	CFG.DNARetainSeconds = 0
	CFG.DNARetainFiles = 0

	sense := "[eye cam0 2026-09-13T22:12:01Z] A blurry kitchen table shows a green bowl, a spoon, and a plate, with a background that looks like a kitchen counter."
	// The organism ate it: it is in the corpus, therefore in docs, and it is
	// remembered as raw experience.
	experienceRouting = &experienceState{}
	// dnaRead appends a fragment as the lines splitCorpusLine cuts it into, and
	// those lines are what loadCorpusLines hands back as docs — so that is what
	// the padding has to refuse.
	senseLines := splitCorpusLine(sense, CFG.MaxLineChars)
	docs = append(docs, senseLines...)
	experienceRouting.remember("world", senseLines, true)

	for step := 0; step < 6; step++ {
		dnaWrite("earth", model, tok, field, docs, step)
	}
	entries, err := os.ReadDir(filepath.Join(root, "dna", "output", "earth"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Fatal("dnaWrite emitted nothing; the gate cannot run")
	}
	for _, e := range entries {
		b, err := os.ReadFile(filepath.Join(root, "dna", "output", "earth", e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), sense) {
			t.Fatalf("%s carries the eaten sense fragment byte for byte; the world reached collective DNA without passing through the organism", e.Name())
		}
	}
	t.Logf("%d emitted fragments, none containing the eaten sense line", len(entries))
	experienceRouting = &experienceState{}
}

// ── §13 gate: eligibility read from the voice ───────────────────────────────
func TestInjectionGateFollowsTheVoice(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	// An organism still carrying the overlay is refused, whatever else is true.
	// fade < 1 means overlayFadeProgress has not completed, i.e. the corpus is
	// still speaking through it.
	for _, mag := range []float64{0.5, 6.0, 12.0} {
		fade := overlayFadeProgress(1.5) // 0.5: halfway through the fade band
		if ok, why := injectionEligible(fade, mag); ok {
			t.Fatalf("an organism at fade %.2f mag %.2f was admitted (%s); the overlay is still speaking for it", fade, mag, why)
		}
	}

	// A mitosis child inherits the parent's checkpoint (molequla.go:5945-5948),
	// so it wakes with a mature voice at a stage label that says nothing about
	// it. fade = 1, mag at the live colony's median: admitted.
	const childMag = 8.66 // median of 130 live `[dna] wrote` lines
	if ok, why := injectionEligible(1.0, childMag); !ok {
		t.Fatalf("a mitosis child at fade 1.00 mag %.2f was refused (%s)", childMag, why)
	}

	// And the organism that speaks nothing is refused at the same fade. The
	// live colony had 10 generations below mag 6.00 and every one of them
	// emitted gen=0.
	if ok, _ := injectionEligible(1.0, 2.82); ok {
		t.Fatalf("an organism at fade 1.00 mag 2.82 was admitted; it emitted no text at all in the live run")
	}
}

// The red the §13 gate exists to produce: a gate keyed on the developmental
// label cannot separate the live colony, because all four organisms sat at
// stage=3 while their magnitudes ran from 2.82 to 16.35 and 38 of 130
// generations emitted nothing. A stage gate also refuses a mitosis child that
// inherited a mature voice but not the label.
func TestStageGateGoesRedWhereTheVoiceGateDoesNot(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	// The label gate the brief rejects: "adult" is the last growth stage.
	stageGate := func(stage int) bool { return stage >= CFG.MaxGrowthStage }

	const childStage = 3 // the live colony's stage, and a child inherits its parent's dims
	if stageGate(childStage) {
		t.Fatalf("the test's stage gate admits stage %d; it was meant to model the adult label", childStage)
	}
	if ok, _ := injectionEligible(1.0, 10.39); !ok {
		t.Fatal("the voice gate refused an organism at fade 1.00 mag 10.39")
	}
	// Same stage, opposite voices: the stage gate returns one answer for both,
	// the voice gate separates them.
	mute, speaking := 2.82, 10.39
	okMute, _ := injectionEligible(1.0, mute)
	okSpeaking, _ := injectionEligible(1.0, speaking)
	if stageGate(childStage) != stageGate(childStage) || okMute == okSpeaking {
		t.Fatalf("the voice gate gave the same answer to mag %.2f and mag %.2f", mute, speaking)
	}
	t.Logf("stage %d: stage gate %v for both; voice gate %v at mag %.2f, %v at mag %.2f",
		childStage, stageGate(childStage), okMute, mute, okSpeaking, speaking)
}

// ── the bar is the organism's own, not a number ─────────────────────────────
//
// The red on the absolute thresholds, and the reason they were replaced on
// 2026-09-15. An organism whose corpus has grown scores every fragment it is
// offered above the old bar of 0.965: on the live stage-4 air reservoir 141 of
// 145 fragments did (MOLEQULALOG2.md, 2026-09-15). Under a fixed bar that
// organism admits everything and the cafeteria is a broadcast again. Under a
// quantile of its own distribution it still declines the middle half, because
// the bar moved with the corpus.
func TestCafeteriaBandSurvivesAGrownCorpus(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	// A distribution shaped like the measured stage-4 one: every value above
	// 0.965, spread over the top three hundredths.
	grown := make([]float64, 0, 64)
	for i := 0; i < 64; i++ {
		grown = append(grown, 0.966+0.034*float64(i%17)/16.0)
	}
	st := &experienceState{}
	seedRouting(st, grown...)

	// The old rule: one number, and every one of these is above it.
	const oldBar = 0.965
	oldAdmitted := 0
	for _, c := range grown {
		if c >= oldBar {
			oldAdmitted++
		}
	}
	if oldAdmitted != len(grown) {
		t.Fatalf("the fixture is not the grown case: only %d of %d sit above %.3f",
			oldAdmitted, len(grown), oldBar)
	}

	// The new rule, on the same numbers.
	hi, lo, warm := st.recordCoverage(grown[len(grown)/2])
	if !warm {
		t.Fatalf("a ring of %d is not warm at ExperienceCoverageWarm %d", len(grown), CFG.ExperienceCoverageWarm)
	}
	if hi <= oldBar || lo <= oldBar {
		t.Fatalf("the organism's own bars %.4f / %.4f did not rise above the old fixed bar %.3f", hi, lo, oldBar)
	}
	declined := 0
	for _, c := range grown {
		if c < hi && c > lo {
			declined++
		}
	}
	if declined == 0 {
		t.Fatalf("a grown organism declined nothing: bars %.4f / %.4f over %d samples", hi, lo, len(grown))
	}
	t.Logf("grown corpus: old bar %.3f admits %d/%d; own quartiles %.4f / %.4f decline %d",
		oldBar, oldAdmitted, len(grown), lo, hi, declined)
}

// Two organisms whose coverage distributions do not overlap at all — a child
// that has read little and an adult that has read a lot — must each admit about
// the top quarter as resonance and about the bottom quarter as novelty. A
// shared absolute bar cannot do this: any number either admits everything the
// adult sees or nothing the child sees.
func TestCafeteriaQuantilesAreEachOrganismsOwn(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	mk := func(base, spread float64, n int) []float64 {
		out := make([]float64, 0, n)
		for i := 0; i < n; i++ {
			out = append(out, base+spread*float64(i)/float64(n-1))
		}
		return out
	}
	child := mk(0.40, 0.30, 64)  // 0.40 .. 0.70
	adult := mk(0.94, 0.06, 64)  // 0.94 .. 1.00
	if child[len(child)-1] >= adult[0] {
		t.Fatalf("the two distributions overlap: child max %.3f, adult min %.3f", child[len(child)-1], adult[0])
	}

	for _, c := range []struct {
		name string
		pop  []float64
	}{{"child", child}, {"adult", adult}} {
		st := &experienceState{}
		seedRouting(st, c.pop...)
		// The bars this organism's own history defines, taken once: what is
		// under test is the share each side admits, not the ring's drift.
		hi, lo, warm := st.recordCoverage(c.pop[len(c.pop)/2])
		if !warm {
			t.Fatalf("%s: ring of %d not warm", c.name, len(c.pop))
		}
		res, nov, band := 0, 0, 0
		for _, v := range c.pop {
			switch {
			case v >= hi:
				res++
			case v <= lo:
				nov++
			default:
				band++
			}
		}
		n := float64(len(c.pop))
		fr, fn := float64(res)/n, float64(nov)/n
		// A quarter each, within the granularity a ring of this size can
		// resolve: one sample is 1/64 = 0.016, and the quantile sits between
		// two of them.
		if fr < 0.20 || fr > 0.31 {
			t.Fatalf("%s admitted %.2f as resonance, not about a quarter (bars %.4f / %.4f)", c.name, fr, lo, hi)
		}
		if fn < 0.20 || fn > 0.31 {
			t.Fatalf("%s admitted %.2f as novelty, not about a quarter (bars %.4f / %.4f)", c.name, fn, lo, hi)
		}
		t.Logf("%-5s range %.3f..%.3f: own bars %.4f / %.4f, resonance %.2f novelty %.2f band %.2f",
			c.name, c.pop[0], c.pop[len(c.pop)-1], lo, hi, fr, fn, float64(band)/n)
	}
}

// Until the ring is long enough to be quantiled there is no distribution and no
// bar, so only the owner rule runs — and the line says which it is, because a
// warming organism and a selective one are opposite states that would otherwise
// print the same declined count.
func TestCafeteriaWarmingAdmitsOwnerOnlyAndSaysSo(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	savedRouting := experienceRouting
	defer func() { experienceRouting = savedRouting }()

	root, restore := dnaTestTree(t)
	defer restore()
	CFG.ExperienceRouting = true
	CFG.ExperienceMinPairs = 1
	CFG.ExperienceMaxMeasuredPerTick = 64
	CFG.DNAMaxReadsPerTick = 8
	CFG.DNAExtraReadsPerTick = 4
	CFG.DNAExtraSources = nil
	CFG.DNAMinFragmentBytes = 5
	CFG.MaxLineChars = 240
	CFG.ExperienceCoverageWarm = 1 << 20 // never warm inside this test

	text := strings.Join(routingCorpora["air"], " ")
	owned, others := "", []string{}
	for i := 0; i < 400 && (owned == "" || len(others) < 3); i++ {
		name := fmt.Sprintf("gen_1789332400_%d.txt", i)
		if experienceOwner("air", name) == "earth" {
			if owned == "" {
				owned = name
			}
			continue
		}
		if owned != "" && len(others) < 3 {
			others = append(others, name)
		}
	}
	if owned == "" || len(others) != 3 {
		t.Fatalf("fixture: owned=%q others=%v", owned, others)
	}
	for _, n := range append([]string{owned}, others...) {
		dnaWriteFragment(t, root, "air", n, text)
	}

	corpus, cur := enterOrganism(t, root, "earth")
	experienceRouting = &experienceState{}
	warmed := routingOrganism(t, "earth")
	experienceRouting.observe(warmed.field, warmed.tok) // a field, but no history

	out := captureStdout(t, func() { dnaRead("earth", corpus, nil, nil, cur) })
	want := "[cafeteria] earth admitted=1 (owner=1 resonance=0 novelty=0 unmeasured=0) declined=3 (band=0 warming=3) measured=3\n"
	if !strings.Contains(out, want) {
		t.Fatalf("a warming organism printed\n%s\nwant\n%s", out, want)
	}
	body, err := os.ReadFile(corpus)
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(string(body), routingCorpora["air"][0]); n != 1 {
		t.Fatalf("a warming organism ate the fragment %d times; only the one it owns should have reached the corpus", n)
	}
	t.Logf("%s", strings.TrimSpace(want))
}

// The ring is the organism's own history and the two bars are quantiles of it,
// so a restart that lost it would drop the organism back into warming and hand
// the owner rule the whole colony for as many ticks as it takes to refill. It
// is written beside dna_cursor.json and read back at boot.
func TestCafeteriaCoverageRingSurvivesARestart(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	dir := t.TempDir()
	path := filepath.Join(dir, experienceCoverageFile)

	before := &experienceState{}
	before.loadCoverage(path) // nothing there yet
	vals := []float64{0.11, 0.42, 0.73, 0.94, 0.55}
	for _, v := range vals {
		before.recordCoverage(v)
	}
	before.saveCoverage()

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the ring was not written beside the cursor: %v", err)
	}

	after := &experienceState{}
	after.loadCoverage(path)
	after.mu.Lock()
	got := append([]float64(nil), after.cov...)
	after.mu.Unlock()
	if len(got) != len(vals) {
		t.Fatalf("the restart read back %d coverages, wrote %d", len(got), len(vals))
	}
	for i := range vals {
		if got[i] != vals[i] {
			t.Fatalf("coverage %d came back %.4f, wrote %.4f", i, got[i], vals[i])
		}
	}

	// And the ring is bounded by the window on the way back in, so a window
	// shrunk between runs does not resurrect an old distribution.
	CFG.ExperienceCoverageWindow = 2
	short := &experienceState{}
	short.loadCoverage(path)
	short.mu.Lock()
	n := len(short.cov)
	tail := append([]float64(nil), short.cov...)
	short.mu.Unlock()
	if n != 2 || tail[0] != vals[3] || tail[1] != vals[4] {
		t.Fatalf("a window of 2 read back %v", tail)
	}
	t.Logf("%d coverages written to %s and read back; window 2 keeps the newest two %v",
		len(vals), experienceCoverageFile, tail)
}

package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Repair 6 (MOLEQULALOG2.md, 2026-09-13; audit C, C-OVL-01/02/05, audit A
// P1-3): cross-graze must reach the logits sampling reads under the default-on
// overlay on a warmed organism; the overlay must fade past the untrained
// threshold instead of vanishing in one step; cross-graze must read only what
// is new in the field.

func grazeTestModel(t *testing.T) (*GPT, *EvolvingTokenizer, *CooccurField, []string) {
	t.Helper()
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 2
	CFG.BlockSize = 24
	CFG.HeadTypes = []string{"content", "content"}
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.Trainer = "notorch"
	docs := []string{
		"the river follows the steepest available gradient through the weakest material",
		"heat arrives from outside and the change comes from within",
		"both find the path of least resistance and both shape the landscape",
		"a voice that hears its siblings is pulled toward what they say right now",
	}
	tok := NewEvolvingTokenizer(docs)
	model := NewGPT(tok)
	field := NewCooccurField()
	field.BuildFromCorpus(tok, docs)
	return model, tok, field, docs
}

// scaleToMag rescales a logit vector so that its mean |logit| equals mag.
func scaleToMag(v []float64, mag float64) []float64 {
	out := make([]float64, len(v))
	cur := meanAbsLogit(v)
	if cur == 0 {
		return out
	}
	for i, x := range v {
		out[i] = x * mag / cur
	}
	return out
}

func maxAbsDiff(a, b []float64) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

// The overlay's authority must be continuous in the transformer magnitude:
// one step below the threshold and one step above must give nearly the same
// logits, while well below the overlay is present and well above it is gone.
func TestOverlayFadesInsteadOfSwitchingOff(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	model, tok, field, _ := grazeTestModel(t)
	CFG.CorpusLogitOverlay = true

	if s := overlayFadeProgress(metaTFGateThreshold); s != 0 {
		t.Fatalf("fade progress at the threshold = %v, want 0", s)
	}
	if s := overlayFadeProgress(metaTFGateThreshold + metaFadeWidth/2); math.Abs(s-0.5) > 1e-12 {
		t.Fatalf("fade progress at mid-band = %v, want 0.5", s)
	}
	if s := overlayFadeProgress(metaTFGateThreshold + metaFadeWidth + 1); s != 1 {
		t.Fatalf("fade progress past the band = %v, want 1", s)
	}

	ids := tok.Encode("the river follows")
	keys := make([][]*Vec, model.NLayer)
	values := make([][]*Vec, model.NLayer)
	var raw []float64
	for pos := 0; pos < len(ids); pos++ {
		raw = model.ForwardStep(ids[pos], pos, keys, values).Data
	}
	scratch := NewOverlayScratch(tok.VocabSize)
	scratch.PrepareStatic(model, field)
	step := func(mag float64) ([]float64, float64) {
		r := scaleToMag(raw, mag)
		out, _, _, w := overlayStep(r, ids, field, model, nil, scratch)
		return out, w
	}

	below, wb := step(metaTFGateThreshold - 1e-3)
	above, wa := step(metaTFGateThreshold + 1e-3)
	if wb != 1 || wa >= 1 || wa <= 0 {
		t.Fatalf("weights across the threshold: below=%v above=%v, want 1 and just under 1", wb, wa)
	}
	rawBelow := scaleToMag(raw, metaTFGateThreshold-1e-3)
	stack := maxAbsDiff(below, rawBelow) // what the overlay adds just under the threshold
	if stack < 1 {
		t.Fatalf("overlay adds only %.3f logits below the threshold; the gate has nothing to protect", stack)
	}
	d := maxAbsDiff(below, above)
	if d > 0.05*stack {
		t.Fatalf("overlay steps by %.3f logits across the threshold (stack %.3f): a cliff, not a fade", d, stack)
	}
	t.Logf("overlay stack just under the threshold %.3f logits; step across it %.4f", stack, d)

	gone, wg := step(metaTFGateThreshold + metaFadeWidth + 0.5)
	if wg != 0 {
		t.Fatalf("weight past the band = %v, want 0", wg)
	}
	if d := maxAbsDiff(gone, scaleToMag(raw, metaTFGateThreshold+metaFadeWidth+0.5)); d != 0 {
		t.Fatalf("a warmed organism's logits are altered by %.3e with the overlay faded out", d)
	}
}

// With the overlay flag on and the organism warmed past the fade, the sibling
// boost must land in the logits that are sampled. Before repair 6 it landed in
// logits.Data while sampling read a detached copy, and the colony lost its
// cross-organism channel exactly in the configuration it is deployed with.
func TestCrossGrazeReachesSamplingWhenOverlayOnAndWarmed(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	model, tok, field, docs := grazeTestModel(t)
	CFG.CorpusLogitOverlay = true
	CFG.CrossGraze = true
	CFG.CrossGrazeCoef = 1e6 // the boosted token must win every draw
	CFG.CrossGrazeTopN = 1
	CFG.AntiFieldProb = 0
	CFG.MaxGenTokens = 6
	CFG.MinGenTokens = 1

	// Warm the organism artificially: scale lm_head so mean |logit| is far
	// past the fade band, the regime a warmed adult sits in.
	for _, row := range model.Base["lm_head"].Rows {
		for j := range row.Data {
			row.Data[j] *= 400
		}
	}
	keys := make([][]*Vec, model.NLayer)
	values := make([][]*Vec, model.NLayer)
	probe := tok.Encode("the")
	mag := meanAbsLogit(model.ForwardStep(probe[0], 0, keys, values).Data)
	if mag <= metaTFGateThreshold+metaFadeWidth {
		t.Fatalf("test organism is not warmed past the fade: mean |logit| = %.3f", mag)
	}

	// The byte token for 'z': absent from every test document, so without the
	// sibling boost the organism has no reason to say it.
	enc := tok.Encode("z")
	if len(enc) != 3 {
		t.Fatalf("Encode(\"z\") = %v, want BOS z EOS", enc)
	}
	want := enc[1]
	model.crossField = &CrossField{
		SelfElement: "earth", Siblings: []string{"air"},
		Recent: map[string][]int{"air": {want}}, RecentCap: 64,
		ScanInterval: time.Hour, LastScan: time.Now(), Last: map[string]string{},
	}
	out := GenerateResonant(model, tok, field, "the", docs, true)
	if !strings.Contains(out, "z") {
		t.Fatalf("sibling boost of 1e6 on 'z' did not reach sampling; generated %q — cross-graze is written where nothing reads it", out)
	}
}

// The repetition penalty must never raise a token: halving a negative logit
// moved it toward zero, up the ranking, exactly where the overlay's unigram
// damping had declared it unlikely (audit C, C-OVL-04).
func TestRepetitionPenaltyNeverRewardsRepetition(t *testing.T) {
	logits := []float64{4.0, -3.0, 1.0, -0.5, 2.0}
	// Recent ids 2 4 3 2: tokens 2, 3, 4 are repeated-window members; the
	// context token is 3 and its earlier successor 2 is the blocked bigram.
	MetaweightsRepetitionPenalty(logits, []int{2, 4, 3, 2})
	want := []float64{4.0, -3.0, 0.1, -0.5, 1.0}
	for i := range want {
		if math.Abs(logits[i]-want[i]) > 1e-12 {
			t.Fatalf("logits after penalty = %v, want %v (positive halved and blocked ×0.2; negative left where it is)", logits, want)
		}
	}
	// A negative blocked successor stays negative as it was.
	logits2 := []float64{0, -3.0, 0, -2.0}
	MetaweightsRepetitionPenalty(logits2, []int{3, 1, 3}) // context 1, its successor 3
	if logits2[3] != -2.0 || logits2[1] != -3.0 {
		t.Fatalf("negative logits moved by the penalty: %v", logits2)
	}
}

// The pasture is read by cursor: a refresh ingests only fragments newer than
// the last one seen from each sibling, so the buffer never re-reads the tree.
func TestCrossFieldCursorReadsOnlyNew(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	docs := []string{"one two three four five six seven eight nine ten"}
	tok := NewEvolvingTokenizer(docs)
	root := t.TempDir()
	dir := filepath.Join(root, "dna", "output", "air")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	write := func(name, text string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text+"\n"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("gen_100_1.txt", "one two")
	write("gen_100_2.txt", "three four")
	c := &CrossField{
		SelfElement: "earth", PastureBase: filepath.Join(root, "dna", "output"),
		Siblings: []string{"air"}, Recent: map[string][]int{}, RecentCap: 64,
		ScanInterval: 0, Last: map[string]string{},
	}
	c.MaybeRefresh(tok)
	n1 := len(c.Recent["air"])
	if n1 == 0 || c.Last["air"] != "gen_100_2.txt" {
		t.Fatalf("first refresh: %d tokens, cursor %q; want tokens and cursor at gen_100_2.txt", n1, c.Last["air"])
	}
	c.MaybeRefresh(tok)
	if n := len(c.Recent["air"]); n != n1 {
		t.Fatalf("a refresh with nothing new changed the buffer: %d -> %d", n1, n)
	}
	write("gen_100_10.txt", "five six") // step 10 sorts after step 2 numerically, before it lexically
	c.MaybeRefresh(tok)
	added := len(c.Recent["air"]) - n1
	wantAdded := len(tok.Encode("five six")) - 2 // BOS/EOS stripped
	if added != wantAdded || c.Last["air"] != "gen_100_10.txt" {
		t.Fatalf("third refresh added %d tokens (want %d), cursor %q (want gen_100_10.txt)", added, wantAdded, c.Last["air"])
	}
}

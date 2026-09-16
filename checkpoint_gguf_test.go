package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ── fixtures ──────────────────────────────────────────────────────────────────

// hybridOrganism is the shape the colony actually runs: hybrid heads, so the
// RRPRAM factors and the frozen gate vectors are in the file and in the burst.
// A content-only fixture would never exercise ntBuildGateVectors, which is the
// one place a float32 checkpoint could move a loss that a float64 one does not.
func hybridOrganism(t *testing.T) (*GPT, *EvolvingTokenizer) {
	t.Helper()
	saved := CFG
	t.Cleanup(func() { CFG = saved })
	CFG.NEmbd, CFG.NLayer, CFG.NHead = 32, 2, 2
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content", "hybrid"}
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.RRPRAMRank = 8
	CFG.CheckpointMinInterval = 0
	tok := NewEvolvingTokenizer(ggufCorpus())
	model := NewGPT(tok)
	model.AddDeltaModule(0.5)
	model.globalStep = 137
	model.growthStepOffset = 11
	model.lastWarmupStage = 2
	model.corpusIngestedTotal = 4242
	return model, tok
}

func ggufCorpus() []string {
	docs := make([]string, 0, 64)
	for i := 0; i < 64; i++ {
		docs = append(docs, fmt.Sprintf("the organism speaks of %d and grows a little further into the field", i))
	}
	return docs
}

// ── 1. the round trip ─────────────────────────────────────────────────────────

// A GGUF checkpoint is float32 by design (docs/resonator_design.md §1.2): the
// gate is therefore not that the weights come back unchanged but that they come
// back as exactly float64(float32(w)) — the value the notorch mirror trains on
// either way. Anything else means a wrong offset, a wrong shape or a wrong
// widening, and every one of those is a silent organism.
func TestGGUFCheckpointRoundTrip(t *testing.T) {
	model, tok := hybridOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	gp := ggufPathFor(path)
	gs, err := os.Stat(gp)
	if err != nil {
		t.Fatalf("no GGUF beside the JSON: %v", err)
	}
	js, _ := os.Stat(path)
	t.Logf("JSON %d B, GGUF %d B (%.2fx)", js.Size(), gs.Size(), float64(js.Size())/float64(gs.Size()))

	before := gptConstructions.Load()
	back, backTok, err := LoadCheckpoint(ggufCorpus(), path)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	if got := gptConstructions.Load(); got != before {
		t.Fatalf("the GGUF path built %d throwaway model(s)", got-before)
	}

	if len(back.Base) != len(model.Base) {
		t.Fatalf("%d base matrices came back for %d saved", len(back.Base), len(model.Base))
	}
	compared := 0
	for name, want := range model.Base {
		got, ok := back.Base[name]
		if !ok {
			t.Fatalf("%s did not come back", name)
		}
		if got.Nout != want.Nout || got.Nin != want.Nin {
			t.Fatalf("%s came back %dx%d, was %dx%d", name, got.Nout, got.Nin, want.Nout, want.Nin)
		}
		for i := range want.Rows {
			for j, v := range want.Rows[i].Data {
				if got.Rows[i].Data[j] != float64(float32(v)) {
					t.Fatalf("%s[%d][%d]: %v came back as %v", name, i, j, v, got.Rows[i].Data[j])
				}
			}
			if len(got.Rows[i].Grad) != len(got.Rows[i].Data) {
				t.Fatalf("%s row %d came back with %d grads for %d weights", name, i, len(got.Rows[i].Grad), len(got.Rows[i].Data))
			}
			compared += len(want.Rows[i].Data)
		}
	}
	if len(back.Deltas) != len(model.Deltas) {
		t.Fatalf("%d delta modules came back for %d", len(back.Deltas), len(model.Deltas))
	}
	for i, mod := range model.Deltas {
		if len(back.Deltas[i]) != len(mod) {
			t.Fatalf("module %d came back with %d adapters for %d", i, len(back.Deltas[i]), len(mod))
		}
		for k, da := range mod {
			gd := back.Deltas[i][k]
			if gd == nil || gd.A.Nout != da.A.Nout || gd.B.Nin != da.B.Nin {
				t.Fatalf("module %d adapter %q did not come back at its shape", i, k)
			}
			for r := range da.A.Rows {
				for c, v := range da.A.Rows[r].Data {
					if gd.A.Rows[r].Data[c] != float64(float32(v)) {
						t.Fatalf("delta %d %s.A[%d][%d] came back wrong", i, k, r, c)
					}
				}
			}
		}
	}
	if back.globalStep != model.globalStep || back.growthStepOffset != model.growthStepOffset ||
		back.lastWarmupStage != model.lastWarmupStage || back.corpusIngestedTotal != model.corpusIngestedTotal {
		t.Fatalf("growth state did not survive: step %d/%d offset %d/%d warmup %d/%d ingested %d/%d",
			back.globalStep, model.globalStep, back.growthStepOffset, model.growthStepOffset,
			back.lastWarmupStage, model.lastWarmupStage, back.corpusIngestedTotal, model.corpusIngestedTotal)
	}
	if len(back.ActiveAlpha) != len(model.ActiveAlpha) {
		t.Fatalf("alpha came back with %d entries for %d", len(back.ActiveAlpha), len(model.ActiveAlpha))
	}
	for i, a := range model.ActiveAlpha {
		if back.ActiveAlpha[i] != a { // written as text, so exact float64, not f32
			t.Fatalf("alpha[%d]: %v came back as %v", i, a, back.ActiveAlpha[i])
		}
	}
	if len(back.InitEmbedSnapshot) != len(model.InitEmbedSnapshot) {
		t.Fatalf("the embedding snapshot came back with %d rows for %d",
			len(back.InitEmbedSnapshot), len(model.InitEmbedSnapshot))
	}
	if CFG.TieEmbeddings && back.Base["lm_head"] != back.Base["wte"] {
		t.Fatal("the embedding tie was not re-established after the GGUF load")
	}
	if backTok.VocabSize != tok.VocabSize || len(backTok.Merges) != len(tok.Merges) ||
		backTok.TrainedChars != tok.TrainedChars || backTok.BPEEnabled != tok.BPEEnabled {
		t.Fatalf("the tokenizer did not survive: %d/%d tokens, %d/%d merges",
			backTok.VocabSize, tok.VocabSize, len(backTok.Merges), len(tok.Merges))
	}
	t.Logf("%d parameters round-tripped through the mapping, %d delta modules, %d tokens",
		compared, len(back.Deltas), backTok.VocabSize)
}

// ── 2. loss parity ────────────────────────────────────────────────────────────

// ggufBurstLoss resumes from `path` and runs `steps` notorch steps on the same
// corpus from the same seed. The tape is destroyed first: Chuck's moment slots
// are positional and survive ntTapeClear, so a second arm in the same process
// would otherwise start where the first one left off, and the two arms would
// differ for a reason that has nothing to do with the file.
func ggufBurstLoss(t *testing.T, path string, steps, seed int) (float64, *GPT) {
	t.Helper()
	docs := ggufCorpus()
	model, tok, err := LoadCheckpoint(docs, path)
	if err != nil {
		t.Fatalf("LoadCheckpoint %s: %v", path, err)
	}
	ntTapeDestroy()
	ntSeed(0xC0FFEE)
	rand.Seed(int64(seed))
	avg, n, _ := ntTrainCore(model, tok, docs, steps, model.BlockSize, func(int) float64 { return 0.001 })
	if n != steps {
		t.Fatalf("%s: %d of %d steps counted", path, n, steps)
	}
	return avg, model
}

// The gate of step 2 in docs/resonator_design.md: the same organism resumed from
// the GGUF and from the JSON, same corpus, same seed, must train identically.
// The notorch mirror flattens every weight to float32 before the first forward
// (ntFlattenMatrix), so float32(json_weight) is bit for bit the float32 the GGUF
// stored — which makes this equality and not a tolerance. It goes red if a
// tensor is written in the wrong order, if a row is off by one, or if anything
// the burst reads comes back rounded that the burst reads in float64.
func TestGGUFLossParityWithJSON(t *testing.T) {
	model, tok := hybridOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	gp := ggufPathFor(path)

	const steps, seed = 32, 20260916
	fromGGUF, _ := ggufBurstLoss(t, path, steps, seed)

	// The same file with its binary sibling out of the way is the JSON arm.
	if err := os.Rename(gp, gp+".off"); err != nil {
		t.Fatalf("hiding the GGUF: %v", err)
	}
	fromJSON, _ := ggufBurstLoss(t, path, steps, seed)
	if err := os.Rename(gp+".off", gp); err != nil {
		t.Fatalf("restoring the GGUF: %v", err)
	}

	if fromGGUF != fromJSON {
		t.Fatalf("%d steps: GGUF resume %.17g, JSON resume %.17g, difference %.3g",
			steps, fromGGUF, fromJSON, fromGGUF-fromJSON)
	}
	if fromGGUF == 0 {
		t.Fatal("both arms produced a zero loss — the burst did not run")
	}
	t.Logf("%d steps from either checkpoint: avg loss %.17g", steps, fromGGUF)
}

// ── 3. the refusals ───────────────────────────────────────────────────────────

// A GGUF cut in half is what a process killed mid-write leaves if the rename
// discipline is ever removed, and what a half-copied file is. The organism must
// start from the JSON and say which file it did not use.
func TestGGUFTruncationFallsBackToJSON(t *testing.T) {
	model, tok := hybridOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	gp := ggufPathFor(path)
	blob, err := os.ReadFile(gp)
	if err != nil {
		t.Fatalf("reading the GGUF: %v", err)
	}
	for _, frac := range []float64{0.5, 0.97, 0.02} {
		cut := int(float64(len(blob)) * frac)
		if err := os.WriteFile(gp, blob[:cut], 0644); err != nil {
			t.Fatalf("truncating: %v", err)
		}
		os.Chtimes(gp, time.Now(), time.Now())
		var back *GPT
		var loadErr error
		out := captureStdout(t, func() {
			back, _, loadErr = LoadCheckpoint(ggufCorpus(), path)
		})
		if loadErr != nil {
			t.Fatalf("a GGUF truncated at %d of %d bytes took the organism down: %v", cut, len(blob), loadErr)
		}
		if back == nil || len(back.Base) != len(model.Base) {
			t.Fatalf("truncation at %d did not produce the JSON organism", cut)
		}
		if !strings.Contains(out, "loading the JSON checkpoint") {
			t.Fatalf("truncation at %d was silent; stdout was %q", cut, out)
		}
		t.Logf("cut at %d of %d bytes: %s", cut, len(blob), strings.TrimSpace(out))
	}
}

// A GGUF from another organism is the failure the identity key exists for: same
// shapes, same tensor names, different tokenizer. Nothing about the bytes is
// wrong, so only the identity can catch it.
func TestGGUFForeignTokenizerRefused(t *testing.T) {
	model, tok := hybridOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	// A second organism that has eaten a different corpus and trained a merge on
	// it, written over the first's binary sibling. The byte alphabet is the same
	// 259 tokens, so every tensor in the foreign file has the name and the shape
	// the loader expects and no shape check can see anything wrong: what differs
	// is the mouth — the merges and the chars trained — which is exactly what the
	// identity key is a hash of.
	other := make([]string, 0, 64)
	for i := 0; i < 64; i++ {
		other = append(other, fmt.Sprintf("QUITE ANOTHER MOUTH %d speaking in letters the first never ate", i))
	}
	otherTok := NewEvolvingTokenizer(other)
	otherTok.BPEEnabled = true
	otherTok.Merges = []MergePair{{"t", "h"}}
	otherModel := NewGPT(otherTok)
	gp := ggufPathFor(path)
	if err := writeCheckpointGGUF(gp, otherModel, otherTok); err != nil {
		t.Fatalf("writing the foreign GGUF: %v", err)
	}
	if otherTok.VocabSize != tok.VocabSize {
		t.Fatalf("the fixture is not a pure identity case: %d tokens against %d",
			otherTok.VocabSize, tok.VocabSize)
	}
	if len(otherModel.Base) != len(model.Base) {
		t.Fatalf("the fixture is not a pure identity case: %d matrices against %d",
			len(otherModel.Base), len(model.Base))
	}

	var back *GPT
	var backTok *EvolvingTokenizer
	var loadErr error
	out := captureStdout(t, func() {
		back, backTok, loadErr = LoadCheckpoint(ggufCorpus(), path)
	})
	if loadErr != nil {
		t.Fatalf("the foreign GGUF took the organism down: %v", loadErr)
	}
	// The weights say whose checkpoint came back: the two organisms were built
	// from different random draws, so one embedding value settles it.
	mine := model.Base["wte"].Rows[7].Data[3]
	theirs := otherModel.Base["wte"].Rows[7].Data[3]
	got := back.Base["wte"].Rows[7].Data[3]
	if got == theirs || got != mine {
		t.Fatalf("wte[7][3] came back %v — mine is %v, the foreign organism's is %v", got, mine, theirs)
	}
	if backTok.BPEEnabled != tok.BPEEnabled || len(backTok.Merges) != len(tok.Merges) {
		t.Fatalf("the foreign tokenizer came back: bpe=%v with %d merges",
			backTok.BPEEnabled, len(backTok.Merges))
	}
	if !strings.Contains(out, "identity") {
		t.Fatalf("the refusal was silent or said something else; stdout was %q", out)
	}
	t.Logf("refused: %s", strings.TrimSpace(out))
}

// A GGUF older than the JSON beside it is what a process killed between the two
// renames leaves. The pair is then inconsistent and only the JSON is current.
func TestGGUFOlderThanJSONRefused(t *testing.T) {
	model, tok := hybridOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	gp := ggufPathFor(path)
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(gp, old, old); err != nil {
		t.Fatalf("ageing the GGUF: %v", err)
	}
	before := gptConstructions.Load()
	var loadErr error
	out := captureStdout(t, func() { _, _, loadErr = LoadCheckpoint(ggufCorpus(), path) })
	if loadErr != nil {
		t.Fatalf("load: %v", loadErr)
	}
	if gptConstructions.Load() == before {
		t.Fatal("the JSON path was not taken — the stale GGUF was used")
	}
	if !strings.Contains(out, "older than the JSON") {
		t.Fatalf("the staleness was not reported; stdout was %q", out)
	}
	t.Logf("%s", strings.TrimSpace(out))
}

// ── 4. the throwaway model ────────────────────────────────────────────────────

// LoadCheckpoint's JSON path calls NewGPT and then replaces every matrix it
// built: 4 834 408 rand.NormFloat64 draws and 69 MB of a stage-4 organism's
// 152 MB load, thrown away. The GGUF path must build none of it. The second half
// of this test is what keeps the first half honest — the counter has to move on
// the JSON path, or it could be measuring nothing at all.
func TestGGUFLoadBuildsNoThrowawayModel(t *testing.T) {
	model, tok := hybridOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	gp := ggufPathFor(path)

	before := gptConstructions.Load()
	if _, _, err := LoadCheckpoint(ggufCorpus(), path); err != nil {
		t.Fatalf("GGUF load: %v", err)
	}
	afterGGUF := gptConstructions.Load()
	if afterGGUF != before {
		t.Fatalf("the GGUF path constructed %d model(s) it does not need", afterGGUF-before)
	}

	if err := os.Rename(gp, gp+".off"); err != nil {
		t.Fatalf("hiding the GGUF: %v", err)
	}
	defer os.Rename(gp+".off", gp)
	if _, _, err := LoadCheckpoint(ggufCorpus(), path); err != nil {
		t.Fatalf("JSON load: %v", err)
	}
	if got := gptConstructions.Load(); got == afterGGUF {
		t.Fatal("the counter did not move on the JSON path either — it cannot go red")
	}
	t.Logf("NewGPT calls: %d on the GGUF path, %d on the JSON path",
		afterGGUF-before, gptConstructions.Load()-afterGGUF)
}

// ── 5. the ragged matrix ──────────────────────────────────────────────────────

// The JSON format admits a matrix with a missing row; a GGUF does not. The
// writer refuses rather than inventing a shape, and SaveCheckpoint then leaves
// the JSON standing alone with no stale binary beside it.
func TestGGUFWriterRefusesARaggedMatrix(t *testing.T) {
	model, tok := hybridOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("first save: %v", err)
	}
	gp := ggufPathFor(path)
	if _, err := os.Stat(gp); err != nil {
		t.Fatalf("the first save wrote no GGUF: %v", err)
	}

	model.Base["wpe"].Rows[1] = nil
	if err := writeCheckpointGGUF(filepath.Join(dir, "ragged.gguf"), model, tok); err == nil {
		t.Fatal("a matrix with a missing row was written as a GGUF")
	}
	if _, err := os.Stat(filepath.Join(dir, "ragged.gguf")); err == nil {
		t.Fatal("the refused write left a file behind")
	}
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("the JSON save did not survive a GGUF refusal: %v", err)
	}
	if _, err := os.Stat(gp); err == nil {
		t.Fatal("the stale GGUF was left beside the fresh JSON")
	}
	if _, _, err := LoadCheckpoint(ggufCorpus(), path); err != nil {
		t.Fatalf("the organism cannot load its JSON after the refusal: %v", err)
	}
	t.Log("ragged matrix refused, JSON written, no binary left behind")
}

// ── 6. why the float32 file may be preferred ──────────────────────────────────

// A GGUF checkpoint stores float32 and a JSON one stores float64, so preferring
// the GGUF looks like throwing away half the mantissa of every weight. It is not,
// and this is the measurement that says so: the notorch mirror flattens every
// trained weight to float32 before the first forward (ntFlattenMatrix) and writes
// float32 back into the float64 rows at the end of the burst (ntUnflattenMatrix,
// through pullBack), so after one burst every weight the trainer touches is
// already exactly float64(float32(w)). The float64 tail of a trained organism
// holds nothing. It goes red the day the trainer keeps a float64 weight, which is
// the day this checkpoint has to stop being F32.
func TestTrainedWeightsAreAlreadyFloat32(t *testing.T) {
	model, tok := hybridOrganism(t)
	docs := ggufCorpus()

	// Before the burst the fixture must have a float64 tail, or the gate proves
	// nothing: NewMatrixParam draws from rand.NormFloat64.
	tail := 0
	for _, row := range model.Base["wte"].Rows {
		for _, v := range row.Data {
			if v != float64(float32(v)) {
				tail++
			}
		}
	}
	if tail == 0 {
		t.Fatal("the fixture has no float64 tail to lose — the gate cannot go red")
	}

	ntTapeDestroy()
	rand.Seed(7)
	if _, n, _ := ntTrainCore(model, tok, docs, 4, model.BlockSize, func(int) float64 { return 0.001 }); n != 4 {
		t.Fatalf("%d of 4 steps ran", n)
	}

	checked := 0
	for _, p := range ntContentParams(model) {
		for i, row := range p.mp.Rows {
			for j, v := range row.Data {
				if v != float64(float32(v)) {
					t.Fatalf("%s[%d][%d] = %.17g still carries a float64 tail after a burst", p.name, i, j, v)
				}
				checked++
			}
		}
	}
	t.Logf("%d float64 values before the burst were not float32-exact; after it, %d trained weights are",
		tail, checked)
}

// ── 7. the measurement ────────────────────────────────────────────────────────

// TestGGUFCheckpointMeasure is the table of step 2: file size, save cost and load
// cost of a real organism's checkpoint in both formats, and the loss both
// resumes train to. One phase per process, because VmHWM is a high-water mark for
// the life of a process and two loads in one process measure the first one twice.
//
//	MOLEQULA_GGUF_MEASURE=<copy of a checkpoint.json> \
//	MOLEQULA_GGUF_PHASE=save|load-json|load-gguf|parity \
//	go test -run TestGGUFCheckpointMeasure -v .
func TestGGUFCheckpointMeasure(t *testing.T) {
	src := os.Getenv("MOLEQULA_GGUF_MEASURE")
	phase := os.Getenv("MOLEQULA_GGUF_PHASE")
	if src == "" || phase == "" {
		t.Skip("set MOLEQULA_GGUF_MEASURE=<copy of a checkpoint> and MOLEQULA_GGUF_PHASE to measure")
	}
	if free := memAvailableMB(); free < 2000 {
		t.Skipf("MemAvailable %d MB < 2000 MB — not measuring on a machine this tight", free)
	}
	if _, err := applyOomScoreAdj(oomScoreAdjPath, 500); err != nil {
		t.Logf("could not raise oom_score_adj: %v", err)
	}
	saved := CFG
	t.Cleanup(func() { CFG = saved })
	CFG.CheckpointMinInterval = 0
	gp := ggufPathFor(src)
	docs := []string{"the organism speaks", "and lo, it grows"}

	hwm := func() int64 { return procFieldMB("/proc/self/status", "VmHWM") }
	rss := func() int64 { return procFieldMB("/proc/self/status", "VmRSS") }
	resetHWM := func() {
		if f, err := os.OpenFile("/proc/self/clear_refs", os.O_WRONLY, 0); err == nil {
			f.WriteString("5\n")
			f.Close()
		}
	}
	size := func(p string) int64 {
		fi, err := os.Stat(p)
		if err != nil {
			return -1
		}
		return fi.Size()
	}

	switch phase {
	case "save":
		os.Remove(gp)
		model, tok, err := LoadCheckpoint(docs, src)
		if err != nil {
			t.Fatalf("LoadCheckpoint: %v", err)
		}
		nParams := 0
		for _, m := range model.Base {
			nParams += m.Nout * m.Nin
		}
		t.Logf("stage %d, %d parameters, vocab %d, %d delta modules",
			model.CurrentGrowthStage(), nParams, len(tok.Tokens), len(model.Deltas))

		outJSON := filepath.Join(t.TempDir(), "probe_ckpt.json")
		runtime.GC()
		resetHWM()
		r0, h0 := rss(), hwm()
		t0 := time.Now()
		if err := saveJSONOnly(outJSON, model, tok); err != nil {
			t.Fatalf("JSON save: %v", err)
		}
		dJSON := time.Since(t0)
		t.Logf("JSON   save: %6.0f ms, %10d B, RSS %d→%d MB, HWM %d→%d MB (+%d)",
			float64(dJSON.Microseconds())/1000, size(outJSON), r0, rss(), h0, hwm(), hwm()-h0)

		outGGUF := ggufPathFor(outJSON)
		runtime.GC()
		resetHWM()
		r1, h1 := rss(), hwm()
		t1 := time.Now()
		if err := writeCheckpointGGUF(outGGUF, model, tok); err != nil {
			t.Fatalf("GGUF save: %v", err)
		}
		dGGUF := time.Since(t1)
		t.Logf("GGUF   save: %6.0f ms, %10d B, RSS %d→%d MB, HWM %d→%d MB (+%d)",
			float64(dGGUF.Microseconds())/1000, size(outGGUF), r1, rss(), h1, hwm(), hwm()-h1)
		t.Logf("ratio: %.2fx smaller, %.2fx faster", float64(size(outJSON))/float64(size(outGGUF)),
			float64(dJSON)/float64(dGGUF))

		// And the pair beside the source, for the load phases.
		if err := SaveCheckpoint(model, tok, src); err != nil {
			t.Fatalf("pair save: %v", err)
		}
		t.Logf("pair on disk: %s %d B, %s %d B", src, size(src), gp, size(gp))

	case "load-json", "load-gguf":
		if phase == "load-json" {
			if err := os.Rename(gp, gp+".off"); err != nil {
				t.Fatalf("hiding the GGUF: %v", err)
			}
			defer os.Rename(gp+".off", gp)
		} else if _, err := os.Stat(gp); err != nil {
			t.Fatalf("no GGUF to load — run the save phase first: %v", err)
		}
		runtime.GC()
		resetHWM()
		r0, h0 := rss(), hwm()
		before := gptConstructions.Load()
		t0 := time.Now()
		model, tok, err := LoadCheckpoint(docs, src)
		if err != nil {
			t.Fatalf("LoadCheckpoint: %v", err)
		}
		d := time.Since(t0)
		nParams := 0
		for _, m := range model.Base {
			nParams += m.Nout * m.Nin
		}
		t.Logf("%s load: %6.0f ms, RSS %d→%d MB (+%d), HWM %d→%d MB (+%d), NewGPT×%d, %d params, vocab %d",
			phase, float64(d.Microseconds())/1000, r0, rss(), rss()-r0, h0, hwm(), hwm()-h0,
			gptConstructions.Load()-before, nParams, len(tok.Tokens))

	case "parity":
		const steps = 32
		out := make([]float64, 0, 2)
		for _, arm := range []string{"gguf", "json"} {
			if arm == "json" {
				if err := os.Rename(gp, gp+".off"); err != nil {
					t.Fatalf("hiding the GGUF: %v", err)
				}
			}
			model, tok, err := LoadCheckpoint(docs, src)
			if err != nil {
				t.Fatalf("%s load: %v", arm, err)
			}
			corpus := probeCorpus()
			ntTapeDestroy()
			ntSeed(0xC0FFEE)
			rand.Seed(20260916)
			avg, n, ms := ntTrainCore(model, tok, corpus, steps, model.BlockSize, func(int) float64 { return 0.001 })
			t.Logf("%-4s resume: %d steps, avg loss %.17g, %.0f ms, HWM %d MB", arm, n, avg, ms, hwm())
			out = append(out, avg)
			if arm == "json" {
				if err := os.Rename(gp+".off", gp); err != nil {
					t.Fatalf("restoring the GGUF: %v", err)
				}
			}
			runtime.GC()
		}
		if out[0] != out[1] {
			t.Fatalf("parity: GGUF %.17g, JSON %.17g, difference %.3g", out[0], out[1], out[0]-out[1])
		}
		t.Logf("parity over %d steps: %.17g from either checkpoint", steps, out[0])

	default:
		t.Fatalf("unknown MOLEQULA_GGUF_PHASE %q", phase)
	}
}

// saveJSONOnly writes just the JSON half, so the two halves of a save can be
// timed and measured apart from each other.
func saveJSONOnly(path string, model *GPT, tok *EvolvingTokenizer) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	err = writeCheckpointJSON(bw, model, tok)
	if err == nil {
		err = bw.Flush()
	}
	f.Close()
	if err != nil {
		os.Remove(path)
	}
	return err
}

// probeCorpus makes a corpus long enough that the burst's window sampling has
// somewhere to land; every character of it is in any organism's byte alphabet.
func probeCorpus() []string {
	docs := make([]string, 0, 64)
	for i := 0; i < 64; i++ {
		docs = append(docs, fmt.Sprintf(
			"the organism %d speaks into the field and the field speaks back, and what returns is not what went in", i))
	}
	return docs
}

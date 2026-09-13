package main

import (
	"bufio"
	"bytes"
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Repair 9 (MOLEQULALOG2.md 2026-09-13): the four organisms all reached stage 4
// within two minutes, each wrote a 110 MB checkpoint, and Android's lmkd took
// Termux and ~20 apps instead of the colony. These are the gates that stand in
// the way of that happening twice, each written so it goes red when the thing it
// guards is removed.

// ── 1. the byte gate ──────────────────────────────────────────────────────────

func TestGrowthGateDecision(t *testing.T) {
	// floor 0 disables the gate completely, whatever the factor.
	if open, need := growthGateDecision(1, 900, 0, 300); !open || need != 0 {
		t.Fatalf("floor 0 must disable the gate: open=%v need=%d", open, need)
	}
	// A stage step is charged as factorPct% of the organism's own peak, on top
	// of the floor: 240 MB peak at 300% = 720, +256 floor = 976 MB.
	if open, need := growthGateDecision(2000, 240, 256, 300); !open || need != 976 {
		t.Fatalf("2000 MB free against 240*3+256: open=%v need=%d, want open need=976", open, need)
	}
	if open, need := growthGateDecision(740, 240, 256, 300); open || need != 976 {
		t.Fatalf("740 MB free (the 2026-09-13 low-water mark) must defer: open=%v need=%d", open, need)
	}
	if open, _ := growthGateDecision(976, 240, 256, 300); !open {
		t.Fatal("free == need must be open (>=)")
	}
	// The factor is what makes this gate different from the mitosis one: at 100%
	// the same organism and the same machine would have been let through.
	if open, _ := growthGateDecision(740, 240, 256, 100); !open {
		t.Fatal("at factor 100% the 2026-09-13 state would pass — the factor is the gate")
	}
	// A negative factor charges nothing but the floor, it never credits memory.
	if _, need := growthGateDecision(1000, 240, 256, -50); need != 256 {
		t.Fatalf("negative factor must charge the floor only, got need=%d", need)
	}
	// Unknown free memory is not a reason to stop growing.
	if open, _ := growthGateDecision(0, 240, 256, 300); !open {
		t.Fatal("unknown MemAvailable must leave the gate open")
	}
}

// The gate on the live machine, with the configured knobs.
func TestGrowthGateOnThisHost(t *testing.T) {
	if _, err := os.Stat("/proc/meminfo"); err != nil {
		t.Skip("no /proc/meminfo on this host")
	}
	free := memAvailableMB()
	peak := ownPeakRSSMB()
	if free <= 0 || peak <= 0 {
		t.Fatalf("free=%d peak=%d, both must be > 0 on a Linux host", free, peak)
	}
	open, gotFree, need := growthMemGateOpen(256, 300)
	if gotFree != free && gotFree <= 0 {
		t.Fatalf("growthMemGateOpen read free=%d, /proc says %d", gotFree, free)
	}
	if want := 256 + peak*3; need < want-peak || need > want+peak {
		t.Fatalf("need=%d MB is not ~256 + 3*%d MB", need, peak)
	}
	// A floor larger than the machine must close it, factor or no factor.
	if closed, _, _ := growthMemGateOpen(free*4, 0); closed {
		t.Fatalf("a %d MB floor against %d MB free must close the gate", free*4, free)
	}
	t.Logf("host: free=%d MB, own peak=%d MB, growth needs %d MB, open=%v", free, peak, need, open)
}

// ── 1b. the two declared ceilings ─────────────────────────────────────────────

// The byte gate defers what the machine cannot afford this minute; MaxGrowthStage
// is the permanent ceiling. A corpus that would carry the organism to adult must
// not move it past the stage the launcher declared.
func TestMaxGrowthStageRefusesTheNextStage(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
		{50000, 64, 2, 4},
		{200000, 128, 4, 4},
	}
	CFG.NEmbd, CFG.NLayer, CFG.NHead = 16, 1, 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.FreezeAfterGrowthSteps = 100

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)
	model.corpusIngestedTotal = 999999 // enough for the last stage in the table

	// Uncapped: the organism grows, one stage at a time.
	CFG.MaxGrowthStage = 3
	if maxGrowthStage() != 3 {
		t.Fatalf("maxGrowthStage = %d, want 3", maxGrowthStage())
	}
	if !model.MaybeGrowArchitecture() {
		t.Fatal("fixture broken: an organism below the ceiling with corpus to spare must grow")
	}
	model.growthFreezeRemaining = 0
	if model.CurrentGrowthStage() != 1 {
		t.Fatalf("stage %d after one growth, want 1", model.CurrentGrowthStage())
	}

	// Capped at the stage it is standing on: no growth, however large the corpus.
	CFG.MaxGrowthStage = 1
	if model.GrowthWanted() {
		t.Fatal("GrowthWanted must be false at the ceiling")
	}
	if model.MaybeGrowArchitecture() {
		t.Fatalf("the organism grew past the ceiling to stage %d", model.CurrentGrowthStage())
	}
	if model.CurrentGrowthStage() != 1 || model.NEmbd != 32 {
		t.Fatalf("the refused growth still changed the model: stage %d embd %d",
			model.CurrentGrowthStage(), model.NEmbd)
	}
	if !model.growthCapLogged {
		t.Fatal("the ceiling must be announced once")
	}

	// Raising the ceiling releases it again — the cap is a ceiling, not a state.
	CFG.MaxGrowthStage = 3
	if !model.MaybeGrowArchitecture() {
		t.Fatal("a raised ceiling must let the organism grow again")
	}
	if model.CurrentGrowthStage() != 2 {
		t.Fatalf("stage %d, want 2", model.CurrentGrowthStage())
	}

	// Out-of-range values mean the last stage, never a panic or a frozen embryo.
	CFG.MaxGrowthStage = 99
	if maxGrowthStage() != 3 {
		t.Fatalf("a ceiling past the table must clamp to the last stage, got %d", maxGrowthStage())
	}
	CFG.MaxGrowthStage = -1
	if maxGrowthStage() != 3 {
		t.Fatalf("a negative ceiling must clamp to the last stage, got %d", maxGrowthStage())
	}
}

// --max-organisms 1 means one organism: the colony's own admit refuses the second.
func TestMaxOrganismsOneRefusesTheSecondAdmit(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE organisms(id TEXT PRIMARY KEY, status TEXT, last_heartbeat REAL)`); err != nil {
		t.Skipf("sqlite exec: %v", err)
	}
	db.Exec(`CREATE TABLE mitosis_lock(organism_id TEXT PRIMARY KEY, acquired_at REAL)`)
	now := float64(time.Now().UnixMilli()) / 1000.0
	db.Exec(`INSERT INTO organisms VALUES(?,?,?)`, "earth", "alive", now)
	earth := &SwarmRegistry{OrganismID: "earth", MeshDB: db}

	if earth.AcquireMitosisSlot(1) {
		t.Fatal("one live organism at --max-organisms 1 must refuse the divide")
	}
	if !earth.AcquireMitosisSlot(4) {
		t.Fatal("fixture broken: the same organism under a cap of 4 must be admitted")
	}
	earth.ReleaseMitosisLock()

	// And the flag really reaches CFG, which is what the gate reads.
	savedArgs, savedCfg := os.Args, CFG
	defer func() { os.Args, CFG = savedArgs, savedCfg }()
	os.Args = []string{"molequla", "--max-organisms", "4", "--max-growth-stage", "4"}
	parseCLIArgs()
	if CFG.MaxOrganisms != 4 {
		t.Fatalf("--max-organisms 4 left CFG.MaxOrganisms at %d", CFG.MaxOrganisms)
	}
	if CFG.MaxGrowthStage != 4 {
		t.Fatalf("--max-growth-stage 4 left CFG.MaxGrowthStage at %d", CFG.MaxGrowthStage)
	}
	os.Args = []string{"molequla", "--max-organisms", "-3"}
	CFG.MaxOrganisms = 16
	parseCLIArgs()
	if CFG.MaxOrganisms != 16 {
		t.Fatalf("a negative cap must be ignored, got %d", CFG.MaxOrganisms)
	}
}

// ── 2. the colony growth lock ─────────────────────────────────────────────────

func growthLockMesh(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	if _, err := db.Exec(`CREATE TABLE growth_lock(organism_id TEXT PRIMARY KEY, acquired_at REAL)`); err != nil {
		db.Close()
		t.Skipf("sqlite exec: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// Two registries on one mesh cannot both hold the growth lock: that is the whole
// point — four organisms that grew together peaked together.
func TestGrowthLockExcludesSibling(t *testing.T) {
	db := growthLockMesh(t)
	earth := &SwarmRegistry{OrganismID: "earth", MeshDB: db}
	water := &SwarmRegistry{OrganismID: "water", MeshDB: db}

	if !earth.AcquireGrowthLock() {
		t.Fatal("an idle colony must admit the first grower")
	}
	if water.AcquireGrowthLock() {
		t.Fatal("a sibling must be refused while another organism is growing")
	}
	// Re-entrant for the holder: a second growth on the same tick is not a
	// deadlock against itself.
	if !earth.AcquireGrowthLock() {
		t.Fatal("the holder must be able to re-take its own lock")
	}
	earth.ReleaseGrowthLock()
	if !water.AcquireGrowthLock() {
		t.Fatal("the lock must pass to the next organism after release")
	}
	water.ReleaseGrowthLock()

	// A refresh keeps the holder's stamp fresh without stealing anyone's lock.
	if !earth.AcquireGrowthLock() {
		t.Fatal("re-acquire after release failed")
	}
	var before float64
	db.QueryRow("SELECT acquired_at FROM growth_lock WHERE organism_id='earth'").Scan(&before)
	db.Exec("UPDATE growth_lock SET acquired_at=? WHERE organism_id='earth'", before-1000)
	water.RefreshGrowthLock() // not the holder — must not touch earth's row
	var afterSibling float64
	db.QueryRow("SELECT acquired_at FROM growth_lock WHERE organism_id='earth'").Scan(&afterSibling)
	if afterSibling != before-1000 {
		t.Fatal("a non-holder's refresh must not touch the holder's row")
	}
	earth.RefreshGrowthLock()
	var afterHolder float64
	db.QueryRow("SELECT acquired_at FROM growth_lock WHERE organism_id='earth'").Scan(&afterHolder)
	if afterHolder <= before-1000 {
		t.Fatalf("the holder's refresh must re-stamp the row: %f", afterHolder)
	}
	earth.ReleaseGrowthLock()

	// A lock older than the TTL belongs to a dead organism and must not block
	// the colony forever.
	db.Exec("INSERT OR REPLACE INTO growth_lock(organism_id, acquired_at) VALUES('fire', ?)",
		float64(0))
	if !earth.AcquireGrowthLock() {
		t.Fatal("a stale lock (age > TTL) must not block a live grower")
	}
	earth.ReleaseGrowthLock()

	// No mesh = solo organism = always admitted.
	solo := &SwarmRegistry{OrganismID: "solo"}
	if !solo.AcquireGrowthLock() {
		t.Fatal("a solo organism with no mesh must always be admitted")
	}
	solo.ReleaseGrowthLock()
}

// ── 3. oom_score_adj ──────────────────────────────────────────────────────────

func TestApplyOomScoreAdj(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("oom_score_adj is a Linux knob")
	}
	path := filepath.Join(t.TempDir(), "oom_score_adj")
	os.WriteFile(path, []byte("-1000\n"), 0644)

	got, err := applyOomScoreAdj(path, 300)
	if err != nil {
		t.Fatalf("applyOomScoreAdj: %v", err)
	}
	if got != "300" {
		t.Fatalf("read back %q, want \"300\" — the organism must outrank nothing", got)
	}
	data, _ := os.ReadFile(path)
	if strings.TrimSpace(string(data)) != "300" {
		t.Fatalf("file holds %q, want 300", strings.TrimSpace(string(data)))
	}

	// 0 means leave it alone — an inherited value stays inherited.
	os.WriteFile(path, []byte("-1000\n"), 0644)
	if got, err := applyOomScoreAdj(path, 0); err != nil || got != "" {
		t.Fatalf("0 must be a no-op, got %q err=%v", got, err)
	}
	data, _ = os.ReadFile(path)
	if strings.TrimSpace(string(data)) != "-1000" {
		t.Fatalf("0 rewrote the file to %q", strings.TrimSpace(string(data)))
	}

	// An unwritable knob is reported, not swallowed.
	if _, err := applyOomScoreAdj(filepath.Join(path, "nope", "oom"), 300); err == nil {
		t.Fatal("a write failure must return an error")
	}
}

// The live knob: this process really does become killable.
func TestApplyOomScoreAdjOnThisProcess(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("oom_score_adj is a Linux knob")
	}
	orig, err := os.ReadFile(oomScoreAdjPath)
	if err != nil {
		t.Skipf("no %s: %v", oomScoreAdjPath, err)
	}
	t.Cleanup(func() { os.WriteFile(oomScoreAdjPath, orig, 0644) })
	got, err := applyOomScoreAdj(oomScoreAdjPath, 300)
	if err != nil {
		t.Skipf("cannot write %s here: %v", oomScoreAdjPath, err)
	}
	if got != "300" {
		t.Fatalf("%s reads %q after the write, want 300", oomScoreAdjPath, got)
	}
	t.Logf("%s: %s → %s", oomScoreAdjPath, strings.TrimSpace(string(orig)), got)
}

// ── 4. the streamed checkpoint ────────────────────────────────────────────────

// checkpointViaEncoder is the old save path, kept here as the oracle: build one
// CheckpointData holding a copy of every weight, hand it to json.Encoder. The
// streamed writer must produce these bytes exactly.
func checkpointViaEncoder(model *GPT, tok *EvolvingTokenizer) ([]byte, error) {
	merges := make([][]string, len(tok.Merges))
	for i, m := range tok.Merges {
		merges[i] = []string{m.A, m.B}
	}
	cfgJSON, _ := json.Marshal(CFG)
	base := make(map[string][][]float64)
	for k, v := range model.Base {
		base[k] = serializeMatrixParam(v)
	}
	deltas := make([]map[string]DeltaJSON, len(model.Deltas))
	for i, mod := range model.Deltas {
		dm := make(map[string]DeltaJSON)
		for name, da := range mod {
			dm[name] = DeltaJSON{A: serializeMatrixParam(da.A), B: serializeMatrixParam(da.B)}
		}
		deltas[i] = dm
	}
	ckpt := CheckpointData{
		Cfg: cfgJSON,
		Tokenizer: TokenizerJSON{
			Tokens:       tok.Tokens,
			BPEEnabled:   tok.BPEEnabled,
			Merges:       merges,
			TrainedChars: tok.TrainedChars,
		},
		Base:                base,
		Alpha:               model.ActiveAlpha,
		Deltas:              deltas,
		InitEmbedSnapshot:   model.InitEmbedSnapshot,
		GlobalStep:          model.globalStep,
		GrowthStepOffset:    model.growthStepOffset,
		LastWarmupStage:     intPtr(model.lastWarmupStage),
		CorpusIngestedTotal: model.corpusIngestedTotal,
	}
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(ckpt)
	return buf.Bytes(), err
}

// smallOrganism builds a grown model with delta modules and a snapshot — every
// field the checkpoint carries, at a size a test can afford.
func smallOrganism(t *testing.T) (*GPT, *EvolvingTokenizer) {
	t.Helper()
	saved := CFG
	t.Cleanup(func() { CFG = saved })
	CFG.NEmbd, CFG.NLayer, CFG.NHead = 16, 1, 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	tok := NewEvolvingTokenizer([]string{"the organism speaks", "and lo, it grows"})
	model := NewGPT(tok)
	model.AddDeltaModule(1.0)
	model.AddDeltaModule(0.5)
	model.corpusIngestedTotal = 999999
	model.MaybeGrowArchitecture()
	model.globalStep = 41
	model.growthStepOffset = 7
	model.lastWarmupStage = 1
	return model, tok
}

// The streamed writer and the old encoder must agree byte for byte: the saving
// is in the allocations, never in the format. Break writeMatrixParamJSON or the
// field order and this goes red immediately.
func TestCheckpointStreamMatchesEncoder(t *testing.T) {
	model, tok := smallOrganism(t)

	want, err := checkpointViaEncoder(model, tok)
	if err != nil {
		t.Fatalf("oracle encode: %v", err)
	}
	var got bytes.Buffer
	bw := bufio.NewWriter(&got)
	if err := writeCheckpointJSON(bw, model, tok); err != nil {
		t.Fatalf("writeCheckpointJSON: %v", err)
	}
	bw.Flush()
	if !bytes.Equal(want, got.Bytes()) {
		t.Fatalf("streamed checkpoint differs from the encoder's (%d vs %d bytes)\nwant tail: %s\ngot  tail: %s",
			len(want), got.Len(), tailOf(want), tailOf(got.Bytes()))
	}
	if len(want) < 1000 {
		t.Fatalf("the fixture is too small to prove anything: %d bytes", len(want))
	}
	t.Logf("identical, %d bytes", len(want))
}

func tailOf(b []byte) string {
	if len(b) > 160 {
		return string(b[len(b)-160:])
	}
	return string(b)
}

// A round trip through the streamed writer restores the same organism.
func TestCheckpointStreamRoundTrip(t *testing.T) {
	model, tok := smallOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	savedPath, savedInterval := CFG.CkptPath, CFG.CheckpointMinInterval
	CFG.CkptPath, CFG.CheckpointMinInterval = path, 0
	defer func() { CFG.CkptPath, CFG.CheckpointMinInterval = savedPath, savedInterval }()

	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}
	back, tokBack, err := LoadCheckpoint([]string{"the organism speaks"}, path)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	if back.NEmbd != model.NEmbd || back.NLayer != model.NLayer || back.NHead != model.NHead {
		t.Fatalf("dims %d/%d/%d != %d/%d/%d", back.NEmbd, back.NLayer, back.NHead,
			model.NEmbd, model.NLayer, model.NHead)
	}
	if len(back.Deltas) != len(model.Deltas) {
		t.Fatalf("%d delta modules, want %d", len(back.Deltas), len(model.Deltas))
	}
	if back.corpusIngestedTotal != model.corpusIngestedTotal || back.globalStep != model.globalStep {
		t.Fatalf("clock lost: ingested=%d step=%d", back.corpusIngestedTotal, back.globalStep)
	}
	if len(tokBack.Tokens) != len(tok.Tokens) {
		t.Fatalf("vocab %d, want %d", len(tokBack.Tokens), len(tok.Tokens))
	}
	for k, m := range model.Base {
		b, ok := back.Base[k]
		if !ok {
			t.Fatalf("matrix %s missing after reload", k)
		}
		if b.Nout != m.Nout || b.Nin != m.Nin {
			t.Fatalf("%s is %dx%d, want %dx%d", k, b.Nout, b.Nin, m.Nout, m.Nin)
		}
		if m.Nout > 0 && b.Rows[0].Data[0] != m.Rows[0].Data[0] {
			t.Fatalf("%s[0][0] = %v, want %v", k, b.Rows[0].Data[0], m.Rows[0].Data[0])
		}
	}
}

// ── 5. the checkpoint on the way out ──────────────────────────────────────────

// A session that ends by SIGTERM must leave its work on disk. The trap is the
// debouncer: the periodic path is throttled to one write per CheckpointMinInterval
// and a shutdown inside that window writes nothing at all — which is what happened
// on 2026-09-13, when every checkpoint kept its growth-time mtime through a stop.
func TestSaveOnShutdownBeatsTheDebouncer(t *testing.T) {
	model, tok := smallOrganism(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "molequla_ckpt.json")
	savedPath, savedInterval := CFG.CkptPath, CFG.CheckpointMinInterval
	CFG.CkptPath, CFG.CheckpointMinInterval = path, 30.0
	defer func() { CFG.CkptPath, CFG.CheckpointMinInterval = savedPath, savedInterval }()

	// A periodic save lands and arms the debouncer.
	if err := SaveCheckpoint(model, tok, ""); err != nil {
		t.Fatalf("first periodic save: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("no checkpoint after the periodic save: %v", err)
	}

	// Training moves on; a second periodic save inside the window is dropped.
	model.globalStep = 4242
	model.Base["wte"].Rows[0].Data[0] = 0.4242
	if err := SaveCheckpoint(model, tok, ""); err != nil {
		t.Fatalf("throttled save returned an error: %v", err)
	}
	if mid, _ := os.ReadFile(path); !bytes.Equal(before, mid) {
		t.Fatal("fixture broken: the debounced save was expected to be dropped")
	}

	// The shutdown save must go through anyway.
	saveOnShutdown("evolution", model, tok, "signal")
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading the checkpoint after shutdown: %v", err)
	}
	if bytes.Equal(before, after) {
		t.Fatal("the shutdown save wrote nothing — a SIGTERM'd session loses its work")
	}
	back, _, err := LoadCheckpoint([]string{"the organism speaks"}, path)
	if err != nil {
		t.Fatalf("the shutdown checkpoint does not load: %v", err)
	}
	if back.globalStep != 4242 {
		t.Fatalf("the shutdown checkpoint holds step %d, want 4242", back.globalStep)
	}
	if back.Base["wte"].Rows[0].Data[0] != 0.4242 {
		t.Fatalf("the shutdown checkpoint holds stale weights: %v", back.Base["wte"].Rows[0].Data[0])
	}
	t.Logf("shutdown save: %d → %d bytes, step 41 → %d", len(before), len(after), back.globalStep)
}

// A shutdown must reach the mutex. ntWarmupTrain holds model.mu for its whole
// phase — 1600 steps at stage 4, half an hour — so without a per-step abort the
// exit path blocks on the lock and the process is killed before it saves.
func TestTrainAbortStopsTheStepLoop(t *testing.T) {
	model, tok := smallOrganism(t)
	docs := []string{"the organism speaks", "and lo, it grows", "nonames"}
	trainAbort.Store(false)
	defer trainAbort.Store(false)

	// Baseline: steps really do run when nothing is shutting down.
	before := model.globalStep
	ntWarmupTrain(model, tok, docs, 3, 8)
	if model.globalStep == before {
		t.Fatalf("fixture broken: 3 warmup steps moved globalStep not at all (%d)", before)
	}

	trainAbort.Store(true)
	defer trainAbort.Store(false)
	if !trainAborting() {
		t.Fatal("trainAborting must report the raised flag")
	}
	atAbort := model.globalStep
	start := time.Now()
	ntWarmupTrain(model, tok, docs, 100000, 8)
	took := time.Since(start)
	if model.globalStep != atAbort {
		t.Fatalf("100000 steps requested under abort ran %d of them", model.globalStep-atAbort)
	}
	if took > 5*time.Second {
		t.Fatalf("an aborted warmup took %s — the exit path would still block on model.mu", took)
	}
	// And the mutex is free the moment it returns, which is the whole point.
	if !model.mu.TryLock() {
		t.Fatal("model.mu still held after an aborted warmup returned")
	}
	model.mu.Unlock()
	t.Logf("aborted warmup of 100000 steps returned in %s with the lock free", took)
}

// ── 6. where the stage-4 peak comes from ──────────────────────────────────────

// The measurement runs only against a read-only copy of a real checkpoint named
// by MOLEQULA_CKPT_MEASURE, and only when the phone has room for it. It writes
// 500 to its own oom_score_adj first, so if the estimate is wrong the test dies
// and Termux does not.
func TestCheckpointMemoryProfile(t *testing.T) {
	src := os.Getenv("MOLEQULA_CKPT_MEASURE")
	if src == "" {
		t.Skip("set MOLEQULA_CKPT_MEASURE=<copy of a checkpoint> to measure the load/save peak")
	}
	const needMB = 2000
	if free := memAvailableMB(); free < needMB {
		t.Skipf("MemAvailable %d MB < %d MB — not measuring on a machine this tight", free, needMB)
	}
	if _, err := applyOomScoreAdj(oomScoreAdjPath, 500); err != nil {
		t.Logf("could not raise oom_score_adj: %v", err)
	}
	fi, err := os.Stat(src)
	if err != nil {
		t.Fatalf("stat %s: %v", src, err)
	}
	t.Logf("checkpoint %s: %.1f MB on disk", src, float64(fi.Size())/(1<<20))

	rss := func() (int64, int64) {
		return procFieldMB("/proc/self/status", "VmRSS"), procFieldMB("/proc/self/status", "VmHWM")
	}
	resetHWM := func() {
		// Clearing the high-water mark makes the next measurement about this
		// step only, not about everything the test binary ever did.
		if f, err := os.OpenFile("/proc/self/clear_refs", os.O_WRONLY, 0); err == nil {
			f.WriteString("5\n")
			f.Close()
		}
	}

	resetHWM()
	rss0, hwm0 := rss()
	model, tok, err := LoadCheckpoint([]string{"hello"}, src)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	rss1, hwm1 := rss()
	nParams := 0
	for _, m := range model.Base {
		nParams += m.Nout * m.Nin
	}
	t.Logf("LoadCheckpoint: RSS %d → %d MB (+%d), HWM %d → %d MB (+%d); stage=%d params=%d vocab=%d",
		rss0, rss1, rss1-rss0, hwm0, hwm1, hwm1-hwm0, model.CurrentGrowthStage(), nParams, len(tok.Tokens))

	out := filepath.Join(t.TempDir(), "out_ckpt.json")
	savedInterval := CFG.CheckpointMinInterval
	CFG.CheckpointMinInterval = 0
	defer func() { CFG.CheckpointMinInterval = savedInterval }()

	runtime.GC()
	resetHWM()
	rss2, hwm2 := rss()
	if err := SaveCheckpoint(model, tok, out); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}
	rss3, hwm3 := rss()
	so, _ := os.Stat(out)
	t.Logf("SaveCheckpoint (streamed): RSS %d → %d MB (+%d), HWM %d → %d MB (+%d); wrote %.1f MB",
		rss2, rss3, rss3-rss2, hwm2, hwm3, hwm3-hwm2, float64(so.Size())/(1<<20))

	// And the old path, for the difference this repair actually bought.
	runtime.GC()
	resetHWM()
	rss4, hwm4 := rss()
	blob, err := checkpointViaEncoder(model, tok)
	if err != nil {
		t.Fatalf("oracle encode: %v", err)
	}
	rss5, hwm5 := rss()
	t.Logf("SaveCheckpoint (old, CheckpointData + json.Encoder): RSS %d → %d MB (+%d), HWM %d → %d MB (+%d); %.1f MB in memory",
		rss4, rss5, rss5-rss4, hwm4, hwm5, hwm5-hwm4, float64(len(blob))/(1<<20))
	if int64(len(blob)) != so.Size() {
		t.Fatalf("the two paths disagree on size: %d vs %d", len(blob), so.Size())
	}
	blob = nil
	runtime.GC()

	free := memAvailableMB()
	peak := ownPeakRSSMB()
	open, need := growthGateDecision(free, peak, int64(CFG.GrowthMinFreeMB), CFG.GrowthPeakFactorPct)
	t.Logf("gate against this organism: peak=%d MB, factor=%d%%, need=%d MB, free=%d MB, open=%v",
		peak, CFG.GrowthPeakFactorPct, need, free, open)
}

// A stage-4 organism built from nothing, so the cost of the save path is
// measurable without a 110 MB file. Guarded by the same memory floor.
func TestStage4SavePeak(t *testing.T) {
	if os.Getenv("MOLEQULA_HEAVY") == "" {
		t.Skip("set MOLEQULA_HEAVY=1 to build a stage-4 organism and measure its save")
	}
	if free := memAvailableMB(); free < 2000 {
		t.Skipf("MemAvailable %d MB < 2000 MB — not building a stage-4 organism here", free)
	}
	if _, err := applyOomScoreAdj(oomScoreAdjPath, 500); err != nil {
		t.Logf("could not raise oom_score_adj: %v", err)
	}
	saved := CFG
	defer func() { CFG = saved }()
	CFG.NEmbd, CFG.NLayer, CFG.NHead = 16, 1, 1
	CFG.BlockSize = 96
	CFG.HeadTypes = []string{"content"}
	CFG.TieEmbeddings = true
	CFG.CheckpointMinInterval = 0

	words := make([]string, 0, 512)
	for i := 0; i < 512; i++ {
		words = append(words, "organism "+strconv.Itoa(i)+" speaks and grows")
	}
	tok := NewEvolvingTokenizer(words)
	model := NewGPT(tok)
	model.corpusIngestedTotal = 999999999
	hwmBefore := procFieldMB("/proc/self/status", "VmHWM")
	for model.CurrentGrowthStage() < 4 && model.MaybeGrowArchitecture() {
		model.growthFreezeRemaining = 0
	}
	nParams := 0
	for _, m := range model.Base {
		nParams += m.Nout * m.Nin
	}
	hwmGrown := procFieldMB("/proc/self/status", "VmHWM")
	t.Logf("grown to stage %d (%d params): HWM %d → %d MB", model.CurrentGrowthStage(), nParams, hwmBefore, hwmGrown)

	out := filepath.Join(t.TempDir(), "ckpt.json")
	runtime.GC()
	rssBefore := procFieldMB("/proc/self/status", "VmRSS")
	if err := SaveCheckpoint(model, tok, out); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}
	hwmSaved := procFieldMB("/proc/self/status", "VmHWM")
	fi, _ := os.Stat(out)
	t.Logf("streamed save of %.1f MB: RSS before %d MB, HWM %d → %d MB",
		float64(fi.Size())/(1<<20), rssBefore, hwmGrown, hwmSaved)

	runtime.GC()
	blob, err := checkpointViaEncoder(model, tok)
	if err != nil {
		t.Fatalf("oracle encode: %v", err)
	}
	hwmOld := procFieldMB("/proc/self/status", "VmHWM")
	t.Logf("old save of the same organism: HWM %d → %d MB (+%d), %.1f MB held in memory",
		hwmSaved, hwmOld, hwmOld-hwmSaved, float64(len(blob))/(1<<20))
	blob = nil
	runtime.GC()
}

// ── 7. the load path, repair 10 ───────────────────────────────────────────────

// Save → load → save. The streamed loader builds *MatrixParam rows straight out
// of the token stream instead of materialising a CheckpointData first, so the
// gate that it still reads the format the streamed writer emits is that the
// second file is the first file. Change the field order in either direction, or
// lose a null row, or drop the grad allocation, and this goes red.
func TestCheckpointRoundTripIsByteIdentical(t *testing.T) {
	model, tok := smallOrganism(t)
	CFG.CheckpointMinInterval = 0

	dir := t.TempDir()
	first := filepath.Join(dir, "first.json")
	if err := SaveCheckpoint(model, tok, first); err != nil {
		t.Fatalf("first save: %v", err)
	}
	back, backTok, err := LoadCheckpoint([]string{"the organism speaks"}, first)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	second := filepath.Join(dir, "second.json")
	if err := SaveCheckpoint(back, backTok, second); err != nil {
		t.Fatalf("second save: %v", err)
	}
	a, err := os.ReadFile(first)
	if err != nil {
		t.Fatalf("reading the first file: %v", err)
	}
	b, err := os.ReadFile(second)
	if err != nil {
		t.Fatalf("reading the second file: %v", err)
	}
	if len(a) < 1000 {
		t.Fatalf("the fixture is too small to prove anything: %d bytes", len(a))
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("the round trip changed the checkpoint (%d vs %d bytes)\nfirst  tail: %s\nsecond tail: %s",
			len(a), len(b), tailOf(a), tailOf(b))
	}

	// Byte equality is about the format; these are about the organism.
	if back.globalStep != model.globalStep || back.growthStepOffset != model.growthStepOffset ||
		back.lastWarmupStage != model.lastWarmupStage {
		t.Fatalf("growth state did not survive: step %d/%d offset %d/%d warmup %d/%d",
			back.globalStep, model.globalStep, back.growthStepOffset, model.growthStepOffset,
			back.lastWarmupStage, model.lastWarmupStage)
	}
	wtIn, wtOut := model.Base["wte"], back.Base["wte"]
	if wtOut.Nout != wtIn.Nout || wtOut.Nin != wtIn.Nin {
		t.Fatalf("wte came back %dx%d, was %dx%d", wtOut.Nout, wtOut.Nin, wtIn.Nout, wtIn.Nin)
	}
	for i := range wtIn.Rows {
		for j := range wtIn.Rows[i].Data {
			if wtOut.Rows[i].Data[j] != wtIn.Rows[i].Data[j] {
				t.Fatalf("wte[%d][%d]: %v came back as %v", i, j, wtIn.Rows[i].Data[j], wtOut.Rows[i].Data[j])
			}
		}
		// A loaded parameter is a trainable one: without grad the next warmup
		// writes into a nil slice.
		if len(wtOut.Rows[i].Grad) != len(wtOut.Rows[i].Data) {
			t.Fatalf("wte row %d came back with %d grads for %d weights", i, len(wtOut.Rows[i].Grad), len(wtOut.Rows[i].Data))
		}
	}
	if CFG.TieEmbeddings && back.Base["lm_head"] != back.Base["wte"] {
		t.Fatal("the embedding tie was not re-established after the load")
	}
	if len(back.Deltas) != len(model.Deltas) {
		t.Fatalf("%d delta modules came back for %d saved", len(back.Deltas), len(model.Deltas))
	}
	t.Logf("round trip identical, %d bytes, %d delta modules", len(a), len(back.Deltas))
}

// A checkpoint written by a process the low-memory killer reached mid-write is a
// prefix of a valid document. The streamed loader walks it token by token, so
// every cut lands inside some Token() or Decode() call: the gate is that each
// one comes back as an error and a nil model, not as a panic taking the colony
// with it, and not as a half-built organism that trains on garbage.
func TestLoadCheckpointRejectsTruncation(t *testing.T) {
	model, tok := smallOrganism(t)
	CFG.CheckpointMinInterval = 0

	dir := t.TempDir()
	full := filepath.Join(dir, "full.json")
	if err := SaveCheckpoint(model, tok, full); err != nil {
		t.Fatalf("save: %v", err)
	}
	blob, err := os.ReadFile(full)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	// Cuts through the head, the base matrices, the deltas and the trailing
	// scalars — the last one drops only the closing brace.
	for _, frac := range []float64{0.01, 0.1, 0.33, 0.5, 0.75, 0.95} {
		cut := int(float64(len(blob)) * frac)
		path := filepath.Join(dir, "cut_"+strconv.Itoa(cut)+".json")
		if err := os.WriteFile(path, blob[:cut], 0644); err != nil {
			t.Fatalf("writing the truncated copy: %v", err)
		}
		m, tk, err := LoadCheckpoint([]string{"the organism speaks"}, path)
		if err == nil {
			t.Fatalf("a checkpoint truncated at %d of %d bytes loaded without an error", cut, len(blob))
		}
		if m != nil || tk != nil {
			t.Fatalf("truncation at %d returned an organism alongside the error", cut)
		}
	}
	short := filepath.Join(dir, "one_brace.json")
	if err := os.WriteFile(short, []byte("{"), 0644); err != nil {
		t.Fatalf("writing the one-byte copy: %v", err)
	}
	if _, _, err := LoadCheckpoint([]string{"x"}, short); err == nil {
		t.Fatal(`a file holding only "{" loaded without an error`)
	}

	// And the untruncated file still loads, so the gate is about the cut and
	// not about the fixture.
	if _, _, err := LoadCheckpoint([]string{"the organism speaks"}, full); err != nil {
		t.Fatalf("the complete checkpoint stopped loading: %v", err)
	}
	t.Logf("six truncations of a %d-byte checkpoint, all refused", len(blob))
}

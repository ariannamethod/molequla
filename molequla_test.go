package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ============================================================
// MatrixParam tests
// ============================================================

func TestNewMatrixParam(t *testing.T) {
	m := NewMatrixParam(3, 4, 0.08)
	if m.Nout != 3 {
		t.Errorf("expected Nout=3, got %d", m.Nout)
	}
	if m.Nin != 4 {
		t.Errorf("expected Nin=4, got %d", m.Nin)
	}
	if len(m.Rows) != 3 {
		t.Errorf("expected 3 rows, got %d", len(m.Rows))
	}
	for i, row := range m.Rows {
		if len(row.Data) != 4 {
			t.Errorf("row %d: expected 4 cols, got %d", i, len(row.Data))
		}
	}
}

func TestMatrixParamGrowCols(t *testing.T) {
	m := NewMatrixParam(2, 3, 0.0) // zero init for easy checking
	// Set known values
	m.Rows[0].Data = []float64{1, 2, 3}
	m.Rows[1].Data = []float64{4, 5, 6}

	m.GrowCols(5, 0.0)

	if m.Nin != 5 {
		t.Errorf("expected Nin=5, got %d", m.Nin)
	}
	// Original data preserved
	if m.Rows[0].Data[0] != 1 || m.Rows[0].Data[1] != 2 || m.Rows[0].Data[2] != 3 {
		t.Errorf("row 0 original data corrupted: %v", m.Rows[0].Data[:3])
	}
	if m.Rows[1].Data[0] != 4 || m.Rows[1].Data[1] != 5 || m.Rows[1].Data[2] != 6 {
		t.Errorf("row 1 original data corrupted: %v", m.Rows[1].Data[:3])
	}
	// New cols exist
	if len(m.Rows[0].Data) != 5 {
		t.Errorf("expected 5 cols after grow, got %d", len(m.Rows[0].Data))
	}
}

func TestMatrixParamGrowColsNoop(t *testing.T) {
	m := NewMatrixParam(2, 5, 0.08)
	m.GrowCols(3, 0.08) // smaller — should be noop
	if m.Nin != 5 {
		t.Errorf("GrowCols to smaller should be noop, got Nin=%d", m.Nin)
	}
}

func TestMatrixParamGrowRows(t *testing.T) {
	m := NewMatrixParam(2, 3, 0.0)
	m.Rows[0].Data = []float64{1, 2, 3}
	m.Rows[1].Data = []float64{4, 5, 6}

	m.GrowRows(4, 0.0)

	if m.Nout != 4 {
		t.Errorf("expected Nout=4, got %d", m.Nout)
	}
	if len(m.Rows) != 4 {
		t.Errorf("expected 4 rows, got %d", len(m.Rows))
	}
	// Original rows preserved
	if m.Rows[0].Data[0] != 1 {
		t.Errorf("original row 0 corrupted")
	}
	if m.Rows[1].Data[0] != 4 {
		t.Errorf("original row 1 corrupted")
	}
	// New rows have correct width
	if len(m.Rows[2].Data) != 3 {
		t.Errorf("new row 2: expected 3 cols, got %d", len(m.Rows[2].Data))
	}
	if len(m.Rows[3].Data) != 3 {
		t.Errorf("new row 3: expected 3 cols, got %d", len(m.Rows[3].Data))
	}
}

func TestMatrixParamGrow(t *testing.T) {
	m := NewMatrixParam(2, 3, 0.08)
	m.Grow(4, 5, 0.08)

	if m.Nout != 4 {
		t.Errorf("expected Nout=4, got %d", m.Nout)
	}
	if m.Nin != 5 {
		t.Errorf("expected Nin=5, got %d", m.Nin)
	}
	// All rows should have new width
	for i, row := range m.Rows {
		if len(row.Data) != 5 {
			t.Errorf("row %d: expected 5 cols, got %d", i, len(row.Data))
		}
	}
}

func TestMatvec(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	// 2x3 matrix @ 3-vec
	m := NewMatrixParam(2, 3, 0.0)
	m.Rows[0].Data = []float64{1, 0, 0}
	m.Rows[1].Data = []float64{0, 1, 0}
	x := NewVec([]float64{3, 7, 11})

	out := m.Matvec(x)
	if len(out.Data) != 2 {
		t.Fatalf("expected 2-element output, got %d", len(out.Data))
	}
	if out.Data[0] != 3.0 {
		t.Errorf("expected out[0]=3, got %f", out.Data[0])
	}
	if out.Data[1] != 7.0 {
		t.Errorf("expected out[1]=7, got %f", out.Data[1])
	}
}

// ============================================================
// Serialization round-trip
// ============================================================

func TestSerializeDeserializeMatrixParam(t *testing.T) {
	m := NewMatrixParam(3, 4, 0.08)
	// Set deterministic values
	for i := range m.Rows {
		for j := range m.Rows[i].Data {
			m.Rows[i].Data[j] = float64(i*10 + j)
		}
	}

	data := serializeMatrixParam(m)
	m2 := deserializeMatrixParam(data)

	if m2.Nout != m.Nout {
		t.Errorf("Nout mismatch: %d vs %d", m2.Nout, m.Nout)
	}
	if m2.Nin != m.Nin {
		t.Errorf("Nin mismatch: %d vs %d", m2.Nin, m.Nin)
	}
	for i := range m.Rows {
		for j := range m.Rows[i].Data {
			if m2.Rows[i].Data[j] != m.Rows[i].Data[j] {
				t.Errorf("[%d][%d] mismatch: %f vs %f", i, j, m2.Rows[i].Data[j], m.Rows[i].Data[j])
			}
		}
	}
}

// ============================================================
// TieEmbeddings — the critical bug fix
// ============================================================

func TestTieEmbeddingsNewGPT(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.TieEmbeddings = true
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5

	tok := NewEvolvingTokenizer([]string{"hello world"})
	model := NewGPT(tok)

	// With TieEmbeddings=true, lm_head and wte must be the SAME pointer
	if model.Base["lm_head"] != model.Base["wte"] {
		t.Fatal("TieEmbeddings=true but lm_head != wte (pointer identity broken)")
	}
}

func TestTieEmbeddingsSaveLoadRoundTrip(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.TieEmbeddings = true
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.GrowthStages = [][4]int{{0, 16, 1, 1}}

	tok := NewEvolvingTokenizer([]string{"hello world"})
	model := NewGPT(tok)
	model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
	for i, row := range model.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		model.InitEmbedSnapshot[i] = snap
	}

	// Save to temp file
	tmpFile := filepath.Join(t.TempDir(), "test_ckpt.json")
	if err := SaveCheckpoint(model, tok, tmpFile); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	// Load back
	model2, _, err := LoadCheckpoint([]string{"hello world"}, tmpFile)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}

	// THE critical check: after load, lm_head must be the SAME pointer as wte
	if model2.Base["lm_head"] != model2.Base["wte"] {
		t.Fatal("TieEmbeddings broken after SaveLoad: lm_head != wte (pointer identity not restored)")
	}

	// Verify dimensions match
	wte := model2.Base["wte"]
	lmHead := model2.Base["lm_head"]
	if wte.Nout != lmHead.Nout || wte.Nin != lmHead.Nin {
		t.Errorf("dimension mismatch after load: wte=%dx%d, lm_head=%dx%d",
			wte.Nout, wte.Nin, lmHead.Nout, lmHead.Nin)
	}
}

func TestTieEmbeddingsGrowPreservesIdentity(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.TieEmbeddings = true
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5

	tok := NewEvolvingTokenizer([]string{"hello"})
	model := NewGPT(tok)

	// Grow wte columns (simulating ontogenesis)
	model.Base["wte"].GrowCols(32, 0.001)

	// Since lm_head IS wte (same pointer), it should also be grown
	if model.Base["lm_head"].Nin != 32 {
		t.Errorf("lm_head.Nin should be 32 after wte grow (same pointer), got %d", model.Base["lm_head"].Nin)
	}
}

// ============================================================
// Growth stages / ontogenesis
// ============================================================

func TestCurrentGrowthStage(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
		{50000, 64, 2, 4},
		{200000, 128, 4, 4},
	}
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true

	tok := NewEvolvingTokenizer([]string{"test"})

	tests := []struct {
		embd, layer, head int
		want              int
	}{
		{16, 1, 1, 0},   // embryo
		{32, 1, 2, 1},   // infant
		{64, 2, 4, 2},   // child
		{128, 4, 4, 3},  // adolescent
		{99, 3, 3, -1},  // legacy (no match)
	}

	for _, tt := range tests {
		CFG.NEmbd = tt.embd
		CFG.NLayer = tt.layer
		CFG.NHead = tt.head
		CFG.HeadTypes = headTypesForNHead(tt.head)
		model := NewGPT(tok)
		got := model.CurrentGrowthStage()
		if got != tt.want {
			t.Errorf("embd=%d layer=%d head=%d: CurrentGrowthStage()=%d, want %d",
				tt.embd, tt.layer, tt.head, got, tt.want)
		}
	}
}

func TestTargetGrowthStage(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
		{50000, 64, 2, 4},
		{200000, 128, 4, 4},
	}
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)

	tests := []struct {
		corpusChars int
		want        int
	}{
		{0, 0},       // embryo
		{10000, 0},   // still embryo
		{20000, 1},   // infant threshold
		{49999, 1},   // still infant
		{50000, 2},   // child
		{199999, 2},  // still child
		{200000, 3},  // adolescent
		{999999, 3},  // stays at max
	}

	for _, tt := range tests {
		got := model.TargetGrowthStage(tt.corpusChars)
		if got != tt.want {
			t.Errorf("corpusChars=%d: TargetGrowthStage()=%d, want %d", tt.corpusChars, got, tt.want)
		}
	}
}

func TestMaybeGrowArchitectureOneStageAtATime(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
		{50000, 64, 2, 4},
		{200000, 128, 4, 4},
	}
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.FreezeAfterGrowthSteps = 100

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)
	model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
	for i, row := range model.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		model.InitEmbedSnapshot[i] = snap
	}
	model.AddDeltaModule(1.0)

	// Even with corpus=999999 (enough for adolescent), should grow only to infant (stage 0→1)
	grew := model.MaybeGrowArchitecture(999999)
	if !grew {
		t.Fatal("MaybeGrowArchitecture should have grown")
	}
	if model.CurrentGrowthStage() != 1 {
		t.Errorf("should be at stage 1 (infant), got %d", model.CurrentGrowthStage())
	}
	if model.NEmbd != 32 {
		t.Errorf("expected NEmbd=32, got %d", model.NEmbd)
	}
	if model.NLayer != 1 {
		t.Errorf("expected NLayer=1, got %d", model.NLayer)
	}
	if model.NHead != 2 {
		t.Errorf("expected NHead=2, got %d", model.NHead)
	}
}

func TestMaybeGrowArchitectureFreezeBlocks(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
	}
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.FreezeAfterGrowthSteps = 100

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)
	model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
	for i, row := range model.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		model.InitEmbedSnapshot[i] = snap
	}
	model.AddDeltaModule(1.0)

	// First growth
	grew := model.MaybeGrowArchitecture(30000)
	if !grew {
		t.Fatal("first growth should succeed")
	}
	if model.growthFreezeRemaining != 100 {
		t.Errorf("expected freeze=100, got %d", model.growthFreezeRemaining)
	}

	// Second growth should be blocked by freeze
	grew = model.MaybeGrowArchitecture(999999)
	if grew {
		t.Fatal("growth during freeze should be blocked")
	}
}

func TestMaybeGrowArchitectureLegacySkips(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
	}
	CFG.NEmbd = 99 // doesn't match any stage
	CFG.NLayer = 3
	CFG.NHead = 3
	CFG.BlockSize = 32
	CFG.HeadTypes = headTypesForNHead(3)
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)

	grew := model.MaybeGrowArchitecture(999999)
	if grew {
		t.Fatal("legacy checkpoint (no matching stage) should not grow")
	}
}

func TestMaybeGrowArchitectureMatrixDimensions(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
	}
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.FreezeAfterGrowthSteps = 100

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)
	model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
	for i, row := range model.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		model.InitEmbedSnapshot[i] = snap
	}
	model.AddDeltaModule(1.0)

	model.MaybeGrowArchitecture(30000)

	// After growth to stage 1 (32 embd), check matrix dims
	wte := model.Base["wte"]
	if wte.Nin != 32 {
		t.Errorf("wte.Nin should be 32 after growth, got %d", wte.Nin)
	}

	wq := model.Base["l0.wq"]
	if wq.Nout != 32 || wq.Nin != 32 {
		t.Errorf("l0.wq should be 32x32, got %dx%d", wq.Nout, wq.Nin)
	}

	fcG := model.Base["l0.fc_g"]
	if fcG.Nout != 128 || fcG.Nin != 32 {
		t.Errorf("l0.fc_g should be 128x32, got %dx%d", fcG.Nout, fcG.Nin)
	}

	fc2 := model.Base["l0.fc2"]
	if fc2.Nout != 32 || fc2.Nin != 128 {
		t.Errorf("l0.fc2 should be 32x128, got %dx%d", fc2.Nout, fc2.Nin)
	}

	// Verify all matrices have consistent row widths (the crash bug)
	for name, m := range model.Base {
		if len(m.Rows) == 0 {
			continue
		}
		for i, row := range m.Rows {
			if len(row.Data) != m.Nin {
				t.Errorf("%s row[%d] has %d cols but Nin=%d", name, i, len(row.Data), m.Nin)
			}
		}
	}
}

// ============================================================
// TieEmbeddings + ontogenesis = the crash scenario
// ============================================================

func TestTieEmbeddingsOntogenesisThenSaveLoad(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.GrowthStages = [][4]int{
		{0, 16, 1, 1},
		{20000, 32, 1, 2},
	}
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.FreezeAfterGrowthSteps = 100

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)
	model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
	for i, row := range model.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		model.InitEmbedSnapshot[i] = snap
	}
	model.AddDeltaModule(1.0)

	// Grow: embryo → infant
	model.MaybeGrowArchitecture(30000)

	// Save
	tmpFile := filepath.Join(t.TempDir(), "ckpt_after_growth.json")
	if err := SaveCheckpoint(model, tok, tmpFile); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	// Load
	model2, _, err := LoadCheckpoint([]string{"test"}, tmpFile)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}

	// THE critical regression test:
	// Before the fix, lm_head would have old dimensions (V x 16) while wte has (V x 32)
	// This caused panic: index out of range [16] with length 16 in Matvec
	if model2.Base["lm_head"] != model2.Base["wte"] {
		t.Fatal("REGRESSION: TieEmbeddings pointer identity broken after growth+save+load")
	}

	wte := model2.Base["wte"]
	if wte.Nin != 32 {
		t.Errorf("wte.Nin should be 32 after growth+load, got %d", wte.Nin)
	}

	// Verify we can do a matvec without panic (the actual crash scenario)
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)
	x := NewVecZero(32) // 32-dim input (grown embedding)
	result := model2.Base["lm_head"].Matvec(x)
	if len(result.Data) != wte.Nout {
		t.Errorf("lm_head matvec output should have %d elements, got %d", wte.Nout, len(result.Data))
	}
}

// ============================================================
// DNA exchange
// ============================================================

func TestDnaReadWriteFilesystem(t *testing.T) {
	tmpDir := t.TempDir()

	// Create dna/output structure
	for _, elem := range []string{"earth", "air", "water", "fire"} {
		os.MkdirAll(filepath.Join(tmpDir, "dna", "output", elem), 0755)
	}

	// Create a corpus file for "earth"
	corpusPath := filepath.Join(tmpDir, "corpus.txt")
	os.WriteFile(corpusPath, []byte("initial corpus\n"), 0644)

	// Simulate air writing DNA
	airDir := filepath.Join(tmpDir, "dna", "output", "air")
	os.WriteFile(filepath.Join(airDir, "gen_1_0.txt"), []byte("I am air, I breathe the wind and carry seeds."), 0644)
	os.WriteFile(filepath.Join(airDir, "gen_2_0.txt"), []byte("The sky speaks in whispers of ancient truths."), 0644)

	// Now test dnaRead from earth's perspective
	// Need to chdir to work_earth so ../dna/ resolves correctly
	workDir := filepath.Join(tmpDir, "work_earth")
	os.MkdirAll(workDir, 0755)

	// Create the symlink structure dnaRead expects (../dna/output/)
	// dnaRead uses relative paths: ../dna/output/{element}/
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(workDir)

	// dnaRead looks for ../dna/output/{elem}/ relative to cwd
	added := dnaRead("earth", corpusPath)

	if added <= 0 {
		t.Errorf("dnaRead should have consumed air's DNA, got added=%d", added)
	}

	// Verify corpus grew
	data, _ := os.ReadFile(corpusPath)
	if len(data) <= len("initial corpus\n") {
		t.Error("corpus should have grown after dnaRead")
	}

	// Verify consumed files are deleted
	entries, _ := os.ReadDir(airDir)
	if len(entries) != 0 {
		t.Errorf("consumed files should be deleted, but %d remain", len(entries))
	}
}

func TestDnaReadSkipsSelf(t *testing.T) {
	tmpDir := t.TempDir()

	// Create dna/output/earth with a file
	earthDir := filepath.Join(tmpDir, "dna", "output", "earth")
	os.MkdirAll(earthDir, 0755)
	os.WriteFile(filepath.Join(earthDir, "gen_1_0.txt"), []byte("Earth's own words should not be consumed."), 0644)

	corpusPath := filepath.Join(tmpDir, "corpus.txt")
	os.WriteFile(corpusPath, []byte("initial\n"), 0644)

	workDir := filepath.Join(tmpDir, "work_earth")
	os.MkdirAll(workDir, 0755)
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(workDir)

	// Earth should NOT consume its own DNA
	added := dnaRead("earth", corpusPath)
	if added != 0 {
		t.Errorf("earth should not consume its own DNA, got added=%d", added)
	}

	// File should still exist
	entries, _ := os.ReadDir(earthDir)
	if len(entries) != 1 {
		t.Errorf("earth's own DNA file should still exist, got %d files", len(entries))
	}
}

func TestDnaReadSkipsShortFiles(t *testing.T) {
	tmpDir := t.TempDir()

	airDir := filepath.Join(tmpDir, "dna", "output", "air")
	os.MkdirAll(airDir, 0755)
	os.WriteFile(filepath.Join(airDir, "gen_1_0.txt"), []byte("short"), 0644) // < 10 chars

	corpusPath := filepath.Join(tmpDir, "corpus.txt")
	os.WriteFile(corpusPath, []byte("initial\n"), 0644)

	workDir := filepath.Join(tmpDir, "work_earth")
	os.MkdirAll(workDir, 0755)
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(workDir)

	added := dnaRead("earth", corpusPath)
	if added != 0 {
		t.Errorf("short files (<10 chars) should be skipped, got added=%d", added)
	}

	// Short file should be deleted (cleaned up)
	entries, _ := os.ReadDir(airDir)
	if len(entries) != 0 {
		t.Errorf("short DNA file should be deleted, got %d files", len(entries))
	}
}

func TestDnaReadEmptyElement(t *testing.T) {
	added := dnaRead("", "/dev/null")
	if added != 0 {
		t.Errorf("empty element should return 0, got %d", added)
	}
}

// ============================================================
// RMSNorm
// ============================================================

func TestRMSNorm(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	x := NewVec([]float64{3.0, 4.0})
	out := RMSNorm(x)
	// rms = sqrt((9+16)/2) = sqrt(12.5)
	// scale = 1/sqrt(12.5 + 1e-5)
	rms := math.Sqrt(12.5 + 1e-5)
	scale := 1.0 / rms
	if math.Abs(out.Data[0]-3.0*scale) > 1e-6 {
		t.Errorf("RMSNorm[0] expected %f, got %f", 3.0*scale, out.Data[0])
	}
	if math.Abs(out.Data[1]-4.0*scale) > 1e-6 {
		t.Errorf("RMSNorm[1] expected %f, got %f", 4.0*scale, out.Data[1])
	}
}

// ============================================================
// CrossEntropyLoss
// ============================================================

func TestCrossEntropyLoss(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	// With logits [0, 0, 1000], softmax ≈ [0, 0, 1], loss for target=2 ≈ 0
	logits := NewVec([]float64{0, 0, 1000})
	loss := CrossEntropyLoss(logits, 2)
	if loss.Data > 0.01 {
		t.Errorf("loss should be ~0 for correct high-confidence prediction, got %f", loss.Data)
	}

	// With logits [1000, 0, 0], loss for target=2 should be large
	logits2 := NewVec([]float64{1000, 0, 0})
	loss2 := CrossEntropyLoss(logits2, 2)
	if loss2.Data < 100 {
		t.Errorf("loss should be large for wrong prediction, got %f", loss2.Data)
	}
}

// ============================================================
// headTypesForNHead
// ============================================================

func TestHeadTypesForNHead(t *testing.T) {
	tests := []struct {
		n    int
		want []string
	}{
		{1, []string{"content"}},
		{2, []string{"content", "hybrid"}},
		{4, []string{"content", "content", "hybrid", "hybrid"}},
		{8, []string{"content", "content", "content", "content", "hybrid", "hybrid", "hybrid", "hybrid"}},
	}

	for _, tt := range tests {
		got := headTypesForNHead(tt.n)
		if len(got) != tt.n {
			t.Errorf("headTypesForNHead(%d): expected %d types, got %d", tt.n, tt.n, len(got))
			continue
		}
		for i, typ := range got {
			if typ != tt.want[i] {
				t.Errorf("headTypesForNHead(%d)[%d]=%s, want %s", tt.n, i, typ, tt.want[i])
			}
		}
	}
}

// ============================================================
// CosineLR
// ============================================================

func TestCosineLR(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.LearningRate = 0.01
	CFG.LRMin = 0.001
	CFG.CosineWarmupSteps = 100
	CFG.MaxTotalSteps = 10000

	// During warmup (step 0, stepsSinceGrowth=0): should be LRMin
	lr := cosineLR(0, 0)
	if math.Abs(lr-CFG.LRMin) > 1e-10 {
		t.Errorf("at warmup start, lr should be %f, got %f", CFG.LRMin, lr)
	}

	// At warmup end (stepsSinceGrowth=99, just before cutoff): should be close to full LR
	lr = cosineLR(99, 99)
	expected := CFG.LRMin + (CFG.LearningRate-CFG.LRMin)*99.0/100.0
	if math.Abs(lr-expected) > 1e-10 {
		t.Errorf("at warmup step 99, lr should be %f, got %f", expected, lr)
	}

	// At stepsSinceGrowth=CosineWarmupSteps, should switch to cosine (not warmup)
	lr = cosineLR(100, 100)
	if lr >= CFG.LearningRate {
		t.Errorf("at step 100 (past warmup), lr should be slightly below LR, got %f", lr)
	}

	// LR should decrease over time (cosine decay)
	lr1 := cosineLR(1000, 1000)
	lr2 := cosineLR(5000, 5000)
	if lr1 <= lr2 {
		t.Errorf("LR should decrease: lr(1000)=%f should be > lr(5000)=%f", lr1, lr2)
	}
}

// ============================================================
// parseCLIArgs
// ============================================================

func TestParseCLIArgs(t *testing.T) {
	// Save original args
	origArgs := os.Args
	defer func() { os.Args = origArgs }()

	os.Args = []string{"molequla", "--element", "earth", "--evolution", "--organism-id", "test-id"}

	id, _, elem, evo := parseCLIArgs()
	if elem != "earth" {
		t.Errorf("expected element=earth, got %s", elem)
	}
	if !evo {
		t.Error("expected evolution=true")
	}
	if id != "test-id" {
		t.Errorf("expected organism-id=test-id, got %s", id)
	}
}

func TestParseCLIArgsDefaults(t *testing.T) {
	origArgs := os.Args
	defer func() { os.Args = origArgs }()

	os.Args = []string{"molequla"}

	id, cfg, elem, evo := parseCLIArgs()
	if id != "" || cfg != "" || elem != "" || evo {
		t.Errorf("defaults should be empty: id=%q cfg=%q elem=%q evo=%v", id, cfg, elem, evo)
	}
}

// ============================================================
// Checkpoint serialization: full round-trip with deltas
// ============================================================

func TestCheckpointRoundTripWithDeltas(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true
	CFG.DeltaRank = 4
	CFG.GrowthStages = [][4]int{{0, 16, 1, 1}}

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)
	model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
	for i, row := range model.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		model.InitEmbedSnapshot[i] = snap
	}
	model.AddDeltaModule(1.0)
	model.globalStep = 42

	tmpFile := filepath.Join(t.TempDir(), "ckpt.json")
	if err := SaveCheckpoint(model, tok, tmpFile); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	model2, tok2, err := LoadCheckpoint([]string{"test"}, tmpFile)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}

	// Check dimensions
	if model2.NEmbd != 16 || model2.NLayer != 1 || model2.NHead != 1 {
		t.Errorf("dimensions wrong: %d/%d/%d", model2.NEmbd, model2.NLayer, model2.NHead)
	}

	// Check global step preserved
	if model2.globalStep != 42 {
		t.Errorf("globalStep should be 42, got %d", model2.globalStep)
	}

	// Check tokenizer round-trip
	if tok2.VocabSize != tok.VocabSize {
		t.Errorf("vocab size mismatch: %d vs %d", tok2.VocabSize, tok.VocabSize)
	}

	// Check deltas exist
	if len(model2.Deltas) == 0 {
		t.Error("deltas should be preserved after load")
	}
}

// ============================================================
// MaybeExpandVocab with TieEmbeddings
// ============================================================

func TestMaybeExpandVocabTieEmbeddings(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.TieEmbeddings = true
	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.DeltaRank = 4

	tok := NewEvolvingTokenizer([]string{"hello"})
	model := NewGPT(tok)

	oldVocab := tok.VocabSize
	model.MaybeExpandVocab(oldVocab + 10)

	// wte should have grown
	if model.Base["wte"].Nout != oldVocab+10 {
		t.Errorf("wte.Nout should be %d, got %d", oldVocab+10, model.Base["wte"].Nout)
	}

	// lm_head should also be grown (same pointer)
	if model.Base["lm_head"].Nout != oldVocab+10 {
		t.Errorf("lm_head.Nout should be %d (tied), got %d", oldVocab+10, model.Base["lm_head"].Nout)
	}
}

// ============================================================
// Checkpoint JSON structure
// ============================================================

func TestCheckpointJSONHasCfg(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	CFG.NEmbd = 16
	CFG.NLayer = 1
	CFG.NHead = 1
	CFG.BlockSize = 32
	CFG.HeadTypes = []string{"content"}
	CFG.HybridAlphaInit = 0.5
	CFG.TieEmbeddings = true
	CFG.GrowthStages = [][4]int{{0, 16, 1, 1}}

	tok := NewEvolvingTokenizer([]string{"test"})
	model := NewGPT(tok)
	model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
	for i, row := range model.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		model.InitEmbedSnapshot[i] = snap
	}

	tmpFile := filepath.Join(t.TempDir(), "ckpt.json")
	SaveCheckpoint(model, tok, tmpFile)

	// Read back and verify cfg is embedded
	data, _ := os.ReadFile(tmpFile)
	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	if _, ok := raw["cfg"]; !ok {
		t.Error("checkpoint JSON should contain 'cfg' field")
	}

	// Verify cfg contains n_embd
	var cfgMap map[string]interface{}
	json.Unmarshal(raw["cfg"], &cfgMap)
	if cfgMap["n_embd"] != float64(16) {
		t.Errorf("cfg.n_embd should be 16, got %v", cfgMap["n_embd"])
	}
}

// ============================================================
// DeltaAdapter
// ============================================================

func TestDeltaAdapterApply(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	da := NewDeltaAdapter(4, 3, 2, 0.0)
	// Zero-init means output should be all zeros
	x := NewVec([]float64{1, 2, 3})
	out := da.Apply(x)
	if len(out.Data) != 4 {
		t.Fatalf("expected 4-element output, got %d", len(out.Data))
	}
}

func TestDeltaAdapterGrowDims(t *testing.T) {
	da := NewDeltaAdapter(4, 3, 2, 0.08)
	da.GrowDims(6, 5)

	if da.A.Nout != 6 {
		t.Errorf("A.Nout should be 6 after grow, got %d", da.A.Nout)
	}
	if da.B.Nin != 5 {
		t.Errorf("B.Nin should be 5 after grow, got %d", da.B.Nin)
	}
}

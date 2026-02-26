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
	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	added := dnaRead("earth", corpusPath, qb, tok)

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
	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	added := dnaRead("earth", corpusPath, qb, tok)
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

	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	added := dnaRead("earth", corpusPath, qb, tok)
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
	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	added := dnaRead("", "/dev/null", qb, tok)
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

// ============================================================
// AUTOGRAD GRADIENT FLOW — the beating heart of backprop
// ============================================================

func TestVecAddGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{1, 2, 3})
	b := NewVec([]float64{4, 5, 6})
	c := a.Add(b)
	// Forward check
	if c.Data[0] != 5 || c.Data[1] != 7 || c.Data[2] != 9 {
		t.Fatalf("Add forward wrong: %v", c.Data)
	}
	// Backward
	c.Grad = []float64{1, 1, 1}
	Backward(c)
	for i := 0; i < 3; i++ {
		if a.Grad[i] != 1.0 {
			t.Errorf("a.Grad[%d]=%f, want 1.0", i, a.Grad[i])
		}
		if b.Grad[i] != 1.0 {
			t.Errorf("b.Grad[%d]=%f, want 1.0", i, b.Grad[i])
		}
	}
}

func TestVecSubGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{5, 7})
	b := NewVec([]float64{2, 3})
	c := a.Sub(b)
	if c.Data[0] != 3 || c.Data[1] != 4 {
		t.Fatalf("Sub forward wrong: %v", c.Data)
	}
	c.Grad = []float64{1, 1}
	Backward(c)
	if a.Grad[0] != 1.0 || a.Grad[1] != 1.0 {
		t.Errorf("a.Grad wrong: %v", a.Grad)
	}
	if b.Grad[0] != -1.0 || b.Grad[1] != -1.0 {
		t.Errorf("b.Grad wrong (should be -1): %v", b.Grad)
	}
}

func TestVecNegGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{3, -2})
	c := a.Neg()
	if c.Data[0] != -3 || c.Data[1] != 2 {
		t.Fatalf("Neg forward wrong: %v", c.Data)
	}
	c.Grad = []float64{1, 1}
	Backward(c)
	if a.Grad[0] != -1.0 || a.Grad[1] != -1.0 {
		t.Errorf("a.Grad wrong: %v (want [-1,-1])", a.Grad)
	}
}

func TestVecMulVecGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{2, 3})
	b := NewVec([]float64{4, 5})
	c := a.MulVec(b)
	if c.Data[0] != 8 || c.Data[1] != 15 {
		t.Fatalf("MulVec forward wrong: %v", c.Data)
	}
	c.Grad = []float64{1, 1}
	Backward(c)
	// d(a*b)/da = b, d(a*b)/db = a
	if a.Grad[0] != 4.0 || a.Grad[1] != 5.0 {
		t.Errorf("a.Grad wrong: %v (want [4,5])", a.Grad)
	}
	if b.Grad[0] != 2.0 || b.Grad[1] != 3.0 {
		t.Errorf("b.Grad wrong: %v (want [2,3])", b.Grad)
	}
}

func TestVecScaleGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{2, 3})
	c := a.Scale(5.0)
	if c.Data[0] != 10 || c.Data[1] != 15 {
		t.Fatalf("Scale forward wrong: %v", c.Data)
	}
	c.Grad = []float64{1, 1}
	Backward(c)
	if a.Grad[0] != 5.0 || a.Grad[1] != 5.0 {
		t.Errorf("a.Grad wrong: %v (want [5,5])", a.Grad)
	}
}

func TestVecAddScalarGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{1, 2})
	c := a.AddScalar(10.0)
	if c.Data[0] != 11 || c.Data[1] != 12 {
		t.Fatalf("AddScalar forward wrong: %v", c.Data)
	}
	c.Grad = []float64{1, 1}
	Backward(c)
	if a.Grad[0] != 1.0 || a.Grad[1] != 1.0 {
		t.Errorf("a.Grad wrong: %v (want [1,1])", a.Grad)
	}
}

func TestVecReLUGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{3, -2, 0, 5})
	c := a.ReLU()
	expected := []float64{3, 0, 0, 5}
	for i, v := range c.Data {
		if v != expected[i] {
			t.Errorf("ReLU[%d]=%f, want %f", i, v, expected[i])
		}
	}
	c.Grad = []float64{1, 1, 1, 1}
	Backward(c)
	expectedGrad := []float64{1, 0, 0, 1} // grad passes only where input > 0
	for i, g := range a.Grad {
		if g != expectedGrad[i] {
			t.Errorf("ReLU grad[%d]=%f, want %f", i, g, expectedGrad[i])
		}
	}
}

func TestVecSiLUGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{0.0})
	c := a.SiLU()
	// SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
	if math.Abs(c.Data[0]) > 1e-10 {
		t.Errorf("SiLU(0)=%f, want 0", c.Data[0])
	}
	c.Grad = []float64{1.0}
	Backward(c)
	// d/dx[x*sigma(x)] at x=0 = sigma(0)(1 + 0*(1-sigma(0))) = 0.5
	if math.Abs(a.Grad[0]-0.5) > 1e-6 {
		t.Errorf("SiLU grad at 0=%f, want 0.5", a.Grad[0])
	}
}

func TestVecDotGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{1, 2, 3})
	b := NewVec([]float64{4, 5, 6})
	c := a.Dot(b)
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	if c.Data != 32 {
		t.Fatalf("Dot forward wrong: %f (want 32)", c.Data)
	}
	c.Grad = 1.0
	Backward(c)
	// d(a.b)/da = b, d(a.b)/db = a
	for i := 0; i < 3; i++ {
		if a.Grad[i] != b.Data[i] {
			t.Errorf("a.Grad[%d]=%f, want %f", i, a.Grad[i], b.Data[i])
		}
		if b.Grad[i] != a.Data[i] {
			t.Errorf("b.Grad[%d]=%f, want %f", i, b.Grad[i], a.Data[i])
		}
	}
}

func TestVecMeanSqGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{3, 4})
	c := a.MeanSq()
	// (9+16)/2 = 12.5
	if math.Abs(c.Data-12.5) > 1e-10 {
		t.Fatalf("MeanSq forward wrong: %f (want 12.5)", c.Data)
	}
	c.Grad = 1.0
	Backward(c)
	// d/da[mean(a^2)] = 2*a/n
	if math.Abs(a.Grad[0]-3.0) > 1e-10 { // 2*3/2 = 3
		t.Errorf("a.Grad[0]=%f, want 3.0", a.Grad[0])
	}
	if math.Abs(a.Grad[1]-4.0) > 1e-10 { // 2*4/2 = 4
		t.Errorf("a.Grad[1]=%f, want 4.0", a.Grad[1])
	}
}

func TestVecElementGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{10, 20, 30})
	c := a.Element(1)
	if c.Data != 20 {
		t.Fatalf("Element(1)=%f, want 20", c.Data)
	}
	c.Grad = 1.0
	Backward(c)
	if a.Grad[0] != 0 || a.Grad[1] != 1 || a.Grad[2] != 0 {
		t.Errorf("Element grad wrong: %v (want [0,1,0])", a.Grad)
	}
}

func TestVecSliceGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{1, 2, 3, 4, 5})
	c := a.Slice(1, 4)
	if len(c.Data) != 3 || c.Data[0] != 2 || c.Data[1] != 3 || c.Data[2] != 4 {
		t.Fatalf("Slice wrong: %v", c.Data)
	}
	// Dot with weight vector creates scalar -> Backward sets scalar grad=1
	w := NewVec([]float64{10, 20, 30})
	loss := c.Dot(w)
	Backward(loss)
	expected := []float64{0, 10, 20, 30, 0}
	for i, g := range a.Grad {
		if math.Abs(g-expected[i]) > 1e-10 {
			t.Errorf("Slice grad[%d]=%f, want %f", i, g, expected[i])
		}
	}
}

func TestConcatGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewVec([]float64{1, 2})
	b := NewVec([]float64{3, 4, 5})
	c := Concat([]*Vec{a, b})
	if len(c.Data) != 5 {
		t.Fatalf("Concat len=%d, want 5", len(c.Data))
	}
	if c.Data[0] != 1 || c.Data[2] != 3 || c.Data[4] != 5 {
		t.Fatalf("Concat data wrong: %v", c.Data)
	}
	w := NewVec([]float64{10, 20, 30, 40, 50})
	loss := c.Dot(w)
	Backward(loss)
	if a.Grad[0] != 10 || a.Grad[1] != 20 {
		t.Errorf("a.Grad wrong: %v (want [10,20])", a.Grad)
	}
	if b.Grad[0] != 30 || b.Grad[1] != 40 || b.Grad[2] != 50 {
		t.Errorf("b.Grad wrong: %v (want [30,40,50])", b.Grad)
	}
}

func TestScalarAddSGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewScalar(3.0)
	b := NewScalar(4.0)
	c := a.AddS(b)
	if c.Data != 7.0 {
		t.Fatalf("AddS forward: %f, want 7", c.Data)
	}
	c.Grad = 1.0
	Backward(c)
	if a.Grad != 1.0 || b.Grad != 1.0 {
		t.Errorf("AddS grad: a=%f b=%f, want 1,1", a.Grad, b.Grad)
	}
}

func TestScalarMulSGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewScalar(3.0)
	b := NewScalar(4.0)
	c := a.MulS(b)
	if c.Data != 12.0 {
		t.Fatalf("MulS forward: %f, want 12", c.Data)
	}
	c.Grad = 1.0
	Backward(c)
	if a.Grad != 4.0 {
		t.Errorf("a.Grad=%f, want 4", a.Grad)
	}
	if b.Grad != 3.0 {
		t.Errorf("b.Grad=%f, want 3", b.Grad)
	}
}

func TestScalarMulFGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewScalar(5.0)
	c := a.MulF(3.0)
	if c.Data != 15.0 {
		t.Fatalf("MulF forward: %f, want 15", c.Data)
	}
	c.Grad = 1.0
	Backward(c)
	if a.Grad != 3.0 {
		t.Errorf("a.Grad=%f, want 3", a.Grad)
	}
}

func TestScalarSigmoidGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	a := NewScalar(0.0)
	c := a.Sigmoid()
	if math.Abs(c.Data-0.5) > 1e-10 {
		t.Fatalf("Sigmoid(0)=%f, want 0.5", c.Data)
	}
	c.Grad = 1.0
	Backward(c)
	// d/dx sigmoid(x) at x=0 = sigma(0)*(1-sigma(0)) = 0.25
	if math.Abs(a.Grad-0.25) > 1e-10 {
		t.Errorf("Sigmoid grad at 0=%f, want 0.25", a.Grad)
	}
}

func TestBackwardChainRule(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	// Test chain: c = (a + b) * a  where a=2, b=3
	// c = (2+3)*2 = 10
	// dc/da = (a+b) + a = 5+2 = 7
	// dc/db = a = 2
	a := NewVec([]float64{2})
	b := NewVec([]float64{3})
	sum := a.Add(b)
	c := sum.MulVec(a)
	c.Grad = []float64{1}
	Backward(c)
	if math.Abs(a.Grad[0]-7.0) > 1e-10 {
		t.Errorf("chain rule: da=%f, want 7", a.Grad[0])
	}
	if math.Abs(b.Grad[0]-2.0) > 1e-10 {
		t.Errorf("chain rule: db=%f, want 2", b.Grad[0])
	}
}

func TestMatvecGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	// 2x3 matrix, 3-vec input
	m := NewMatrixParam(2, 3, 0.0)
	m.Rows[0] = NewVecWithGrad([]float64{1, 0, 0})
	m.Rows[1] = NewVecWithGrad([]float64{0, 1, 0})
	x := NewVec([]float64{3, 7, 11})
	out := m.Matvec(x)
	// out = [3, 7]
	if out.Data[0] != 3.0 || out.Data[1] != 7.0 {
		t.Fatalf("Matvec forward: %v, want [3,7]", out.Data)
	}
	out.Grad = []float64{1, 1}
	Backward(out)
	// d(loss)/d(x) = W^T @ grad_out = [[1,0],[0,1],[0,0]]^T @ [1,1] = [1,1,0]
	if x.Grad[0] != 1.0 || x.Grad[1] != 1.0 || x.Grad[2] != 0.0 {
		t.Errorf("Matvec x.Grad: %v, want [1,1,0]", x.Grad)
	}
	// d(loss)/d(W[0]) = grad_out[0] * x = 1 * [3,7,11] = [3,7,11]
	if m.Rows[0].Grad[0] != 3.0 || m.Rows[0].Grad[1] != 7.0 {
		t.Errorf("Matvec W[0].Grad: %v, want [3,7,11]", m.Rows[0].Grad)
	}
}

func TestNewVecNoGradWhenDisabled(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	v := NewVec([]float64{1, 2, 3})
	if v.Grad != nil {
		t.Error("NewVec should not allocate grad when gradEnabled=false")
	}
}

func TestNewVecWithGradAlwaysAllocates(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	v := NewVecWithGrad([]float64{1, 2, 3})
	if v.Grad == nil {
		t.Error("NewVecWithGrad should always allocate grad")
	}
	if len(v.Grad) != 3 {
		t.Errorf("NewVecWithGrad grad len=%d, want 3", len(v.Grad))
	}
}

// Test loss computation gradient flows through to logits
func TestCrossEntropyLossGrad(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	logits := NewVec([]float64{1.0, 2.0, 3.0})
	loss := CrossEntropyLoss(logits, 2) // target = class 2
	loss.Grad = 1.0
	Backward(loss)

	// softmax probs for [1,2,3]: p = [0.09, 0.245, 0.665]
	// grad for CE: p - one_hot(target) = [0.09, 0.245, -0.335]
	// grad[target] should be negative (push logit up)
	if logits.Grad[2] >= 0 {
		t.Errorf("grad for correct class should be negative (push logit up), got %f", logits.Grad[2])
	}
	// grad for wrong classes should be positive (push logits down)
	if logits.Grad[0] <= 0 || logits.Grad[1] <= 0 {
		t.Errorf("grad for wrong classes should be positive, got [%f, %f]", logits.Grad[0], logits.Grad[1])
	}
	// Sum of softmax grads = 0
	gradSum := logits.Grad[0] + logits.Grad[1] + logits.Grad[2]
	if math.Abs(gradSum) > 1e-6 {
		t.Errorf("CE grad sum should be ~0, got %f", gradSum)
	}
}

// ============================================================
// TOKENIZER BPE
// ============================================================

func TestNewEvolvingTokenizerBaseVocab(t *testing.T) {
	tok := NewEvolvingTokenizer([]string{"hello"})
	// 256 byte tokens + BOS + EOS + PAD = 259
	if tok.VocabSize != 259 {
		t.Errorf("base vocab size=%d, want 259", tok.VocabSize)
	}
	if tok.Stoi["<BOS>"] != 256 {
		t.Errorf("BOS id=%d, want 256", tok.Stoi["<BOS>"])
	}
	if tok.Stoi["<EOS>"] != 257 {
		t.Errorf("EOS id=%d, want 257", tok.Stoi["<EOS>"])
	}
	if tok.Stoi["<PAD>"] != 258 {
		t.Errorf("PAD id=%d, want 258", tok.Stoi["<PAD>"])
	}
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	tok := NewEvolvingTokenizer([]string{"test"})
	text := "Hello, World!"
	ids := tok.Encode(text)
	decoded := tok.Decode(ids)
	if decoded != text {
		t.Errorf("roundtrip failed: encoded %q, decoded %q", text, decoded)
	}
}

func TestEncodeHasBOSEOS(t *testing.T) {
	tok := NewEvolvingTokenizer([]string{"test"})
	ids := tok.Encode("hi")
	if ids[0] != tok.Stoi["<BOS>"] {
		t.Errorf("first token should be BOS, got %d", ids[0])
	}
	if ids[len(ids)-1] != tok.Stoi["<EOS>"] {
		t.Errorf("last token should be EOS, got %d", ids[len(ids)-1])
	}
}

func TestEncodeEmptyString(t *testing.T) {
	tok := NewEvolvingTokenizer([]string{"test"})
	ids := tok.Encode("")
	// Should just be BOS + EOS
	if len(ids) != 2 {
		t.Errorf("empty string should encode to [BOS, EOS], got len=%d", len(ids))
	}
}

func TestTokenToBytes(t *testing.T) {
	tests := []struct {
		tok  string
		want []byte
	}{
		{"0x48", []byte{0x48}},       // 'H'
		{"0x65", []byte{0x65}},       // 'e'
		{"0x48+0x65", []byte{0x48, 0x65}}, // "He"
		{"<BOS>", nil},                // special token
	}
	for _, tt := range tests {
		got := tokenToBytes(tt.tok)
		if tt.want == nil && got != nil {
			t.Errorf("tokenToBytes(%q)=%v, want nil", tt.tok, got)
		} else if tt.want != nil {
			if len(got) != len(tt.want) {
				t.Errorf("tokenToBytes(%q) len=%d, want %d", tt.tok, len(got), len(tt.want))
			} else {
				for i := range got {
					if got[i] != tt.want[i] {
						t.Errorf("tokenToBytes(%q)[%d]=%d, want %d", tt.tok, i, got[i], tt.want[i])
					}
				}
			}
		}
	}
}

func TestUnicodeSegment(t *testing.T) {
	segs := unicodeSegment("Hello 42!")
	// "Hello" = letters, " " = space, "42" = digits, "!" = punctuation
	if len(segs) != 4 {
		t.Fatalf("unicodeSegment(\"Hello 42!\") = %d segments, want 4", len(segs))
	}
	if string(segs[0]) != "Hello" {
		t.Errorf("seg[0]=%q, want \"Hello\"", string(segs[0]))
	}
	if string(segs[1]) != " " {
		t.Errorf("seg[1]=%q, want \" \"", string(segs[1]))
	}
	if string(segs[2]) != "42" {
		t.Errorf("seg[2]=%q, want \"42\"", string(segs[2]))
	}
	if string(segs[3]) != "!" {
		t.Errorf("seg[3]=%q, want \"!\"", string(segs[3]))
	}
}

func TestUnicodeSegmentEmpty(t *testing.T) {
	segs := unicodeSegment("")
	if segs != nil {
		t.Errorf("unicodeSegment(\"\") should return nil, got %v", segs)
	}
}

func TestUnicodeSegmentUTF8(t *testing.T) {
	segs := unicodeSegment("Привет")
	// All letters — should be one segment
	if len(segs) != 1 {
		t.Errorf("unicodeSegment(\"Привет\") = %d segments, want 1", len(segs))
	}
	if string(segs[0]) != "Привет" {
		t.Errorf("seg[0]=%q, want \"Привет\"", string(segs[0]))
	}
}

func TestTrainBPEMergesTokens(t *testing.T) {
	docs := []string{"aaaa bbbb aaaa bbbb aaaa"}
	tok := NewEvolvingTokenizer(docs)
	origVocab := tok.VocabSize
	tok.TrainBPE(docs, 5)

	if tok.VocabSize <= origVocab {
		t.Errorf("BPE should add tokens: vocab was %d, now %d", origVocab, tok.VocabSize)
	}
	if len(tok.Merges) == 0 {
		t.Error("BPE should produce merges")
	}
}

func TestBPEEncodeDecodeRoundTrip(t *testing.T) {
	docs := []string{"the cat sat on the mat the cat sat on the mat"}
	tok := NewEvolvingTokenizer(docs)
	tok.TrainBPE(docs, 20)
	tok.BPEEnabled = true

	text := "the cat"
	ids := tok.Encode(text)
	decoded := tok.Decode(ids)
	if decoded != text {
		t.Errorf("BPE roundtrip failed: %q -> %v -> %q", text, ids, decoded)
	}
}

func TestBPECompressionRatio(t *testing.T) {
	text := "abcabc abcabc abcabc abcabc"
	docs := []string{text}
	tok := NewEvolvingTokenizer(docs)

	// Without BPE: each byte = 1 token (plus BOS+EOS)
	idsNoBPE := tok.Encode(text)
	noBPELen := len(idsNoBPE)

	// Train BPE and re-encode
	tok.TrainBPE(docs, 30)
	tok.BPEEnabled = true
	idsWithBPE := tok.Encode(text)
	withBPELen := len(idsWithBPE)

	if withBPELen >= noBPELen {
		t.Errorf("BPE should compress: without=%d, with=%d", noBPELen, withBPELen)
	}
}

func TestMaybeEnableBPE(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.EnableBPEAfterChars = 100
	CFG.BPENumMerges = 10

	smallDoc := "short"
	tok := NewEvolvingTokenizer([]string{smallDoc})
	enabled := tok.MaybeEnableBPE([]string{smallDoc})
	if enabled {
		t.Error("should not enable BPE with small corpus")
	}

	bigDoc := make([]byte, 200)
	for i := range bigDoc {
		bigDoc[i] = 'a' + byte(i%26)
	}
	tok2 := NewEvolvingTokenizer([]string{string(bigDoc)})
	enabled2 := tok2.MaybeEnableBPE([]string{string(bigDoc)})
	if !enabled2 {
		t.Error("should enable BPE with large corpus")
	}
	if !tok2.BPEEnabled {
		t.Error("BPEEnabled should be true after MaybeEnableBPE returns true")
	}
}

// ============================================================
// SAMPLING
// ============================================================

func TestSoftmaxProbsSumToOne(t *testing.T) {
	probs := SoftmaxProbs([]float64{1, 2, 3, 4})
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("softmax probs sum=%f, want 1.0", sum)
	}
}

func TestSoftmaxProbsMonotone(t *testing.T) {
	probs := SoftmaxProbs([]float64{1, 2, 3})
	if probs[0] >= probs[1] || probs[1] >= probs[2] {
		t.Errorf("softmax should be monotone increasing: %v", probs)
	}
}

func TestSoftmaxProbsNumericalStability(t *testing.T) {
	// Very large logits should not cause NaN/Inf
	probs := SoftmaxProbs([]float64{1000, 1001, 1002})
	for i, p := range probs {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Errorf("softmax[%d] = %f with large logits", i, p)
		}
	}
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax sum with large logits=%f", sum)
	}
}

func TestTopKTopPSampleRespectsTopK(t *testing.T) {
	// Uniform probs, k=1 should always pick the first (highest)
	probs := []float64{0.4, 0.3, 0.2, 0.1}
	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		idx := TopKTopPSample(probs, 1, 1.0, 0.0, 1.0)
		counts[idx]++
	}
	if counts[0] != 100 {
		t.Errorf("top-k=1 should always pick idx 0, got counts: %v", counts)
	}
}

func TestTopKTopPSampleRespectsTopP(t *testing.T) {
	// With p=0.5, only first token (0.6) should be in nucleus
	probs := []float64{0.6, 0.2, 0.1, 0.1}
	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		idx := TopKTopPSample(probs, 0, 0.5, 0.0, 1.0)
		counts[idx]++
	}
	if counts[0] != 100 {
		t.Errorf("top-p=0.5 with probs[0]=0.6 should always pick idx 0, got: %v", counts)
	}
}

func TestTopKTopPSampleRespectsMinP(t *testing.T) {
	// minP=0.5 means only tokens with prob >= 0.5 * max_prob are kept
	probs := []float64{0.8, 0.1, 0.05, 0.05}
	// threshold = 0.5 * 0.8 = 0.4, only probs[0] passes
	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		idx := TopKTopPSample(probs, 0, 1.0, 0.5, 1.0)
		counts[idx]++
	}
	if counts[0] != 100 {
		t.Errorf("minP=0.5 should filter to only idx 0, got: %v", counts)
	}
}

func TestTopKTopPSampleValidIndex(t *testing.T) {
	probs := SoftmaxProbs([]float64{1, 2, 3, 4, 5})
	for i := 0; i < 50; i++ {
		idx := TopKTopPSample(probs, 3, 0.9, 0.05, 0.95)
		if idx < 0 || idx >= len(probs) {
			t.Fatalf("sample returned invalid index %d for len=%d", idx, len(probs))
		}
	}
}

func TestTopKTopPSampleZeroProbs(t *testing.T) {
	probs := []float64{0, 0, 0, 0}
	// Should not panic, return some valid index
	idx := TopKTopPSample(probs, 0, 1.0, 0.0, 1.0)
	if idx < 0 || idx >= len(probs) {
		t.Fatalf("zero probs: invalid index %d", idx)
	}
}

func TestClipParams(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	v := NewVec([]float64{1, 2})
	v.Grad = []float64{5.0, -3.0}
	ClipParams([]*Vec{v}, 2.0)
	if v.Grad[0] != 2.0 {
		t.Errorf("grad[0] should be clipped to 2.0, got %f", v.Grad[0])
	}
	if v.Grad[1] != -2.0 {
		t.Errorf("grad[1] should be clipped to -2.0, got %f", v.Grad[1])
	}
}

func TestClipParamsNoop(t *testing.T) {
	gradEnabled.Store(true)
	defer gradEnabled.Store(false)

	v := NewVec([]float64{1})
	v.Grad = []float64{0.5}
	ClipParams([]*Vec{v}, 1.0)
	if v.Grad[0] != 0.5 {
		t.Errorf("grad should not be clipped when within range: %f", v.Grad[0])
	}
}

// ============================================================
// SYNTROPY TRACKER
// ============================================================

func TestNewSyntropyTracker(t *testing.T) {
	st := NewSyntropyTracker()
	if st.LastAction != "none" {
		t.Errorf("initial action=%q, want \"none\"", st.LastAction)
	}
	if len(st.BurstHistory) != 0 {
		t.Errorf("initial BurstHistory len=%d, want 0", len(st.BurstHistory))
	}
}

func TestSyntropyTrackerRecordBurst(t *testing.T) {
	st := NewSyntropyTracker()
	st.RecordBurst("boost", 3.0, 2.5)
	st.RecordBurst("dampen", 2.5, 2.3)
	if len(st.BurstHistory) != 2 {
		t.Errorf("BurstHistory len=%d, want 2", len(st.BurstHistory))
	}
	if st.BurstHistory[0].Action != "boost" {
		t.Errorf("burst[0].Action=%q, want \"boost\"", st.BurstHistory[0].Action)
	}
}

func TestSyntropyTrackerBurstHistoryCap(t *testing.T) {
	st := NewSyntropyTracker()
	for i := 0; i < 20; i++ {
		st.RecordBurst("test", float64(i), float64(i-1))
	}
	if len(st.BurstHistory) != 16 {
		t.Errorf("BurstHistory should cap at 16, got %d", len(st.BurstHistory))
	}
}

func TestSyntropyTrackerActionEffectiveness(t *testing.T) {
	st := NewSyntropyTracker()
	st.RecordBurst("boost", 3.0, 2.0) // delta = -1.0
	st.RecordBurst("boost", 2.0, 1.5) // delta = -0.5
	st.RecordBurst("dampen", 1.5, 1.6) // delta = +0.1

	mean, count := st.ActionEffectiveness("boost")
	if count != 2 {
		t.Errorf("boost count=%d, want 2", count)
	}
	// mean = (-1.0 + -0.5) / 2 = -0.75
	if math.Abs(mean-(-0.75)) > 1e-10 {
		t.Errorf("boost mean delta=%f, want -0.75", mean)
	}

	mean2, count2 := st.ActionEffectiveness("dampen")
	if count2 != 1 || math.Abs(mean2-0.1) > 1e-10 {
		t.Errorf("dampen: mean=%f count=%d, want 0.1/1", mean2, count2)
	}

	_, count3 := st.ActionEffectiveness("unknown")
	if count3 != 0 {
		t.Errorf("unknown action count=%d, want 0", count3)
	}
}

func TestSyntropyDecideActionSteady(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	st := NewSyntropyTracker()
	st.SyntropyTrend = 0.0 // flat
	st.FieldDeviation = 1.0 // in sweet spot
	st.PurposeAlignment = 0.0

	d := st.DecideAction()
	if d.Action != "steady" {
		t.Errorf("expected steady, got %q", d.Action)
	}
	if d.LRMultiplier != 1.0 {
		t.Errorf("steady LR multiplier=%f, want 1.0", d.LRMultiplier)
	}
}

func TestSyntropyDecideActionDampen(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	st := NewSyntropyTracker()
	st.SyntropyTrend = -0.05 // dissolving
	st.FieldDeviation = 1.0
	st.PurposeAlignment = 0.0

	d := st.DecideAction()
	if d.Action != "dampen" {
		t.Errorf("expected dampen, got %q", d.Action)
	}
	if d.LRMultiplier >= 1.0 {
		t.Errorf("dampen should reduce LR, got %f", d.LRMultiplier)
	}
}

func TestSyntropyDecideActionGround(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	st := NewSyntropyTracker()
	st.SyntropyTrend = 0.0
	st.FieldDeviation = 100.0 // way too high
	st.PurposeAlignment = 0.0

	d := st.DecideAction()
	if d.Action != "ground" {
		t.Errorf("expected ground (high deviation), got %q", d.Action)
	}
}

func TestSyntropyDecideActionExplore(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	st := NewSyntropyTracker()
	st.SyntropyTrend = 0.0
	st.FieldDeviation = 0.001 // too low = parroting
	st.PurposeAlignment = 0.0

	d := st.DecideAction()
	if d.Action != "explore" {
		t.Errorf("expected explore (low deviation), got %q", d.Action)
	}
}

func TestSyntropyDecideActionRealign(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	st := NewSyntropyTracker()
	st.SyntropyTrend = 0.0
	st.FieldDeviation = 1.0
	st.PurposeAlignment = -0.5 // purpose opposes gamma

	d := st.DecideAction()
	if d.Action != "realign" {
		t.Errorf("expected realign (negative purpose alignment), got %q", d.Action)
	}
	if d.LRMultiplier >= 1.0 {
		t.Errorf("realign should halve LR, got %f", d.LRMultiplier)
	}
}

func TestSyntropyDecideActionSelfMetaLearning(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	st := NewSyntropyTracker()
	// Record history showing "boost" consistently makes things worse
	for i := 0; i < 4; i++ {
		st.RecordBurst("boost", 2.0, 2.2) // loss went UP
	}

	// Set up conditions for "boost"
	st.SyntropyTrend = 0.05 // rising
	st.FieldDeviation = 1.0 // sweet spot
	st.PurposeAlignment = 0.1 // not enough for amplify

	d := st.DecideAction()
	// Self-meta-learning should downgrade "boost" to "steady" since it historically hurts
	if d.Action != "steady" {
		t.Errorf("self-meta-learning should downgrade boost to steady, got %q", d.Action)
	}
}

func TestSyntropyIsSustainedOverload(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.SyntropyWindow = 8
	CFG.EntropyHigh = 1.5

	st := NewSyntropyTracker()
	// Not enough history
	if st.isSustainedOverload() {
		t.Error("should not be overloaded with no history")
	}

	// Fill with high entropy
	for i := 0; i < 8; i++ {
		st.EntropyHistory = append(st.EntropyHistory, 2.0) // all above EntropyHigh
	}
	st.SyntropyTrend = -0.05 // dissolving

	if !st.isSustainedOverload() {
		t.Error("should detect sustained overload")
	}
}

func TestSyntropyShouldHibernateNoPeers(t *testing.T) {
	st := NewSyntropyTracker()
	// No SwarmInfo
	if st.shouldHibernate() {
		t.Error("should not hibernate without peers")
	}
}

// ============================================================
// QUANTUM BUFFER
// ============================================================

func TestNewQuantumBuffer(t *testing.T) {
	qb := NewQuantumBuffer()
	if qb.AccumulatedBytes != 0 {
		t.Errorf("initial AccumulatedBytes=%d, want 0", qb.AccumulatedBytes)
	}
	if qb.TotalTokens != 0 {
		t.Errorf("initial TotalTokens=%d, want 0", qb.TotalTokens)
	}
}

func TestQuantumBufferFeed(t *testing.T) {
	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	qb.Feed("hello world", tok)
	if qb.AccumulatedBytes != 11 {
		t.Errorf("AccumulatedBytes=%d, want 11", qb.AccumulatedBytes)
	}
	if qb.TotalTokens == 0 {
		t.Error("TotalTokens should be > 0 after Feed")
	}
	if len(qb.UniqueTokens) == 0 {
		t.Error("UniqueTokens should be > 0 after Feed")
	}
}

func TestQuantumBufferNovelty(t *testing.T) {
	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()

	// All unique text -> high novelty
	qb.Feed("abcdefghij", tok)
	bytes1, novelty1 := qb.SnapshotStats()
	if bytes1 != 10 {
		t.Errorf("bytes=%d, want 10", bytes1)
	}
	if novelty1 < 0.5 {
		t.Errorf("novelty should be high for unique text, got %f", novelty1)
	}

	// Repeated text -> lower novelty
	qb2 := NewQuantumBuffer()
	qb2.Feed("aaaaaaaaaa", tok)
	_, novelty2 := qb2.SnapshotStats()
	if novelty2 >= novelty1 {
		t.Errorf("repeated text should have lower novelty: %f >= %f", novelty2, novelty1)
	}
}

func TestQuantumBufferShouldTrigger(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.QBMinBytes = 100
	CFG.QBMinNovelty = 0.99
	CFG.QBCooldownSeconds = 0.0 // no cooldown for testing

	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	qb.Feed("aaaa", tok)
	if qb.ShouldTrigger() {
		t.Error("should not trigger with < QBMinBytes and low novelty")
	}

	// Feed enough bytes
	bigText := make([]byte, 200)
	for i := range bigText {
		bigText[i] = byte('a' + i%26)
	}
	qb2 := NewQuantumBuffer()
	qb2.Feed(string(bigText), tok)
	if !qb2.ShouldTrigger() {
		t.Error("should trigger with enough bytes")
	}
}

func TestQuantumBufferReset(t *testing.T) {
	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	qb.Feed("hello world this is a test", tok)
	if qb.AccumulatedBytes == 0 {
		t.Fatal("should have bytes before reset")
	}
	qb.Reset()
	if qb.AccumulatedBytes != 0 {
		t.Errorf("AccumulatedBytes=%d after reset, want 0", qb.AccumulatedBytes)
	}
	if qb.TotalTokens != 0 {
		t.Errorf("TotalTokens=%d after reset, want 0", qb.TotalTokens)
	}
	if len(qb.UniqueTokens) != 0 {
		t.Errorf("UniqueTokens len=%d after reset, want 0", len(qb.UniqueTokens))
	}
}

func TestQuantumBufferCooldown(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.QBMinBytes = 10
	CFG.QBCooldownSeconds = 9999.0 // very long cooldown

	tok := NewEvolvingTokenizer([]string{"test"})
	qb := NewQuantumBuffer()
	qb.Reset() // sets LastBurstTime to now
	qb.Feed("this is enough bytes for trigger", tok)

	if qb.ShouldTrigger() {
		t.Error("should not trigger during cooldown")
	}
}

// ============================================================
// COOCCUR FIELD
// ============================================================

func TestNewCooccurField(t *testing.T) {
	cf := NewCooccurField()
	if cf.Built {
		t.Error("new CooccurField should not be built")
	}
	if cf.Unigram == nil || cf.BigramByFirst == nil || cf.TrigramByContext == nil {
		t.Error("maps should be initialized")
	}
}

func TestCooccurFieldBuildFromCorpus(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 3

	tok := NewEvolvingTokenizer([]string{"the cat sat"})
	cf := NewCooccurField()
	cf.BuildFromCorpus(tok, []string{"the cat sat"})

	if !cf.Built {
		t.Error("field should be built after BuildFromCorpus")
	}
	if len(cf.Unigram) == 0 {
		t.Error("unigram counts should not be empty")
	}
	if len(cf.BigramByFirst) == 0 {
		t.Error("bigram counts should not be empty")
	}
}

func TestCooccurFieldIngestTokens(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	ids := []int{1, 2, 3, 1, 2}
	cf.IngestTokens(ids)

	// Unigram: 1 appears 2x, 2 appears 2x, 3 appears 1x
	if cf.Unigram[1] != 2 {
		t.Errorf("unigram[1]=%f, want 2", cf.Unigram[1])
	}
	if cf.Unigram[3] != 1 {
		t.Errorf("unigram[3]=%f, want 1", cf.Unigram[3])
	}

	// Bigram: (1,2) appears 2x
	if cf.BigramByFirst[1] == nil || cf.BigramByFirst[1][2] != 2 {
		t.Errorf("bigram[1][2] should be 2")
	}
}

func TestCooccurFieldIngestTokensWeighted(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	ids := []int{10, 20}
	cf.IngestTokensWeighted(ids, 3.0)

	if cf.Unigram[10] != 3.0 {
		t.Errorf("weighted unigram[10]=%f, want 3", cf.Unigram[10])
	}
	if cf.BigramByFirst[10][20] != 3.0 {
		t.Errorf("weighted bigram[10][20]=%f, want 3", cf.BigramByFirst[10][20])
	}
}

func TestCooccurFieldAbsorbUserWords(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.UserBoostStrength = 1.0
	CFG.UserBoostDecay = 0.7

	cf := NewCooccurField()
	cf.AbsorbUserWords([]int{5, 10})

	if cf.UserBoost[5] != 1.0 {
		t.Errorf("UserBoost[5]=%f, want 1.0", cf.UserBoost[5])
	}
	if cf.UserBoost[10] != 1.0 {
		t.Errorf("UserBoost[10]=%f, want 1.0", cf.UserBoost[10])
	}
}

func TestCooccurFieldDecayUserBoost(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.UserBoostDecay = 0.5

	cf := NewCooccurField()
	cf.UserBoost = map[int]float64{
		1: 1.0,
		2: 0.01, // will decay below 0.01 threshold
	}

	cf.DecayUserBoost()

	if math.Abs(cf.UserBoost[1]-0.5) > 1e-10 {
		t.Errorf("UserBoost[1]=%f after decay, want 0.5", cf.UserBoost[1])
	}
	if _, exists := cf.UserBoost[2]; exists {
		t.Error("UserBoost[2] should be deleted (decayed below threshold)")
	}
}

func TestCooccurFieldSampleNextBigram(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	// Set up strong bigram: after token 5, token 10 always follows
	cf.BigramByFirst = map[int]map[int]float64{
		5: {10: 100.0},
	}
	cf.Unigram = map[int]float64{5: 1, 10: 1}

	// With context ending in 5, should strongly prefer 10
	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		next := cf.SampleNext([]int{5}, 20, 0.5)
		counts[next]++
	}
	if counts[10] < 90 {
		t.Errorf("strong bigram should dominate sampling: got token 10 only %d/100 times", counts[10])
	}
}

func TestCooccurFieldSampleNextTrigram(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	// Strong trigram: [3,5] -> 7
	cf.TrigramByContext = map[[2]int]map[int]float64{
		{3, 5}: {7: 100.0},
	}
	cf.Unigram = map[int]float64{3: 1, 5: 1, 7: 1}

	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		next := cf.SampleNext([]int{3, 5}, 20, 0.5)
		counts[next]++
	}
	if counts[7] < 90 {
		t.Errorf("strong trigram should dominate: got token 7 only %d/100 times", counts[7])
	}
}

func TestCooccurFieldSampleNextFourgram(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	cf.FourgramByCtx = map[[3]int]map[int]float64{
		{1, 2, 3}: {4: 100.0},
	}
	cf.Unigram = map[int]float64{1: 1, 2: 1, 3: 1, 4: 1}

	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		next := cf.SampleNext([]int{1, 2, 3}, 20, 0.5)
		counts[next]++
	}
	if counts[4] < 90 {
		t.Errorf("strong fourgram should dominate: got token 4 only %d/100 times", counts[4])
	}
}

func TestCooccurFieldSampleNextFallbackToUnigram(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	cf.Unigram = map[int]float64{0: 100.0} // token 0 dominates

	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		next := cf.SampleNext([]int{99}, 10, 0.5) // no bigrams for 99
		counts[next]++
	}
	if counts[0] < 80 {
		t.Errorf("unigram fallback should prefer token 0: got %d/100", counts[0])
	}
}

func TestCooccurFieldSampleNextUserBoost(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	cf.Unigram = map[int]float64{1: 1.0, 2: 1.0}
	cf.UserBoost = map[int]float64{2: 10.0} // massive boost to token 2

	counts := make(map[int]int)
	for i := 0; i < 200; i++ {
		next := cf.SampleNext([]int{99}, 10, 1.0)
		counts[next]++
	}
	if counts[2] < counts[1] {
		t.Errorf("user boost should favor token 2: got 1=%d, 2=%d", counts[1], counts[2])
	}
}

func TestCooccurFieldSampleNextValidIndex(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.CooccurWindowSize = 2

	cf := NewCooccurField()
	cf.Unigram = map[int]float64{0: 1, 1: 1, 2: 1}
	vocabSize := 5
	for i := 0; i < 50; i++ {
		idx := cf.SampleNext([]int{0}, vocabSize, 1.0)
		if idx < 0 || idx >= vocabSize {
			t.Fatalf("SampleNext returned invalid index %d for vocabSize=%d", idx, vocabSize)
		}
	}
}

// ============================================================
// RoPE (Rotary Position Embedding)
// ============================================================

func TestRoPERotatePreservesNorm(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	// RoPE is a rotation — it should approximately preserve vector norm
	data := []float64{1, 0, 0, 1, 1, 0, 0, 1}
	x := NewVec(data)
	normBefore := 0.0
	for _, v := range x.Data {
		normBefore += v * v
	}

	rotated := RoPERotate(x, 5, 8)
	normAfter := 0.0
	for _, v := range rotated.Data {
		normAfter += v * v
	}

	if math.Abs(normBefore-normAfter) > 1e-6 {
		t.Errorf("RoPE should preserve norm: before=%f, after=%f", normBefore, normAfter)
	}
}

func TestRoPERotatePosition0IsIdentity(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	// At position 0, cos=1 sin=0, so rotation should be identity
	data := []float64{1, 2, 3, 4}
	x := NewVec(data)
	rotated := RoPERotate(x, 0, 4)

	for i := range data {
		if math.Abs(rotated.Data[i]-data[i]) > 1e-10 {
			t.Errorf("RoPE at pos=0 should be identity: [%d] %f vs %f", i, rotated.Data[i], data[i])
		}
	}
}

func TestRoPERotateDifferentPositions(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	data := []float64{1, 0, 1, 0, 1, 0, 1, 0}
	x := NewVec(data)
	r1 := RoPERotate(x, 1, 8)
	r2 := RoPERotate(x, 2, 8)

	different := false
	for i := range r1.Data {
		if math.Abs(r1.Data[i]-r2.Data[i]) > 1e-10 {
			different = true
			break
		}
	}
	if !different {
		t.Error("RoPE at different positions should produce different vectors")
	}
}

// ============================================================
// DELTA ADAPTER — more thorough tests
// ============================================================

func TestDeltaAdapterApplyZeroIsIdentity(t *testing.T) {
	gradEnabled.Store(false)
	defer gradEnabled.Store(true)

	// Zero-initialized adapter should produce zero output
	da := NewDeltaAdapter(4, 3, 2, 0.0)
	x := NewVec([]float64{1, 2, 3})
	out := da.Apply(x)
	for i, v := range out.Data {
		if v != 0 {
			t.Errorf("zero adapter output[%d]=%f, want 0", i, v)
		}
	}
}

func TestDeltaAdapterDimensions(t *testing.T) {
	da := NewDeltaAdapter(8, 4, 2, 0.08)
	if da.A.Nout != 8 || da.A.Nin != 2 {
		t.Errorf("A dims: %dx%d, want 8x2", da.A.Nout, da.A.Nin)
	}
	if da.B.Nout != 2 || da.B.Nin != 4 {
		t.Errorf("B dims: %dx%d, want 2x4", da.B.Nout, da.B.Nin)
	}
}

func TestDeltaAdapterParams(t *testing.T) {
	da := NewDeltaAdapter(4, 3, 2, 0.08)
	params := da.Params()
	// A has 4 rows, B has 2 rows = 6 total Vec params
	if len(params) != 6 {
		t.Errorf("Params len=%d, want 6", len(params))
	}
}

func TestDeltaAdapterMaybeGrowOut(t *testing.T) {
	da := NewDeltaAdapter(4, 3, 2, 0.08)
	da.MaybeGrowOut(6)
	if da.A.Nout != 6 {
		t.Errorf("A.Nout=%d after MaybeGrowOut(6), want 6", da.A.Nout)
	}
	// B should not change
	if da.B.Nout != 2 {
		t.Errorf("B.Nout should stay 2, got %d", da.B.Nout)
	}
}

// ============================================================
// CORPUS UTILITIES
// ============================================================

func TestNormalizeText(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"  hello  world  ", "hello world"},
		{"line1\nline2", "line1 line2"},
		{"tabs\there", "tabs here"},
		{"", ""},
	}
	for _, tt := range tests {
		got := normalizeText(tt.in)
		if got != tt.want {
			t.Errorf("normalizeText(%q)=%q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestLoadCorpusLines(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "corpus.txt")
	os.WriteFile(tmpFile, []byte("line1\nline2\nline3\n"), 0644)

	lines := loadCorpusLines(tmpFile)
	if len(lines) != 3 {
		t.Errorf("loadCorpusLines: got %d lines, want 3", len(lines))
	}
	if lines[0] != "line1" || lines[2] != "line3" {
		t.Errorf("loadCorpusLines: wrong content: %v", lines)
	}
}

func TestLoadCorpusLinesNonexistent(t *testing.T) {
	lines := loadCorpusLines("/nonexistent/path/file.txt")
	if lines != nil {
		t.Errorf("nonexistent file should return nil, got %v", lines)
	}
}

func TestSaveCorpusLines(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "out.txt")
	lines := []string{"alpha", "beta", "gamma"}
	saveCorpusLines(tmpFile, lines)

	// Read back
	loaded := loadCorpusLines(tmpFile)
	if len(loaded) != 3 {
		t.Fatalf("saved %d lines, loaded %d", len(lines), len(loaded))
	}
	for i := range lines {
		if loaded[i] != lines[i] {
			t.Errorf("line[%d]: got %q, want %q", i, loaded[i], lines[i])
		}
	}
}

func TestReservoirMixKeep(t *testing.T) {
	existing := []string{"a", "b", "c"}
	newSents := []string{"d", "e"}
	result := reservoirMixKeep(existing, newSents, 5)

	if len(result) > 5 {
		t.Errorf("reservoir should cap at maxLines=5, got %d", len(result))
	}
	if len(result) < 3 {
		t.Errorf("should have at least existing lines, got %d", len(result))
	}
	// All existing lines should be preserved when room allows
	for _, e := range existing {
		found := false
		for _, r := range result {
			if r == e {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("existing line %q should be preserved when maxLines > existing", e)
		}
	}
}

func TestReservoirMixKeepSmallMax(t *testing.T) {
	existing := []string{"a", "b", "c", "d", "e"}
	newSents := []string{"x"}
	result := reservoirMixKeep(existing, newSents, 3)

	if len(result) != 3 {
		t.Errorf("reservoir should cap at maxLines=3, got %d", len(result))
	}
}

// ============================================================
// HELPERS
// ============================================================

func TestSliceEqual(t *testing.T) {
	if !sliceEqual([]int{1, 2, 3}, []int{1, 2, 3}) {
		t.Error("equal slices should be equal")
	}
	if sliceEqual([]int{1, 2}, []int{1, 3}) {
		t.Error("different slices should not be equal")
	}
	if sliceEqual([]int{1}, []int{1, 2}) {
		t.Error("different length slices should not be equal")
	}
	if sliceEqual(nil, nil) != true {
		t.Error("nil slices should be equal")
	}
}

func TestIntPtr(t *testing.T) {
	p := intPtr(42)
	if *p != 42 {
		t.Errorf("intPtr(42)=%d, want 42", *p)
	}
}

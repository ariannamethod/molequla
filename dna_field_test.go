package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The DNA tree is a field, not a queue: every organism must be able to eat
// every fragment a sibling wrote. Before repair 3 dnaRead removed a fragment
// after the first reader appended it, so the second reader found an empty
// directory and the organism that lost the scan stopped growing
// (MOLEQULALOG2.md, 2026-09-13, "DNA is a race").
//
// Layout under a temp root, mirroring launcher.sh: <root>/<element>/ is each
// organism's working directory, <root>/dna/output/<element>/ its emissions.
func dnaTestTree(t *testing.T, sources ...string) (root string, restore func()) {
	t.Helper()
	root = t.TempDir()
	for _, e := range append([]string{"earth", "air", "water"}, sources...) {
		if err := os.MkdirAll(filepath.Join(root, e), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(root, "dna", "output", e), 0755); err != nil {
			t.Fatal(err)
		}
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return root, func() { os.Chdir(cwd) }
}

func dnaWriteFragment(t *testing.T, root, element, name, text string) {
	t.Helper()
	p := filepath.Join(root, "dna", "output", element, name)
	if err := os.WriteFile(p, []byte(text+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
}

func dnaCountFiles(t *testing.T, root, element string) int {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(root, "dna", "output", element))
	if err != nil {
		t.Fatal(err)
	}
	return len(entries)
}

// enterOrganism chdirs into <root>/<element>, returns its corpus path and a
// cursor loaded from its working directory (a restart re-reads the same file).
func enterOrganism(t *testing.T, root, element string) (corpus string, cur *dnaCursor) {
	t.Helper()
	if err := os.Chdir(filepath.Join(root, element)); err != nil {
		t.Fatal(err)
	}
	corpus = filepath.Join(root, element, "corpus.txt")
	if _, err := os.Stat(corpus); err != nil {
		os.WriteFile(corpus, nil, 0644)
	}
	return corpus, loadDNACursor(filepath.Join(root, element, dnaCursorFile))
}

var dnaTestFrags = []string{
	"the river follows the steepest available gradient through the weakest material",
	"heat arrives from outside and the change comes from within",
	"both find the path of least resistance and both shape the landscape",
}

func TestDNAFieldEveryReaderEatsEveryFragment(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.DNAMinFragmentBytes = 5
	CFG.DNAMaxReadsPerTick = 8

	root, restore := dnaTestTree(t)
	defer restore()

	total := 0
	for i, f := range dnaTestFrags {
		dnaWriteFragment(t, root, "earth", "gen_100_"+string(rune('1'+i))+".txt", f)
		total += len(f)
	}

	// Reader 1: air.
	airCorpus, airCur := enterOrganism(t, root, "air")
	if got := dnaRead("air", airCorpus, nil, nil, airCur); got != total {
		t.Fatalf("air consumed %d bytes, want %d", got, total)
	}
	// The fragments must still be there for the next reader.
	if n := dnaCountFiles(t, root, "earth"); n != len(dnaTestFrags) {
		t.Fatalf("earth's fragments after the first reader: %d files, want %d — the reader consumed the field", n, len(dnaTestFrags))
	}

	// Reader 2: water.
	waterCorpus, waterCur := enterOrganism(t, root, "water")
	if got := dnaRead("water", waterCorpus, nil, nil, waterCur); got != total {
		t.Fatalf("water consumed %d bytes, want %d — the second reader lost the race", got, total)
	}
	// A second pass by the same reader must not eat the same fragments twice.
	if got := dnaRead("water", waterCorpus, nil, nil, waterCur); got != 0 {
		t.Fatalf("water consumed %d bytes on a second pass, want 0", got)
	}
	// A restart (fresh cursor loaded from disk) must not re-eat either.
	_, waterCur2 := enterOrganism(t, root, "water")
	if got := dnaRead("water", waterCorpus, nil, nil, waterCur2); got != 0 {
		t.Fatalf("water consumed %d bytes after a restart, want 0 — the cursor did not persist", got)
	}
	// A new fragment after the cursor is eaten exactly once.
	dnaWriteFragment(t, root, "earth", "gen_101_1.txt", dnaTestFrags[0])
	if got := dnaRead("water", waterCorpus, nil, nil, waterCur2); got != len(dnaTestFrags[0]) {
		t.Fatalf("water consumed %d bytes of the new fragment, want %d", got, len(dnaTestFrags[0]))
	}
}

func TestDNAFieldReadsAreCappedPerTick(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.DNAMinFragmentBytes = 5
	CFG.DNAMaxReadsPerTick = 2

	root, restore := dnaTestTree(t)
	defer restore()
	for i, f := range dnaTestFrags {
		dnaWriteFragment(t, root, "earth", "gen_100_"+string(rune('1'+i))+".txt", f)
	}
	corpus, cur := enterOrganism(t, root, "air")
	first := dnaRead("air", corpus, nil, nil, cur)
	second := dnaRead("air", corpus, nil, nil, cur)
	third := dnaRead("air", corpus, nil, nil, cur)
	want1 := len(dnaTestFrags[0]) + len(dnaTestFrags[1])
	if first != want1 || second != len(dnaTestFrags[2]) || third != 0 {
		t.Fatalf("capped reads: got %d / %d / %d bytes, want %d / %d / 0", first, second, third, want1, len(dnaTestFrags[2]))
	}
}

func TestDNAFieldExtraSourceIsFood(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.DNAMinFragmentBytes = 5
	CFG.DNAMaxReadsPerTick = 8
	CFG.DNAExtraSources = []string{"world"}

	root, restore := dnaTestTree(t, "world")
	defer restore()
	obs := "a blurry photo shows a desk with a chair, a laptop, and a green object"
	dnaWriteFragment(t, root, "world", "gen_200_1.txt", obs)

	corpus, cur := enterOrganism(t, root, "air")
	if got := dnaRead("air", corpus, nil, nil, cur); got != len(obs) {
		t.Fatalf("air consumed %d bytes from world, want %d", got, len(obs))
	}
	if n := dnaCountFiles(t, root, "world"); n != 1 {
		t.Fatalf("world's fragment was removed by a reader: %d files, want 1", n)
	}
	// cross-graze sees the same sources.
	cf := NewCrossField("air", "../dna/output")
	found := false
	for _, s := range cf.Siblings {
		if s == "world" {
			found = true
		}
	}
	if !found {
		t.Fatalf("cross-graze siblings %v do not include the extra source", cf.Siblings)
	}
}

func TestDNAFieldWriterPrunesItsOwnByAge(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()

	root, restore := dnaTestTree(t)
	defer restore()
	now := time.Now().Unix()
	dnaWriteFragment(t, root, "earth", "gen_"+itoa64(now-7200)+"_1.txt", dnaTestFrags[0]) // two hours old
	dnaWriteFragment(t, root, "earth", "gen_"+itoa64(now-10)+"_2.txt", dnaTestFrags[1])   // fresh
	if err := os.Chdir(filepath.Join(root, "earth")); err != nil {
		t.Fatal(err)
	}
	if removed := dnaPruneOwn("earth", 30*time.Minute); removed != 1 {
		t.Fatalf("pruned %d fragments, want 1", removed)
	}
	if n := dnaCountFiles(t, root, "earth"); n != 1 {
		t.Fatalf("%d fragments left, want 1", n)
	}
	// Order of names is numeric, not lexical: step 10 comes after step 9.
	if !dnaNewer("gen_100_10.txt", "gen_100_9.txt") || dnaNewer("gen_100_9.txt", "gen_100_10.txt") {
		t.Fatal("fragment order must compare <unix>,<step> numerically")
	}
}

func itoa64(v int64) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

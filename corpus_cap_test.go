package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// In --evolution nothing writes the messages table (REPL only), and before
// repair 5 updateReservoirCorpus returned before touching the file whenever
// that table was empty — so the corpus dnaRead appends to grew without bound
// (26× in twenty minutes on the pod, 3.2× in ten on phone-1). The cap must
// hold on line count and on bytes, with no REPL traffic at all.
func corpusCapDB(t *testing.T) (dbPath string) {
	t.Helper()
	return filepath.Join(t.TempDir(), "memory.sqlite3")
}

func writeCorpus(t *testing.T, path string, lines []string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
}

func fileBytes(t *testing.T, path string) int64 {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return fi.Size()
}

func TestCorpusCapHoldsWithoutMessages(t *testing.T) {
	saved := CFG
	defer func() { CFG = saved }()
	CFG.MaxLineChars = 240
	db, err := initDB(corpusCapDB(t))
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	defer db.Close()
	corpus := filepath.Join(t.TempDir(), "corpus.txt")

	// Too many lines: trimmed to maxLines, and a second pass changes nothing.
	var many []string
	for i := 0; i < 120; i++ {
		many = append(many, "line number "+itoa64(int64(i))+" of a corpus that keeps growing")
	}
	writeCorpus(t, corpus, many)
	updateReservoirCorpus(db, corpus, 50)
	if n := len(loadCorpusLines(corpus)); n > 50 {
		t.Fatalf("corpus holds %d lines after the periodic trim, cap is 50 — the cap depends on REPL messages", n)
	}
	b1 := fileBytes(t, corpus)
	updateReservoirCorpus(db, corpus, 50)
	if b2 := fileBytes(t, corpus); b2 != b1 {
		t.Fatalf("a corpus already under the cap was rewritten (%d -> %d bytes)", b1, b2)
	}

	// Under the line cap but over the byte cap: DNA fragments are single
	// 5 KB lines, so ten of them are 50 KB against a 50 × 240 = 12 KB budget.
	var fat []string
	for i := 0; i < 10; i++ {
		fat = append(fat, strings.Repeat("dna fragment text ", 300)) // 5400 chars each
	}
	writeCorpus(t, corpus, fat)
	updateReservoirCorpus(db, corpus, 50)
	if b := fileBytes(t, corpus); b > 50*240 {
		t.Fatalf("corpus is %d bytes after the trim, byte cap is %d — lines are not bounded by MaxLineChars on disk", b, 50*240)
	}

	// Under both caps: the file is left alone byte for byte.
	small := []string{"one small line", "another small line", "a third"}
	writeCorpus(t, corpus, small)
	before := fileBytes(t, corpus)
	updateReservoirCorpus(db, corpus, 50)
	if after := fileBytes(t, corpus); after != before {
		t.Fatalf("a corpus under both caps was rewritten (%d -> %d bytes)", before, after)
	}
}

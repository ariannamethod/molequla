package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Routing repair 3. dnaWrite pads a fragment toward CFG.DNAFragmentTargetBytes
// (5000) and dnaRead appended it as one line, while loadCorpusLines truncates
// every line at CFG.MaxLineChars (240) on every read — so the field, the
// overlay and both trainers saw the first 240 bytes of an eaten fragment and
// nothing else. Measured on the live run before this repair: 4.7-4.8 % of the
// bytes of a 5 KB sibling fragment reached docs, and a place fragment lost the
// last 23.3 %, which is where the clause about having moved lives.

func TestSplitCorpusLineKeepsSentencesWhole(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want []string
	}{
		{"two sentences", "The fog lifted. The phone did not move.",
			[]string{"The fog lifted.", "The phone did not move."}},
		{"a decimal is not a sentence end", "It is 22.5 °C now.",
			[]string{"It is 22.5 °C now."}},
		{"a timestamp is not a sentence end", "Sunset 2026-09-14T18:48. Done.",
			[]string{"Sunset 2026-09-14T18:48.", "Done."}},
		{"a question and an exclamation end one too", "Who? Nobody! Fine.",
			[]string{"Who?", "Nobody!", "Fine."}},
		{"newlines fold into the line", "One.\nTwo.", []string{"One.", "Two."}},
		{"nothing at all", "   \n  ", nil},
	} {
		got := splitCorpusLine(tc.in, CFG.MaxLineChars)
		if len(got) != len(tc.want) {
			t.Fatalf("%s: splitCorpusLine(%q) = %q, want %q", tc.name, tc.in, got, tc.want)
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Fatalf("%s: line %d = %q, want %q", tc.name, i, got[i], tc.want[i])
			}
		}
	}
}

// The bound loadCorpusLines enforces has to hold on the way in as well, or the
// split has only moved the truncation somewhere else.
func TestSplitCorpusLineRespectsTheByteBound(t *testing.T) {
	long := strings.Repeat("unbroken ", 400) // 3600 B, no sentence end at all
	lines := splitCorpusLine(long, CFG.MaxLineChars)
	if len(lines) < 2 {
		t.Fatalf("a 3600 B run with no sentence end became %d line(s)", len(lines))
	}
	total := 0
	for i, ln := range lines {
		if len(ln) > CFG.MaxLineChars {
			t.Fatalf("line %d is %d B, over CFG.MaxLineChars = %d", i, len(ln), CFG.MaxLineChars)
		}
		total += len(ln)
	}
	// Nothing is thrown away but the spaces the cuts fell on.
	if total < len(strings.TrimSpace(long))-len(lines) {
		t.Fatalf("split kept %d B of %d — content was dropped, not cut", total, len(long))
	}
	// A sentence end longer than the bound is still cut on a rune boundary.
	for _, ln := range splitCorpusLine(strings.Repeat("щ", 400)+".", CFG.MaxLineChars) {
		if !strings.HasPrefix(ln, "щ") || strings.Contains(ln, "�") {
			t.Fatalf("a multi-byte run was cut inside a rune: %q", ln)
		}
	}
}

// The gate the audit asked for: a sentence from the tail of a 5 KB fragment has
// to be in the bigram table BuildFromCorpus hands generation. Red before the
// split — the tail never reaches docs, so its characters are not even in the
// tokenizer's vocabulary.
func TestTailOfAnEatenFragmentReachesTheBigramTable(t *testing.T) {
	root, restore := dnaTestTree(t, "world")
	defer restore()

	// A fragment the shape dnaWrite emits: padded past DNAFragmentTargetBytes,
	// one run of prose. The filler uses no z and no q, so the pair below can
	// only have come from the last sentence.
	var b strings.Builder
	for i := 0; b.Len() < CFG.DNAFragmentTargetBytes; i++ {
		fmt.Fprintf(&b, "The water ran over the flat stone number %d and made a sound like rain. ", i)
	}
	const tail = "Then the zq marker settled on the riverbed."
	b.WriteString(tail)
	frag := b.String()
	if len(frag) < CFG.DNAFragmentTargetBytes {
		t.Fatalf("fixture is %d B, want at least %d", len(frag), CFG.DNAFragmentTargetBytes)
	}
	dnaWriteFragment(t, root, "world", "gen_1789337521_2.txt", frag)

	withArgs(t, []string{"--element", "earth", "--dna-extra-sources", "world"}, func() {
		CFG.DNAExtraSources = nil
		parseCLIArgs()
		if err := os.Chdir(filepath.Join(root, "earth")); err != nil {
			t.Fatal(err)
		}
		corpus := filepath.Join(root, "earth", "nonames_earth.txt")
		if err := os.WriteFile(corpus, []byte("The river is a place.\n"), 0644); err != nil {
			t.Fatal(err)
		}
		if added := dnaRead("earth", corpus, nil, nil, &dnaCursor{Last: map[string]string{}}); added == 0 {
			t.Fatal("dnaRead ate nothing")
		}

		docs := loadCorpusLines(corpus)
		tok := NewEvolvingTokenizer(docs)
		cf := NewCooccurField()
		cf.BuildFromCorpus(tok, docs)

		ids := tok.Encode(tail)
		if len(ids) < 3 {
			t.Fatalf("the tail encoded to %d ids", len(ids))
		}
		// Every step of the tail sentence, BOS aside, has to be a transition
		// the field knows.
		for i := 1; i < len(ids)-1; i++ {
			if cf.BigramByFirst[ids[i]][ids[i+1]] == 0 {
				t.Fatalf("bigram %d→%d of the tail sentence is missing from the field: "+
					"docs hold %d lines, %d bytes, the fragment was %d B",
					ids[i], ids[i+1], len(docs), corpusBytes(docs), len(frag))
			}
		}
	})
}

func corpusBytes(docs []string) int {
	n := 0
	for _, d := range docs {
		n += len(d)
	}
	return n
}

// dnaRead appends, and a corpus file whose last line has no newline of its own
// swallowed the first sentence of every fragment eaten after it — seen in the
// live probe of 2026-09-15 as "...microorganism[eye cam0 2026-09-13T22:12:01Z]
// A blurry...". One malformed line per fragment, and the sentence it ruins is
// the one the header is on.
func TestDNAReadDoesNotGlueAFragmentOntoAnUnterminatedCorpus(t *testing.T) {
	root, restore := dnaTestTree(t, "world")
	defer restore()

	const head = "[eye cam0 2026-09-13T22:12:01Z] A blurry kitchen table shows a green bowl."
	dnaWriteFragment(t, root, "world", "gen_1789337521_2.txt", head+" The light is dim.")

	withArgs(t, []string{"--element", "earth", "--dna-extra-sources", "world"}, func() {
		CFG.DNAExtraSources = nil
		parseCLIArgs()
		if err := os.Chdir(filepath.Join(root, "earth")); err != nil {
			t.Fatal(err)
		}
		corpus := filepath.Join(root, "earth", "nonames_earth.txt")
		// No trailing newline — the shape saveCorpusLines does not produce but
		// a hand-edited or truncated corpus does.
		if err := os.WriteFile(corpus, []byte("A handful of healthy soil contains more microorganisms"), 0644); err != nil {
			t.Fatal(err)
		}
		if added := dnaRead("earth", corpus, nil, nil, &dnaCursor{Last: map[string]string{}}); added == 0 {
			t.Fatal("dnaRead ate nothing")
		}
		lines := loadCorpusLines(corpus)
		for _, ln := range lines {
			if ln == head {
				return // the fragment's first sentence is a line of its own
			}
		}
		t.Fatalf("the fragment's first sentence is not a line of its own; loadCorpusLines gave %q", lines)
	})
}

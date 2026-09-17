package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The gates for the one line a resume owes the log. Before them LoadCheckpoint
// printed only on the paths that *refuse* the binary sibling, so a session that
// resumed cleanly and a session that found no checkpoint at all left the same
// mark in stdout — none — and `molequla-run/*/*.stdout` could not be asked which
// file an organism was rebuilt from. The refusal lines are not under test here;
// they have their own gates in checkpoint_gguf_test.go and must keep firing
// beside the resume line, which is the third case below.

// ckptResumeLine is the format the whole toolchain agrees on: `[ckpt] resumed
// from <path> — <size> MB, read in <n> ms` with an optional `, peak +<n> MB`.
// phone1/daily.sh parses the same three numbers out of it, so a change here is
// a change there.
var ckptResumeLine = regexp.MustCompile(
	`\[ckpt\] resumed from (\S+) — ([0-9.]+) MB, read in ([0-9]+) ms`)

// resumeSaid returns the captured resume line's path, or "" when none was said.
func resumeSaid(t *testing.T, out string) string {
	t.Helper()
	m := ckptResumeLine.FindStringSubmatch(out)
	if m == nil {
		return ""
	}
	return m[1]
}

// 1. The binary sibling won. The line must name the `.gguf`, not the `.json`
// the call was given: which file was actually read is the whole point.
func TestResumeFromGGUFIsSaid(t *testing.T) {
	model, tok := hybridOrganism(t)
	path := filepath.Join(t.TempDir(), "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	gp := ggufPathFor(path)
	before := gptConstructions.Load()
	var loadErr error
	out := captureStdout(t, func() { _, _, loadErr = LoadCheckpoint(ggufCorpus(), path) })
	if loadErr != nil {
		t.Fatalf("load: %v", loadErr)
	}
	if gptConstructions.Load() != before {
		t.Fatal("the JSON path was taken — this fixture is not testing the GGUF resume")
	}
	if got := resumeSaid(t, out); got != gp {
		t.Fatalf("the resume named %q, want %q; stdout was %q", got, gp, out)
	}
	if strings.Contains(out, "not used") || strings.Contains(out, "older than") {
		t.Fatalf("a clean GGUF resume printed a refusal too: %q", out)
	}
	t.Logf("%s", strings.TrimSpace(out))
}

// 2. No sibling at all — what every checkpoint written by the C, Rust and JS
// cores looks like, and what every organism's first save looked like before
// 2026-09-16. The line must name the `.json` and must not read as a refusal.
func TestResumeFromJSONIsSaid(t *testing.T) {
	model, tok := hybridOrganism(t)
	path := filepath.Join(t.TempDir(), "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := os.Remove(ggufPathFor(path)); err != nil {
		t.Fatalf("removing the sibling: %v", err)
	}
	var loadErr error
	out := captureStdout(t, func() { _, _, loadErr = LoadCheckpoint(ggufCorpus(), path) })
	if loadErr != nil {
		t.Fatalf("load: %v", loadErr)
	}
	if got := resumeSaid(t, out); got != path {
		t.Fatalf("the resume named %q, want %q; stdout was %q", got, path, out)
	}
	if strings.Contains(out, "not used") || strings.Contains(out, "older than") {
		t.Fatalf("a JSON with no sibling printed a refusal: %q", out)
	}
	t.Logf("%s", strings.TrimSpace(out))
}

// 3. A refused sibling must print both: its own refusal, saying why the binary
// was not used, and the resume line, saying which file the organism came from.
// One without the other is the old silence in a new place — a refusal alone
// does not prove the JSON read succeeded, and a resume alone hides that a
// sibling existed and was wrong.
func TestRefusedSiblingStillSaysTheResume(t *testing.T) {
	model, tok := hybridOrganism(t)
	path := filepath.Join(t.TempDir(), "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	// The foreign-mouth fixture of TestGGUFForeignTokenizerRefused, reduced to
	// what makes the identity differ: a different corpus and a trained merge.
	other := make([]string, 0, 64)
	for i := 0; i < 64; i++ {
		other = append(other, fmt.Sprintf("QUITE ANOTHER MOUTH %d speaking in letters the first never ate", i))
	}
	otherTok := NewEvolvingTokenizer(other)
	otherTok.BPEEnabled = true
	otherTok.Merges = []MergePair{{"t", "h"}}
	gp := ggufPathFor(path)
	if err := writeCheckpointGGUF(gp, NewGPT(otherTok), otherTok); err != nil {
		t.Fatalf("writing the foreign GGUF: %v", err)
	}
	var loadErr error
	out := captureStdout(t, func() { _, _, loadErr = LoadCheckpoint(ggufCorpus(), path) })
	if loadErr != nil {
		t.Fatalf("load: %v", loadErr)
	}
	if !strings.Contains(out, "not used") || !strings.Contains(out, "identity") {
		t.Fatalf("the refusal stopped firing: %q", out)
	}
	if got := resumeSaid(t, out); got != path {
		t.Fatalf("the resume named %q, want the JSON %q; stdout was %q", got, path, out)
	}
	t.Logf("%s", strings.TrimSpace(out))
}

// 4. First boot: no checkpoint on disk, the organism climbs from an embryo. It
// was silent, which made "no line" mean the same as "resumed from the GGUF" to
// anything reading the file.
func TestFirstBootWithNoCheckpointIsSaid(t *testing.T) {
	saved := CFG
	t.Cleanup(func() { CFG = saved })
	path := filepath.Join(t.TempDir(), "molequla_ckpt.json")
	var loadErr error
	out := captureStdout(t, func() { _, _, loadErr = LoadCheckpoint(ggufCorpus(), path) })
	if loadErr == nil {
		t.Fatal("loading a checkpoint that does not exist returned no error")
	}
	if !strings.Contains(out, "[ckpt] no checkpoint at "+path) ||
		!strings.Contains(out, "embryo") {
		t.Fatalf("the first boot was silent or said something else; stdout was %q", out)
	}
	if resumeSaid(t, out) != "" {
		t.Fatalf("a boot with no checkpoint claimed a resume: %q", out)
	}
	t.Logf("%s", strings.TrimSpace(out))
}

// 5. A checkpoint that exists and cannot be parsed ends in an embryo too, and
// it is the one case where that is a loss rather than a beginning. It gets its
// own wording so the day's table does not read it as a first boot.
func TestUnreadableCheckpointIsSaid(t *testing.T) {
	saved := CFG
	t.Cleanup(func() { CFG = saved })
	path := filepath.Join(t.TempDir(), "molequla_ckpt.json")
	if err := os.WriteFile(path, []byte(`{"base":`), 0o644); err != nil {
		t.Fatal(err)
	}
	var loadErr error
	out := captureStdout(t, func() { _, _, loadErr = LoadCheckpoint(ggufCorpus(), path) })
	if loadErr == nil {
		t.Fatal("a truncated checkpoint loaded without error")
	}
	if !strings.Contains(out, "could not be read") || !strings.Contains(out, "embryo") {
		t.Fatalf("an unreadable checkpoint was silent; stdout was %q", out)
	}
	if strings.Contains(out, "no checkpoint at") {
		t.Fatalf("a present-but-broken checkpoint read as a first boot: %q", out)
	}
	t.Logf("%s", strings.TrimSpace(out))
}

// 6. The numbers in the line have to be the read's, not decoration. Size is the
// file's own size on disk, and the two checkpoints in one fixture differ in size
// by more than any rounding, so a line that reported the wrong file's size —
// the JSON's while reading the GGUF, say — goes red here.
func TestResumeLineReportsTheFileItRead(t *testing.T) {
	model, tok := hybridOrganism(t)
	path := filepath.Join(t.TempDir(), "molequla_ckpt.json")
	if err := SaveCheckpoint(model, tok, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	gp := ggufPathFor(path)
	gi, err := os.Stat(gp)
	if err != nil {
		t.Fatal(err)
	}
	ji, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if gi.Size() >= ji.Size() {
		t.Fatalf("the fixture cannot tell the two sizes apart: GGUF %d, JSON %d", gi.Size(), ji.Size())
	}
	out := captureStdout(t, func() {
		if _, _, err := LoadCheckpoint(ggufCorpus(), path); err != nil {
			t.Errorf("load: %v", err)
		}
	})
	m := ckptResumeLine.FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("no resume line; stdout was %q", out)
	}
	want := fmt.Sprintf("%.1f", float64(gi.Size())/(1<<20))
	if m[2] != want {
		t.Fatalf("the line reported %s MB for a %d-byte file (want %s MB); the JSON is %d bytes",
			m[2], gi.Size(), want, ji.Size())
	}
	t.Logf("%s", strings.TrimSpace(out))
}

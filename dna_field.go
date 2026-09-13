package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// The DNA tree as a field.
//
// dnaWrite drops gen_<unix>_<step>.txt into ../dna/output/<self>/. Every other
// organism reads that directory. Until repair 3 (MOLEQULALOG2.md, 2026-09-13)
// the reader removed a fragment after appending it to its own corpus, so a
// fragment fed exactly one organism — whoever scanned the directory first that
// tick — and the loser of the scan stopped growing. Now the reader never
// deletes: each organism keeps a cursor per source (the last file name it ate,
// ordered by the <unix>,<step> pair in the name), persisted next to its
// checkpoint so a restart does not re-eat its corpus; the writer prunes its
// own directory by age; a tick reads at most DNAMaxReadsPerTick new files.
//
// Extra read-only sources (CFG.DNAExtraSources, e.g. "world" written by the
// eye) sit beside the four elements as directories under ../dna/output; they
// are food, not organisms, and are pruned by whoever writes them.
// ═══════════════════════════════════════════════════════════════════════════════

const dnaCursorFile = "dna_cursor.json"

// dnaSources lists the ../dna/output subdirectories this organism eats from:
// every element but itself, then the configured extra sources.
func dnaSources(element string) []string {
	out := make([]string, 0, len(dnaElements)+len(CFG.DNAExtraSources))
	for _, e := range dnaElements {
		if e != element {
			out = append(out, e)
		}
	}
	for _, s := range CFG.DNAExtraSources {
		if s != "" && s != element {
			out = append(out, s)
		}
	}
	return out
}

// dnaFragOrder parses gen_<unix>_<step>.txt into its ordering pair. A name
// that does not fit the pattern is not a fragment and is ignored by readers.
func dnaFragOrder(name string) (unix int64, step int64, ok bool) {
	if !strings.HasPrefix(name, "gen_") || !strings.HasSuffix(name, ".txt") {
		return 0, 0, false
	}
	body := strings.TrimSuffix(strings.TrimPrefix(name, "gen_"), ".txt")
	parts := strings.SplitN(body, "_", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}
	u, err1 := strconv.ParseInt(parts[0], 10, 64)
	s, err2 := strconv.ParseInt(parts[1], 10, 64)
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	return u, s, true
}

// dnaNewer reports whether fragment `name` comes after `last` in write order.
// An empty `last` means nothing has been eaten from that source yet.
func dnaNewer(name, last string) bool {
	if last == "" {
		return true
	}
	nu, ns, ok := dnaFragOrder(name)
	if !ok {
		return false
	}
	lu, ls, ok := dnaFragOrder(last)
	if !ok {
		return true
	}
	if nu != lu {
		return nu > lu
	}
	return ns > ls
}

// dnaCursor is one organism's read position into every source it eats from.
type dnaCursor struct {
	Last map[string]string `json:"last"` // source → last consumed file name
	path string
}

// loadDNACursor reads the cursor at path, or starts an empty one there.
func loadDNACursor(path string) *dnaCursor {
	c := &dnaCursor{Last: map[string]string{}, path: path}
	if data, err := os.ReadFile(path); err == nil {
		var on struct {
			Last map[string]string `json:"last"`
		}
		if json.Unmarshal(data, &on) == nil && on.Last != nil {
			c.Last = on.Last
		}
	}
	return c
}

// save writes the cursor atomically (temp file then rename), so a crash mid-write
// leaves the previous position rather than a truncated file.
func (c *dnaCursor) save() {
	if c == nil || c.path == "" {
		return
	}
	data, err := json.Marshal(struct {
		Last map[string]string `json:"last"`
	}{c.Last})
	if err != nil {
		return
	}
	tmp := c.path + ".tmp"
	if os.WriteFile(tmp, data, 0644) == nil {
		os.Rename(tmp, c.path)
	}
}

// dnaListNew returns the fragments in dir newer than `last`, in write order.
func dnaListNew(dir, last string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	type frag struct {
		name string
		unix int64
		step int64
	}
	frags := make([]frag, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		u, s, ok := dnaFragOrder(e.Name())
		if !ok || !dnaNewer(e.Name(), last) {
			continue
		}
		frags = append(frags, frag{e.Name(), u, s})
	}
	sort.Slice(frags, func(i, j int) bool {
		if frags[i].unix != frags[j].unix {
			return frags[i].unix < frags[j].unix
		}
		return frags[i].step < frags[j].step
	})
	out := make([]string, len(frags))
	for i, f := range frags {
		out[i] = f.name
	}
	return out
}

// dnaPruneOwn removes this organism's own fragments older than retain, by the
// <unix> in the file name. The writer is the only one allowed to delete; a
// reader that fell behind by more than retain loses the oldest fragments and
// nothing else.
func dnaPruneOwn(element string, retain time.Duration, keep int) int {
	if element == "" || (retain <= 0 && keep <= 0) {
		return 0
	}
	dir := filepath.Join("../dna/output", element)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	cutoff := int64(0)
	if retain > 0 {
		cutoff = time.Now().Add(-retain).Unix()
	}
	removed := 0
	var kept []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		u, _, ok := dnaFragOrder(e.Name())
		if !ok {
			continue
		}
		if u < cutoff {
			if os.Remove(filepath.Join(dir, e.Name())) == nil {
				removed++
			}
			continue
		}
		kept = append(kept, e.Name())
	}
	// Count bound (repair 5): of what the age bound left, keep the newest
	// `keep` fragments and drop the rest, oldest first.
	if keep > 0 && len(kept) > keep {
		sort.Slice(kept, func(i, j int) bool { return dnaNewer(kept[j], kept[i]) }) // oldest first
		for _, name := range kept[:len(kept)-keep] {
			if os.Remove(filepath.Join(dir, name)) == nil {
				removed++
			}
		}
	}
	return removed
}

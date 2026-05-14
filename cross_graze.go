package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// CrossField — Dario-style cross-organism logit injection.
//
// Q's interference layer (postgpt_q.c:1384 `raw[i] += c_doc * doc_signal[i]`)
// picks heavy tokens from a doc and boosts them mid-generation. Stanley's
// graze (stanley.c + graze.c:289) splices a foreign vocab token from a
// mmap'd GGUF when chambers signal hunger. Both are doc-shaped knowledge
// injection.
//
// Here the «doc» is **the sibling organism's recent emission stream**. Per
// Oleg 2026-05-14 PM: «как в дарио только вместо доков, слова, метрики
// и проч». Each organism reads its peers' recent DNA fragments (already
// mirrored to ../dna/seen/<sibling>/ by dnaRead, commit e5c1685),
// tokenizes them, keeps a rolling per-sibling buffer, and during its own
// generation adds a rank-decay logit boost to the token ids the siblings
// just emitted. The host organism's voice gets pulled toward what its
// peers are saying RIGHT NOW, not just what the corpus contains.
//
// Direct cross-pollination at the logit level. Mid-emission, not after-burst.
// The «metrics» half (sibling entropy / syntropy / loss) is wired via the
// `MetricBoost` field — modulates per-sibling coef when set, no-op when nil.
// ═══════════════════════════════════════════════════════════════════════════════

// CrossField is the per-organism sibling-pasture state. One per running
// organism; main() instantiates it when --cross-graze AND --element are both
// set. Refresh runs once per generation entry (lazy, throttled by
// ScanInterval). Apply pushes a coef-scaled rank-decay boost into the
// caller's overlaidLogits before sampling.
type CrossField struct {
	SelfElement  string                            // own element label
	PastureBase  string                            // ../dna/seen relative to organism CWD
	Siblings     []string                          // other elements
	Recent       map[string][]int                  // sibling → ring buffer of recent token ids
	RecentCap    int                               // per-sibling buffer size
	LastScan     time.Time                         // throttle FS reads
	ScanInterval time.Duration                     // min gap between rescans
	SeenFiles    map[string]bool                   // dedup of ingested gen_*.txt files
	SeenCap      int                               // hard cap on SeenFiles before purging oldest sibling/* prefix
	MetricBoost  func(sibling string) float64      // optional per-sibling coef multiplier
	mu           sync.Mutex
}

// NewCrossField constructs a CrossField for the given own element. Siblings
// are the standard four-element ecology minus self. PastureBase is typically
// "../dna/seen" relative to the organism's workdir (matches dnaRead mirror
// path molequla.go:5410).
func NewCrossField(element, pastureBase string) *CrossField {
	all := []string{"earth", "air", "water", "fire"}
	sibs := make([]string, 0, len(all)-1)
	for _, e := range all {
		if e != element {
			sibs = append(sibs, e)
		}
	}
	return &CrossField{
		SelfElement:  element,
		PastureBase:  pastureBase,
		Siblings:     sibs,
		Recent:       make(map[string][]int, len(sibs)),
		RecentCap:    64,
		ScanInterval: 30 * time.Second,
		SeenFiles:    make(map[string]bool, 256),
		SeenCap:      2048, // ~8h × 3 siblings × ~1 emission/min keeps under bound
	}
}

// MaybeRefresh walks PastureBase/<sibling>/gen_*.txt for files not yet
// ingested, tokenizes their text, appends to per-sibling ring buffer.
// Throttled by ScanInterval — calling every token step would be O(FS) hot.
func (c *CrossField) MaybeRefresh(tok *EvolvingTokenizer) {
	if c == nil || tok == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if time.Since(c.LastScan) < c.ScanInterval {
		return
	}
	c.LastScan = time.Now()
	bosID, hasBos := tok.Stoi[tok.BOS]
	eosID, hasEos := tok.Stoi[tok.EOS]
	for _, sib := range c.Siblings {
		dir := filepath.Join(c.PastureBase, sib)
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		type fileMtime struct {
			name  string
			mtime time.Time
		}
		files := make([]fileMtime, 0, len(entries))
		for _, e := range entries {
			n := e.Name()
			if !strings.HasPrefix(n, "gen_") || !strings.HasSuffix(n, ".txt") {
				continue
			}
			info, err := e.Info()
			if err != nil {
				continue
			}
			files = append(files, fileMtime{n, info.ModTime()})
		}
		sort.Slice(files, func(i, j int) bool { return files[i].mtime.Before(files[j].mtime) })
		for _, f := range files {
			key := sib + "/" + f.name
			if c.SeenFiles[key] {
				continue
			}
			c.SeenFiles[key] = true
			data, err := os.ReadFile(filepath.Join(dir, f.name))
			if err != nil {
				continue
			}
			text := strings.TrimSpace(string(data))
			if text == "" {
				continue
			}
			ids := tok.Encode(text)
			// Strip BOS/EOS sentinel if present.
			if hasBos && len(ids) > 0 && ids[0] == bosID {
				ids = ids[1:]
			}
			if hasEos && len(ids) > 0 && ids[len(ids)-1] == eosID {
				ids = ids[:len(ids)-1]
			}
			c.Recent[sib] = append(c.Recent[sib], ids...)
			if len(c.Recent[sib]) > c.RecentCap {
				c.Recent[sib] = c.Recent[sib][len(c.Recent[sib])-c.RecentCap:]
			}
		}
	}
	// Hard cap on SeenFiles dedup map — wipe to empty when over SeenCap.
	// Loses dedup state across the rebuild, so a few duplicate ingestions
	// possible right after wipe; acceptable vs unbounded growth over 24h+
	// runs (Opus audit P1, 2026-05-14).
	if c.SeenCap > 0 && len(c.SeenFiles) > c.SeenCap {
		c.SeenFiles = make(map[string]bool, c.SeenCap/2)
	}
}

// Apply adds Dario-style rank-decay logit boost from the most recent topN
// tokens of each sibling into the host's `logits` slice in place.
//
// For each sibling, the most recent token gets `coef`, the second most recent
// gets `coef/2`, rank k gets `coef/(1+k)`. Matches Q's interf_signal_chunk
// 1/(1+rank) normalisation (postgpt_q.c:809-818).
//
// If MetricBoost is set, the per-sibling coef is multiplied by
// `MetricBoost(sibling)` — gateway for the metrics half of «слова, метрики
// и проч». MetricBoost defaults to nil (1.0 implicit).
func (c *CrossField) Apply(logits []float64, coef float64, topN int) int {
	if c == nil || coef == 0 || len(logits) == 0 {
		return 0
	}
	if topN <= 0 {
		topN = 8
	}
	V := len(logits)
	c.mu.Lock()
	defer c.mu.Unlock()
	boosted := 0
	for _, sib := range c.Siblings {
		seq := c.Recent[sib]
		if len(seq) == 0 {
			continue
		}
		sibCoef := coef
		if c.MetricBoost != nil {
			if m := c.MetricBoost(sib); m > 0 {
				sibCoef *= m
			}
		}
		for rank := 0; rank < topN; rank++ {
			idx := len(seq) - 1 - rank
			if idx < 0 {
				break
			}
			tid := seq[idx]
			if tid < 0 || tid >= V {
				continue
			}
			logits[tid] += sibCoef / float64(1+rank)
			boosted++
		}
	}
	return boosted
}

// Stats returns a one-line summary for debug logging — total tokens cached
// per sibling.
func (c *CrossField) Stats() string {
	if c == nil {
		return ""
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	var b strings.Builder
	fmt.Fprintf(&b, "[graze] %s pasture:", c.SelfElement)
	for _, sib := range c.Siblings {
		fmt.Fprintf(&b, " %s=%d", sib, len(c.Recent[sib]))
	}
	return b.String()
}

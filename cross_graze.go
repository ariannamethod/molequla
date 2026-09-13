package main

import (
	"fmt"
	"os"
	"path/filepath"
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
// и проч». Each organism reads its peers' recent DNA fragments straight from
// the field they are written to (../dna/output/<sibling>/, the same tree
// dnaRead eats from; repair 3), tokenizes them, keeps a rolling per-sibling
// buffer, and during its own generation adds a rank-decay logit boost to the
// token ids the siblings just emitted. The host organism's voice gets pulled
// toward what its peers are saying RIGHT NOW, not just what the corpus
// contains.
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
	SelfElement  string                       // own element label
	PastureBase  string                       // ../dna/output relative to organism CWD (repair 3)
	Siblings     []string                     // every DNA source this organism reads
	Recent       map[string][]int             // sibling → ring buffer of recent token ids
	RecentCap    int                          // per-sibling buffer size
	LastScan     time.Time                    // throttle FS reads
	ScanInterval time.Duration                // min gap between rescans
	Last         map[string]string            // sibling → last fragment ingested (repair 6: a cursor, like dnaRead's)
	MetricBoost  func(sibling string) float64 // optional per-sibling coef multiplier
	mu           sync.Mutex
}

// NewCrossField constructs a CrossField for the given own element. Siblings
// are every DNA source this organism reads (dnaSources: the other elements
// plus CFG.DNAExtraSources). PastureBase is "../dna/output" relative to the
// organism's workdir — the same tree dnaRead eats from, since readers no
// longer mirror fragments into a seen/ tree (repair 3).
func NewCrossField(element, pastureBase string) *CrossField {
	sibs := dnaSources(element)
	return &CrossField{
		SelfElement:  element,
		PastureBase:  pastureBase,
		Siblings:     sibs,
		Recent:       make(map[string][]int, len(sibs)),
		RecentCap:    64,
		ScanInterval: 30 * time.Second,
		Last:         make(map[string]string, len(sibs)),
	}
}

// MaybeRefresh walks PastureBase/<sibling>/ for fragments newer than the one
// last ingested from that sibling — the same numeric <unix>,<step> order and
// the same dnaListNew that dnaRead uses — tokenizes them and appends to the
// per-sibling ring buffer. A cursor per sibling replaces the dedup map of
// seen names that was wiped when it grew past 2048 entries and then re-read
// every file in the tree under the model lock (audit A, P1-3). Throttled by
// ScanInterval — calling every token step would be O(FS) hot.
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
		for _, name := range dnaListNew(dir, c.Last[sib]) {
			data, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				continue
			}
			c.Last[sib] = name
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

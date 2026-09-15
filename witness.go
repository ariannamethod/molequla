package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// The mycelium as a witness (repair 7, MOLEQULALOG2.md 2026-09-13).
//
// It sees the whole ecology and says what it sees. It never steers: no
// field_steering row, no write into dna/, no delete — the arrow points
// outward only (stdout, a jsonl log). The contract is what the Go cores
// actually write, verified in audit C §2.7:
//
//   inputs   mesh.db / organisms — the columns SwarmRegistry populates
//            (id, pid, stage, n_params, syntropy, entropy, last_heartbeat,
//            parent_id, status, element, global_step); the DNA field
//            ../dna/output/<source>/ by name and size, read-only.
//   compute  am_method_field_entropy / _syntropy and am_harmonic_forward
//            through cgo (witness_cgo.go); the decision ladder of
//            am_method_step re-stated in Go, without its AML side effects.
//   outputs  one line per tick on stdout and one JSON record per tick in
//            witness.jsonl in the working directory.
//
// What it does not compute, and why: field coherence and organism
// resonance need a gamma vector per organism, and no core writes one — the
// old mycelium read columns that did not exist and reported a constant.
// The decision-effectiveness classes of the old mycelium compared a
// snapshot with itself inside one tick; here every delta is across ticks.
//
// Run it as a fifth process from a sibling of the organism directories, so
// ../dna/output resolves to the same tree:  molequla --witness [--once]
// [--witness-interval 1].
// ═══════════════════════════════════════════════════════════════════════════════

var (
	witnessMode     bool
	witnessInterval = 1.0
	witnessOnce     bool
)

const (
	witnessLiveWindow  = 60.0 // s — the governor's live window (AcquireMitosisSlot)
	witnessHistoryLen  = 16   // AM_METHOD_HISTORY_LEN
	witnessDNAEvery    = 5    // DNA field scan every N ticks (the old sentinel's cadence)
	witnessActionsKept = 8    // the "stuck in dampen loop" window
	witnessLogFile     = "witness.jsonl"
)

type witnessOrganism struct {
	ID         string  `json:"id"`
	PID        int     `json:"pid"`
	Stage      int     `json:"stage"`
	NParams    int     `json:"n_params"`
	Syntropy   float64 `json:"syntropy"`
	Entropy    float64 `json:"entropy"`
	Heartbeat  float64 `json:"last_heartbeat"`
	ParentID   string  `json:"parent_id,omitempty"`
	Status     string  `json:"status"`
	Element    string  `json:"element"`
	GlobalStep int     `json:"global_step"`
	// The voice (routing repair 6): the raw transformer magnitude at the first
	// step of the organism's last generation, and how far the corpus overlay has
	// faded out of it — 1 means the overlay is gone.
	GenMag      float64 `json:"gen_mag"`
	OverlayFade float64 `json:"overlay_fade"`
}

type witnessDNASource struct {
	Files  int    `json:"files"`
	Bytes  int64  `json:"bytes"`
	Newest string `json:"newest,omitempty"`
}

type witnessPulse struct {
	Novelty float64 `json:"novelty"`
	Arousal float64 `json:"arousal"`
	Entropy float64 `json:"entropy"`
}

// witnessSnapshot is one tick's record — everything the line on stdout says
// and the numbers behind it.
type witnessSnapshot struct {
	T          float64                     `json:"t"`
	Step       int                         `json:"step"`
	Organisms  []witnessOrganism           `json:"organisms"`
	FieldH     float64                     `json:"field_entropy"`
	FieldS     float64                     `json:"field_syntropy"`
	Trend      float64                     `json:"trend"`
	Action     string                      `json:"action"`
	Strength   float64                     `json:"strength"`
	TargetID   string                      `json:"target_id,omitempty"`
	Harmonic   wHarmonic                   `json:"harmonic"`
	Bias       map[string]float64          `json:"action_bias,omitempty"`
	Pulse      witnessPulse                `json:"pulse"`
	Alerts     []string                    `json:"alerts,omitempty"`
	DNA        map[string]witnessDNASource `json:"dna,omitempty"`
	DNAEvents  []string                    `json:"dna_events,omitempty"`
	World      *witnessWorld               `json:"world,omitempty"`
	SchemaNote string                      `json:"schema_note,omitempty"`
}

// witnessWorld is the world ledger seen from the witness: how many facts it
// holds, how many of them are still open, and when it last learned something.
// Read through the query_only handle; the writer is `molequla --world-ingest`.
type witnessWorld struct {
	Facts    int     `json:"facts"`
	Open     int     `json:"open"`
	Recorded float64 `json:"newest_recorded_at,omitempty"`
}

// witnessReadWorld returns nil when world_facts does not exist — a node whose
// senses have never written a fact has no world memory, which is a state and
// not an error.
func witnessReadWorld(db *sql.DB) *witnessWorld {
	var w witnessWorld
	var open sql.NullInt64
	var recorded sql.NullFloat64
	err := db.QueryRow(`SELECT COUNT(*), SUM(valid_to IS NULL), MAX(recorded_at) FROM world_facts`).
		Scan(&w.Facts, &open, &recorded)
	if err != nil || w.Facts == 0 {
		return nil
	}
	w.Open = int(open.Int64)
	w.Recorded = recorded.Float64
	return &w
}

// witnessState is what persists between ticks: the entropy history the
// trend is computed over, the previous organism set and field entropy for
// the pulse, the last actions for the loop alert, the last DNA scan.
type witnessState struct {
	step     int
	hist     []float64
	prevIDs  map[string]bool
	prevH    float64
	havePrev bool
	actions  []string
	dna      map[string]witnessDNASource
	dnaBase  string
}

func newWitnessState(dnaBase string) *witnessState {
	witnessCMu.Lock()
	wHarmonicInit()
	wMethodInit()
	witnessCMu.Unlock()
	return &witnessState{prevIDs: map[string]bool{}, dnaBase: dnaBase}
}

// witnessOpenMesh opens mesh.db for reading only: query_only makes any
// write through this handle an error, so "the witness never writes" is a
// property of the connection, not of discipline.
func witnessOpenMesh(path string) (*sql.DB, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("witness: no mesh at %s: %w", path, err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec("PRAGMA query_only=1"); err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

// witnessReadField reads the organisms a Go core writes and nothing more. A
// schema mismatch is an error, never an empty field: the old mycelium
// swallowed `no such column` and reported "no organisms alive" for the life
// of every run.
func witnessReadField(db *sql.DB, now float64) ([]witnessOrganism, error) {
	rows, err := db.Query(`SELECT id, pid, stage, n_params, syntropy, entropy, last_heartbeat,
		COALESCE(parent_id,''), status, COALESCE(element,''), COALESCE(global_step,0),
		COALESCE(gen_mag,0), COALESCE(overlay_fade,0)
		FROM organisms WHERE status='alive' AND last_heartbeat > ? ORDER BY id`, now-witnessLiveWindow)
	if err != nil {
		return nil, fmt.Errorf("witness: mesh schema: %w", err)
	}
	defer rows.Close()
	var out []witnessOrganism
	for rows.Next() {
		var o witnessOrganism
		if err := rows.Scan(&o.ID, &o.PID, &o.Stage, &o.NParams, &o.Syntropy, &o.Entropy, &o.Heartbeat,
			&o.ParentID, &o.Status, &o.Element, &o.GlobalStep, &o.GenMag, &o.OverlayFade); err != nil {
			return nil, fmt.Errorf("witness: mesh row: %w", err)
		}
		out = append(out, o)
	}
	return out, rows.Err()
}

// witnessScanDNA reads the DNA field by name and size — every writer's
// directory under base — and touches nothing.
func witnessScanDNA(base string, sources []string) map[string]witnessDNASource {
	out := make(map[string]witnessDNASource, len(sources))
	for _, src := range sources {
		entries, err := os.ReadDir(filepath.Join(base, src))
		if err != nil {
			continue
		}
		var s witnessDNASource
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			if _, _, ok := dnaFragOrder(e.Name()); !ok {
				continue
			}
			if fi, err := e.Info(); err == nil {
				s.Bytes += fi.Size()
			}
			s.Files++
			if s.Newest == "" || dnaNewer(e.Name(), s.Newest) {
				s.Newest = e.Name()
			}
		}
		out[src] = s
	}
	return out
}

// witnessTrend is am_method_step's entropy trend: mean of the four samples
// before the last four minus mean of the last four (positive = organizing),
// once at least four samples exist.
func witnessTrend(hist []float64) float64 {
	n := len(hist)
	if n < 4 {
		return 0
	}
	var recent, earlier float64
	rc, ec := 0, 0
	for i := 0; i < n && i < 8; i++ {
		v := hist[n-1-i]
		if i < 4 {
			recent += v
			rc++
		} else {
			earlier += v
			ec++
		}
	}
	if rc == 0 || ec == 0 {
		return 0
	}
	return earlier/float64(ec) - recent/float64(rc)
}

// witnessDecide is am_method_step's ladder without its first rung: the
// coherence clause needs gamma, which no core writes. Strengths are the C's.
func witnessDecide(n int, entropy, trend float64) (string, float64) {
	switch {
	case n == 0:
		return "wait", 0
	case trend > 0.05:
		return "amplify", math.Min(1, trend*5)
	case trend < -0.05:
		return "dampen", math.Min(1, math.Abs(trend)*5)
	case entropy > 2.0:
		return "ground", math.Min(1, (entropy-1.5)*0.5)
	case entropy < 0.5:
		return "explore", math.Min(1, (1-entropy)*0.5)
	}
	return "sustain", 0.1
}

// witnessHarmonicBias is the old HarmonicNet action-bias mapping on the
// harmonics alone (mycelium.py:739-760); the resonance-based rungs are
// omitted because resonance is 0 without gamma and their bias always fired.
func witnessHarmonicBias(h wHarmonic, histLen int) map[string]float64 {
	bias := map[string]float64{}
	if histLen < 4 {
		return bias
	}
	amp := h.Harmonics[h.Dominant]
	switch {
	case h.Dominant <= 1 && amp > 0.1:
		bias["dampen"] = math.Min(amp*2, 1)
	case h.Dominant <= 1 && amp < -0.1:
		bias["amplify"] = math.Min(math.Abs(amp)*2, 1)
	case h.Dominant >= 4:
		bias["ground"] = math.Min(math.Abs(amp)*3, 1)
	}
	return bias
}

// observe is one tick over an already-read field and DNA scan; pure of the
// database so it can be gated on synthetic organisms.
func (w *witnessState) observe(now float64, orgs []witnessOrganism, dna map[string]witnessDNASource) witnessSnapshot {
	w.step++
	snap := witnessSnapshot{T: now, Step: w.step, Organisms: orgs}
	n := len(orgs)

	// Field means and harmonics through the C core, serialised.
	witnessCMu.Lock()
	wMethodClear()
	wHarmonicClear()
	zeros := make([]float32, wGammaDim)
	for i, o := range orgs {
		wMethodPushOrganism(i, o.Entropy, o.Syntropy)
		wHarmonicPushGamma(i, zeros, o.Entropy)
	}
	if n > 0 {
		snap.FieldH = wMethodFieldEntropy()
		snap.FieldS = wMethodFieldSyntropy()
		wHarmonicPushEntropy(snap.FieldH)
		snap.Harmonic = wHarmonicForward(w.step)
	}
	witnessCMu.Unlock()

	if n > 0 {
		w.hist = append(w.hist, snap.FieldH)
		if len(w.hist) > witnessHistoryLen {
			w.hist = w.hist[len(w.hist)-witnessHistoryLen:]
		}
	}
	snap.Trend = witnessTrend(w.hist)
	snap.Action, snap.Strength = witnessDecide(n, snap.FieldH, snap.Trend)
	snap.Bias = witnessHarmonicBias(snap.Harmonic, len(w.hist))
	if len(snap.Bias) == 0 {
		snap.Bias = nil
	}
	if n > 0 {
		best := orgs[0]
		for _, o := range orgs[1:] {
			if o.Entropy < best.Entropy {
				best = o
			}
		}
		snap.TargetID = best.ID
	}

	// Pulse, across ticks: novelty from the organism set, arousal from the
	// field entropy delta, entropy as Shannon over organism entropies.
	ids := make(map[string]bool, n)
	for _, o := range orgs {
		ids[o.ID] = true
	}
	if len(w.prevIDs) > 0 {
		changed := 0
		for id := range ids {
			if !w.prevIDs[id] {
				changed++
			}
		}
		for id := range w.prevIDs {
			if !ids[id] {
				changed++
			}
		}
		snap.Pulse.Novelty = float64(changed) / math.Max(1, float64(n))
	} else if n > 0 {
		snap.Pulse.Novelty = 1
	}
	w.prevIDs = ids
	if w.havePrev && n > 0 {
		snap.Pulse.Arousal = math.Min(1, math.Abs(snap.FieldH-w.prevH)*2)
	}
	if n > 0 {
		w.prevH, w.havePrev = snap.FieldH, true
	}
	if n >= 2 {
		total := 1e-10
		for _, o := range orgs {
			total += o.Entropy
		}
		var hs float64
		for _, o := range orgs {
			p := o.Entropy / total
			hs -= p * math.Log(p+1e-10)
		}
		snap.Pulse.Entropy = hs
	}

	// Alerts (the old FieldMonitor's, minus the coherence one).
	w.actions = append(w.actions, snap.Action)
	if len(w.actions) > witnessActionsKept {
		w.actions = w.actions[len(w.actions)-witnessActionsKept:]
	}
	if n == 0 {
		snap.Alerts = append(snap.Alerts, "no organisms alive")
	} else {
		if snap.FieldH > 2.5 {
			snap.Alerts = append(snap.Alerts, fmt.Sprintf("entropy high: %.3f", snap.FieldH))
		}
		if len(w.actions) >= witnessActionsKept {
			all := true
			for _, a := range w.actions {
				if a != "dampen" {
					all = false
					break
				}
			}
			if all {
				snap.Alerts = append(snap.Alerts, "stuck in dampen loop")
			}
		}
	}

	// DNA field: what changed since the last scan.
	if dna != nil {
		snap.DNA = dna
		if w.dna != nil {
			keys := make([]string, 0, len(dna))
			for k := range dna {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				cur, prev := dna[k], w.dna[k]
				switch {
				case cur.Newest != prev.Newest && cur.Newest != "":
					snap.DNAEvents = append(snap.DNAEvents, fmt.Sprintf("%s wrote %s", k, cur.Newest))
				case cur.Files < prev.Files:
					snap.DNAEvents = append(snap.DNAEvents, fmt.Sprintf("%s pruned %d", k, prev.Files-cur.Files))
				}
			}
		}
		w.dna = dna
	}
	return snap
}

// line is the stdout form of a snapshot: the numbers and the field, in the
// old status-line shape.
func (s witnessSnapshot) line() string {
	var b strings.Builder
	fmt.Fprintf(&b, "[witness] step=%d organisms=%d action=%s(%.2f) H=%.3f S=%.3f trend=%+.3f",
		s.Step, len(s.Organisms), s.Action, s.Strength, s.FieldH, s.FieldS, s.Trend)
	if s.TargetID != "" {
		fmt.Fprintf(&b, " target=%s", s.TargetID)
	}
	fmt.Fprintf(&b, " harm=k%d:%+.3f conf=%.2f pulse=%.2f/%.2f/%.2f",
		s.Harmonic.Dominant, s.Harmonic.Harmonics[s.Harmonic.Dominant], s.Harmonic.StrengthMod,
		s.Pulse.Novelty, s.Pulse.Arousal, s.Pulse.Entropy)
	if len(s.Organisms) > 0 {
		b.WriteString(" |")
		for _, o := range s.Organisms {
			// …/entropy/step/fade — the fade is the organism's own voice
			// against the corpus overlay carrying it (routing repair 6).
			fmt.Fprintf(&b, " %s:s%d/%dk/%.2f/%d/f%.2f",
				o.ID, o.Stage, o.NParams/1000, o.Entropy, o.GlobalStep, o.OverlayFade)
		}
	}
	if len(s.DNA) > 0 {
		keys := make([]string, 0, len(s.DNA))
		for k := range s.DNA {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteString(" | dna")
		for _, k := range keys {
			fmt.Fprintf(&b, " %s=%df/%dK", k, s.DNA[k].Files, s.DNA[k].Bytes/1024)
		}
	}
	if len(s.DNAEvents) > 0 {
		fmt.Fprintf(&b, " | %s", strings.Join(s.DNAEvents, "; "))
	}
	if s.World != nil {
		fmt.Fprintf(&b, " | world %d facts/%d open", s.World.Facts, s.World.Open)
	}
	if len(s.Alerts) > 0 {
		fmt.Fprintf(&b, "  !! %s", strings.Join(s.Alerts, "; "))
	}
	return b.String()
}

// runWitness is the process: read, observe, say, sleep. Returns the exit code.
func runWitness(interval float64, once bool) int {
	meshPath := filepath.Join(swarmDir, "mesh.db")
	db, err := witnessOpenMesh(meshPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	defer db.Close()
	logf, err := os.OpenFile(witnessLogFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Fprintln(os.Stderr, "witness: log:", err)
		return 2
	}
	defer logf.Close()
	fmt.Fprintf(os.Stderr, "[witness] mesh=%s dna=../dna/output interval=%.2fs log=%s\n", meshPath, interval, witnessLogFile)

	w := newWitnessState("../dna/output")
	sources := dnaSources("")
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	if interval <= 0 {
		interval = 1
	}
	for {
		now := float64(time.Now().UnixMilli()) / 1000.0
		orgs, err := witnessReadField(db, now)
		var snap witnessSnapshot
		if err != nil {
			// Say it, every tick, and keep watching: a schema the witness
			// cannot read is a finding, not a reason to fall silent.
			snap = w.observe(now, nil, nil)
			snap.SchemaNote = err.Error()
			snap.Alerts = append(snap.Alerts, err.Error())
		} else {
			var dna map[string]witnessDNASource
			if w.step%witnessDNAEvery == 0 {
				dna = witnessScanDNA(w.dnaBase, sources)
			}
			snap = w.observe(now, orgs, dna)
		}
		// The world ledger, read only (ROADMAP 10). The writer is a separate
		// process — `molequla --world-ingest`, world_ledger.go — because this
		// handle carries PRAGMA query_only and the §11 arrow says it keeps
		// carrying it. The witness says how large the world's memory is and
		// how much of it is still open; it does not touch a row of it.
		snap.World = witnessReadWorld(db)
		if once {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(snap)
			return 0
		}
		fmt.Println(snap.line())
		if rec, err := json.Marshal(snap); err == nil {
			logf.Write(append(rec, '\n'))
		}
		select {
		case <-sigCh:
			fmt.Fprintln(os.Stderr, "[witness] stopped")
			return 0
		case <-time.After(time.Duration(interval * float64(time.Second))):
		}
	}
}

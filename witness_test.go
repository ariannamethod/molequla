package main

import (
	"database/sql"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Repair 7 (MOLEQULALOG2.md, 2026-09-13): the mycelium as a Go witness. The
// gates here are the lessons of audit C §2: read the mesh the cores write
// (not a fixture's schema), never swallow a schema error, compare across
// ticks, verify the C harmonics against the formula, and write nothing back.

func witnessTestMesh(t *testing.T) string {
	t.Helper()
	saved := swarmDir
	swarmDir = t.TempDir()
	t.Cleanup(func() { swarmDir = saved })
	return swarmDir
}

func nowSec() float64 { return float64(time.Now().UnixMilli()) / 1000.0 }

func TestWitnessReadsWhatGoWrites(t *testing.T) {
	dir := witnessTestMesh(t)
	a := NewSwarmRegistry("earth", "earth")
	if err := a.Register(); err != nil {
		t.Fatal(err)
	}
	defer a.MeshDB.Close()
	b := NewSwarmRegistry("air", "air")
	if err := b.Register(); err != nil {
		t.Fatal(err)
	}
	defer b.MeshDB.Close()
	a.Heartbeat(2, 262144, 0.10, 1.25, 4200)
	b.Heartbeat(3, 1100000, -0.05, 0.80, 9800)

	db, err := witnessOpenMesh(filepath.Join(dir, "mesh.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	orgs, err := witnessReadField(db, nowSec())
	if err != nil {
		t.Fatal(err)
	}
	if len(orgs) != 2 {
		t.Fatalf("witness sees %d organisms, want 2: %+v", len(orgs), orgs)
	}
	air, earth := orgs[0], orgs[1] // ORDER BY id
	if air.ID != "air" || air.Stage != 3 || air.NParams != 1100000 || air.GlobalStep != 9800 || math.Abs(air.Entropy-0.80) > 1e-9 || air.Element != "air" {
		t.Fatalf("air read back wrong: %+v", air)
	}
	if earth.ID != "earth" || earth.Stage != 2 || earth.NParams != 262144 || earth.GlobalStep != 4200 || math.Abs(earth.Syntropy-0.10) > 1e-9 {
		t.Fatalf("earth read back wrong: %+v", earth)
	}

	b.MarkHibernating()
	orgs, err = witnessReadField(db, nowSec())
	if err != nil {
		t.Fatal(err)
	}
	if len(orgs) != 1 || orgs[0].ID != "earth" {
		t.Fatalf("after hibernation the field must hold earth alone, got %+v", orgs)
	}

	// The witness's handle cannot write: the property lives in the connection.
	if _, err := db.Exec(`UPDATE organisms SET entropy=0 WHERE id='earth'`); err == nil {
		t.Fatal("the witness handle wrote to the mesh; query_only is not in force")
	}
}

// A schema the witness cannot read is an error, never an empty field.
func TestWitnessSchemaMismatchIsAnError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mesh.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	if _, err := db.Exec(`CREATE TABLE organisms(id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER, n_params INTEGER, entropy REAL, last_heartbeat REAL, status TEXT, element TEXT)`); err != nil {
		t.Fatal(err)
	}
	db.Exec(`INSERT INTO organisms VALUES('a',1,2,100,1.0,?,'alive','earth')`, nowSec())
	db.Close()
	w, err := witnessOpenMesh(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if orgs, err := witnessReadField(w, nowSec()); err == nil {
		t.Fatalf("a table without syntropy/global_step read as %d organisms and no error — the old mycelium's silence", len(orgs))
	}
}

func witnessOrgs(entropies ...float64) []witnessOrganism {
	out := make([]witnessOrganism, len(entropies))
	for i, h := range entropies {
		out[i] = witnessOrganism{ID: "o" + string(rune('1'+i)), Stage: 2, NParams: 1000, Entropy: h, Syntropy: 0.1, Status: "alive"}
	}
	return out
}

// Every delta is across ticks: the old mycelium compared a snapshot with
// itself inside one tick and every effectiveness signal was null.
func TestWitnessDeltasAreAcrossTicks(t *testing.T) {
	w := newWitnessState("")
	s1 := w.observe(1, witnessOrgs(1.0, 1.0), nil)
	if s1.Pulse.Novelty != 1 || s1.Pulse.Arousal != 0 {
		t.Fatalf("first observation: novelty=%v arousal=%v, want 1 and 0", s1.Pulse.Novelty, s1.Pulse.Arousal)
	}
	if math.Abs(s1.FieldH-1.0) > 1e-6 {
		t.Fatalf("field entropy through the C core = %v, want 1.0", s1.FieldH)
	}
	s2 := w.observe(2, witnessOrgs(1.4, 1.4), nil)
	if math.Abs(s2.Pulse.Arousal-0.8) > 1e-6 || s2.Pulse.Novelty != 0 {
		t.Fatalf("H 1.0 -> 1.4 across ticks: arousal=%v novelty=%v, want 0.8 and 0", s2.Pulse.Arousal, s2.Pulse.Novelty)
	}
	s3 := w.observe(3, witnessOrgs(1.4, 1.4), nil)
	if s3.Pulse.Arousal != 0 {
		t.Fatalf("unchanged field: arousal=%v, want 0", s3.Pulse.Arousal)
	}
	s4 := w.observe(4, append(witnessOrgs(1.4, 1.4), witnessOrganism{ID: "o3", Entropy: 1.4, Status: "alive"}), nil)
	if math.Abs(s4.Pulse.Novelty-1.0/3) > 1e-9 {
		t.Fatalf("one organism of three appeared: novelty=%v, want 1/3", s4.Pulse.Novelty)
	}
	s5 := w.observe(5, nil, nil)
	if s5.Action != "wait" || len(s5.Alerts) == 0 || s5.Alerts[0] != "no organisms alive" {
		t.Fatalf("empty field: action=%s alerts=%v", s5.Action, s5.Alerts)
	}
}

func TestWitnessLadderAndTrend(t *testing.T) {
	check := func(n int, h, tr float64, wantA string, wantS float64) {
		t.Helper()
		a, s := witnessDecide(n, h, tr)
		if a != wantA || math.Abs(s-wantS) > 1e-12 {
			t.Fatalf("decide(n=%d H=%v trend=%v) = %s(%v), want %s(%v)", n, h, tr, a, s, wantA, wantS)
		}
	}
	check(0, 1.0, 0, "wait", 0)
	check(2, 1.0, 0.1, "amplify", 0.5)
	check(2, 1.0, -0.3, "dampen", 1)
	check(2, 2.4, 0, "ground", 0.45)
	check(2, 0.2, 0, "explore", 0.4)
	check(2, 1.0, 0, "sustain", 0.1)

	if tr := witnessTrend([]float64{2, 2, 2, 2, 1, 1, 1, 1}); math.Abs(tr-1) > 1e-12 {
		t.Fatalf("trend = %v, want earlier(2) - recent(1) = 1", tr)
	}
	if tr := witnessTrend([]float64{1, 2, 3}); tr != 0 {
		t.Fatalf("trend on three samples = %v, want 0", tr)
	}
	if tr := witnessTrend([]float64{3, 3, 1, 1, 1, 1}); math.Abs(tr-2) > 1e-12 {
		t.Fatalf("trend on six samples = %v, want 2", tr)
	}
}

// The C harmonics are the sine DFT of the entropy history; the formula is
// the ground truth here, computed in float64 beside the C's float32.
func TestWitnessHarmonicsMatchTheDFT(t *testing.T) {
	witnessCMu.Lock()
	defer witnessCMu.Unlock()
	wHarmonicInit()
	wHarmonicClear()
	wHarmonicPushGamma(0, make([]float32, wGammaDim), 1.0)
	const T, k = 16, 2
	seq := make([]float64, T)
	for tt := 0; tt < T; tt++ {
		seq[tt] = 1.0 + math.Sin(2*math.Pi*float64(k+1)*float64(tt)/float64(T))
		wHarmonicPushEntropy(seq[tt])
	}
	r := wHarmonicForward(1)
	for j := 0; j < 8; j++ {
		var sum float64
		for tt := 0; tt < T; tt++ {
			sum += seq[tt] * math.Sin(2*math.Pi*float64(j+1)*float64(tt)/float64(T))
		}
		if want := sum / T; math.Abs(r.Harmonics[j]-want) > 1e-4 {
			t.Fatalf("harmonic %d = %v, formula gives %v", j, r.Harmonics[j], want)
		}
	}
	if r.Dominant != k || math.Abs(r.Harmonics[k]-0.5) > 1e-4 {
		t.Fatalf("dominant=%d amp=%v, want k=%d with amplitude 0.5", r.Dominant, r.Harmonics[k], k)
	}
	if math.Abs(r.StrengthMod-0.475) > 1e-6 { // 0.3 + 0.7 · conf_t(16/16) · conf_n(1/4)
		t.Fatalf("strength_mod = %v, want 0.475", r.StrengthMod)
	}
	bias := witnessHarmonicBias(r, T)
	if v, ok := bias["dampen"]; ok || v != 0 {
		t.Fatalf("dominant k=2 must give no dampen/amplify bias, got %v", bias)
	}
}

// The witness writes nothing back: no table appears in the mesh, no DNA file
// changes, over a run of ticks with the field scanned.
func TestWitnessNeverWritesBack(t *testing.T) {
	dir := witnessTestMesh(t)
	a := NewSwarmRegistry("earth", "earth")
	if err := a.Register(); err != nil {
		t.Fatal(err)
	}
	defer a.MeshDB.Close()
	a.Heartbeat(2, 1000, 0, 1.0, 10)

	base := filepath.Join(t.TempDir(), "dna", "output")
	if err := os.MkdirAll(filepath.Join(base, "earth"), 0755); err != nil {
		t.Fatal(err)
	}
	frag := filepath.Join(base, "earth", "gen_100_1.txt")
	os.WriteFile(frag, []byte("the river follows the gradient\n"), 0644)
	before, _ := os.Stat(frag)

	db, err := witnessOpenMesh(filepath.Join(dir, "mesh.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	tables := func() int {
		var n int
		db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table'`).Scan(&n)
		return n
	}
	tablesBefore := tables()

	w := newWitnessState(base)
	var sawFile, sawEvent bool
	for i := 0; i < 6; i++ {
		orgs, err := witnessReadField(db, nowSec())
		if err != nil {
			t.Fatal(err)
		}
		var dna map[string]witnessDNASource
		if i%witnessDNAEvery == 0 {
			if i == witnessDNAEvery {
				os.WriteFile(filepath.Join(base, "earth", "gen_100_2.txt"), []byte("heat arrives\n"), 0644)
			}
			dna = witnessScanDNA(base, []string{"earth", "air"})
		}
		snap := w.observe(nowSec(), orgs, dna)
		if snap.DNA != nil && snap.DNA["earth"].Files >= 1 {
			sawFile = true
		}
		for _, e := range snap.DNAEvents {
			if e == "earth wrote gen_100_2.txt" {
				sawEvent = true
			}
		}
		_ = snap.line()
	}
	if !sawFile || !sawEvent {
		t.Fatalf("witness did not see the field: file=%v event=%v", sawFile, sawEvent)
	}
	if n := tables(); n != tablesBefore {
		t.Fatalf("mesh gained tables under the witness: %d -> %d", tablesBefore, n)
	}
	var fs int
	db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE name='field_steering'`).Scan(&fs)
	if fs != 0 {
		t.Fatal("field_steering exists — the arrow back is being drawn")
	}
	after, _ := os.Stat(frag)
	if after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		t.Fatal("a DNA fragment changed under the witness")
	}
	if n, _ := os.ReadDir(filepath.Join(base, "earth")); len(n) != 2 {
		t.Fatalf("DNA directory holds %d files, want 2 (nothing deleted)", len(n))
	}
}

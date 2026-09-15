package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The world ledger of change (ROADMAP 10; molequla_new_logic.md §4-6, §15-16).
// Every gate here goes red when the thing it names is broken on purpose; the
// runs are in MOLEQULALOG2.md, 2026-09-15.

// worldTestDB is an empty mesh with the ledger's table migrated in.
func worldTestDB(t *testing.T) *sql.DB {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mesh.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := worldMigrate(db); err != nil {
		t.Fatal(err)
	}
	return db
}

// fact builds one observation the way facts.jsonl carries it.
func fact(source, subject, predicate, object string, validFrom float64, prov map[string]interface{}) worldFact {
	f := worldFact{Source: source, Subject: subject, Predicate: predicate, Object: object, validFrom: validFrom, prov: prov}
	if prov != nil {
		raw, _ := json.Marshal(prov)
		f.Provenance = raw
	}
	return f
}

func worldRowsOf(t *testing.T, db *sql.DB) []struct {
	Object    string
	ValidFrom float64
	ValidTo   sql.NullFloat64
	Recorded  float64
	Predicate string
} {
	t.Helper()
	rows, err := db.Query(`SELECT object, valid_from, valid_to, recorded_at, predicate FROM world_facts ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var out []struct {
		Object    string
		ValidFrom float64
		ValidTo   sql.NullFloat64
		Recorded  float64
		Predicate string
	}
	for rows.Next() {
		var r struct {
			Object    string
			ValidFrom float64
			ValidTo   sql.NullFloat64
			Recorded  float64
			Predicate string
		}
		if err := rows.Scan(&r.Object, &r.ValidFrom, &r.ValidTo, &r.Recorded, &r.Predicate); err != nil {
			t.Fatal(err)
		}
		out = append(out, r)
	}
	return out
}

func worldOpenCount(t *testing.T, db *sql.DB) int {
	t.Helper()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM world_facts WHERE valid_to IS NULL`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// §5: four passes on a still phone are one fact, not four, and no fragment
// after the first. The old prose path wrote a fragment every pass.
func TestWorldLedgerRepeatedObservationSaysNothing(t *testing.T) {
	db := worldTestDB(t)
	f := func(vf float64) worldFact {
		return fact("place", "phone", "at_place", "Neve X, Be'er-Sheva, Israel", vf, nil)
	}
	first, err := worldIngest(db, []worldFact{f(100)}, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 1 {
		t.Fatalf("the first fix said %d lines, want 1: %v", len(first), first)
	}
	for i, vf := range []float64{200, 300, 400} {
		got, err := worldIngest(db, []worldFact{f(vf)}, vf)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("repeat %d emitted %v — repeated identical state must produce no fragment", i+1, got)
		}
	}
	rows := worldRowsOf(t, db)
	if len(rows) != 1 {
		t.Fatalf("four identical observations left %d rows, want 1: %+v", len(rows), rows)
	}
	if n := worldOpenCount(t, db); n != 1 {
		t.Fatalf("%d open rows, want exactly 1", n)
	}
}

// A correction closes and opens; it never erases (§5). Both rows stay, the old
// one carrying the new one's valid_from as its valid_to, and exactly one line.
func TestWorldLedgerContradictionClosesAndOpens(t *testing.T) {
	db := worldTestDB(t)
	if _, err := worldIngest(db, []worldFact{
		fact("place", "phone", "at_place", "Neve X, Be'er-Sheva, Israel", 100, nil),
	}, 100); err != nil {
		t.Fatal(err)
	}
	got, err := worldIngest(db, []worldFact{
		fact("place", "phone", "at_place", "Ramot, Be'er-Sheva, Israel", 260,
			map[string]interface{}{"moved_m": 1900.0}),
	}, 260)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("a contradiction emitted %d lines, want exactly 1: %v", len(got), got)
	}
	want := "The phone moved from Neve X, Be'er-Sheva, Israel to Ramot, Be'er-Sheva, Israel, 1900 m from the last fix."
	if got[0].Line != want {
		t.Fatalf("change line:\n got %q\nwant %q", got[0].Line, want)
	}
	rows := worldRowsOf(t, db)
	if len(rows) != 2 {
		t.Fatalf("%d rows after a correction, want 2 — nothing is deleted: %+v", len(rows), rows)
	}
	if !rows[0].ValidTo.Valid || rows[0].ValidTo.Float64 != 260 {
		t.Fatalf("the old row's valid_to = %+v, want 260 (the new fact's valid_from)", rows[0].ValidTo)
	}
	if rows[0].Object != "Neve X, Be'er-Sheva, Israel" {
		t.Fatalf("the old row was overwritten: object = %q", rows[0].Object)
	}
	if rows[1].ValidTo.Valid {
		t.Fatalf("the new row is not open: valid_to = %+v", rows[1].ValidTo)
	}
	if n := worldOpenCount(t, db); n != 1 {
		t.Fatalf("%d open rows after a correction, want 1", n)
	}
}

// Valid time and recorded time are two clocks. A fact that held at noon and
// arrived at midnight keeps both, and a fact that arrives after a newer one has
// already replaced it is history — stored, closed, and silent.
func TestWorldLedgerRecordedAtDiffersFromValidFrom(t *testing.T) {
	db := worldTestDB(t)
	if _, err := worldIngest(db, []worldFact{
		fact("place", "phone", "at_place", "Ramot, Be'er-Sheva, Israel", 1000, nil),
	}, 5000); err != nil {
		t.Fatal(err)
	}
	rows := worldRowsOf(t, db)
	if rows[0].ValidFrom != 1000 || rows[0].Recorded != 5000 {
		t.Fatalf("valid_from=%v recorded_at=%v, want 1000 and 5000 — the two clocks are not the same clock",
			rows[0].ValidFrom, rows[0].Recorded)
	}

	// Late ingest: an older observation arriving now contradicts a belief that
	// has since been revised. It is history, not news.
	late, err := worldIngest(db, []worldFact{
		fact("place", "phone", "at_place", "Neve X, Be'er-Sheva, Israel", 400, nil),
	}, 6000)
	if err != nil {
		t.Fatal(err)
	}
	if len(late) != 0 {
		t.Fatalf("a late fact emitted %v — the revision stands and says nothing new", late)
	}
	rows = worldRowsOf(t, db)
	if len(rows) != 2 {
		t.Fatalf("%d rows, want 2 — a late fact is still kept: %+v", len(rows), rows)
	}
	var lateRow = rows[1]
	if lateRow.ValidFrom != 400 || !lateRow.ValidTo.Valid || lateRow.ValidTo.Float64 != 1000 || lateRow.Recorded != 6000 {
		t.Fatalf("the late row = %+v, want valid 400→1000 recorded 6000", lateRow)
	}
	if n := worldOpenCount(t, db); n != 1 {
		t.Fatalf("%d open rows after a late ingest, want 1", n)
	}
}

// The cursor: re-running the ingest over the same facts.jsonl changes nothing.
func TestWorldLedgerCursorIsIdempotent(t *testing.T) {
	run := t.TempDir()
	facts := filepath.Join(run, "facts.jsonl")
	// Two observations of one triple and one of another: re-reading the file
	// from the start is then visible in the table even though it emits no
	// fragment — the older of the two arrives after the newer and is filed as
	// history. That is what makes this a gate on the cursor and not on the
	// dedupe behind it.
	lines := []string{
		`{"source":"place","subject":"phone","predicate":"at_place","object":"Neve X, Be'er-Sheva, Israel","valid_from":"2026-09-13T22:10:52Z","provenance":{"accuracy_m":13}}`,
		`{"source":"place","subject":"phone","predicate":"at_place","object":"Ramot, Be'er-Sheva, Israel","valid_from":"2026-09-13T22:19:24Z","provenance":{"accuracy_m":14,"moved_m":1900}}`,
		`{"source":"eye","subject":"rear camera","predicate":"interpreted_as","object":"A close-up view shows a keyboard with white keys.","valid_from":"2026-09-13T22:18:27Z","provenance":{"camera":"0"}}`,
	}
	blob := []byte(strings.Join(lines, "\n") + "\n")
	if err := os.WriteFile(facts, blob, 0644); err != nil {
		t.Fatal(err)
	}
	dna := filepath.Join(run, "dna", "output")
	mesh := filepath.Join(run, "mesh.db")
	cursor := filepath.Join(run, worldCursorFile)

	countFrags := func() int {
		e, _ := os.ReadDir(filepath.Join(dna, worldDNASource))
		return len(e)
	}
	ledgerRows := func() int {
		db, err := sql.Open("sqlite", mesh)
		if err != nil {
			t.Fatal(err)
		}
		defer db.Close()
		var n int
		db.QueryRow(`SELECT COUNT(*) FROM world_facts`).Scan(&n)
		return n
	}

	l, err := openWorldLedger(mesh, facts, dna, cursor)
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	first := l.tick(9000)
	offset := l.cursor.Offset
	l.Close()
	if len(first) != 3 {
		t.Fatalf("first pass emitted %d lines, want 3: %v", len(first), first)
	}
	if offset != int64(len(blob)) {
		t.Fatalf("the cursor stopped at %d of %d bytes — it did not advance over what it read",
			offset, len(blob))
	}
	rows1, frags1 := ledgerRows(), countFrags()
	if frags1 != 3 {
		t.Fatalf("%d fragments after the first pass, want 3", frags1)
	}

	// A second ledger over the same file, same cursor: nothing new.
	l2, err := openWorldLedger(mesh, facts, dna, cursor)
	if err != nil {
		t.Fatal(err)
	}
	again := l2.tick(9100)
	l2.Close()
	if len(again) != 0 {
		t.Fatalf("re-running the ingest emitted %v — the cursor did not hold", again)
	}
	if n := ledgerRows(); n != rows1 {
		t.Fatalf("rows %d -> %d on a re-run; the ingest is not idempotent", rows1, n)
	}
	if n := countFrags(); n != frags1 {
		t.Fatalf("fragments %d -> %d on a re-run", frags1, n)
	}

	// And with the cursor thrown away: every line is read again. No fragment
	// comes of it — nothing in the file contradicts what is open — but the
	// first fix arrives after the one that replaced it and is filed as the
	// history it is, which is exactly the work the cursor saves.
	os.Remove(cursor)
	l3, err := openWorldLedger(mesh, facts, dna, cursor)
	if err != nil {
		t.Fatal(err)
	}
	third := l3.tick(9200)
	l3.Close()
	if len(third) != 0 || countFrags() != frags1 {
		t.Fatalf("a lost cursor emitted %v and left %d fragments, want none and %d",
			third, countFrags(), frags1)
	}
	if n := ledgerRows(); n != rows1+1 {
		t.Fatalf("a lost cursor left %d rows, want %d: the superseded fix is re-filed as history, once",
			n, rows1+1)
	}
}

// The fragments the ledger writes are fragments: dnaListNew parses and orders
// them, and they carry the bracketed header senses.sh writes.
func TestWorldLedgerFragmentsAreDNA(t *testing.T) {
	dir := t.TempDir()
	changes := []worldChange{
		{Line: "The phone moved from A to B.", Row: 7, At: 1789347628},
		{Line: "A person entered the rear camera's frame.", Row: 8, At: 1789347628},
	}
	names, err := worldWriteChanges(dir, changes)
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 {
		t.Fatalf("%d fragments written, want 2", len(names))
	}
	for _, n := range names {
		if _, _, ok := dnaFragOrder(n); !ok {
			t.Fatalf("%q is not a fragment dnaListNew would order", n)
		}
	}
	if !dnaNewer(names[1], names[0]) {
		t.Fatalf("%q does not sort after %q — ledger order is not fragment order", names[1], names[0])
	}
	body, err := os.ReadFile(filepath.Join(dir, worldDNASource, names[0]))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(body), "[world 2026-09-14T01:00:28Z] The phone moved") {
		t.Fatalf("fragment body %q has no bracketed header", string(body))
	}
	if len(body) < CFG.DNAMinFragmentBytes {
		t.Fatalf("a %d-byte fragment is below DNAMinFragmentBytes=%d and a reader would delete it",
			len(body), CFG.DNAMinFragmentBytes)
	}

	// A name an organ took in the same second is stepped past, not overwritten.
	taken := filepath.Join(dir, worldDNASource, "gen_1789347700_9.txt")
	if err := os.WriteFile(taken, []byte("[eye cam0 ...] the organ was here\n"), 0644); err != nil {
		t.Fatal(err)
	}
	more, err := worldWriteChanges(dir, []worldChange{{Line: "The sky changed from fog to overcast.", Row: 9, At: 1789347700}})
	if err != nil {
		t.Fatal(err)
	}
	if more[0] == "gen_1789347700_9.txt" {
		t.Fatal("the ledger took a name an organ already held")
	}
	if b, _ := os.ReadFile(taken); !strings.Contains(string(b), "the organ was here") {
		t.Fatal("the organ's fragment was overwritten")
	}
}

// The move gate is a measured knob, not a shape of the code: a fix wandering
// inside its own accuracy is not a move, a walk down the street is.
func TestWorldLedgerMoveGate(t *testing.T) {
	saved := CFG.WorldMoveMeters
	defer func() { CFG.WorldMoveMeters = saved }()

	// 31.2657873,34.7544114 is the live fix; +0.0002° of latitude is 22 m.
	const home = "31.2657873,34.7544114"
	const nudged = "31.2659873,34.7544114"
	const away = "31.2757873,34.7544114"
	if d, ok := worldDistanceM(home, nudged); !ok || d < 20 || d > 25 {
		t.Fatalf("haversine says %v m for +0.0002 deg of latitude, want ~22", d)
	}
	if d, _ := worldDistanceM(home, away); d < 1080 || d > 1140 {
		t.Fatalf("haversine says %v m for +0.01 deg of latitude, want ~1112", d)
	}

	CFG.WorldMoveMeters = 50
	db := worldTestDB(t)
	if _, err := worldIngest(db, []worldFact{fact("place", "phone", "at_position", home, 100, nil)}, 100); err != nil {
		t.Fatal(err)
	}
	got, err := worldIngest(db, []worldFact{fact("place", "phone", "at_position", nudged, 200, nil)}, 200)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("22 m under a 50 m gate emitted %v — fix noise is not movement", got)
	}
	got, err = worldIngest(db, []worldFact{fact("place", "phone", "at_position", away, 300,
		map[string]interface{}{"place": "Ramot, Be'er-Sheva, Israel"})}, 300)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !strings.HasPrefix(got[0].Line, "The phone moved 1112 m and is still at Ramot") {
		t.Fatalf("1112 m over a 50 m gate gave %v", got)
	}

	// The gate can fail: lower it and the same 22 m becomes a move.
	CFG.WorldMoveMeters = 5
	db2 := worldTestDB(t)
	if _, err := worldIngest(db2, []worldFact{fact("place", "phone", "at_position", home, 100, nil)}, 100); err != nil {
		t.Fatal(err)
	}
	got, err = worldIngest(db2, []worldFact{fact("place", "phone", "at_position", nudged, 200, nil)}, 200)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("a 5 m gate must call 22 m a move; got %v", got)
	}
}

// §6: the eye's sentence is an interpretation with provenance, never a fact
// about the room. The predicate is interpreted_as and the camera, the weights
// and the conditions of the pass survive with it.
func TestWorldLedgerStoresPerceptionAsInterpretation(t *testing.T) {
	db := worldTestDB(t)
	prov := map[string]interface{}{
		"camera": "1", "lens": "front",
		"model": "yent_eye_ours_q6_k.gguf", "mmproj": "yent_eye_smolvlm2_lora_v2_mmproj_q8_0.gguf",
		"edge": 1024.0, "rss_mb": 1020.0,
		"conditions": map[string]interface{}{"mem_mb": 2635.0, "cpus": "4-7", "colony": "asleep"},
	}
	balcony := "A bathroom with a shower curtain, a light above the shower, and a small trash can."
	if _, err := worldIngest(db, []worldFact{
		fact("eye", "front camera", "interpreted_as", balcony, 1789340433, prov),
	}, 1789340433); err != nil {
		t.Fatal(err)
	}
	var pred, obj, raw string
	if err := db.QueryRow(`SELECT predicate, object, provenance FROM world_facts
		WHERE source='eye' AND predicate<>'sees_person'`).Scan(&pred, &obj, &raw); err != nil {
		t.Fatal(err)
	}
	if pred != "interpreted_as" {
		t.Fatalf("the eye's sentence is stored under %q; §6 says it is never `is`", pred)
	}
	if obj != balcony {
		t.Fatalf("the sentence was rewritten: %q", obj)
	}
	var back map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &back); err != nil {
		t.Fatalf("provenance is not JSON: %q", raw)
	}
	for _, key := range []string{"camera", "model", "mmproj", "conditions"} {
		if _, ok := back[key]; !ok {
			t.Fatalf("provenance lost %q: %v", key, back)
		}
	}
}

// The change vocabulary, one row per sentence the organisms can be handed.
func TestWorldLedgerChangeVocabulary(t *testing.T) {
	cases := []struct {
		name string
		prev *worldRow
		f    worldFact
		want string
	}{
		{"first place", nil,
			fact("place", "phone", "at_place", "Neve X, Be'er-Sheva, Israel", 1, nil),
			"The phone is at Neve X, Be'er-Sheva, Israel. Where it was before that, the ledger does not know."},
		{"moved", &worldRow{Object: "Neve X"},
			fact("place", "phone", "at_place", "Ramot", 1, map[string]interface{}{"moved_m": 312.0}),
			"The phone moved from Neve X to Ramot, 312 m from the last fix."},
		{"first sky", nil, fact("place", "sky", "reported_as", "fog", 1, nil), "The sky is fog."},
		{"fog lifted", &worldRow{Object: "fog"}, fact("place", "sky", "reported_as", "clear sky", 1, nil),
			"The sky changed from fog to clear sky."},
		{"first frame", nil, fact("eye", "rear camera", "interpreted_as", "A keyboard.", 1, nil),
			"The rear camera opened on: \"A keyboard.\""},
		{"scene changed", &worldRow{Object: "A keyboard."},
			fact("eye", "rear camera", "interpreted_as", "A dark room with a chair.", 1, nil),
			"The rear camera's scene changed from \"A keyboard.\" to \"A dark room with a chair.\""},
		{"speech", nil, fact("ears", "microphone", "hearing", "speech", 1, map[string]interface{}{"window_s": 12.0}),
			"Speech was heard through the microphone for 12 s."},
		{"silence first is not news", nil, fact("ears", "microphone", "hearing", "silence", 1, nil), ""},
		{"speech stopped", &worldRow{Object: "speech"},
			fact("ears", "microphone", "hearing", "silence", 1, map[string]interface{}{"window_s": 12.0}),
			"The microphone stopped hearing speech: 12 s of room noise instead."},
		{"person entered", &worldRow{Object: "no"}, fact("eye", "rear camera", "sees_person", "yes", 1, nil),
			"A person entered the rear camera's frame."},
		{"person left", &worldRow{Object: "yes"}, fact("eye", "rear camera", "sees_person", "no", 1, nil),
			"A person left the rear camera's frame."},
		{"first sighting cannot have entered", nil, fact("eye", "rear camera", "sees_person", "yes", 1, nil), ""},
		{"a first position says nothing at_place has not", nil,
			fact("place", "phone", "at_position", "31.2,34.7", 1, nil), ""},
	}
	for _, c := range cases {
		if got := worldChangeLine(c.prev, c.f); got != c.want {
			t.Errorf("%s:\n got %q\nwant %q", c.name, got, c.want)
		}
	}
}

// A person entering and leaving is read out of the eye's own sentences; the
// derivation is lexical and says so in its provenance.
func TestWorldLedgerPersonEntersAndLeaves(t *testing.T) {
	db := worldTestDB(t)
	empty := "An empty room has a ceiling fan, a white floor, and a metal frame for a door."
	withHand := "A bathroom with a shower curtain hanging, a person's hand reaching out, and a light fixture above."

	step := func(sentence string, vf float64) []string {
		got, err := worldIngest(db, []worldFact{
			fact("eye", "front camera", "interpreted_as", sentence, vf, map[string]interface{}{"camera": "1"}),
		}, vf)
		if err != nil {
			t.Fatal(err)
		}
		var lines []string
		for _, c := range got {
			lines = append(lines, c.Line)
		}
		return lines
	}

	if lines := step(empty, 100); len(lines) != 1 || !strings.HasPrefix(lines[0], "The front camera opened on:") {
		t.Fatalf("the first frame said %v", lines)
	}
	lines := step(withHand, 200)
	if len(lines) != 2 {
		t.Fatalf("a hand appearing said %v, want a scene change and a person entering", lines)
	}
	if lines[1] != "A person entered the front camera's frame." {
		t.Fatalf("second line = %q", lines[1])
	}
	lines = step(empty, 300)
	if len(lines) != 2 || lines[1] != "A person left the front camera's frame." {
		t.Fatalf("the hand leaving said %v", lines)
	}
	// The same empty room again: no scene change, no person change, nothing.
	if lines = step(empty, 400); len(lines) != 0 {
		t.Fatalf("an unchanged frame said %v", lines)
	}

	var raw string
	if err := db.QueryRow(`SELECT provenance FROM world_facts WHERE predicate='sees_person' AND object='yes'`).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	var prov map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &prov); err != nil {
		t.Fatal(err)
	}
	if prov["derived"] != "lexical" || prov["from"] != "interpreted_as" || prov["text"] != withHand {
		t.Fatalf("a derived fact must carry what it was derived from and how: %v", prov)
	}

	// The eye's own phrasing for an empty room, from the live run of
	// 2026-09-14T01:00:47Z. "no people" is not a person, and the first live
	// ingest read one there.
	negated := "A dark room with a chair, a table, and a blanket, with no people or text visible."
	if m := worldMatchWords(negated, CFG.WorldPersonWords); len(m) != 0 {
		t.Fatalf("%q matched %v — a negated word is not a sighting", negated, m)
	}
	if m := worldMatchWords("A person's hand reaching out.", CFG.WorldPersonWords); len(m) != 2 {
		t.Fatalf("the negation guard swallowed a real sighting: %v", m)
	}
	lines = step(negated, 500)
	if len(lines) != 1 {
		t.Fatalf("an empty room said %v, want only the scene change", lines)
	}

	// The vocabulary is a knob: empty, and nothing is derived at all.
	saved := CFG.WorldPersonWords
	defer func() { CFG.WorldPersonWords = saved }()
	CFG.WorldPersonWords = nil
	if d := worldDerive(fact("eye", "front camera", "interpreted_as", withHand, 500, nil)); d != nil {
		t.Fatalf("an empty vocabulary still derived %+v", d)
	}
}

// The witness's one-way rule (repair 7) under the ledger: world_facts is the
// only table that appears and no pre-existing table is touched — not
// organisms, not messages, not the locks. The companion of
// TestWitnessNeverWritesBack, which covers the witness without a facts file.
func TestWorldLedgerNeverWritesPreexistingTables(t *testing.T) {
	dir := witnessTestMesh(t)
	a := NewSwarmRegistry("earth", "earth")
	if err := a.Register(); err != nil {
		t.Fatal(err)
	}
	defer a.MeshDB.Close()
	a.Heartbeat(2, 262144, 0.10, 1.25, 4200)
	mesh := filepath.Join(dir, "mesh.db")

	probe, err := sql.Open("sqlite", mesh)
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	defer probe.Close()
	tableSet := func() []string {
		rows, err := probe.Query(`SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()
		var out []string
		for rows.Next() {
			var n string
			rows.Scan(&n)
			out = append(out, n)
		}
		sort.Strings(out)
		return out
	}
	// Every row of every pre-existing table, as text, before and after.
	dump := func() string {
		var b strings.Builder
		for _, tbl := range tableSet() {
			if tbl == "world_facts" {
				continue
			}
			rows, err := probe.Query(`SELECT * FROM "` + tbl + `"`)
			if err != nil {
				continue
			}
			cols, _ := rows.Columns()
			for rows.Next() {
				cells := make([]interface{}, len(cols))
				ptrs := make([]interface{}, len(cols))
				for i := range cells {
					ptrs[i] = &cells[i]
				}
				rows.Scan(ptrs...)
				fmt.Fprintf(&b, "%s%v\n", tbl, cells)
			}
			rows.Close()
		}
		return b.String()
	}
	tablesBefore, dumpBefore := tableSet(), dump()

	run := t.TempDir()
	facts := filepath.Join(run, "facts.jsonl")
	os.WriteFile(facts, []byte(strings.Join([]string{
		`{"source":"place","subject":"phone","predicate":"at_place","object":"Neve X","valid_from":100}`,
		`{"source":"place","subject":"phone","predicate":"at_place","object":"Ramot","valid_from":200}`,
		`{"source":"eye","subject":"rear camera","predicate":"interpreted_as","object":"A keyboard with white keys.","valid_from":210}`,
	}, "\n")+"\n"), 0644)

	l, err := openWorldLedger(mesh, facts, filepath.Join(run, "dna", "output"), filepath.Join(run, worldCursorFile))
	if err != nil {
		t.Fatal(err)
	}
	lines := l.tick(9000)
	l.Close()
	if len(lines) == 0 {
		t.Fatal("the ledger emitted nothing; this gate would pass vacuously")
	}

	after := tableSet()
	var added []string
	before := map[string]bool{}
	for _, n := range tablesBefore {
		before[n] = true
	}
	for _, n := range after {
		if !before[n] {
			added = append(added, n)
		}
	}
	if len(added) != 1 || added[0] != "world_facts" {
		t.Fatalf("the ledger added %v, want exactly [world_facts]", added)
	}
	if len(after) < len(tablesBefore) {
		t.Fatalf("tables disappeared: %v -> %v", tablesBefore, after)
	}
	if d := dump(); d != dumpBefore {
		t.Fatalf("a pre-existing table changed under the ledger:\nbefore\n%s\nafter\n%s", dumpBefore, d)
	}
	var fs int
	probe.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE name='field_steering'`).Scan(&fs)
	if fs != 0 {
		t.Fatal("field_steering exists — the arrow back is being drawn")
	}

	// §11's arrow, after the writer has been through the same file: the
	// witness's own connection still refuses every write, the ledger's table
	// included, and reads what the writer left.
	wdb, err := witnessOpenMesh(mesh)
	if err != nil {
		t.Fatal(err)
	}
	defer wdb.Close()
	if _, err := wdb.Exec(`UPDATE organisms SET entropy=0 WHERE id='earth'`); err == nil {
		t.Fatal("the witness handle wrote to organisms after the ingest ran; query_only is not in force")
	}
	if _, err := wdb.Exec(`INSERT INTO world_facts(source,subject,predicate,object,valid_from,recorded_at)
		VALUES('witness','phone','at_place','somewhere',1,1)`); err == nil {
		t.Fatal("the witness handle wrote a fact; the ledger has one writer and it is not the witness")
	}
	if _, err := wdb.Exec(`UPDATE world_facts SET valid_to=1 WHERE valid_to IS NULL`); err == nil {
		t.Fatal("the witness handle closed a fact")
	}
	w := witnessReadWorld(wdb)
	if w == nil || w.Facts == 0 || w.Open == 0 {
		t.Fatalf("the witness cannot see the ledger it may not write: %+v", w)
	}
	if w.Open > w.Facts {
		t.Fatalf("world summary is nonsense: %+v", w)
	}
}

// The witness's side of the ledger: a summary, read through the query_only
// handle, and nothing at all when no organ has ever written a fact.
func TestWitnessReadsTheLedgerItMayNotWrite(t *testing.T) {
	dir := witnessTestMesh(t)
	a := NewSwarmRegistry("earth", "earth")
	if err := a.Register(); err != nil {
		t.Fatal(err)
	}
	defer a.MeshDB.Close()
	a.Heartbeat(2, 1000, 0, 1.0, 10)
	mesh := filepath.Join(dir, "mesh.db")

	wdb, err := witnessOpenMesh(mesh)
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	defer wdb.Close()
	if w := witnessReadWorld(wdb); w != nil {
		t.Fatalf("a node whose senses never spoke reports a world of %+v, want nothing", w)
	}

	run := t.TempDir()
	facts := filepath.Join(run, "facts.jsonl")
	os.WriteFile(facts, []byte(strings.Join([]string{
		`{"source":"place","subject":"phone","predicate":"at_place","object":"Neve X","valid_from":100}`,
		`{"source":"place","subject":"sky","predicate":"reported_as","object":"fog","valid_from":100}`,
		`{"source":"place","subject":"sky","predicate":"reported_as","object":"clear sky","valid_from":400}`,
	}, "\n")+"\n"), 0644)
	l, err := openWorldLedger(mesh, facts, filepath.Join(run, "dna", "output"), filepath.Join(run, worldCursorFile))
	if err != nil {
		t.Fatal(err)
	}
	l.tick(9000)
	l.Close()

	w := witnessReadWorld(wdb)
	if w == nil {
		t.Fatal("the witness sees no ledger after three facts were filed")
	}
	if w.Facts != 3 || w.Open != 2 {
		t.Fatalf("world = %+v, want 3 facts of which 2 open (the fog is closed)", w)
	}
	if w.Recorded != 9000 {
		t.Fatalf("newest recorded_at = %v, want 9000", w.Recorded)
	}
}

// facts.jsonl is read as the organs write it: RFC3339 stamps, a provenance
// object, and a trailing partial line left alone until it is finished.
func TestWorldLedgerReadsWhatBashWrites(t *testing.T) {
	run := t.TempDir()
	path := filepath.Join(run, "facts.jsonl")
	complete := `{"source":"ears","subject":"microphone","predicate":"hearing","object":"speech","valid_from":"2026-09-13T22:21:08Z","provenance":{"window_s":12,"asr":"ears","model":"ggml-tiny.bin"}}` + "\n"
	partial := `{"source":"place","subject":"phone","predicate":"at_pl`
	if err := os.WriteFile(path, []byte(complete+partial), 0644); err != nil {
		t.Fatal(err)
	}
	cur := loadWorldCursor(filepath.Join(run, worldCursorFile))
	facts, err := worldReadFacts(path, cur, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 {
		t.Fatalf("%d facts, want 1 — the half-written line must wait: %+v", len(facts), facts)
	}
	if facts[0].validFrom != 1789338068 {
		t.Fatalf("valid_from = %v, want 1789338068 (2026-09-13T22:21:08Z)", facts[0].validFrom)
	}
	if w, ok := worldProvNumber(facts[0].prov, "window_s"); !ok || w != 12 {
		t.Fatalf("window_s = %v (%v)", w, ok)
	}
	if cur.Offset != int64(len(complete)) {
		t.Fatalf("cursor at %d, want %d — the partial line is not consumed", cur.Offset, len(complete))
	}

	// The rest of the line arrives; the next read picks it up from there.
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0644)
	f.WriteString(`ace","object":"Neve X","valid_from":"2026-09-13T22:10:52Z"}` + "\n")
	f.Close()
	facts, err = worldReadFacts(path, cur, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 || facts[0].Predicate != "at_place" || facts[0].Object != "Neve X" {
		t.Fatalf("the finished line read back as %+v", facts)
	}

	// A file that shrank was rotated: the cursor goes back to the start.
	os.WriteFile(path, []byte(complete), 0644)
	facts, err = worldReadFacts(path, cur, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 || cur.Offset != int64(len(complete)) {
		t.Fatalf("after rotation: %d facts, cursor %d", len(facts), cur.Offset)
	}
}

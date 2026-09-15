package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// The world ledger of change (ROADMAP item 10; molequla_new_logic.md §4-6,
// §15-16).
//
// The senses already write prose into ../dna/output/{world,sound,place}/. That
// prose is a stream of state: four passes on a still phone leave four fragments
// that say the same thing, and an organism eating them learns that the world
// repeats. §5 wants the other thing — the change. So beside the fragment every
// organ now writes one structured line into $MOLEQULA_RUN/senses/facts.jsonl,
// and a writer turns those lines into a bitemporal table:
//
//   world_facts(source, subject, predicate, object,
//               valid_from, valid_to, recorded_at, provenance)
//
// valid_from / valid_to is when the fact held in the world; recorded_at is when
// molequla learned it. The two are different clocks and a late ingest proves it.
// Nothing is deleted and nothing is overwritten: a contradicting observation
// closes the old row with valid_to and opens a new one, and both stay.
//
// §6 — perception is not ground truth. The eye's sentence is stored under the
// predicate `interpreted_as` with the camera, the weights, the quantisation and
// the conditions of the pass in `provenance`; it is never stored as `is`. The
// balcony that Ocelli read as a bathroom is a true record of an interpretation
// and a false record of a room, and the table says which of the two it holds.
//
// A change, and only a change, becomes a fragment: one line into
// ../dna/output/world/ under the same gen_<unix>_<seq>.txt name dnaListNew
// orders by, with the same bracketed header senses.sh writes. Repeated identical
// state produces nothing.
//
// The writer is its own process — `molequla --world-ingest [--once]`, run by
// phone1/senses.sh at the end of every pass and by the scheduler on its own.
// It is deliberately not the witness: the witness's mesh handle carries
// PRAGMA query_only (witness.go:167) and §11's arrow says it keeps carrying it,
// so
// a witness that wrote the ledger would be the first hole in it. The witness
// reads world_facts on its tick and prints how large the world's memory is;
// that is all it may do with it.
//
// What the writer may touch is one table. world_facts is the only thing it
// creates and the only thing it writes; `organisms`, `messages` and the locks
// are read-only to it in practice and gated in test.
// ═══════════════════════════════════════════════════════════════════════════════

const (
	worldCursorFile = "world_cursor.json"
	worldDNASource  = "world"
)

// worldIngestMode is `--world-ingest`: this process is the ledger's writer and
// loads no model. Set in parseCLIArgs, dispatched in main, beside witnessMode.
var worldIngestMode bool

// worldFact is one observation as an organ writes it into facts.jsonl. Only
// source, subject, predicate and object are required; valid_from defaults to
// the ingest clock, which is the honest reading of a line that did not say
// when it held.
type worldFact struct {
	Source     string          `json:"source"`
	Subject    string          `json:"subject"`
	Predicate  string          `json:"predicate"`
	Object     string          `json:"object"`
	ValidFrom  json.RawMessage `json:"valid_from,omitempty"`
	Provenance json.RawMessage `json:"provenance,omitempty"`

	validFrom float64
	prov      map[string]interface{}
}

// worldRow is a row of world_facts as the ledger reads it back.
type worldRow struct {
	ID        int64
	Source    string
	Subject   string
	Predicate string
	Object    string
	ValidFrom float64
}

// worldChange is one emitted change: the sentence, and the id of the row that
// opened it — which is also the sequence number of its fragment, so fragment
// order and ledger order are the same order.
type worldChange struct {
	Line string
	Row  int64
	At   float64
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema
// ─────────────────────────────────────────────────────────────────────────────

// worldMigrate creates the table if it is missing and is safe to run against a
// mesh.db that already has it — the same shape as the global_step ALTER in
// initMeshDB: run every open, fail on nothing that is already there.
func worldMigrate(db *sql.DB) error {
	// `id INTEGER PRIMARY KEY` and not AUTOINCREMENT: AUTOINCREMENT keeps its
	// high-water mark in sqlite_sequence, which this mesh already has because
	// `messages` uses it — the ledger would be writing a row of a table an
	// organism owns, and the one-way gate caught exactly that. Nothing here is
	// ever deleted, so the plain rowid is monotonic anyway.
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS world_facts(
		id INTEGER PRIMARY KEY,
		source TEXT NOT NULL,
		subject TEXT NOT NULL,
		predicate TEXT NOT NULL,
		object TEXT NOT NULL,
		valid_from REAL NOT NULL,
		valid_to REAL,
		recorded_at REAL NOT NULL,
		provenance TEXT)`); err != nil {
		return err
	}
	// The one query the ledger makes per fact: the open row of a triple.
	if _, err := db.Exec(`CREATE INDEX IF NOT EXISTS world_facts_open
		ON world_facts(source, subject, predicate, valid_to)`); err != nil {
		return err
	}
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// The cursor into facts.jsonl
// ─────────────────────────────────────────────────────────────────────────────

// worldCursor is a byte offset into facts.jsonl, persisted in the writer's
// working directory so re-running the ingest over the same file changes nothing. A file shorter
// than the offset was rotated or truncated, and the cursor goes back to zero;
// re-reading old lines is harmless anyway, because an observation identical to
// the open row is not a change (worldIngest).
type worldCursor struct {
	Offset int64 `json:"offset"`
	path   string
}

func loadWorldCursor(path string) *worldCursor {
	c := &worldCursor{path: path}
	if data, err := os.ReadFile(path); err == nil {
		var on struct {
			Offset int64 `json:"offset"`
		}
		if json.Unmarshal(data, &on) == nil && on.Offset >= 0 {
			c.Offset = on.Offset
		}
	}
	return c
}

// save writes the cursor atomically, so a crash mid-write leaves the previous
// position rather than a truncated file (dnaCursor.save does the same).
func (c *worldCursor) save() {
	if c == nil || c.path == "" {
		return
	}
	data, err := json.Marshal(struct {
		Offset int64 `json:"offset"`
	}{c.Offset})
	if err != nil {
		return
	}
	tmp := c.path + ".tmp"
	if os.WriteFile(tmp, data, 0644) == nil {
		os.Rename(tmp, c.path)
	}
}

// worldReadFacts returns the facts written after the cursor and advances it to
// the end of the last complete line it read. A trailing partial line — an organ
// caught mid-append — is left for the next tick.
func worldReadFacts(path string, cur *worldCursor, now float64) ([]worldFact, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if fi.Size() < cur.Offset {
		cur.Offset = 0 // rotated or truncated
	}
	if fi.Size() == cur.Offset {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if _, err := f.Seek(cur.Offset, 0); err != nil {
		return nil, err
	}
	var out []worldFact
	off := cur.Offset
	rd := bufio.NewReader(f)
	for {
		line, err := rd.ReadString('\n')
		if err != nil {
			break // no newline yet: the line is still being written
		}
		off += int64(len(line))
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var fact worldFact
		if json.Unmarshal([]byte(line), &fact) != nil {
			continue // a line that is not JSON is not a fact; the offset still moves
		}
		if fact.Source == "" || fact.Subject == "" || fact.Predicate == "" {
			continue
		}
		fact.validFrom = worldParseTime(fact.ValidFrom, now)
		fact.prov = worldParseProv(fact.Provenance)
		out = append(out, fact)
	}
	cur.Offset = off
	return out, nil
}

// worldParseTime accepts either an RFC3339 string (what `date -u +%FT%TZ`
// prints) or a unix number, and falls back to the ingest clock.
func worldParseTime(raw json.RawMessage, fallback float64) float64 {
	if len(raw) == 0 {
		return fallback
	}
	var num float64
	if json.Unmarshal(raw, &num) == nil && num > 0 {
		return num
	}
	var s string
	if json.Unmarshal(raw, &s) != nil || s == "" {
		return fallback
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05Z0700", "2006-01-02 15:04:05"} {
		if t, err := time.Parse(layout, s); err == nil {
			return float64(t.UnixMilli()) / 1000.0
		}
	}
	if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
		return v
	}
	return fallback
}

func worldParseProv(raw json.RawMessage) map[string]interface{} {
	if len(raw) == 0 {
		return nil
	}
	var m map[string]interface{}
	if json.Unmarshal(raw, &m) != nil {
		return nil
	}
	return m
}

func worldProvString(prov map[string]interface{}, key string) (string, bool) {
	if prov == nil {
		return "", false
	}
	switch v := prov[key].(type) {
	case string:
		return v, v != ""
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64), true
	}
	return "", false
}

func worldProvNumber(prov map[string]interface{}, key string) (float64, bool) {
	if prov == nil {
		return 0, false
	}
	switch v := prov[key].(type) {
	case float64:
		return v, true
	case string:
		f, err := strconv.ParseFloat(v, 64)
		return f, err == nil
	}
	return 0, false
}

// ─────────────────────────────────────────────────────────────────────────────
// Sameness, and the one measured gate in it
// ─────────────────────────────────────────────────────────────────────────────

// worldSameObject decides whether an observation repeats the open row or
// contradicts it. Text is compared as text; a position is compared in metres,
// because a fix that wanders inside its own accuracy has not moved and a ledger
// that says it has would emit a change line every pass. CFG.WorldMoveMeters is
// that gate and it is the same 50 m senses.sh already uses against
// senses/place.last — measured against a network fix that reports 13-14 m of
// accuracy on this phone.
func worldSameObject(predicate, a, b string) bool {
	if predicate == "at_position" {
		if d, ok := worldDistanceM(a, b); ok {
			return d <= CFG.WorldMoveMeters
		}
	}
	return a == b
}

// worldDistanceM is the haversine between two "lat,lon" strings, in metres —
// the same formula as dist_m in phone1/senses.sh.
func worldDistanceM(a, b string) (float64, bool) {
	la1, lo1, ok1 := worldLatLon(a)
	la2, lo2, ok2 := worldLatLon(b)
	if !ok1 || !ok2 {
		return 0, false
	}
	const r = 6371000.0
	rad := math.Pi / 180
	dla := (la2 - la1) * rad
	dlo := (lo2 - lo1) * rad
	s := math.Sin(dla/2)*math.Sin(dla/2) +
		math.Cos(la1*rad)*math.Cos(la2*rad)*math.Sin(dlo/2)*math.Sin(dlo/2)
	return 2 * r * math.Atan2(math.Sqrt(s), math.Sqrt(1-s)), true
}

func worldLatLon(s string) (float64, float64, bool) {
	parts := strings.SplitN(s, ",", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}
	la, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	lo, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	return la, lo, true
}

// ─────────────────────────────────────────────────────────────────────────────
// Derived facts
// ─────────────────────────────────────────────────────────────────────────────

// worldDerive returns the facts the ledger reads out of an organ's own fact.
// Today there is one: whether a person is in the camera's frame, taken
// lexically from the eye's sentence. It is a weak reading and its provenance
// says so — `derived: lexical`, with the words that matched and the sentence
// they matched in — but it is what turns two unrelated descriptions into
// "a person entered the frame", which is the §5 line ROADMAP 10 asks for.
// CFG.WorldPersonWords is the vocabulary and an empty list turns the derivation
// off entirely.
func worldDerive(f worldFact) []worldFact {
	if f.Source != "eye" || f.Predicate != "interpreted_as" || len(CFG.WorldPersonWords) == 0 {
		return nil
	}
	matched := worldMatchWords(f.Object, CFG.WorldPersonWords)
	object := "no"
	if len(matched) > 0 {
		object = "yes"
	}
	prov := map[string]interface{}{
		"derived": "lexical",
		"from":    "interpreted_as",
		"text":    f.Object,
	}
	if len(matched) > 0 {
		prov["matched"] = matched
	}
	if cam, ok := worldProvString(f.prov, "camera"); ok {
		prov["camera"] = cam
	}
	raw, err := json.Marshal(prov)
	if err != nil {
		return nil
	}
	return []worldFact{{
		Source:     f.Source,
		Subject:    f.Subject,
		Predicate:  "sees_person",
		Object:     object,
		Provenance: raw,
		validFrom:  f.validFrom,
		prov:       prov,
	}}
}

// worldMatchWords returns the vocabulary words present in text as whole words,
// lowercased, in the order of the vocabulary and without repeats. A word
// immediately preceded by one of CFG.WorldNegationWords does not count: the eye
// writes "with no people or text visible" about an empty room, and the first
// live run of this rule read a person into it (MOLEQULALOG2.md, 2026-09-15).
// It is a lexical rule guarding a lexical rule and it will miss longer-range
// negation; the provenance of every derived fact says `derived: lexical` so
// that a later reading can disagree with it.
func worldMatchWords(text string, vocab []string) []string {
	var tokens []string
	cur := strings.Builder{}
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, strings.ToLower(cur.String()))
			cur.Reset()
		}
	}
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			cur.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	negated := map[string]bool{}
	for _, w := range CFG.WorldNegationWords {
		negated[strings.ToLower(w)] = true
	}
	words := map[string]bool{}
	for i, tok := range tokens {
		if i > 0 && negated[tokens[i-1]] {
			continue
		}
		words[tok] = true
	}
	var out []string
	for _, w := range vocab {
		if words[strings.ToLower(w)] {
			out = append(out, strings.ToLower(w))
		}
	}
	return out
}

// ─────────────────────────────────────────────────────────────────────────────
// The change vocabulary
// ─────────────────────────────────────────────────────────────────────────────

// worldChangeLine is the sentence a change becomes, or "" when the change is
// not worth a fragment. prev is nil when the ledger has never held this triple.
// Everything here is English prose for an organism to eat; nothing in it is a
// key another program parses.
func worldChangeLine(prev *worldRow, f worldFact) string {
	subject := f.Subject
	switch f.Predicate {
	case "at_place":
		if prev == nil {
			return fmt.Sprintf("The %s is at %s. Where it was before that, the ledger does not know.", subject, f.Object)
		}
		if m, ok := worldProvNumber(f.prov, "moved_m"); ok {
			return fmt.Sprintf("The %s moved from %s to %s, %.0f m from the last fix.", subject, prev.Object, f.Object, m)
		}
		return fmt.Sprintf("The %s moved from %s to %s.", subject, prev.Object, f.Object)

	case "at_position":
		if prev == nil {
			return "" // the at_place line of the same pass already said it
		}
		d, ok := worldDistanceM(prev.Object, f.Object)
		if !ok {
			return ""
		}
		if place, have := worldProvString(f.prov, "place"); have {
			return fmt.Sprintf("The %s moved %.0f m and is still at %s.", subject, d, place)
		}
		return fmt.Sprintf("The %s moved %.0f m.", subject, d)

	case "reported_as":
		if prev == nil {
			return fmt.Sprintf("The %s is %s.", subject, f.Object)
		}
		return fmt.Sprintf("The %s changed from %s to %s.", subject, prev.Object, f.Object)

	case "interpreted_as":
		if f.Source == "ears" {
			return fmt.Sprintf("The %s heard: \"%s\"", subject, f.Object)
		}
		if prev == nil {
			return fmt.Sprintf("The %s opened on: \"%s\"", subject, f.Object)
		}
		return fmt.Sprintf("The %s's scene changed from \"%s\" to \"%s\"", subject, prev.Object, f.Object)

	case "hearing":
		secs, haveSecs := worldProvNumber(f.prov, "window_s")
		if f.Object == "speech" {
			if haveSecs {
				return fmt.Sprintf("Speech was heard through the %s for %.0f s.", subject, secs)
			}
			return fmt.Sprintf("Speech was heard through the %s.", subject)
		}
		if prev == nil {
			return "" // silence that was always silence is not news
		}
		if haveSecs {
			return fmt.Sprintf("The %s stopped hearing speech: %.0f s of room noise instead.", subject, secs)
		}
		return fmt.Sprintf("The %s stopped hearing speech.", subject)

	case "sees_person":
		if prev == nil {
			return "" // entered and left both need a before
		}
		if f.Object == "yes" {
			return fmt.Sprintf("A person entered the %s's frame.", subject)
		}
		return fmt.Sprintf("A person left the %s's frame.", subject)
	}
	// An unknown predicate still belongs in the table; it just does not get a
	// hand-written sentence, and the generic one says exactly what happened.
	if prev == nil {
		return fmt.Sprintf("The %s: %s is %s.", f.Source, subject, f.Object)
	}
	return fmt.Sprintf("The %s: %s changed from %s to %s.", f.Source, subject, prev.Object, f.Object)
}

// ─────────────────────────────────────────────────────────────────────────────
// Ingest
// ─────────────────────────────────────────────────────────────────────────────

// worldOpenRow returns the open row of a triple — the one with no valid_to —
// or nil. There is at most one by construction: every insert of an open row
// closes the previous one in the same transaction.
func worldOpenRow(db *sql.DB, f worldFact) (*worldRow, error) {
	var r worldRow
	err := db.QueryRow(`SELECT id, source, subject, predicate, object, valid_from
		FROM world_facts
		WHERE source=? AND subject=? AND predicate=? AND valid_to IS NULL
		ORDER BY valid_from DESC, id DESC LIMIT 1`,
		f.Source, f.Subject, f.Predicate).
		Scan(&r.ID, &r.Source, &r.Subject, &r.Predicate, &r.Object, &r.ValidFrom)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &r, nil
}

// worldIngest writes the facts into the ledger and returns the changes worth a
// fragment. The four cases, in order:
//
//   - nothing open for the triple: the fact opens one, and the ledger says what
//     it has learned for the first time;
//   - the open row says the same thing: nothing is written and nothing is said
//     (§5 — repeated identical state produces no fragment);
//   - the fact is newer and says something else: the open row is closed at the
//     fact's valid_from, the fact opens a new row, and the change is a line;
//   - the fact is older than the open row — a late ingest, recorded_at far past
//     valid_from: it is inserted as history, already closed at the open row's
//     valid_from, and says nothing. It contradicts a belief molequla has since
//     revised, and the revision stands.
func worldIngest(db *sql.DB, facts []worldFact, now float64) ([]worldChange, error) {
	var changes []worldChange
	for _, f := range facts {
		all := append([]worldFact{f}, worldDerive(f)...)
		for _, fact := range all {
			if fact.validFrom <= 0 {
				fact.validFrom = now
			}
			prev, err := worldOpenRow(db, fact)
			if err != nil {
				return changes, err
			}
			if prev != nil && worldSameObject(fact.Predicate, prev.Object, fact.Object) {
				continue
			}
			prov := ""
			if len(fact.Provenance) > 0 {
				prov = string(fact.Provenance)
			}
			if prev != nil && fact.validFrom <= prev.ValidFrom {
				// Late ingest: history, closed on arrival, no line.
				if _, err := db.Exec(`INSERT INTO world_facts
					(source, subject, predicate, object, valid_from, valid_to, recorded_at, provenance)
					VALUES(?,?,?,?,?,?,?,?)`,
					fact.Source, fact.Subject, fact.Predicate, fact.Object,
					fact.validFrom, prev.ValidFrom, now, prov); err != nil {
					return changes, err
				}
				continue
			}
			if prev != nil {
				if _, err := db.Exec(`UPDATE world_facts SET valid_to=? WHERE id=?`,
					fact.validFrom, prev.ID); err != nil {
					return changes, err
				}
			}
			res, err := db.Exec(`INSERT INTO world_facts
				(source, subject, predicate, object, valid_from, valid_to, recorded_at, provenance)
				VALUES(?,?,?,?,?,NULL,?,?)`,
				fact.Source, fact.Subject, fact.Predicate, fact.Object,
				fact.validFrom, now, prov)
			if err != nil {
				return changes, err
			}
			id, _ := res.LastInsertId()
			if line := worldChangeLine(prev, fact); line != "" {
				changes = append(changes, worldChange{Line: line, Row: id, At: now})
			}
		}
	}
	return changes, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Fragments
// ─────────────────────────────────────────────────────────────────────────────

// worldWriteChanges drops one fragment per change into <dnaBase>/world/, under
// the gen_<unix>_<seq>.txt name dnaListNew orders by and behind the same
// bracketed header senses.sh writes. The sequence is the ledger row id, so the
// order of the fragments is the order of the ledger; a name already taken by an
// organ's own fragment in the same second is stepped past rather than
// overwritten, because the two writers share a directory and not a counter.
func worldWriteChanges(dnaBase string, changes []worldChange) ([]string, error) {
	if len(changes) == 0 {
		return nil, nil
	}
	dir := filepath.Join(dnaBase, worldDNASource)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	var written []string
	for _, c := range changes {
		stamp := time.Unix(int64(c.At), 0).UTC().Format("2006-01-02T15:04:05Z")
		body := fmt.Sprintf("[world %s] %s\n", stamp, c.Line)
		seq := c.Row
		var name string
		for {
			name = fmt.Sprintf("gen_%d_%d.txt", int64(c.At), seq)
			if _, err := os.Stat(filepath.Join(dir, name)); os.IsNotExist(err) {
				break
			}
			seq++
		}
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0644); err != nil {
			return written, err
		}
		written = append(written, name)
	}
	return written, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// The ledger as a witness step
// ─────────────────────────────────────────────────────────────────────────────

// worldLedger is what the ingest process holds between passes: the write
// handle on world_facts, the cursor into facts.jsonl, and where the fragments
// go.
type worldLedger struct {
	db        *sql.DB
	facts     string
	dnaBase   string
	cursor    *worldCursor
	ingested  int
	emitted   int
	lastError string
}

// openWorldLedger opens the writer's handle on the mesh — the one handle in
// the tree that may write world_facts, and it writes nothing else.
func openWorldLedger(meshPath, factsPath, dnaBase, cursorPath string) (*worldLedger, error) {
	db, err := sql.Open("sqlite", meshPath)
	if err != nil {
		return nil, err
	}
	if err := worldMigrate(db); err != nil {
		db.Close()
		return nil, err
	}
	return &worldLedger{db: db, facts: factsPath, dnaBase: dnaBase, cursor: loadWorldCursor(cursorPath)}, nil
}

func (l *worldLedger) Close() {
	if l != nil && l.db != nil {
		l.db.Close()
	}
}

// tick reads whatever the organs appended since the last tick, writes it into
// the ledger and leaves the changes in the DNA field. It returns the lines it
// emitted, for the witness's own record.
func (l *worldLedger) tick(now float64) []string {
	if l == nil || l.db == nil {
		return nil
	}
	facts, err := worldReadFacts(l.facts, l.cursor, now)
	if err != nil {
		if !os.IsNotExist(err) {
			l.lastError = err.Error()
		}
		return nil
	}
	if len(facts) == 0 {
		l.cursor.save()
		return nil
	}
	changes, err := worldIngest(l.db, facts, now)
	if err != nil {
		l.lastError = err.Error()
	}
	if _, err := worldWriteChanges(l.dnaBase, changes); err != nil {
		l.lastError = err.Error()
	}
	// The cursor advances only after the facts are in the table: a crash
	// between the two re-reads the lines, and an observation identical to the
	// open row is not a change, so re-reading them costs nothing.
	l.cursor.save()
	l.ingested += len(facts)
	l.emitted += len(changes)
	out := make([]string, 0, len(changes))
	for _, c := range changes {
		out = append(out, c.Line)
	}
	return out
}

// runWorldIngest is the writer process: read what the organs appended, file it,
// leave the changes in the DNA field, say what changed. phone1/senses.sh runs
// it with --once at the end of every pass; without --once it stays up and
// re-reads on the witness's interval, which is what the scheduler would use to
// run it beside a colony session. Returns the exit code.
//
//	molequla --world-ingest [--once] [--world-facts <path>]
//
// Run it from a sibling of the organism directories, like the witness, so that
// ../dna/output and ../senses/facts.jsonl resolve to the same tree.
func runWorldIngest(interval float64, once bool) int {
	if CFG.WorldFactsPath == "" {
		fmt.Fprintln(os.Stderr, "world-ingest: --world-facts is empty; nothing to read")
		return 2
	}
	meshPath := filepath.Join(swarmDir, "mesh.db")
	if _, err := os.Stat(meshPath); err != nil {
		fmt.Fprintf(os.Stderr, "world-ingest: no mesh at %s: %v\n", meshPath, err)
		return 2
	}
	l, err := openWorldLedger(meshPath, CFG.WorldFactsPath, "../dna/output", worldCursorFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "world-ingest:", err)
		return 2
	}
	defer l.Close()
	fmt.Fprintf(os.Stderr, "[world] mesh=%s facts=%s dna=../dna/output/%s\n",
		meshPath, CFG.WorldFactsPath, worldDNASource)
	if interval <= 0 {
		interval = 1
	}
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	for {
		now := float64(time.Now().UnixMilli()) / 1000.0
		before := l.ingested
		for _, line := range l.tick(now) {
			fmt.Printf("[world] %s\n", line)
		}
		if l.lastError != "" {
			fmt.Fprintln(os.Stderr, "[world]", l.lastError)
			l.lastError = ""
		}
		if once {
			fmt.Fprintf(os.Stderr, "[world] %d facts read, %d changes emitted\n",
				l.ingested-before, l.emitted)
			return 0
		}
		select {
		case <-sigCh:
			fmt.Fprintf(os.Stderr, "[world] stopped after %d facts, %d changes\n", l.ingested, l.emitted)
			return 0
		case <-time.After(time.Duration(interval * float64(time.Second))):
		}
	}
}

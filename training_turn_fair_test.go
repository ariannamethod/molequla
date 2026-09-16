package main

import (
	"fmt"
	"testing"
	"time"
)

// The turn made yieldable and the queue made fair (training_turn.go). Step 1
// bounded how many tapes exist at once and left unbounded how long one phase may
// hold the one tape: the tick loop took one turn around all three sub-phases of a
// stage-5 warmup, and a stage-5 warmup measures between 1 200.9 s and 4 796.3 s
// for its 2 000 steps in molequla-run/*/*.stdout. These gates are the ceiling
// that cuts a phase at a step boundary, the ordering that hands out equal seconds
// of tape instead of equal turns, and a replay of a session shaped from those
// measurements through both orderings on one binary.

// withTrainTurn sets the two knobs for one test and puts them back. Passing the
// pre-2026-09-16 values (0, false) is how every gate here is shown red.
func withTrainTurn(t *testing.T, ceiling float64, fair bool) {
	t.Helper()
	savedCeiling, savedFair := CFG.TrainTurnCeilingSeconds, CFG.TrainTurnFairSpend
	CFG.TrainTurnCeilingSeconds, CFG.TrainTurnFairSpend = ceiling, fair
	t.Cleanup(func() {
		CFG.TrainTurnCeilingSeconds, CFG.TrainTurnFairSpend = savedCeiling, savedFair
	})
}

// withSimClock replaces the queue's clock with one the test drives, so a
// two-hour session is replayed in milliseconds and an arrival time is where the
// test puts it rather than where the scheduler happened to land. Only the queue
// reads this clock; the ceiling and the refresher keep the real one.
func withSimClock(t *testing.T, at *float64) {
	t.Helper()
	saved := trainTurnNow
	trainTurnNow = func() float64 { return *at }
	t.Cleanup(func() { trainTurnNow = saved })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ordering: equal seconds of tape, not equal turns
// ═══════════════════════════════════════════════════════════════════════════════

// A waiter counts only once it has polled: AcquireTrainingTurn inserts the row
// and then tests it, so an organism that has not yet asked is invisible to the
// comparator. Live that window is CFG.TrainingTurnPollSeconds against a burst of
// tens of seconds, so every sibling is registered long before the tape frees.
// These gates reproduce that shape rather than assuming it away: a holder keeps
// the turn while the waiters queue behind it, and only then lets go.
// The holder takes the lock row directly rather than through
// AcquireTrainingTurn: a fixture whose own success depends on the ordering under
// test would report the ordering's verdict in its own voice, and the first red
// run of this file did exactly that — "fire could not take the turn" where the
// finding was 4 turns each.
func queueBehind(t *testing.T, holder *SwarmRegistry, waiters ...*SwarmRegistry) {
	t.Helper()
	if _, err := holder.MeshDB.Exec(
		"INSERT OR REPLACE INTO training_lock(organism_id, acquired_at) VALUES(?,?)",
		holder.OrganismID, trainTurnNow()); err != nil {
		t.Fatalf("the fixture could not busy the tape: %v", err)
	}
	for _, sr := range waiters {
		if sr.AcquireTrainingTurn(trainTurnBurst) {
			t.Fatalf("%s took a held turn", sr.OrganismID)
		}
	}
	holder.ReleaseTrainingLock()
}

// admitOne lets every waiter poll a free turn and returns the one the queue chose.
func admitOne(t *testing.T, waiters ...*SwarmRegistry) *SwarmRegistry {
	t.Helper()
	var winner *SwarmRegistry
	for _, sr := range waiters {
		if sr.AcquireTrainingTurn(trainTurnBurst) {
			if winner != nil {
				t.Fatalf("two organisms took the same turn: %s and %s", winner.OrganismID, sr.OrganismID)
			}
			winner = sr
		}
	}
	if winner == nil {
		t.Fatal("nobody was admitted to a free turn")
	}
	return winner
}

// §2.3's last key was the whole of the ordering below a grown organism, and
// longest-wait is not fairness: it hands every waiter the same number of turns,
// and a turn is not the same amount of tape. Measured 2026-09-16 the adult's
// burst is a median 76.0 s and the teen's 58.0 s, so equal turns give the adult
// 31 % more of the tape than the teen. Here three waiters differ in what they
// have already had, their arrival order is the exact inverse of it, and each is
// polled in arrival order so that a comparator reading arrival would let earth
// through on the first round.
func TestTheLeastSpentOrganismGoesFirstNotTheEarliestArrival(t *testing.T) {
	withTrainTurn(t, 120.0, true)
	withTrainingTTL(t, 1e9) // the TTL is another gate's subject; never stale here
	now := 1000.0
	withSimClock(t, &now)
	r := trainingTurnMesh(t, "fire", "earth", "water", "air")
	fire, earth, water, air := r[0], r[1], r[2], r[3]

	// Arrival is the inverse of spend: earth asks first and has had the most.
	earth.trainSpentMs.Store(600_000)
	water.trainSpentMs.Store(300_000)
	air.trainSpentMs.Store(120_000)

	pool := []*SwarmRegistry{earth, water, air}
	for i, want := range []string{"air", "water", "earth"} {
		queueBehind(t, fire, pool...)
		winner := admitOne(t, pool...)
		if winner.OrganismID != want {
			t.Fatalf("turn %d went to %q, want %q — the order is by arrival, not by spend", i+1, winner.OrganismID, want)
		}
		winner.ReleaseTrainingLock()
		// The organism that has trained leaves the queue for its tick body, as
		// the tick loop does; the two behind it are still owed a turn.
		next := pool[:0]
		for _, sr := range pool {
			if sr != winner {
				next = append(next, sr)
			}
		}
		pool = next
		now += 1.0
	}
}

// Least-spent-first must not starve the organism that has trained most: once its
// siblings have caught up it goes again. Starvation is a bug, not a policy. air
// starts 150 s behind earth and every turn is one 60 s burst, so air takes the
// first three and then they alternate.
func TestLeastSpentFirstDoesNotStarveTheSpender(t *testing.T) {
	withTrainTurn(t, 120.0, true)
	withTrainingTTL(t, 1e9)
	now := 1000.0
	withSimClock(t, &now)
	r := trainingTurnMesh(t, "fire", "earth", "air")
	fire, earth, air := r[0], r[1], r[2]

	earth.trainSpentMs.Store(200_000)
	air.trainSpentMs.Store(50_000)
	turns := map[string]int{}
	for i := 0; i < 8; i++ {
		queueBehind(t, fire, earth, air)
		winner := admitOne(t, earth, air)
		winner.AddTrainingSpent(60 * time.Second)
		winner.ReleaseTrainingLock()
		turns[winner.OrganismID]++
		now += 60.0
	}
	if turns["earth"] == 0 {
		t.Fatalf("the organism that had trained most got none of 8 turns: %v", turns)
	}
	if turns["air"] <= turns["earth"] {
		t.Fatalf("the organism that had trained least did not get more turns: %v", turns)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// The ceiling: a long phase releases the turn and keeps what it ran
// ═══════════════════════════════════════════════════════════════════════════════

// The chunk loop itself, over a phase whose length the test chooses. What must
// hold: every step asked for is eventually run; the turn is genuinely given back
// between chunks, which shows as a queue row written fresh on each chunk (the row
// only comes back new if ReleaseTrainingLock deleted it); the lock belongs to the
// holder while a chunk is inside it; and the grown organism's head-of-queue key
// is spent on the first chunk only — without that decay the release is theatre,
// because nothing outranks the first key.
func TestALongPhaseReleasesTheTurnBetweenChunksAndKeepsItsSteps(t *testing.T) {
	withTrainTurn(t, 120.0, true)
	withTrainingTTL(t, 1e9)
	now := 1000.0
	withSimClock(t, &now)
	savedPoll := CFG.TrainingTurnPollSeconds
	CFG.TrainingTurnPollSeconds = 0.01
	t.Cleanup(func() { CFG.TrainingTurnPollSeconds = savedPoll })
	r := trainingTurnMesh(t, "water", "air")
	water, air := r[0], r[1]

	const wantSteps = 1000
	chunks := 0
	var priorities []int
	var sinces []float64
	ran := ntTrainInTurns(water, true, trainTurnGrown, "warmup", wantSteps, func(remaining int) int {
		chunks++
		var p int
		var since float64
		if err := water.MeshDB.QueryRow(
			"SELECT priority, since FROM training_queue WHERE organism_id='water'").Scan(&p, &since); err != nil {
			t.Fatalf("chunk %d: the holder has no queue row: %v", chunks, err)
		}
		priorities = append(priorities, p)
		sinces = append(sinces, since)
		var holder string
		if err := water.MeshDB.QueryRow("SELECT COALESCE(MIN(organism_id),'') FROM training_lock").Scan(&holder); err != nil {
			t.Fatalf("chunk %d: %v", chunks, err)
		}
		if holder != "water" {
			t.Fatalf("chunk %d ran with the lock held by %q", chunks, holder)
		}
		now += 1.0 // so the next chunk's row is distinguishable from this one's
		step := 100
		if remaining < step {
			step = remaining
		}
		return step
	})
	// The turn is free once the phase is done, and a sibling takes it.
	if !air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the turn was not free after the last chunk")
	}
	air.ReleaseTrainingLock()

	if ran != wantSteps {
		t.Fatalf("the phase ran %d of %d steps — a yield lost the steps it had taken", ran, wantSteps)
	}
	if chunks != 10 {
		t.Fatalf("the phase took %d chunks of 100 steps, want 10", chunks)
	}
	for i := 1; i < len(sinces); i++ {
		if sinces[i] <= sinces[i-1] {
			t.Fatalf("chunk %d kept the queue row of chunk %d (since %.1f then %.1f): the turn was never given back",
				i+1, i, sinces[i-1], sinces[i])
		}
	}
	if priorities[0] != trainTurnGrown {
		t.Fatalf("the first chunk asked at priority %d, want trainTurnGrown (%d)", priorities[0], trainTurnGrown)
	}
	for i, p := range priorities[1:] {
		if p != trainTurnBurst {
			t.Fatalf("chunk %d asked at priority %d, want trainTurnBurst (%d): a grown organism that keeps the first key wins back every turn it releases",
				i+2, p, trainTurnBurst)
		}
	}
}

// A chunk that can run nothing must end the phase rather than spin: the loop
// pays a tape rebuild on every attempt, so an attempt that never advances is an
// unbounded cost. (ntTrainCore returns zero steps when no corpus line is long
// enough to train on.)
func TestAPhaseWhoseChunkRunsNothingStops(t *testing.T) {
	withTrainTurn(t, 120.0, true)
	withTrainingTTL(t, 1e9)
	now := 1000.0
	withSimClock(t, &now)
	r := trainingTurnMesh(t, "earth")
	earth := r[0]

	calls := 0
	ran := ntTrainInTurns(earth, true, trainTurnBurst, "burst", 500, func(int) int {
		calls++
		if calls > 4 {
			t.Fatalf("a chunk that ran no steps was retried %d times", calls)
		}
		return 0
	})
	if ran != 0 || calls != 1 {
		t.Fatalf("ran=%d calls=%d, want 0 and 1", ran, calls)
	}
}

// The ceiling over the real step loop, not over a stand-in: ntTrainCore must stop
// at a step boundary with the deadline behind it, run at least one step so the
// chunk loop advances, mirror what it ran back into model.Base, and charge the
// post-growth freeze counter for the steps it ran and not the steps it was asked
// for. That last one is the duplicated-invariant pattern this tree has paid for
// once already (ff6ad49): a counter charged per request instead of per step would
// have the colony leave its freeze the moment a burst is cut in two.
func TestTheCeilingCutsARealPhaseAndChargesOnlyWhatItRan(t *testing.T) {
	saved := CFG
	t.Cleanup(func() { CFG = saved })
	model, tok, _ := parityModel(t, 2)
	docs := []string{
		"a phase cut at the ceiling keeps the steps it has already taken",
		"the tape mirrors its weights back on the way out whatever stopped it",
		"progress is kept because pullBack does not care why the loop ended",
	}
	const ask = 20

	// Disarmed: the whole request runs, and this is the arm that proves the cut
	// below is the ceiling's doing and not the model's.
	disarmTrainTurnCeiling()
	g0 := model.globalStep
	full := ntBurstTrain(model, tok, docs, ask, 1e-4)
	if full != ask || model.globalStep-g0 != ask {
		t.Fatalf("with no ceiling the burst ran %d of %d steps (globalStep +%d)", full, ask, model.globalStep-g0)
	}

	// Armed and already past: one step, then out.
	model.growthFreezeRemaining = ask
	g1 := model.globalStep
	trainTurnDeadlineMs.Store(time.Now().UnixMilli() - 1000)
	cut := ntBurstTrain(model, tok, docs, ask, 1e-4)
	disarmTrainTurnCeiling()
	if cut != 1 {
		t.Fatalf("the ceiling cut the burst at %d steps, want exactly 1 (a chunk must always advance, and must not run past the deadline)", cut)
	}
	if got := model.globalStep - g1; got != cut {
		t.Fatalf("the burst reported %d steps and advanced globalStep by %d", cut, got)
	}
	if model.growthFreezeRemaining != ask-cut {
		t.Fatalf("the freeze counter fell from %d to %d for %d step(s) run", ask, model.growthFreezeRemaining, cut)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// The replay: one synthetic session through both orderings
// ═══════════════════════════════════════════════════════════════════════════════

// Every figure below is measured, from the four colony sessions in
// molequla-run/*/*.stdout (2026-09-15T20:21 to 2026-09-16T21:57), and none is
// chosen:
//
//   - burst seconds: the per-organism median of the 41 bursts of the 2026-09-16
//     20:00Z session, the session MOLEQULALOG2 reports (earth 76.0, air 58.0,
//     water 75.7, fire 71.6);
//   - think seconds: the per-organism median gap between one burst's end= stamp
//     and the next one's start= stamp, less the wait the trainer printed between
//     them, over the same 41 bursts — the tick body, which is what an organism
//     does when it is not on the tape (earth 874.0, air 474.7, water 786.8,
//     fire 509.5);
//   - the warmup: water's measured stage-5 warmup, 1 968.3 + 1 521.8 + 1 306.2 =
//     4 796.3 s for 2 000 steps at 0.4 steps/s;
//   - re-entry: 370 ms, the mean over 162 timed phases of a burst line's
//     start=/end= wall minus the step-loop ms printed on the same line, which is
//     ntNewMirror + register + pullBack + free + malloc_trim.
//
// The two arms differ exactly as the two versions of the code do. Without the
// ceiling the warmup is one phase of 4 796.3 s holding one turn taken at
// trainTurnGrown, which is what the tick loop did around its three sub-phases.
// With it the warmup is the three sub-phases, each cut into chunks at the
// ceiling, and only the first chunk of the first sub-phase carries the grown key.
//
// What is modelled rather than measured: that a waiter is admitted the instant
// the turn frees (a live waiter is up to CFG.TrainingTurnPollSeconds behind), and
// that think time is a constant per organism rather than a distribution. Both
// apply identically to the two arms, so the comparison between them stands; the
// absolute total is a model's and not a session's.
const (
	simSession = 7200.0 // the scheduler's dur=7200
	simReentry = 0.370  // measured mean re-entry cost of a chunk boundary
	// A stage-5 warmup is 2 000 steps, and the two complete ones in the artifacts
	// are four times apart in wall time: earth 610.3 + 298.8 + 291.7 = 1 200.9 s at
	// 1.67 steps/s (earth/earth.stdout, the bursts around it carry start= stamps,
	// so it ran under the serialised lock) and water 1 968.3 + 1 521.8 + 1 306.2 =
	// 4 796.3 s at 0.42 steps/s (water/water.stdout, its neighbouring bursts carry
	// no stamps, so it predates the lock and cannot be dated from these files).
	// The serialised one is what the code produces today and is the arm that
	// asserts; the other is the worst case the artifacts hold, and it is replayed
	// beside it because a ceiling is worth what the longest phase costs.
	simWarmupFast = 1200.9 // earth, under the serialised lock
	simWarmupSlow = 4796.3 // water, before it
)

// simWarmupSerialised and simWarmupPreLock are the sub-phases behind those two
// totals, in the 40/30/30 split the tick loop uses.
var (
	simWarmupSerialised = []float64{610.3, 298.8, 291.7}
	simWarmupPreLock    = []float64{1968.3, 1521.8, 1306.2}
)

const (
	simThinking = iota
	simWanting
	simTraining
)

type simOrg struct {
	sr    *SwarmRegistry
	burst float64 // seconds of tape per micro-burst
	think float64 // seconds of tick body between bursts
	start float64 // when it first asks

	warmupLeft []float64 // warmup phases still owed, in order; empty for no warmup
	grownKey   bool      // the grown key of §2.3, worth one chunk
	inWarmup   bool
	phaseLeft  float64 // seconds of the current phase still owed
	chunked    bool    // this phase has already had a chunk, so the next pays re-entry
	state      int
	until      float64
	asked      float64

	burstWait    float64
	burstWaits   int
	bursts       int
	warmupWait   float64
	warmupChunks int
	warmupRan    float64
	tape         float64
}

// priority is what the organism asks at, which is ntTrainInTurns's rule: the
// grown key on the first chunk of a warmup, trainTurnBurst for everything after.
func (o *simOrg) priority() int {
	if o.inWarmup && o.grownKey {
		return trainTurnGrown
	}
	return trainTurnBurst
}

// nextPhase loads the next unit of work: the next warmup phase if one is owed,
// otherwise a micro-burst.
func (o *simOrg) nextPhase() {
	o.chunked = false
	if len(o.warmupLeft) > 0 {
		o.inWarmup, o.phaseLeft = true, o.warmupLeft[0]
		return
	}
	o.inWarmup, o.phaseLeft = false, o.burst
}

// simRun replays the session against the live AcquireTrainingTurn. It is a
// single-server discrete-event loop: nothing is trained, and the only thing
// under test is which organism the SQL admits next.
func simRun(t *testing.T, ceiling float64, fair bool, subPhases []float64) []*simOrg {
	t.Helper()
	withTrainTurn(t, ceiling, fair)
	withTrainingTTL(t, 1e9) // staleness is another gate's subject
	now := 1000.0
	withSimClock(t, &now)
	r := trainingTurnMesh(t, "earth", "air", "water", "fire")
	base := now

	// Without the ceiling the three sub-phases are one turn, as the tick loop took
	// it; with the ceiling they are queued separately and cut into chunks.
	warmup := append([]float64(nil), subPhases...)
	if ceiling <= 0 {
		total := 0.0
		for _, p := range subPhases {
			total += p
		}
		warmup = []float64{total}
	}
	orgs := []*simOrg{
		{sr: r[0], burst: 76.0, think: 874.0, start: base + 1},
		{sr: r[1], burst: 58.0, think: 474.7, start: base + 2},
		{sr: r[2], burst: 75.7, think: 786.8, start: base, warmupLeft: warmup, grownKey: true},
		{sr: r[3], burst: 71.6, think: 509.5, start: base + 3},
	}
	for _, o := range orgs {
		o.state, o.until = simThinking, o.start
	}
	end := base + simSession
	holder := -1
	guard := 0

	for now < end {
		guard++
		if guard > 200000 {
			t.Fatal("the replay made no progress: nobody could be admitted to a free turn")
		}
		// The next instant at which anything can change.
		next := end
		for _, o := range orgs {
			if o.state != simWanting && o.until < next {
				next = o.until
			}
		}
		if holder < 0 {
			for _, o := range orgs {
				if o.state == simWanting {
					next = now // somebody is waiting for a free turn: decide now
					break
				}
			}
		}
		if next > now {
			now = next
		}

		// A phase that has run its chunk gives the turn back.
		if holder >= 0 && orgs[holder].until <= now {
			o := orgs[holder]
			o.sr.ReleaseTrainingLock()
			holder = -1
			switch {
			case o.phaseLeft > 1e-9:
				// Cut at the ceiling: straight back into the queue for the rest.
				o.state, o.asked = simWanting, now
				o.sr.AcquireTrainingTurn(o.priority())
			case o.inWarmup:
				o.warmupLeft = o.warmupLeft[1:]
				if len(o.warmupLeft) > 0 {
					// The tick loop runs the sub-phases back to back.
					o.nextPhase()
					o.state, o.asked = simWanting, now
					o.sr.AcquireTrainingTurn(o.priority())
				} else {
					o.inWarmup = false
					o.state, o.until = simThinking, now+o.think
				}
			default:
				o.bursts++
				o.state, o.until = simThinking, now+o.think
			}
		}

		// Everyone whose think time is up joins the queue.
		for _, o := range orgs {
			if o.state == simThinking && o.until <= now {
				o.nextPhase()
				o.state, o.asked = simWanting, now
				o.sr.AcquireTrainingTurn(o.priority())
			}
		}

		if holder >= 0 {
			continue
		}
		// The turn is free. Everyone who wants it asks — in reverse order, so an
		// ordering that read the polling sequence would be caught here.
		winner := -1
		for i := len(orgs) - 1; i >= 0; i-- {
			o := orgs[i]
			if o.state != simWanting {
				continue
			}
			if o.sr.AcquireTrainingTurn(o.priority()) {
				if winner >= 0 {
					t.Fatalf("two organisms took the same turn: %s and %s",
						orgs[winner].sr.OrganismID, o.sr.OrganismID)
				}
				winner = i
			}
		}
		if winner < 0 {
			if next >= end {
				break
			}
			continue
		}
		o := orgs[winner]
		wait := now - o.asked
		if o.inWarmup {
			o.warmupChunks++
			o.warmupWait += wait
			o.grownKey = false // the key is spent on the first chunk
		} else {
			o.burstWaits++
			o.burstWait += wait
		}
		run := o.phaseLeft
		if ceiling > 0 && run > ceiling {
			run = ceiling
		}
		cost := 0.0
		if o.chunked {
			cost = simReentry // every chunk after a phase's first rebuilds the tape
		}
		o.chunked = true
		o.phaseLeft -= run
		if o.inWarmup {
			o.warmupRan += run
		}
		o.tape += run + cost
		o.sr.AddTrainingSpent(time.Duration((run + cost) * float64(time.Second)))
		o.state, o.until = simTraining, now+run+cost
		holder = winner
	}
	if holder >= 0 {
		orgs[holder].sr.ReleaseTrainingLock()
	}
	return orgs
}

func simReport(name string, orgs []*simOrg, warmupTotal float64) string {
	var burstWait, warmupWait, warmupRan, tape float64
	var bursts, burstWaits, chunks int
	out := ""
	for _, o := range orgs {
		burstWait += o.burstWait
		warmupWait += o.warmupWait
		warmupRan += o.warmupRan
		tape += o.tape
		bursts += o.bursts
		burstWaits += o.burstWaits
		chunks += o.warmupChunks
		out += fmt.Sprintf("    %-6s bursts=%2d burst_wait=%7.1fs (n=%2d) warmup=%7.1fs in %2d chunks waiting %7.1fs | tape %7.1fs\n",
			o.sr.OrganismID, o.bursts, o.burstWait, o.burstWaits, o.warmupRan, o.warmupChunks, o.warmupWait, o.tape)
	}
	return fmt.Sprintf("%s\n  burst waiting %.1fs over %d waits | bursts %d | warmup %.1fs of %.1fs in %d chunks, waiting %.1fs | tape busy %.1fs of %.0fs\n%s",
		name, burstWait, burstWaits, bursts, warmupRan, warmupTotal, chunks, warmupWait, tape, simSession, out)
}

// The deliverable: the same synthetic session through the ordering of
// origin/main (no ceiling, longest wait) and through this one (a 120 s ceiling,
// least spent first). air has the shortest burst and the shortest think — the
// organism whose bursts are short and frequent, the one longest-wait behind a
// warmup loses — so its wait is the assertion; the totals are logged because the
// number the log has to beat is a total.
func TestTheSameSessionWaitsLessUnderTheNewTurn(t *testing.T) {
	totals := func(orgs []*simOrg) (wait float64, bursts int) {
		for _, o := range orgs {
			wait += o.burstWait
			bursts += o.bursts
		}
		return
	}
	airOf := func(orgs []*simOrg) *simOrg {
		for _, o := range orgs {
			if o.sr.OrganismID == "air" {
				return o
			}
		}
		t.Fatal("no air in the replay")
		return nil
	}

	for _, arm := range []struct {
		name   string
		phases []float64
		total  float64
		assert bool
	}{
		{"the stage-5 warmup measured under the serialised lock (1 200.9 s)", simWarmupSerialised, simWarmupFast, true},
		{"the longest stage-5 warmup in the artifacts (4 796.3 s, pre-lock)", simWarmupPreLock, simWarmupSlow, false},
	} {
		before := simRun(t, 0, false, arm.phases)
		t.Log("\n" + arm.name + "\n" + simReport("  origin/main (one turn per warmup, longest-wait)", before, arm.total))
		after := simRun(t, 120.0, true, arm.phases)
		t.Log("\n" + arm.name + "\n" + simReport("  this branch (120 s ceiling, least-spent-first)", after, arm.total))
		if !arm.assert {
			continue
		}
		if a, b := airOf(after), airOf(before); a.burstWait >= b.burstWait {
			t.Fatalf("air waited %.1fs under the new turn against %.1fs under the old one — the short-burst organism gained nothing",
				a.burstWait, b.burstWait)
		}
		beforeWait, beforeBursts := totals(before)
		afterWait, afterBursts := totals(after)
		if afterWait >= beforeWait {
			t.Fatalf("burst waiting %.1fs against %.1fs: the session did not get cheaper", afterWait, beforeWait)
		}
		if afterBursts <= beforeBursts {
			t.Fatalf("%d bursts completed against %d: the colony did not train more", afterBursts, beforeBursts)
		}
	}
}

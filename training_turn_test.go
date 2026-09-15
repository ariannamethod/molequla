package main

import (
	"testing"
	"time"
)

// Step 1 of docs/resonator_design.md: the training lock as the colony's burst
// admission gate. Until now AcquireGrowthLock serialized growth and the warmup
// behind it, and the micro-bursts of four organisms ran concurrently — four
// tapes, four mirrors, four sets of activations and gradients, four glibc
// arenas. These gates are the queue behind that lock: one holder at a time, the
// order of §2.3, a dead holder bounded by the TTL, a slow one not bounded at
// all.

// registryPair returns two registries on one mesh in a temp swarmDir.
func trainingTurnMesh(t *testing.T, ids ...string) []*SwarmRegistry {
	t.Helper()
	witnessTestMesh(t)
	out := make([]*SwarmRegistry, 0, len(ids))
	for _, id := range ids {
		sr := NewSwarmRegistry(id, id)
		if err := sr.Register(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { sr.ReleaseTrainingLock(); sr.MeshDB.Close() })
		out = append(out, sr)
	}
	return out
}

// withTrainingTTL sets the TTL for one test and puts it back.
func withTrainingTTL(t *testing.T, ttl float64) {
	t.Helper()
	saved := CFG.TrainingLockTTLSeconds
	CFG.TrainingLockTTLSeconds = ttl
	t.Cleanup(func() { CFG.TrainingLockTTLSeconds = saved })
}

func TestOnlyOneOrganismIsInsideATrainingPhase(t *testing.T) {
	r := trainingTurnMesh(t, "earth", "air")
	earth, air := r[0], r[1]

	if !earth.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the first organism to ask was refused the turn")
	}
	if air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("two organisms hold the training turn at once — four tapes is exactly what step 1 removes")
	}
	earth.ReleaseTrainingLock()
	if !air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the turn was not freed by the release")
	}
}

// §2.3, first key: an organism that has just grown is at stage N+1 untrained,
// and leaving it there is the duplicated-invariant bug the tree already paid
// for. It goes to the head of the queue however long the others have waited.
func TestTheGrownOrganismGoesToTheHeadOfTheQueue(t *testing.T) {
	r := trainingTurnMesh(t, "earth", "air", "water")
	earth, air, water := r[0], r[1], r[2]

	if !earth.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("earth was refused the first turn")
	}
	// water asks first and waits; air grows and asks after it.
	if water.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("water took a held turn")
	}
	time.Sleep(5 * time.Millisecond)
	if air.AcquireTrainingTurn(trainTurnGrown) {
		t.Fatal("air took a held turn")
	}

	earth.ReleaseTrainingLock()
	if water.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the longer wait beat the grown organism — the ordering key of §2.3 is inverted")
	}
	if !air.AcquireTrainingTurn(trainTurnGrown) {
		t.Fatal("the grown organism did not get the head of the queue")
	}
}

// §2.3, last key: a plateaued adult still gets turns, it just gets them last.
// Starvation is a bug, not a policy. With no loss-trend column in the mesh
// (see the note on trainTurnBurst) longest-wait is the whole of the ordering
// below the grown organism.
func TestTheLongestWaitGoesFirst(t *testing.T) {
	r := trainingTurnMesh(t, "earth", "air", "water")
	earth, air, water := r[0], r[1], r[2]

	if !earth.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("earth was refused the first turn")
	}
	if air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("air took a held turn")
	}
	time.Sleep(5 * time.Millisecond)
	if water.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("water took a held turn")
	}

	earth.ReleaseTrainingLock()
	if water.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the shorter wait went first")
	}
	if !air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the longest waiter was not admitted")
	}
	// And the one that just trained goes to the back: its queue row is gone,
	// so its next `since` is later than everyone still waiting.
	air.ReleaseTrainingLock()
	var n int
	if err := air.MeshDB.QueryRow("SELECT COUNT(*) FROM training_queue WHERE organism_id='air'").Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("the organism that just trained is still in the queue (%d rows)", n)
	}
}

// A holder killed with kill -9 releases nothing and refreshes nothing. The TTL
// is what bounds it — the same arithmetic AcquireGrowthLock uses, and the only
// reason the lock has a TTL at all.
func TestADeadHolderFreesTheTurnAfterTheTTL(t *testing.T) {
	withTrainingTTL(t, 30.0)
	r := trainingTurnMesh(t, "earth", "air")
	earth, air := r[0], r[1]

	if !earth.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("earth was refused the first turn")
	}
	if air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("air took a held turn")
	}
	// kill -9: the process is gone, so the row stops being re-stamped. Age it
	// past the TTL by hand rather than sleeping thirty seconds.
	stale := float64(time.Now().UnixMilli())/1000.0 - CFG.TrainingLockTTLSeconds - 1
	if _, err := earth.MeshDB.Exec("UPDATE training_lock SET acquired_at=? WHERE organism_id='earth'", stale); err != nil {
		t.Fatal(err)
	}
	if _, err := earth.MeshDB.Exec("UPDATE training_queue SET seen=? WHERE organism_id='earth'", stale); err != nil {
		t.Fatal(err)
	}
	if !air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("a dead holder blocks the colony past its TTL")
	}
}

// A holder that is merely slow must not be preempted: a warmup at stage 4 is
// 1600 steps and minutes long, far past any TTL worth having for a dead
// process. The refresher is why the two cases are different.
func TestTheRefresherKeepsASlowHolder(t *testing.T) {
	withTrainingTTL(t, 30.0)
	r := trainingTurnMesh(t, "earth", "air")
	earth, air := r[0], r[1]

	if !earth.AcquireTrainingTurn(trainTurnGrown) {
		t.Fatal("earth was refused the first turn")
	}
	// The turn has lasted longer than a TTL; the refresher has been running.
	stale := float64(time.Now().UnixMilli())/1000.0 - CFG.TrainingLockTTLSeconds - 1
	if _, err := earth.MeshDB.Exec("UPDATE training_lock SET acquired_at=? WHERE organism_id='earth'", stale); err != nil {
		t.Fatal(err)
	}
	earth.RefreshTrainingLock()
	if air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("a live holder was preempted — the TTL is bounding a slow turn instead of a dead one")
	}
	// The refresher only ever touches its own row: a non-holder's refresh
	// writes nothing, so it can neither steal the turn nor prolong it.
	air.RefreshTrainingLock()
	var holders int
	var holder string
	if err := air.MeshDB.QueryRow("SELECT COUNT(*), COALESCE(MIN(organism_id),'')  FROM training_lock").Scan(&holders, &holder); err != nil {
		t.Fatal(err)
	}
	if holders != 1 || holder != "earth" {
		t.Fatalf("after a non-holder's refresh the lock table holds %d rows, first %q — want one row, earth's", holders, holder)
	}
}

// The refresher really runs, and it stops on release. Three refreshes fall
// inside one TTL by construction (trainingLockRefreshSeconds), so a TTL of
// 150 ms re-stamps every 50 ms.
func TestTheRefresherRunsOnItsOwnClock(t *testing.T) {
	withTrainingTTL(t, 0.15)
	r := trainingTurnMesh(t, "earth", "air")
	earth, air := r[0], r[1]

	if !earth.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("earth was refused the first turn")
	}
	time.Sleep(500 * time.Millisecond) // more than three TTLs
	if air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the holder expired while its refresher was running")
	}
	earth.ReleaseTrainingLock()
	time.Sleep(300 * time.Millisecond)
	if !air.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("the released turn was not free")
	}
}

// Without a mesh an organism is solo and always admitted: the lock must never
// be the thing that stops a single organism from training.
func TestSoloOrganismIsAlwaysAdmitted(t *testing.T) {
	sr := NewSwarmRegistry("solo", "earth")
	if !sr.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("a solo organism was refused the turn")
	}
	if !sr.WaitTrainingTurn(trainTurnBurst, "burst") {
		t.Fatal("a solo organism waited for a turn nobody else can hold")
	}
	sr.ReleaseTrainingLock()
}

// WaitTrainingTurn blocks instead of returning, and returns as soon as the
// holder releases.
func TestWaitTrainingTurnBlocksUntilTheTurnIsFree(t *testing.T) {
	saved := CFG.TrainingTurnPollSeconds
	CFG.TrainingTurnPollSeconds = 0.02
	defer func() { CFG.TrainingTurnPollSeconds = saved }()
	r := trainingTurnMesh(t, "earth", "air")
	earth, air := r[0], r[1]

	if !earth.AcquireTrainingTurn(trainTurnBurst) {
		t.Fatal("earth was refused the first turn")
	}
	done := make(chan time.Duration, 1)
	go func() {
		t0 := time.Now()
		air.WaitTrainingTurn(trainTurnBurst, "burst")
		done <- time.Since(t0)
	}()
	select {
	case <-done:
		t.Fatal("the waiter returned while the turn was held")
	case <-time.After(120 * time.Millisecond):
	}
	earth.ReleaseTrainingLock()
	select {
	case waited := <-done:
		if waited < 100*time.Millisecond {
			t.Fatalf("the waiter returned after %v, before the release", waited)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the waiter never woke after the release")
	}
}

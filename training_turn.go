package main

import (
	"fmt"
	"sync/atomic"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// The turn, made yieldable (docs/resonator_design.md §2.3, after step 1).
//
// Step 1 serialised the colony's training phases on training_lock and that did
// what it was built for: one tape at a time. What it did not bound is how long
// one phase may hold the turn. Measured across the four colony sessions in
// molequla-run/*/*.stdout (2026-09-15T20:21 to 2026-09-16T21:57):
//
//   - an ordinary micro-burst of 32 steps is a median 58.9 s and at most 103.3 s
//     (n = 233 timed bursts);
//   - a stage-5 warmup is three sub-phases taking one turn between them, and the
//     two complete ones on disk are 610.3 + 298.8 + 291.7 = 1 200.8 s at
//     1.67 steps/s (earth, under the serialised lock) and 1 968.3 + 1 521.8 +
//     1 306.2 = 4 796.3 s at 0.42 steps/s (water, before it — the bursts either
//     side carry no start= stamp) for the same 2 000 steps;
//   - behind earth's, in the 2026-09-15 20:00Z session, three siblings each
//     waited about a thousand seconds for a single burst (air 1 005.4 s, fire
//     988.5 s, water 958.1 s), and that session's burst waiting totalled 3 425.8 s
//     over 15 waits against 363.4 s, 503.0 s and 444.7 s in the three sessions
//     after it, none of which contains a warmup at all.
//
// So the thing to bound is a long phase, not the queue. Here a phase runs as a
// sequence of chunks: take the turn, train until the ceiling, mirror the weights
// back, release, re-queue for what is left. Re-entry is not free — ntNewMirror,
// register, pullBack, free and malloc_trim cost a mean of 370 ms and at most
// 1 169 ms on this phone (n = 162, from each burst line's start=/end= wall minus
// the step-loop ms printed beside it) — and CFG.TrainTurnCeilingSeconds is set
// from those two numbers: 120 s is above the longest burst measured, so an
// ordinary burst is never cut, and spends 0.31 % of the tape on rebuilding it.
//
// Two invariants hold the mechanism up. A chunk always runs at least one step,
// so the loop cannot spin; and the steps a chunk ran stay run, because
// ntTrainCore's pullBack mirrors the tape into model.Base on the way out
// whatever stopped it — the same guarantee trainAborting() has always had.
// ═══════════════════════════════════════════════════════════════════════════════

// trainTurnDeadlineMs is when the phase inside the current turn must stop, in
// Unix milliseconds; 0 means no ceiling is armed and ntTrainCore runs to the end
// of its step count. It is process-wide because a process trains one phase at a
// time — the tick loop is the only caller — and it is read every step, which is
// why it is an atomic and not a mutex.
var trainTurnDeadlineMs atomic.Int64

// armTrainTurnCeiling starts the clock on one turn. seconds <= 0 disarms, which
// is what an organism with no colony to yield to gets: a ceiling there would buy
// nothing and pay the re-entry cost for it.
func armTrainTurnCeiling(seconds float64) {
	if seconds <= 0 {
		trainTurnDeadlineMs.Store(0)
		return
	}
	trainTurnDeadlineMs.Store(time.Now().UnixMilli() + int64(seconds*1000.0))
}

// disarmTrainTurnCeiling lifts the ceiling, so a phase that runs outside a turn
// is never cut by a deadline left behind by the phase before it.
func disarmTrainTurnCeiling() { trainTurnDeadlineMs.Store(0) }

// trainTurnCeilingReached reports whether the armed ceiling has passed. The step
// loops call it at a step boundary, never inside one.
func trainTurnCeilingReached() bool {
	d := trainTurnDeadlineMs.Load()
	return d != 0 && time.Now().UnixMilli() >= d
}

// ntTrainInTurns runs one training phase of `steps` steps as a sequence of
// turns, and returns the steps that ran.
//
// `chunk` is the phase itself — ntBurstTrain or ntWarmupTrain — called with the
// steps still owed and returning the steps it took. It is a function and not the
// trainer directly because the warmup's three sub-phases differ only in their
// sequence-length override, and because the gate needs a phase whose length it
// chooses rather than one it has to train for.
//
// `priority` is spent on the first chunk only. §2.3's first key is the organism
// "whose warmup has not run"; after one chunk that sentence is false, and a
// grown organism that kept the key would win back every turn it released, which
// is the ceiling doing nothing at a cost of 370 ms per attempt.
func ntTrainInTurns(swarm *SwarmRegistry, gated bool, priority int, what string, steps int, chunk func(remaining int) int) int {
	if steps <= 0 || chunk == nil {
		return 0
	}
	if !gated || swarm == nil {
		// Nothing to yield to. One phase, no ceiling; the spend is still charged
		// because it is the organism's own record of its session.
		t0 := time.Now()
		ran := chunk(steps)
		swarm.AddTrainingSpent(time.Since(t0))
		return ran
	}
	done := 0
	p := priority
	for done < steps {
		if !swarm.WaitTrainingTurn(p, what) {
			break // shutting down; the steps already taken are in model.Base
		}
		armTrainTurnCeiling(CFG.TrainTurnCeilingSeconds)
		t0 := time.Now()
		ran := chunk(steps - done)
		held := time.Since(t0)
		disarmTrainTurnCeiling()
		swarm.AddTrainingSpent(held)
		swarm.ReleaseTrainingLock()
		if ran <= 0 {
			break // no corpus line long enough to train on; asking again would spin
		}
		done += ran
		p = trainTurnBurst
		if trainAborting() {
			break
		}
		if done < steps {
			fmt.Printf("[trainer] %s yielded the turn after %.1fs, %d of %d steps left (spent %.1fs this session)\n",
				what, held.Seconds(), steps-done, steps, swarm.TrainingSpentSeconds())
		}
	}
	return done
}

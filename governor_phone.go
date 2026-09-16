package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Governor pieces for a small machine (repair 4, MOLEQULALOG2.md 2026-09-13).
//
// The cascade governor counts heads (MaxOrganisms) and counts them by heartbeat
// freshness. Three things broke that on a phone: there was no byte budget at
// all; the post-growth warmup runs inline for minutes without a tick, so a
// growing organism dropped out of the live count exactly when its memory
// peaked; and a hibernating organism kept its process, and its weights, in
// RAM because main parked on a signal it never received. Here: a byte gate
// before division built from the organism's own peak RSS, a heartbeat keeper
// that repeats the last known state on its own clock, and the wait that ends
// the process when the trainer loop ends.
// ═══════════════════════════════════════════════════════════════════════════════

// parseProcKB extracts a "<key>: <n> kB" field from /proc text (meminfo or a
// process status file). Returns 0 when the field is absent or unparsable.
func parseProcKB(text, key string) int64 {
	sc := bufio.NewScanner(strings.NewReader(text))
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, key+":") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		v, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil || v < 0 {
			return 0
		}
		return v
	}
	return 0
}

// procFieldMB reads one kB field from a /proc file in MB; 0 means unknown.
func procFieldMB(path, key string) int64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	return parseProcKB(string(data), key) / 1024
}

// memAvailableMB is MemAvailable from /proc/meminfo; 0 means unknown.
func memAvailableMB() int64 { return procFieldMB("/proc/meminfo", "MemAvailable") }

// ownPeakRSSMB is this process's high-water RSS (VmHWM); 0 means unknown. A
// child loads the parent's checkpoint and walks the same stages, so the
// parent's own peak is the measured footprint of the organism it is about to
// spawn.
func ownPeakRSSMB() int64 { return procFieldMB("/proc/self/status", "VmHWM") }

// memGateDecision is the byte gate as a pure function: a child needs the
// parent's peak RSS plus floorMB of headroom for the rest of the machine.
// floorMB <= 0 disables the gate; unknown free memory (<= 0) opens it, since a
// kernel that hides MemAvailable is not a reason to stop reproducing.
func memGateDecision(freeMB, peakMB, floorMB int64) (open bool, needMB int64) {
	if floorMB <= 0 {
		return true, 0
	}
	needMB = floorMB + peakMB
	if freeMB <= 0 {
		return true, needMB
	}
	return freeMB >= needMB, needMB
}

// mitosisMemGateOpen applies memGateDecision to the live machine. It is a
// pre-check beside AcquireMitosisSlot, not a replacement for the head count.
func mitosisMemGateOpen(floorMB int64) (open bool, freeMB, needMB int64) {
	freeMB = memAvailableMB()
	open, needMB = memGateDecision(freeMB, ownPeakRSSMB(), floorMB)
	return
}

// growthGateDecision is the byte gate before ontogenesis, the same arithmetic as
// memGateDecision over a different cost: a stage transition does not add a
// process, it multiplies this one. The cost is expressed as factorPct percent of
// the organism's own peak RSS, measured from what a stage step actually does to
// VmHWM (MOLEQULALOG2.md 2026-09-13, repair 9). factorPct <= 0 charges nothing
// but the floor; floorMB <= 0 disables the gate entirely.
func growthGateDecision(freeMB, peakMB, floorMB int64, factorPct int) (open bool, needMB int64) {
	if factorPct < 0 {
		factorPct = 0
	}
	return memGateDecision(freeMB, peakMB*int64(factorPct)/100, floorMB)
}

// growthMemGateOpen applies growthGateDecision to the live machine.
func growthMemGateOpen(floorMB int64, factorPct int) (open bool, freeMB, needMB int64) {
	freeMB = memAvailableMB()
	open, needMB = growthGateDecision(freeMB, ownPeakRSSMB(), floorMB, factorPct)
	return
}

// growthMemGateCheck is the gate as the trainer calls it: one line per deferral,
// and the caller tries again on a later tick because nothing about the decision
// is remembered.
func growthMemGateCheck() bool {
	open, freeMB, needMB := growthMemGateOpen(int64(CFG.GrowthMinFreeMB), CFG.GrowthPeakFactorPct)
	if !open {
		fmt.Printf("[growth] deferred: free=%d MB need=%d MB\n", freeMB, needMB)
	}
	return open
}

// oomScoreAdjPath is the live knob; tests pass a temp file instead.
const oomScoreAdjPath = "/proc/self/oom_score_adj"

// applyOomScoreAdj raises this process's OOM badness so the phone's lmkd reaches
// for the organism before it reaches for the terminal that owns the session.
// Processes started under Magisk su inherit oom_score_adj=-1000, which made the
// colony unkillable and cost Termux and ~20 apps on 2026-09-13. v == 0 leaves the
// value untouched; non-linux is a no-op. Returns the value read back.
func applyOomScoreAdj(path string, v int) (string, error) {
	if v == 0 || runtime.GOOS != "linux" {
		return "", nil
	}
	if err := os.WriteFile(path, []byte(strconv.Itoa(v)+"\n"), 0644); err != nil {
		return "", err
	}
	got, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(got)), nil
}

// beatKeeper repeats the organism's last reported heartbeat on its own clock,
// so the mesh sees the organism as alive while the tick loop is blocked inside
// an inline warmup after growth. SwarmRegistry.Heartbeat feeds it the fresh
// values; the keeper re-sends them, except the peak, which it reads for itself
// at every beat. Stop() silences it for good, so a hibernating organism is
// never written back as alive.
//
// The peak is the one field the keeper must not replay. Measured 2026-09-15 at
// 20:31Z: the witness printed `earth … p490` while /proc/24540/status held
// `VmHWM: 1732008 kB` — 1691 MB, understated 3.45x, and unmoved for 401
// witness lines (MOLEQULALOG2.md, "the third session"). The tick loop reads
// ownPeakRSSMB() at the call site but beats only on tickCount%10, and a stage-5
// tick is one ~85 s burst, so ten of them are a long way apart; between them
// this keeper replayed a cached figure every 20 s. §4.2 of
// docs/resonator_design.md keys the sleep policy on the largest C_i, so the
// organism doing the most work was the one whose column rotted most.
//
// The other fields stay replayed on purpose. Stage, parameter count, syntropy,
// entropy, global step, generation magnitude and overlay fade are training
// state the tick loop owns: they are read under model.mu and are only true as
// of the tick that computed them, so re-deriving them here would mean taking
// that lock — the lock a multi-minute warmup is holding, which is the whole
// reason this keeper exists. The peak is not training state. It belongs to the
// process, not to the model; it costs one /proc/self/status read; and it is
// monotone, so a fresh reading can never contradict a cached one, only
// supersede it.
type beatKeeper struct {
	mu      sync.Mutex
	swarm   *SwarmRegistry
	stage   int
	nParams int
	syn     float64
	ent     float64
	step    int
	mag     float64
	fade    float64
	peak    int64 // VmHWM in MB (resonator design, step 0)
	set     bool
	stopped bool
	// peakFn reads the live high-water RSS. It is ownPeakRSSMB on a running
	// organism and a stub in the gate, because a test that grows the real
	// process to drive this would be measuring the Go runtime's allocator
	// rather than the keeper.
	peakFn func() int64
}

func newBeatKeeper(swarm *SwarmRegistry) *beatKeeper {
	return &beatKeeper{swarm: swarm, peakFn: ownPeakRSSMB}
}

// Set records the latest state the tick loop reported.
func (b *beatKeeper) Set(stage, nParams int, syn, ent float64, step int, mag, fade float64, peak int64) {
	if b == nil {
		return
	}
	b.mu.Lock()
	b.stage, b.nParams, b.syn, b.ent, b.step, b.set = stage, nParams, syn, ent, step, true
	b.mag, b.fade, b.peak = mag, fade, peak
	b.mu.Unlock()
}

// Stop silences the keeper permanently.
func (b *beatKeeper) Stop() {
	if b == nil {
		return
	}
	b.mu.Lock()
	b.stopped = true
	b.mu.Unlock()
}

// beat sends one heartbeat with the last recorded training state and a peak
// read now. Returns false when nothing has been recorded yet, the keeper is
// stopped, or there is no mesh.
func (b *beatKeeper) beat() bool {
	if b == nil || b.swarm == nil {
		return false
	}
	b.mu.Lock()
	stage, nParams, syn, ent, step, set, stopped := b.stage, b.nParams, b.syn, b.ent, b.step, b.set, b.stopped
	mag, fade, peak := b.mag, b.fade, b.peak
	read := b.peakFn
	b.mu.Unlock()
	if !set || stopped {
		return false
	}
	// VmHWM only ever rises, so the larger of the two is the true one: a read
	// that fails returns 0 and leaves the cached figure standing, and a read
	// that succeeds is never older than the cache.
	if read != nil {
		if fresh := read(); fresh > peak {
			peak = fresh
		}
	}
	b.swarm.Heartbeat(stage, nParams, syn, ent, step, mag, fade, peak)
	return true
}

// Run repeats the heartbeat every interval until stop is closed.
func (b *beatKeeper) Run(stop <-chan struct{}, interval time.Duration) {
	if b == nil || interval <= 0 {
		return
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			b.beat()
		}
	}
}

// trainAbort is raised once, on the way out, and never lowered: the step loops
// in notorch_trainer.go read it every step and return early. Closing `stop` is
// not enough, because a warmup holds model.mu for its whole phase — 1600 steps
// at stage 4 — so a shutdown that only closes `stop` waits half an hour for the
// mutex and the checkpoint is written after the session has already been killed
// (2026-09-13: every ckpt on disk kept its growth-time mtime through stop.sh).
var trainAbort atomic.Bool

// trainAborting reports whether the process is shutting down. Training steps
// stop at the next step boundary; the weights trained so far are mirrored back
// by ntTrainCore's pullBack, so an aborted phase is progress kept, not lost.
func trainAborting() bool { return trainAbort.Load() }

// waitEvolution parks main in evolution mode until the shutdown channel closes
// or the trainer loop ends on its own (hibernation). `shutdown` is closed by the
// signal handler main arms before the bootstrap climb (repair 10), so a signal
// that landed minutes earlier — during a first launch's warmup — is still
// observable here. On shutdown it raises the train abort and closes stop so the
// trainer winds down at the next step and the next tick; when the trainer is
// already gone there is nothing to stop and the process simply ends, releasing
// the organism's memory to the colony. Returns the reason for the caller's log
// line.
func waitEvolution(shutdown <-chan struct{}, done <-chan struct{}, stop chan struct{}) string {
	select {
	case <-shutdown:
		trainAbort.Store(true)
		close(stop)
		return "signal"
	case <-done:
		return "trainer-exit"
	}
}

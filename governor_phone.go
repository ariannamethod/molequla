package main

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"sync"
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

// beatKeeper repeats the organism's last reported heartbeat on its own clock,
// so the mesh sees the organism as alive while the tick loop is blocked inside
// an inline warmup after growth. SwarmRegistry.Heartbeat feeds it the fresh
// values; the keeper only re-sends them. Stop() silences it for good, so a
// hibernating organism is never written back as alive.
type beatKeeper struct {
	mu      sync.Mutex
	swarm   *SwarmRegistry
	stage   int
	nParams int
	syn     float64
	ent     float64
	step    int
	set     bool
	stopped bool
}

func newBeatKeeper(swarm *SwarmRegistry) *beatKeeper {
	return &beatKeeper{swarm: swarm}
}

// Set records the latest state the tick loop reported.
func (b *beatKeeper) Set(stage, nParams int, syn, ent float64, step int) {
	if b == nil {
		return
	}
	b.mu.Lock()
	b.stage, b.nParams, b.syn, b.ent, b.step, b.set = stage, nParams, syn, ent, step, true
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

// beat sends one heartbeat with the last recorded state. Returns false when
// nothing has been recorded yet, the keeper is stopped, or there is no mesh.
func (b *beatKeeper) beat() bool {
	if b == nil || b.swarm == nil {
		return false
	}
	b.mu.Lock()
	stage, nParams, syn, ent, step, set, stopped := b.stage, b.nParams, b.syn, b.ent, b.step, b.set, b.stopped
	b.mu.Unlock()
	if !set || stopped {
		return false
	}
	b.swarm.Heartbeat(stage, nParams, syn, ent, step)
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

// waitEvolution parks main in evolution mode until a signal arrives or the
// trainer loop ends on its own (hibernation). On a signal it closes stop so
// the trainer winds down; when the trainer is already gone there is nothing
// to stop and the process simply ends, releasing the organism's memory to the
// colony. Returns the reason for the caller's log line.
func waitEvolution(sigCh <-chan os.Signal, done <-chan struct{}, stop chan struct{}) string {
	select {
	case <-sigCh:
		close(stop)
		return "signal"
	case <-done:
		return "trainer-exit"
	}
}

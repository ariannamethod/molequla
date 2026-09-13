package main

import (
	"database/sql"
	"os"
	"testing"
	"time"
)

func TestParseProcKB(t *testing.T) {
	meminfo := "MemTotal:        7603028 kB\nMemFree:          330000 kB\nMemAvailable:    2257168 kB\nBuffers:           12 kB\n"
	if got := parseProcKB(meminfo, "MemAvailable"); got != 2257168 {
		t.Fatalf("MemAvailable = %d, want 2257168", got)
	}
	status := "VmPeak:\t 1269308 kB\nVmHWM:\t   97776 kB\nVmRSS:\t   65000 kB\n"
	if got := parseProcKB(status, "VmHWM"); got != 97776 {
		t.Fatalf("VmHWM = %d, want 97776", got)
	}
	if got := parseProcKB(status, "VmRSS"); got != 65000 {
		t.Fatalf("VmRSS = %d, want 65000 (prefix VmRSS must not match VmRSSx)", got)
	}
	if got := parseProcKB("MemTotal: 1 kB\n", "MemAvailable"); got != 0 {
		t.Fatalf("missing field must read 0, got %d", got)
	}
	if got := parseProcKB("MemAvailable: junk kB\n", "MemAvailable"); got != 0 {
		t.Fatalf("unparsable field must read 0, got %d", got)
	}
}

// The byte gate as arithmetic: a child costs the parent's peak plus the floor.
func TestMemGateDecision(t *testing.T) {
	if open, need := memGateDecision(100, 900, 0); !open || need != 0 {
		t.Fatalf("floor 0 must disable the gate: open=%v need=%d", open, need)
	}
	if open, _ := memGateDecision(0, 900, 256); !open {
		t.Fatal("unknown free memory must leave the gate open")
	}
	if open, need := memGateDecision(1000, 300, 256); !open || need != 556 {
		t.Fatalf("1000 MB free against 300+256 must be open: open=%v need=%d", open, need)
	}
	if open, need := memGateDecision(500, 300, 256); open || need != 556 {
		t.Fatalf("500 MB free against 300+256 must be closed: open=%v need=%d", open, need)
	}
	if open, _ := memGateDecision(556, 300, 256); !open {
		t.Fatal("free == need must be open (>=)")
	}
}

func TestMemGateOnThisHost(t *testing.T) {
	if _, err := os.Stat("/proc/meminfo"); err != nil {
		t.Skip("no /proc/meminfo on this host")
	}
	free := memAvailableMB()
	if free <= 0 {
		t.Fatalf("memAvailableMB = %d on a Linux host, want > 0", free)
	}
	peak := ownPeakRSSMB()
	if peak <= 0 {
		t.Fatalf("ownPeakRSSMB = %d, want > 0 for a running process", peak)
	}
	if open, _, need := mitosisMemGateOpen(1); !open || need != 1+peak {
		t.Fatalf("gate with a 1 MB floor: open=%v need=%d free=%d peak=%d", open, need, free, peak)
	}
	if open, _, _ := mitosisMemGateOpen(free * 4); open {
		t.Fatalf("gate must be closed with %d MB free against a %d MB floor", free, free*4)
	}
	t.Logf("host: free=%d MB, own peak=%d MB", free, peak)
}

func meshForKeeperTest(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	// The columns Heartbeat writes, global_step included (repair 7): a fixture
	// narrower than the real schema made the UPDATE fail silently and this
	// test red the day the column was added.
	if _, err := db.Exec(`CREATE TABLE organisms(id TEXT PRIMARY KEY, stage INTEGER, n_params INTEGER, syntropy REAL, entropy REAL, last_heartbeat REAL, status TEXT, global_step INTEGER)`); err != nil {
		db.Close()
		t.Skipf("sqlite exec: %v", err)
	}
	return db
}

func heartbeatRow(t *testing.T, db *sql.DB, id string) (age float64, stage int, status string) {
	t.Helper()
	var hb float64
	if err := db.QueryRow(`SELECT last_heartbeat, stage, status FROM organisms WHERE id=?`, id).Scan(&hb, &stage, &status); err != nil {
		t.Fatal(err)
	}
	return float64(time.Now().UnixMilli())/1000.0 - hb, stage, status
}

// The keeper must keep the organism's heartbeat fresh in the mesh without the
// tick loop, carry the last state Heartbeat reported, and fall silent after
// hibernation so a sleeping organism is never written back as alive.
func TestBeatKeeperRefreshesMeshWithoutTicks(t *testing.T) {
	db := meshForKeeperTest(t)
	defer db.Close()
	stale := float64(time.Now().UnixMilli())/1000.0 - 600 // ten minutes ago
	db.Exec(`INSERT INTO organisms(id,stage,n_params,syntropy,entropy,last_heartbeat,status) VALUES(?,?,?,?,?,?,?)`, "a", 2, 1000, 0.1, 0.2, stale, "alive")
	sr := &SwarmRegistry{OrganismID: "a", MeshDB: db}

	stop := make(chan struct{})
	defer close(stop)
	sr.StartKeeper(stop, 20*time.Millisecond)
	time.Sleep(60 * time.Millisecond)
	if age, _, _ := heartbeatRow(t, db, "a"); age < 500 {
		t.Fatalf("keeper beat before any state was reported (age %.0f s)", age)
	}

	sr.Heartbeat(3, 1100000, 0.3, 0.4, 4200) // the tick loop reports once, then blocks
	time.Sleep(120 * time.Millisecond)
	age, stage, status := heartbeatRow(t, db, "a")
	if age > 5 {
		t.Fatalf("heartbeat is %.1f s old after the keeper ran, want fresh", age)
	}
	if stage != 3 || status != "alive" {
		t.Fatalf("keeper sent stage=%d status=%s, want the last Heartbeat values 3/alive", stage, status)
	}

	sr.MarkHibernating()
	time.Sleep(80 * time.Millisecond)
	if _, _, status := heartbeatRow(t, db, "a"); status != "sleeping" {
		t.Fatalf("a hibernating organism was written back as %q by the keeper", status)
	}
}

// Hibernation ends the trainer loop; main must end with it instead of parking
// on a signal that never comes.
func TestWaitEvolutionEndsWhenTrainerExits(t *testing.T) {
	// trainAbort is process-global and, in a live organism, never lowered again;
	// a test that raises it must put it back or every trainer test after this one
	// returns without a step.
	trainAbort.Store(false)
	defer trainAbort.Store(false)

	sigCh := make(chan os.Signal, 1)
	done := make(chan struct{})
	stop := make(chan struct{})
	res := make(chan string, 1)
	go func() { res <- waitEvolution(sigCh, done, stop) }()
	close(done)
	select {
	case r := <-res:
		if r != "trainer-exit" {
			t.Fatalf("got %q, want trainer-exit", r)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waitEvolution did not return after the trainer exited")
	}
	select {
	case <-stop:
		t.Fatal("stop must stay open when the trainer already exited")
	default:
	}
	if trainAborting() {
		t.Fatal("a trainer that exited on its own must not raise the shutdown abort")
	}

	sigCh2 := make(chan os.Signal, 1)
	stop2 := make(chan struct{})
	sigCh2 <- os.Interrupt
	if r := waitEvolution(sigCh2, make(chan struct{}), stop2); r != "signal" {
		t.Fatalf("got %q, want signal", r)
	}
	select {
	case <-stop2:
	default:
		t.Fatal("a signal must close stop so the trainer winds down")
	}
	// Closing stop is not enough on its own: a warmup holds model.mu for its
	// whole phase, so the step loop has to be told to stop too.
	if !trainAborting() {
		t.Fatal("a signal must raise the train abort, or the exit path waits on model.mu")
	}
}

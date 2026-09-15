package main

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// mesh.db is written by four organisms' heartbeats, by the training queue and
// its refresher (resonator design §2.3, step 1) and by the world ledger's
// ingest, each from its own process. Until CFG.MeshBusyTimeoutMS the tree
// carried exactly one pragma — journal_mode=WAL, molequla.go — and no
// busy_timeout anywhere, so the first collision returned SQLITE_BUSY at once
// and the losing write was dropped. The 20:00Z session of 2026-09-15 lost four
// heartbeats that way in its first six minutes. A dropped heartbeat is not a
// dropped statistic: the keeper beats every 20 s against a 60 s liveness
// window (molequla.go, StartKeeper), so three in a row make a live organism
// dead to DiscoverPeers and to the mitosis cap.

// meshBusyHoldLock opens a second connection on the mesh and holds a write
// transaction on it for d, which is what a contending writer must survive.
// It returns once the lock is actually held.
func meshBusyHoldLock(t *testing.T, path string, d time.Duration) *sync.WaitGroup {
	t.Helper()
	held := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		db, err := sql.Open("sqlite", path)
		if err != nil {
			t.Error(err)
			close(held)
			return
		}
		defer db.Close()
		db.SetMaxOpenConns(1)
		tx, err := db.Begin()
		if err != nil {
			t.Error(err)
			close(held)
			return
		}
		// BEGIN alone is deferred; the write lock is taken by the first write.
		if _, err := tx.Exec("INSERT OR REPLACE INTO messages(from_id,to_id,type,payload,ts) VALUES('holder','','hold','',0)"); err != nil {
			t.Error(err)
			tx.Rollback()
			close(held)
			return
		}
		close(held)
		time.Sleep(d)
		tx.Commit()
	}()
	<-held
	return &wg
}

// TestMeshWriteWaitsForAHeldWriteLock is the gate on the defect. With the
// pragma the heartbeat waits for the holder and lands; without it — the state
// of origin/main — it is refused and the organism goes invisible.
func TestMeshWriteWaitsForAHeldWriteLock(t *testing.T) {
	for _, tc := range []struct {
		name    string
		busyMS  int
		wantErr bool
	}{
		{"with busy_timeout", 5000, false},
		{"at the old zero wait", 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			savedCFG := CFG.MeshBusyTimeoutMS
			CFG.MeshBusyTimeoutMS = tc.busyMS
			defer func() { CFG.MeshBusyTimeoutMS = savedCFG }()

			dir := witnessTestMesh(t)
			sr := NewSwarmRegistry("water", "water")
			if err := sr.Register(); err != nil {
				t.Fatal(err)
			}
			defer sr.MeshDB.Close()
			// One connection in the pool, so the handle under test is the one
			// meshDSN built and not a second the pool opened for the holder.
			sr.MeshDB.SetMaxOpenConns(1)

			path := filepath.Join(dir, "mesh.db")
			// 300 ms: longer than any single mesh write measured on this phone
			// (the heaviest, the ledger's 50-fact pass, is 14.3-26.0 ms) and
			// short enough that the test costs a third of a second.
			wg := meshBusyHoldLock(t, path, 300*time.Millisecond)

			out := captureStdout(t, func() {
				sr.Heartbeat(5, 40000000, 0.5, 0.5, 490, 1.0, 0.0, 1451)
			})
			wg.Wait()

			refused := strings.Contains(out, "mesh refused")
			if tc.wantErr && !refused {
				t.Fatalf("a zero-wait write through a held lock was NOT refused; the defect is not being reproduced: %q", out)
			}
			if !tc.wantErr {
				if refused {
					t.Fatalf("busy_timeout=%d ms did not carry the heartbeat through a 300 ms held lock: %q", tc.busyMS, out)
				}
				var beat float64
				if err := sr.MeshDB.QueryRow("SELECT last_heartbeat FROM organisms WHERE id='water'").Scan(&beat); err != nil {
					t.Fatal(err)
				}
				if beat <= 0 {
					t.Fatalf("the heartbeat said nothing and wrote nothing: last_heartbeat=%v", beat)
				}
			}
		})
	}
}

// TestMeshDSNReachesEveryConnectionInThePool is why the pragma rides the DSN
// and is not executed after the open: busy_timeout belongs to a connection,
// and database/sql opens as many as it likes.
func TestMeshDSNReachesEveryConnectionInThePool(t *testing.T) {
	saved := CFG.MeshBusyTimeoutMS
	CFG.MeshBusyTimeoutMS = 5000
	defer func() { CFG.MeshBusyTimeoutMS = saved }()

	witnessTestMesh(t)
	sr := NewSwarmRegistry("air", "air")
	if err := sr.Register(); err != nil {
		t.Fatal(err)
	}
	defer sr.MeshDB.Close()
	sr.MeshDB.SetMaxOpenConns(4)

	// Force four distinct connections to exist at once, then read the pragma
	// back on each. A pragma applied by db.Exec would be on one of them.
	var wg sync.WaitGroup
	got := make([]int, 4)
	errs := make([]error, 4)
	start := make(chan struct{})
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			conn, err := sr.MeshDB.Conn(context.Background())
			if err != nil {
				errs[i] = err
				return
			}
			defer conn.Close()
			<-start
			errs[i] = conn.QueryRowContext(context.Background(), "PRAGMA busy_timeout").Scan(&got[i])
		}(i)
	}
	time.Sleep(50 * time.Millisecond)
	close(start)
	wg.Wait()
	for i := range got {
		if errs[i] != nil {
			t.Fatal(errs[i])
		}
		if got[i] != 5000 {
			t.Fatalf("connection %d carries busy_timeout=%d, want 5000: the pragma did not reach the whole pool", i, got[i])
		}
	}
}

// TestMeshBusyTimeoutDoesNotMaskASchemaError: the wait is for a lock, never for
// a schema. A heartbeat against a table too narrow for it must still be said
// at once — TestHeartbeatSaysWhenTheMeshRefusesTheWrite is the gate on that
// line, and this is the gate that the busy_timeout handle did not silence it.
func TestMeshBusyTimeoutDoesNotMaskASchemaError(t *testing.T) {
	saved := CFG.MeshBusyTimeoutMS
	CFG.MeshBusyTimeoutMS = 5000
	defer func() { CFG.MeshBusyTimeoutMS = saved }()

	dir := witnessTestMesh(t)
	path := filepath.Join(dir, "mesh.db")
	db, err := sql.Open("sqlite", meshDSN(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var busy int
	if err := db.QueryRow("PRAGMA busy_timeout").Scan(&busy); err != nil {
		t.Fatal(err)
	}
	if busy != 5000 {
		t.Fatalf("the handle under test carries busy_timeout=%d, not the 5000 this gate is about", busy)
	}
	// The pre-repair-7 schema: no global_step, no gen_mag, no overlay_fade,
	// no peak_rss_mb.
	if _, err := db.Exec(`CREATE TABLE organisms(
		id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER,
		n_params INTEGER, syntropy REAL, entropy REAL,
		last_heartbeat REAL, parent_id TEXT,
		status TEXT DEFAULT 'alive', element TEXT)`); err != nil {
		t.Fatal(err)
	}
	sr := &SwarmRegistry{OrganismID: "fire", Element: "fire", MeshDB: db}

	t0 := time.Now()
	out := captureStdout(t, func() { sr.Heartbeat(2, 262144, 0.1, 1.2, 4200, 3.25, 0.40, 231) })
	elapsed := time.Since(t0)

	if !strings.Contains(out, "[ecology]") || !strings.Contains(out, "fire") || !strings.Contains(out, "global_step") {
		t.Fatalf("busy_timeout swallowed a schema error the mesh must report: %q", out)
	}
	// And it was reported now, not after the timeout: a missing column is not
	// a lock and must not be waited on.
	if elapsed > time.Second {
		t.Fatalf("a schema error took %v to be said; busy_timeout is being applied to the wrong failure", elapsed)
	}
}

// TestMeshBusyTimeoutZeroIsTheOldBehaviour: the knob turns off to exactly what
// the tree did before, so the default can be argued rather than assumed.
func TestMeshBusyTimeoutZeroIsTheOldBehaviour(t *testing.T) {
	saved := CFG.MeshBusyTimeoutMS
	defer func() { CFG.MeshBusyTimeoutMS = saved }()
	CFG.MeshBusyTimeoutMS = 0
	if got := meshDSN("/x/mesh.db"); got != "/x/mesh.db" {
		t.Fatalf("a zero timeout still decorated the DSN: %q", got)
	}
	CFG.MeshBusyTimeoutMS = 5000
	want := fmt.Sprintf("/x/mesh.db?_pragma=busy_timeout(%d)", 5000)
	if got := meshDSN("/x/mesh.db"); got != want {
		t.Fatalf("meshDSN = %q, want %q", got, want)
	}
}

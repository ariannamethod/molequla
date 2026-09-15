package main

// Memory attribution hook. Off unless MOLEQULA_PPROF names a directory, and
// when off it costs one already-resolved string comparison per call site.
//
// It answers two questions that /proc alone cannot. First, how the resident set
// splits between the Go heap and the C heap: runtime.MemStats accounts for
// everything the Go allocator owns, and what VmRSS holds beyond that is the
// notorch tape, OpenBLAS and sqlite. Second, whether the Go side is holding
// garbage at that moment: the second line is taken after a forced collection
// and a release to the OS, so the delta is exactly what a FreeOSMemory would
// have won at that instant.

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

var (
	memProfOnce sync.Once
	memProfDir  string
	memProfSeq  int64
)

func memProfTarget() string {
	memProfOnce.Do(func() {
		d := strings.TrimSpace(os.Getenv("MOLEQULA_PPROF"))
		if d == "" {
			return
		}
		if err := os.MkdirAll(d, 0o755); err != nil {
			fmt.Printf("[memprof] disabled: %v\n", err)
			return
		}
		memProfDir = d
	})
	return memProfDir
}

// smapsRollupMB returns one field of /proc/self/smaps_rollup in MB, -1 if the
// kernel does not provide the file or the key.
func smapsRollupMB(key string) int64 {
	b, err := os.ReadFile("/proc/self/smaps_rollup")
	if err != nil {
		return -1
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(ln, key+":") {
			continue
		}
		f := strings.Fields(ln)
		if len(f) < 2 {
			return -1
		}
		kb, err := strconv.ParseInt(f[1], 10, 64)
		if err != nil {
			return -1
		}
		return kb / 1024
	}
	return -1
}

// memSnapshot reports the split at `label` and writes a heap profile beside it.
// Call it at a point whose cost is being attributed; it is a no-op when the
// environment variable is unset.
func memSnapshot(label string) {
	dir := memProfTarget()
	if dir == "" {
		return
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	rss := procFieldMB("/proc/self/status", "VmRSS")
	hwm := procFieldMB("/proc/self/status", "VmHWM")
	anon := smapsRollupMB("Anonymous")
	swap := smapsRollupMB("Swap")
	mb := func(v uint64) int64 { return int64(v / (1 << 20)) }
	// What the Go runtime keeps resident is everything it took from the kernel
	// minus what it has handed back; the rest of VmRSS is the C side.
	goRes := mb(ms.Sys - ms.HeapReleased)
	arena, mmapped, inUse, cfree := cHeapStats()
	m := func(v int64) int64 { return v / (1 << 20) }
	fmt.Printf("[memprof] %s rss=%dMB hwm=%dMB anon=%dMB swap=%dMB | go: alloc=%dMB heapSys=%dMB heapIdle=%dMB heapRel=%dMB sys=%dMB res=%dMB | c: arena=%dMB mmap=%dMB inuse=%dMB frag=%dMB | rss-goRes=%dMB\n",
		label, rss, hwm, anon, swap,
		mb(ms.HeapAlloc), mb(ms.HeapSys), mb(ms.HeapIdle), mb(ms.HeapReleased), mb(ms.Sys), goRes,
		m(arena), m(mmapped), m(inUse), m(cfree), rss-goRes)

	seq := atomic.AddInt64(&memProfSeq, 1)
	name := filepath.Join(dir, fmt.Sprintf("heap-%03d-%s.pprof", seq, strings.NewReplacer("/", "_", " ", "_").Replace(label)))
	f, err := os.Create(name)
	if err != nil {
		fmt.Printf("[memprof] %s: %v\n", label, err)
		return
	}
	runtime.GC()
	if err := pprof.WriteHeapProfile(f); err != nil {
		fmt.Printf("[memprof] %s: %v\n", label, err)
	}
	f.Close()

	// Second line: what the Go side could have returned to the OS right here.
	// Off unless asked for, because the collection it forces changes the very
	// timeline the first line is measuring.
	if os.Getenv("MOLEQULA_PPROF_FREE") == "" {
		return
	}
	debug.FreeOSMemory()
	runtime.ReadMemStats(&ms)
	arena, mmapped, inUse, cfree = cHeapStats()
	fmt.Printf("[memprof] %s after-free rss=%dMB alloc=%dMB heapSys=%dMB heapRel=%dMB sys=%dMB | c: arena=%dMB mmap=%dMB inuse=%dMB frag=%dMB\n",
		label, procFieldMB("/proc/self/status", "VmRSS"),
		mb(ms.HeapAlloc), mb(ms.HeapSys), mb(ms.HeapReleased), mb(ms.Sys),
		m(arena), m(mmapped), m(inUse), m(cfree))
}

// memTapeSnapshot prints the live notorch tape census. Call it between the
// backward and the clear, where the graph, its gradients and the moment slots
// are all alive at once — that instant is the step's true C-side cost.
func memTapeSnapshot(label string, T, D, nLayer, vocab int) {
	if memProfTarget() == "" {
		return
	}
	entries, params, out, grad, pbytes, slots := ntTapeCensus()
	mb := func(v int64) float64 { return float64(v) / (1 << 20) }
	arena, mmapped, inUse, cfree := cHeapStats()
	fmt.Printf("[memprof] tape %s entries=%d params=%d | act=%.1fMB grad=%.1fMB mirror=%.1fMB moments=%.1fMB total=%.1fMB | T=%d D=%d L=%d V=%d | c: arena=%.0fMB mmap=%.0fMB inuse=%.0fMB frag=%.0fMB | rss=%dMB\n",
		label, entries, params, mb(out), mb(grad), mb(pbytes), mb(slots),
		mb(out+grad+pbytes+slots), T, D, nLayer, vocab,
		mb(arena), mb(mmapped), mb(inUse), mb(cfree),
		procFieldMB("/proc/self/status", "VmRSS"))
}

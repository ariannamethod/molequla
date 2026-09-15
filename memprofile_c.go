package main

/*
#cgo CFLAGS: -I/usr/local/include/ariannamethod -O2
#cgo darwin CFLAGS: -I/opt/homebrew/include
#include <notorch.h>
#include <malloc.h>
#include <stdint.h>

// One glibc malloc accounting snapshot. mallinfo2 is the 64-bit form; the
// original mallinfo truncates to int and wraps above 2 GB, which on a phone
// running a training tape is not a theoretical concern.
typedef struct { uint64_t arena, hblkhd, uordblks, fordblks, keepcost; } ntx_malloc;
static void ntx_mallinfo(ntx_malloc* out) {
    struct mallinfo2 mi = mallinfo2();
    out->arena    = (uint64_t)mi.arena;     // sbrk heap, bytes
    out->hblkhd   = (uint64_t)mi.hblkhd;    // mmap'd blocks, bytes
    out->uordblks = (uint64_t)mi.uordblks;  // in use in the arena
    out->fordblks = (uint64_t)mi.fordblks;  // free in the arena, still ours
    out->keepcost = (uint64_t)mi.keepcost;  // releasable top of the arena
}

// One census of the live notorch tape: how many entries it holds and how many
// bytes sit in their outputs, their gradients, the mirrored parameters and the
// Chuck moment slots. Taken between the backward and the clear it is the exact
// cost of one step rather than an estimate from the shapes.
//
// The moment bytes are derived, not read: nt_tape_param allocates exactly two
// moment tensors of the parameter's own length for every entry that owns an
// optimizer slot, and none for one registered frozen (slot == -1).
typedef struct {
    uint64_t entries, params, out_bytes, grad_bytes, param_bytes, slot_bytes;
} ntx_tape_census;
static void ntx_tape_census_read(ntx_tape_census* c) {
    c->entries = c->params = c->out_bytes = c->grad_bytes = 0;
    c->param_bytes = c->slot_bytes = 0;
    nt_tape* tp = nt_tape_get();
    if (!tp) return;
    c->entries = (uint64_t)tp->count;
    for (int i = 0; i < tp->count; i++) {
        nt_tape_entry* e = &tp->entries[i];
        uint64_t ob = e->output ? (uint64_t)e->output->len * sizeof(float) : 0;
        if (e->is_param) {
            c->params++;
            c->param_bytes += ob;
            if (e->slot >= 0) c->slot_bytes += 2 * ob;
        } else {
            c->out_bytes += ob;
        }
        if (e->grad) c->grad_bytes += (uint64_t)e->grad->len * sizeof(float);
    }
}
*/
import "C"

// cHeapStats reports what glibc's allocator holds for the C side — the notorch
// tape and its tensors, OpenBLAS buffers, sqlite. arena+mmapped is what the
// process took from the kernel; inUse is what is actually held, and free is
// what the allocator is keeping back rather than returning.
func cHeapStats() (arena, mmapped, inUse, free int64) {
	var mi C.ntx_malloc
	C.ntx_mallinfo(&mi)
	return int64(mi.arena), int64(mi.hblkhd), int64(mi.uordblks), int64(mi.fordblks)
}

// ntTapeCensus reports the live tape: entry count, parameter count, and bytes in
// activations, gradients, mirrored parameters and Chuck moment slots.
func ntTapeCensus() (entries, params, outBytes, gradBytes, paramBytes, slotBytes int64) {
	var c C.ntx_tape_census
	C.ntx_tape_census_read(&c)
	return int64(c.entries), int64(c.params), int64(c.out_bytes), int64(c.grad_bytes),
		int64(c.param_bytes), int64(c.slot_bytes)
}

package main

/*
#include <malloc.h>
static int ntx_malloc_trim(void) { return malloc_trim(0); }
*/
import "C"

// The notorch tape allocates and frees every activation and every gradient of
// every step: at stage 4 that is 31.3 MB of activations and 30.5 MB of their
// gradients, built and torn down 32 times per burst. glibc raises its dynamic
// mmap threshold as those blocks are freed, so after the first bursts the
// tensors come from the sbrk arena instead of their own mappings, and a free
// there returns nothing to the kernel — it only lengthens a free list the
// process keeps.
//
// Measured on this phone, one organism restored from the live stage-4 earth
// checkpoint (MOLEQULALOG2.md, 2026-09-15): between bursts the arena stood at
// 150 MB with 1 MB of it in use, and the second burst took it to 270 MB with
// 4 MB in use. The bytes are not leaked — every tensor is freed — and they are
// not fragmentation the allocator needs: they are pages nobody asked it to
// give back. malloc_trim walks the free lists of every arena and hands the
// whole pages among them to the kernel, which is exactly this shape.
//
// releaseTrainingHeap is called once per training phase, after the tape is
// clear and the mirror is freed, where nothing the organism owns is in flight.
func releaseTrainingHeap() {
	if !CFG.TrimHeapAfterTrain {
		return
	}
	C.ntx_malloc_trim()
}

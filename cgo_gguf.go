package main

/*
#cgo CFLAGS: -I/usr/local/include/ariannamethod -O2
#cgo darwin CFLAGS: -I/opt/homebrew/include
#include <gguf.h>
#include <stdlib.h>

// A vector of C strings, for the type-9 array keys — the tokenizer's tokens and
// merges and the head topology. Built element by element from Go because cgo
// cannot hand a []*C.char to C in one piece.
static char** ggx_strv_new(int n) { return (char**)calloc((size_t)n, sizeof(char*)); }
static void   ggx_strv_set(char** v, int i, char* s) { v[i] = s; }
static void   ggx_strv_free(char** v, int n) {
    if (!v) return;
    for (int i = 0; i < n; i++) free(v[i]);
    free(v);
}

// The tensor directory, by index. The loader walks what the file declares rather
// than asking for names it expects, so a checkpoint with one matrix more or less
// than this binary would build is read as it was written.
static const char* ggx_tensor_name(const gguf_file* gf, int i) { return gf->tensors[i].name; }
static uint32_t    ggx_tensor_ndim(const gguf_file* gf, int i) { return gf->tensors[i].ndim; }
static uint64_t    ggx_tensor_dim (const gguf_file* gf, int i, int d) {
    return d < (int)gf->tensors[i].ndim ? gf->tensors[i].shape[d] : 0;
}

// One row of an F32 tensor, widened into the caller's float64 row. The mapping is
// the only copy: no float32 buffer stands between the file and the organism's
// array, which is the whole reason the checkpoint stopped being JSON.
static int ggx_row_f64(const gguf_file* gf, int idx, uint64_t row, double* dst, uint64_t n) {
    if (!gf || !dst || idx < 0 || idx >= (int)gf->n_tensors) return -1;
    const gguf_tensor_info* t = &gf->tensors[idx];
    if (t->dtype != GGUF_TYPE_F32) return -1;
    if (n == 0 || row + 1 > t->n_elements / n) return -1;
    uint64_t off = t->offset + row * n * 4;
    if (off + n * 4 > gf->data_size) return -1;
    const float* src = (const float*)(const void*)(gf->data + off);
    for (uint64_t i = 0; i < n; i++) dst[i] = (double)src[i];
    return 0;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════════════════
// CGO bridge to notorch's GGUF reader and writer (gguf.h).
//
// The reader has been in notorch since the eye; the writer landed 2026-09-16 and
// is what lets an organism trained here put its weights in a file something else
// can map. Alignment is fixed at 32 inside the library and general.alignment is
// deliberately not written — gguf_open computes the data offset without reading
// it, so a file that declared another alignment would not load.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Writing ──

type ggufWriter struct{ w *C.gguf_writer }

// ggufWriteOpen creates (truncating) path. Every later call on a failed writer
// returns an error without touching the file, so one check at close is enough —
// but each wrapper here reports, because a Go caller reads errors, not flags.
func ggufWriteOpen(path string) (*ggufWriter, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	w := C.gguf_write_open(cp)
	if w == nil {
		return nil, fmt.Errorf("gguf_write_open %s", path)
	}
	return &ggufWriter{w: w}, nil
}

func ggufErr(rc C.int, what string) error {
	if rc != 0 {
		return errors.New(what)
	}
	return nil
}

func (g *ggufWriter) kvStr(key, val string) error {
	ck, cv := C.CString(key), C.CString(val)
	defer C.free(unsafe.Pointer(ck))
	defer C.free(unsafe.Pointer(cv))
	return ggufErr(C.gguf_write_kv_str(g.w, ck, cv), "gguf kv str "+key)
}

func (g *ggufWriter) kvU32(key string, v uint32) error {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	return ggufErr(C.gguf_write_kv_u32(g.w, ck, C.uint32_t(v)), "gguf kv u32 "+key)
}

func (g *ggufWriter) kvI32(key string, v int32) error {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	return ggufErr(C.gguf_write_kv_i32(g.w, ck, C.int32_t(v)), "gguf kv i32 "+key)
}

func (g *ggufWriter) kvU64(key string, v uint64) error {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	return ggufErr(C.gguf_write_kv_u64(g.w, ck, C.uint64_t(v)), "gguf kv u64 "+key)
}

func (g *ggufWriter) kvBool(key string, v bool) error {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	b := C.int(0)
	if v {
		b = 1
	}
	return ggufErr(C.gguf_write_kv_bool(g.w, ck, b), "gguf kv bool "+key)
}

func (g *ggufWriter) kvStrArray(key string, vals []string) error {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	n := C.int(len(vals))
	v := C.ggx_strv_new(n)
	if v == nil {
		return errors.New("gguf kv array " + key + ": out of memory")
	}
	defer C.ggx_strv_free(v, n)
	for i, s := range vals {
		C.ggx_strv_set(v, C.int(i), C.CString(s))
	}
	return ggufErr(C.gguf_write_kv_str_array(g.w, ck,
		(**C.char)(unsafe.Pointer(v)), C.uint64_t(len(vals))), "gguf kv array "+key)
}

// declF32 declares one F32 tensor. shape[0] is the fastest-moving dimension, so
// a molequla matrix of Nout rows by Nin columns declares {Nin, Nout}.
func (g *ggufWriter) declF32(name string, shape []uint64) error {
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	return ggufErr(C.gguf_write_tensor_decl(g.w, cn, C.uint32_t(len(shape)),
		(*C.uint64_t)(unsafe.Pointer(&shape[0])), C.GGUF_TYPE_F32), "gguf decl "+name)
}

// tensorBegin / chunkF32 / tensorEnd deliver one declared tensor a row at a time,
// which is why the writer's peak is one row and not one matrix.
func (g *ggufWriter) tensorBegin(name string) error {
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	return ggufErr(C.gguf_write_tensor_begin(g.w, cn), "gguf begin "+name)
}

func (g *ggufWriter) chunkF32(row []float32) error {
	if len(row) == 0 {
		return nil
	}
	return ggufErr(C.gguf_write_tensor_chunk_f32(g.w,
		(*C.float)(unsafe.Pointer(&row[0])), C.uint64_t(len(row))), "gguf chunk")
}

func (g *ggufWriter) tensorEnd() error {
	return ggufErr(C.gguf_write_tensor_end(g.w), "gguf tensor end")
}

// close finishes the file. It returns an error having REMOVED a file that would
// have been incomplete, so a failed close leaves nothing to load.
func (g *ggufWriter) close() error {
	rc := C.gguf_write_close(g.w)
	g.w = nil
	return ggufErr(rc, "gguf_write_close")
}

func (g *ggufWriter) abort() {
	if g.w != nil {
		C.gguf_write_abort(g.w)
		g.w = nil
	}
}

// ── Reading ──

type ggufFile struct{ f *C.gguf_file }

func ggufOpen(path string) (*ggufFile, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	f := C.gguf_open(cp)
	if f == nil {
		return nil, fmt.Errorf("gguf_open %s", path)
	}
	return &ggufFile{f: f}, nil
}

func (g *ggufFile) close() {
	if g.f != nil {
		C.gguf_close(g.f)
		g.f = nil
	}
}

func (g *ggufFile) nTensors() int { return int(g.f.n_tensors) }

func (g *ggufFile) tensorName(i int) string {
	return C.GoString(C.ggx_tensor_name(g.f, C.int(i)))
}

// tensorShape returns the declared dimensions, shape[0] first.
func (g *ggufFile) tensorShape(i int) []uint64 {
	nd := int(C.ggx_tensor_ndim(g.f, C.int(i)))
	out := make([]uint64, nd)
	for d := 0; d < nd; d++ {
		out[d] = uint64(C.ggx_tensor_dim(g.f, C.int(i), C.int(d)))
	}
	return out
}

func (g *ggufFile) findTensor(name string) int {
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	return int(C.gguf_find_tensor(g.f, cn))
}

// readRow widens row `row` of tensor `idx` into dst, straight from the mapping.
func (g *ggufFile) readRow(idx int, row int, dst []float64) error {
	if len(dst) == 0 {
		return nil
	}
	if C.ggx_row_f64(g.f, C.int(idx), C.uint64_t(row),
		(*C.double)(unsafe.Pointer(&dst[0])), C.uint64_t(len(dst))) != 0 {
		return fmt.Errorf("gguf tensor %d row %d of %d values is not readable", idx, row, len(dst))
	}
	return nil
}

// kvU64 reads an unsigned scalar (u32, u64 or bool) by key.
func (g *ggufFile) kvU64(key string) (uint64, bool) {
	kv := g.kv(key)
	if kv == nil {
		return 0, false
	}
	switch kv._type {
	case 4: // UINT32
		return uint64(*(*C.uint32_t)(unsafe.Pointer(&kv.val))), true
	case 10: // UINT64
		return uint64(*(*C.uint64_t)(unsafe.Pointer(&kv.val))), true
	case 7: // BOOL
		if *(*C.uint8_t)(unsafe.Pointer(&kv.val)) != 0 {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// kvI32 reads a signed 32-bit scalar by key — last_warmup_stage is -1 on an
// organism that has never warmed up, so this one cannot be folded into kvU64.
func (g *ggufFile) kvI32(key string) (int32, bool) {
	kv := g.kv(key)
	if kv == nil || kv._type != 5 { // INT32
		return 0, false
	}
	return int32(*(*C.int32_t)(unsafe.Pointer(&kv.val))), true
}

// kvStr reads a string value by key. gguf_open keeps the first 255 bytes of one,
// which is why every long string this format writes goes through ggufReadStrKV.
func (g *ggufFile) kvStr(key string) (string, bool) {
	kv := g.kv(key)
	if kv == nil || kv._type != 8 { // STRING
		return "", false
	}
	return C.GoString((*C.char)(unsafe.Pointer(&kv.val))), true
}

func (g *ggufFile) kv(key string) *C.gguf_kv {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	return C.gguf_get_kv(g.f, ck)
}

// ggufReadStrArray re-scans the file for a type-9 string array. gguf_open skips
// arrays, so the tokenizer comes back through this rather than through the open
// handle.
func ggufReadStrArray(path, key string) ([]string, error) {
	cp, ck := C.CString(path), C.CString(key)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(ck))
	var n C.int
	arr := C.gguf_read_str_array(cp, ck, &n)
	if arr == nil {
		return nil, fmt.Errorf("gguf: %q is not a string array in %s", key, path)
	}
	out := make([]string, int(n))
	base := (*[1 << 20]*C.char)(unsafe.Pointer(arr))
	for i := 0; i < int(n); i++ {
		out[i] = C.GoString(base[i])
		C.free(unsafe.Pointer(base[i]))
	}
	C.free(unsafe.Pointer(arr))
	return out, nil
}

// ggufReadStrKV reads one string value without opening the tensor data, and
// without the 255-byte truncation gguf_open's union imposes.
func ggufReadStrKV(path, key string, cap int) (string, bool) {
	cp, ck := C.CString(path), C.CString(key)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(ck))
	buf := make([]byte, cap)
	if C.gguf_read_str_kv(cp, ck, (*C.char)(unsafe.Pointer(&buf[0])), C.int(cap)) != 0 {
		return "", false
	}
	n := 0
	for n < cap && buf[n] != 0 {
		n++
	}
	return string(buf[:n]), true
}

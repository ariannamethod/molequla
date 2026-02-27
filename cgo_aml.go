package main

/*
#cgo CFLAGS: -I${SRCDIR}/ariannamethod -O2 -fopenmp
#cgo LDFLAGS: -lm -lpthread -lgomp
#include "ariannamethod.h"
#include "ariannamethod.c"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════════════════
// AML C Bridge — CGO bindings to ariannamethod.c autograd
// "No Python. No PyTorch. No Go autograd. Just C and AML."
// ═══════════════════════════════════════════════════════════════════════════════

var amlInitialized bool

func amlInit() {
	if amlInitialized {
		return
	}
	C.am_init()
	C.am_persistent_mode(1)
	amlInitialized = true
}

func amlExec(script string) error {
	cs := C.CString(script)
	defer C.free(unsafe.Pointer(cs))
	rc := C.am_exec(cs)
	if rc != 0 {
		errMsg := C.GoString(C.am_get_error())
		return fmt.Errorf("aml_exec: %s", errMsg)
	}
	return nil
}

func amlSetArray(name string, data []float32) {
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	C.am_set_var_array(cn, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

func amlSetMatrix(name string, data []float32, rows, cols int) {
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	C.am_set_var_matrix(cn, (*C.float)(unsafe.Pointer(&data[0])), C.int(rows), C.int(cols))
}

func amlGetArray(name string) []float32 {
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	var clen C.int
	ptr := C.am_get_var_array(cn, &clen)
	if ptr == nil || clen <= 0 {
		return nil
	}
	length := int(clen)
	result := make([]float32, length)
	cSlice := (*[1 << 30]C.float)(unsafe.Pointer(ptr))[:length:length]
	for i := 0; i < length; i++ {
		result[i] = float32(cSlice[i])
	}
	return result
}

func amlGetFloat(name string) float32 {
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	return float32(C.am_get_var_float(cn))
}

func amlClear() {
	C.am_persistent_clear()
	amlInitialized = false
}

package main

/*
#cgo CFLAGS: -I${SRCDIR}/ariannamethod
#include "ariannamethod.h"
*/
import "C"

import (
	"sync"
	"unsafe"
)

// cgo bindings the witness uses from the AML core (repair 7). The C is
// compiled once, inside cgo_aml.go's unit; this file only declares the calls.
//
// HN (HarmonicNet) and M (METHOD) in ariannamethod.c are file-static process
// globals — one instance per process, not reentrant. Every clear/push/forward
// sequence runs under witnessCMu. am_method_step is deliberately not bound:
// it advances AML field physics and executes AML statements per action, and
// a witness does not steer.

var witnessCMu sync.Mutex

const wGammaDim = int(C.AM_HARMONIC_GAMMA_DIM)

func wHarmonicInit()                 { C.am_harmonic_init() }
func wHarmonicClear()                { C.am_harmonic_clear() }
func wHarmonicPushEntropy(h float64) { C.am_harmonic_push_entropy(C.float(h)) }

// wHarmonicPushGamma registers one organism for the forward pass. Go cores
// write no gamma vector into mesh.db, so the witness pushes zeros: layer 2
// (pairwise gamma cosines) then reads 0 for every pair and is not reported.
func wHarmonicPushGamma(id int, gamma []float32, entropy float64) {
	var p *C.float
	if len(gamma) > 0 {
		p = (*C.float)(unsafe.Pointer(&gamma[0]))
	}
	C.am_harmonic_push_gamma(C.int(id), p, C.int(len(gamma)), C.float(entropy))
}

// wHarmonic is am_harmonic_forward's result: the sine DFT of the field
// entropy history (layer 1), the dominant harmonic and the confidence
// multiplier (layer 3).
type wHarmonic struct {
	Harmonics   [8]float64 `json:"harmonics"`
	Dominant    int        `json:"dominant"`
	StrengthMod float64    `json:"strength_mod"`
	N           int        `json:"n"`
}

func wHarmonicForward(step int) wHarmonic {
	r := C.am_harmonic_forward(C.int(step))
	var out wHarmonic
	for k := 0; k < len(out.Harmonics); k++ {
		out.Harmonics[k] = float64(r.harmonics[k])
	}
	out.Dominant = int(r.dominant_freq)
	out.StrengthMod = float64(r.strength_mod)
	out.N = int(r.n_organisms)
	return out
}

func wMethodInit()  { C.am_method_init() }
func wMethodClear() { C.am_method_clear() }

// wMethodPushOrganism pushes one organism snapshot; gamma magnitude and
// cosine are 0 because nothing writes them (see wHarmonicPushGamma).
func wMethodPushOrganism(id int, entropy, syntropy float64) {
	C.am_method_push_organism(C.int(id), C.float(entropy), C.float(syntropy), 0, 0)
}
func wMethodFieldEntropy() float64  { return float64(C.am_method_field_entropy()) }
func wMethodFieldSyntropy() float64 { return float64(C.am_method_field_syntropy()) }

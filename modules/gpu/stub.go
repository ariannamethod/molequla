//go:build !linux || !cuda

package gpu

// ═══════════════════════════════════════════════════════════════════════════════
// GPU stubs — every build that is not linux + `-tags cuda`, which is every
// build the phones and macOS run.
//
// The CUDA toolchain is wired only through notorch_cuda.go on Linux. Everywhere
// else the package is pure Go and compiles no cgo at all: Ready() returns false,
// so the dispatcher in molequla.go takes the CPU/BLAS path and Matvec is never
// reached at runtime. Signatures mirror the exported API in bindings_cuda.go and
// forward_cuda.go exactly.
// ═══════════════════════════════════════════════════════════════════════════════

func Init() int                                          { return -1 }
func Shutdown()                                          {}
func Ready() bool                                        { return false }
func CacheWeight(name string, h []float32) bool          { return false }
func Matvec(key string, x []float32, nout int) []float32 { return nil }

# Audit B — molequla training paths and vendored libraries

Node: phone-1 (Galaxy A56, Ubuntu 24.04 chroot, aarch64, glibc 2.39). Read-only:
nothing was built, nothing was run, no git state was changed. Every number below
is cited to a file and line that was read, or to a tool output reproduced inline.

Trees compared:

- molequla `/data/data/com.termux/files/home/arianna/molequla`, HEAD `4fc05d5`
  (`git log -1`, 2026-09-13).
- canon notorch `/data/data/com.termux/files/home/arianna/notorch`, HEAD
  `5091547` (2026-09-13), `notorch.c` 9085 lines, `notorch.h` 833 lines (`wc -l`).
- vendored copy `molequla/ariannamethod/notorch.{c,h}`, 4739 / 694 lines,
  last touched by `e5c66fb` 2026-05-14 (`git log -- ariannamethod/notorch.c`).

## 0. The finding that reframes the rest: there are THREE notorch versions here, and the vendored one is not the one that runs

`cgo_notorch.go:4-7` compiles against `-I/usr/local/include/ariannamethod` and
`#include <notorch.h>`; `cgo_notorch_cpu.go:11` links `-L/usr/local/lib
-lnotorch`. Nothing in the Go build compiles `ariannamethod/notorch.c`: the only
`#include` of a vendored `.c` is `cgo_aml.go:11` (`#include "ariannamethod.c"`),
and `ariannamethod.c` does not include `notorch.c` (`grep -n notorch
ariannamethod/ariannamethod.c` returns only comments and the unrelated
`am_notorch_step` delta routine at line 6898). The vendored `notorch.c` is
compiled by exactly one thing in the tree — `ariannamethod/Makefile:34,45` —
producing `libaml.so`, which is loaded only by the Python tier
(`tests/test_all.sh:63,181,252`), and which never calls a single `nt_*` tape
symbol.

So the training tape that the phone actually executes comes from
`/usr/local/lib/libnotorch.a`, dated Aug 10 01:41, `ar t` = `notorch.o gguf.o`,
built with BLAS (`nm -u` shows `cblas_sgemm`, `cblas_sgemv`, `cblas_sger`
undefined) and with threads (`pthread_create`). Its exported set includes
`nt_qmatvec`/`nt_qmatvec_i8` but not `nt_relu`/`nt_seq_gate`, so it sits between
the vendored May state and canon HEAD — around `a30329b` (2026-08-10).

Three-way divergence, then:

| copy | date | reachable from the Go trainer |
|---|---|---|
| `ariannamethod/notorch.{c,h}` | 2026-05-14 | header only (see P1-5); `.c` never compiled |
| `/usr/local/lib/libnotorch.a` + `/usr/local/include/ariannamethod/notorch.h` | 2026-08-10 | **this is what runs** |
| canon `~/arianna/notorch` | 2026-09-13 | nothing |

`MOLEQULALOG2.md:34-40` records the vendored drift as open but treats the
vendored copy as the thing molequla uses. It is not.

---

## 1. VENDOR DELTA MAP

### 1.1 Symbols molequla calls

`grep -oh "C\.nt_[a-z_0-9]*" *.go | sort -u` gives 32 names. Thirty are reachable
on the CPU build; two (`nt_gpu_dispatch_count`, `nt_gpu_dispatch_reset`) are
declared only inside `cgo_notorch_cuda.go:19-22` (build tag `cuda`) and are
absent from the installed archive — correctly, since `gpu_notorch_stub.go:17`
replaces them. All thirty CPU-path symbols are present in the installed archive
(checked one by one against `nm -g --defined-only /usr/local/lib/libnotorch.a`).

Lifecycle: `nt_tensor_new`, `nt_tensor_new2d`, `nt_tensor_free`,
`nt_tensor_sync_cpu`, `nt_tape_start`, `nt_tape_clear`, `nt_tape_destroy`,
`nt_tape_get`, `nt_tape_param`, `nt_tape_param_frozen`, `nt_tape_no_decay`,
`nt_tape_record`. Autograd: `nt_tape_backward`, `nt_tape_clip_grads`,
`nt_tape_chuck_step`, `nt_nan_guard_new`, `nt_nan_guard_check`. Forward:
`nt_seq_embedding`, `nt_rope`, `nt_seq_rmsnorm`, `nt_seq_linear`,
`nt_mh_causal_attention`, `nt_rrpram_lowrank_attention`, `nt_add`, `nt_mul`,
`nt_silu`, `nt_seq_cross_entropy`. Mode: `nt_train_mode`, `nt_seed`,
`nt_set_gpu_mode`.

### 1.2 Which of them changed in canon, and whether it matters on ARM CPU

`diff -u -F '^[a-zA-Z_].*(' ariannamethod/notorch.c <canon>/notorch.c` = 5060
lines, 107 `-` lines and 4453 `+` lines. Classified by hunk:

**Changed in a way that reaches molequla's CPU path — exactly one function.**

`nt_tape_chuck_step` (canon `notorch.c:2620-2624`) replaced

```c
if (!e->is_param || !e->grad) continue;            // vendored, ariannamethod/notorch.c
```

with

```c
if (!e->is_param) continue;
if (!e->grad) { param_idx++; continue; }   // registered param w/o grad this step: keep slot alignment, skip update
```

The same rewrite is in `nt_tape_adam_step` (2431), `nt_tape_adamw_step` (2461)
and `nt_tape_accum_grads` (2812). It is present at the oldest commit in the local
canon clone, `51bc611` 2026-08-10 (`git show 51bc611:notorch.c | grep -n "keep
slot alignment"` → lines 2421/2451/2611/2802), so the installed Aug-10 archive
already carries it. It is the root of P0-1 below.

**Changed but CUDA-only, therefore dead on this phone.** `nt_tape_clip_grads`
(canon 2737-2779) gained the `gpu_nrm2_batch` single-readback restructure dated
2026-06-03 in its own comment; the CPU arm is bit-identical to the vendored loop.
`nt_tape_chuck_step` gained the same batching at 2591-2618, inside `#ifdef
USE_CUDA`. `NT_OP_MUL` backward gained a `gpu_mul_backward` arm (canon hunk at
old line 590). All of this is the 2026-05/06 CPU-sync work the brief asks about:
it is entirely `#ifdef USE_CUDA`, and the phone builds without `-tags cuda`
(`gpu_notorch_stub.go:1`, `cgo_notorch_cpu.go:1`). **A resync buys the phone none
of it.**

**Changed cosmetically / hardening.** `nt_tensor_new(int)` → `nt_tensor_new(size_t)`
(canon `notorch.h:42`; installed `notorch.h:42` still `int`), landed at or before
`51bc611`, with matching `nt_tensor_new((size_t)T * D)` casts at every call site.
`nt_tape_clear` (canon 308-310) and `nt_tape_record/record3/record4/param`
(355, 375, 397, 416-419) now reset `e->frozen = 0` defensively. Benign for
molequla, mildly good.

**Added and unreachable.** The 3854-line block appended after `nt_blas_matvec`
(diff hunk `@@ -4502,6 +5000,3854 @@`) is the packed-kernel family —
`nt_qmatvec`, `nt_qmatvec_i8`, `nt_qmatvec_i8_rows/_gather`, `nt_qmatmul*`,
`nt_quant_act*`, `nt_quantize_row`, the `NT_QMV_POOL` persistent worker pool,
`nt_qmv_set_thread_min`, `nt_qmv_planned_threads` (declared canon
`notorch.h:505-565`). New tape ops `NT_OP_RELU` (35) and `NT_OP_SEQ_GATE` (36),
`NT_OP_RRPRAM_BCAST` backward, `nt_rrpram_broadcast_attention` with a new `rank`
argument, image ops (`nt_conv2d`, `nt_group_norm`, `nt_im2col`,
`nt_upsample_nearest`, `nt_attention`).

**None of the packed kernels touch molequla's dense f32 matvec.** The trainer's
hot op is `nt_seq_linear`, and canon `notorch.c:3124-3155` shows it calls
`cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, out_dim, in_dim, ...)`
on f32 weights; `nt_qmatvec*` operate on GGUF-packed `uint8_t*` buffers and have
no f32 entry point. The thread pool is the *quantized* matvec pool. `nt_par_for`
lives in `harness/runtime.c`, not in `notorch.c`, and the 2026-09-13 attention
parallelisation applies to the harness inference loop, not to the tape's
`nt_mh_causal_attention` (whose only diff hunks are `size_t` casts and a GPU
guard). GGUF mmap is in `gguf.c`, which molequla never calls.

**Answer to "which canon changes would molequla gain from a resync": on a
CPU-only ARM phone, none that affect speed or numerics.** What a resync would
change is the Chuck slot semantics — in the wrong direction (P0-1) — and it would
break the build at `cgo_notorch.go:52`, `C.nt_tensor_new(C.int(length))`, because
canon's parameter is now `size_t`.

### 1.3 molequla-specific hunks in the vendored copy: there are none

The 107 `-` lines are, in full: the banner comment (`// fuck torch`, dropped
upstream); the pre-`size_t` allocation signatures; the dead statics `tape_find` /
`tape_ensure` (vendored `notorch.c` around old line 458, removed upstream); the
dead profiler statics `now_ms` / `g_alloc_bytes` and `#include <sys/time.h>`
(old line 2685); and the old shapes of the four optimizer loops and the CUDA
branches listed above. Not one hunk is a molequla-side edit. The vendored copy is
an unmodified, stale snapshot, so `CLAUDE.md:97-99`'s requirement to "name the
molequla-specific delta" resolves to: **the delta is empty**. That is worth
writing down once, because it means the vendored `.c` can be deleted or replaced
wholesale without preserving anything.

Same for the headers: `diff ariannamethod/notorch.h
/usr/local/include/ariannamethod/notorch.h` is 64 lines and contains no `<` line
of substance — the installed header is a strict superset except for one arity
change, `nt_rrpram_broadcast_attention(..., int rank)` (installed `notorch.h:453`
vs vendored `notorch.h:442`), a function molequla does not call. Likewise
`diff ariannamethod/ariannamethod.h /usr/local/include/ariannamethod/ariannamethod.h`
is 51 lines, all `>` (the co-occurrence block `AM_COOC_MAX`, `cooc_src[]`,
`am_cooc_*`).

---

## 2. Training entry points and the post-growth freeze counter

### 2.1 How many paths exist, and which are live

| function | file:line | decrements `growthFreezeRemaining` | reachable |
|---|---|---|---|
| `ntWarmupTrain` | `notorch_trainer.go:430` | yes, 456-460 | yes — `molequla.go:6494-6496`, `6985-6987` |
| `ntBurstTrain` | `notorch_trainer.go:393` | yes, 406-410 | yes — `molequla.go:6564` |
| `amlTrainSteps` | `aml_trainer.go:140` | yes, 238-242 | only via `notorch_trainer.go:432` when `CFG.Trainer=="aml"` |
| `amlBurstTrain` | `aml_trainer.go:260` | yes, 333-337 | only via `notorch_trainer.go:395` |
| `trainSteps` | `molequla.go:6317` | yes, 6346-6349 | **no caller** (`grep -rn "\btrainSteps(" *.go` → definition only) |
| `notorchTrainSteps` | `molequla.go:6054` | yes, 6238-6243 | **no live caller** — the only call site is commented out at `molequla.go:6499` |

So: every path that can run today decrements. The `ff6ad49` regression (neo,
2026-05-15; its message names `trainSteps`, the notorch path, and the two AML
functions) is closed and has stayed closed. But the counter now lives in **six**
places, two of which are unreachable — the repo's own "if a counter lives in N
places, N is the bug" (`CLAUDE.md:76-79`) has gotten worse, not better, since the
fix.

Two second-order defects in the decrement itself:

- The four live paths subtract `steps` wholesale, but their loops `continue`
  without doing work when a sampled document tokenizes to fewer than 2 ids
  (`notorch_trainer.go:291-293`, `aml_trainer.go:183-185`). On a corpus of short
  lines the freeze counter drains faster than steps are actually taken.
  `notorchTrainSteps`, the dead one, is the only path that decrements per
  executed step (`molequla.go:6239`) — and even it skips the decrement on its own
  `continue` at 6069 and 6123.
- `ntTrainCore` itself never touches the counter; the decrement sits in the two
  wrappers. `ntBurstTrain`/`ntWarmupTrain` return before locking `model.mu` when
  `docs` is empty or `steps <= 0` (`398`, `435`), so no decrement happens then.
  That matches `amlTrainSteps:141` / `amlBurstTrain:261`. Consistent.

### 2.2 Do the notorch and AML paths agree? No — on four counts.

**Learning-rate schedule: they agree.** `ntWarmupTrain`'s `lrFor`
(`notorch_trainer.go:446-454`) computes `cosineLR(gs, gs-growthStepOffset)`, then
`× embryoEmbd/NEmbd`, then `× CFG.PostGrowthLRScale` when frozen. `amlTrainSteps`
(`aml_trainer.go:155-160`, recomputed per step at 227-232) does the same three
operations with the same operands. The burst paths also agree: both use
`burstLR × embryoEmbd/NEmbd` with no `PostGrowthLRScale`
(`notorch_trainer.go:404` vs `aml_trainer.go:271`).

**Gradient clipping: they do not agree.** The notorch path clips at global norm
1.0 before every optimizer step (`notorch_trainer.go:350`,
`ntTapeClipGrads(1.0)`). The AML script generated at `aml_trainer.go:50-56` emits
`TAPE BACKWARD loss` then `TAPE CHUCK_STEP lr loss` and no clip — even though AML
implements one (`ariannamethod/ariannamethod.c:4296-4300`, `TAPE CLIP_GRADS`,
backed by `am_tape_clip_grads` at 2158). `--trainer aml` therefore runs Chuck on
unclipped gradients.

**NaN guard: they do not agree.** The notorch path gates the optimizer step on
`nt_nan_guard_check` (`notorch_trainer.go:282, 349-352`) — a NaN batch is
detected, gradients are zeroed, and the step is skipped. The AML path has no
guard; it filters NaN only out of the *reported average* (`aml_trainer.go:214`)
after the step has already been applied.

**Chuck state across bursts and on growth: they do not agree, and the difference
is structural.** notorch keeps Chuck's positional `m`/`v` slots alive across
bursts (`cgo_notorch.go:86-87`) and wipes them only on growth, via
`ntOnGrowth()` → `ntTapeNeedsReset` → `ntTapeDestroy()`
(`notorch_trainer.go:30-34, 276-280`), called from `molequla.go:6664` and
`molequla.go:7020` — both growth sites, correctly. The AML path has no such
notion: `amlInit()` (`cgo_aml.go:40-47`) calls `am_init()`, whose first statement
is `am_tape_destroy()` (`ariannamethod.c`, `am_init` body), and every
`amlTrainSteps`/`amlBurstTrain` ends with `amlClear()` → `am_persistent_clear()`
and `amlInitialized = false` (`cgo_aml.go:95-98`). So the AML trainer restarts
Chuck momentum from zero on **every 32-step burst**, while notorch carries it for
the organism's whole stage. These are not two implementations of one trainer.

**Beyond the brief, but in the same wiring:** the AML path pushes and pulls only
`wte`, `wpe`, `lm_head` and the seven per-layer content matrices
(`aml_trainer.go:104-136`). It never touches `l*.wr_a` / `l*.wr_b` or
`l*.h*.alpha`, which inference does use (`molequla.go:886-887, 2921-2922`). Under
`--trainer aml` the RRPRAM organ is not trained at all.

---

## 3. `ariannamethod/ariannamethod.c:1500` — compiler range artefact, not a bug

The site, read verbatim:

```c
1484        case AM_OP_RMSNORM: {
1487            if (e->parent1 >= 0) {
1492                int n = out_len;
1493                float ss = 0;
1494                for (int i = 0; i < n; i++) ss += px->output->data[i] * px->output->data[i];
1495                float rms = sqrtf(ss / n + 1e-6f);
1500                float* gx = (float*)calloc(n, sizeof(float));
```

`out_len` is assigned once, at `ariannamethod.c:1314`, inside `am_tape_backward`:
`int out_len = e->output->len;` where `e->output` is an `AM_Array*`.

`AM_Array.len` (declared `ariannamethod.h:264-275`) is written in exactly one
place in the whole file — `grep -nE "(->|\.)len[[:space:]]*=[^=]"
ariannamethod/ariannamethod.c` returns a single hit, `ariannamethod.c:1066`,
inside `am_array_new`:

```c
1060 AM_Array* am_array_new(int len) {
1061     if (len <= 0 || len > AM_MAX_ARRAY_SIZE) return NULL;
...
1066     arr->len = len;
```

with `AM_MAX_ARRAY_SIZE 1048576` (`ariannamethod.h:262`). Every AM_Array in the
process therefore has `len ∈ [1, 1048576]`, and `out_len` at 1314 is provably in
that range. **A negative or zero length is not reachable.**

The warning is a value-range artefact: gcc reads `e->output->len` as a plain
`int` through a pointer, cannot see the constructor invariant across the
translation unit, and therefore must assume `[INT_MIN, INT_MAX]`; converting that
to the `size_t` parameter of `calloc` yields a range that includes values above
`PTRDIFF_MAX`, which is what `-Walloc-size-larger-than=` reports. Even in the
impossible case the code is safe: with `n < 0` the `calloc` returns `NULL` and the
`if (gx)` at 1501 skips the body; with `n == 0` the two preceding loops execute
zero iterations, `ss` stays 0, and `rms` at 1495 becomes `sqrtf(0.0f/0 + 1e-6f)`
→ NaN, but nothing is dereferenced and nothing is written.

Why this site fires and the syntactically identical `calloc(out_len,
sizeof(float))` at 1451 (SiLU), 1474 (softmax), 1517 (GELU) and 1544 (dropout)
apparently do not is not determined here — no compile was run in this audit, so I
am not going to assert a cause. The one structural difference at 1500 is that the
count is the local copy `n`, which has already been used as a divisor at 1495;
divisor use lets VRP exclude zero and leaves a split range, which is the shape
that the allocation warning keys on.

**Severity P2, no functional repair required.** If the warning is to be silenced
rather than suppressed, the honest edit is a guard that states the invariant the
compiler cannot see — `if (n <= 0) break;` before 1493 — which also removes the
`ss / n` divide-by-zero. Not a correctness fix; a legibility one.

---

## 4. Dead on a CPU-only ARM phone, and what could degrade numerics there

### 4.1 CUDA paths: compiled out, but still consulted

Build tags are clean — `cgo_notorch_cuda.go:1` (`//go:build cuda`),
`gpu_bindings_linux.go:1` and `gpu_forward.go:1` (`//go:build linux && cuda`),
with `!cuda` counterparts at `gpu_notorch_stub.go:1`, `gpu_bindings_stub.go:1`,
`gpu_forward_stub.go:1`. Nothing CUDA is compiled on the phone. What survives is
the *consultation*:

- `ntSetGPUForStage(model.CurrentGrowthStage())` is called at
  `molequla.go:6421` (trainer start) and `molequla.go:6665` (after every growth).
  On this build the body is `func ntSetGPUForStage(stage int) {}`
  (`gpu_notorch_stub.go:20`). The comment at 6665 — "flip CPU→GPU at teen/adult"
  — describes a thing that cannot happen here.
- `ntGPUDispatchCount()` returns a hardcoded 0 (`gpu_notorch_stub.go:17`) and is
  interpolated into **every** completion line the trainer prints
  (`notorch_trainer.go:413-414` and `463-464`), so every burst and warmup line on
  the phone ends `| gpu-dispatch=0`. That is the string in the log lines
  `MOLEQULALOG2.md:94-95` quotes as the throughput evidence.
- `ntGPUEnable()` is called once at `molequla.go:6825` and returns
  `(false, "trainer on CPU/BLAS (built without -tags cuda)")`.
- `--gpu` is still parsed and sets `CFG.UseGPU` (`molequla.go:6285-6290`); the
  dispatcher gates on `CFG.UseGPU && gpuReady()` and `gpuReady()` is false on this
  build (`gpu_forward_stub.go` header comment), so the flag is silently inert.

One CUDA concern has a real CPU cost. `notorch_trainer.go:168-173` freezes the
per-head blend gate — sigmoid precomputed in Go, registered via
`ntTapeParamFrozen` — explicitly "to keep `nt_sigmoid`/`nt_scale_by_t` off the
tape (notorch GPU-sync bug class)". On a CPU-only build there is no GPU sync bug
class, and the price is that `l*.h*.alpha` is never trained by any path: it is
allocated at `molequla.go:1889, 2265, 2291`, read at `notorch_trainer.go:182-187`
and at inference, and pinned forever at `CFG.HybridAlphaInit = 0.5`
(`molequla.go:318`), i.e. gate weight `sigmoid(0.5) = 0.622`. The workaround for a
device this phone does not have is costing it a trainable parameter.

Also dead: `ntTrainMode` and `ntSeed` (`cgo_notorch.go:158-166`) have zero
callers anywhere in the tree. `nt_train_mode` only gates dropout (canon
`notorch.c:2854-2856, 4471`), which molequla does not use, and the default is
already `1`, so nothing breaks — but the bridge advertises a mode switch that is
never thrown.

### 4.2 SIMD on aarch64 — the shim is not the risk; the hardcoded paths are

`ariannamethod/notorch_simd.h:1-24` is `<immintrin.h>`, AVX2+FMA, x86_64 only,
and the only way to reach it is `-DUSE_SIMD`, which `ariannamethod/Makefile:59-66`
refuses on non-x86_64 with an explicit error and which no cgo directive sets.
`ariannamethod/notorch.c:26-40` shows the include is behind `#ifdef USE_SIMD` and
`#error`s if combined with `USE_BLAS`. The installed archive resolves
`cblas_sgemm/sgemv/sger` externally (`nm -u`), proving it was built `USE_BLAS`,
not `USE_SIMD`. **The ARM build uses OpenBLAS for every f32 matmul; the AVX2 shim
is never reached, and there is no silent scalar fallback.** Canon has 73 NEON
references in `notorch.c` against 0 in the vendored copy, but all of them are in
the quantized `nt_qmatvec*` kernels — unreachable from a dense f32 tape.

The real aarch64 hazard is in the cgo directives, which name x86 multiarch paths
that do not exist on this machine:

- `cgo_notorch.go:6` — `-I/usr/include/x86_64-linux-gnu/openblas-pthread/`
- `cgo_notorch_cpu.go:11` — `-L/usr/lib/x86_64-linux-gnu/openblas-pthread/`
- `cgo_aml.go:8-9` — both of the above

`ls` confirms neither exists here; what exists is
`/usr/include/aarch64-linux-gnu/openblas-pthread` and
`/usr/lib/aarch64-linux-gnu/openblas-pthread`. The build only works because
`/usr/include/aarch64-linux-gnu` and `/usr/lib/aarch64-linux-gnu` are default
multiarch search directories, so `<cblas.h>` and `-lopenblas` fall through to the
Debian *alternatives* symlink. On this box that symlink happens to resolve
correctly — `readlink -f /usr/lib/aarch64-linux-gnu/libopenblas.so.0` →
`/usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so`. On a box
where the serial alternative is selected, the x86 build keeps its explicit
pthread path and the ARM build silently loses BLAS threading, with no message and
no failure. That is a perf cliff hidden behind an architecture-specific string.

Numerics themselves are not at risk: no `-ffast-math` anywhere, and the build
command recorded at `MOLEQULALOG2.md:76-77` uses `-O3 -march=native -mtune=native
-DUSE_BLAS`, all IEEE-preserving.

One item I could not settle read-only and will not assert: `capColonyThreads()`
(`molequla.go`) sets `OPENBLAS_NUM_THREADS` with `os.Setenv` from `main()`.
OpenBLAS reads that variable in a library constructor, which runs before `main`.
Whether the setting takes effect in the current process, or only decorates the
log line `[cpu] effective cores=4` quoted at `MOLEQULALOG2.md:130`, needs a
measurement (`openblas_get_num_threads()` at first burst, or a step-rate A/B with
the variable exported from the shell). Flagging, not claiming.

---

## 5. Memory: per-organism trainer allocation at adult (embd 320, 6 layers)

Everything here is an **estimate computed from the code**, not a measurement. No
adult has run on this phone yet; the largest measured figure in the tree is peak
`VmHWM` 231-240 MB at adolescent (`MOLEQULALOG2.md:136-138`).

Shape at adult, from `CFG.GrowthStages[5] = {500000, 320, 6, 8}`
(`molequla.go:268`): `D = 320`, `NLayer = 6`, `NHead = 8`, `HeadDim = 40`,
`T = BlockSize = 96` (`molequla.go:260`), `R = RRPRAMRank = 32`
(`molequla.go:299`), `V ≈ 643` (259 base tokens at `molequla.go:1399-1400` plus
`BPENumMerges = 384` at `molequla.go:313`).

Parameter floats, from the shapes in `ntContentParams`
(`notorch_trainer.go:48-59`) and `ensureRRPRAMFactors` (`molequla.go`, `wr_a` =
`NHead·NEmbd × R`, `wr_b` = `NHead·R × BlockSize`):

```
wte       643 × 320                      =    205,760
lm_head   643 × 320                      =    205,760
per layer 4·D² + 2·(4D·D) + (D·4D)       =  1,638,400     × 6 = 9,830,400
wr_a      (8·320) × 32           =  81,920
wr_b      (8·32)  × 96           =  24,576  → 106,496     × 6 =   638,976
                                                   total  = 10,880,896 floats
                                                          = 43.5 MB as f32
```

**Allocated once per burst, freed at burst end.** The weight mirror:
`ntTensorNew2D` + `ntTensorSet` per param (`notorch_trainer.go:227-231`), freed by
the `defer` at 232-236 — **43.5 MB**. The frozen gate vectors, `2 × T·D` floats
per RRPRAM layer (`notorch_trainer.go:191-201, 253-259`), freed by the `defer` at
261-273 — 368,640 floats, **1.5 MB**. On the Go heap and transient:
`ntFlattenMatrix` builds a fresh `[]float32` per param on the way in (line 229)
and `ntTensorGet` builds another on the way out (line 368) — **~43.5 MB of GC
churn each way, per burst**.

**Allocated and freed every step.** `ntTapeClear()` (`notorch_trainer.go:353`)
releases the whole tape each iteration. Counting the ops
`ntBuildForward` records (`notorch_trainer.go:104-134`): 20 entries per layer
with the RRPRAM branch live, so ~186 tape entries per step (50 params + 12 frozen
+ 2 inputs + 6×20 + final norm + logits + CE) — comfortably under
`NT_TAPE_MAX_ENTRIES 8192` (canon `notorch.h:87`), and 50 optimizer slots under
`NT_TAPE_MAX_PARAMS 512`. Activation floats ≈ `30·T·D` per layer:

```
forward activations   6 × 30 × 30,720 + T·D + T·V  ≈ 5,622,048 floats ≈ 22.5 MB
their backward grads  ≈ same shape                                    ≈ 22.5 MB
parameter grads       = 10,880,896 floats                             ≈ 43.5 MB
per-op backward scratch (dw/dx calloc per SEQ_MATVEC, freed in-case)  ≈  1.6 MB peak
                                                   per-step peak      ≈ 88 MB
```

**Kept, across steps and across bursts.** Chuck's `m` and `v`, allocated by
`nt_tape_param` (canon `notorch.c:422-439`) and living in `g_tape.adam[]` until
`nt_tape_destroy`: `2 × 43.5` = **87 MB**. Plus the static `g_tape` struct itself
— `entries[8192]` + `adam[512]` + `chuck_params[512]` (canon `notorch.h:207-215`)
— on the order of 0.5 MB in BSS.

**Not the trainer's, but in the same process:** `model.Base` is the canonical
`float64` store (`notorch_trainer.go:20-21`), so the same 10,880,896 values cost
**87 MB** again on the Go side, before deltas, the co-occurrence field and the
corpus.

Order of magnitude for one adult organism: **~220 MB C-side at the peak of a
step (mirror + tape + Chuck), plus ~87 MB Go-side base weights** — call it 300 MB
and change, against 231-240 MB measured at adolescent where the same arithmetic
gives ~50 MB of trainer, i.e. at adolescent the trainer is a minority of RSS and
at adult it becomes the majority. Four adults on an 8 GB phone is upward of
1.2 GB of trainer alone. Estimate, from the code; it wants a measurement.

And one term that is not in the table above because it should not exist: see
P1-1, the Chuck moments that are never freed on growth.

---

## Findings, by severity

### P0-1 — RRPRAM factors are silently untrained from stage 2 onward

**Mechanism.** molequla registers, per RRPRAM layer, one trainable `wr` and two
*frozen* gate vectors, in that order (`notorch_trainer.go:331-334`):

```go
wrIdx[l] = ntTapeParam(wrTensors[l])
gateCIdx[l] = ntTapeParamFrozen(gateCT[l])
gateRIdx[l] = ntTapeParamFrozen(gateRT[l])
```

`nt_tape_param_frozen` sets `e->is_param = 1` and `e->frozen = 1` and
**deliberately does not allocate a Chuck slot** — "INTENTIONAL: do NOT increment
`g_tape.n_params`" (canon `notorch.c:478`, identical in the vendored copy).
`tape_acc_grad` returns immediately for a frozen entry (canon `notorch.c:488`),
so after backward a frozen gate has `e->grad == NULL`.

`nt_tape_chuck_step` then walks the tape (canon `notorch.c:2621-2624`):

```c
for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
    nt_tape_entry* e = &g_tape.entries[i];
    if (!e->is_param) continue;
    if (!e->grad) { param_idx++; continue; }
```

A frozen gate matches `is_param && !grad`, so it **consumes a slot index it was
never given**. Each RRPRAM layer inflates `param_idx` by 2.

**Arithmetic at each stage.** With `P = 2 + 7·NLayer` content params and `L`
RRPRAM layers, `n_params = P + L`:

| stage | D | L | P | n_params | outcome |
|---|---|---|---|---|---|
| embryo | 16 | 1 | — | — | `CFG.HeadTypes = {"content"}` → no RRPRAM, unaffected |
| infant | 32 | 1 | 9 | 10 | `wr_0` gets `adam[9]`, loop exits — correct |
| child | 64 | 2 | 16 | 18 | `wr_0` → `adam[16]`; loop exits at i=18; **`wr_1` never updated** |
| adolescent | 128 | 4 | 30 | 34 | `wr_0` → `adam[30]`; `wr_1` → `adam[33]` (**wr_3's moments**); **`wr_2`, `wr_3` never updated** |
| adult | 320 | 6 | 44 | 50 | `wr_0` → `adam[44]`; `wr_1` → `adam[47]` (wrong); **`wr_2..wr_5` never updated** |

RRPRAM becomes active at infant, because growth sets
`CFG.HeadTypes = headTypesForNHead(newHead)` (`molequla.go:2377`) and
`headTypesForNHead(2)` returns `{"content","hybrid"}` (`molequla.go`), which makes
`layerHasHybrid()` true. The colony run of `MOLEQULALOG2.md:133-138` reached child
and adolescent, so this was live on that run.

Nothing reports it. The weights are registered, receive gradients, are clipped
into the global norm, and are mirrored back out unchanged; the loss curve simply
reflects a smaller model than the one that is written to disk and run. No error,
no log line, no NaN.

**Minimal repair.** Two options, both one line.

Upstream (correct, fixes it for every caller) — canon `notorch.c:2623-2624`, plus
the identical lines at 2431/2461/2812:

```c
    if (e->frozen) continue;                    /* frozen params own no slot */
    if (!e->grad) { param_idx++; continue; }
```

molequla-side (works without touching the lib) — stop registering the gates as
params. `ntTapeParamFrozen` exists to keep the gate on the tape so gradient flows
*through* it into `attn`; `nt_tape_record(t, NT_OP_NONE, -1, -1, 0)` — which
`ntTapeInput` (`cgo_notorch.go:101-103`) already wraps — gives exactly that
without `is_param = 1`. Change `notorch_trainer.go:333-334` to use `ntTapeInput`
and the inflation disappears. This needs a check that `NT_OP_MUL` backward
accumulates into a non-param `NT_OP_NONE` parent (it does; `tape_acc_grad` only
refuses on `frozen`), and the gate then accumulates a useless gradient buffer of
`T·D` floats per layer per step — 1.5 MB at adult, freed by `ntTapeClear`.

**Gate that goes red.** Train two steps at child with a distinguishable
`l1.wr_a`, read it back through `ntTensorGet`, assert it moved. Today it will not.

### P0-2 — the trained model is not the model that runs: `wpe`

`ForwardStep` adds a learned absolute position embedding to every token:

```go
2850  tokEmb := gpt.Base["wte"].Rows[tokenID]
2851  posEmb := gpt.Base["wpe"].Rows[posID%gpt.BlockSize]
2852  x := tokEmb.Add(posEmb)
```

(`molequla.go:2850-2852`), and *also* applies RoPE per head at 2908 and 2913. The
default trainer omits `wpe` entirely: `ntBuildForward` calls
`ntSeqEmbedding(wte, -1, tokIdx, T, D)` with `wpe_idx = -1` and the comment "WTE
only — RoPE handles position" (`notorch_trainer.go:104`); `nt_seq_embedding`
honours `-1` by skipping the position term (canon `notorch.c:3048-3082`), and
`ntContentParams` (`notorch_trainer.go:48-59`) never registers `wpe`.

So on the default path `wpe` stays at its initialisation — `NewMatrixParam(96,
NEmbd, 0.08)` at `molequla.go:1866`, grown with 0.001 fill at 2231 — a fixed
random 96×320 offset added to every hidden state at inference and never seen
during training. The AML trainer *does* train it (`seq_embed(wte, wpe, tokens,
seq_len)`, `aml_trainer.go:29`; pushed and pulled at 106 and 123), so the two
trainers optimise two different models and a `--trainer` flip changes the
objective, not just the backend.

**Minimal repair** is a decision, not a patch, and it belongs to Oleg: either
register `wpe` in `ntContentParams` and pass its index to `ntSeqEmbedding`, or
delete the `wpe` add at `molequla.go:2851` and make RoPE the only position
mechanism in all four cores. Deleting is the smaller change and matches the
trainer's stated design; adding is the smaller risk to existing checkpoints.
Whichever, `README.md` has to say which.

### P1-1 — Chuck moments leak on every growth event

`ntTapeDestroy()` is the post-growth Chuck wipe (`notorch_trainer.go:276-280`).
It is reached at the *top* of the next burst, after the previous burst's final
`ntTapeClear()` at line 353. But `nt_tape_clear` sets `g_tape.n_params = 0`
(canon `notorch.c:319`), and `nt_tape_destroy` frees optimizer state with

```c
for (int i = 0; i < g_tape.n_params; i++) {
    if (g_tape.adam[i].m) { nt_tensor_free(g_tape.adam[i].m); ... }
```

(canon `notorch.c`, `nt_tape_destroy` body — byte-identical in the vendored
copy, so this is long-standing, not a canon regression). With `n_params == 0` the
loop frees nothing, and the following `memset(&g_tape, 0, sizeof(g_tape))` drops
every pointer. The *semantic* goal still happens — the next `nt_tape_param` sees a
null `m` and allocates fresh — but the previous stage's `m`/`v` are unreachable.

Estimated leak, `2 × params × 4 B` per growth: ~0.1 MB (embryo→infant), 0.3,
2.0, 10.6, 37.7 MB — **about 51 MB abandoned by the time an organism is adult**,
on top of the 87 MB the adult's own moments occupy. Estimate.

**Minimal repair**, upstream: bound the free loop by `NT_TAPE_MAX_PARAMS` instead
of `n_params`, which is safe because the slots are null unless allocated.
molequla-side there is no clean fix — the state is private to the library.

### P1-2 — the progressive-sequence warmup is inert from infant onward

`molequla.go:6490-6496` and `6981-6987` implement "backprop with progressive
sequence length (short→full)":

```go
ntWarmupTrain(model, tok, docs, earlySteps, 8)   // very short seqs, batch=1
ntWarmupTrain(model, tok, docs, midSteps, 16)    // short seqs, batch=1
ntWarmupTrain(model, tok, docs, lateSteps, 32)   // medium seqs, batch=1
```

`ntWarmupTrain` honours the override (`notorch_trainer.go:441-443`), and then
`ntTrainCore` overwrites it (`notorch_trainer.go:218-222`):

```go
hasRRPRAM := layerHasHybrid()
if hasRRPRAM {
    seqLen = model.BlockSize
}
```

`layerHasHybrid()` is true from infant on (see P0-1). So from stage 1 every one
of those three calls runs at `seqLen = 96`, not 8/16/32 — the warmup costs
roughly 3-12× the per-step work the comment promises, and the measured rates in
`MOLEQULALOG2.md:93-95` (96-174 steps/s at embryo, 100-105 at infant, 34-41 at
child) are the rates of a flat-96 warmup at infant and child, and of a genuinely
progressive one only at embryo.

The reason for the pin is legitimate — op 33 packs `Wr_b` at width `BlockSize`
and assumes `T_r == T` (`notorch_trainer.go:216-218`; canon `notorch.c:3689`,
`int T_r = T; /* assumption */`). **Minimal repair:** either drop the three-call
ladder and say one warmup call at `BlockSize`, or repack `Wr_b` to the burst's
`T` before each call so the ladder means something. Do not leave a comment
describing behaviour that stopped at stage 0.

### P1-3 — the two trainers are not interchangeable, and the flag implies they are

`--trainer aml` (`molequla.go:6276-6279`) is presented as a backend choice. Per
§2.2 it changes: gradient clipping (on → off), the NaN guard (on → off), Chuck
momentum (stage-long → reset every burst), and the parameter set (RRPRAM `wr_a` /
`wr_b` trained → not trained). **Minimal repair:** emit `TAPE CLIP_GRADS 1.0`
between `TAPE BACKWARD` and `TAPE CHUCK_STEP` in `amlModelScript`
(`aml_trainer.go:54-55`) — AML already implements it at
`ariannamethod.c:4296-4300` — and write the remaining differences into the flag's
help text rather than leaving them to be discovered from a loss curve.

### P1-4 — the freeze counter does not freeze anything on the live paths

`MaybeGrowArchitecture` sets `growthFreezeRemaining = CFG.FreezeAfterGrowthSteps`
(500) at `molequla.go:2426`, described at 2425 as "only train deltas until new
weights stabilize", and `MaybeGrowArchitecture` refuses to grow again while it is
positive (`molequla.go:2206-2208`). Only the dead `trainSteps` actually
implements the base freeze (`molequla.go:6341-6357`: base params omitted from the
step when frozen). On both live paths the flag does one thing — multiply LR by
`CFG.PostGrowthLRScale = 0.3` (`notorch_trainer.go:450-452`,
`aml_trainer.go:158-160`). The base is fully trained throughout the "freeze".
**Minimal repair:** rename the field and the comment to what it is — a
post-growth LR damper and a re-growth cooldown — or implement the freeze on the
notorch path by registering the base tensors through `ntTapeParamFrozen`. The
second option collides with P0-1 until P0-1 is fixed.

### P1-5 — the vendored headers shadow the installed ones, silently

Both `cgo_aml.go:4` (`-I${SRCDIR}/ariannamethod`) and `cgo_notorch.go:4`
(`-I/usr/local/include/ariannamethod`) are in the same Go package, so cgo
concatenates them; the directories both contain `notorch.h` and
`ariannamethod.h`, and `-I` dirs are searched for `<...>` as well as `"..."`.
Go processes package files in sorted order and `cgo_aml.go` sorts before
`cgo_notorch.go`, so `${SRCDIR}/ariannamethod` is searched first and
`#include <notorch.h>` at `cgo_notorch.go:7` most likely resolves to the **May
vendored header**, while the linker binds the **August archive**.

Right now this is harmless, and I checked why rather than assuming: the two
headers agree on `struct nt_tensor` and `struct nt_tape_entry` field-for-field,
agree on every `#define NT_OP_*` (`diff` of the define block is empty), agree on
`nt_nan_guard`, and differ in exactly one prototype — `nt_rrpram_broadcast_attention`
gained `int rank` (installed `notorch.h:453`) — a function molequla does not
call. So it compiles and it is correct, by luck.

It will stop being correct the next time canon changes an arity on a symbol
molequla does call. The signature-change rate is not zero: canon HEAD already
moved `nt_tensor_new(int)` → `nt_tensor_new(size_t)` (canon `notorch.h:42`),
which `cgo_notorch.go:52` passes `C.int(length)` to.

**Minimal repair:** decide which tree owns the header and make the build say so.
Either drop `ariannamethod/notorch.h` from the include path (it is only needed by
`ariannamethod/Makefile`, which compiles inside that directory anyway), or vendor
everything and stop linking `/usr/local/lib/libnotorch.a`. Not both, silently.
**Verification without building:** `go build -n` prints the gcc command lines
without running them; the `-I` order is visible there. That check should be in
`tests/test_all.sh`.

### P1-6 — the vendored tree is four months stale, and the resync has a known blocker

`ariannamethod/notorch.c` is 4739 lines at 2026-05-14; canon is 9085 at
2026-09-13; `ariannamethod/ariannamethod.c` is likewise behind
`/usr/local/include/ariannamethod/ariannamethod.h` by the whole co-occurrence
block. `CLAUDE.md:97-99` forbids letting this drift silently, and
`MOLEQULALOG2.md:67-68` already lists the resync as open.

Per §1.2-1.3 the resync's technical value on this node is small — no CPU-path
kernel improvement reaches the f32 tape — and it carries P0-1. Its real value is
that it removes a four-month-stale copy of a file that looks authoritative and
is not. **When it is done:** `cgo_notorch.go:52` must become
`C.nt_tensor_new(C.size_t(length))`, and the log entry must say the
molequla-specific delta is empty (§1.3), because that is the fact and it is
easier to state than to re-derive next time.

### P2-1 — `ntTapeNoDecay` is a no-op under Chuck

`notorch_trainer.go:318` ("wte — no weight decay on embeddings") and 332
("low-rank factors: no weight decay (double-shrink)") call
`nt_tape_no_decay`. That flag is read in exactly one place in canon —
`nt_tape_adamw_step`, `notorch.c:2472` — and `nt_tape_chuck_step` contains zero
references to it (`sed -n '/^void nt_tape_chuck_step/,/^}/p' notorch.c | grep -c
no_decay` → 0). Chuck applies no weight decay at all, so there is nothing to
suppress. **Minimal repair:** delete the two calls and the two comments, or add
decay to Chuck deliberately. The current state documents an effect that does not
occur.

### P2-2 — `ariannamethod.c:1500`

See §3. Range artefact, not reachable. Optional `if (n <= 0) break;` before 1493.

### P2-3 — the Dockerfile drops the `-a` that the repo calls mandatory

`CLAUDE.md:57-59` states `-a` is mandatory ("without it Go reuses stale compiled
C"). `Dockerfile:37` is `go build -trimpath -ldflags="-s -w" -o /out/molequla .`.
Benign in a cold container, a trap the moment the layer is cached with a changed
`ariannamethod.c`. **Minimal repair:** add `-a`.

### P2-4 — dead trainer wiring

`trainSteps` (`molequla.go:6317`) and `notorchTrainSteps` (`molequla.go:6054`,
its call site commented out at 6499) have no callers, and both carry
freeze-counter logic that reads as live. `ntTrainMode` and `ntSeed`
(`cgo_notorch.go:158-166`) have no callers. `notorchStep`
(`molequla.go:5973`) exists only for the dead DFA path. **Minimal repair:**
delete, or move behind a build tag with a comment saying what would revive them.
Six copies of one invariant is how `ff6ad49` happened.

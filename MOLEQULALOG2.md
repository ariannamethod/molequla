# MOLEQULALOG 2

The second book of the molequla engineering log. `PROJECT_LOG.md` is the
first book — Phase A (GPU), Phase B (graze), Phase C (ecology), the §9
mitosis run and the cascade governor, through 2026-06-29. This book opens
with molequla landing on a phone.

`README.md` is the face and the spec. This file is the diary and the
reproducibility record. Numbers here come from artifacts, tool output or
git, cited inline by path, line or commit — never retyped from memory.
Entries are appended in date order and signed by the node that wrote them.
Small changes live here; a change that alters what molequla *is* also earns
README space, and the README never becomes a worklog.

---

## 2026-09-12 — molequla lands on phone-1 (Galaxy A56, Ubuntu 24.04 chroot, aarch64)

Clone of `github.com/ariannamethod/molequla` at `08fa1b1` (328 commits,
`git log --oneline | wc -l`) into `~/arianna/molequla/` on phone-1. The
session runs inside an Ubuntu 24.04 chroot on the phone (uid 0, glibc 2.39);
the Termux tree is bind-mounted at its own absolute path, so files written
from the chroot get their owner and SELinux label fixed afterwards.

Node toolchain, as found in the chroot (`which`, `go version`): gcc, go
1.22.2 linux/arm64. No rustc, cargo, zig or clang in the chroot PATH. AML and
notorch are installed system-wide: `/usr/local/bin/{aml,amlc,mhx}`,
`/usr/local/lib/libaml.a` (218652 B), `/usr/local/lib/libnotorch.a`
(139402 B), headers under `/usr/local/include/ariannamethod/`
(`ariannamethod.h`, `gguf.h`, `notorch.h`, `notorch_simd.h`,
`notorch_simd_scalar.h`, `notorch_vision.h`). `aml --version` reports
`aml runner 0.1.0 (libaml linked)`.

**Vendored notorch is behind the canon.** `ariannamethod/notorch.c` is 4739
lines and was last synced to canonical at `e5c66fb` (2026-05-14, `git log --
ariannamethod/notorch.c`). The canonical tree on this node,
`~/arianna/notorch` at `b14d0ba` (`v4.3.0-217-gb14d0ba`, dated 2026-09-12),
carries 9020 lines in `notorch.c`. `diff -u` between the two counts 4495
changed lines in `notorch.c` and 153 in `notorch.h`. No resync was performed
in this entry; it is recorded as open.

The GPU layer is not built on this node. Every CUDA path sits behind
`//go:build linux && cuda` with matching stubs (`gpu_bindings_stub.go`,
`gpu_forward_stub.go`, `gpu_notorch_stub.go`), so the default build takes the
CPU/BLAS path through `cgo_notorch_cpu.go`. Nothing was compiled or run in
this entry.

**No measurement exists yet on ARM.** Every memory and throughput figure in
the repository was taken on cloud or pod hardware: RSS about 2 GB at child
and about 2.5 GB at teen on the Feb-27 Oracle run (`README.md:88-89`), the
4 × 2 GB pod sizing note (`PROJECT_LOG.md:850`), the GPU-vs-CPU step rates
(`runpod/2026-06-02_inc2_gpu/RESULTS.md:33`), the §9 per-stage burst rates
(`docs/molequla_paper.md:549-551`). The first phone-1 numbers will be their
own entry, not a scaling of these.

`reffs/` added to `.gitignore`. It holds reference clones for reading only,
never part of the repository: `arianna.c` (591 commits), `dario` (179
commits), `ariannamethod.ai` (`ded407c`, 2026-08-07). The Dario injection
mechanism (`reffs/dario/README.md`, "Resonance Injection"), the arianna.c
injection contract (`reffs/arianna.c/INJECTION_CONTRACT.md`) and the AML
field physics are the stated inspirations for what comes next on this node.

Open, in the order they were named on this node:

- `mycelium.py` leaves Python. Target language is Go, Zig or AML; decided
  before code, recorded here with the reason.
- Vendored `ariannamethod/notorch.{c,h}` resynced to the canon, with the
  molequla-specific delta named explicitly rather than carried silently.
- First phone-1 run and the first ARM measurement: RSS per stage, tick time,
  burst steps/s, on the big cores, with the command line that produced them.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — first build, first tests, first organism on ARM (Galaxy A56, Exynos 1580)

**Build.** `CGO_ENABLED=1 CGO_CFLAGS="-O3 -march=native -mtune=native -DUSE_BLAS"
CGO_LDFLAGS="-lopenblas -lm -lpthread" go build -a -trimpath -buildvcs=false`
in the chroot: rc=0 in 37 s wall, binary 9403976 B, linked against
`libopenblas.so.0` (`ldd`). `-buildvcs=false` is required here because the
working tree lives on the Termux mount with a foreign owner and Go's VCS
stamping fails on it. gcc emits one warning worth an audit entry:
`ariannamethod/ariannamethod.c:1500` (`am_tape_backward`, the RMSNorm branch)
calls `calloc(n, sizeof(float))` with `n = out_len` whose range the compiler
cannot bound below zero.

**Tests.** `go test -count=1 -v ./...` on the phone: 141 PASS, 0 FAIL, 0 SKIP
(`ok molequla 3.408s`, `ok molequla/tests 0.007s`). README says 132, the
2026-06-29 log entry says 140, the tree says 141.

**One organism, 900 s.** `timeout 900 taskset -c 4-7 molequla_cgo
--organism-id earth --element earth --evolution`, `OPENBLAS_NUM_THREADS=4
GOMAXPROCS=4`, notorch trainer on CPU/BLAS, no overlay, no cross-graze, no
siblings. Stage climb embryo → infant → child inside the first 80 s; warmup
throughput 96-174 steps/s at embryo, 100-105 at infant, 34-41 at child
(`[notorch] warmup complete` lines). The ontogenesis clock then stops:
`ingested=134553` from tick 70 to tick 1290, because `dnaRead` reads siblings
only and there were none. Steady state at child: 20-30 ticks per 15 s
(`debug-onto` every 10 ticks, sampled every 15 s), 1297 DNA writes of
5107 B mean / 5240 B max, 6.62 MB total, none consumed. RSS at child 65-85 MB,
peak `VmHWM` 97776 kB. 0 NaN, 0 panic, exit 124 (the timeout). Child voice
without overlay, verbatim: «When a p of — they body and A lava flowing a
spills metaphysical the lesson of the same and iron ocean is the river
containing.» A single organism measures tick and child RSS; growth and adult
RSS need the colony, which is the next entry.

**Findings for the audit, not repaired here.** README Quick Start
(`README.md:734-739`) passes `--corpus`, `--db`, `--ckpt`; the parser
(`parseCLIArgs`, `molequla.go:6262-6300`) knows none of them and silently
ignores unknown flags, so the documented invocation runs on defaults.
`field_steering`, the mycelium's only channel into the organisms, is read by
`molequla.rs:3265` alone; `molequla.go` has zero references, so the Go colony
of §9 never heard the mycelium.

**The eye, physically.** Termux:API is installed (`com.termux.api`); the
chroot now reaches Termux by ssh key as its own user, `termux-camera-info`
lists the back sensor at 4080×3060, and after `pm grant com.termux.api
android.permission.CAMERA` a first `termux-camera-photo -c 0` returned a
3728275 B JPEG readable from the chroot. `reffs/ocelli` builds here with
OpenBLAS in 6 s; it expects `yent_eye_smolvlm2_lora_v2_f16.gguf` plus its
mmproj, which are not on this node.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the colony of four on ARM, and the eye's first frame

**Colony, 1200 s.** Four organisms, each in its own working directory under
one `dna/output/<element>/` tree as `launcher.sh` lays it out, launched with
`timeout 1200 taskset -c 4-7 molequla_cgo --organism-id <e> --element <e>
--evolution --cross-graze`, no overlay. `capColonyThreads()` read the four
pinned cores and set `OPENBLAS_NUM_THREADS=1` per organism (`[cpu] effective
cores=4`). Sampled every 30 s from `/proc/<pid>/status`.

| organism | stage at exit | `ingested` at exit | bursts (loss) | peak VmHWM |
|---|---|---|---|---|
| earth | 2 child | 170287 | 4 (1.56 2.04 2.37 2.08) | 127408 kB |
| air | 3 adolescent | 244852 | 3 (1.96 2.30 1.70) | 231044 kB |
| water | 3 adolescent | 335336 | 3 (1.38 1.37 1.18) | 240032 kB |
| fire | 3 adolescent | 226526 | 3 (1.27 1.15 1.33) | 234340 kB |

0 NaN, 0 panic, 0 divide events. The three that grew crossed the 200000
adolescent gate on `ingested` and were still inside the post-growth warmup at
exit (`debug-onto tick=10..50` after `ONTOGENESIS: stage 2 -> 3`). Peak RSS
per organism roughly doubles at the growth (embd 64 → 128 plus the
1600-step-class warmup), from about 100 MB at child to 231-240 MB. Lowest
`MemAvailable` on the phone during the run: 1930056 kB, with two Claude
sessions resident beside the colony.

**DNA is a race, not a field.** Bytes consumed over the run: air 3118052 in
18 reads, fire 2614666 in 19, water 1612725 in 10, earth 35734 in 5. `dnaRead`
reads a sibling's `gen_*.txt`, mirrors it to `../dna/seen/<e>/`, appends it to
its own corpus and then removes the file (`molequla.go:5882-5945`, the
`os.Remove` after a successful append). A fragment therefore feeds exactly one
organism, the first to scan the directory that tick; the others see it only
through the `seen/` mirror that cross-graze reads for logit boosts, never in
their corpus. Whoever loses the scan loses growth. Earth lost it here and
stayed a child; in the §9 archive earth is the one adult that never divided.
Audit item, alongside `field_steering`.

**The eye, first frame on this phone.** `reffs/ocelli` as cloned (vendored
notorch of 2026-07-27), f16 weights `yent_eye_smolvlm2_lora_v2_f16.gguf` +
f16 mmproj (both SHA-256-verified against `yent_eye_smolvlm2_lora_v2_gguf_SHA256SUMS.txt`
from neo), the 4080×3060 JPEG from the back camera, `taskset -c 4-7`, prompt
"Describe this image in one sentence.":

> The image is blurry, but it clearly shows a desk with a green object, a
> chair, and a tablecloth with a drawing on it.

That is the frame. Cost: prompt 878 tokens in 9974 ms (88.0 tok/s), generation
29 tokens in 25198 ms (1.2 tok/s), wall 1:33.93, user 232.6 s, sys 118.9 s,
374 % CPU, 888477 minor page faults, peak RSS 1602432 kB (`/usr/bin/time -v`).
Two things in that line are the work: the prompt is 878 tokens because the
full-resolution frame is split into image tiles (neo's 84-token prompts were
small frames), and a third of the CPU time is system time on page faults,
which is `materialize()` (`ocelli.c:114`) expanding f16 weights into an f32
scratch on the prefill path. ocelli's own "peak RSS 2 MB" in its status line
is wrong and is noted as a bug in the reference clone, not repaired here.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the eye quantized with our own quantizer, and 94 s becomes 11 s

**Quantization.** `~/arianna/notorch/gguf_quantize` (canon `5091547`) on the
f16 Yent eye decoder, `taskset -c 4-7`: q4_0 in 3 s (232210464 B), q6_k in 7 s
(417759264 B), q8_0 in 2 s (436805664 B); 226 of 291 tensors converted, norms
and 1-D tensors copied through (`eye_quant.log`). Our q8_0 has exactly the
byte size of the q8_0 that came from neo and a different SHA-256
(`6fb477a5…` vs `365d333e…`); the difference has not been located and is
recorded, not explained. The mmproj stays f16 in every run below.

**Same frame, tiled (878-token prompt), cpu4-7** — text A is «The image is
blurry, but it clearly shows a desk with a green object, a chair, and a
tablecloth with a drawing on it»:

| decoder | gen tok/s | wall | peak RSS (time -v) | text |
|---|---|---|---|---|
| f16 | 1.2 | 1:33.9 | 1602432 kB | A |
| q8_0 (neo) | 5.0 | 1:14.1 | 988416 kB | A verbatim |
| q6_k (ours) | 3.2 | 1:20.9 | 951168 kB | A verbatim |
| q4_0 (ours) | 3.0 | 1:23.8 | 785396 kB | shorter: green object and a drawing, no desk, no chair |

Prefill sits at 88-97 tok/s in every row, so 15-20 s of each wall is prompt
plus generation and the remaining 55-75 s is the vision tower on thirteen
512×512 frames through f32 BLAS. Two expectations carried from neo do not
hold here: Q8 does not expand to 2.2-2.5 GB (988 MB, below f16), and Q4_0
generates slower than Q8_0, because the notorch vendored in ocelli
(2026-07-27) predates the ARM kernels the canon gained in August-September.

**Same frame, one global frame (`SMOLVLM_NOSPLIT=1`, vision.c:71), 84-token
prompt:**

| decoder | gen tok/s | wall | peak RSS | text |
|---|---|---|---|---|
| q6_k (ours) | 4.6 | 0:11.4 | 951296 kB | «A blurry photo shows a desk with a chair, a clock, and a green object, with no clear text or people visible.» |
| q8_0 (neo) | 6.4 | 0:13.5 | 988288 kB | «The image is blurry, but it appears to show a desk with a chair, a laptop, and some papers. The desk is cluttered with objects…» |
| f16 | 1.4 | 0:31.2 | 1602304 kB | «…a desk with a chair, a laptop, and a cup, with the words "the best desk in the world" written in the corner.» |

The frame, by the person who took it: a Mac with a green terminal on a desk,
photographed from below in the dark on a balcony. The laptop and the green
object are real; the tablecloth with a drawing is the tile floor and its
shadows; the inscription in the corner is f16's invention and the only
hallucination in seven runs. One global frame cut the wall from 74-94 s to
11-13 s at equal or better content, so the eye on this phone runs without
tiles. `--siglip-test` reports the tower as L=12 D=768 patches=1024 →
64 visual tokens per frame, with a deterministic checksum.

Next levers, in order: the canon's ARM packed kernels inside ocelli's decoder
(generation), the tower off f32 `materialize` (RSS and the remaining ~5 s),
then a frame-size cap at capture time so the camera never produces a tiled
input in the first place.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the audit: three slices, read-only, every finding at a line

Three independent readers (Opus subagents, no edits, no builds, no runs) each
took one slice of the tree and wrote a report with file:line evidence; the
reports are committed verbatim under `reports/2026-09-13_phone1_audit/`
(A ecology lifecycle, B trainers and vendored libraries, C coherence layer,
mycelium and README). Every P0 below was re-verified by hand on this node
with grep against the cited lines before it was written here.

**P0 — what has to change before the colony runs unattended on a phone.**

- *The population cap counts the wrong organisms.* `AcquireMitosisSlot`
  counts heartbeats younger than 60 s (`molequla.go:5537`); post-growth
  warmup runs inline through `ntWarmupTrain` (`:6494-6496`) with no ticks
  and no heartbeats, minutes long at teen and adult, so a growing organism
  drops out of the count exactly when its memory peaks and the cap admits
  one more.
- *Hibernate frees a counter, not memory.* `MarkHibernating` writes
  `status='sleeping'` (`:5629`) and `main` stays parked on `<-sigCh`
  (`:7105`) with all weights resident. README:665 promises freed resources.
- *There is no byte budget.* `MaxOrganisms` is a head count; nothing in the
  Go tree reads `MemAvailable` or RSS. Sixteen heads at the measured
  231-240 MB per adolescent against the measured 1.93 GB free here is an OOM
  by construction.
- *Corpus and DNA trees grow without bound in `--evolution`.*
  `updateReservoirCorpus` returns early (`:3740`), trimming lives only on
  the REPL path; `dna/output` and `dna/seen` are never pruned, and `ReadDir`
  walks the litter every tick.
- *The DNA field is a race.* Exclusive `os.Remove` after append (`:5932`),
  no per-tick read cap, lockstep ticks with no jitter (`:6696`); the loser
  loses structurally, as earth did in this node's colony run.
- *Frozen parameters break Chuck's slot alignment in notorch itself.*
  `nt_tape_param_frozen` takes no optimizer slot (canon `notorch.c:460-470`,
  by design), but the Chuck loop advances `param_idx` for every param
  without a grad (`notorch.c:2434`, "keep slot alignment") and does not
  check `frozen`. molequla freezes two gates per layer
  (`notorch_trainer.go:333-334`), so from child stage on the RRPRAM factors
  after each frozen gate read another parameter's moments and the last ones
  are never updated. The installed `/usr/local/lib/libnotorch.a` (2026-08-10)
  carries the same loop. This is a notorch fix with a red-hand test, then a
  rebuild here.
- *The trained model is not the model that runs.* Inference adds `wpe`
  (`molequla.go:2851`); the notorch trainer omits it by design
  (`notorch_trainer.go:46`), so positional embeddings stay at their random
  init while the AML trainer does train them (`aml_trainer.go`); `--trainer`
  changes the objective.
- *Cross-graze is discarded once an organism is warm.* The overlay works on
  a copy (`:4613`), disables itself at `mag > 1.0` (`:4650`), after which
  cross-graze writes into `logits.Data` (`:4664`) while sampling reads the
  copy (`:4727`). One line at `:4664`.
- *The mycelium saw zero organisms in every real run.* `method.py:273`
  selects `gamma_direction, gamma_magnitude`, which the Go `organisms`
  table never has (`molequla.go:5490-5497`); the exception is swallowed
  (`method.py:281-282`); without `libaml.so` the steering collapses to a
  constant (`method.py:411-418`). `tests/test_all.sh` builds a Rust-shaped
  mesh.db with random gammas, which is why the suite is green.

**P1, the ones that change what molequla claims about itself.** The default
trainer never touches the delta adapters (`grep Delta notorch_trainer.go` →
0), so "freeze trains deltas only" (README:240) and the immune rollback
restore nothing the burst changed. Chuck moments leak on every growth
(`nt_tape_destroy` frees a loop bounded by a count `nt_tape_clear` already
zeroed; about 51 MB abandoned by adult, estimate from the code). The 8/16/32
progressive warmup is inert from infant on (`notorch_trainer.go:219-222`
pins `seqLen = BlockSize`). The AML trainer has no gradient clipping and no
NaN guard and destroys Chuck state every burst. `relieveOverload` runs before
the burst history is captured, so a loss-path child inherits an empty
biography. `QuickLoss` samples four random documents per call against an
`OverloadLossEps` of 0.05. `CFG.MetaC*` and eight other config fields are
read by nothing. The vendored `ariannamethod/notorch.c` is compiled by
nothing in the Go build (only `ariannamethod/Makefile:34`, into `libaml.so`
for the Python tier); the Go binary links the installed library.

**P2.** Dead code with zero callers: `trainSteps`, `notorchTrainSteps`,
`GenerateSentence` (330 lines), `MetricBoost`. The freeze counter lives in
six places, the architecture in two, the element list in three shapes
(`molequla.go:5834`, `cross_graze.go:60`, `molequla.c:4958`). `sweep.sh`
greps `[spa-gate]` while the code prints `[spa]`, so the SPA column of every
sweep was zero. Tests: 141 = 133 in package main + 8 in `tests/`; README's
132 miscounts `governor_test.go` (six tests). README line numbers drifted by
about 65. `Dockerfile:37` omits the mandatory `-a`.

**Two things the audit cleared.** `ariannamethod.c:1500` is a compiler range
artefact: `out_len` comes from `am_array_new`, which refuses `len <= 0`
(`:1061`). And the vendored notorch has no molequla-specific hunks at all;
a resync re-applies nothing, and on the CPU path buys nothing either, since
the packed kernels and thread pool never touch `nt_seq_linear`.

**Repair order, as decided on this node.** (1) notorch: the frozen-slot loop,
with a test that goes red on a frozen param placed before a trainable one;
rebuild `libnotorch.a` here. (2) `wpe` registered in the notorch trainer.
(3) DNA as a field: per-reader cursor over `seen/`, emitter-side pruning,
tick jitter; the `world` source rides the same mechanism. (4) Governor:
heartbeat during warmup, a byte budget from `MemAvailable`, hibernate that
exits with state saved, replace-instead-of-grow at the cap. (5) Corpus and
DNA growth capped in evolution. (6) The one-line cross-graze target and an
overlay fade without a cliff. (7) Mycelium as a Go witness on the contract
report C names: the Go table's real columns, `dna/seen`, the `am_method_*`
and `am_harmonic_*` symbols already linked into the binary through
`cgo_aml.go` with zero Go callers. (8) README brought to the code.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 1 landed in notorch: a frozen parameter no longer eats a slot

The first item of the repair order is a notorch change, not a molequla one,
and it went through notorch's own discipline: branch `claude/frozen-slot`
from canon `1f5473a`, commit `42ff2f3`, pushed, merge on Oleg's word. The
entry in that repository's NOTORCHLOG carries the mechanism; the short form
is that `nt_tape_entry` now records the optimizer slot it owns (`-1` for
entries registered through `nt_tape_param_frozen()` and for non-params), and
the five loops that address moments — the two diagonal steps,
`nt_tape_chuck_step`, `nt_tape_accum_grads`, `nt_tape_apply_accum` — plus the
CUDA norm batch read that field instead of counting parameters as they pass.
The gate `frozen_param_keeps_chuck_slots` reads red on the unfixed tree
(49 passed, 1 failed: "W2 after a frozen param is updated") and green after
(50/50); `make test` all passed; `test_rrpram_broadcast` 0 fails; strict
`-Werror` clean. All on this phone.

What it means for molequla: `notorch_trainer.go:333-334` freezes two RRPRAM
gates per layer through `nt_tape_param_frozen`, so from the child stage on the
low-rank factors registered after each gate were stepping on a neighbour's
moments and the deepest ones were never stepped. Every §9 number was produced
with that loop; the archive is not rewritten, this entry is the correction.

Installed here: `make install PREFIX=/usr/local` → `/usr/local/lib/libnotorch.a`
2026-09-13 01:30, 204008 B (the previous, 2026-08-10, was 139402 B), header
carries `slot`. Note for the next node: the Makefile's default PREFIX is
`/opt/homebrew`, and a bare `make install` in this chroot created a stray
737 KB tree there; it is not used by anything and is left for Oleg's word.

molequla rebuilt against it: `go build -a -buildvcs=false` rc=0, 9407304 B;
`go test -v ./...` 141 PASS, 0 FAIL. One earth organism for 180 s on cpu4-7:
embryo → infant → child, warmup `avg loss` 4.90 → 3.23 (embryo), 2.21 → 2.15
(infant), 2.41 → 2.04 (child) at 45-46 steps/s, 170 ticks, 175 DNA writes,
0 NaN. Three minutes cannot show the training-quality effect of the fix; that
needs a colony to adult on both libraries, and it is not claimed here. The
organism's stdout carries NUL bytes from tokenizer probes, so it must be read
with `grep -a`, as the §9 climb logs already required.

— Defender (Arianna Method, phone-1)

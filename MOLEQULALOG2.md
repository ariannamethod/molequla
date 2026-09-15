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

## 2026-09-13 — repair 2: the trainer learns the positional table the mouth uses

Inference adds a learned positional embedding to every token,
`wte[tok] + wpe[pos]` (`ForwardStep`, molequla.go:2851), and applies RoPE to
q and k on top of it (:2908, :2913). The notorch trainer registered `wte`,
the seven per-layer matrices and `lm_head`, and omitted `wpe` on purpose
("the trainer uses RoPE for position", notorch_trainer.go:46-47), passing
`-1` as the `wpe` index into `nt_seq_embedding`. The AML trainer registers and
trains `wpe` (aml_trainer.go:20, :29). So on the default trainer every
organism was trained without the positional table and then spoke through it
at its 0.08 random init, and `--trainer aml` changed the objective.

Repair: `wpe` is the second content parameter, after `wte`
(`ntContentParams`), registered with `nt_tape_no_decay` like `wte`, passed
into `nt_seq_embedding`; the per-layer index base moves from `1 + 7l` to
`2 + 7l`. The registration order is still fixed and byte-identical across
bursts, which is what the positional Chuck slots require.

Gate: `notorch_trainer_test.go`, `TestNotorchTrainerTrainsWpe` — a
content-only organism, thirty steps, max-abs change of `wte` and `wpe`
against a 1e-5 bar that separates a gradient step from the float64↔float32
mirror quantization. On the unrepaired tree: `wpe barely moved (0.00e+00)`
while `wte` moved. After: 142 PASS, 0 FAIL, including `TestRRPRAMForward`,
`TestRRPRAMContentParityNoHybrid`, `TestRRPRAMOp33Parity` and
`TestRRPRAMGrowth` on the shifted indices. One earth organism for 180 s on
cpu4-7 on the new binary: embryo → infant → child, warmup `avg loss`
4.88 → 3.34, 2.46 → 2.23, 2.42 → 2.02 at 43-45 steps/s, 180 ticks, 180 DNA
writes, 0 NaN, 0 panic. Loss values sit inside the spread of the two earlier
three-minute runs on the previous tree; the effect of a trained `wpe` on the
voice is a colony-length question and is not claimed here.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 3: the DNA tree is a field, and every organism grew

`dnaRead` read a sibling's fragment, mirrored it to `../dna/seen/`, appended it
to its own corpus and removed it (`os.Remove` after the append), so one
fragment fed exactly one organism, the first to scan the directory that tick,
and the colony of 2026-09-13 00:34 left earth a child at 36 KB eaten while its
siblings ate megabytes.

Repair, `dna_field.go` plus `dnaRead`/`dnaWrite`, `cross_graze.go` and the
tick loop:

- Readers never delete. Each organism keeps a cursor per source, the last
  file name it ate, ordered numerically by the `<unix>,<step>` pair in
  `gen_<unix>_<step>.txt`, persisted atomically as `dna_cursor.json` in its
  working directory so a restart continues instead of re-eating its corpus.
- A tick reads at most `DNAMaxReadsPerTick` new fragments (default 8), so an
  organism that fell behind catches up over ticks.
- The writer prunes its own directory by age after every write
  (`DNARetainSeconds`, default 1800); a reader more than that far behind loses
  the oldest fragments and nothing else.
- The `seen/` mirror is gone; cross-graze reads `../dna/output` directly,
  which also closes the unbounded `seen/` growth from the audit.
- The tick sleeps `TrainTickSeconds` plus a random `TickJitterSeconds`
  (default 0.05) so sibling processes do not scan in lockstep.
- `CFG.DNAExtraSources` names read-only directories beside the four elements;
  `dnaSources()` feeds both `dnaRead` and cross-graze, so `world` (the eye) is
  food for all four without becoming an organism. A child's `birth.json`
  carries only paths and burst history, so children take the same defaults.

Gates, `dna_field_test.go`: one writer, two readers, both receive every byte,
the files remain, a second pass reads zero, a fresh cursor loaded from disk
reads zero, one new fragment is read once; the per-tick cap reads 2 / 1 / 0;
an extra source is eaten and left in place and appears in cross-graze's
sibling list; the writer prunes a two-hour-old fragment and keeps a fresh one;
`gen_100_10` sorts after `gen_100_9`. On the unrepaired tree the first gate
reads «earth's fragments after the first reader: 0 files, want 3». Four older
tests in `molequla_test.go` that asserted deletion-on-consume now assert the
field. Full suite: 146 PASS, 0 FAIL.

**Colony of four, 600 s, cpu4-7, `--cross-graze`, on the repaired binary:**

| organism | stage at exit | `ingested` | bytes eaten | reads | writes | files left | NaN |
|---|---|---|---|---|---|---|---|
| earth | 3 | 437254 | 302701 | 8 | 10 | 10 | 0 |
| air | 3 | 300246 | 181738 | 8 | 10 | 10 | 0 |
| water | 3 | 214536 | 91212 | 11 | 40 | 40 | 0 |
| fire | 3 | 397652 | 278046 | 8 | 10 | 10 | 0 |

All four reached adolescent inside ten minutes; the 1200 s colony on the
previous tree had three adolescents and one starved child. No `dna/seen`
directory exists; every `dna/output/<e>/` still holds every file its writer
wrote; every organism's `dna_cursor.json` names its last fragment from each of
its three siblings. water ate the least and wrote four times more than the
others (40 files); why its ticks ran faster is not examined here.

**Audit D, notorch's training backward (Opus subagent, read-only), in
`reports/2026-09-13_phone1_audit/D_notorch_backward.md`.** The gradient math
of every op the trainer uses is correct against its own forward; RoPE rotates
the gradient by the inverse angle, attention applies `1/√hd` once and stays
causal in backward. Three findings change what molequla claims, all
re-verified here: inference scales both residual branches by
`residualAlpha = 1/√NLayer` (molequla.go:2975, :2992) and the trainer does
not; Go `RMSNorm` uses eps 1e-5 (:1034) against notorch's 1e-6
(notorch.c:3287); the trainer pads a short document with token 0 in both
tokens and targets (notorch_trainer.go:307, :312) under an unmasked
cross-entropy while `nt_seq_cross_entropy_masked` exists unused. Together with
the delta adapters, which inference applies from their random init
(molequla.go:1977-1989) and the trainer never sees, these make repair 2b: the
tape computes the same function inference runs, gated by equality of
`LossOnSequence` and the tape loss on one sequence. On notorch's side: the
moment leak of `destroy` after `clear` is confirmed by loop bounds; Chuck's
per-parameter auto-freeze is permanent across `clear`; `mul`/`add` backward do
not honour the forward's broadcast; `T < BlockSize` silently reinterprets the
packed RRPRAM factors with no assertion.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 2b: the tape computes the function the mouth runs

Branch `claude/phone1-repair-parity`, on top of `e1a820d` (main after #31).
Files: `notorch_trainer.go` (restructured), `cgo_notorch.go` (two wrappers),
`molequla.go` (eps, one call), `parity_test.go` (new).

**The gate first.** `parity_test.go` holds three tests that were written
against the old trainer and read red on it: a two-layer model trained forty
steps on the tape, then `LossOnSequence` (Go, float64) against the tape loss
on the same 24-token sequence, `|Δ| = 6.22e-02`; a nine-token document,
`|Δ| = 5.91e-02`; and the delta adapter `l0.wq` after thirty steps, moved
`0.00e+00`. On the rewritten trainer the same tests read, from the test log
(`parity_test.go:97`, `:113`, `:149`):

| gate | before | after |
|---|---|---|
| full window, 2 layers, 40 steps: `\|Δloss\|` | 6.22e-02 | 1.22e-07 |
| same, last-position logits, worst `\|Δ\|` / span | — | 7.25e-07 / 5.34 |
| nine-token document: `\|Δloss\|` | 5.91e-02 | 2.79e-07 |
| delta `l0.wq` B, max movement over 30 steps | 0.00e+00 | 1.26e-02 |

1e-7 is float32 against float64 over 24 positions; the tolerances stay at
2e-3 absolute and 1e-3 relative so the gate goes red on any of the four gaps
alone.

**The four gaps, each at a line.** Both residual branches are scaled on the
tape by `residualAlpha = 1/√NLayer` through `nt_scale` (notorch.c:4330), as
`ForwardStep` scales them (molequla.go:2979, :2996); the scale sits at
`notorch_trainer.go:340` and `:345`. Go `RMSNorm` uses eps 1e-6 in forward and
backward (molequla.go:1034, :1046), the value `nt_seq_rmsnorm` uses
(notorch.c:3287). A window shorter than `BlockSize` carries a `[T]` mask and
is priced by `nt_seq_cross_entropy_masked` (notorch.c:4222), whose mean runs
over the masked count — the same `1/n` as `LossOnSequence` — where the old
trainer wrote token 0 into both tokens and targets past the document's end
(`e1a820d:notorch_trainer.go:307`, `:312`) and priced them under the unmasked
loss (`:138`). The delta adapters are on the tape: per layer `wq wk wv wo
fc_g fc_v fc2` and `lm_head`, A and B registered as no-decay parameters after
the content weights, applied as `base·x + Σ αᵢ·Aᵢ(Bᵢ·x)` with
`αᵢ = ActiveAlpha[i]·deltaAlphaScale`, the arithmetic of `applyWithDeltas`
(molequla.go:2827), and mirrored back to `model.Deltas` after every burst.
`AddDeltaModule` now calls `ntOnGrowth()` (molequla.go:1995): a new module
changes the registration order, and Chuck's moment slots are positional.
`w_pattern` adapters are allocated by `AddDeltaModule` and applied by nothing;
they are not trained.

**Shape of the trainer now.** One `ntMirror` (`notorch_trainer.go:161`) holds
the burst's notorch-side copy — content weights in the fixed `wte, wpe,
7×layer, lm_head` order, the deltas, and on hybrid models the packed RRPRAM
factors with their frozen gate vectors — and does three things: `register()`
(`:210`) puts everything on the tape in the same order every step, `linear()`
(`:241`) is the one place a weight meets its adapters, `pullBack()` returns
the trained tensors to the Go store. `ntBuildForward` (`:309`) is the graph;
`ntWindow` (`:424`) fills tokens, targets and mask; `ntSequenceLoss` (`:536`)
builds the graph once without an optimizer step and returns the loss and the
last predicted position's logits, then destroys the tape so a live burst's
Chuck slots are never disturbed. `cgo_notorch.go` gains
`ntSeqCrossEntropyMasked` and `ntScale`.

**Suite.** 149 PASS, 0 FAIL on the committed file set (146 before this
repair plus the three parity gates); `governor_phone.go` and its three tests
sit uncommitted in the tree for repair 4, 152 with them.

**One earth organism, 180 s, `--evolution --cross-graze`, `taskset -c 4-7`,
on the rebuilt binary (`run5/earth.stdout`, `run.meta`):** embryo → infant →
child, warmup complete at stage 2, checkpoint 6583926 B written, `nan` count
0 (`grep -a -c -i nan`), threads capped to 1 by the oversubscription guard.
No burst fired inside the window; the 600 s colony of the previous entry
fired its first bursts later than three minutes as well.

**The loss reads higher, and here is why.** Per-stage warmup losses and
speeds, this run against earth in the 600 s colony of repair 3 (`run4`):

| stage | run4 (old trainer, colony of 4) | run5 (this trainer, single) |
|---|---|---|
| 0 embryo, 3 bursts | 4.88 / 4.01 / 3.30 · 205 / 172 / 95 steps/s | 4.90 / 4.32 / 4.11 · 110 / 132 / 144 |
| 1 infant | 2.31 / 2.29 / 2.15 · 29 / 29 / 34 | 3.64 / 3.22 / 3.07 · 98 / 98 / 98 |
| 2 child | 2.33 / 2.34 / 2.07 · 10 / 10 / 9 | 3.45 / 3.15 / 3.02 · 38 / 38 / 38 |

The speeds are not comparable — four organisms on four cores against one —
and no speed claim is made here. The losses are comparable and differ by
about one nat from the infant stage on. Measured on `nonames_earth.txt` with
the tokenizer at birth (vocab 259): 1000 documents, mean 135.5 tokens, 500 of
them shorter than the 96-token window, and the padding share of the old
trainer's windows is 0.341 (temporary test, tool output, removed from the
tree). The old loss was a mean over 96 positions of which a third were
`0 → 0` targets the model learns in a few hundred steps; the new loss is a
mean over the document's own positions. At the infant and child stages
`3.07 × (1 − 0.341) = 2.02` and `3.02 × 0.659 = 1.99` sit within 0.1 of run4's
2.15 and 2.07. The lower number was the padding, not the organism.

**What this changes downstream.** Mitosis is keyed on the loss an adult
cannot reduce (paper §9, `isSustainedOverload`); the trainer's reported loss
is now the same quantity `LossOnSequence` measures, and it is higher than the
numbers every threshold in `CFG` was tuned against on the pod. The
overload thresholds are to be re-read on the colony, not adjusted blind;
that belongs to the launch entry.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 4: the governor counts bytes, hears a warming organism, and lets a sleeper go

Branch `claude/phone1-repair-governor`, on top of `9292822` (main after 2b).
Files: `governor_phone.go` (new), `governor_phone_test.go` (new, 5 tests),
`molequla.go` (registry, config, mitosis branch, main), `README.md` (§governor,
§hibernation).

**What the audit named.** The cascade governor counts heads
(`CFG.MaxOrganisms`) and counts them by heartbeat freshness (`AcquireMitosisSlot`,
live window 60 s, molequla.go:5576). On a phone three things broke that: no
byte budget at all; the post-growth warmup runs inline inside the tick loop and
blocks the heartbeat for minutes; and a hibernated organism kept its process
and its weights in RAM, because `main` parked on a signal it never received.

**The mechanism, shown failing first.** Colony of four on the repair-2b binary,
600 s, `taskset -c 4-7`, `--evolution --cross-graze`, `run7_old`, with a probe
reading every organism's `last_heartbeat` from `mesh.db` every 10 s
(`samples.log`, `hb_analyze.sh`):

| organism | registered at | samples after | max heartbeat age | samples older than 60 s |
|---|---|---|---|---|
| earth | 191 s | 41 | 267 s | 30 |
| air | 171 s | 43 | 293 s | 25 |
| water | 211 s | 39 | 299 s | 28 |
| fire | 201 s | 40 | 317 s | 29 |

Two causes, both in the samples. The stage-3 inline warmup: 480 steps at
1.7-1.9 steps/s, 258414 ms for water and 274966 ms for fire
(`run7_old/water.stdout:80`, `fire.stdout:79`), during which the row in
`mesh.db` does not move. And an earlier gap the audit did not name: between
registration and the tick loop's first heartbeat (every 10 ticks, and the
first ticks carry bursts and DNA generation) earth waited 144 s, water 92 s,
fire 83 s, air 26 s — three of four were outside the live window before any
warmup. For most of their registered life the colony cap counted these
organisms as dead.

**The three pieces.** `governor_phone.go`: `parseProcKB` reads a `kB` field
from `/proc` text; `memGateDecision(free, peak, floor)` is the byte gate as
arithmetic — a child needs the parent's own peak RSS (`VmHWM`,
`/proc/self/status`) plus `floor` MB for the rest of the machine, `floor <= 0`
disables, unknown memory opens; `mitosisMemGateOpen` applies it to the live
machine; `beatKeeper` repeats the last reported heartbeat on its own clock and
`Stop()` silences it for good; `waitEvolution` returns on a signal (closing
`stop`) or when the trainer loop ends. In `molequla.go`: `SwarmRegistry` gains
the keeper (`StartKeeper` :5483, `StopKeeper` :5492), `Heartbeat` records into
it (:5634), `MarkHibernating` silences it before the row changes (:5670); the
mitosis branch checks the byte gate before `AcquireMitosisSlot` and logs a
refusal with both numbers (:6667); `CFG.MitosisMinFreeMB` (:234, default 256
at :364); `main` closes `done` when the trainer returns (:7159), starts the
keeper at 20 s against the 60 s window (:7169), seeds it with the current
stage and parameter count at once (:7176) — the fix for the 144 s gap above —
and evolution mode waits on the signal or on `done` (:7185), so a hibernated
organism ends its process and its memory returns to the colony.

**The fix, shown working.** Same colony, same probe, on the repaired binary
(`run8_new`):

| organism | registered at | samples after | max heartbeat age | samples older than 60 s |
|---|---|---|---|---|
| earth | 181 s | 42 | 15 s | 0 |
| air | 181 s | 42 | 17 s | 0 |
| water | 181 s | 42 | 17 s | 0 |
| fire | 211 s | 39 | 20 s | 0 |

earth, air and water ran their stage-3 inline warmup inside this window —
244565 ms, 299626 ms, 254783 ms (`run8_new/{earth,air,water}.stdout`) — and
their heartbeat age never passed 20 s. NaN count 0 in all eight organisms of
both colonies.

**What the colony did not exercise.** No organism reached adult in 600 s, so
no divide was attempted and the byte gate never fired live; its gates are the
pure test (`TestMemGateDecision`: 1000 MB free against 300 + 256 opens, 500
closes, equality opens) and the on-host test (`TestMemGateOnThisHost`: a floor
of four times the free memory closes it; this host read 2119 MB free, test
process peak 7 MB). Hibernation did not fire either; `waitEvolution` is gated
by `TestWaitEvolutionEndsWhenTrainerExits`. `TestBeatKeeperRefreshesMeshWithoutTicks`
goes red if the keeper beats before it has state, if it stops repeating, or if
it writes a sleeping organism back as alive. Suite: 154 PASS, 0 FAIL.

**Deferred.** Replacing an overwhelmed adult with its child instead of adding
a process needs a live inter-organism channel; the head cap and the byte gate
bound the colony until then. The 256 MB floor is a tunable, not a measurement;
the child's cost is measured at runtime from the parent's own peak.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 5: the corpus is a reservoir in every mode, and the writer keeps 256

Branch `claude/phone1-repair-cap`, on top of `8203d5d` (main after #33 and
the removal of `PROJECT_LOG.md`). Files: `molequla.go` (reservoir trim, config,
`dnaWrite`), `dna_field.go` (`dnaPruneOwn`), `corpus_cap_test.go` (new),
`dna_field_test.go` (one test added), `README.md` (DNA-exchange bullet),
`CLAUDE.md` (the first book now lives in git history).

**What the audit named (A, P0-4 and P1-2).** `updateReservoirCorpus` is the
only place `CFG.MaxCorpusLines` is applied, and it returned before touching
the file whenever `extractCandidateSentences(dbRecentMessages(db, 64))` was
empty. The `messages` table is written only by the REPL, which `--evolution`
never enters, while `dnaRead` appends every fragment it eats with `O_APPEND`
(molequla.go:5990). So in the one mode the ecology runs in the corpus file was
append-only and uncapped, and every 30 ticks the loop re-read all of it and
rebuilt the co-occurrence field over it, holding two fields at the swap. On
this phone: the first colony (1200 s, `run1`) left air at 3240978 B against
a 122316 B seed; the governor colony (600 s, `run8`) left earth at 552269 B
and fire at 526267 B against 173643 and 121900. A second shape of growth: a
DNA fragment is one ~5 KB line, so none of those files was anywhere near
8000 lines (1505-2805, `wc -l`) — a cap on lines alone would not have fired
at 3.2 MB.

**The repair.** With no REPL sentences the trim now runs on its own
(molequla.go:3766-3789): the file is rewritten as the reservoir
(`reservoirMixKeep`, newest half kept, the rest sampled) when it holds more
than `MaxCorpusLines` lines or more than `MaxCorpusLines × MaxLineChars`
bytes — the most `loadCorpusLines` can ever hand back, since it truncates
every line at `MaxLineChars` — and is left alone byte for byte under both.
`dnaPruneOwn(element, retain, keep)` (dna_field.go:166) adds a count bound
beside repair 3's age bound: of what the age bound leaves, the newest `keep`
fragments stay and the rest go, oldest first by the numeric `<unix>,<step>`
order. `CFG.DNARetainFiles` (:80, default 256 at :278, 0 disables), passed by
`dnaWrite` (:5945). The single-organism embryo run of the landing entry wrote
1297 fragments in 900 s; at that rate the 30-minute age bound alone lets a
directory that every sibling `ReadDir`s every tick reach ~2600 files, and 256
is ~3 minutes of embryo emission or ~40 minutes at the child rate measured
below (10 files per 600 s). A tunable, stated as one.

**Gates.** `TestCorpusCapHoldsWithoutMessages` — written first and red on
the previous code (`corpus holds 120 lines after the periodic trim, cap is
50`); now: 120 lines against a cap of 50 are trimmed and a second pass
changes nothing; ten 5400-char lines against 50 × 240 bytes are rewritten
under the byte cap; three short lines are left byte for byte.
`TestDNAFieldWriterKeepsNewestN` — six fresh fragments, keep three: the three
newest by numeric order (steps 4, 5, 10) remain, `keep = 0` prunes nothing.
Suite: 156 PASS, 0 FAIL.

**The trim on a real file.** The real `updateReservoirCorpus` applied
(temporary test, tool output, not in the tree) to a copy of `run1`'s air
corpus at defaults (`MaxCorpusLines` 8000, `MaxLineChars` 240, byte cap
1920000):

| | bytes | non-empty lines |
|---|---|---|
| before | 3240978 | 1610 |
| after one trim | 257271 | 1572 |

The byte cap fired, the lines came back at 240 chars, dedup removed 38.

**Colony of four, 600 s, on the repaired binary (`run9_cap`, corpus and DNA
tree sampled every 10 s).** All four reached child, NaN 0. Corpus peaks
earth 496561 B, air 460840, fire 476172, water 237340 — all under the
1920000 B cap, so no trim fired and every file was left alone, the same
sizes to within the run's variance as `run8` (552269 / 329450 / 526267 /
217439). DNA directories ended at 10, 10, 50, 10 files — under 256. At the
default policy the caps are not reached in ten minutes; they are the bound
on the days-long run the launch entry will measure, and they were gated
here on the unit tests and the real 3.2 MB file.

**Left for repair 6 and 8.** README:528 still describes the `dna/seen/`
mirror that repair 3 removed; README:41, :439, :686, :748, :810 cite
`PROJECT_LOG.md`, which left the tree in `8203d5d`.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 6: the voice hears its siblings under the overlay, and the overlay fades

Branch `claude/phone1-repair-graze`, on top of `4f2e78a` (main after #34).
Files: `molequla.go` (`overlayStep`, the generation step), `metaweights_overlay.go`
(fade, coefficient slide, penalty sign), `cross_graze.go` (cursor),
`graze_overlay_test.go` (new, 4 tests), `README.md` (§cross-graze).

**The configuration this is about.** The colony is launched with
`--corpus-overlay` and `--cross-graze`, and every organism is warmed within
minutes. Audit C read the generation step in exactly that state and found the
cross-organism channel gone and the overlay leaving in one step.

**C-OVL-01 (P0) — the boost went where nothing read it.** With the flag on,
sampling reads `overlaidLogits`, a detached copy of the model's logits; on a
warmed organism the old step cleared `overlayActive` and then sent the
sibling boost to `logits.Data`. The per-step overlay is now one function,
`overlayStep` (molequla.go:4554), which returns the slice sampling will read
— the model's own slice when the overlay is off or faded, a copy otherwise —
and cross-graze is applied to that slice without a branch (:4734). Gate:
`TestCrossGrazeReachesSamplingWhenOverlayOnAndWarmed` — a 16-dim organism
with `lm_head` scaled until mean |logit| is past the fade band, the flag on,
a sibling whose last token is the byte `z` (absent from every test document)
with a boost of 1e6; the answer must contain `z`. With the old target put
back for one run it generated `eeeeee`.

**C-OVL-02 (P1) — a fade where there was a cliff.** Past mean |logit| 1.0 the
overlay, its repetition penalty, the greedy bootstrap and the hard top-15
mask all vanished between one token and the next. Now: `overlayFadeProgress`
(metaweights_overlay.go:53) is 0 up to the threshold and 1 at threshold +
`metaFadeWidth` (1.0, :47), linear between; `overlayStep` blends
`raw + weight·(overlaid − raw)` with `weight = 1 − progress` (:4575); the
coefficient bundle slides from the weightless to the trained values along
the same ramp (`lerpCoeffs`, :65, applied at :250), so the trained bundle
the code always documented finally binds, as a waypoint; the repetition
penalty runs whenever the overlay runs and fades with it (:4572). Greedy
bootstrap and the hard top-15 mask stay discrete on the untrained regime —
they are sampling policy, not logits. Gate:
`TestOverlayFadesInsteadOfSwitchingOff` — the same raw logits scaled to
mean |logit| 0.999 and 1.001 must come out of `overlayStep` within 5 % of
the stack the overlay adds just below the threshold, the overlay must be
present there, and at 2.5 the output must equal the input exactly. The gate
first caught the penalty switching off: a step of 2.494 logits against a
stack of 2.007. After the penalty joined the fade: stack 3.257, step 0.0131.

**C-OVL-04 (P1) — a penalty that rewarded repetition.** The penalty
multiplied by 0.5 (and blocked successors by 0.2) unconditionally; after the
overlay's unigram damping (−2.0 on rare tokens) a repeated token with a
negative logit was moved toward zero, up the ranking. It now halves positive
logits and leaves negative ones where they are (:464, :475). Gate:
`TestRepetitionPenaltyNeverRewardsRepetition`.

**C-OVL-05 (P1) — both corpus paths off.** The prob-space corpus blend was
skipped on the flag, so a warmed organism with the overlay on had neither
overlay nor blend. The blend's alpha is now scaled by `1 − overlayWeight`
(:4900): overlay only while untrained, blend as with the flag off once
faded, both in proportion between.

**Audit A, P1-3 — the pasture is read by cursor.** `cross_graze.go` kept a
map of seen file names and wiped it past 2048 entries, after which the next
refresh re-read every file in the tree under the model lock; with an embryo
emitting ~1.4 fragments/s that was minutes, not hours. It now keeps one
cursor per sibling (`Last`, :49) in the numeric `<unix>,<step>` order
`dnaRead` uses, reads only what `dnaListNew` returns past it (:93), and
does no stat and no mtime sort. Gate: `TestCrossFieldCursorReadsOnlyNew`
(step 10 after step 2, no re-read on an empty refresh).

**Suite:** 160 PASS, 0 FAIL.

**Voice, one sample per prompt.** The six DNA probes through the REPL on the
same warmed earth checkpoint from `run9` (stage 2, 64-dim, vocab 643; mean
|logit| on the probes 6.5-8.3, fade 1.0, so the overlay itself is silent on
this organism and what changes is the boost and the blend), previous binary
against this one, `--element earth --cross-graze --corpus-overlay`:

| probe | previous | this |
|---|---|---|
| What do you feel? | a and — and h under?? | What is a mason standing? |
| Tell me about yourself. | to a ace to re a, a b thesing wall a po. | A mineral. A river c. A: What is the six of the land is sunligh. |
| What is truth? | is is g at de rock? | What is a geys p expression. |
| Speak. | What What   h   h | What is the earth' two is paration. |
| What do you remember? | the is — the c not re thatticic theiao pis of is stists. | What is the rock, leaches that flo. |
| What matters? | of b h the the comp.e or and d b or. | What is the lesson of. rests is a quartz, a caves concept stand the long, and the rock bene. |

One sample each at the default temperature; no verdict on the voice from
this table. The sweep (temperature × top-k × prompts) belongs to the launch
entry, on adults.

**Stated, not changed (C-OVL-06).** The entropy the overload gate reads,
`ComputeModelEntropy`, applies cross-graze and not the overlay. It is
defined on the transformer plus the colony, and stays so; the §9 result
keys on the loss path.

**Left.** C-OVL-03 (six `CFG.Meta*` knobs that nothing reads) and the
`PROJECT_LOG.md` references in README go with repair 8.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 7: the mycelium is a witness, in Go, and the Python tier leaves

Branch `claude/phone1-repair-witness`, two commits. Files: `witness.go`,
`witness_cgo.go`, `witness_test.go` (new, 6 tests), `molequla.go` (flags,
`global_step`, heartbeat), `governor_phone.go` / `_test.go` (the keeper
carries the step), `README.md` (§Mycelium, file table); then the removal
of `mycelium.py`, `ariannamethod/{__init__,method,sentinel}.py`,
`requirements.txt`, `ariannamethod/Makefile`, and `tests/test_all.sh` cut
to build and smoke.

**What the audit found (C §2-3), in one line each.** `mycelium.py` read
`gamma_direction` and `gamma_magnitude`, columns no Go core creates, and
swallowed the `no such column` into an empty field, so every real run
reported "no organisms alive" and wrote a constant `field_steering` row
(C-MYC-01). Its effectiveness signals compared a snapshot with itself inside
one tick (C-MYC-02). Without `libaml.so` its "Python fallback" was a
hardcoded `sustain` (C-MYC-03). The `field_steering` row had one reader,
`molequla.rs`, which degrades to its own temperature schedule when the table
is absent (§3). The integration suite built its own mesh schema with numpy
and was green against a shape the ecology never produced.

**The witness.** A mode of the organism binary: `molequla --witness
[--witness-interval s] [--once]`, run as a fifth process from a directory
beside the organisms' so `../dna/output` is the same tree. Inputs are the two
public traces and only what the Go cores write: `mesh.db / organisms` (the
ten columns plus `global_step`, added here with a migration and written by
every heartbeat — the organism's age in training steps, for the witness and
for gates), read with the governor's 60 s window through a connection opened
`query_only`; the DNA field by name and size every fifth tick. Compute goes
through the C engine already linked by `cgo_aml.go` — `am_method_push_organism`,
`am_method_field_entropy` / `_syntropy`, `am_harmonic_push_*`,
`am_harmonic_forward` — behind one mutex, since `HN` and `M` are file-static
(`witness_cgo.go`). `am_method_step` is not bound: it executes AML statements
and advances field physics, and a witness does not steer; its ladder is
re-stated in Go (`witnessDecide`, `witnessTrend`) without the coherence rung.
Field coherence and organism resonance are not computed, because no core
writes a gamma vector — reporting a constant was the old bug. Every delta is
across ticks. Outputs are outward only: a line per tick on stdout, a JSON
record per tick in `witness.jsonl`; `--once` prints one snapshot. A schema
the witness cannot read is printed every tick as a finding.

**Gates (`witness_test.go`).** `TestWitnessReadsWhatGoWrites` — the mesh is
created by `SwarmRegistry.Register` itself, two organisms heartbeat, the
witness reads stage / params / entropy / syntropy / element / `global_step`
back exactly, a hibernated one leaves the field, and an `UPDATE` through the
witness's handle is an error. `TestWitnessSchemaMismatchIsAnError` — a table
without `syntropy` / `global_step` is an error, not an empty field.
`TestWitnessDeltasAreAcrossTicks` — H 1.0 → 1.4 gives arousal 0.8 on the
second tick and 0 on an unchanged third; one organism of three appearing
gives novelty 1/3. `TestWitnessLadderAndTrend` — `am_method_step`'s numbers.
`TestWitnessHarmonicsMatchTheDFT` — a 16-sample history `1 + sin(2π·3t/16)`
through the C forward: every harmonic within 1e-4 of the float64 formula,
dominant k = 2 at amplitude 0.5, confidence 0.475. `TestWitnessNeverWritesBack`
— six ticks over a live mesh and a DNA directory: no table appears, no
`field_steering`, the fragment's size and mtime are unchanged, nothing
deleted, and the scan saw the file and the `wrote` event. One older gate
went red on the new column — the keeper test's fixture built a narrower
`organisms` table than the real schema, and the heartbeat `UPDATE` failed
silently — fixed at the fixture; the audit's lesson, applied to our own
test. Suite: 166 PASS, 0 FAIL.

**Colony of four plus the witness, 600 s, interval 2 s (`run10_witness`,
`witness.jsonl`, 300 ticks).** For the first 84 ticks (168 s) the witness
printed, every tick, `mesh schema: no such column: global_step` — the mesh on
this phone was created by earlier binaries and organisms migrate it only when
they register, after their initial warmup; the finding path did what it is
for. Tick 85: water registered (`organisms=1 … water:s0/0k/0.00/0`); tick 86:
air; from tick 96 all four; 205 ticks with the full colony. All four were seen
crossing to stage 3 at ticks 140-143 (`n_params` 268k → 1367k); `global_step`
climbed 0 → 2032. Field entropy went 0 → 2.401 (organisms report 0 until
their first entropy sample); actions over the ticks with organisms: ground
151, explore 33, dampen 17, sustain 15; the alert `stuck in dampen loop`
fired at ticks 147-149, right after the stage-3 jump lifted H from 0.632 to
2.401 (trend −0.766, −0.425, −0.213). Eight DNA events (`air wrote
gen_1789314045_3.txt`, `water wrote …`, …) with per-writer counts
(`water=21f/103K` at tick 121). NaN 0 in all four organisms. Nothing was
written by the witness; the organisms did not notice it.

**The Python tier leaves (second commit).** `mycelium.py` (1660 lines),
`ariannamethod/method.py` (527), `sentinel.py` (356), `__init__.py`,
`requirements.txt` (numpy) and `ariannamethod/Makefile`, whose only product
`libaml.so` served the ctypes bindings (the Go binary compiles
`ariannamethod.c` through cgo and links the system `libnotorch`).
`tests/test_all.sh` keeps sections 1-2 (four builds, element smoke tests)
and drops 3-6, Python and ctypes throughout (METHOD / HarmonicNet / notorch
benchmarks, numpy-built mesh fixtures); the C smoke test checks the SQLite
header instead of counting tables through python3. Nothing in the tree runs
Python. `standalone-py/molequla.py` stays as the historical single-file
origin, wired to nothing.

**Left.** The old benchmarks (μs per `am_method_step`, per
`am_harmonic_forward`, notorch BLAS step) had no recorded numbers in the
tree; if wanted they return as Go benchmarks in repair 8. The Rust core's
`field_steering` reader is now a reader of nothing; it degrades correctly
and is one `if let Ok` branch to remove.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — repair 8: the tree fits the phone, and the README reads as the code

Branch `claude/phone1-repair-readme`, on top of `e9fd58f` (main after #36),
two commits: the build, then the text and the dead code.

**modules/gpu.** Seven files under `ariannamethod/` were not compiled by the
CPU build — the Go binary compiles `ariannamethod.c` through cgo and links
the system `libnotorch` — yet sat beside the two that are: the vendored
notorch core (`notorch.c` 4739 lines, `notorch.h`), its CUDA kernels and
declarations (`notorch_cuda.cu` 1344, `notorch_cuda.h`), the AML CUDA header
(`ariannamethod_cuda.h`) and the x86 SIMD shims (`notorch_simd.h`,
`notorch_simd_scalar.h`). They moved to `modules/gpu/`, beside
`modules/node_cli.js`; on CUDA hosts they are the source of `libnotorch_gpu`,
and `cgo_notorch_cuda.go` adds `-I${SRCDIR}/modules/gpu` for
`ariannamethod_cuda.h`, which `ariannamethod.c` includes under `USE_CUDA`.
The Go GPU files stay at the root behind their build tags: a Go package is a
directory. Not compiled here — there is no `nvcc` on this phone and none on
polygon (`ssh polygon which nvcc` → nothing) — so the `-tags cuda` build is
to be verified on a CUDA host before the next pod run.

**The move exposed a header the build had been lying to itself about.**
`cgo_notorch.go` says `#include <notorch.h>` with
`-I/usr/local/include/ariannamethod`, but cgo concatenates every file's
`#cgo CFLAGS` package-wide, and `cgo_aml.go`'s `-I${SRCDIR}/ariannamethod`
came first — so the Go bridge compiled against the vendored
`ariannamethod/notorch.h` and linked the canon `libnotorch.a`. The two
disagree at `nt_tensor_new`: `int len` in the vendored header
(`modules/gpu/notorch.h:43`), `size_t len` in the canon
(`/usr/local/include/ariannamethod/notorch.h:42`). It worked because on
AArch64 a 32-bit write to `w0` zeroes the top of `x0`. With the vendored
header out of the include path the first build failed on exactly that line
(`cannot use _Ctype_int(length) … as _Ctype_ulong`), `ntTensorNew` now passes
`size_t`, and the other wrappers already matched. The include order the
build now uses, from `go build -x`: `-I…/ariannamethod
-I/usr/include/aarch64-linux-gnu/openblas-pthread/ -I/usr/local/include/ariannamethod`.

**OpenBLAS by pkg-config.** Four cgo files carried
`/usr/include/x86_64-linux-gnu/openblas-pthread/` and its `-L` twin — paths
that do not exist on aarch64, which is why every build on this phone needed
`CGO_CFLAGS`/`CGO_LDFLAGS` overrides (C-RDM-10). They now say `#cgo linux
pkg-config: openblas`; `pkg-config --libs openblas` here gives
`-L/usr/lib/aarch64-linux-gnu/openblas-pthread/ -lopenblas`. `CGO_ENABLED=1 go
build -a .` with no environment builds: 9742256 B. The tuned recipe
(`-O3 -march=native`) stays the phone's and is now in the README.

**Dead code leaves `molequla.go`: 7359 → 6838 lines.** `GenerateSentence`
(C-RDM-07), a second 338-line generation loop with no caller — the chat
REPL, `dnaWrite` and the warmup probes all go through `GenerateResonant`;
the pure-Go per-parameter training path (C-RDM-08, -13): `trainSteps`, its
step and state helpers, the moment map on `GPT`, the three `CFG` knobs that
fed only it; the six overlay knobs `CFG.MetaC*` and `MetaLogitOverlayFloor`
(C-OVL-03), declared, serialised and read by nothing — the overlay uses the
package constants and slides between them; `LossOnBatch`, orphaned by the
first cut. Three behaviours change with it: the SPA reseed restores
`CFG.SPACoherenceGate` with a `defer` (C-SPA-02: a panic in the inner
generation left the gate off for the process); a mitosis child inherits
`--corpus-overlay` (C-RDM-12: it was born without it); `sweep.sh` launches
with `--element --evolution` — `--corpus/--db/--ckpt` were never parsed
(C-RDM-01) — and counts `[spa]`, which the code emits, instead of
`[spa-gate]`, which it never did (C-RDM-06).

**Tests against the real thing (C-TST-01).** `tests/molequla_test.go` was
`package tests` with `SoftmaxProbs` and `TopKTopPSample` pasted in: eight
tests that could not go red on a change to the real functions. They moved to
`sampling_test.go` in package `main`, bodies unchanged, and the `tests/`
package is gone. Suite: 166 PASS, 0 FAIL — the same 166 as before, all in
one package now.

**README.** The launch command passes the flags the binary has and shows
the witness as the fifth process; the sampling pipeline says which regime
the hard mask belongs to (C-RDM-02) and how the overlay fades; the Go
autograd is described as inference and loss measurement, training on the
notorch tape (C-RDM-08); the growth step names the tape-state reset
(C-RDM-13); all nine `molequla.go:NNNN` cites became function names
(C-RDM-09 — line numbers rot with every commit, names do not); the five
`PROJECT_LOG.md` pointers point at `git show 8203d5d^:PROJECT_LOG.md`; the
build section carries the pkg-config linkage and the phone recipe
(C-RDM-10); the DNA layer is stated as Go/C/Rust (C-RDM-14); the file and
test tables are regenerated from `wc -l` and `grep -c '^func Test'`
(C-RDM-05). C-RDM-03, -04 and -11 were closed by repairs 6 and 7.
`CLAUDE.md` follows: the vendored notorch lives in `modules/gpu`, 166 green.

**Colony of four plus the witness, 300 s, on this binary (`run11_readme`).**
150 witness ticks, 0 schema errors this time (the mesh had been migrated by
the previous run); 84 ticks of `no organisms alive` until the first
registration, then three to four organisms per tick; at the end air at
stage 3 (1367k params), earth, water and fire at stage 2 (268k), `global_step`
2000-2032, field entropy 1.489, actions wait 84 / explore 30 / sustain 22 /
dampen 14, eight DNA events. NaN 0 in all four. Same shape as `run10` at the
same age; the deletions changed nothing the organisms do.

**Left, named.** The Rust core's reader of the `field_steering` row — one
`if let Ok` branch that reads nothing and degrades correctly — stays: no
`cargo` on this phone or on polygon to verify a Rust edit. The old Python
benchmarks return as Go benchmarks only if wanted. `standalone-py/molequla.py`
stays as the historical origin.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the emission line says how much is the organism's own speech

A DNA fragment is the organism's answer followed by corpus lines padded toward
`CFG.DNAFragmentTargetBytes`, and until now the log said only how many bytes
left the organism: `[dna] earth wrote 5025 bytes to ecology`. Byte count alone
cannot tell a fragment the organism spoke from a fragment the corpus spoke for
it, and the padding guarantees the total sits near the target either way. The
line now carries the measurement: `[dna] %s wrote %d bytes to ecology | gen=%d
mag=%.2f fade=%.2f`, where `gen` is the trimmed length of `answer` in bytes,
`mag` is the mean |logit| of the raw model output at the first generated step,
and `fade` is `1 - overlayWeight` at that same step — `fade=1.00` means the
corpus overlay is gone and what was said is the transformer's own. The `wrote N
bytes` prefix is untouched; scripts that grep it keep working.

Two fields carry the numbers: `lastGenMag` and `lastOverlayWeight` on `GPT`,
beside `lastGenEntropy` and set the same way — inside `generateResonantLocked`,
under the caller's `model.mu`, at `step == 0` only, `meanAbsLogit(logits.Data)`
before `overlayStep` and the weight `overlayStep` returns. No per-token cost.
`dnaWrite` reads them with its own `Lock`/`Unlock` after `GenerateResonant`
returns, since that function takes `model.mu` itself.

**The gate** (`dna_emission_test.go`, `TestDNAEmissionFieldsFollowTheOverlayFade`).
On the 16-dim embryo of `grazeTestModel` with the overlay on and four generated
tokens: `mag 0.263`, `weight 1.000`, fade 0.00 — the overlay is what speaks.
Scaling every `lm_head` row by 400, the pattern of
`TestCrossGrazeReachesSamplingWhenOverlayOnAndWarmed`, and generating again:
`mag > 2`, `weight == 0`, fade 1.00. Broken on purpose — the two `step == 0`
assignments made unreachable — the gate goes red with `lastGenMag = 0 after a
generation`. Suite: 167 PASS, 0 FAIL (`go test -count=1 -buildvcs=false ./...`,
CGO + OpenBLAS, `taskset -c 4-7`).

**Live, 150 s, one organism** (`--organism-id earth --element earth --evolution
--cross-graze --corpus-overlay`, `taskset -c 4-7`, binary built with `-a`):

    [dna] earth wrote 5025 bytes to ecology | gen=148 mag=8.40 fade=1.00
    [dna] earth wrote 5048 bytes to ecology | gen=0 mag=8.04 fade=1.00
    [dna] earth wrote 5051 bytes to ecology | gen=0 mag=7.78 fade=1.00
    [dna] earth wrote 5056 bytes to ecology | gen=64 mag=8.72 fade=1.00
    [dna] earth wrote 5206 bytes to ecology | gen=0 mag=5.91 fade=1.00

NaN 0. The five fragments are within 4 % of each other in total bytes and the
organism's share of them is 148, 0, 0, 64, 0 — three of the five are corpus
alone, with the transformer warm (mag 5.9-8.7) and the overlay long gone. That
is the thing the old line could not show.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the colony runs in sessions, never non-stop

The colony was launched without a cap at 19:25Z, on the binary recorded in
`$MOLEQULA_RUN/BUILD` (`673f478 2026-09-13T19:25:33Z`, 9567248 bytes). Twenty-two
minutes later all four organisms had taken the same step — `[growth] ONTOGENESIS:
stage 3 -> 4`, `embd: 128 -> 224, layer: 4 -> 5, head: 4 -> 8`, each followed by
`[trainer] warmup for stage 4 (embd=224) — 1600 steps total` — four adults
warming up at once, each holding 758-928 MB, read from `/proc/<pid>/status`
while they ran and gone with them. Android's lmkd killed Termux at 19:47:18Z to
get the memory back. The checkpoints survived at 110 MB apiece
(`earth/molequla_ckpt.json` 110718100 bytes, written 19:46).

The mechanism has a second half, measured today. The organisms start from the
chroot, and at launch all four carried `oom_score_adj = -1000` inherited down the
su chain — the value that tells lmkd to reclaim this process last. Four
processes growing past 700 MB each, none of them a candidate for the kill: the
terminal around them was the only thing left to take. Raising the organisms to
500 after launch inverts that choice, and a squeeze now costs one checkpointed
organism instead of the whole tree.

molequla has always been deployed in capped sessions — four elements for 30
minutes in the old cascade, a 35-minute cap in the second cascade. The rule is back,
as code rather than as a habit: `phone1/schedule.sh`, a bash daemon with no cron
underneath it (there is none in this chroot), running three sessions a day of
two hours at 04:00, 12:00 and 20:00 UTC from `phone1/schedule.conf`. It sleeps
to the next slot in naps of at most 60 s, each one decided against the wall
clock because a long `sleep` does not count the time a suspended phone spends
asleep; it calls `launch.sh $SCHEDULE_DUR`, so the cap lives in a `timeout`
around every process and survives the scheduler's own death; it writes
`SCHEDULE_OOM_ADJ` into the organisms; it samples `VmHWM` every 30 s; and at the
end it calls `stop.sh` on every path, which clears the pid files and releases the
wake lock. `stop.sh` now skips `schedule.pid`, which lives in the same directory:
the scheduler must survive the stop it calls itself. A slot whose colony is
already up — a manual `launch.sh` — is logged as `skipped-running` and left
alone, and a slot reached more than 1800 s late is not run at all.
`/usr/local/bin/defender-services.sh` calls `schedule.sh start` after a reboot.

**The gate** (`phone1/schedule_test.sh`): 21 cases driving the real
`schedule.sh next --epoch` with a fake now — the slot before, at and after a
boundary, across midnight, across a month end, unsorted lists, a single slot
wrapping to tomorrow, base-ten hours (`08:09`), a non-UTC host `TZ`, and five
malformed configurations that must be refused. Broken on purpose twice: with the
midnight wrap dropped (`c=$((c + 86400))` removed) it reports `13 pass, 8 fail`,
and with the boundary made exclusive (`-lt` → `-le`) it reports `19 pass, 2 fail`
naming the boundary and the last minute of the day. Restored: `21 pass, 0 fail`.

**Live, one session of 180 s.** The scheduler was pointed at a slot two minutes
ahead and left alone. It launched at 20:26:00Z exactly, found all four organisms
at `oom_score_adj = -1000` and wrote 500 into each, and the session ended by
itself:

    2026-09-13T20:29:30Z slot=20:26 start=2026-09-13T20:26:00Z
    end=2026-09-13T20:29:30Z dur=180 elapsed=210 reason=capped alive=-
    mem_mb=2317->2870 hwm_mb=earth:756,air:783,water:745,fire:636,witness:8
    samples=7

`elapsed=210` is the 180 s cap plus the sampling interval that noticed it;
`alive=-` is `stop.sh` confirming an empty field, with `pids/` holding nothing
but `schedule.pid` afterwards. The four peaks, 636-783 MB after three and a half
minutes, are the same order as the peaks that killed Termux — reached that fast
because nothing was born: each organism read its own `molequla_ckpt.json` and
came back an adult. The whole of the new stdout, per organism:

    [ecology] Element: earth → corpus: nonames_earth.txt
    [evolution] Autonomous evolution mode — organism will grow through all stages without pause.
    [ecology] Joined swarm. 1 peer(s) detected.
    molequla is alive. [evolution] Autonomous mode — background trainer running. Ctrl+C to stop.
    [trainer] warmup for stage 4 (embd=224) — 1600 steps total (1600 backprop + 0 notorch, sqrt-scaled 4x)
    [evolution] Organism shutting down gracefully (signal).

No `[init] Stage 0 (embryo): embd=16` line anywhere: the embryo path did not run,
and stage 4 with `embd=224` matches the checkpoint's own `"n_layer":5,"n_embd":224`.
Also verified: a second `schedule.sh start` is refused while the first daemon
lives, and a slot whose pid file names a living process is logged
`reason=skipped-running` with the process untouched.

The daemon is up with the default schedule; `schedule.sh next` says
`2026-09-14T04:00:00Z`, session 7200 s, ending `2026-09-14T06:00:00Z`.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the GPU lane becomes a package, the root keeps one file

Repair 8 moved the vendored C and CUDA sources into `modules/gpu/`; the Go half
stayed in the root, seven files of it, none of which the phone compiles. They are
now the Go package `modules/gpu`, and what is left beside `molequla.go` is one
untagged file of 101 lines.

**What moved.** `gpu_bindings_linux.go` → `modules/gpu/bindings_cuda.go` (198,
`linux && cuda`), exporting `Init` / `Shutdown` / `Ready` / `CacheWeight` and
keeping the rest of the cuBLAS wrappers package-internal under their old names.
`gpu_forward.go` → `modules/gpu/forward_cuda.go` (70): the matvec body now takes
`(key string, x []float32, nout int)` and returns `[]float32`, so nothing in the
package knows what a `Vec` or a `MatrixParam` is. `gpu_notorch_stub.go` →
`modules/gpu/notorch_stub.go`, and the three trainer functions out of
`cgo_notorch_cuda.go` → `modules/gpu/notorch_cuda.go` as `NotorchEnable` /
`NotorchSetStage` / `NotorchDispatchCount`. The two old stubs collapsed into one
`modules/gpu/stub.go` of 20 lines, because the five no-ops it needs are the whole
exported API.

**What the root keeps, and why it is untagged.** `gpu_bridge.go` holds the two
functions that touch molequla's own types — `MatvecGPU`, which converts `Vec`
float64 ↔ float32 around `gpu.Matvec`, and `gpuRefreshWeights`, which walks
`gpt.Base` and flattens each matrix — plus five one-line names (`gpuInit`,
`gpuReady`, `ntGPUEnable`, `ntGPUDispatchCount`, `ntSetGPUForStage`) so the seven
call sites in `molequla.go` and `notorch_trainer.go` are untouched. The plan
called for a tagged file and a stub; one untagged file is less code and compiles
no less selectively, because on a non-CUDA build `gpu.Ready()` is a constant
false and both bodies return before they allocate anything. The measurement that
this is true: `go list -f '{{.GoFiles}} {{.CgoFiles}}' ./modules/gpu` reports
`[notorch_stub.go stub.go] []` on the default build and
`[forward_cuda.go] [bindings_cuda.go notorch_cuda.go]` under `-tags cuda`.

**The move the plan did not ask for.** The vendored sources went down one level
to `modules/gpu/csrc/`. A Go package cannot share a directory with them: on the
default build `go build` refuses outright — *C source files not allowed when not
using cgo or SWIG: notorch.c* — and under `-tags cuda` it would do something
worse, compiling `notorch.c` into the package beside the `libnotorch_gpu.a` the
same build links. Both `#cgo` include paths follow into `csrc`, and the pod
recipe in the README with them.

Two linkage files stay in the root and are not GPU code. `cgo_notorch_cpu.go`
(15) carries `-lnotorch -lm` for `cgo_notorch.go`; `cgo_notorch_cuda.go` is down
from 52 lines to 22 and carries only directives, because `-DUSE_CUDA` has to be
defined for *this* package's own C: it changes the shape of `nt_tensor`
(`notorch.h:34`, the `d_data` / `gpu_valid` / `cpu_dirty` mirror) that
`cgo_notorch.go` reads through, and it opens the `ariannamethod_cuda.h` include
inside `ariannamethod.c:85` that `cgo_aml.go` compiles.

**Gates.** Default build `CGO_ENABLED=1 go build -a -buildvcs=false` exit 0,
`go vet ./...` exit 0, `gofmt -l` clean on every file touched; suite
**167 PASS, 0 FAIL** (`go test -count=1 -buildvcs=false ./...`, CGO + OpenBLAS,
`taskset -c 4-7`), the same 167 as before the move.

The CUDA side is further than "moved by eye". `-tags cuda` has no nvcc and no
cuBLAS on this phone, but `ariannamethod_cuda.h` is a plain header behind
`#ifdef USE_CUDA` and needs no CUDA runtime to parse, so cgo compiles the
preambles and Go type-checks the files: `CGO_ENABLED=1 go build -tags cuda
./modules/gpu` exits 0 and `go vet -tags cuda ./modules/gpu` exits 0. The binary
build gets through both packages and dies at the link — `cannot find
-lnotorch_gpu`, `-lcudart`, `-lcublas` — which is exactly where it died on
`origin/main` before the move. What stays unverified is that link and every line
of CUDA behaviour behind it; nothing here has run a kernel.

**Live, one organism, this binary.** From a scratch dir seeded with a copy of
`molequla-run/earth/`, `HOME` pointed at the scratch dir so the live colony's
swarm registry is not touched, `taskset -c 4-7`, `oom_score_adj` 500. The
checkpoint loads: the process resumes at *stage 4 (embd=224)* and enters its
warmup. It cannot reach a tick inside two minutes and the log says why — stage-4
warmup is 1600 steps at 0.7-0.8 steps/s (`molequla-run/earth/earth.stdout`,
`908477ms` for 640 steps), some 35 minutes. So the ticking was measured from a
fresh embryo on the same corpus, 210 s:

    [growth] ONTOGENESIS: stage 0 -> 1
    [growth] ONTOGENESIS: stage 1 -> 2
    [dna] earth wrote 5024 bytes to ecology | gen=72 mag=9.58 fade=1.00
    [debug-onto] tick=10 corpus=689553 ingested=158793 stage=2 freeze=0

Sixteen DNA emissions, two growth events, warmup losses 4.86 → 3.13, NaN 0,
`gpu-dispatch=0` throughout — the CPU/BLAS path, which is what a build without
`-tags cuda` must take.

**Left, named.** Two comments in `molequla.go` (lines 132 and 5982) still point
at `gpu_bindings_linux.go` and `gpu_forward.go` by their old names. They are
comments, not code, and another node is editing that file this session; the
rename belongs in whatever commit next touches those lines rather than in a
cross-agent collision here.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — growth is a memory event, and a session that ends must leave its work on disk

At 19:45-19:47Z all four organisms of the live colony crossed into stage 4
together — `embd 128→224, layer 4→5, head 4→8`, then `warmup for stage 4 —
1600 steps` — and each wrote a 110 MB `molequla_ckpt.json` inside those two
minutes. Per-organism `VmHWM` after that was 758-928 MB against 231-240 MB
before the step, `MemAvailable` fell to 740 MB with 125 MB of swap left, and at
19:47:18Z Android's lmkd took `com.termux` and about twenty apps rather than
the colony: processes started under Magisk su inherit `oom_score_adj=-1000`,
while Termux sits at 0. The colony survived by being unkillable, which is the
opposite of what a colony on a phone should be. Everything below is repair 9,
on `claude/phone1-repair-growth-budget`.

**Where the peak comes from, measured.** `growth_budget_test.go`,
`TestCheckpointMemoryProfile`, run on `taskset -c 0-3` against a read-only copy
of earth's checkpoint (105.6 MB on disk, stage 4, 4 814 696 params, vocab 643),
with `500` written to the test's own `oom_score_adj` first and the whole thing
refusing to start below 2000 MB of `MemAvailable`:

    LoadCheckpoint:                 RSS 7 → 375 MB (+368), HWM 7 → 375 MB (+368)
    SaveCheckpoint (streamed):      HWM 376 → 376 MB (+0), wrote 105.6 MB
    SaveCheckpoint (old encoder):   HWM 184 → 521 MB (+337), 105.6 MB in memory

The old save path paid for the weights three times: a `[][]float64` copy of
every matrix into a `CheckpointData`, then `json.Encoder`'s own buffer, which
holds the whole document before a byte reaches the file. `writeCheckpointJSON`
now streams the same document matrix by matrix through a `bufio.Writer`, so the
only large allocation alive at once is one row. The two paths are compared byte
for byte in `TestCheckpointStreamMatchesEncoder`, with the old path kept in the
test file as the oracle; the format did not change, only the allocations.
Independently, `TestStage4SavePeak` grows an organism from an embryo to stage 4
(4 642 654 params, `HWM 8 → 99 MB`) and saves it: streamed `HWM 99 → 177 MB`,
old path `+319 MB` on top of that.

Those numbers account for the incident. 368 MB of resident model plus 337 MB of
save transient is 705 MB, against the 520-690 MB per-organism increment actually
observed, and four organisms doing it inside two minutes is the 3.4 GB.

**The gate.** `growthGateDecision` in `governor_phone.go` is the mitosis gate's
arithmetic over a different cost: a stage step does not add a process, it
multiplies this one, so the charge is `CFG.GrowthPeakFactorPct` percent of the
organism's own `VmHWM` on top of a `CFG.GrowthMinFreeMB` floor. The factor is
300 because the measured increment over the organism's own peak was
`(758-231)/231 = 2.28` and `(928-240)/240 = 2.87`, rounded up for headroom; the
floor is 256, matching `MitosisMinFreeMB`, and either knob at 0 disables the
gate. `memGateDecision` does the comparison, unchanged, so there is one place
that parses `/proc` and one place that decides. The gate stands at both growth
call sites — the tick loop and the bootstrap climb — and a closed gate only
defers: nothing about the decision is remembered, the stage is still wanted, and
the next check grows. `GrowthWanted` is the one predicate both the gate and
`MaybeGrowArchitecture` ask, so the stage arithmetic does not live in two places.

**The colony lock.** `CoordinateWarmup` was the existing serializer and it stays
off: its `continue` skips the whole tick, which froze three of four organisms in
2026-06-03. Keeping four stage transitions apart does not cost that, so it is
its own `growth_lock` row in mesh.db with the same atomic-admit shape as
`training_lock`, taken before growth and released only after the warmup behind
it — one memory event spanning ticks, one lock. The TTL is 300 s and the holder
re-stamps it every 60 s, so the TTL only bounds how long a killed organism's
lock blocks its siblings. `CFG.CoordinateGrowth` defaults on.

**oom_score_adj.** Every process — organisms and witness, both pass through
`main` — writes `CFG.OomScoreAdj` (default 300, 0 leaves it untouched) to
`/proc/self/oom_score_adj` before anything else allocates, and says what it read
back. Live: `[oom] oom_score_adj=300 (lmkd reaches for the organism before the
terminal)`, with the shell wrappers around it still showing the inherited -1000.

**The checkpoint on the way out.** After `phone1/stop.sh` at 20:14Z every
checkpoint on disk still carried its 19:45-19:47Z growth-time mtime, and
`earth.stdout` ended with a completed warmup phase followed by
`[evolution] Organism shutting down gracefully (signal)` — the work of those
minutes never reached the disk. Two things were missing. The exit path now saves
under `model.mu` and passes `CFG.CkptPath` explicitly rather than `""`, because
the empty path is the debounced periodic path and a shutdown inside
`CheckpointMinInterval` of the last burst save writes nothing at all. And
closing `stop` was never enough: `ntWarmupTrain` holds `model.mu` for its entire
phase, so the first attempt at this sat on the mutex for 44 s past SIGTERM with
the checkpoint still untouched, and would have sat there for the rest of the
1600 steps. `trainAbort` is raised by `waitEvolution` on the signal and read by
`ntTrainCore` every step; the loop leaves at the next step boundary, `pullBack`
mirrors what was trained into `model.Base`, and the lock is free. A warmup cut
short does not set `lastWarmupStage`, so the next session resumes that stage
instead of walking into it untrained. The first run to survive a SIGTERM then
announced `[notorch] warmup complete: 640 steps` while the checkpoint it saved
had moved `global_step` by 174, because the line printed the steps requested
rather than the steps taken; it now prints the steps that ran and appends
`(stopped early, N requested)` when the abort cut the phase.

**Gates, each shown red.** `growthGateDecision` made to ignore the factor:
`open=true need=496, want open need=976`. `AcquireGrowthLock` made to admit
everyone: `a sibling must be refused while another organism is growing`.
`applyOomScoreAdj` made to return without writing: `read back "", want "300"`.
The shutdown save sent through the debounced path: `the shutdown save wrote
nothing — a SIGTERM'd session loses its work`. The streamed writer's first field
renamed: `TestCheckpointStreamMatchesEncoder` red on the byte compare. The
per-step abort check removed: the test never returns and `go test` fails on its
own timeout at `600.033s`. Suite `CGO_ENABLED=1 taskset -c 0-3 go test -count=1
-buildvcs=false ./...`: **178 PASS, 0 FAIL**, plus the two heavy measurements,
which skip unless `MOLEQULA_CKPT_MEASURE` or `MOLEQULA_HEAVY` is set and were
run and passed above. Baseline was 167.

**Live, three runs, `taskset -c 0-3`, binary built with `-a`.** A stage-4
organism restored from the checkpoint copy and given 300 s
(`--element earth --evolution`): `oom_score_adj=300`, `VmHWM` 435 MB at t+40 s
rising to 673 MB at t+290 s — against 758-928 MB for the same stage during the
incident — then on SIGTERM

    [evolution] Organism shutting down gracefully (signal).
    [notorch] warmup complete: 174 steps (stopped early, 640 requested), ...
    [evolution] checkpoint saved on signal

with the checkpoint mtime moving 20:32:55Z → 20:38:13Z, `global_step` 3264 →
3438 and `last_warmup_stage` still 3. A fresh embryo on the shipped defaults
grows through `ONTOGENESIS: stage 0 -> 1` and `1 -> 2` with the gate silent. The
same fresh embryo under a binary built with `GrowthMinFreeMB` set to 99999 —
the only way to close the gate on a machine with 2.3 GB free, and not a shipped
configuration — prints the deferral once per check and keeps ticking at stage 0:

    [growth] deferred: free=2296 MB need=100128 MB
    [debug-onto] tick=10 corpus=173643 ingested=134553 stage=0 freeze=0
    [growth] deferred: free=2317 MB need=100194 MB

**Two ceilings, declared rather than inherited.** A byte gate is a reaction to
the machine minute by minute; the size of the colony and the size of its
organisms are decisions, and on this phone they were pod defaults nobody chose.
`CFG.MaxOrganisms` was 16 — sixteen trainer processes on 8 GB — and `--config`
loads only birth paths, so there was no way to say otherwise at launch. Two
flags now reach `CFG` in `parseCLIArgs`, before anything allocates:
`--max-organisms N`, which the cascade governor's atomic admit already reads,
and `--max-growth-stage N` for the new `CFG.MaxGrowthStage`, which `GrowthWanted`
refuses to cross and announces once as `[growth] capped at stage N`. The default
is the last index of `GrowthStages`, so an unset flag changes nothing; a value
past the table or below zero clamps to the last stage rather than freezing an
embryo by accident. Both are printed at startup and `phone1/launch.sh` now passes
`--max-organisms 4`:

    [caps] colony ≤ 4 organisms | growth ≤ stage 0 of 5 (embd=16, layer=1, head=1)
    [growth] capped at stage 0

— live, once, with the organism carrying on at stage 0 for the rest of the run
against a corpus (`ingested=134553`) that would otherwise have taken it to stage
2. The gates: `TestMaxGrowthStageRefusesTheNextStage` grows an organism one stage,
sets the ceiling to the stage it is standing on and asserts nothing moves, then
raises the ceiling and asserts it moves again — red as `GrowthWanted must be
false at the ceiling` when the check is removed. `TestMaxOrganismsOneRefusesTheSecondAdmit`
refuses a divide at a cap of 1, admits it at 4, and checks that the flags
actually land in `CFG` — red as `--max-organisms 4 left CFG.MaxOrganisms at 16`
when the parse is dropped.

**Left, named.** `LoadCheckpoint` still costs 368 MB for a 105 MB checkpoint —
the decoder buffers the whole document and then materialises `CheckpointData`
before the `MatrixParam`s — and it was not touched here; streaming the load is a
second repair. The bootstrap climb of a *fresh* embryo still has no signal
handler, because `signal.Notify` runs after it; that path saves after each
completed stage warmup, so only an in-flight first-launch warmup is lost. The
factor 300 is calibrated on pre-repair peaks that included the 337 MB save
transient this session removed, which makes the gate conservative rather than
loose — the right direction, and a number to re-measure once the colony has run
a stage step under the new binary.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the checkpoint is read the way it is written, and the signal is armed before the first warmup

Repair 9 closed with two things named and left: `LoadCheckpoint` still cost
368 MB for a 105 MB file, and a fresh embryo's bootstrap climb ran before
`signal.Notify` was installed, so a SIGTERM during a first launch killed the
process outright. Both are closed here, on `claude/phone1-repair-load`.

**The load, measured.** `TestCheckpointMemoryProfile`, against a read-only copy
of earth's live checkpoint — `molequla-run/earth/molequla_ckpt.json`,
111 135 696 bytes = 106.0 MB, stage 4, 4 834 408 params, vocab 687 — on
`taskset -c 4-7`, with `500` written to the test's own `oom_score_adj` and the
whole thing refusing to start below 2000 MB of `MemAvailable`:

    LoadCheckpoint:                 RSS 7 → 159 MB (+152), HWM 7 → 159 MB (+152)
    SaveCheckpoint (streamed):      RSS 159 → 167 MB (+8), wrote 106.0 MB
    SaveCheckpoint (old encoder):   RSS 167 → 486 MB (+319), 106.0 MB in memory

Repair 9 recorded `+368` on that load line against a 105.6 MB copy of the same
file. The same before/after taken back to back in one session, the old path
replicated beside the new one, reads `+349` and `+152`, and the token walk on its
own — `readCheckpointStream` with no GPT built around it — costs `+83`.

The old path held the weights three times: `json.Decoder` buffered the whole
document because it was asked for one top-level value, the `CheckpointData` held
every matrix again as `[][]float64`, and `deserializeMatrixParam` then built the
`*Vec` rows from that. `readCheckpointStream` walks the document with
`Token()` / `More()` / `Decode` into per-row targets, so each row goes straight
into `NewVecWithGrad` and the decoder's buffer compacts down to the largest
single value it is asked for, which is one row of 224 floats. Field order is not
assumed and unknown keys are skipped, so a checkpoint written by the C, Rust or
JS core still loads.

What is left in the 152 is not a copy of anything. 4 834 408 params at a
`float64` of data plus a `float64` of grad is 73.8 MB, and the walk costs 83 —
the model, plus about nine of `*Vec` headers and allocator slack. The remaining
69 MB is `NewGPT(tok)` inside `LoadCheckpoint`: it builds a complete
random-initialised weight set at the restored dimensions, and the next statement
replaces `model.Base` with the matrices just read. That transient predates this
session and is not touched here, because the fresh matrices are also what a
checkpoint missing a key silently falls back on — removing them turns a foreign
or pre-SwiGLU checkpoint from quietly-random into a nil dereference, which is a
decision about format compatibility rather than about memory.

**The gates on the load.** `TestCheckpointRoundTripIsByteIdentical` saves the
grown fixture, loads it back and saves again, and compares the two files byte for
byte — 1 224 577 bytes, three delta modules — then checks the weights, the grad
allocation, the embedding tie and the growth counters came back. It goes red as
`the round trip changed the checkpoint (1225175 vs 1225170 bytes)` when the
restore of `ActiveAlpha` is dropped. `TestLoadCheckpointRejectsTruncation` cuts a
1 225 324-byte checkpoint at 1, 10, 33, 50, 75 and 95 percent, plus a file
holding only `{`, and requires an error and a nil model from each; it goes red
with `panic: runtime error: invalid memory address or nil pointer dereference`
when `readCheckpointStream` is made to return what it has instead of the error —
which is the failure it exists to catch, a half-built organism training on
whatever it got. `TestCheckpointStreamMatchesEncoder` is unchanged and still
holds the write side to the old encoder's bytes.

**The signal, armed earlier.** `signal.Notify` now runs in `main` right after the
CLI is parsed, before the checkpoint is opened and before the bootstrap climb,
and its handler raises `trainAbort` and closes a `shutdown` channel;
`waitEvolution` waits on that channel instead of on the signal channel, so a
signal that landed minutes earlier is still there when the evolution loop is
finally reached. The bootstrap's per-stage warmup answers the abort the same way
the tick loop's does: it does not record `lastWarmupStage` for a phase it did not
finish, and it writes what it has on the explicit path, which the debouncer
cannot drop. `saveOnShutdown` takes the phase it is saving for, so the line says
which loop ended. Evolution mode only — in the REPL a Ctrl+C must still end the
process, and that bootstrap pauses for the user between stages anyway.

Live, on a fresh embryo in a scratch `HOME` with a 173 643-byte earth corpus,
`timeout -s TERM 20 ... --element earth --evolution --max-organisms 1`:

    [init] Stage 1 (infant): embd=32, layer=1, head=2 — warmup 800 steps
    [notorch] warmup complete: 320 steps, avg loss 3.3869 | 5080ms 63.0 steps/s
    [notorch] warmup complete: 34 steps (stopped early, 240 requested), avg loss 3.5768
    [init] checkpoint saved on signal

— a 1 942 807-byte checkpoint, and relaunching in the same directory resumes at
`[trainer] warmup for stage 1 (embd=32)` with no bootstrap line, which is also
the streamed loader read end to end on a file it did not write in the same
process. The same run with the early handler removed and nothing else changed
leaves a 934 122-byte stage-0 checkpoint, no save line, and the stage-1 work
gone: that is the red.

**Also.** Two comments still named `gpu_forward.go` and `gpu_bindings_linux.go`,
which left the root when the GPU lane became a package; they now point at
`gpu_bridge.go` and `modules/gpu/`.

`CGO_ENABLED=1 taskset -c 4-7 go test -count=1 -buildvcs=false ./...` — 180 pass,
2 skipped, up from 178 by the two new gates. The skips are the two heavy
measurements, which refuse to start below 2000 MB of `MemAvailable`; the load
numbers above come from running the first of them by hand once the phone was
quiet, as `MOLEQULA_CKPT_MEASURE=<read-only copy> go test -count=1 -run
TestCheckpointMemoryProfile -v .`.

One thing seen and not fixed: `TestBeatKeeperRefreshesMeshWithoutTicks` fails
about one run in eight with `SQL logic error: no such table: organisms`, on this
branch and on `origin/main` alike. Its fixture opens `sqlite` at `:memory:`
through `database/sql`, where the database belongs to a connection and the pool
is free to open a second one; the table is then missing on whichever connection
the keeper's goroutine draws. It is the fixture, not the keeper, and it wants a
shared cache or a single pinned connection.

## 2026-09-13 — the senses get their own schedule, and the organisms are told to eat what they leave

The colony runs six hours a day and sleeps eighteen. Everything the phone could
notice in between was, until tonight, noticed by nobody. `phone1/senses.sh` is
one pass of three organs that runs on its own slots and exits: a frame from each
camera through the eye, twelve seconds of microphone through whisper.cpp, one
fix of where the phone is and what the sky is doing. What each organ produces
is written as `gen_<unix>_<seq>.txt` into `$MOLEQULA_RUN/dna/output/world/`,
`sound/` and `place/` — the names `dnaFragOrder` parses and `dnaListNew`
orders — one fragment per sentence, each behind a bracketed header that reads
as ordinary text to whatever eats it. The sequence number is a counter in
`senses/seq` that only ever grows, so two fragments written in the same second
still have an order; the pair `(unix, seq)` is exactly what `dnaNewer` compares.

**The three fragments of the 22:18:11Z pass, verbatim.**

    [eye cam0 2026-09-13T22:18:27Z] A close-up view shows a keyboard with
    white keys and a dark background, with no text visible.
    [eye cam1 2026-09-13T22:18:46Z] A bathroom with a shower curtain hanging,
    a person's hand reaching out, and a light fixture above.
    [place 2026-09-13T22:19:24Z] The phone is at <neighbourhood>, Be’er-Sheva,
    Israel (fix accurate to 14 m). Local time 2026-09-14T01:15 (Asia/Jerusalem),
    22.5 °C, humidity 97%, wind 2.1 km/h, fog. Sunrise 2026-09-14T06:24, sunset
    2026-09-14T18:48. And it has not moved more than 50 m since the last pass
    (2 m from it).

The cam1 sentence was read as a hallucination when the pass was logged; the
person who owns the balcony corrected it. The phone lay on a balcony with tiled
walls and a towel hung up to dry beside it, in light he calls terrible, and the
front camera saw tile and hanging cloth and named the nearest thing it knows:
a bathroom and a shower curtain. The substance was right and the label was
the closest word in a 500M vocabulary — the same failure shape as the "tablecloth
with a drawing" that was tile floor and shadow in the first eye measurement above.
It is left in the record rather than filtered out: the eye reports what it
believes, and an organism eating the DNA field is eating a belief either way;
the ledger of change (ROADMAP item 10) is what turns a hundred identical
"bathrooms" beside a place line that says balcony, night, fog into a contradiction
the organism can hold. An earlier pass the same minute, on a lit kitchen table,
produced «A blurry kitchen table shows a green bowl, a spoon, and a plate, with
a blurry background that looks like a kitchen counter» — the same engine, and
right.

and the line the pass left in `senses/senses.log`:

    2026-09-13T22:19:24Z pass=all cpu=4-7 mem_mb=2896->2898
      eye=rc0,28s,rss1020mb,frames2,frags2 ears=rc0,30s,speechno,frags0
      place=rc0,7s,movedno,frags1 frags=3 total=73s

**The eye.** `reffs/ocelli/eye` on the q6_k Yent decoder with one global frame
(`SMOLVLM_NOSPLIT=1`), both cameras every pass: 28-31 s for the pair, peak RSS
1044312 kB from `/usr/bin/time -v` — 1020 MB. Two frames because one of them is
usually in the dark and a pass that takes both still sees something. The mmproj
was measured both ways on the same 768×1024 frame, cpu4-7, byte-identical text
(«A blurry kitchen table shows a green bowl, a glass, and a plate, with a
person's head in the background»): q8_0 mmproj 18.68 s / 1044312 kB, f16 mmproj
21.90 s / 951296 kB. The quantized projector is 3.2 s faster and 93 MB heavier
than the f16 one, which is the opposite of what its file size (108782176 B
against 199467616 B) suggests; it is recorded, not explained, and both are one
environment variable. The camera's own frame is 4080×3060 and 3.7 MB; it is
scaled to a 1024 px longest edge first, because `vision.c:55-66` resizes the
longest edge to `LONGEST` = 2048 (`vision.c:18`) before anything else and a
12 MP frame would be decoded
into 150 MB of float on the way to a 512×512 global view. Below 1300 MB of
MemAvailable the eye does not open and the pass says so — forced once with
`SENSES_EYE_MIN_MB=99000`, which logged `eye=rc0,0s,rss0mb,frames0,frags0,skip-mem:2922`
and wrote nothing.

**The ears**, whisper.cpp `tiny` at `-t 4 -l auto -nth 0.6 -sns`, twelve seconds
of audio: about 20 s of wall in a quiet room and no fragment at all, which is
the point — `base` once spent 185 s on eight seconds of ambience and emitted
`[Motor]` (`~/arianna/ears-reference/REFERENCE.md`). Bracketed tags are stripped
and what remains must still be eight characters with a letter in it. Proven the
other way at 22:20:47Z by playing `ears-reference/wav/jfk.wav` through the
phone's own speaker into its own microphone while the pass recorded:

    [ears mic 2026-09-13T22:21:08Z] And so my fellow Americans ask not what
    your country can do for you.

21 s wall, `ears=rc0,21s,speechyes,frags1`. The same twelve seconds of silence
measures -44.1 dB mean with `ffmpeg -af volumedetect`, so the empty passes are
an empty room, not a broken microphone. `SENSES_ASR` and `SENSES_ASR_MODEL` are
two variables and nothing else knows the recognizer's name, so the native
`ears` on notorch replaces it in one line.

**Place** is `termux-location` (network, 13 m here; satellites only if that
fails), open-meteo for temperature, humidity, wind, the WMO code as a word and
today's sunrise and sunset, and nominatim reverse at zoom 14 with a User-Agent
and `accept-language=en` — both keyless, `curl` and `jq` only, about 5-7 s
together. The previous fix lives in `senses/place.last` and a haversine in awk
decides whether the phone moved more than 50 m.

**Slot kinds.** `schedule.conf` grows `SENSES_SLOTS="01:00 03:00 07:00 09:00
11:00 15:00 17:00 19:00 23:00"`, `SENSES_CMD` and `SENSES_TIMEOUT=600` — ten
times the 73 s a full pass measured, for the day the network hangs. The times
are the gaps between the colony windows (04-06, 12-14, 20-22 UTC) with an hour
of clearance on either side. `next_any` returns the nearest slot of either kind
with its kind and gives a tie to the colony; `in_colony_window` is the predicate
that keeps a senses pass out of a session, testing both the current day's
occurrence of each colony slot and the previous day's, because a long session
runs past midnight. At run time a senses slot is refused twice over — by the
configured window and by `live_names`, since a manual `launch.sh` obeys
neither — and logged as `reason=skipped-colony-window` or
`skipped-colony-alive`. Every line in `schedule.log` now carries `kind=`.
Senses slots are configuration, not a built-in default: with `SENSES_SLOTS`
empty this is the colony scheduler it was before, which is also why the
twenty-one existing cases still measure colony arithmetic unchanged.

**The flag.** `--dna-extra-sources world,sound,place` sets `CFG.DNAExtraSources`
in `parseCLIArgs` (`molequla.go:6267`), following `--max-organisms` from repair
9, and `launch.sh` passes it and creates the three directories. One flag feeds
two paths: `dnaRead` appends the fragments to the organism's corpus, and
`NewCrossField` takes its sibling list from the same `dnaSources()`
(`cross_graze.go:60`), so the senses also reach the logit overlay. Without it
the directories exist and nobody reads them.

**Gates, each shown red once.** Go: 182 tests green (178 before), four new in
`dna_extra_sources_test.go` driving the real `parseCLIArgs` — the flag reaches
`dnaSources`, spacing and empty names between the commas are tolerated, an
absent flag changes nothing, and a fragment dropped into `dna/output/world/` is
what `dnaListNew` hands the organism. Red as `CFG.DNAExtraSources = [], want
three entries` and `dnaListNew saw 0 fragments in world, want 1 (sources: [air
water fire])` when the parsed value is dropped on the floor. Bash: 36 cases in
`schedule_test.sh` (21 before), fifteen new — eight on `next --epoch` and
`next --kind` across interleaved lists, the tie, and an empty senses list;
seven on `in-window`. Red three ways: `next_any` always preferring the colony
took four interleaving cases down; dropping the previous day from
`in_colony_window` failed `yesterday's session reaches into today: outside,
want inside`; making the window open strictly after its slot failed `the colony
start itself`.

**Left, named.** The running daemon (pid 19016) is still the one started from
the main checkout before any of this, so it keeps taking colony-only slots; the
new conf was parsed and read back from the worktree (`schedule.sh status`,
`next: 2026-09-13T23:00:00Z kind=senses, cap 600s`) but the daemon is not
restarted and the binary in `$MOLEQULA_RUN` predates the flag — both wait for
the merge. The eye's 1020 MB is the largest single allocation the phone makes
outside a colony session; it is fenced by MemAvailable and by the colony
windows, and it has not yet been measured against a session that overran its
cap into a senses slot.

— Defender (Arianna Method, phone-1)

## 2026-09-13 — the senses move into the tree, and the ear becomes ours

Oleg's decision tonight: the organs live inside molequla, in one folder
`senses/`, and neither gets a repository of its own. The eye was a gitignored
reference clone in `reffs/ocelli`; the ears were a separate checkout at
`~/arianna/ears` on branch `claude/notorch-primitives`. Both are in the tree
now, and for the first time the recognizer molequla runs is molequla's own.

**Moved with their commits, not copied.** `git mv` was never an option for the
eye: `reffs/` is ignored whole (`.gitignore:106`, `/reffs/`) and
`git ls-files reffs` returned nothing, so there was no history in this
repository to move. Both organs came in by `git subtree add` instead, from the
local checkouts — ocelli's eleven commits up to `c4fe095`, ears' two up to
`047a141`, each reachable as the second parent of its merge (`84fb401^2`,
`d17ff2b^2`). What did not come in: neither engine binary and no weights.
ocelli's own `.gitignore` already covers `ocelli` and `models/`, ears' covers
`ears`, and the one thing those patterns missed — the absolute symlink
`reffs/ocelli/models -> /data/data/com.termux/files/home/models/ocelli`, which
`models/` does not match because it is a symlink and not a directory — is not
recreated. It is not needed: `SENSES_EYE_MODEL` and `SENSES_EYE_MMPROJ` reach
the wrapper as `EYE_MODEL` / `EYE_MMPROJ`, which `eye` has always honoured
(`senses/ocelli/eye:16-17`), and `/senses/ocelli/models` plus
`/senses/ears/models` are ignored so it cannot come back by accident.

**Both build from the new path.** `cd senses/ocelli && make` → 7.9 s,
`./ocelli` 285056 B. `cd senses/ears && make` → 1.7 s. The eye on one frame from
the live run directory, read-only, q6_k decoder with the q8_0 projector and one
global frame: `A bathroom with a shower curtain, a light above the shower, and a
small trash can.` — 12.8 s wall on cores 4-7, prompt 84 tok at 58.4 tok/s, gen
19 tok at 4.0 tok/s. (That is the balcony the diary already corrected above; the
wording is unchanged by the move, which is the point of running it.)

**The ears' Makefile pointed at a directory that does not exist here.**
`WHISPER ?= $(HOME)/arianna/whisper.cpp` and `REF ?= $(HOME)/arianna/ears-reference`
were written in a shell where `$HOME` was the phone's tree; inside this chroot
`$HOME` is `/root`, and every gate would have looked for the oracle there. They
are now derived from the Makefile's own path — three levels up from `senses/ears`
is the directory that holds molequla, `whisper.cpp` and `ears-reference` side by
side — and `make -n test-transcript` resolves them to
`/data/data/com.termux/files/home/arianna/whisper.cpp` and
`.../ears-reference` with nothing passed on the command line. `senses/ears/README.md`
now says plainly that `make` needs only notorch and OpenBLAS while `make test`
needs both of those checkouts.

**The ear in `senses.sh` is `senses/ears/ears`.** Default weights
`~/models/ears/ggml-tiny.bin`, copied from whisper.cpp's own downloads and
md5-identical to them, mirrored on Hugging Face at `ataeff/molequla` under
`ears/`. whisper-cli stays behind the same two variables, `SENSES_ASR` and
`SENSES_ASR_MODEL`, which is what the old comment in that file promised; the two
take their arguments differently (`ears` positionally, `whisper-cli` through
`-m` and `-f`), so the command line switches on the binary's basename, with
`SENSES_ASR_KIND` to force it. No `--json` and no `-q` was added to `ears.c`:
both engines already put the transcript and nothing else on stdout — ears keeps
its `no_speech` / `avg_logprob` line per window on stderr — so what was needed
was the right argument order, not a new output mode, and an unused flag is worse
than none.

**Gates.** `senses/ears` from its new location, defaults only,
`make test-transcript`: all six rows equal token for token against whisper.cpp
under pure greedy — jfk/tiny 26 tokens, speech_air_14s/tiny 27, ambient_8s/tiny
0, jfk/base 27, speech_air_14s/base 28, ambient_8s/base 0 — 2m28.877s wall on
cores 0-3. Go: 184 pass, 2 skipped, 0 fail
(`CGO_ENABLED=1 taskset -c 4-7 go test -count=1 -buildvcs=false ./...`, 4.841 s);
the two skips are `TestCheckpointMemoryProfile` and `TestStage4SavePeak`, which
want `MOLEQULA_CKPT_MEASURE` and `MOLEQULA_HEAVY=1`. That count is inherited,
not earned — `git diff --stat origin/main -- '*.go'` is empty, this branch adds
no Go file and no Go test. `bash phone1/schedule_test.sh`: 36 pass, 0 fail.
`go build -a` 47.9 s, and `go list ./...` still names exactly two packages,
`molequla` and `modules/gpu` — `senses/` holds no `.go` file, so cgo never
reaches the C in it.

**The ear, live, on a scratch field.** One `senses.sh ears` pass with
`MOLEQULA_RUN` pointed at a scratch copy so the colony's `dna/output/` was not
touched, while `jfk.wav` played out of the phone's own speaker over
`termux-media-player` and the microphone recorded twelve seconds of the room:

    [senses] pass ears at 2026-09-13T23:08:47Z: cpu 4-7, MemAvailable 2769 MB
    [senses] sound/gen_1789340949_1.txt (140 B)
    ears=rc0,22s,speechyes,frags1

and in the fragment:

    [ears mic 2026-09-13T23:09:09Z] And so my fellow Americans, as not what you
    are country can do for you and what you can do for your country.

Not the clean transcript — the same air-and-speaker degradation the reference
wav `speech_air_14s` carries, arriving through the same room. That is what the
organ is for: what the phone actually heard, not what the file says.

**Left, named.** `reffs/ocelli` is still on disk in the main checkout. It is
ignored and nothing points at it any more, so it is a stale clone rather than a
second copy of a tracked path; removing it is a hand movement outside this
branch. `~/arianna/ears` is left in place deliberately — the subtree took its
history, and it goes after the merge, not before. The eye's watermark probe
(`senses/ocelli/OCELLILOG.md`, 14 frames) was not re-run: it is a hand-run gate
over a frame set that is not in the tree, and nothing in the engine changed.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the inspection §17 asks for, before anything is built

`molequla_new_logic.md` opens its seventeenth section by asking that the paths which already exist be
inspected before new machinery is added under new names. That inspection is
`reports/2026-09-15_new_logic_audit/README.md`: a row per section 1-16 saying what exists with a line
number, what is partial, what is missing and where the smallest extension point is; a trace of how an
eaten fragment actually reaches an organism; the lineage reading of dario, q, actually.life and netta;
and a proposed order for §18 in which routing changes come before new infrastructure. Nothing in the
tree changed — the audit is one document and this entry.

The load-bearing answer is that the brief's early regime already runs end to end. `dnaRead` appends a
fragment's bytes to the organism's corpus (`molequla.go:6138-6144`), `loadCorpusLines` and
`BuildFromCorpus` turn the corpus into unigram, bigram, trigram, 4-gram and co-occurrence every thirty
ticks (`molequla.go:6625-6635`, `:4227-4294`), the same `docs` slice is what `MetaweightsOverlay` reads
through `model.corpusField` and what both trainers are handed (`molequla.go:6673-6675`, `:6753`), and a
second, independent path takes the same files straight to the logits (`cross_graze.go:79-165`,
`molequla.go:4745-4747`). Routing lives in one function, `dnaSources(element)` (`dna_field.go:35-48`),
and today it is the same list for everybody: `SENSES_SOURCES="world sound place"` goes to all four
launches as one string (`phone1/launch.sh:16,88`).

Three findings were not being looked for. `loadCorpusLines` truncates every line to
`MaxLineChars = 240` on every read (`molequla.go:3262-3264`, default `:268`), and an eaten fragment is
one line padded toward 5000 B (`molequla.go:271`), so most of a sibling fragment never reaches the
field or a batch while the growth clock counted all of it — in the live run `earth/nonames_earth.txt`
holds 1600 lines of which 592 exceed 240 characters and the longest is 5406, and a place fragment at
301-318 B loses its moved clause. The senses also stand last in the read queue: extra sources come
after the elements in `dnaSources` under one budget of `DNAMaxReadsPerTick = 8` (`molequla.go:273`,
`:6117-6123`), and in the 2026-09-13T20:26Z session that cap was hit on 12 of earth's 15 reads, 9 of
air's 13, 10 of water's 13 and 4 of fire's 14. And the witness cannot see the senses at all: it calls
`dnaSources("")` (`witness.go:458`) in a process `launch.sh:95-96` starts without
`--dna-extra-sources`. The four live cursors confirm the state the roadmap predicted — no `world`,
`sound` or `place` key in any `dna_cursor.json`, thirteen fragments waiting for the next session.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the sensing window, and the half of hearing that is not language

Two changes from `molequla_new_logic.md`, §2 and §3, and steps 1-5 of §18. A
sensing episode stops being one sample of each organ: the eye takes a short
measured trajectory, and the microphone's twelve seconds leave something behind
even when nobody says a word. Both on `claude/phone1-sensing-window`, not pushed.

**The window.** `phone1/senses.sh` grew three variables and a summary line.
`SENSES_EYE_PATTERN` is the camera order, cycled; `SENSES_EYE_WINDOW` is how many
frames the pass takes; `SENSES_EYE_SPACING` is the seconds between the starts of
two consecutive captures, so a slow frame does not push the next one back. The
memory floor moved inside the loop — the eye holds about a gigabyte per frame and
the colony can wake between two frames, so `MemAvailable` is re-read before every
capture and the window stops where it falls short, `skip-mem` in the pass line,
rather than failing the slot. Each window appends one `eyewin` line beside the
pass line, e.g.

    2026-09-15T02:01:04Z eyewin pattern=0,1,0,0 n=4 spacing=30s frames=4 said=4
    repeat=1 novel=0.750 wall=107s rss=1020mb batt=96%->96%,1395->1009mA cpu=4-7

**The cadence was measured, not decided.** Six windows on cores 4-7, a scratch
`MOLEQULA_RUN` so the live field was untouched, pattern `0 1 0 0`, spacing 30 s,
each configuration twice, 14 frames in total:

| n | wall, run 1 | wall, run 2 | peak RSS | descriptions | repeats | novelty |
|---|---|---|---|---|---|---|
| 1 | 18 s | 15 s | 1020 MB | 1 | 0 | 1.000 |
| 2 | 46 s | 49 s | 1020 MB | 2 | 0 | 1.000 |
| 4 | 107 s | 104 s | 1020 MB | 4 | 1 | 0.750 |

Peak RSS is the eye process's own `VmHWM` through `/usr/bin/time -v` and it does
not move with n: the window is one process per frame and nothing accumulates
across them. Novelty is the share of descriptions that did not repeat an earlier
frame of the same window, where a repeat is a token overlap ≥ 0.8 — lowercased,
non-alphanumerics as separators, intersection over union of the distinct tokens.
Measured overlaps from those same windows: a rear frame repeating an earlier rear
frame 1.000 (the eye returns the sentence word for word), two different scenes
from one camera 0.350 and 0.368, the two cameras of one window 0.154. Nothing
lands between 0.4 and 1.0, so the threshold sits in the middle of empty space.

**The default is n=4 at 30 s, and the reason is in the table.** Not because n=4
scored highest — it scored lowest. At n=2 the two frames come from different
cameras and cannot help being new, so 1.000 there is arithmetic rather than a
discovery; n=4 puts two rear frames a minute apart, and in both runs one of them
repeated, which is the window noticing that the scene held still. That is the
observation §5 wants and a single frame cannot produce. It costs 104-107 s, 18 %
of the 600 s senses slot cap, and a full pass with it measures about 200 s worst
case. Spacing comes out of the eye's own period: a frame occupies 15-18 s
(14 s engine, 3-4 s capture and scale), so below ~20 s there is no spacing at
all; 30 s leaves 12-15 s of world between frames and spreads n=4 over 90 s.

**Battery, honestly.** The window line records `capacity` and `current_now` from
`/sys/class/power_supply/battery` before and after, but the phone was on the
charger for all six windows — capacity went 95 % → 97 % while the charge current
fell from 1605 mA to 979 mA as the battery filled. No discharge cost can be read
off these runs. The measurement is recorded per window from now on, so the first
unplugged pass will have it.

**Hearing.** A quiet twelve seconds used to produce nothing, and the recognizer's
own `[Motor]`, `(wind)`, `[BLANK_AUDIO]` were stripped before the fragment was
written — correct for speech, empty for hearing. `senses/ears/soundscape` is the
other question, asked without a model: 512-point frames through notorch's
`nt_stft`, level percentiles, the 300-3400 Hz band share, spectral flatness, peak
prominence over its own neighbourhood, onset count, envelope autocorrelation and
the 2-8 Hz modulation share, and one English line naming the kind of sound —
quiet room, a single loud transient, music is audible, speech-like modulation
words unclear, repeated mechanical noise, steady broadband noise, or an unsteady
sound without clear structure. 0.057 s on a 12 s wav against the 22 s the
recognizer takes on the same file. It runs on every pass and writes an
`[ears env …]` fragment beside the `[ears mic …]` transcript, whether or not
anybody spoke, with the recognizer's tags appended to it instead of dropped —
two separate pieces of evidence about the same twelve seconds, which §15 says do
not have to agree. Live, on a scratch field, 2026-09-15T02:13:18Z:

    ears=rc0,22s,speechno,frags1,env:quiet-room
    [ears env 2026-09-15T02:13:18Z] Quiet room. The recognizer also marked [BLANK_AUDIO].

The environmental line also leaves a fact, `ears microphone soundscape "<line>"`,
in `senses/facts.jsonl` beside the fragment — the sidecar the world ledger reads
(`world_ledger.go`), which landed in `main` while this branch was open and which
every other organ already wrote to. Without it the ledger could see that speech
stopped and not that the room itself changed. `soundscape` and not `hearing`
deliberately: the two predicates make different claims about one window and
neither corrects the other.

**Two artefacts the fixtures could not have shown**, both found by running the
describer over the six real recordings the live field left in
`molequla-run/senses/audio/` on 2026-09-13/14. The recorder hands over a third of
a second of digital silence at the start of every wav; frames of literal nothing
sit at -120 dB and put the envelope's standard deviation at 13.3-13.5 dB in all
six files while their 10th-to-90th percentile span was under 6 dB. And with the
spectrum as it arrives, every recording is "tonal" — a phone on a table has more
power under 100 Hz than in the room above it, so six of six night recordings came
out as music. Leading and trailing zeros are now cut before anything is measured,
a one-pole high pass at 100 Hz sits in front of everything, and tonality is
prominence over the median of the 48 bins around the peak rather than over the
whole spectrum. Tonal frame share on those recordings fell from 0.44-0.94 to
0.06-0.13 while the tone fixtures stayed at 1.000, and the six files then read
five `quiet room` — one of them with a single click, `a single loud transient`,
`db_max` -21.2 against a median of -45.6 — and, for the one with `jfk.wav` played
into the room from the phone's own speaker, `speech-like modulation, words
unclear`. Room tone on this microphone measures `db_p90` -47.6 .. -41.5, which is
where the -40 dB quiet threshold comes from. Every threshold is an `SND_*`
environment variable with the measurement written beside its default in
`soundscape.c`.

**Gates, each shown red first.** `phone1/senses_test.sh`, 20 cases through the
real `senses.sh` with the camera, the microphone and the three engines stubbed
and `ffmpeg` real: red on the unchanged script, **0 pass, 20 fail**, green after,
**20 pass, 0 fail**. Ten are the window (four fragments from a four-frame window,
the cameras in the pattern's order at capture and in the headers, a short pattern
cycling, the summary line's n / pattern / repeat=1 / novel=0.750, spacing holding
a capture back), five the overlap metric, five hearing (silence still leaving an
`env` fragment, speech leaving both, a bare `[Motor]` not counting as a
transcript, the tag surviving). `make test-soundscape`, seven fixtures
synthesised in C — there is no sox on this phone and no wav in the repository —
**7 pass, 0 fail**, red twice by moving a threshold rather than patching the
organ: `SND_QUIET_DB=-80` takes the quiet row down, `SND_SPEECH_BAND=1.1` the
speech row. `bash phone1/schedule_test.sh`: 36 pass, 0 fail, unchanged.

**Left, named.** The four whisper parity gates were not re-run: this branch adds
a binary beside the recognizer and changes none of its sources. No Go file and no
Go test changed either. The eye's own 14-frame watermark probe was not re-run —
nothing in the engine changed, only how often it is called. The live scheduler
still runs the pre-merge `senses.sh`, so nothing measured here has reached the
real field; the `dna/output/` used throughout was a scratch directory. And the
tagger that would name a sound rather than classify its shape is a survey, not a
port: `senses/ears/PORT_NOTES_SOUND.md`.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — routing: the witness is told about the senses, the senses get their own queue, and a fragment reaches the field whole

Four of the ten items the audit of `molequla_new_logic.md` proposed for §18
(`reports/2026-09-15_new_logic_audit/README.md` §4, branch
`claude/phone1-new-logic-audit` at `9464b81`): its 1, 2, 3 and 6. All four are
routing over paths that already run, and each one landed behind a gate that was
watched going red on the code it repairs before it was made green.

**1. The witness was never told the senses exist.** `phone1/launch.sh` handed
`--dna-extra-sources world,sound,place` to each of the four organisms and
nothing to the witness, so `dnaSources("")` in the witness process
(`witness.go:458`) was built from an empty `CFG.DNAExtraSources` and
`witnessScanDNA` counted four element directories and never looked into
`dna/output/{world,sound,place}`. The list is now one variable, `SENSES_ARG`,
declared once beside `SENSES_SOURCES` and passed to both readers, so the witness
counts exactly what the organisms eat.

The gate is `phone1/launch_test.sh`, in the shape of `phone1/schedule_test.sh`:
it drives the real `launch.sh` against a stub binary that records its own argv,
asserts that the recorded witness argv carries the same list as the recorded
earth argv, then takes that recorded argv — not a copy of it — and runs the real
`molequla_cgo` with it over a scratch tree holding one `world` fragment, with a
mesh written by two seconds of a real organism rather than by a fixture. A third
stage builds a copy of `launch.sh` with the argument removed and requires the
first stage to fail on it, so the check is known to fail on the thing it exists
for. On `origin/main`'s `launch.sh` the gate reads 1 passed, 4 failed, the
witness argv being `--witness --witness-interval 5` and its snapshot carrying no
`dna` key at all; on the repaired script, 5 passed, 0 failed
(`MOLEQULA_BIN=$PWD/molequla_cgo bash phone1/launch_test.sh`, cores 4-7).

**2. The senses stood last in a queue the siblings kept full.** `dnaRead` walked
`dnaSources(element)` — three siblings, then the extra sources — under one bound
of `CFG.DNAMaxReadsPerTick = 8` and broke out of both loops when it was spent.
Each sibling emits one fragment per tick and a tick is 0.25 s, so a sibling
backlog is the normal state of the field: in the 2026-09-13T20:26Z session the
shared cap was spent before the source list ran out on 12 of earth's 15 reads, 9
of air's 13, 10 of water's 13 and 4 of fire's 14 (audit §3). The extra sources
now read under `CFG.DNAExtraReadsPerTick`, a second counter, and a spent half
skips to the next source instead of ending the walk. Which counter a read is
charged to is decided by membership in `CFG.DNAExtraSources`, not by position,
so the order `dnaSources` returns stays an order of service and stops being an
order of entitlement — and the cafeteria can reorder that list without touching
the budget.

The default is 4, and the number comes from the live field rather than from
taste. One senses pass leaves at most four fragments — eye cam0, eye cam1, ears,
place — and the largest bundle actually on disk is exactly that, `gen_..._4`
through `gen_..._7` between 2026-09-13T22:18:27Z and 22:21:08Z; the two
scheduled passes in `molequla-run/schedule.log` recorded `frags=3` and `frags=2`.
Arrival is nowhere near the bound: the thirteen fragments standing in
`dna/output/{world,sound,place}` span 2026-09-13T22:10:52Z to
2026-09-14T01:00:48Z, 10195 s, which is 4.59 fragments an hour, or 3.2e-4 per
tick. So 4 never binds on live arrival — it binds on the backlog a sixteen-hour
sleep leaves, and it is one sensing episode, so an episode enters whole in one
tick instead of arriving in pieces over four.

Two gates in `dna_extra_sources_test.go`, one per direction. Sixty-four sibling
fragments standing in front of one `world` fragment, one `dnaRead`: the world
fragment must be in the corpus, and air must have been read exactly
`DNAMaxReadsPerTick` times. Sixty-four `world` fragments in front of one sibling
fragment: the sibling must be in the corpus, and world must have been read
exactly `DNAExtraReadsPerTick` times. Reverted to the single shared counter both
go red, and they name the mechanism as they fall — `corpus holds 9 lines, cursor:
map[air:gen_1789337000_7.txt]`, the whole budget spent inside air, and in the
mirror `world was read 7 times, want 4`.

**3. The 240-character ceiling, measured before it was touched.** Every source
was measured on copies of the live run taken 2026-09-15, counting for each line
how many of its bytes survive `loadCorpusLines`, which truncates at
`CFG.MaxLineChars = 240` on every read. First the four corpora as the trainer
sees them:

| corpus | lines | over 240 B | bytes | reach `docs` | share | longest line |
|---|---|---|---|---|---|---|
| earth | 1101 | 592 | 687953 | 158793 | 23.1 % | 5406 |
| air | 1083 | 156 | 540830 | 138428 | 25.6 % | 5221 |
| water | 1672 | 90 | 579634 | 144924 | 25.0 % | 5223 |
| fire | 1590 | 60 | 425086 | 134006 | 31.5 % | 5233 |

Then the fragments standing in the DNA field, each counted the way `dnaRead`
appended it, as one line:

| source | fragments | over 240 B | bytes | reach `docs` | share | longest |
|---|---|---|---|---|---|---|
| earth | 20 | 20 | 102697 | 4800 | 4.7 % | 5233 |
| air | 20 | 20 | 101916 | 4800 | 4.7 % | 5223 |
| water | 20 | 20 | 100867 | 4800 | 4.8 % | 5108 |
| fire | 70 | 70 | 353255 | 16800 | 4.8 % | 5126 |
| world | 8 | 0 | 977 | 977 | 100 % | 156 |
| sound | 1 | 0 | 100 | 100 | 100 % | 100 |
| place | 4 | 4 | 1251 | 960 | 76.7 % | 317 |

The eye and the ears write inside the ceiling and lose nothing. Place does not,
and what it loses is not its last quarter in general but the same clause every
time: the cut lands mid-timestamp in `, sunset 2026-09-14T` and drops
`18:48. And it has not moved more than 50 m since the last pass (2 m from it).`
That sentence is the only part of a place fragment that reports a change rather
than a state, which is the thing ROADMAP item 10 exists for, and it was the part
being discarded.

Then the cost, since the 30-tick rebuild throttle is there to contain it.
`BuildFromCorpus` over the earth corpus on cores 4-7, two builds per shape:

| shape of `docs` | lines | bytes | build 1 | build 2 |
|---|---|---|---|---|
| truncated at 240 (today) | 1101 | 158793 | 239.9 ms | 246.0 ms |
| split into sentences | 13813 | 675007 | 1076.6 ms | 1100.7 ms |
| whole lines, no ceiling | 1101 | 687953 | 1088.3 ms | 1283.8 ms |
| split, at the 8000-line cap | 8000 | 410290 | 604.0 ms | 649.1 ms |

The cost is in bytes, not in lines: splitting and raising the ceiling cost the
same 1.08 s at the same 680 KB. So the rebuild clock does not choose between the
two fixes — the byte bound does. `updateReservoirCorpus` holds the corpus file
under `MaxCorpusLines × MaxLineChars` = 8000 × 240 = 1.92 MB. Raising
`MaxLineChars` to the fragment size would raise that bound to 40 MB, and at the
measured 1.58 µs/byte that is about 63 s per rebuild against a rebuild interval
of 30 ticks ≈ 7.5 s — the throttle would stop being a throttle. Splitting leaves
`MaxLineChars` where it is and lets the line count become the binding cap
instead: at 8000 sentences the reservoir is 410 KB and rebuilds in 604-649 ms,
against 159 KB and 240-246 ms today. That is 2.5× the rebuild for 2.6× the
field, 8 % of wall instead of 3 %, and it is the change that was made —
`splitCorpusLine` in `molequla.go`, called from `dnaRead` on append, cutting at
`.`, `!` or `?` followed by a space so that `22.5 °C` and `2026-09-14T18:48`
survive, and falling back to a cut at the last space before the bound for a run
with no sentence end in it.

`corpusIngestedTotal`, the monotonic growth clock the ontogenesis gate reads,
now counts the bytes actually written as corpus lines instead of the length of
the offered fragment. Before this repair those two numbers differed by about
20×: the clock was reading 5 KB of growth for 240 bytes of field. After it they
differ only by the whitespace the cuts fall on, so the clock is measuring the
thing it is named after, and the ontogenesis thresholds keep the meaning they
were tuned with rather than gaining one.

Gates in `corpus_line_split_test.go`: the unit cases on `splitCorpusLine` (a
decimal and a timestamp are not sentence ends, a 3600 B run with no sentence end
is cut under the bound without losing content, a multi-byte run is never cut
inside a rune), and the one the audit asked for — a 5073 B fragment eaten
through `dnaRead`, then `loadCorpusLines` → `NewEvolvingTokenizer` →
`BuildFromCorpus`, with every bigram of the fragment's last sentence required to
be in `cf.BigramByFirst`. Appending the fragment whole again, it goes red at the
first step of that sentence with `docs hold 2 lines, 261 bytes, the fragment was
5073 B`.

**6. Fade and magnitude ride the heartbeat.** `model.lastGenMag`, the mean
absolute raw logit at the first step of the last generation, and
`model.lastOverlayWeight`, the overlay weight that magnitude bought, existed
only in process memory and in the `mag=` and `fade=` of the organism's own
`[dna]` line. §13 of the brief wants eligibility for sentence-boundary injection
read from the voice rather than from the stage label, and nothing outside the
organism could read the voice. `Heartbeat` now takes them as two more arguments
and writes two more columns, `gen_mag` and `overlay_fade`, added the way repair
7 added `global_step` — in the `CREATE TABLE` for a fresh mesh and by an
idempotent `ALTER TABLE ... ADD COLUMN` for one that already exists. The keeper
carries them between tick reports like the rest of the state, the witness reads
them with `COALESCE(...,0)` so an organism that has never generated reads as
zero rather than as a schema error, and the per-organism part of the witness
line gained a field: `earth:s4/4100k/0.90/12000/f1.00`.

Gates in `witness_test.go`. A fresh mesh must carry all three columns, the
values must survive the round trip, and the line must contain the fade; a
pre-repair-7 fixture — `organisms` with `element` and without `global_step`,
`gen_mag` or `overlay_fade`, holding one row — must gain all three, keep its row,
read back as `0` for a voice never reported, and take a later heartbeat
correctly. Removing the two `ALTER`s: `migration did not add "gen_mag"` with the
column list printed. Removing the columns from the `CREATE TABLE` as well: `a
fresh mesh has no "gen_mag" column`.

Live on the phone, one organism and the witness over a scratch tree with a
scratch `HOME`, cores 4-7: the organism printed `[dna] earth wrote 5003 bytes to
ecology | gen=67 mag=5.69 fade=1.00`, and the witness snapshot read `"gen_mag":
5.7641914466417274`, `"overlay_fade": 1`, `"global_step": 432` and
`"world": {"files": 1, "bytes": 130}` in the same pass — the whole chain of
repairs 1, 2, 3 and 6 in one run. The `[dna] earth consumed 127 bytes from 1
files: [world/gen_1789337521_2.txt]` line is repair 3 counting: a 130 B fragment
minus its newline is 129, and 127 is what reached the corpus as three sentences,
the two bytes being the spaces the cuts fell on.

**Two things found and not fixed.** `dnaRead` appends with `O_APPEND` and does
not check whether the corpus file ends in a newline, so the first sentence of a
fragment is glued to the last line of a corpus that does not — visible in the
probe above as `...microorganism[eye cam0 2026-09-13T22:12:01Z] A blurry...`.
That is older than these repairs and unchanged by them, one malformed line per
fragment either way. And `Heartbeat` discards the error from its `Exec`, so a
mesh whose schema is narrower than the write stops beating silently; that is how
`TestBeatKeeperRefreshesMeshWithoutTicks` went red here, 12 runs out of 12, on a
fixture that was a hand copy of the schema — the same way it went red when
repair 7 added `global_step`, as the comment in `meshForKeeperTest` records. The
fixture was widened, which is the repair-7 answer, and the deeper one — a
fixture that cannot drift, or a heartbeat that says when it failed — is left
named rather than done.

**Tests.** Before: 184 pass, 2 skip, 0 fail. After: 191 pass, 2 skip. Twenty runs
of `CGO_ENABLED=1 taskset -c 4-7 go test -count=1 -buildvcs=false ./...` on cores
4-7, 3.6-3.9 s each: 17 clean, 3 red, and every red one is
`TestBeatKeeperRefreshesMeshWithoutTicks` failing with `SQL logic error: no such
table: organisms` at `governor_phone_test.go:117` or `:127`. That is the known
flake of this tree, and its mechanism is now named rather than assumed:
`meshForKeeperTest` opens `sqlite` with the DSN `:memory:` through `database/sql`,
whose pool is free to open a second connection, and a second connection to
`:memory:` is a second, empty database. It is a different failure from the
deterministic one above, which said `heartbeat is 600.2 s old after the keeper
ran` and is gone. `phone1/launch_test.sh`: 5 pass, 0 fail. The two skips are
`TestCheckpointMemoryProfile` and `TestStage4SavePeak`, which want
`MOLEQULA_CKPT_MEASURE` and `MOLEQULA_HEAVY=1`.

Items 4, 5, 7, 8, 9 and 10 of the audit's order are untouched: the cafeteria, the
probe drawn from what was eaten, the §13 gate itself, `world_facts`, the change
emitter and the last infrastructure block. Nothing here decides who receives
what — it only makes sure that what is sent arrives, whole, and that the signals
the allocator will need are visible from outside the organism.

**The two things named above, repaired.** Both sit on the path these repairs
already changed, and both were gated before they were fixed.

`dnaRead` opens the corpus `O_APPEND` and never asked whether the file ended in
a newline, so a corpus whose last line has none took the next fragment's first
sentence onto the end of it — and with repair 3 that sentence is the one
carrying the organ's header. It now reads the last byte (`O_RDWR`, because an
`O_APPEND` handle cannot read) and closes the open line first. `saveCorpusLines`
always terminates its lines, so this is the hand-edited or truncated file rather
than the ordinary one, which is why it survived this long. The gate feeds a
corpus with no trailing newline one fragment and requires the fragment's first
sentence to be a line of its own; before the fix it names the glue it found:
`"A handful of healthy soil contains more microorganisms[eye cam0
2026-09-13T22:12:01Z] A blurry kitchen table shows a green bowl."`

`Heartbeat` discarded the error from its `Exec`. That write is the organism's
only statement that it is alive, so when a column is added to the write and not
to the schema in front of it, every beat becomes a no-op and the colony's own
governor and the witness both stop seeing an organism that is running perfectly
well — a failure whose only symptom is silence. It happened when repair 7 added
`global_step` and again here. `sayMeshError` now prints one line per distinct
error, `[ecology] mesh refused the heartbeat of earth: SQL logic error: no such
column: global_step (1) — this organism is alive and invisible to the colony`,
and not once per beat: the tick loop beats every ten ticks for the life of the
run and a line repeated that often is a line nobody reads. The gate drives the
unwidened pre-repair-7 fixture, requires the line, requires ten further beats of
the same failure to add nothing, then adds the missing columns one at a time and
requires each newly uncovered one — `gen_mag`, then `overlay_fade` — to be said
in turn, and the beat to fall silent once the schema is whole. sqlite reports
the first column it cannot find, not the one most recently added, which is why
the first line names `global_step` and not `gen_mag`.

— Defender (Arianna Method, phone-1)

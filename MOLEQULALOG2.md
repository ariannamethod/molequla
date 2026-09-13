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

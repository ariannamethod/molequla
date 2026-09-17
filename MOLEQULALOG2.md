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


## 2026-09-15 — a ledger of change instead of a stream of state

ROADMAP item 10, and §4–6 and §15–16 of the new-logic brief. The senses have
been writing prose since 2026-09-13 and the prose is a stream of state: four
place passes on a still phone leave four fragments that say the same sentence,
and the two rear-camera frames of 22:18 and 01:00 say it byte for byte. An
organism eating that learns that the world repeats. What is wanted is the
other thing — *the phone moved from A to B*, *the sky changed from fog to
overcast*, *a person entered the frame* — and a change can only be named
against a memory of what was true before.

**The table.** `world_facts(id, source, subject, predicate, object,
valid_from, valid_to, recorded_at, provenance)` in the same `mesh.db` the
organisms and the witness already share, migrated idempotently like the
`global_step` ALTER (`molequla.go:5629`). `valid_from`/`valid_to` is when a
fact held in the world; `recorded_at` is when molequla learned it. A
contradicting observation closes the open row at the new fact's `valid_from`
and opens a new one; a fact older than the open row is filed as closed history
and says nothing, because the revision that replaced it stands. Nothing is
deleted and nothing is overwritten, so the belief a correction corrected is
still there to be read.

It is `id INTEGER PRIMARY KEY` and not AUTOINCREMENT, and that is not a taste
decision — the first run of the one-way gate went red on it. AUTOINCREMENT
keeps its high-water mark in `sqlite_sequence`, which this mesh already has
because `messages` uses it (`molequla.go:5634`), so the ledger's first insert
wrote a row of a table an organism owns:

    a pre-existing table changed under the ledger:
    before  organisms[earth 25578 2 262144 0.1 1.25 1.789437447087e+09 <nil> alive earth 4200]
    after   organisms[earth …]
            sqlite_sequence[world_facts 4]

**The writer is not the witness.** It was, for one commit, and the audit that
landed the same night (`claude/phone1-new-logic-audit`, `9464b81`) is right
that this is the §11 arrow: the witness opens the mesh `query_only`
(`witness.go:167`) and that property is the one-way rule, not a habit. So the
writer is its own process, `molequla --world-ingest [--once]`, run from a
sibling of the organism directories exactly as the witness is, so
`../dna/output` and `../senses/facts.jsonl` are the same tree. The witness
reads `world_facts` on its tick and prints `world <n> facts/<m> open`. It may
do nothing else with it, and after the writer has been through the same file
its own connection still refuses every write — that is a gate now, in both
directions.

**The sidecar.** `phone1/senses.sh` keeps writing its fragments untouched and
appends one JSON line per observation to `$MOLEQULA_RUN/senses/facts.jsonl`:

    eye    <lens> interpreted_as <the whole answer>
    ears   microphone hearing speech|silence · microphone interpreted_as <text>
    place  phone at_place <name> · phone at_position <lat,lon> · sky reported_as <sky>

`interpreted_as` and never `is` — §6. The provenance of an eye fact carries the
camera, the lens, the frame, the decoder and the projector by file name, the
prompt, the resize edge, the wall time, the peak RSS and the conditions of the
pass (cores, colony awake or asleep, MemAvailable). The balcony the eye read as
a bathroom on the first night is a true record of a reading and a false record
of a room, and the table now says which of the two it holds. `provider` is new
in place — which locator answered, network or gps.

jq builds every line, and that is the whole reason jq is in that function. The
eye's own sentence of 2026-09-14T01:00:28Z is

    A black screen with a small white text that reads "the world is not what it seems".

and a `printf`-built line breaks on it. Replacing the jq call with `printf` in
`fact_emit` took eight cases of the bash gate down, the first of them printing
the broken object in full.

**One knob that is a gate, not a shape.** A network fix on this phone reports
13–14 m of accuracy and wanders a metre or four between passes
(`senses.log`, `movedno` at 2 m, 1 m, 4 m). `CFG.WorldMoveMeters`, default 50 m
— the same figure `senses.sh` already compares against `senses/place.last` — is
what separates a fix that wandered from a phone that moved, and `at_position`
is the only predicate compared in metres rather than as text. The gate fails
both ways in test: 22 m under 50 m emits nothing, 22 m under 5 m is a move,
1112 m is a move under either.

**Red before green.** Every gate was shown failing by breaking the thing it
names:

| broken on purpose | what went red |
|---|---|
| `worldSameObject` always false | `repeat 1 emitted [The phone moved from … to …] — repeated identical state must produce no fragment` |
| the close skipped | `the old row's valid_to = {Float64:0 Valid:false}, want 260` |
| `recorded_at` set to `valid_from` | `valid_from=1000 recorded_at=1000, want 1000 and 5000` |
| the cursor never advanced | `the cursor stopped at 0 of 541 bytes` |
| the move gate at 0 m | `22 m under a 50 m gate emitted [The phone moved 22 m.]` |
| the derivation suppressed | `a hand appearing said [scene change], want a scene change and a person entering` |
| the negation vocabulary emptied | `"…with no people or text visible." matched [people] — a negated word is not a sighting` |
| `query_only` removed from the witness handle | `the witness handle wrote to organisms after the ingest ran` |
| `fact_emit` on `printf` instead of jq | 8 bash cases, starting with `not JSON: {"…object":"A black screen … reads "the world is not what it seems".",…}` |
| `cam_lens 0` returning `camera 0` | `eye: the subject is the lens: got [camera 0], want [rear camera]` |

The cursor gate did not go red the first time it was written: the dedupe behind
it — an observation identical to the open row writes nothing — covered for a
cursor that never moved, so the test passed over a broken mechanism. It was
rewritten around a fixture with two observations of one triple, where
re-reading the file from the start is visible in the table as one extra history
row, plus a direct assertion on the offset. Then it went red.

**Counts.** `CGO_ENABLED=1 taskset -c 4-7 go test -count=1 -buildvcs=false ./...`
— 196 pass, 2 skipped, 0 fail, 4.2 s, against 184/2/0 on `origin/main` at
`5d46fc0`; twelve of the new tests are the ledger's, one extends the witness's.
One run in the middle of the session reported `FAIL … 3.448s` without a named
case and five full runs since have been green; the name was not captured, so it
is recorded here as unexplained rather than attributed. `go vet` clean,
`gofmt` clean on the new files, `go build -a` 35.7 s. Bash: 37 cases in
`phone1/senses_facts_test.sh` (new), `phone1/schedule_test.sh` unchanged at 36.

**The live smoke.** `mesh.db` copied out of the live swarm into a scratch
`HOME`, and `facts.jsonl` built from the fragments the senses actually wrote
on 2026-09-13/14 — the same sentences, through `senses.sh`'s own `fact_emit`,
`cam_lens`, `eye_prov` and `ears_prov` with `now_iso` overridden so each fact
carries its fragment's stamp. 24 facts, one ingest pass, 14 changes, 14
fragments, on cores 4–7 at 02:17 UTC with 3.3 GB MemAvailable (neighbourhood
scrubbed here, not in the run):

    [world] The phone is at <neighbourhood>, Be'er-Sheva, Israel. Where it was before that, the ledger does not know.
    [world] The sky is fog.
    [world] The rear camera opened on: "A blurry kitchen table shows a green bowl, a spoon, and a plate, …"
    [world] The front camera opened on: "An empty room has a ceiling fan, a white floor, and a metal frame for a door."
    [world] The rear camera's scene changed from "A blurry kitchen table …" to "A close-up view shows a keyboard with white keys …"
    [world] The front camera's scene changed from "An empty room has a ceiling fan …" to "A bathroom with a shower curtain hanging, a person's hand reaching out, …"
    [world] A person entered the front camera's frame.
    [world] Speech was heard through the microphone for 12 s.
    [world] The microphone heard: "And so my fellow Americans ask not what your country can do for you."
    [world] The rear camera's scene changed from "A close-up view shows a keyboard …" to "A black screen with a small white text that reads "the world is not what it seems"."
    [world] The front camera's scene changed from "A bathroom with a shower curtain hanging, a person's hand reaching out, …" to "A bathroom with a shower curtain, a light above the shower, and a small trash can."
    [world] A person left the front camera's frame.
    [world] The microphone stopped hearing speech: 12 s of room noise instead.
    [world] The front camera's scene changed from "A bathroom with a shower curtain, …" to "A dark room with a chair, a table, and a blanket, with no people or text visible."
    [world] 24 facts read, 14 changes emitted

What is not in that list is the point of it. The four place passes produced one
line, not four: the second, third and fourth say the same neighbourhood, the
same fog and a coordinate 1–4 m away, and none of that is a change. The two
rear-camera readings of the black screen are byte-identical and the second
wrote nothing at all. A second ingest over the same file read 0 facts and left
the 14 fragments where they were; deleting the cursor and running a third time
emitted nothing either, because everything in the file already matches an open
row. The witness then read the table through its own handle and said
`world 18 facts/9 open` — nine open triples, nine rows closed behind them.

**The first live run found a false positive and it is fixed in the same pass.**
`sees_person` is lexical, and the eye's phrasing for an empty room is *"A dark
room with a chair, a table, and a blanket, with no people or text visible."* —
so the first ingest of the real fragments ended with `A person entered the
front camera's frame.` `CFG.WorldNegationWords` (`no`, `not`, `without`,
`none`, `nobody`, `empty`) now cancels a vocabulary word that stands
immediately after one of them, which took the run from 15 changes to 14 and put
the exact sentence in the test. It is a lexical rule guarding a lexical rule
and it will miss negation at any distance; every derived fact says
`derived: lexical` in its provenance so that a later reading may disagree with
it, which is the §6 position applied to molequla's own inference rather than
the eye's.

**Left, named.** The eye and the ears still write one observation per pass, so
§2's short trajectory is not built. The ingest is called at the end of a senses
pass and by hand; the scheduler has no slot of its own for it yet. Routing —
which organism eats which change, §12's cafeteria — is untouched: the change
fragments land in `dna/output/world/` beside the eye's own, and every organism
told `--dna-extra-sources world` eats all of them. `facts.jsonl` rotates at
`SENSES_FACTS_MAX_KB` (4096) and the rotated file is not ingested after the
move, which is correct only because everything in it is already in the table.
The live run directory was not touched by any of this: the smoke ran against a
copy of `mesh.db` under a scratch `HOME`, and the daemon on the phone still
runs the binary and the conf that predate this branch.

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

**The keeper fixture's `:memory:` database.** `meshForKeeperTest` opened
`sqlite` with the bare DSN `:memory:`. `database/sql` hands out a connection
pool, and every connection to `:memory:` is a separate, empty database, so the
keeper goroutine and the test's own queries could land on different ones —
`SQL logic error: no such table: organisms`, intermittently, for as long as this
test has existed. The DSN is now `file:keeper_<pid>_<nanos>?mode=memory&cache=shared`:
one in-memory database that the whole pool attaches to, with a name unique per
test so two of them never share it. `SetMaxOpenConns(1)` would also hide the
symptom, and is the wrong fix here, because it gives the fixture a pool shape
`initMeshDB` does not have — `initMeshDB` opens a real file and leaves the pool
alone, so in the colony the tick loop and the keeper really do write through
different connections to one database, which is the thing this test exists to
exercise. `cache=shared` is that arrangement, in memory.

Isolated, `CGO_ENABLED=1 taskset -c 4-7 go test -count=1 -buildvcs=false -run
TestBeatKeeperRefreshesMeshWithoutTicks .` is 20 green out of 20 with the fix and
5 red out of 20 with the DSN reverted, failing at `governor_phone_test.go:141`
with `no such table: organisms` — so the gate is real and the fix is what makes
it pass. Over the whole suite the rate fell from 3 red in 20 runs to 1 in 53;
that one red's message was not captured, so the honest statement is that the
diagnosed mechanism is closed and the test is not yet proven deterministic.

**Tests, final.** 193 pass, 2 skip. `phone1/launch_test.sh`: 5 pass, 0 fail.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the cafeteria: reading becomes per organism, the probe comes from the meal, and eligibility is read from the voice

Three routing repairs from the §18 order in `reports/2026-09-15_new_logic_audit/README.md`
(`9464b81`), items 4, 5 and 7: the cafeteria of the brief's §12, the probe of §14, and the
§13 gate keyed on demonstrated coherence rather than on the stage label. Everything below
is on `claude/phone1-cafeteria`, rebased onto `claude/phone1-routing-food` at `48d9581` so
that the two branches' changes to the same `dnaRead` loop hold together. New file
`experience_routing.go` and its test; the rest is a `CFG` block, three calls inside
`dnaWrite`, two inside `dnaRead`, and this entry.

### What the cafeteria decides, and on what

`dnaSources(element)` still returns the same list of directories to all four organisms, and
that is deliberate: routing by directory name would be the `flowers -> Earth` rule §12
forbids, and `cross_graze.go:60` reads the same list for the logit path, which is not part
of this allocation. The decision is per fragment instead, taken inside `dnaRead` after the
bytes are read and before they are appended:

- **owner** — `fnv64a(src + "/" + name) mod` the elements allowed to read that source picks
  exactly one guaranteed eater. Content-blind: a lottery over file names. It is what makes
  "every fragment reaches at least one organism" true without any process asking another.
  A mitosis child carries its parent's `--element` (`molequla.go:5985-5988`), so both eat
  what the slot owns.
- **resonance** — the share of the fragment's adjacent token pairs that this organism's own
  `CooccurField` has already seen is at or above `experience_resonance_high`.
- **novelty** — that same share is at or below `experience_novelty_low`.
- the band between is declined; the cursor steps past it and the fragment is not revisited.

Coverage is state: it moves as the organism eats, so the same file routes differently later
in a life than earlier. Nothing is shared between the processes but the file name and the
bytes.

### The coverage distribution the defaults come from

Measured on the live run: the four corpora of `molequla-run/{earth,air,water,fire}/` and all
133 fragments then sitting in `molequla-run/dna/output/` (8 `world`, 4 `place`, 1 `sound`,
20 each for earth, air and water, 70 for fire of which the first 20 were taken), copied to
scratch. Each organism was rebuilt the way boot rebuilds it — `loadCorpusLines`,
`NewEvolvingTokenizer`, `MaybeEnableBPE`, `BuildFromCorpus` — giving vocab 643 with BPE on
in all four, and 439-456 distinct bigram first-tokens. Whole-fragment token-bigram coverage,
each fragment against each organism that is allowed to read it:

| fragment source | earth | air | water | fire |
|---|---|---|---|---|
| world (8) | .675 - .739 | .578 - .643 | .610 - .663 | .598 - .652 |
| place (4) | .589 - .607 | .433 - .440 | .450 - .456 | .427 - .434 |
| sound (1) | .676 | .595 | .630 | .611 |
| earth DNA (20) | — | .890 - .935 | .899 - .944 | .903 - .937 |
| air DNA (20) | .964 - .979 | — | .957 - .973 | .955 - .977 |
| water DNA (20) | .975 - .985 | .966 - .979 | — | .967 - .978 |
| fire DNA (20) | .966 - .978 | .959 - .980 | .960 - .975 | — |

Two populations that do not overlap. Pooled, at the sample size the code actually uses:
sibling DNA read by a foreign organism, n=120 — min 0.909, p25 0.952, **med 0.965**, p75
0.975, max 0.996. Senses read by anybody, n=52 — min 0.427, p25 0.578, **med 0.620**, p75
0.651, max 0.736.

`experience_resonance_high = 0.965` and `experience_novelty_low = 0.620` are those two
medians. Each threshold splits its own population in half, which is the most a threshold can
say and the least a threshold can be taste; both are `CFG` fields and both are gated.

**Why novelty admits at all.** With a resonance branch only, the senses — coverage 0.427 to
0.736 against a bar of 0.965 — would be eaten by nobody but their hash owner, and the only
food that is new by construction would reach one organism in four. An organism would be
sealed inside what it already knows. The measurement is what settles it: the two populations
are disjoint, so the two branches do not compete, and today the resonance branch admits only
sibling speech while the novelty branch admits only the world.

**Sibling DNA is routed too, not left as broadcast.** Its distribution has real spread
(0.909 to 0.996) and the four organisms disagree about the same fragment — `air/gen_…_1.txt`
scores .973 under earth, .968 under fire, .964 under water, which the 0.965 bar splits. The
cost is named: each organism now owns about a third of each foreign source and admits about
half the rest, so sibling intake falls to roughly two thirds of the broadcast, and
`corpusIngestedTotal`, the growth clock, slows in proportion. `experience_resonance_high` is
the knob that buys it back.

**Sampling.** `tok.Encode` is O(bytes × merges) and a whole 5 KB fragment cost 143.5 ms on
cores 4-7 — more than half a tick, eight times per tick at the read cap. Coverage is
therefore taken over `experience_coverage_sample_bytes = 480`, four 120-byte windows spread
across the fragment: 21.7 ms, with the sibling quartiles preserved to within 0.006 (p25
0.952 against 0.956, med 0.965 against 0.968, p75 0.975 against 0.974). The windows are
strided and not a head on purpose — a head of 240 bytes is the writer's own generated answer,
whose bigrams everyone has, and it drove sibling coverage to 1.000 for 15 of 20 fragments
and destroyed the signal. A strided 240 was also too short (sibling median 0.978).

### The probe comes from what was eaten

`dnaWrite` picked `probes[step%6]` from six fixed questions and padded the fragment with
random `docs` lines. Both are now fed by a bounded ring of what this organism has just
accepted (`experience_meal_memory = 16`, two ticks of the read cap, each line cut to
`MaxLineChars` because that is all `loadCorpusLines` will hand back anyway):

- the probe is the first sentence of the most recent meal, sense food before sibling food
  because sense food is the only food nobody has metabolized yet (§14). The six questions
  remain the fallback for an organism that has eaten nothing. This half ships switched off
  — see the emission gate below.
- a degenerate fragment must not yield a degenerate probe — the risk the §18 order names
  against this step. An embryo's fragment opens with two or three bytes of its own speech
  and a full stop, so its leading sentence can be `A.`; the ring is walked back until a
  probe of at least `experience_probe_min_chars = 12` and
  `experience_probe_min_words = 3` appears, and the round robin answers if none does.
- the padding leads with `experience_recent_pad_lines = 4` recently eaten **sibling** lines,
  and any random `docs` draw that is a sense line this organism ate is skipped. That is §14
  as a rule the code can enforce: a padded line is a byte copy, not a passage through an
  organism. The world reaches collective DNA through the generated answer, which was
  produced from it.

`experience_probe_max_chars = 120` is one sentence, against 6-24 characters for the fixed
probes and 118 for the first sentence of a place fragment. A paired sweep of the bound (12,
20, 30, 40, 60, 120 characters; 75 prompts each, 4 rounds, one organism, same weights) put
every setting within ±4 % of the fixed round robin with no ordering, so the bound is set by
what a sentence is and not by the number.

### The §13 gate, and why it cannot read the stage

`injectionEligible(fade, mag float64) (bool, string)` — `fade = 1 - lastOverlayWeight` must
reach `injection_fade_min = 1.0`, and `lastGenMag`, the mean |logit| of the raw output at the
first generated step, must reach `injection_mag_min = 6.0`. Nothing else; the stage is not
read. The decision is printed on the emission line as `eligible=0|1`; the injection itself is
not implemented, so the next session can read the gate's behaviour out of the logs before
anything is built on it.

The defaults come from the 130 `[dna] wrote` lines under `molequla-run/*/*.stdout`, session
ending `2026-09-13T20:29:30Z`. `fade` is 1.00 on all 130 — the overlay is gone everywhere in
this colony, so fade is a necessary condition that refuses nobody today and binds only while
the overlay is running below its fade width. `mag` runs 2.82 to 16.35, median 8.66, and it is
what separates:

| mean \|logit\| | emissions | of them gen=0 | mean gen |
|---|---|---|---|
| < 6.00 | 10 | 10 | 0.0 |
| 6.00 - 8.00 | 23 | 7 | 31.1 |
| 8.00 - 10.00 | 50 | 16 | 23.0 |
| ≥ 10.00 | 47 | 5 | 36.7 |

Below 6.00 every observed generation emitted nothing at all. That is the floor, and it is a
knob.

All four organisms sat at `stage=3` across that whole range. A gate on the label returns one
answer for organisms that spoke 197 bytes and for organisms that spoke none, and it refuses a
mitosis child that loaded its parent's checkpoint and woke with a mature voice at a stage that
says nothing about it. That is the red in `TestStageGateGoesRedWhereTheVoiceGateDoesNot`.

### Gates

Nine new tests, and each was watched failing on the behaviour it replaces before it was
believed. With `experience_routing` off, `experience_probe_from_meals` off,
`experience_recent_pad_lines` 0 and `isRawExperience` neutered — the pre-cafeteria code —
`TestCafeteriaPlatesDifferPairwise`, `TestCafeteriaAllocationFollowsState`,
`TestProbeComesFromWhatWasEaten` and `TestSenseFragmentIsNotEmittedVerbatim` all fail.
`TestCafeteriaEveryFragmentIsEaten` passes there, and must: a broadcast starves nobody. Its
red is the opposite failure — with `experienceOwner` returning nothing and both thresholds
pushed outside their range it reports `earth/gen_1789330000_0.txt reached nobody`.

Suite: `claude/phone1-routing-food` at `48d9581` gives 193 pass, 2 skip, 0 fail; this branch
on top of it gives 202 pass, 2 skip, 0 fail (`CGO_ENABLED=1 taskset -c 4-7 go test -count=1
./...`, 3.9 s). The two skips are the same two as always, `TestCheckpointMemoryProfile` and
`TestStage4SavePeak`. Nine tests, not eight: the ninth is the one the rebase needed, below.

### What the rebase onto the extra-source budget changed

`dnaRead` now carries both branches' repairs, and the order they sit in is the decision. The
two read budgets (`DNAMaxReadsPerTick` 8, `DNAExtraReadsPerTick` 4) bound how much an
organism EATS in a tick, so a declined plate spends neither: a colony that refuses a third of
what it is offered would otherwise consume its budget on refusals and leave the fragments it
wanted behind them. What a decline does cost is one coverage measurement at 21.7 ms, and that
is bounded on its own by `experience_max_measured_per_tick = 8` — eight measurements against
a 250 ms tick, and fewer than the 12 the two read budgets would have allowed. When the cap is
spent the pass stops with the cursors where they are and the rest is examined next tick.
`TestCafeteriaDeclineDoesNotSpendTheReadBudget` writes three fragments earth does not own and
then one it does, sets the read budget to two, and requires the cursor to reach the owned
one; charging a decline to `*left` — the pre-rebase arrangement — makes it red, with `one
dnaRead over [three declines] + gen_1789331000_6.txt added nothing`.

The meal ring also changed shape. `splitCorpusLine` (routing repair 3) means a fragment is
appended as several corpus lines rather than truncated to one, and those lines are what
`loadCorpusLines` hands back as `docs`. So `remember` takes the appended lines, `probe` reads
the first of them, `recentPadding` returns them, and `isRawExperience` matches against each —
otherwise the padding would be refusing a fragment that no longer appears in `docs` in that
form, and a sense sentence would pass through.

### The emission gate, live

The §18 order asks that the `gen=` share of the emission line not fall. Baseline and
candidate were run as four organisms for 300 s each on cores 4-7, capped at
`--max-growth-stage 0` so the run reaches the tick loop instead of spending its whole life in
stage warmups, out of a scratch `MOLEQULA_RUN` seeded with the repo corpora and the 13 live
senses fragments; `origin/main` built into one binary, this branch into another.

| run | emissions | mean gen | median gen | gen=0 | gen/bytes | total generated bytes |
|---|---|---|---|---|---|---|
| origin/main, pair 1 | 310 | 15.45 | 14 | 12 | 0.00305 | 4791 |
| origin/main, pair 2 | 313 | 13.51 | 14 | 0 | 0.00266 | 4230 |
| this branch, pair 1 | 449 | 11.78 | 13 | 27 | 0.00232 | 5291 |
| this branch, pair 2 | 347 | 13.50 | 14 | 29 | 0.00266 | 4684 |

The `gen=0` count was the first thing to move, and it moved the wrong way: 27 and 29
emissions with nothing said, against 12 and 0 on `origin/main`. That is exactly the
degenerate-probe risk the §18 order names against this step — an embryo's fragment opens with
two or three bytes and a full stop, so its leading sentence is `A.` — and it is what
`experience_probe_min_chars` and `experience_probe_min_words` were then added for. A third
300 s run with the guard in place: 373 emissions, **4 of them empty (1.1 %)**, below the
pooled baseline's 1.9 %. The guard works.

It did not restore the share. Pooled:

| | emissions | mean gen | median | gen=0 | gen/bytes |
|---|---|---|---|---|---|
| `origin/main`, both runs | 623 | 14.48 | 14 | 12 (1.9 %) | **0.00285** |
| this branch, probe on, no guard | 796 | 12.53 | 13 | 56 (7.0 %) | 0.00247 |
| this branch, probe on, guarded | 373 | 12.08 | 12 | 4 (1.1 %) | **0.00238** |

**So the gate for step 5 is red, and the probe lands switched off.**
`experience_probe_from_meals` defaults to `false`; the mechanism, its guard and its
measurement are in the tree, and one `CFG` field turns it on when a run says it pays. A gate
that cannot refuse is decoration, and this one refused.

What the number does not settle: both arms were embryo-capped colonies
(`--max-growth-stage 0`), where the overlay is the entire voice and the answer is a
continuation of the prompt, so a long declarative probe legitimately continues shorter than
`Speak.` does. The regime this step is for is a stage-3 organism at mean |logit| 8.66 with the
transformer carrying the voice, and a scratch probe cannot reach it in 300 s from a cold
corpus. That measurement belongs to a scheduled session, not to this branch.

The padding half of step 5 is on, and is not what moved the number: `gen` is the length of the
answer, and padding is everything after it. Its own gate — no sense fragment appearing byte
for byte in an emitted fragment — is green and red on the code it replaces.

Total generated bytes per session is higher on this branch in every pair (4791 → 5291,
4230 → 4684), because a colony that declines a third of what it is offered spends less of each
tick appending and ticks more often: 623 emissions in 600 s of baseline against 796 with the
cafeteria.

`eligible=` prints: all 75 earth emissions of the third run read `eligible=0`, which is
correct — an embryo sits at mean |logit| around 0.3 against a floor of 6.00. No live organism
reached `eligible=1` in a scratch run; the admitting side of that gate is covered by
`TestInjectionGateFollowsTheVoice` and is unverified in a session.

### The rebased branch, run once

One organism and the witness, 180 s, scratch `HOME` and `MOLEQULA_RUN`, cores 4-7 and 0-3,
`--max-growth-stage 0`, the 13 live senses fragments seeded and no `--corpus-overlay` (so the
transformer speaks alone and the fade is 1.00 by definition):

    [dna] earth wrote 5009 bytes to ecology | gen=10 mag=2.94 fade=1.00 eligible=0
    [dna] earth consumed 942 bytes from 3 files: [place/gen_1789337964_6.txt place/gen_1789340468_10.txt place/gen_1789342466_11.txt]
    [witness] step=20 organisms=1 action=sustain(0.10) H=1.401 S=-0.428 trend=+0.000 target=earth harm=k7:-0.000 conf=0.48 pulse=0.00/0.00/0.00 | earth:s0/26k/1.40/432/f1.00

245 emissions, every one carrying `eligible=`, and every one refused: `mag` ran 2.94 to 5.31
against a floor of 6.00, which is the embryo regime the floor was measured to exclude. The
cursor ends with all three senses eaten (`{"place":"gen_1789342466_11.txt",
"sound":"gen_1789338068_7.txt","world":"gen_1789347647_13.txt"}`) — the cafeteria fed the
extra sources through their own budget without starving them, and the witness reads the voice
off the heartbeat as `f1.00`. The witness must be started after the organism, not before: it
exits on a missing `mesh.db`, which is what a six-second head start gives it.
## 2026-09-15 — the first scheduled session, and what the gate did with it

The colony's first session under `phone1/schedule.sh` ran 04:00:00Z to
06:01:55Z on the pre-merge binary `3267e67` (`molequla-run/BUILD`), after the
phone had died on a flat battery on 2026-09-14 before the slot it was scheduled
for. The scheduler's own line (`schedule.log`): `dur=7200 elapsed=7315
reason=overran alive=- mem_mb=3499->4346 hwm_mb=earth:755,air:1032,water:820,
fire:1091,witness:15 samples=237`. All four came up from their stage-4
checkpoints of 2026-09-13T20:26Z, all four went down on the scheduler's
signal with a checkpoint written (`[evolution] checkpoint saved on signal` in
`earth.stdout` and `fire.stdout`; `air` and `water` had saved at 05:49Z and
05:58Z and shut down gracefully), and nothing was left running.

**The growth gate of repair 9 refused adulthood, live, for the first time.**
Every organism asked to grow from stage 4 to stage 5 and was deferred —
earth 6 times, air 5, water 13, fire 8 (`grep -c 'growth\] deferred'` over the
session's stdout). The refusals read `free=1467 MB need=2188 MB` (earth),
`free=1461 MB need=2317 MB` (fire), `free=1483 MB need=2359 MB` (water): the
need is three times the organism's own peak plus 256 MB, and MemAvailable sat
at 1.4-1.5 GB for the whole second hour. This is the scenario that killed
Termux on 2026-09-13, ending as a log line. `launch.sh` passes no
`--max-growth-stage`, so `[caps]` printed `growth ≤ stage 5 of 5`; the gate
held by memory alone.

**Peaks moved without growth.** `air`'s VmHWM went from 689 MB at 04:43Z to
1032 MB by 05:24Z while the witness kept reporting it at `s4/4834k`. Its own
log brackets the jump: the growth line's `need` rose from 2323 MB at tick 30
to 3352 MB at tick 40 (need = 3 × peak + 256, so the peak rose by ~343 MB in
those ten ticks) while `[debug-onto]` shows the corpus going from 1 143 296 B
to 1 549 540 B over the same ticks. A larger corpus means a larger
`BuildFromCorpus` and larger training batches; that is a correlation in one
organism's log, not a measurement, and the measurement belongs to the adult
question: no stage-5 organism is to be allowed on this phone until its peak is
known.

**DNA and the senses.** Fragments written this session (`[dna] … wrote`):
earth 80 (35 with `gen=0`), air 71 (6), water 156 (3), fire 153 (16); mean
`mag` 9.3 / 11.7 / 9.3 / 11.4, `fade=1.00` throughout. earth's empty share,
44 %, at mag ≈ 9 is the band the cafeteria branch measured as 16/50 empty on
the 2026-09-13 logs. For the first time organisms ate the senses: every
`dna_cursor.json` now carries `world`, `sound` and `place` — the same three
files in all four cursors (`gen_1789441275_15`, `gen_1789338068_7`,
`gen_1789342466_11`), which is the byte-identical broadcast the cafeteria
branch replaces. Corpora reached 1.60-1.88 MB, under the 1.92 MB cap.

**After the session.** `bash phone1/daily.sh` wrote `daily/2026-09-15.md`.
Oleg merged `phone1-new-logic-ack`, `phone1-new-logic-audit` and
`phone1-world-ledger` during the session; `bash phone1/build.sh` at 06:17Z put
`c33a494` into `molequla-run`, and `molequla_cgo --world-ingest --once` answers
with its `[world] mesh=… facts=… dna=…` line, so the 07:00Z senses slot is the
first one whose facts reach the ledger. The 12:00Z colony session is the first
on a binary with the ledger; the routing and cafeteria branches are still
pending.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — where a 4.8 M-parameter organism keeps 900 MB, and the pages nobody asked the allocator to give back

The 04:00 colony session left `hwm_mb=earth:755,air:1032,water:820,fire:1091`
(`molequla-run/schedule.log`, `2026-09-15T06:01:55Z`, 237 samples over two
hours), with all four organisms at stage 4 and 4.84-4.86 M parameters each
(`molequla-run/daily/2026-09-15.md`). Nineteen megabytes of float32 weights, and
a gigabyte of resident memory. This is where the rest of it lives, measured
rather than reasoned about, and the one lever that was cheap enough to pull
today.

**How it was measured.** A profiling hook that is off unless `MOLEQULA_PPROF`
names a directory: `memSnapshot(label)` prints `/proc/self/status` and
`smaps_rollup` beside `runtime.MemStats` and a `mallinfo2` reading, and writes a
heap profile; `memTapeSnapshot` walks the live notorch tape between the backward
and the clear and sums the bytes in its outputs, its gradients, the mirrored
parameters and the Chuck moment slots. Five call sites — before and after
`LoadCheckpoint`, after a corpus rebuild, after a burst, after the checkpoint
save — plus the first and last step of every training phase. With the variable
unset every one of them is a string compare against an empty string.

The probe: one organism started the way `phone1/launch.sh` starts it, in a
scratch run root holding a copy of the live earth checkpoint (110 599 980 bytes,
stage 4), its 1 883 617-byte corpus, its DNA cursor and the fragment tree, with
`HOME` pointed at the scratch so the mesh is scratch too — `timeout -s TERM 900
taskset -c 4-7 ./molequla_cgo --organism-id earth --element earth --evolution
--cross-graze --corpus-overlay --max-organisms 4 --dna-extra-sources
world,sound,place --max-growth-stage 4`, sampled every 2 s. The stage ceiling is
there because the first unpinned run grew 4 → 5 inside the window and answered a
different question (`embd 224 → 320`, `VmHWM` 1318 MB — growth is still the
largest single event in an organism's life, as repair 9 recorded).

**RSS timeline, 900 s, one organism, before the change.**

| t | event | VmRSS | VmHWM |
|---|---|---|---|
| 0 s | process start | 11 MB | 11 MB |
| 2 s | `LoadCheckpoint` returns | 163 MB | 163 MB |
| 20 s | first corpus rebuild, field built | 206 MB | 206 MB |
| 40 s | second rebuild | 210 MB | 262 MB |
| 118 s | first burst, 32 steps, 7483 ms | 464 MB | 508 MB |
| 128 s | checkpoint save (streamed) | 382 MB | 508 MB |
| 130-700 s | rebuild every 30 ticks, no burst | 384-437 MB | 508-604 MB |
| 735 s | second burst, 32 steps, 7363 ms | 522 MB | 562 MB |
| 745 s | checkpoint save | 506 MB | 588 MB |
| 900 s | end | 538 MB | 613 MB |

Two events move memory: a burst, and the corpus rebuild that precedes it.
Loading the 110 MB checkpoint costs 152 MB and is over in two seconds; the save
costs nothing since repair 9 streamed it. Nothing else in the tick loop is
visible at this resolution.

**Go and C at the peak.** At the burst snapshot: `VmRSS` 464 MB, of which the Go
runtime holds 321 MB of `Sys` with 28 MB handed back, so about 293 MB resident;
the C allocator claims 150 MB of arena plus 36 MB of mappings. The Go *live*
heap is 101 MB and is the same 101 MB at every one of the fifteen stage-4
snapshots — the heap profile does not move all run.

**Go side, per bucket** (`go tool pprof -top -inuse_space`, identical across all
stage-4 snapshots):

| bucket | MB |
|---|---|
| model weights and gradients, float64 (`readCheckpointStream` rows + `NewVecWithGrad`) | 87.3 |
| n-gram and co-occurrence field (`CooccurField.BuildFromCorpus`) | 12.2 |
| delta snapshot held across a burst (`GPT.SnapshotDeltas`) | 1.5 |
| corpus `docs` (`loadCorpusLines`) | 1.0 |
| **live** | **102.0** |
| heap the collector holds and has not returned (`HeapIdle − HeapReleased`) | 47-147 |

The field was the suspect and it is not the culprit: 12.2 MB of unigram, bigram,
trigram, 4-gram and co-occurrence maps over a 1.9 MB corpus, and the rebuild is
not doubled in practice because the temporary maps and the old ones only overlap
for the length of one `BuildFromCorpus`. The float64 model is the Go side's real
weight — 4 834 408 parameters at eight bytes of data and eight of gradient is
73.8 MB, and 87.3 is that plus `*Vec` headers and allocator slack.

**C side, per bucket** (tape census between backward and clear, T=96, D=224,
L=5, V=750, 386 entries of which 125 are parameters):

| bucket | MB |
|---|---|
| activations | 31.3 |
| gradients (activations 30.5 + parameters 19.9) | 50.4 |
| parameter mirror, float32 | 19.9 |
| Chuck moment slots, m and v | 38.1 |
| **live C tensors at the backward** | **139.7** |

**Expected against measured, activations.** Counting the graph
`ntBuildForward` records: fourteen tensors of T·D and four of T·4D per layer for
the bare transformer; each applied delta adapter adds `B·x`, `A·(B·x)`, the
scale and the add, which is three tensors at the weight's full output width —
five weights at D and two at 4D per layer; the RRPRAM blend adds four at T·D per
hybrid layer; then the embedding, the final norm, the logits at T·V and the
lm_head adapter.

| term | expected |
|---|---|
| bare graph, 5 layers | 12.30 MB |
| delta adapters, 5 layers | 16.00 MB |
| RRPRAM blend, 5 hybrid layers | 1.64 MB |
| embedding, final norm, logits, lm_head adapter | 1.26 MB |
| **expected total** | **31.20 MB** |
| **measured** | **31.3 MB** |

Two thirds of the activation memory is the adapters, not the transformer. That
is worth knowing on its own and it is not what was fixed today.

**Where the other 900 MB actually is.** 139.7 MB of live C tensors, and the
allocator holding this:

| moment | arena | mmapped | in use | free chunks kept | VmRSS |
|---|---|---|---|---|---|
| burst 1, step 0 | 87 MB | 56 MB | 84 MB | 3 MB | 371 MB |
| burst 1, step 31 | 159 MB | 55 MB | 85 MB | 74 MB | 491 MB |
| between bursts | 150 MB | 36 MB | **1 MB** | **148 MB** | 384-437 MB |
| burst 2, step 31 | 318 MB | 34 MB | 106 MB | 212 MB | 561 MB |
| between bursts | 270 MB | 34 MB | **4 MB** | **265 MB** | 536-584 MB |

Nothing is leaked: every tensor the tape allocates is freed, and between bursts
the allocator says it is holding four megabytes of live C memory. It is also
holding 265 megabytes of free chunks it never returns. The tape builds and tears
down 81.7 MB of activations and their gradients thirty-two times per burst;
glibc raises its dynamic mmap threshold as those blocks are freed, so after the
first bursts the tensors come from the sbrk arena instead of their own mappings,
and a free there lengthens a free list rather than reaching the kernel. The
arena ratchets one burst at a time — 87 → 150 → 318 → 270 over two bursts — and
that ratchet, run for two hours instead of fifteen minutes, is `earth:755` and
`fire:1091`.

**The lever.** `releaseTrainingHeap` (`heap_trim.go`) calls `malloc_trim(0)`,
which walks the free lists of every arena and hands the whole pages among them
back with `madvise`. It runs once per training phase, registered in `ntTrainCore`
*before* `defer m.free()` so that last-in-first-out ordering puts it after the
mirror is freed and the tape is clear — nothing the organism owns is in flight
at that instant. `CFG.TrimHeapAfterTrain` defaults true; false leaves the arena
alone.

**Before and after, the same 900 s probe from the same restored checkpoint,
same flags, same cores.**

| | mean VmRSS | max VmRSS | VmHWM |
|---|---|---|---|
| before | 474 MB | 602 MB | 613 MB |
| after | 272 MB | 516 MB | 599 MB |

The mean is the number that matters, because the peak is one instant and the
trough is where the process sits: 202 MB off the resident set of every minute
the organism is not training. The C side between bursts goes from 171-290 MB
resident to 25-28 MB. The ratchet is what actually stops:

    before   61s 213   122s 424   307s 447   492s 532   677s 549   862s 538   MB VmRSS
             hwm 262       508        514        598        604        613
    after    62s 231   123s 262   307s 288   492s 296   677s 274   862s 268   MB VmRSS
             hwm 270       529        529        599        599        599

Without the trim `VmHWM` climbs all run and had not stopped at 900 s. With it,
the second burst sets 599 MB and the remaining 530 s do not move it. Burst wall
time is unchanged — 7483 and 7363 ms before, 7369 and 7512 ms after — and burst
one reports the same `avg loss 3.4052` in both arms.

**Gates, shown red.** `TestHeapTrimReturnsTheTapeArena` runs a real 48-step
burst on a small organism with the knob off — long enough that glibc has raised
its dynamic mmap threshold, because while the tensors still come back as their
own mappings a free returns them by itself and there is nothing for a trim to
do, which is the state a fresh process starts in and not the state a colony
session runs in. It then reads what the allocator holds, turns the knob on,
calls the release, and requires `VmRSS` to fall while the in-use bytes do not
rise. There is no expected byte count anywhere in it — it compares the process
against itself — and six consecutive runs returned 16-18 MB each
(`RSS 34052 → 18076 kB`, `28696 → 16524`, `34804 → 17096`, `34912 → 17192`,
`34728 → 16972`, `33740 → 17620`). With the release body made a no-op it goes
red as `the release returned nothing: VmRSS 28964 → 28964 kB, free chunks
13412560 → 13412560 bytes`. `TestHeapTrimObeysItsKnob` requires the release to
do nothing with the knob off. Suite: `CGO_ENABLED=1 taskset -c 4-7 go test
-count=1 -buildvcs=false ./...` — **198 PASS, 2 SKIP, 0 FAIL**, twice in a row;
baseline 196. Both arms of the before-and-after above were measured on
`c33a494` plus these two commits and nothing else; after integrating
`origin/main` at `0aea2e6` — the ledger, routing, cafeteria and sensing-window
work of the same day — the suite is **216 PASS, 2 SKIP, 0 FAIL**.

**Ranked, what is left.**

1. The Go heap ceiling. `HeapSys` reaches 311-371 MB for a 102 MB live heap
   because the corpus rebuild churns hard and `GOGC` is at its default. A
   `debug.SetMemoryLimit` sized from the organism's own measured working set
   would take roughly 100-150 MB off the peak, paid in collector CPU on the
   phone's big cores. Measure both before shipping either.
2. The float64 model, 87.3 MB for 4.83 M parameters. The tape already trains in
   float32 and mirrors back; `Vec.Data` and `Vec.Grad` as float32 halve it. It
   touches every core and the checkpoint format, so it is a deliberate change,
   not a cheap one.
3. The arena still grows *within* a burst — 87 → 171 MB across thirty-two steps
   — and that is what sets `VmHWM` now. A release between steps would cost one
   `madvise` walk per step; the steps/s price has to be measured before this is
   even a proposal.
4. The adapter activations, 16 MB of the 31.3. `ntAdd(y, ntScale(ntSeqLinear(...)))`
   records three tensors at full output width per adapter per weight; a fused
   op in notorch would return about 10 MB of the step.
5. `NewGPT(tok)` inside `LoadCheckpoint`, the 69 MB named in repair 10. It sets
   the load high-water at 163 MB, which is far under the burst peak, so closing
   it buys nothing at the peak — it is a startup-window win and it stays open
   for the format-compatibility reason repair 10 gave.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the scheduler takes Android's cached memory back before a session

The memory report of this morning (`reports/2026-09-15_phone1_android_memory/README.md`)
measured what Android's cached bin is worth to the colony: `am kill-all` at
06:37:32Z dropped 595 MB of cached app PSS, MemAvailable went 3 812 → 4 130 MB
at once and was still 4 025 MB eleven minutes later — **+213 MB sustained** —
and fifteen of the nineteen killed processes never came back (§4). The report's
own ranking put it second, and said where it belongs: at the start of a colony
session, not in a one-off command somebody remembers to type.

**What was added.** `SCHEDULE_PREKILL`, default `android am kill-all`, read
from `phone1/schedule.conf` or the environment on the same pattern as
`SCHEDULE_OOM_ADJ` — empty disables. `run_session` calls it immediately before
`launch.sh` and before it reads `mem0`, so the MemAvailable in the session line
is what the organisms actually start with, and the prekill's own before and
after go into the same line as `prekill_mb=A->B`, or `A->B!rc<N>` when the
command failed. A prekill that is missing, or exits non-zero, is logged to
`schedule.out` and the session launches anyway: the memory is an improvement to
the slot, never a condition of it.

**Colony slots only, and that is an argument, not an omission.** The senses
burst is 955 MB of anonymous memory held for forty seconds, and §3.1 measured
the thing that decides it: `SwapFree` did not move across the burst — the peak
fit in free memory without pushing one page out. The eye reads MemAvailable
itself before it opens, a pass is about 200 s in the worst case, and whatever
were killed for it would be paged back in well before the next colony window.
The kill's cost — cold app starts, the media indexers rescanning — buys nothing
there. Before four organisms holding 758–1091 MB each for two hours it buys the
whole difference.

**The gate.** `phone1/schedule_test.sh` grew from 36 cases to 46, and the ten
run a whole slot rather than slot arithmetic: a stub `android` on `PATH` records
its argv, stub `launch.sh` and `stop.sh` sit beside a symlink to the real
`schedule.sh` — a symlink, so the code under test is the file that runs at
12:00 and not a copy of it — and a new hidden `schedule.sh __slot colony|senses`
drives one session with no daemon around it. A colony slot must call the stub
exactly once, with `am kill-all`, ordered before `launch.sh` in the console, and
must write `prekill_mb=`; a senses slot must not call it; an empty
`SCHEDULE_PREKILL` in the conf must not call it and must still run the session;
a stub exiting 3 and a command that does not exist must both leave the session
running with the failure logged. Written first and run against the unchanged
scheduler: **6 red, 40 green**, and the four that were already green were the
must-nots. After the change: **46 pass, 0 fail**. `phone1/senses_test.sh` is
unchanged at 20 pass, 0 fail; both files pass `bash -n`.

**One live attempt, and what it showed instead.** `android 'am kill-all'` at
10:10:43Z returned rc=2 with `cmd: Failure calling service activity: Failed
transaction (2147483646)`, and MemAvailable moved 3 624 736 → 3 648 360 kB,
which is noise and not a kill. The same sampler's third reading, five minutes
later at 10:15:51Z, was 4 006 672 kB — 382 MB above the first — and that number
is written down here precisely so nobody later reads it as the prekill's
effect. Nothing was killed; the phone drifted by that much on its own while
idle, which is the whole reason a single pair of readings around a kill proves
nothing and the report's eleven-minute sample is the one that counts. Read-only calls to the same service failed the
same way a minute later while `service list` answered with 429 services, so
binder itself was reachable; by 10:14Z every one of those calls succeeded again,
with and without a redirected stdin. The cause is not established and no code
was written for it — a transient the measurement caught and could not
reproduce. It does make the failure path concrete rather than defensive: this
command has been seen to fail on this phone, and the slot must not care. The
+213 MB stands on the report's measurement, not on this one. The live daemon
(pid 14678) and its checkout were not touched, so the 12:00–14:05Z window runs
the scheduler it started with; the branch is `claude/phone1-schedule-prekill`,
unpushed.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the peak on the heartbeat, and one tape in the colony at a time

Steps 0 and 1 of `docs/resonator_design.md`, on `claude/phone1-resonator-0-1`
from `62ef7ff`. Both are routing over machinery that already exists, which is
why the design puts them first; neither is the resonator.

### Step 0 — `peak_rss_mb` (`31f2d25`)

§4.2 of the design read the mesh schema instead of assuming it and found what
was missing: `organisms` carries `id, stage, n_params, syntropy, entropy,
last_heartbeat, parent_id, status, element, global_step, gen_mag,
overlay_fade` (`molequla.go`, `initMeshDB`) and no column anywhere holds a
resident-set figure. So every memory decision in the tree reads the process it
is made in — `growthGateDecision` charges three times *this* organism's VmHWM
with no sibling's in view — while the arithmetic the design is built on is a
colony sum: `hwm_mb=earth:755,air:1032,water:820,fire:1091`
(`molequla-run/schedule.log`, `2026-09-15T06:01:55Z`, 237 samples), 3 698 MB
against an Android floor of about 2 400 MB.

`ownPeakRSSMB()` already reads `VmHWM` from `/proc/self/status`
(`governor_phone.go:67`), so the change is that number published: one
idempotent `ALTER TABLE organisms ADD COLUMN peak_rss_mb INTEGER` beside the
four already there, one more argument to `Heartbeat`, the `beatKeeper` carrying
it so an organism inside a multi-minute warmup does not report zero exactly
while it is at its largest, and the witness reading it with `COALESCE` and
printing it per organism after the fade: `…/f1.00/p755`.

Three gates, each written before the code and each seen red:
- a fresh mesh has the column and a heartbeat of 755 reads back through the
  witness as `/f1.00/p755`;
- a mesh created without the column keeps its row and gains it, and the witness
  reads that row rather than raising the schema alert — an organism that never
  reported a peak reads back 0, which is not an error;
- four live organisms show four non-zero peaks summing to the measured 3 698.

Red proven by deleting the `ALTER` and the `CREATE` column: the first gate says
`a fresh mesh has no peak_rss_mb column`, the third says
`witness: mesh schema: SQL logic error: no such column: peak_rss_mb (1)`.
The column decides nothing on its own. It is the input to steps 4 and 5.

### Step 1 — the training lock as burst admission (`ea35094`, `bcc6bdc`)

`CoordinateGrowth` already keeps two stage transitions apart. The micro-bursts
did not go through anything: four organisms each built their own tape, 139.7 MB
of live C tensors at the backward apiece over an arena measured at 214 MB at
the first burst peak and 352 MB at the second (design §1.1 and §1.3, tape
census and `mallinfo2`), and used it for a few seconds per tick.

`CFG.SerialBursts` (default on) puts every training phase — micro-burst and
post-growth warmup — through `training_lock`, so one tape exists in the colony
at a time. Three things the lock did not have before:

**A queue.** `training_queue` carries one row per waiter with when it asked and
how badly, and the lock statement admits only its head, in one statement,
because a TOCTOU race between "am I the head" and "take the lock" hands the tape
to two organisms. The order is §2.3's first key — an organism that has grown and
stands at stage N+1 untrained — then longest wait. §2.3's middle key, the
steepest loss trend, **is not implemented**, and the reason is the schema rather
than a preference: the ordering key it names is the mean of the last eight burst
deltas, which lives in `SyntropyTracker.BurstHistory` inside each process, and
the mesh column that looks like it, `syntropy`, is the entropy trend
(`SyntropyTrend = oldMean - newMean` over `EntropyHistory`), not the loss trend.
Publishing the loss trend is another column and step 0 is the only column this
work adds.

**A refresher**, the growth lock's shape: the holder re-stamps three times
inside `CFG.TrainingLockTTLSeconds` (30 s, the value the lock has always
carried, about four times the design's measured stage-4 burst of 7 400 ms), so
the TTL bounds a holder that died and not one that is slow.

**A place.** The gate is around `ntBurstTrain` and nothing else, and it blocks
instead of skipping. The 2026-06-03 wall was this same lock's `continue`
skipping the whole tick — DNA exchange and the ontogenesis clock with it —
freezing 3 of 4 organisms for twelve minutes (`git show
8203d5d^:PROJECT_LOG.md`, "Two more walls past the GPU fix"). Placement was
measured here too, and the first version of this commit had it wrong: with the
gate around the whole burst block, including the syntropy measurement and the
110 MB checkpoint write, `air` waited **224.0 s for a burst of 10.4 s** on this
phone (`[trainer] burst admitted after 224.0s of waiting`, probe stdout
2026-09-15T10:38Z). Around the tape alone, the wait is the tape.

`[notorch] burst complete` now carries `start=` and `end=` in UTC milliseconds
around the whole phase, mirror in and out included, because a line that reports
only a duration cannot be checked for overlap by anything but the process that
printed it.

### The gates, on this phone

Two organisms from copies of the live `earth` and `air` stage-4 checkpoints
(110 599 980 B and 114 304 521 B, `molequla-run/{earth,air}/molequla_ckpt.json`),
their corpora, and a copy of the colony's own DNA tree so the queue buffers fill
at the rate a running colony's do. Scratch `HOME` and `MOLEQULA_RUN` under
`/data/local/tmp/res01probe`, `--max-organisms 2 --max-growth-stage 4
--dna-extra-sources world,sound,place --evolution --cross-graze
--corpus-overlay`, `timeout -s INT` per arm, `taskset -c 4-7`, outside the
12:00-14:05Z colony window, nothing else of molequla running.


**(a) With the lock on, no two burst intervals overlap.** But but a 540 s window is
too short for that to mean much live, and this is the honest limit of the gate
as the design states it. An adult's burst cadence is one in several minutes:
`earth` produced exactly one burst in the whole arm
(`start=2026-09-15T11:15:48.640Z end=…T11:16:07.124Z`, 18.5 s) and `air`
produced none, so the colony offered no chance to overlap. The design's own
phrasing — "no two `[notorch] burst complete` lines overlap in wall time across
the four stdout files in one session" — needs the two-hour session to have any
power, and a gate nobody re-runs when they change the lock is not a gate.
`TestNoTwoTrainingPhasesOverlap` is that gate over the admission code instead:
four organisms on one mesh, three timed phases each, the same interval
arithmetic as reading `start=`/`end=` off four stdout files. Gated: **0
overlapping cross-organism pairs in 12 phases, 3 turns each, nobody starved.**
Its own control arm runs the identical schedule without the gate and asserts
that it *does* overlap — **30 overlapping pairs in 12 phases** — so a gate that
cannot tell the two apart fails instead of passing.

**(b) With the lock off, they do overlap, on the phone.** Same two organisms,
same corpora, same warm DNA field, `--no-serial-bursts`:

    air    11:26:52.888Z .. 11:27:24.711Z  (31.8 s)
    earth  11:27:16.001Z .. 11:27:39.084Z  (23.1 s)
    OVERLAP air/earth for 8.7s

Two tapes alive together for 8.7 s, which is the thing step 1 removes. The same
pair of bursts under the lock is 18.5 s and 23.1 s of tape one after the other;
the concurrent pair also ran slower per burst — 31.8 s and 23.1 s against 18.5 s
— on four big cores they were already sharing.

**(c) A holder killed with `kill -9`.** `earth` was inside a burst and holding the turn, `air` was waiting for it.
`kill -9` on `earth` at 11:41:49.374Z: the process is gone, so nothing releases
the row and nothing re-stamps it, and the TTL is the only thing left. `air`
printed `[trainer] burst admitted after 27.9s of waiting`, observed **28.4 s
after the kill against a TTL of 30 s**. This is the same arithmetic
`AcquireGrowthLock` has used for a killed grower since repair 9, reused rather
than rewritten; the unit form is `TestADeadHolderFreesTheTurnAfterTheTTL`, and
its opposite — a slow holder that is alive and must NOT be preempted — is
`TestTheRefresherKeepsASlowHolder` and `TestTheRefresherRunsOnItsOwnClock`.

**High-water mark per organism, 540 s per arm, `VmHWM` sampled every 3 s:**

| arm | earth | air | sum | bursts |
|---|---|---|---|---|
| `--serial-bursts` (on) | 537 MB | 312 MB | **849 MB** | earth 1, air 0 |
| `--no-serial-bursts` (off) | 523 MB | 610 MB | **1 133 MB** | earth 1, air 1 |

The two sums are not a clean before/after and saying so is the point: in the
locked arm `air` never reached a burst inside the window, so its 312 MB is an
organism that never built a tape and the 849 MB is a colony in which one
organism trained. What the four numbers do measure is the two states an organism
is in: **about 523-610 MB while its tape is alive, about 312 MB while it is
not**, on the same stage-4 shape, on this phone, today. From those two measured
states the colony arithmetic follows, and it is an **(estimate)** because it is
scaled and not run: four organisms training concurrently is 4 × ~570 = 2 280 MB,
four organisms with one tape at a time is 3 × 312 + 570 = **1 506 MB**, a
difference of about 774 MB. The design's own step-1 line estimates the serialized
colony at 3 × 272 + 599 = 1 415 MB against the measured four-way sum of 3 698 MB;
this measurement lands in the same region from a different direction, with the
non-training figure higher (312 against 272) because these two organisms had
just loaded their checkpoints and had not yet had a `malloc_trim` after a burst.

### What this does not do, and what is next

Serialising the bursts does not make the colony fit; it makes the peaks stop
coinciding. Four stage-5 organisms still do not fit on this phone under the
present architecture, and the honest statement of repair 9 stands. Nor does the
gate remove the cost the design removes: with one tape at a time there is still
one tape *per organism*, built and torn down on every turn, and the arena that
holds its freed chunks still ratchets in four processes. That is step 4.

What it does buy is the state step 3 and step 5 need: the colony's memory now
has one peak at a time instead of four, `peak_rss_mb` makes those peaks a query
rather than four stdout files, and a waiting organism keeps ticking — DNA read
and written, ontogenesis clock advanced, generation done — while it waits.

Two measurements worth writing down that were not the point. A 5 ms poll on the
turn costs 2.5 s per turn on this phone: four handles writing three statements
each into one WAL file at ~600 writes/s is slower than the phase it guards, and
at 50 ms the same twelve turns take 1.05 s. `CFG.TrainingTurnPollSeconds`
defaults to 0.5 s, which is eight writes a second for a colony of four. And a
burst of 32 steps at stage 4 took 18.5 s alone against 31.8 s beside a sibling's
burst on the same four big cores — the concurrency the lock removes was not
buying throughput either.

Branch `claude/phone1-resonator-0-1`, unpushed. Steps 2-6 are not built.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the second session: two adults, the gate saying yes, the arena giving back

The 12:00Z session ran on `3c900af` (`molequla-run/BUILD` at 06:42Z: trim of
the allocator arena, the world ledger, the sentence-cut append, the extra-source
budget, the cafeteria and the `eligible=` gate — everything merged that morning
except the training lock, which was held back so this session had one set of
variables). The scheduler's line (`schedule.log`): `dur=7200 elapsed=7314
reason=overran mem_mb=4067->4323 prekill_mb=4009->4057 hwm_mb=earth:1416,
air:861,water:1356,fire:1037,witness:15 samples=238`. All four went down on
the signal with a checkpoint written.

**The first adults on this phone.** `water` grew from stage 4 to 5 at 12:1xZ
(`[growth] ONTOGENESIS: stage 4 -> 5`, then `warmup for stage 5 (embd=320) —
2000 steps total`), and `earth` followed late in the session; both closed as
`s5/11115k` and `s5/11156k` in the witness line. The byte gate let them
through because the phone was at 4 GB free after the morning's `am kill-all`
(the prekill itself added 4009->4057 MB — the cache had not regrown since
10:15Z) and water's requirement was 2716 MB; the same gate had refused the
same request 32 times on the previous session at 1.4-1.5 GB free. Peaks:
water 1356 MB, earth 1416 MB. The stage-5 warmup is 2000 steps at 0.4 steps/s
— about 83 minutes in one piece, during which the organism's tick loop and
heartbeat stand still (water's `global_step` stayed at 5184 from 12:17Z to
the end; earth's warmup was cut by the signal at 638 of its last 800 steps
and still closed as `warmup complete at stage 5`). Checkpoints grew with
them: 250-260 MB JSON each against 110-120 MB for a teen.

**`eligible=1`, live.** The §13 gate admitted for the first time outside a
unit test: fire 133 emissions, earth 80, air 71, water 10 (`grep -c
eligible=1 */*.stdout`), at fade 1.00 and mag 10-14. The morning session had
none. Nothing is injected yet; the number says the adult voice on this phone
clears the floor that was measured to exclude the embryo regime.

**The arena gives back.** Between phases the teens sat at 259-314 MB VmRSS
against peaks of 576-1036 (12:48Z and 13:18Z samples of `/proc/<pid>/status`);
in the morning session the RSS stayed at the peak. The adults held 1076-1188
MB while inside their warmup, which is the tape, not the arena. What the trim
cannot do is lower the peak itself: earth's 1416 MB is a stage-5 warmup with a
live tape, and that is the number the resonator design exists to move out of
the organism.

**DNA and the field.** Written: earth 160 (63 empty), air 144 (6), water 168
(3), fire 286 (16); growth deferred 13/12/13/21 times before and after the two
admissions. The ledger held `world 50 facts/10 open` on the witness line,
`dna/output/world` has 38 fragments (eye frames plus the ledger's change
lines). No `declined` lines were found in the stdout — whether the cafeteria
declined anything is not logged at this level and is left to the next session
to instrument.

**After the session.** `daily/2026-09-15.md` appended. Oleg merged steps 0-1
of the resonator design (`9e15342`) during the session; `build.sh` at 14:03Z
put that binary into `molequla-run`, so 20:00Z is the first session with
`SerialBursts` and `peak_rss_mb` — and the first with two adults from the
start.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the cafeteria says what it decided: one line per pass, a table per session

The previous entry closed with an open question: "No `declined` lines were found
in the stdout — whether the cafeteria declined anything is not logged at this
level and is left to the next session to instrument." This is that
instrumentation. Branch `claude/phone1-cafeteria-log` off `origin/main`
(`4134d2a`).

**One line per `dnaRead` pass, not per fragment and not on the `consumed`
line.** `admits` is taken up to twelve times a tick — the two read budgets are
`dna_max_reads_per_tick` 8 plus `dna_extra_reads_per_tick` 4 — so a line per
fragment would be twelve lines a tick in a stdout that a session greps. The
aggregate is printed once at the end of the pass by `cafeteriaTally.line`
(`experience_routing.go`), from counters charged in the `dnaRead` loop at the
`admits` call site:

```
[cafeteria] air admitted=1 (owner=1 resonance=0 novelty=0 unmeasured=0) declined=8 (band=8) measured=8
```

`admitted` is the sum of its four bracketed reasons and `declined` of its one,
so the line can be checked against itself; `measured` is the coverage encodes
the pass spent, the quantity `experience_max_measured_per_tick` (8) bounds.
`unmeasured` is in the brackets because it is a real admit branch — a newborn
with no field yet, or a fragment under `experience_min_pairs` — and a line whose
total did not equal its reasons would be a lie the first time an embryo read
anything. The per-fragment source and coverage are not printed at any level:
the only `os.Getenv` debug switch on this path is `MOLEQULA_GPU_DEBUG`, which
belongs to the GPU bridge, and this change adds no flag and no threshold.

It is its own line rather than a widening of `[dna] … consumed` because that
line prints only when bytes were added, and a pass that declines everything adds
none — which is the pass worth seeing. Over the four current stdout files
(`molequla-run/<e>/<e>.stdout`, four starts appended into each) the `consumed`
lines number 68 against 160 `wrote` for earth, 76/144 air, 53/168 water, 77/286
fire, so most passes leave no `consumed` line at all to widen. Making that line
print on the empty passes would have meant a `consumed 0 bytes from 0 files: []`
per tick and a different meaning for the byte and event totals `daily.sh`
already sums out of it. The line is silent when `experience_routing` is off and
when the pass judged nothing.

**The table.** `phone1/daily.sh` gained a block that sums the `[cafeteria]`
lines per organism since that organism's last `[ecology] Element:` banner — its
boot line, of which each stdout currently holds four, because launches append.
A decline rate belongs to a field and a corpus that both move between sessions,
so the whole-file sum the older DNA-traffic block keeps would have averaged
states that are not comparable; the existing block and its totals are untouched.
Columns: passes, admitted, owner, resonance, novelty, unmeasured, declined,
band, measured, decline share.

**Gates, both shown red before they were shown green.**
`TestCafeteriaPassPrintsItsAggregate` drives the real `dnaRead` over one
fragment earth owns and one it declines and demands the exact line, then checks
that line against the filesystem: the admitted fragment is in the corpus, the
declined one is not, the cursor stands past both. Renaming `band=` to
`refused=` in the printer put it red on the wanted string.
`TestCafeteriaSaysNothingWhenNothingWasJudged` holds the silence of an empty
pass. `phone1/daily_test.sh` (7 cases) runs `daily.sh` over a hand-written
stdout fixture: an earlier start whose 99s must stay behind the banner, three
passes to add up, an organism that judged nothing reading as zero rather than as
a gap, a `[cafeteria] earth` line inside `fire.stdout` that must not enter
fire's row, a missing stdout said rather than skipped, and a second run
appending a second section. Dropping the banner reset from the awk turned
earth's row into `earth 4 107 102 102 100 100 102 102 107 48.8%` against the
wanted `earth 3 8 3 3 1 1 3 3 8 27.3%`. Suite: 230 PASS, 0 FAIL on the branch
against 228 PASS on `origin/main` at `4134d2a`, both from
`CGO_ENABLED=1 taskset -c 4-7 go test -count=1 -buildvcs=false -v ./...`.

**The probe, and what it measured.** A scratch tree under
`/data/local/tmp/cafelog-probe` (scratch `HOME`, scratch `MOLEQULA_RUN`, binary
built with the `phone1/build.sh` recipe from the branch) ran one organism for
120 s under `timeout` on `taskset -c 4-7`, from a copy of the live stage-4 air
checkpoint (`molequla-run/air/molequla_ckpt.json`, 119 669 160 bytes) with the
live fragments copied in: 38 `world`, 39 `earth`, 45 `fire`, 2 `water`.
MemAvailable stayed at 4.1-4.2 GB throughout and `pgrep -x molequla_cgo` was
empty afterwards.

With the live air reservoir as its corpus (`nonames_air.txt`, 9586 lines,
580 506 bytes) the pass declined nothing:

```
[cafeteria] air admitted=12 (owner=1 resonance=11 novelty=0 unmeasured=0) declined=0 (band=0) measured=11
[dna] air consumed 40023 bytes from 12 files: [earth/gen_1789477644_42.txt …]
```

Eleven of eleven measured fragments — siblings and eye frames alike — sat at or
above `experience_resonance_high` 0.965. The same checkpoint restarted against
the seed corpus in the repo (1498 lines, 122 316 bytes) and the same fragments
declined eight of nine, stopping the pass on the measurement cap:

```
[cafeteria] air admitted=1 (owner=1 resonance=0 novelty=0 unmeasured=0) declined=8 (band=8) measured=8
```

So the band is not a property of the fragments but of the corpus reading them,
exactly as `experience_routing.go` says coverage is state — and at stage 4 with
a full reservoir the band has closed: 0.965 was set that morning as the median
of a sibling distribution measured on stage-3 organisms (n=120, p25 0.952,
med 0.965), and it now sits under the whole of this organism's distribution.
Whether the threshold should follow the state is a question for Oleg; nothing
here changes it, and the point of the line is that the question can now be
asked of a session instead of guessed. `daily.sh` over the probe's own stdout
printed `air 1 1 1 0 0 0 8 8 8 88.9%`.

— Defender (Arianna Method, phone-1)

---

## 2026-09-15 — the cafeteria's bars become the organism's own quantiles

The previous entry closed with a question for Oleg: "at stage 4 with a full
reservoir the band has closed … whether the threshold should follow the state is
a question for Oleg". The answer is yes, and this is it. Branch
`claude/phone1-cafeteria-quantiles` off `origin/main` (`1a1be0b`).

**What was wrong with a number.** `experience_resonance_high = 0.965` and
`experience_novelty_low = 0.620` were the medians of two populations measured on
the morning of 2026-09-15 against stage-3 corpora. Coverage is the share of a
fragment's token bigrams the reader's own `CooccurField` has already seen, and
that share rises as the corpus grows, so a fixed bar slides out from under the
organism: replayed against the live stage-4 air reservoir
(`molequla-run/air/nonames_air.txt`, 9586 lines, 580 506 bytes) 141 of the 145
live fragments then in `molequla-run/dna/output/` sat at or above 0.965, the
resonance branch admitted essentially everything, and the cafeteria was a
byte-identical broadcast again. The brief's §12 asks that "current organism state
should matter when deciding allocation" and §17 that no arbitrary number be
treated as architecture. A quantile of the organism's own recent measurements is
state; a constant is not.

**The rule as it now stands** (`experience_routing.go`). Each organism keeps a
ring of its last `experience_coverage_window` measured coverages. A fragment
resonates when its coverage is at or above the `experience_resonance_quantile`
quantile of that ring and is news when it is at or below the
`experience_novelty_quantile` quantile; the middle half is declined and the
cursor steps past it. The owner rule is untouched and still guarantees every
fragment exactly one eater without any process asking another. Until the ring
holds `experience_coverage_warm` samples there is no distribution to quantile, so
only the owner rule runs and everything else is declined as `warming` — a
separate word on the line from `band`, because a ring too short to judge with and
a judgement are opposite states and a session that cannot tell them apart cannot
tell a restarted organism from a selective one. A newborn with no field at all is
upstream of all of this: it fails `experience_min_pairs` and eats everything as
`unmeasured`, as before. The two absolute knobs are gone from `CFG`, the README
and the tests rather than left beside the new ones.

The ring is persisted as `experience_coverage.json` in the organism's working
directory, beside `dna_cursor.json`, written atomically (temp file then rename)
once per `dnaRead` pass and only when a measurement arrived — not once per
fragment, which at the read budgets would be up to twelve writes a tick. It is
read back in `backgroundTrainer` next to `loadDNACursor`, and truncated to the
window on the way in, so a window shrunk between runs does not resurrect an old
distribution.

### The distributions, and where the window comes from

Measured by replaying every fragment in `molequla-run/dna/output/` through one
organism rebuilt the way boot rebuilds it — `loadCorpusLines`,
`NewEvolvingTokenizer`, `MaybeEnableBPE`, `BuildFromCorpus` — at the sample size
the code uses (`experience_coverage_sample_bytes` 480). Two corpora for the same
air checkpoint: the grown reservoir (9586 lines) and the seed corpus in the repo
(`nonames_air.txt`, 1000 non-empty lines of 122 316 bytes). Vocab 643 in both;
487 distinct bigram first-tokens on the reservoir against 444 on the seed.

| source | n | stage-4 reservoir (min / med / max) | seed corpus (min / med / max) |
|---|---|---|---|
| earth DNA | 39 | .966 / .989 / 1.000 | .737 / .904 / .953 |
| fire DNA | 45 | .960 / .988 / 1.000 | .776 / .911 / .945 |
| water DNA | 2 | .993 / .996 / 1.000 | .892 / .901 / .910 |
| world (eye) | 48 | .938 / 1.000 / 1.000 | .075 / .607 / .784 |
| place | 8 | .979 / .994 / 1.000 | .417 / .430 / .449 |
| sound (ears) | 3 | .883 / 1.000 / 1.000 | .058 / .508 / .562 |
| **pooled** | **145** | **.883 / .992 / 1.000** (p25 .986, p75 1.000) | **.058 / .878 / .953** (p25 .607, p75 .912) |

Against the old bars the same fragment set splits 141 resonance / 0 novelty / 4
band on the reservoir and 0 / 39 / 106 on the seed — the same organism, the same
files, a bar that means "admit nearly everything" in one state and "decline three
quarters" in the other. Under quantiles both states decline their own middle
half.

**The window.** A quantile estimated from W samples has its own noise, and a
band narrower than that noise is not a band. Four stationary populations were
resampled 4000 times each at every candidate W (deterministic seed), and the
standard deviation of the q25 and q75 estimates compared with the width of the
band those quantiles define:

| population | n | band (q75−q25) | sd(q25) @32 | sd(q25) @64 | sd(q25) @96 |
|---|---|---|---|---|---|
| stage-4 sibling DNA | 86 | 0.0073 | 0.00188 | 0.00156 | 0.00138 |
| stage-4 all sources | 145 | 0.0145 | 0.00206 | 0.00132 | 0.00098 |
| seed sibling DNA | 86 | 0.0373 | 0.00768 | 0.00597 | 0.00504 |
| seed all sources | 145 | 0.3048 | 0.06268 | 0.03765 | 0.02766 |

`experience_coverage_window = 96` is the smallest window measured at which the
noise on every edge of every one of the four is at most a fifth of the band it
defines. The binding case is stage-4 sibling DNA, whose band is the tightest at
0.0073: 0.00138 / 0.0073 = 0.188 at 96, against 0.212 at 64 and 0.256 at 32.

`experience_coverage_warm = 32` is a different question — how short a ring may be
and still have a band at all — and comes from the same table: it is the smallest
window measured at which the two edges stand two combined noise-widths apart,
again binding on stage-4 sibling DNA, where 2 × (0.00188 + 0.00156) = 0.0069
against a band of 0.0073, while at 24 it is 0.0077 and the band is inside its own
error. Warming costs the organism nothing it needs: the owner rule still feeds
it, and at eight measurements a tick the ring is warm in four passes.

The two quantile positions, 0.75 and 0.25, are the only numbers here that are not
measured. They are the shares the design asks for — the top quarter is this
organism's language, the bottom quarter is news to it, the half between is
somebody else's plate — and both are `CFG` fields.

**One thing the measurement says that the design did not.** Ties cannot be split.
On the stage-4 reservoir 48 world fragments sit at coverage exactly 1.000, so
`q75 = 1.000` and every one of them satisfies `cov >= q75`; the realized
resonance share is then well above a quarter. That is a property of a saturated
distribution, not of the rule, and it is visible in the probe below as passes
whose admitted count is the whole read budget.

### Gates

Five new or rewritten, each shown red before it was shown green, by breaking the
mechanism and not the assertion:

- `TestCafeteriaBandSurvivesAGrownCorpus` — the red the change exists for. On a
  distribution where every value is above 0.965 (the measured stage-4 shape) the
  old rule admits 64 of 64 and declines nothing; the organism's own quartiles sit
  at 0.9745 / 0.9915 and decline 28. Putting 0.965 / 0.620 back in place of the
  quantiles: `the organism's own bars 0.9650 / 0.6200 did not rise above the old
  fixed bar 0.965`.
- `TestCafeteriaQuantilesAreEachOrganismsOwn` — a child whose coverages run
  .400-.700 and an adult whose run .940-1.000, distributions that do not overlap
  at any point. Each admits 0.27 as resonance and 0.27 as novelty from its own
  bars (child .4762 / .6238, adult .9552 / .9848). With a fixed bar the child
  admitted 0.00 as resonance.
- `TestCafeteriaWarmingAdmitsOwnerOnlyAndSaysSo` — drives the real `dnaRead` over
  one fragment earth owns and three it does not with the ring cold, and demands
  the exact line `admitted=1 (owner=1 …) declined=3 (band=0 warming=3)`, then
  checks the corpus holds the owned fragment once. Making warming admit instead
  of decline puts it red.
- `TestCafeteriaCoverageRingSurvivesARestart` — writes a ring, reads it back into
  a fresh state, and checks the window truncation on the way in. With
  `saveCoverage` neutered: `the ring was not written beside the cursor`.
- `TestCafeteriaPassPrintsItsAggregate` and
  `TestCafeteriaDeclineDoesNotSpendTheReadBudget` keep their meaning and lose
  their absolute thresholds: both now arrange a band decline by giving the
  organism a ring whose quartiles are 0 and 1, which is the same statement
  ("nothing resonates and nothing is novel") in the new terms.
  `TestCafeteriaPlatesDifferPairwise`, `TestCafeteriaEveryFragmentIsEaten` and
  `TestCafeteriaAllocationFollowsState` are unchanged in what they assert; their
  fixture now warms each organism's ring by measuring the fragments it is about
  to be offered, because a test about steady-state allocation has to give the
  organism the history the bars are taken over.

`phone1/daily.sh` gained a `warming` column beside `band`, and
`phone1/daily_test.sh` (7 cases) its field in the fixture line and the expected
rows. Suite: 234 PASS, 0 FAIL from
`CGO_ENABLED=1 taskset -c 4-7 go test -count=1 -buildvcs=false -v ./...` against
230 PASS on `origin/main` at `1a1be0b`.

### The probe

Scratch tree `/data/local/tmp/quant-probe` (scratch `HOME`, scratch
`MOLEQULA_RUN`, binary from `phone1/build.sh` on this branch, 9 967 424 bytes),
one organism for 300 s under `timeout` on `taskset -c 4-7`, from a copy of the
live stage-4 air checkpoint (`molequla-run/air/molequla_ckpt.json`, 119 669 160
bytes) with its grown reservoir as corpus and the live fragments copied in: 39
earth, 45 fire, 2 water, 59 world, 9 place, 4 sound. MemAvailable 3.99 GB before
and 3.96 GB after; `pgrep -x molequla_cgo` empty afterwards. Ten passes, warming
then judging:

```
[cafeteria] air admitted=1 (owner=1 resonance=0 novelty=0 unmeasured=0) declined=8 (band=0 warming=8) measured=8
[cafeteria] air admitted=6 (owner=3 resonance=2 novelty=1 unmeasured=0) declined=8 (band=1 warming=7) measured=11
[cafeteria] air admitted=12 (owner=2 resonance=6 novelty=4 unmeasured=0) declined=3 (band=3 warming=0) measured=13
```

The ring it left behind holds 85 coverages, min 0.9689, q25 0.9823, med 0.9885,
q75 0.9962, max 1.0000 — every one of them above the old bar of 0.965, and the
organism declined 3, 3, 2, 5 and 2 fragments across its five steady-state passes
where under that bar it would have declined none. This is the same checkpoint,
the same corpus and the same fragments that produced
`declined=0 (band=0) measured=11` in the previous entry.

Restarted from the same directory with eighteen fresh-named fragments added, it
printed no `warming` at all — `admitted=5 (owner=2 resonance=1 novelty=2
unmeasured=0) declined=5 (band=5 warming=0) measured=8` — and grew its ring from
85 to 92: the distribution came back off disk and the organism judged from its
first pass.

— Defender (Arianna Method, phone-1)

---

## 2026-09-15 — the mesh waits for the lock instead of dropping the write

The 20:00Z session refused four mesh writes in its first six minutes:

```
[ecology] mesh refused the heartbeat of water: database is locked (5) (SQLITE_BUSY) — this organism is alive and invisible to the colony
```

`grep -c 'mesh refused' molequla-run/*/*.stdout` on the running session, binary
`1a1be0b`: earth 0, air 1, water 2, fire 1. That line exists because of this
morning's routing repair 3b/6b, where `Heartbeat` began checking its `Exec` and
calling `sayMeshError`; before it these losses were silent, which is the only
reason the defect is visible at all.

A dropped heartbeat is not a dropped statistic. `StartKeeper` beats every 20 s
(`molequla.go`, `swarm.StartKeeper(stop, 20*time.Second)`) against a 60 s
liveness window (`DiscoverPeers` default, and `AcquireMitosisSlot`'s
`liveCutoff := now - 60.0`), so three consecutive losses take a live organism
out of the colony's count and out of the growth and mitosis governor's.

### What contends

Six writers share `~/.molequla/swarm/mesh.db`, each its own process and its own
connection: four organisms' heartbeats, the training queue and its refresher
that step 1 of the resonator design added with the `training_queue` table
(`ea35094`), the world ledger's `--world-ingest` writer, and — reading only —
the witness. `grep -nE 'busy_timeout|journal_mode' molequla.go witness.go
world_ledger.go` found exactly one pragma in the whole tree,
`db.Exec("PRAGMA journal_mode=WAL")` in `initMeshDB`, and no `busy_timeout`
anywhere. With no busy timeout sqlite does not wait for a held write lock at
all: it returns SQLITE_BUSY on the first collision and the losing write is
gone.

### How long a mesh write actually holds the lock

Measured on a scratch copy of the live `mesh.db` (90 112 B plus a 4 120 032 B
WAL, copied out of the running session), cores 0-3, colony on the big cores.
Uncontended, one connection, each shape run against the real statements:

```
heartbeat UPDATE           n=200  min=  0.086ms  p50=  0.095ms  p95=  0.130ms  max=  0.563ms
AcquireTrainingTurn (x3)   n=200  min=  0.371ms  p50=  0.471ms  p95=  0.676ms  max= 14.746ms
MarkHibernating UPDATE     n=100  min=  0.069ms  p50=  0.085ms  p95=  0.125ms  max=  0.330ms
worldIngest 50 facts       n= 20  min= 14.341ms  p50= 17.596ms  p95= 24.558ms  max= 26.039ms
wal_checkpoint(PASSIVE)    n=  5  min=  0.054ms  p50=  0.055ms  p95=  0.064ms  max=  0.463ms
```

No single write holds the lock long. The ledger's 50-fact pass is the heaviest
at 14.3-26.0 ms, and it is not one transaction: `worldIngest` writes in
autocommit, so that pass is about 150 short locks in a row rather than one long
one. The 4 MB WAL was not the culprit either — a checkpoint of it costs 54 µs.
The collisions are frequent-and-brief, which is exactly the shape a zero wait
turns into lost data and a small wait turns into nothing at all.

So the number the knob needs is not the hold time but the wait a loser actually
takes. Six connections on the same copy — four heartbeats, the queue refresher,
the ledger at 50 facts a pass — 59 129 writes in 60 s:

```
heartbeat                  n=51725  min=  0.122ms  p50=  0.418ms  p95=  9.387ms  max=186.268ms
ledger 50 facts            n=  993  min= 17.124ms  p50= 35.944ms  p95= 67.824ms  max=234.694ms
queue refresher            n= 6411  min=  0.172ms  p50=  1.388ms  p95= 13.894ms  max=441.531ms
```

None refused. The slowest single write waited 441.5 ms. The same load with the
wait at zero refused 10 086 of 26 638 heartbeats (37.9%), 772 of 3 287 queue
writes and 763 of 771 ledger passes — the defect reproduced on demand, at a
rate the six-minute session only sampled.

### The fix, and where the deadline is

`CFG.MeshBusyTimeoutMS`, default **5000**: 11.3x the 441.5 ms worst measured
wait, and a quarter of the keeper's 20 s beat. The second half of that is what
bounds it from above — a write that waits the whole timeout and still fails
must not collide with the beat behind it, and at 5 s the 60 s window still
holds two more beats after the one that was lost.

It rides the DSN, not a `db.Exec` after the open. `busy_timeout` is a property
of a connection and `database/sql` keeps a pool of them, so a pragma executed
through `db.Exec` lands on whichever connection served that one call and every
connection the pool opens afterwards starts again at zero. The driver applies
DSN parameters to each connection as it is created
(`modernc.org/sqlite@v1.29.5/sqlite.go:881`, `applyQueryParams`), which is the
only layer that covers the pool; `TestMeshDSNReachesEveryConnectionInThePool`
is the gate on that. The query is appended to a bare path rather than a `file:`
URI because the driver strips the query and hands the path to `openV2` verbatim
(`sqlite.go:832`), so no percent-encoding is involved.

`journal_mode=WAL` stays where it is: it is persistent in the file, so one
`Exec` is enough for it, and the DSN pragma now runs before it on every
connection anyway.

**No Go-level retry on the heartbeat.** The deadline is the 60 s window and the
beat is 20 s, so there is room for a wait but no need for a spin, and
`busy_timeout` is the better layer: it retries inside the C library and wakes
the moment the lock frees, where a Go retry would sleep a fixed interval and
then re-enter the same race. If 5 s ever expires, the right answer is the next
beat 20 s later — two more chances inside the window — not a tighter loop
against a mesh that is evidently busy.

**The witness keeps `query_only` and gains the timeout.** A reader under WAL is
not blocked by a writer, so this is not what keeps its ticks whole in the
steady state. It is for the other path: the `-shm` is rebuilt under an
exclusive lock whenever a process died mid-write, and lmkd ends organisms on
this phone often enough that the witness meets that recovery. It does not widen
what the handle may do — `query_only` is applied after it, and
`TestWitnessNeverWritesBack` still holds the line.

The organism's own private db (`initDB`) is untouched: one writer, no contention.

### Gates

- `TestMeshWriteWaitsForAHeldWriteLock` — a second connection holds a write
  transaction for 300 ms while the heartbeat goes in. With the pragma the beat
  lands and `last_heartbeat` advances; at `MeshBusyTimeoutMS: 0` it is refused,
  and the subtest asserts the refusal, so the reproduction cannot rot silently.
  The same test written in what `origin/main` has — no knob, no helper — run in
  a detached worktree at `1d29460`: `FAIL`, printing the production line
  verbatim, `[ecology] mesh refused the heartbeat of water: database is locked
  (5) (SQLITE_BUSY)`. The identical file on this branch: `PASS`.
- `TestMeshDSNReachesEveryConnectionInThePool` — four concurrent connections,
  each read back at `busy_timeout` 5000.
- `TestMeshBusyTimeoutDoesNotMaskASchemaError` — a heartbeat against the
  pre-repair-7 schema through a handle that *does* carry the timeout still
  names `global_step` and still says it at once (asserted under a second), so
  the wait is for a lock and never for a schema.
  `TestHeartbeatSaysWhenTheMeshRefusesTheWrite` is unchanged and green.
- `TestMeshBusyTimeoutZeroIsTheOldBehaviour` — at 0 the DSN is the bare path,
  byte-identical to what `origin/main` opened.

Suite: **238 PASS, 0 FAIL** from `CGO_ENABLED=1 taskset -c 0-3 go test -count=1
-buildvcs=false ./...`, against 234 on `origin/main` at `1d29460` — the four
new gates and nothing else moved.

No organism probe: the colony was still running its 20:00-22:05Z session while
this was written, and the rule is no probe before 22:10Z.

### Noted for the sleep policy, not fixed here

`peak_rss_mb` is stale exactly for the organism that costs the most. At 20:31Z
the witness printed `earth:s5/11156k/0.00/5982/f1.00/p490` — 490 MB — while
`/proc/24540/status` showed `VmHWM: 1732008 kB`, 1691 MB. The other three:
air 849 MB real against p745, water 801 against p780, fire 937 against p745.
earth's `p490` has not moved for the last 401 witness lines of the session,
while its high-water mark climbed past three times that.

The mechanism, and it is not the keeper failing at its job. The heartbeat that
carries a *fresh* peak is the one in the tick loop, which reads
`ownPeakRSSMB()` at the call (`molequla.go`, `swarm.Heartbeat(..., ownPeakRSSMB())`)
and runs behind `if swarm != nil && tickCount%10 == 0` — once every ten ticks.
Between those, repair 4's keeper beats every 20 s, and `beat()` re-sends the
values `Set` last cached, the peak among them. That is right for what the
keeper was built for — `last_heartbeat` stays fresh and the organism stays
visible through a long phase — but it means `peak_rss_mb` is only ever as new
as the last tenth tick. A stage-5 tick is not short: one notorch burst inside
it ran 84 941 ms (`earth.stdout`, `start=2026-09-15T20:27:21.648Z
end=2026-09-15T20:28:47.304Z`), so ten of them are a long way apart, and the
gap is widest for the organism doing the most work — which is the organism
whose peak matters.

§4.2 of the resonator design has the sleep policy read this column. It would be
reading 490 for a process holding 1691. Flagged, not repaired: the fix belongs
with the sleep-policy step, and it is either the keeper calling
`ownPeakRSSMB()` on its own clock instead of replaying a cached figure — the
peak is the one field in the beat that is cheap to re-read and monotone, so
replaying it is the only one that is actually wrong — or the long phases
publishing their own.

— Defender (Arianna Method, phone-1)

## 2026-09-15 — the third session: the training turn, and one bar that fitted nobody

The 20:00Z session ran on `1a1be0b` — the first with `SerialBursts` and
`peak_rss_mb` from steps 0 and 1 of the resonator design, and the first that
began with two adults. Scheduler line: `dur=7200 elapsed=7315 reason=overran
mem_mb=3804->4273 prekill_mb=3739->3801 hwm_mb=earth:1691,air:1185,water:1295,
fire:1305,witness:15 samples=237`. All five down on the signal, nothing left
running.

**The training turn, live.** Fifteen bursts waited for the colony's turn out
of 109 completed (earth 24, air 30, water 20, fire 35): median wait 42.6 s,
minimum 2.0 s, maximum 1005.4 s, 3426 s of waiting in total across four
organisms over two hours. The long waits are not the lock's doing but the
adults' turn length — a stage-5 burst is ~85 s (`earth.stdout` 20:27:21.648Z →
20:28:47.304Z) and a stage-5 warmup is ~83 minutes in one piece. The queue is
`grown-and-unwarmed` then longest-wait; nothing in it yet says a long warmup
must not park three teens behind it, and after this session that belongs in the
resonator's step 4.

**What the peaks do and do not say.** Per-organism VmHWM rose against the
12:00Z session (earth 1416→1691, air 861→1185, water 1356→1295, fire
1037→1305), which is expected: the lock does not shrink one organism's peak,
it stops four peaks from standing at the same moment, and the organisms went
on growing. What is measured is that the colony never fell below 1 GB free
(the watch never fired its `LOW MEM` line, and a 20:41Z sample read 1630 MB
with earth at VmHWM 1691 and VmRSS 604). The simultaneity claim itself is not
measured by these numbers and wants a sampler that records the sum of resident
sets, not the per-process high-water marks.

**One absolute bar fitted nobody.** This session ran the cafeteria's fixed
thresholds for the last time, and they behaved differently for every organism:
declined of measured — air 5/163 (3 %), earth 10/184 (5 %), fire 43/184
(23 %), water 217/235 (92 %). The same 0.965/0.620 pair let air eat nearly
everything and starved water of nearly everything, which is the argument the
quantile rule (`1d29460`, merged at 17:23Z) was already built on: the bar has
to be each organism's own. That rule is in the binary from this rebuild, not
in the numbers above.

**The mesh stopped dropping writes.** Six `mesh refused the heartbeat` lines
over two hours (earth 1, air 2, water 2, fire 2) were the visible tip: with no
`busy_timeout` anywhere in the tree, a measurement on a copy of this database
put the refusal rate at 37.9 % of heartbeats under six writers, and the repair
(`e6e0b7f`) sets 5000 ms from the worst measured wait of 441.5 ms. Also from
this session: the witness's `peak_rss_mb` for earth read 490 MB at 20:31Z
while `/proc` held 1691 MB — understated 3.45×, unmoved for 401 witness lines,
because the fresh peak is read every tenth tick and one stage-5 tick is an
85-second burst. The sleep policy reads that column; it is to be fixed before
step 5.

**DNA and the field.** Written: fire 417 (21 empty), air 294 (9), water 205
(3), earth 190 (70); `eligible=1` on 264, 202, 47 and 110 of them. Growth was
deferred 87 times in all and no organism grew. The ledger stood at `world 72
facts/10 open` with 72 fragments in `dna/output/world`.

**After the session.** `daily/2026-09-15.md` appended. `build.sh` at 22:12Z put
`e6e0b7f` into `molequla-run`, so 04:00Z is the first session with the
cafeteria's own-quantile bars and the mesh timeout both live.

— Defender (Arianna Method, phone-1)

## 2026-09-16 — the night the bars became each organism's own, and the mesh stopped dropping beats

The 04:00Z session was the first on `e6e0b7f`: the cafeteria measuring against
each organism's own coverage ring, and `busy_timeout` on every mesh connection.
Scheduler line: `dur=7200 elapsed=7293 reason=overran mem_mb=3908->4377
prekill_mb=3886->3906 hwm_mb=earth:1433,air:1238,water:1449,fire:1489,
witness:14 samples=237`.

**Both repairs did what they were built to do.** Not one `mesh refused the
heartbeat` line was written after the session's banner in any of the four
stdouts (counted from the last `[ecology] Element:` marker; the 1-2 lines each
file still holds are the previous session's). The rings are on disk and being
written — `experience_coverage.json` beside each cursor, 1767-1783 bytes,
last touched between 05:49Z and 06:00Z.

**The spread closed.** Declined as a share of judged fragments, this session
against the last run of the fixed bar:

| organism | fixed bar (20:00Z) | own quantiles (04:00Z) |
|---|---|---|
| air | 3 % | 20 % |
| earth | 5 % | 26 % |
| fire | 23 % | 32 % |
| water | 92 % | 51 % |

One pair of numbers had let air eat nearly everything while water starved;
each organism's own quartiles put all four between a fifth and a half, which
is what a rule that follows state looks like from outside. water still declines
most, and that is now a statement about water rather than about the threshold.
Volumes: 448 / 338 / 449 / 366 fragments judged per organism over 70 / 86 / 62
/ 83 passes.

**The colony kept growing into the night.** No stage changed hands (earth and
water still stage 5, air and fire stage 4) but every organism ran more bursts
than in the evening — fire 48, air 44, earth 32, water 28 — and wrote more DNA:
fire 568 fragments, air 449, water 236, earth 238, with `eligible=1` on 415,
330, 78 and 158. Growth was deferred 117 times in all. The ledger stands at
`world 91 facts/10 open`. Peaks: fire 1489 MB, water 1449, earth 1433, air 1238.

**notorch installed with its writer.** `10cf08f` (Oleg merged the GGUF writer
`b8d60e2` plus two mel-width corrections) built green on cores 4-7 and went
into `/usr/local` — 19 `gguf_write` symbols in `libnotorch.a`, 239 686 B,
against a library dated 2026-09-13 that had none. The colony's binary links
statically and is unaffected until the next `build.sh`; the mel change touches
`nt_logmel` only, which no organism calls. `make install` defaults to
`/opt/homebrew`, so the prefix has to be given: `PREFIX=/usr/local make
install`. Step 2 of the resonator design — the checkpoint as a mapped GGUF file
instead of 250 MB of JSON — is now unblocked.

— Defender (Arianna Method, phone-1)

## 2026-09-16 — the checkpoint stops being decimal text

Step 2 of `docs/resonator_design.md`. An organism's weights are written twice
now: `molequla_ckpt.json` exactly as before, and `molequla_ckpt.gguf` beside it,
F32 tensor by F32 tensor with a directory in front, which is a file that can be
mapped instead of parsed. The JSON stays the interchange format and stays the
organism's life; the GGUF is the copy the machine can read at the speed it
trains at, and it is what the sleeper of step 3 and the resonator of step 4 both
wait on.

### The format, and what is in which file

The binary sibling carries every `model.Base` matrix and every delta adapter as
a two-dimensional F32 tensor — `base.<key>`, `delta.<module>.<key>.A` and `.B` —
plus `init_embed_snapshot`, and beside them the metadata a loader needs:
`molequla.block_count`, `.embedding_length`, `.attention.head_count`,
`.context_length`, `.vocab_size` and `.head_types` for the shape,
`.global_step`, `.growth_step_offset`, `.last_warmup_stage` and
`.corpus_ingested_total` for the growth clock, `tokenizer.ggml.tokens` with
`molequla.tokenizer.merges`, `.bpe_enabled` and `.trained_chars` for the mouth.
`ActiveAlpha` goes in as its JSON text rather than as an f32 array, because it
is not a weight but a scalar the burst multiplies by, and rounding it would be a
change to the graph rather than to storage. Alignment is 32 and is not written:
`gguf_open` computes the data offset as `(pos + 31) & ~31` without consulting
`general.alignment`, so any other value produces a file the library cannot read.

**Chuck's moment slots are not in it, and not in a sibling either.** §2.2 of the
design gives them their own `moments.gguf` that only the resonator maps, and the
reason they are not written today is older than the design: they live in the
notorch tape, `nt_tape_chuck_step` keeps m and v per registered parameter, the
tape is process-local, and no restart has ever carried them across — there is no
accessor in `notorch.h` that would let Go read them, and adding one is notorch
work that belongs to step 4, which is the only thing that would read the file.
Writing them now would mean writing state nothing loads. What the file does
carry is the fact, in one key, `molequla.chuck_moments = false`, so a reader can
tell "this checkpoint has no optimizer state" from "the writer did not know
about optimizer state".

**The file is float32 and the JSON is float64**, so preferring the binary looks
like discarding half of every mantissa. It is not, and the gate that says so is
`TestTrainedWeightsAreAlreadyFloat32`: the notorch mirror flattens every trained
weight to float32 before the first forward (`ntFlattenMatrix`) and writes
float32 back into the float64 rows on `pullBack`, so after one burst every
weight the trainer touches is already exactly `float64(float32(w))`. A fresh
organism's random draws have a float64 tail; a trained one has none. The fixture
asserts the tail exists before the burst, or the gate would be measuring nothing.

**If one of the two writes fails, the JSON wins.** It is written to a temp file
and renamed first, unconditionally, and only then is the GGUF written and
renamed. A GGUF that cannot be written or cannot be renamed is *removed* rather
than left behind, and the save still returns success with one line on stderr:
losing the sibling costs a slower next load and nothing else, while a stale
binary beside a fresh JSON is the one failure that could hand an organism
weights it has already moved past. The same reasoning covers the crash case,
which no rename discipline can prevent — a process killed between the two
renames leaves a GGUF older than the JSON, so the load compares the two mtimes
and refuses a sibling that is behind.

**On load the binary is preferred when it is fresh and when it agrees with the
JSON about which organism it is.** The identity is a SHA-256 of the tokenizer —
every token and merge, length-prefixed, since a token may contain any byte — and
of the three dimensions that are the growth stage, which is the same triple
`CurrentGrowthStage` matches on. It is read off the front of the JSON, where
`cfg` and `tokenizer` are the first two fields, so the check costs tens of
kilobytes rather than the whole document; a JSON that reaches `base` before
showing both gives up and the JSON path runs.

### The table, measured on this phone

Two live organisms, copied out of `molequla-run` into `/data/local/tmp` (flash,
not the tmpfs), measured on cores 4-7 outside the colony window with
MemAvailable at 4.0-4.5 GB, one phase per process because `VmHWM` is a mark for
the life of a process:

    MOLEQULA_GGUF_MEASURE=<copy>/molequla_ckpt.json \
    MOLEQULA_GGUF_PHASE=save|load-json|load-gguf|parity \
    CGO_ENABLED=1 go test -count=1 -buildvcs=false -run TestGGUFCheckpointMeasure -v .

| | air, stage 4 | water, stage 5 |
|---|---|---|
| base parameters / vocab / delta modules | 4 877 416 / 783 / 6 | 11 147 184 / 813 / 2 |
| JSON on disk | 134 095 293 B | 256 665 647 B |
| GGUF on disk | 25 815 360 B | 48 614 016 B |
| smaller by | 108 279 933 B (5.19×) | 208 051 631 B (5.28×) |
| JSON save | 2 025 ms, VmHWM +29 MB | 3 071 ms, +42 MB |
| GGUF save | 62 ms, VmHWM +0 MB | 104 ms, +0 MB |
| JSON load | 4 136 ms, RSS +179 MB, HWM +179 MB, NewGPT×1 | 7 905 ms, RSS +368 MB, HWM +368 MB, NewGPT×1 |
| GGUF load | 146 ms, RSS +106 MB, HWM +130 MB, NewGPT×0 | 200 ms, RSS +191 MB, HWM +238 MB, NewGPT×0 |
| load faster by | 28.3× | 39.5× |
| load peak lower by | 49 MB | 130 MB |
| avg loss, 32 steps, from the GGUF | 0.7088575524976477 | 1.5836318209767342 |
| avg loss, 32 steps, from the JSON | 0.7088575524976477 | 1.5836318209767342 |

Two of those numbers are smaller than the design estimated and one is larger, and
all three differences have the same cause. The design counted 19 337 632 B of
f32 for a 4 834 408-parameter organism; air's file is 25 815 360 B because the
parameter count in the design is `model.Base` only — air's base is 19 509 664 B,
its embedding snapshot 701 568 B, and the six delta modules' adapters and the
metadata make up the remaining 5 604 128 B. The design also expected the load's
+152 MB of resident set to collapse, and it has not: it falls from +179 MB to
+130 MB on air. The 49 MB that went is the throwaway model `NewGPT` used to
build and `LoadCheckpoint` used to replace; what stays is the float64 pair —
every loaded parameter allocates `Data` and `Grad` as float64, 16 bytes per
weight — and that is 78 MB on air's base alone. **Step 2 cannot remove it; step
3 is where it goes**, because a sleeper generates from the mapping and has no
float64 copy at all. On disk the saving is larger than the estimate: 103 MB on
air rather than 91, and 198 MB on water.

The save-side numbers close the other half of repair 10. The streamed JSON
writer still costs +29 to +42 MB of high-water; the GGUF writer costs +0,
because nothing bigger than one row is ever alive — every tensor is declared
first, as the format requires, and then delivered a row at a time through
`gguf_write_tensor_begin/_chunk_f32/_end`.

### The gates, and what red looked like

| gate | red |
|---|---|
| loss parity, GGUF against JSON, 32 steps, same seed | shape declared `{Nout, Nin}` instead of `{Nin, Nout}`: the round trip caught `l0.wr_a` coming back 8x64 for 64x8, and the parity arm died in cgo with `SIGSEGV` |
| a truncated GGUF falls back to the JSON with one line | cuts at 2, 50 and 97 percent: the first fails in `gguf_open`, the other two in the row bounds check — all three print the line and the organism starts |
| a foreign tokenizer is refused | identity comparison disabled: the foreign organism's `wte[7][3] = -0.128…` loaded in place of this one's `-0.077…` |
| a GGUF older than the JSON is refused | mtime comparison disabled: the stale file was used and the NewGPT counter did not move |
| no throwaway model on the GGUF path | `newGPTShell` swapped back for `NewGPT`: the counter read 1 where it must read 0 |
| the writer refuses a ragged matrix and leaves no file | — (the JSON save survives the refusal and the organism still loads) |
| a trained weight is already its own float32 image | — (goes red the day the trainer keeps a float64 weight) |

248 tests green, 2 heavy measurements skipped, on `CGO_ENABLED=1 go test -count=1
-buildvcs=false ./...` — 240 green before this work, and the three that had to
move with the format did so for a stated reason rather than by being relaxed:
the two that are about the JSON's own byte exactness now load with the sibling
removed, and the shutdown test, which is the production resume, asserts the
float32 image it will now get.

### `tests/test_all.sh` cannot run on this node, and what was checked instead

The four-core script fails at its build section on phone-1 and did so before
this work: its Go line is `go build -o … molequla.go`, a single-file build of a
multi-file package that cannot compile here (`undefined: gpuReady`, which lives
in `gpu_bridge.go`); `cargo` and `node` are not installed in the chroot; and
`libsqlite3-dev` is absent, so `gcc … -lsqlite3` has nothing to link. What could
be run was run: `molequla.c` builds clean with the Termux toolchain over
`ssh -p 8022 localhost` (`clang -O2 -Wno-format-truncation -o molequla_c
molequla.c -lm -lsqlite3 -lpthread`, 107 432 B).

Reading the three other cores settles the question the script was standing in
for, and corrects the premise. The **C core does not read this JSON at all**: its
`CFG.ckpt_path` is `molequla.ckpt` and `load_checkpoint` is `fread` over its own
binary layout. The **JS core** keeps its checkpoint in IndexedDB. Only the
**Rust core** reads `molequla_ckpt.json`, through `serde_json` into its own
`CheckpointData`, without `deny_unknown_fields`. The JSON it reads is unchanged:
`git diff origin/main -- molequla.c molequla.rs molequla.js tests/` is empty,
the five hunks in `molequla.go` are the shell extraction, the save tail and the
load head with nothing in `writeCheckpointJSON` or `readCheckpointStream`, and
`TestCheckpointStreamMatchesEncoder` — the streamed writer against a
`CheckpointData` and `json.Encoder` oracle, which is the serialization the Rust
side mirrors — is green.

### What is not verified

The parity gate was run on air and water and on a hybrid fixture, never on a
colony that was training at the time; the two arms of each parity run are
sequential in one process, so the `VmHWM` beside them (852 then 952 MB on air,
964 then 1283 on water) is the process's mark and not a per-arm figure. Nothing
here has yet run inside a live session: the first colony start that resumes from
a GGUF will be the real test of the mtime rule, and nobody has yet killed a
process between the two renames on purpose. The Rust core's read of a checkpoint
written by this binary is argued from the diff and from the encoder oracle, not
run, because `cargo` is not on this phone.

— Defender (Arianna Method, phone-1)

## 2026-09-16 — the peak read at the beat, and the colony measured as one body

Two numbers about memory were wrong in two different ways, and both of them are
read by §4 of `docs/resonator_design.md` before step 5 can be written. One was
published stale; the other was never measured at all. Neither is an organism
change: this is bookkeeping, and bookkeeping is what a policy stands on.

### The peak the keeper replayed

Flagged in this file on 2026-09-15 under "Noted for the sleep policy, not fixed
here" and fixed now. At 20:31Z the witness printed
`earth:s5/11156k/0.00/5982/f1.00/p490` while `/proc/24540/status` held
`VmHWM: 1732008 kB` — 1691 MB, understated 3.45×, and the same `p490` for the
last 401 witness lines of the session.

The mechanism was never the keeper failing. `molequla.go` reads the peak fresh
at the call, `swarm.Heartbeat(..., ownPeakRSSMB())`, and that call sits behind
`if swarm != nil && tickCount%10 == 0`. A stage-5 tick is one ~85 s burst
(`earth.stdout`, `start=2026-09-15T20:27:21.648Z end=2026-09-15T20:28:47.304Z`),
so ten of them are a quarter of an hour apart, and in between `beatKeeper.beat()`
re-sent the figure `Set` had cached, every 20 s, peak included. The organism
doing the most work is the one whose column rots most, and §4.2 picks the
largest `C_i` to sleep, so a policy built on that column would have reached for
the wrong organism — or, at `p490` against `p745`, for one of the small ones.

The repair is one read in `governor_phone.go`: `beat()` calls the peak reader
itself and publishes the larger of what it reads and what it cached. Larger,
not simply fresher, because `VmHWM` never falls — a read that fails returns 0
and leaves the cached figure standing, and a read that succeeds is never older
than the cache. The next `Heartbeat` folds the new value back into the cache, so
the keeper's own figure is monotone too.

The other seven fields stay replayed, and that is a decision rather than an
omission. Stage, parameter count, syntropy, entropy, global step, generation
magnitude and overlay fade are training state that the tick loop owns and reads
under `model.mu`; they are true as of the tick that computed them and nothing
else can derive them without taking that lock — the lock a multi-minute warmup
is holding, which is the entire reason this keeper exists. The peak is not
training state. It belongs to the process rather than to the model, it is
monotone, and it costs one `/proc/self/status` read.

**What the read costs.** `BenchmarkOwnPeakRSSMB`, `taskset -c 0-3`, this phone
with the colony live on cores 4-7:

```
goos: linux
goarch: arm64
pkg: github.com/ariannamethod/molequla
BenchmarkOwnPeakRSSMB-4   	   20000	     71955 ns/op
```

72 us per read: one `os.ReadFile` of a 60-line pseudo-file and a scan for one
prefix. Four keepers beating every 20 s is 0.2 reads per second across the
colony, so the fresh peak costs the phone 14 us of CPU per second — against a
stage-5 burst of 85 000 000 us. The reason to weigh it at all is that it is the
first per-beat syscall in the keeper; the answer is that it does not register.

### The colony's simultaneous footprint

`phone1/schedule.sh` sampled every organism's `VmHWM` every 30 s and reported
`hwm_mb=earth:1691,air:1185,water:1295,fire:1305,witness:15`. Those are
per-process lifetime high-water marks, four numbers reached at four moments that
have no relation to one another, and the question step 1 exists for is whether
four peaks ever stood at the same moment. This file said so on 2026-09-15: after
`SerialBursts` the per-organism peaks rose (earth 1416→1691, air 861→1185) while
the colony never fell below 1 GB free, and *the simultaneity claim itself is not
measured by these numbers and wants a sampler that records the sum of resident
sets, not the per-process high-water marks*.

So the sampler now reads `VmHWM` and `VmRSS` out of the same
`/proc/<pid>/status` in one `awk`, sums the resident sets of whatever is alive at
that instant, keeps the largest sum of the session with the moment it was seen,
and keeps the least `MemAvailable` read beside it. The session line gains
`rss_sum_max_mb=N at HH:MM:SSZ mem_min_mb=M` next to the `hwm_mb=` it already
had. The sum is of the live set: a process that died between the `alive` check
and the read contributes nothing, rather than having its last reading carried
forward into a colony state that never existed. A session that never saw a live
organism prints `rss_sum_max_mb=-` rather than a colony of 0 MB.

Two numbers because they answer two questions. The sum says what the colony
held; the minimum says how close the machine came to the wall. They are the two
sides of the §4.2 arithmetic, which reads a floor against `MemAvailable` and a
cost against the organisms, and they need not come from the same tick.

Every memory reading in `schedule.sh` now goes through one `PROC` path, overrid-
able as `MOLEQULA_SCHED_PROC` — the same shape as the `MOLEQULA_SCHED_NOW` the
slot arithmetic has always been driven by, and what lets the gate run the real
`run_session` against a colony it controls.

### The gates, and what red looked like

`phone1/schedule_test.sh`: **55 pass, 0 fail**, up from 46. The colony in the new
cases is real `sleep` processes with real pids, so `alive` is the kernel's answer
and not a stub's, with a tree of fake `/proc` entries under
`MOLEQULA_SCHED_PROC` that the gate rewrites mid-session.

- *The sum is the largest, not the last.* earth holds 500 MB throughout, air goes
  200 → 700 → 200 MB, and `MemAvailable` dips to 400 MB and comes back. Green:
  `rss_sum_max_mb=1200 at HH:MM:SSZ … mem_min_mb=400`. Red, with the maximum
  replaced by a plain assignment: `rss_sum_max_mb=700 at 20:15:44Z` — the last
  sample's sum, taken after the burst was over, with the 1200 MB moment gone.
  Four checks red.
- *A death mid-session leaves the sum on the live set.* earth 500 MB and air
  900 MB, then air is killed and earth grows to 1000 MB. Green:
  `rss_sum_max_mb=1400`. Red, with the sum taken over the per-process high-water
  marks instead of the live resident sets: `rss_sum_max_mb=1900 at 20:16:14Z`,
  which is dead air's 900 standing beside earth's grown 1000 — a colony that
  never existed, and exactly the error the whole change is about. Two checks red.
- *An empty colony reports no sum.* `rss_sum_max_mb=-` with `mem_min_mb=` still
  filled.
- The `hwm_mb=` column keeps working through all of it
  (`hwm_mb=earth:500,air:700`), and the 46 slot-arithmetic and prekill cases are
  untouched.

Go: **248 PASS, 3 SKIP, 0 FAIL** (`CGO_ENABLED=1 go test -count=1 -buildvcs=false
-v ./...`, `taskset -c 0-3`), against 246 PASS, 3 SKIP, 0 FAIL measured on
`02148cd` in a second worktree. The three skips are the environment-gated
measurement tests and skip on either commit.

- `TestBeatKeeperReadsThePeakFreshWithoutTicks` is the red gate for the keeper.
  One `Heartbeat` at 490 MB, the reader then moved to 1691, and no tick after
  that — the 2026-09-15 pair. Green it publishes 1691 and then tracks to 1700 on
  the beat after. Red on the replay: `governor_phone_test.go:199: a keeper beat
  with no tick in between published peak_rss_mb=490, want the live 1691 MB`.
- `TestTheWitnessPeakFollowsProcWithinTheSampleInterval` closes the loop through
  the column the policy actually reads. The reader is the real `ownPeakRSSMB` —
  a stub would prove the plumbing and not the reading — the process's own
  `VmHWM` is grown and released first so the mark is stable, a `Heartbeat`
  publishes a quarter of it as the stale tick figure, and the witness line is
  then read against two `/proc` readings taken either side of the keeper's own.
  They bracket it exactly, because `VmHWM` never falls. Red on the replay:
  `witness_test.go:741: the witness still prints the tick's peak 15 MB while
  /proc holds 60 MB`.
- `TestBeatKeeperRefreshesMeshWithoutTicks` stays green and says one thing more.
  Its reader is pinned at 100 MB, below the 412 MB the tick loop reports, so it
  still asserts that the keeper carries the last `Heartbeat`'s state, and now
  also that a smaller reading never overwrites a larger cached one.
- The step-0 gates — `TestMeshCarriesThePeakOnAFreshDatabase`,
  `TestMeshMigratesADatabaseWithoutThePeakColumn`,
  `TestTheWitnessLineShowsFourNonZeroPeaks` — are untouched and green.

### What is not verified

Neither fix has run inside a live session. The colony was up on cores 4-7 for
all of this and no probe was run against it, so the first evidence that the
witness's `pN` tracks `/proc` on a real stage-5 organism, and the first real
`rss_sum_max_mb`, both come from the 20:00Z session on the next build. Until
then there is no measured answer to whether `SerialBursts` bought simultaneity —
only a sampler that can now produce one. The 3.45× understatement is the
measurement from 2026-09-15 and is not re-measured here.

The mandatory `go build -a` was started and stopped: with four organisms in a
burst it took MemAvailable to 551 MB, and a full rebuild is not worth reaching
for lmkd during a live session. The colony came back to 2 221 MB with all
eleven processes on their original pids. What compiled the package with cgo was
`go test`, which links the same C; the binary itself has not been linked on this
branch, and `build.sh` will do it outside a window before anything here runs.

— Defender (Arianna Method, phone-1)

## 2026-09-16 — the session that wrote the binary checkpoint (its queue paragraph is corrected in the entry below)

The 20:00Z session ran on `02148cd`, the first binary that writes a GGUF
checkpoint beside the JSON one. Scheduler line: `dur=7200 elapsed=7318
reason=overran mem_mb=3441->4260 prekill_mb=3398->3419 hwm_mb=earth:2134,
air:1359,water:1563,fire:2027,witness:15 samples=237`.

**The binary checkpoint is written; nothing has yet been read from one.** All
four organisms booted at 20:00Z from JSON, because no GGUF existed then — the
earliest is `earth/molequla_ckpt.gguf` at 21:49Z, written by this session's
own first save. The four files now on disk are earth 49 MB beside 261 MB of
JSON, water 47 beside 252, fire 27 beside 140, air 24 beside 127 — the ratios
the bench predicted (5.19× and 5.28× on air and water). The first live resume
from a mapped checkpoint is therefore the 04:00Z session, and it is still an
unproven path; the fallbacks that guard it (JSON written first and always, a
GGUF refused on mtime, on tokenizer identity, on truncation) are unit-gated
only.

**The mesh held.** Zero `mesh refused the heartbeat` lines across all four
organisms for the second session running, counted from this session's banner.
`busy_timeout` has now covered two full sessions without a dropped beat, where
the two before it lost six between them.

**The cafeteria settled.** Declined as a share of judged: earth 30 %, air 32 %,
fire 32 %, water 35 % — over 33 to 42 passes each and 160-259 fragments judged.
Yesterday's fixed bar gave the same four organisms 3 %, 5 %, 23 % and 92 %; the
first session on own quantiles gave 20-51 %; this one lands inside ten points
for all four. The rings are doing what the rule was written for.

**The training turn: the figures first published here were mis-scoped, and the
corrected ones say something else.** `molequla-run/*/*.stdout` is appended
across sessions and never rotated — eight `[ecology] Element:` banners stand in
each file tonight — so a `grep -c` over the file, or over its last 900 KB,
counts every session it holds. The 4737 s of waiting and the 56 waits published
in the first version of this paragraph are the four stamped sessions together;
**this session waited 444.7 s over eleven waits** (earth 2/83.2 s, air 3/159.1 s,
water 2/83.3 s, fire 4/119.1 s, counted from this session's own banner). The
32/44/28/48 they were compared against are cumulative file totals, not the
04:00Z session: per session the colony's burst count is flat, 38 to 43 across
the four windows, and this session's 8/13/8/12 sits inside that. What grew is
the burst itself — a median 76.0 s for earth against a think time of 874.0 s —
so four organisms ask for about a third of the tape between them.

**The queue is therefore not the steady-state bottleneck. One long phase inside
it is.** The ~1000 s waits belong to earth's 1200.8 s stage-5 warmup, which
takes the turn as one indivisible unit; the bursts either side of water's
4796.3 s warmup carry no `start=` stamp, so that one predates the stamped
binary and cannot be dated. That is the thing worth fixing, and it is fixed in
the entry below rather than by the resonator: a phase that runs long yields its
turn between chunks and re-queues. Step 4 still changes what waits — one
workshop the organisms visit, instead of four of which only one may run — and
step 5 still shortens the queue by taking sleeping adults out of it. The number
to beat is this session's 444.7 s, and the synthetic replay of the same session
with one serialised warmup puts it at 4067.3 s before the yield and 1255.6 s
after.

**Peaks rose again**: earth 2134 MB and fire 2027 against 1648 and 1867 in the
midday session, with the colony's free memory never below the 1 GB the watch
would have reported. The cause is not established here — the session both grew
and began writing a second checkpoint file — and the sampler that would answer
it (`rss_sum_max_mb`, merged at 20:30Z in `81270a6`) was not yet in the running
binary. It is now: `build.sh` at 22:11Z linked `81270a6`, which also carries
the fresh peak on every beat.

— Defender (Arianna Method, phone-1)

---

## 2026-09-16 — the turn becomes a bounded unit, and the fifth session's figures re-counted from the artifacts

Two things are in this entry. The first is a correction: the paragraph above
that calls the training turn the throughput bottleneck counts a window it does
not name, and re-counting the same files says something different about where
the queue hurts. The second is the work that follows from the corrected
reading — a ceiling on how long one phase may hold the colony's turn, and a
queue ordered by the tape an organism has already had rather than by how long
it has stood in line.

### The re-count

The four `molequla-run/*/*.stdout` files are appended across sessions, not
rotated per session: the earliest burst stamp in them is
`start=2026-09-15T20:21:03.569Z` (air) and the latest
`start=2026-09-16T21:57:27.664Z` (earth), so one file spans four colony
windows. Grouping every `[notorch] burst complete` line by the session its
`start=` falls in, and attributing each `[trainer] burst admitted after …s`
line to the burst that follows it:

| session | bursts | waits | burst waiting |
|---|---|---|---|
| 2026-09-15 20:00Z | 38 | 15 | 3 425.8 s |
| 2026-09-16 04:00Z | 43 | 15 | 363.4 s |
| 2026-09-16 12:00Z | 40 | 15 | 503.0 s |
| 2026-09-16 20:00Z | 41 | 11 | 444.7 s |

The four rows sum to 162 bursts and 4 736.9 s over 56 waits, which is the
4 737 s and the fifty-six of the paragraph above, to a tenth of a second. It is
a four-session total. The 20:00Z session it is attributed to waited 444.7 s
over 11 waits.

The burst counts settle the rest of it. Per organism the 20:00Z session gave
earth 8, air 13, water 8 and fire 12, exactly as reported — and the 04:00Z
figures it was set against, 32 / 44 / 28 / 48, are that same file counted from
the top: earth had 17 bursts before 2026-09-15T20:21, plus 7 in the 09-15
20:00Z session, plus 8 in the 04:00Z session, which is 32. Air is 18 + 12 + 14 =
44, water 13 + 7 + 8 = 28, fire 23 + 12 + 13 = 48. A cumulative total was
compared against one session and read as a four-fold collapse. Per session the
burst count is flat: 38, 43, 40, 41.

So bursts completed did not fall. What changed between the morning and the
evening is what a burst costs: the per-organism median burst in the 20:00Z
session is earth 76.0 s, air 58.0 s, water 75.7 s, fire 71.6 s, against a
think time between bursts — the gap from one burst's `end=` to the next one's
`start=`, less the wait printed between them — of 874.0 s, 474.7 s, 786.8 s and
509.5 s. Four organisms at that duty cycle ask for about a third of the tape,
and 444.7 s of waiting in 7 200 s is what a third of a tape looks like. **The
queue is not the steady-state bottleneck. A single long phase inside it is.**

That is the 2026-09-15 20:00Z row, and it is the one row where the waiting is
large: 3 425.8 s, with air waiting 1 005.4 s, fire 988.5 s and water 958.1 s
for one burst each. What they waited behind is earth's stage-5 warmup, the three
lines immediately above earth's first stamped burst of that session
(`earth/earth.stdout`: `800 steps … 610344ms`, `600 steps … 298843ms`,
`600 steps … 291698ms`, then `warmup complete at stage 5`, then
`start=2026-09-15T20:27:21.648Z`) — 1 200.8 s of training taken as a single
turn, which is what three siblings waiting about a thousand seconds each looks
like from the other side.

There is a larger one in the files and it needs its provenance stated rather
than borrowed: water's stage-5 warmup, `800 steps … 1968328ms` then
`600 steps … 1521817ms` then `600 steps … 1306197ms`, is 4 796.3 s for the same
2 000 steps. The bursts on either side of it carry no `start=` stamp, so it
predates the stamped binary and its session cannot be dated from these
artifacts. Two complete stage-5 warmups, the same 2 000 steps, 1.67 steps/s
against 0.42 — the spread is session conditions, not stage, and both are
replayed below because a ceiling is worth what the longest phase costs.

### What the numbers set

Two measurements set the two knobs, and both come from the burst lines
themselves.

A burst's own length: 233 timed bursts across the four sessions, median 58.9 s,
p90 83.5 s, longest 103.3 s (fire). So a ceiling of 120 s never cuts an
ordinary burst, and the 16 % headroom is above the largest burst any of the
four has produced at stage 5.

The cost of yielding: each burst line carries `start=` and `end=` beside its
own step-loop `ms`, and the difference between them is everything a phase does
outside the step loop — `ntNewMirror`, `register`, `pullBack`, `free`,
`malloc_trim`. Over the 162 lines that carry both stamps it is a mean of 370 ms,
median 315 ms, p90 599 ms, maximum 1 169 ms. At a 120 s ceiling that is 0.31 %
of training time spent rebuilding the tape, and the whole 4 796.3 s warmup
becomes 41 chunks costing about 15 s of re-entry.

The ceiling is in seconds and not in steps because steps/s spans 215.4 at the
embryo (`air/air.stdout`, `warmup complete: 160 steps … 743ms`) and 0.4 at
stage 5. A chunk of fixed step count would be half a second at one end of
ontogeny and 33 min at the other.

### What landed

`CFG.TrainTurnCeilingSeconds` (120.0) and `CFG.TrainTurnFairSpend` (true), and
`training_turn.go` behind them.

A phase now runs as a sequence of chunks through `ntTrainInTurns`: take the
turn, arm the ceiling, train, mirror the weights back, release, re-queue for
what is left. `ntTrainCore` reads the deadline at a step boundary exactly where
it already reads `trainAborting()`, and the guarantee is the one the abort path
already had — `pullBack` mirrors the tape into `model.Base` on the way out
whatever stopped the loop, so a cut phase is progress kept. Two invariants make
it safe: the deadline is checked only at `step > 0`, so a chunk always advances
and the loop cannot spin; and a chunk that runs nothing ends the phase instead
of paying another rebuild for another nothing.

`trainTurnGrown`, §2.3's first key, is now worth one chunk. §2.3 keys it on an
organism "whose warmup has not run", and after one chunk that sentence is
false. Without the decay the ceiling would be theatre: nothing outranks the
first key, so a warmup would release the turn and win it straight back, at
370 ms a time. The gate for it reads the holder's `priority` column from inside
each chunk.

The queue's second key is now the seconds of tape an organism has had this
session. `training_queue` gained a `spent REAL` column by the same idempotent
`ALTER` as the organisms table's five, fed from an atomic on `SwarmRegistry`
that `ntTrainInTurns` charges around every chunk, and written into the row on
every poll exactly as `priority` is — the figure belongs to the process, and
`ReleaseTrainingLock` deletes the row so that the organism which has just
trained goes to the back. Longest-wait stays available as
`CFG.TrainTurnFairSpend=false`, which is what makes the before/after below
measurable on one binary.

Longest-wait is not fairness because it hands out equal *turns*, and a turn is
not an equal amount of tape: at 76.0 s against 58.0 s, equal turns give the
adult 31 % more of it than the teen. The red run says it in the gate's own
voice — with `TrainTurnFairSpend=false` two organisms 150 s apart in spend take
`map[air:4 earth:4]` of eight turns.

§2.3's middle key, the steepest loss trend, is still not implemented, and the
reason has changed from "there is no column" to a measured one. It could travel
with the request exactly as `spent` does. It would sit below `spent`, and
`spent` is a float of accumulated milliseconds that two organisms never hold in
common, so the tiebreak would never fire. It is left to step 4, where the
resonator's request row carries the trend anyway.

### The before and after

The same synthetic session through both orderings, driving the real
`AcquireTrainingTurn` against a real `mesh.db` with the queue's clock injected
so that 7 200 s replay in milliseconds (`TestTheSameSessionWaitsLessUnderTheNewTurn`).
Every input is one of the measurements above: the four per-organism burst
medians and think times of the 20:00Z session, a measured stage-5 warmup, and
370 ms per chunk boundary. The arms differ as the two versions of the code do —
without the ceiling the warmup is one turn taken at `trainTurnGrown`, with it the
three sub-phases are queued separately and cut into chunks of which only the
first carries the key. It is run twice, once with each of the two stage-5
warmups the artifacts hold.

The warmup measured under the serialised lock, 1 200.9 s, which is what the code
produces today:

```
  origin/main (one turn per warmup, longest-wait)
  burst waiting 4067.3s over 35 waits | bursts 34 | warmup 1200.8s of 1200.9s in 1 chunks, waiting 0.0s | tape busy 3612.6s of 7200s
    earth  bursts= 7 burst_wait= 1304.2s (n= 7)
    air    bursts=11 burst_wait= 1394.8s (n=11)
    water  bursts= 6 burst_wait=   36.5s (n= 6) warmup= 1200.8s in  1 chunks
    fire   bursts=10 burst_wait= 1331.8s (n=11)

  this branch (120 s ceiling, least-spent-first)
  burst waiting 1255.6s over 39 waits | bursts 39 | warmup 1200.8s of 1200.9s in 12 chunks, waiting 540.8s | tape busy 3879.5s of 7200s
    earth  bursts= 8 burst_wait=  372.3s (n= 8)
    air    bursts=13 burst_wait=  433.2s (n=13)
    water  bursts= 6 burst_wait=  103.4s (n= 6) warmup= 1200.8s in 12 chunks waiting 540.8s
    fire   bursts=12 burst_wait=  346.8s (n=12)
```

And the longest one in the files, 4 796.3 s, which predates the lock:

```
  origin/main (one turn per warmup, longest-wait)
  burst waiting 14692.7s over 14 waits | bursts 14 | warmup 4796.3s of 4796.3s in 1 chunks, waiting 0.0s | tape busy 5752.1s of 7200s
    earth  bursts= 3 burst_wait= 4844.2s (n= 3)
    air    bursts= 5 burst_wait= 4921.2s (n= 5)
    water  bursts= 2 burst_wait=    0.0s (n= 2) warmup= 4796.3s in  1 chunks
    fire   bursts= 4 burst_wait= 4927.3s (n= 4)

  this branch (120 s ceiling, least-spent-first)
  burst waiting 2422.5s over 32 waits | bursts 31 | warmup 4796.3s of 4796.3s in 41 chunks, waiting 1944.0s | tape busy 6960.0s of 7200s
    earth  bursts= 7 burst_wait=  529.5s (n= 8)
    air    bursts=13 burst_wait=  728.1s (n=13)
    water  bursts= 0 burst_wait=    0.0s (n= 0) warmup= 4796.3s in 41 chunks waiting 1944.0s
    fire   bursts=11 burst_wait= 1164.9s (n=11)
```

**4 067.3 s → 1 255.6 s** on the first, with bursts completed 34 → 39 and the
tape going from 3 612.6 s busy to 3 879.5 s; **14 692.7 s → 2 422.5 s** on the
second, bursts 14 → 31, tape 5 752.1 s → 6 960.0 s of 7 200. The warmup finishes
in all four runs — under the ceiling it fills the tape the siblings are not
using instead of blocking it — and it pays for that in waiting of its own,
540.8 s over 12 chunks and 1 944.0 s over 41, which is named here rather than
netted out of the totals.

The per-organism figures are the check on the model. Under the new turn on the
worst-case arm the three siblings complete 7, 13 and 11 bursts in a session that
contains a stage-5 warmup, where the 2026-09-16 20:00Z session — which has no
`warmup complete` line in it at all — completed 8, 13, 8 and 12. For a sibling,
a ceiling turns a session with an eighty-minute warmup in it into a session
without one.

Against the 4 737 s of the paragraph above: that figure is four sessions, of
which 3 425.8 s is the one session with a stage-5 warmup in it. The nearest
like-for-like is the first arm, 4 067.3 s of burst waiting for one modelled
session containing one measured serialised warmup, which the ceiling and the
spend order take to 1 255.6 s. The replay is a model — a waiter is admitted the
instant the turn frees, where a live one is up to
`CFG.TrainingTurnPollSeconds` behind, and think time is a constant per organism
rather than a distribution — so its absolute total belongs to the model and not
to a session. Both approximations apply to both arms of both runs.

### The gates, and what red looked like

Six new gates in `training_turn_fair_test.go`; the suite is 254 PASS / 3 SKIP
against 248 / 3 on `0e1e0bc`
(`CGO_ENABLED=1 go test -count=1 -buildvcs=false ./...`, `taskset -c 4-7`).
Each was made red by removing the mechanism, not by editing the assertion:

- ordering, with `CFG.TrainTurnFairSpend=false`:
  `turn 1 went to "earth", want "air" — the order is by arrival, not by spend`,
  and `the organism that had trained least did not get more turns: map[air:4 earth:4]`;
- the grown key's decay, with `p = trainTurnBurst` dropped from the chunk loop:
  `chunk 2 asked at priority 1, want trainTurnBurst (0)`;
- the ceiling over the real step loop, with the `trainTurnCeilingReached()`
  break removed from `ntTrainCore`:
  `the ceiling cut the burst at 20 steps, want exactly 1`;
- the replay, with its second arm run at the old knobs: the two arms come out
  identical and the comparison fails.

The fixture that busies the tape took one round of correction of its own. It
first held the turn through `AcquireTrainingTurn`, which is the thing under
test, so the red run reported `fire could not take the turn to queue the others
behind it` where the finding was four turns each. It now writes the lock row
directly.

`TestNoTwoTrainingPhasesOverlap` stays green with its ungated control arm
overlapping first (`ungated: 30 overlapping cross-organism pairs in 12
phases`): yielding between chunks does not let two phases run at once. The TTL
gate is unchanged and green — a holder killed with `kill -9` still frees the
turn after `CFG.TrainingLockTTLSeconds`, and the refresher still keeps a slow
one.

Also fixed on the way, because chunking would otherwise have created it: the
post-growth freeze counter was decremented by the steps a phase was *asked*
for. A phase cut in two would have charged it twice for one run of steps, which
is the duplicated-invariant pattern of `ff6ad49` with a new cause. Both
`ntBurstTrain` and `ntWarmupTrain` now charge `model.globalStep - g0`, and both
return it, which is what the chunk loop subtracts from what it still owes.

### Not verified

No colony has run this. The next window is 04:00Z and the binary in
`molequla-run` predates the branch (`BUILD`: `81270a6 2026-09-16T22:11:05Z`);
what a live session does to the 444.7 s and to the warmup's wall time is
unmeasured, and so is the re-entry cost at stage 5 specifically — the 370 ms mean
is pooled across all four organisms and all stages. The first-launch climb in
`main()` still takes no turn and has no ceiling, because it runs before
`swarm.Register()` and there is no queue to join; it is reached only by an
organism whose checkpoint did not load, so it is colony genesis rather than every
session, and it is unchanged and unmeasured. The consequence of the grown key's
decay in a
session with a saturated tape is untested in the live case: the replay's
siblings leave the tape idle two thirds of the time and the warmup finishes,
but a colony with four adults and no idle tape would give a warmup a quarter of
the session, and 4 796.3 s does not fit in 1 800 s.

— Defender (Arianna Method, phone-1)

---

## 2026-09-17 — the cap that did not count the hours the phone slept

The 11:00 senses slot took the 12:00 colony session with it. `schedule.log`
has the whole shape of it in one line:

```
2026-09-17T13:28:17Z kind=senses slot=11:00 start=2026-09-17T11:12:01Z
end=2026-09-17T13:28:17Z timeout=600 elapsed=8176 reason=ok frags=5 mem_mb=3494
```

`reason=ok`, under a cap of 600 s, after 8176 s. The cap did not fire and
report a timeout; it reported success. The work itself finished — five
fragments, every organ rc=0 — and the next line in `schedule.out` is
`next slot 2026-09-17T15:00:00Z`, so the colony window that opened at 12:00Z
while the pass was still running was never mentioned again. The daemon is one
process and it sat inside `run_senses` from 11:12Z to 13:28Z; by the time it
looked at the clock again the 12:00 slot was 5 297 s old, past the 1 800 s
catch-up, and gone. The phone was unplugged through all of it
(`senses/senses.log`: `batt=90%->89%,-547->-750mA`).

### Where the hours went, and where they did not

The pass line splits the 8176 s three ways: `eye=rc0,6238s`,
`place=rc1,1373s`, `ears=rc0,44s`. The same window at 09:00 the same morning
took 216 s in total, with `eye=rc0,83s` and `place=rc1,76s`.

The eye's 6 238 s is not evidence about caps. `senses.sh:406-408` runs the ocelli
engine as `/usr/bin/time -v taskset -c "$CPUS" bash "$SENSES_EYE" …` with no
`timeout` around it at all, so a frame that takes twenty-five minutes is
outside every cap in the tree by construction. The eye window's own line says
`n=4 spacing=30s frames=4 … wall=6759s rss=1020mb`, and the four frames landed
517 s, 3 049 s and 2 421 s apart (`world/gen_1789644293_135.txt` …
`gen_1789650280_138.txt`) on a 30 s spacing.

`place` is the one that convicts the cap. Its no-fix branch is exactly two
statements — `timeout 45 $SENSES_SSH 'termux-location -p network -r once'`,
then `timeout 90 $SENSES_SSH 'termux-location -p gps -r once'`
(`senses.sh:699` and `:704`) — and then it returns. 135 s of cap, no other
work, and it reported 1 373 s of wall clock, against 76 s on the same path at
09:00. A cap that lets 1 373 s pass on a 135 s budget is not counting wall
seconds.

It is counting `CLOCK_MONOTONIC`. `readelf --dyn-syms /usr/bin/timeout`
(coreutils 9.4, aarch64) imports `alarm@GLIBC_2.17` and no `timer_create`:
the cap is `ITIMER_REAL`, an hrtimer on the monotonic base, and that clock
stops while the phone is suspended. Measured on the A56 today, from one
process reading all three clocks: `CLOCK_BOOTTIME` 289 341 s against
`CLOCK_MONOTONIC` 225 488 s — 63 853 s of this boot that the cap's clock never
counted — and 53.3 s of a 575 s window lost to suspend while a session was
awake on the phone. `loop()` has known this since it was written: it naps in
minutes and decides each nap against `date`, "because a long sleep does not
count the time the phone spends suspended". The cap never got the same
treatment.

Two things are **not** established. How much of those 8176 s the phone spent
suspended is not recoverable: Android's battery history offers `-running`, and
over the same 575 s window it claimed 543.7 s of not-running against the
kernel's 53.3 s, so it is not a suspend measure and its 4 576 s for the
incident window means nothing; the kernel ring buffer holds about two and a
half minutes of `dmesg` and cannot reach back to 11:12Z. And the cap failing
by a factor of ten on `place` is a demonstration of the mechanism, not a
reconstruction of the whole pass.

### The repair

`run_capped` in `schedule.sh` replaces `timeout` on the senses path. The pass
runs under `setsid`, so its pid is its process group; the watchdog polls
`date -u +%s` in one-second naps, and when the wall clock says the cap is
spent it sends SIGTERM to the *group* and SIGKILL after `SCHEDULE_GRACE`.
Group, not child: `senses.sh` spawns ssh and the eye, and the old line was
`out="$(timeout … )"`, a command substitution that does not return until the
last grandchild closes the pipe — which is why killing the capped child alone
would have freed nothing, and why the daemon stayed blocked for the whole
8 176 s. The output now goes to `$MOLEQULA_RUN/schedule.cap.out` and is read
from there. The group is swept on the clean path too, and the session line
carries `strays=` when there was anything to sweep.

The session line also carries `suspend_s=`, the wall seconds of the pass that
`CLOCK_MONOTONIC` did not count, read from `/proc/timer_list`'s `now at N
nsecs` — which is the monotonic base. `/proc/uptime` is not: on this kernel
its first field is `CLOCK_BOOTTIME` (288 397 s against 224 600 s read
together), so it counts the suspend and would always answer zero.

The colony path needed nothing of this. `run_session` already polls
`date -u +%s` against `t0 + SCHEDULE_DUR + SCHEDULE_GRACE` and calls `stop.sh`
when it passes, and that is what has ended every colony session in the log —
six of six between 2026-09-15 and 2026-09-17 end `reason=overran` at
7 293-7 318 s of a 7 200 s session, which is the daemon's deadline plus one
sample tick, not the `timeout 7200` inside `launch.sh`. That inner `timeout`
has the same monotonic weakness and is kept for what the README says it is, a
backstop for the case where the daemon dies first. What the log cannot say is
whether it fails to fire or merely fails to kill: an organism that ignores
SIGTERM would produce the same `overran` line.

Second half of the incident: `after_slot`. When any slot returns, the loop now
asks which slots of either kind opened while it ran. A senses slot inside a
colony window is not one — it would have been skipped anyway. The newest of
the rest is run late if it is still inside `SCHEDULE_CATCHUP`, which is the
promise the loop already makes to a slot it reaches late; everything else is
logged as `reason=overran-into` with `by=<kind>@<slot>`, so a lost colony
window is a line in `schedule.log` and not an absence. And a senses pass that
finds a colony window nearer than its own cap does not start at all:
`reason=skipped-colony-soon`.

### The gate

`phone1/schedule_test.sh` is 71 cases, was 55, all green. The sixteen new ones
drive the real `run_senses` through the `__slot` verb and the real
`after_slot` through a new `__after`, against a stub that forks a grandchild
holding stdout open — ssh and the eye in miniature.

Red looks like this. Running the new gate against `origin/main`'s
`schedule.sh` (`git show origin/main:phone1/schedule.sh`), the first case
fails with `elapsed=24 against a 5s cap, line … timeout=5 elapsed=24
reason=ok`: the same shape as the incident, three orders of magnitude smaller.
Six of the sixteen go red there — the blocked substitution, `suspend_s`,
`skipped-colony-soon` and three on the swallowed window. The other ten are
regression gates and pass on `main` too, because a gate cannot suspend a phone:
with the machine awake, `CLOCK_MONOTONIC` is the wall clock and `timeout` fires
on time. That is exactly the failure mode this incident is about — the cap
works on every machine that never sleeps.

Not run: a live senses pass. The daemon (pid 10110, restarted 14:15Z on the
pre-fix script) was left alone and `molequla-run` untouched; the branch has not
been near a 15:00 or 17:00 slot. What a real pass reports in `suspend_s`, and
whether SIGTERM alone ends an ocelli engine mid-inference or it takes the
SIGKILL, is unmeasured.

— Defender (Arianna Method, phone-1)

## 2026-09-17 — the eye's own cap, and every cap in the pass counted in wall seconds

The morning's incident left two things undone, and this is both of them. The
outer cap counts wall seconds now (`b4a5f45`), but catching an overrun out there
costs the whole pass; and the command that actually spent the 6238 s — the eye's
engine — had no cap around it at all, so the only thing that could ever end a
hung frame was the 600 s slot cap taking the pass with it. On
`claude/phone1-eye-cap`, not pushed.

### One frame of the eye, measured

The engine on this phone, the same invocation `senses.sh` makes
(`SMOLVLM_NOSPLIT=1`, the q6_k Yent decoder with the q8_0 projector,
`/usr/bin/time -v taskset -c …`), over the eight frames the live 15:00 and 17:00
windows of today left in `molequla-run/senses/frames` (copied out, the live field
untouched), MemAvailable 2941-2984 MB throughout:

| frame | cores 4-7 | cores 0-3 | peak RSS |
|---|---|---|---|
| `20260917T150001Z_cam0` | 13 s | 40 s | 1020 MB |
| `20260917T150031Z_cam1` | 14 s | 42 s | 1020 MB |
| `20260917T150101Z_cam0` | 13 s | | 1020 MB |
| `20260917T150131Z_cam0` | 13 s | | 1019 MB |
| `20260917T170000Z_cam0` | 14 s | | 1020 MB |
| `20260917T170030Z_cam1` | 14 s | | 1020 MB |
| `20260917T170100Z_cam0` | 13 s | | 1020 MB |
| `20260917T170130Z_cam0` | 14 s | | 1020 MB |

Thirteen or fourteen seconds, eight times out of eight, and 1019-1020 MB
whatever the frame is — the same peak the window measured on 2026-09-15 at every
n. The little cores are the other end of it: the same two frames took 40 s and
42 s on cores 0-3, which is the class a pass gets while the colony is awake
(`senses.sh`: `CPUS="${SENSES_CPUS:-0-3}"` when `colony_alive`), three times the
big-core figure.

The live log agrees from a different angle. Over the fourteen windows between
2026-09-16T03:01Z and 2026-09-17T17:01Z, excluding the incident,
`senses.log`'s `eye=rc0,<N>s` over four frames runs 46-83 s, which is 11.5-20.75 s
a frame; the 20.75 s window is 09:02Z, the same pass whose `place` branch went to
the satellites. The incident's own window is 6238 s over four frames, and its
four frames landed 517 s, 3049 s and 2421 s apart on a 30 s spacing.

### 90 s, and the two sides it came from

`SENSES_EYE_FRAME_TIMEOUT` defaults to 90 s. From below: the slowest frame this
phone produces when nothing is wrong is 42 s, on the little cores, and 90 s is
2.1 times that and 6.4 times the big-core median of 13.5 s. From above: the slot
cap is 600 s, and a window of four that loses every frame to this cap costs
4 × 90 = 360 s, which still leaves the ears (21-25 s in every line of the log)
and the no-fix place branch (135 s of cap) inside the slot — where 120 s a frame
would put 480 + 160 over it. Nothing between 42 s and 90 s has been measured on
this phone at all.

What the cap trades away is named honestly: the incident's four frames did come
back — `said=4`, four fragments — after 517 s, 3049 s and 2421 s. Under this
default all four would have been dropped. That is the intended trade. A frame
that answers fifty minutes after its shutter is a description of a world that is
gone, the window it belonged to reported `spacing=30s` for frames an hour apart,
and the slot it ran in is 600 s long.

### The cap counts seconds, and the kill reaches the engine

`cap_run` in `senses.sh` is the shape of `run_capped` in `schedule.sh:240`
rather than a call into it, and the two stay apart because of how they are
invoked. `run_capped` caps one command per slot — the whole pass — from a daemon
that owns `$MOLEQULA_RUN` and is the only writer of `$MOLEQULA_RUN/schedule.cap.*`.
`cap_run` caps nine short commands *inside* that pass, as its grandchild, and
four of the nine need the command's stdout back as a shell value. One shared
function would have to grow an out-parameter and per-call scratch names for a
caller that needs neither, and `senses.sh` would stop being the single file that
runs from any checkout by path — which is how `schedule.conf`'s `SENSES_CMD`,
the gate and every hand-run pass reach it. What is shared is the three lessons:
`setsid` so the command's pid is its own process group, output to a file and
never through `out="$(…)"`, and the group swept on the clean path as well as the
capped one.

The open question the previous entry left — whether SIGTERM alone ends an ocelli
engine mid-inference — is answered. With the engine running under `setsid`, the
group holds four processes: `/usr/bin/time`, the `eye` wrapper, the subshell of
its command substitution, and `ocelli` itself. `kill -TERM -- -<pgid>` emptied
that group in 186 ms, first poll, with no `ocelli` left anywhere on the phone.
So `SENSES_CAP_GRACE` at 5 s is 27 times what the escalation has ever needed,
and the SIGKILL behind it has not yet been seen to do anything. It also settles
why the kill has to be aimed at the group: `senses/ocelli/eye` runs the engine as
`out=$("$DIR/ocelli" …)`, so a signal to the wrapper alone would leave a
gigabyte of weights resident with nobody reading it.

### Nine caps, not six

Six `timeout` calls are gone from `senses.sh` — the camera grab (60 s), the
describer (60 s), the recorder (30 s), the two location fixes (45 s and 90 s) and
the ledger ingest (120 s) — each replaced by `cap_run` at the same number, each
number now a variable with its measurement beside it. The seventh cap is the
eye's new one. The last two came out of the gate going red for a reason it was
not written for: the ears' case hung for 62 s at a 2 s cap, because
`$SENSES_SSH "termux-microphone-record -q; rm -f …"` and its twin after the
recording were plain uncapped `ssh` calls, and so was the eye's `rm -f` of the
previous frame. Three remote commands whose result is ignored, which could hang
the pass exactly the way the engine did. They now carry their own organ's
number.

A capped frame is a logged fact and not a silence: `cam<N>:timeout` in the pass
line beside the notes that were already there (`cam0:capture`, `cam1:scale`,
`skip-mem:1180`), `eye=rc124`, and the frame counted in `frames=` but not in
`said=`, so the `eyewin` line says three frames taken and two described. And the
frame is dropped whole. The return is before the engine's output is parsed,
because what a killed engine has written is half a sentence: the gate's stub
flushes `OURS: "A half-written sen` and then hangs, and neither
`dna/output/world/` nor `facts.jsonl` gets a line out of it.

### The gate

`phone1/senses_test.sh` is 42 cases, was 20, all green. Four drive `cap_run`
through the library door `senses_facts_test.sh` opened, and the rest drive the
real script with the hardware faked: an engine that hangs on the second frame of
three, an engine that forks a grandchild before it hangs, an `ssh` that hangs on
whichever remote command `SENSES_TEST_SLOW` names, and a describer and an ingest
that do nothing but sleep.

Red looks like this, running the new cases against `origin/main`'s `senses.sh`
(`git show origin/main:phone1/senses.sh` into a scratch `phone1/`):

```
25 pass, 17 fail

FAIL a hung frame leaves the other two fragments: got '3', want '2'
FAIL the window took its three frames and heard two: got '… eyewin pattern=0,1,0
n=3 spacing=0s frames=3 said=3 repeat=1 novel=0.667 wall=61s …'
FAIL a capped frame writes no fragment: '…/dna/output/world/gen_1789667606_2.txt'
FAIL and no fact line for it: got '3', want '2'
FAIL the engine's grandchild is killed with the group: pid 8087 is still alive
FAIL and the branch costs four seconds, not forty: 106s
```

Seventeen of the twenty-two go red. The window waits out the whole 60 s hang and
then counts the hung frame as a description, `said=3`; the half sentence
`OURS: "A half-written sen` becomes a fragment and a third fact line, which is
the eye reporting something it never finished seeing; the engine's grandchild is
still alive after the pass has returned; and the two location fixes cost 106 s
against the 4 s they cost now, because on `main` the substitution around
`timeout 45` does not return until the stub's own `sleep 60` closes the pipe —
the incident's mechanism at a twentieth of the scale. The five new cases that
pass on `main` are the five that only read the note and not the clock: with the
phone awake `timeout` does fire, eventually, so `cam0:capture` and `record` and
`no-fix` do land in the log — after 60 s, 90 s and 106 s.

### One live window, capped

A real pass on a scratch field, `MOLEQULA_RUN` under the scratchpad and
`molequla-run` untouched, 2026-09-17T18:01:51Z, cores 4-7, MemAvailable 2958 MB:

```
2026-09-17T18:02:28Z eyewin pattern=0,1 n=2 spacing=0s frames=2 said=2 repeat=0
novel=1.000 wall=37s rss=1020mb batt=100%->100%,-6->-4mA cpu=4-7
```

Both frames came back — `wall_s` 14 and 16 in their facts against the 13-14 s
measured outside the cap, so the quarter-second poll is inside that difference —
peak RSS 1020 MB, two fragments, two facts, and `pgrep` for `eye|ocelli|termux-camera`
found nothing afterwards. The two sentences were «A black screen with a small
white text that reads "the world is not what it seems"» and «A room with a white
ceiling and a small air conditioning unit on the wall, with a small hole in the
wall above the unit», which is the room the phone is in. The engine came from the
phone's main checkout (`SENSES_EYE=…/molequla/senses/ocelli/eye`): a worktree has
no built `ocelli` in it, and the first attempt at this window recorded
`eye=rc4,cam0:rc4,cam1:rc4` — `engine not built` — which is the wrapper's own
exit code arriving through the cap unchanged.

### What is not established

The frame distribution is eight frames on an idle phone and two on the little
cores, all of one scene at one hour; it is not a distribution over the day's
lighting or over a phone that is busy with something else. The 90 s default has
therefore never been approached by a healthy frame, which is the point, and also
means the number has not been tested from the side that matters — no frame
between 42 s and 90 s has ever been observed, and if one exists this cap will
kill it. No cap has ever fired on the real engine inside a pass either: the
186 ms figure comes from SIGTERM sent to a live engine's group by hand, and the
window that fires in the gate is a stub. Nothing here measures a suspended
phone: like the outer cap's gate, these cases pass on `main` wherever
`CLOCK_MONOTONIC` is the wall clock, and the replaced caps go red there only
where a cap has to fire at all.

— Defender (Arianna Method, phone-1)

## 2026-09-17 — an organism says which checkpoint it resumed from

Steps 3 and 4 of `docs/resonator_design.md` both need to know which file an
organism was rebuilt from, and until today the log could not say. `LoadCheckpoint`
(`molequla.go`) and `loadCheckpointGGUF` (`checkpoint_gguf.go`) printed only on
the paths that *refuse* the binary sibling — `not used (%v)`, `not used (the
JSON's identity is unreadable: %v)`, `is older than the JSON checkpoint`. Success
was silent, and so was a boot that found no checkpoint at all, so "resumed from
the mapped GGUF", "resumed from the JSON" and "no GGUF existed yet" all left the
same mark in `molequla-run/*/*.stdout`: none. The GGUF sibling was first written
at 2026-09-16T21:49Z, which makes tonight's 20:00Z window the first session that
can resume from one, and without this it would not have said whether it did.

### One line, whichever file won

```
[ckpt] resumed from molequla_ckpt.gguf — 28.2 MB, read in 176 ms, peak +156 MB
[ckpt] resumed from molequla_ckpt.json — 267.9 MB, read in 8926 ms, peak +396 MB
```

Those two are measured, not illustrative: a scratch resume of the live air and
water checkpoints copied out of `molequla-run`, each in its own directory with
its own `HOME` and `MOLEQULA_RUN`, pinned with `taskset -c 4-7` and cut by
`timeout` just after the load. Air had its binary sibling beside it and read it;
water's copy had the sibling removed, so it took the JSON path.

The cross-format ratio needs one organism and one pair, so air was resumed a
second time from a fresh copy of its JSON with no sibling beside it:

```
[ckpt] resumed from molequla_ckpt.json — 146.9 MB, read in 3258 ms, peak +208 MB
```

28.2 MB in 176 ms against 146.9 MB in 3 258 ms is **18.5x**, on the same
organism's own two files. Step 2 recorded 146 ms and 4 136 ms for the same pair
and called it 28x; both of these are slower and faster respectively on
checkpoints that have grown since and read from `/root` rather than from the run
root, and one run per format is not a distribution. The first attempt at this
measurement was wrong for a reason worth writing down: it compared air's GGUF
against *water's* JSON, two different organisms and two file sizes, and called
the quotient a ratio. The second attempt was wrong too — it read air's JSON out
of the probe directory the earlier probe had run in for ninety seconds, so the
file was the 310.3 MB checkpoint that probe had itself saved, not the 146.9 MB
one the colony holds.

The path in the line is a bare basename because `phone1/launch.sh` cds into the
organism's directory (`cd "$dir"`) and `CFG.CkptPath` is relative. A mitosis
child is told an absolute `ckpt_path` in its birth config and prints that, so
both shapes occur and both are in the table's fixture. That was not a guess
about the format — the first version of the day's table filtered a line by
finding `/<element>/` in its path, which rejected every line the colony actually
prints, and the probe above is what found it.

`peak +N MB` is `VmHWM` before and after the read (`ownPeakRSSMB`,
`governor_phone.go`). It is a floor rather than a total, and it is a measurement
of the read only because the read happens at boot, before the organism is
anywhere near its largest. When the high-water is already above what the read
needs the subtraction stops meaning anything, so the line drops the field
instead of printing `+0 MB` — which is what happens under `go test`, where the
binary has loaded larger organisms already, and why the gates below assert the
size and the millisecond and not the peak.

The three silent boots now say so too, in three wordings rather than one, since
a checkpoint that was there and could not be parsed is a loss of weights and not
a beginning:

```
[ckpt] no checkpoint at molequla_ckpt.json (open molequla_ckpt.json: no such file or directory) — the organism starts from an embryo
[ckpt] molequla_ckpt.json could not be read (checkpoint field "base": EOF) — the organism starts from an embryo
[ckpt] zero-warmup mode — no checkpoint read, the organism starts from an embryo
```

The refusal lines are untouched. A refused sibling prints its refusal *and* the
resume line naming the JSON, and the resume clock and the high-water are re-read
after a refusal so the JSON read is not charged for the binary attempt.

### Countable per session

`phone1/daily.sh` gains a table above the DNA traffic, read per organism since
that organism's last `[ecology] Element:` banner — the same scoping the
cafeteria table uses, and against the same error. Counting over the whole file
was mine on 2026-09-16 (the re-count entry above): these stdout files are
appended across every session ever run, so a whole-file reading answers about
some earlier boot. On a copy of tonight's real files with the two probe lines
spliced in as a new session:

```
org    from   file                  size-MB  read-ms  peak-MB
earth  -      (no [ckpt] line)            -        -        -
air    gguf   molequla_ckpt.gguf       28.2      176     +156
water  json   molequla_ckpt.json      267.9     8926     +396
fire   -      (no [ckpt] line)            -        -        -
```

Earth and fire read `(no [ckpt] line)` because there is no `[ckpt]` line in
their whole history — that is the gap this entry closes, now visible instead of
silent.

### Gates, and what red looked like

Six in `checkpoint_says_test.go`. On `origin/main` four of them reported
`stdout was ""` — nothing printed at all — and the fifth,
`TestRefusedSiblingStillSaysTheResume`, captured only
`[ckpt] …molequla_ckpt.gguf not used (identity 3227d7d4f5c2 does not match the
JSON checkpoint's 331b264dc25b) — loading the JSON checkpoint` with no resume
line after it. The five checkpoint gates the step-2 entry landed —
`TestGGUFLossParityWithJSON`, `TestGGUFTruncationFallsBackToJSON`,
`TestGGUFForeignTokenizerRefused`, `TestGGUFOlderThanJSONRefused`,
`TestGGUFLoadBuildsNoThrowawayModel` — are green beside them, with their
refusals firing verbatim.

Four more in `phone1/daily_test.sh`, on a fixture where earth has two starts
that resumed from different files, air's last start printed nothing while the
one before it resumed, and fire resumed and then, in a further start appended to
the same file, found no checkpoint. Removing the banner reset from the table's
awk turns air's row from `air - (no [ckpt] line) - - -` into
`air gguf molequla_ckpt.gguf 28.2 131 +126` — the previous session's answer
reported for a session that has none, which is exactly the shape of the
2026-09-16 error. That was run, not reasoned: the reset was deleted, the gate
went to `11 pass, 1 fail`, the reset was restored.

`CGO_ENABLED=1 go test -count=1 -buildvcs=false ./...` on `3fd1092` is 254 pass,
3 skip (not the 246 the brief carried); with these six it is 260 pass, 3 skip.
`bash phone1/daily_test.sh` is 14 pass, 0 fail, up from 10.

### What is not established

One full-suite run of six on this branch failed
`TestBeatKeeperRefreshesMeshWithoutTicks` — `a hibernating organism was written
back as "alive"` — and the other five were clean, as were eight runs of the
suite on `3fd1092` in a detached worktree. The test asserts a state change
through `mesh.db` inside an 80 ms sleep against a 20 ms keeper interval, which
is a window four little cores under a full suite can miss, and it passed three
times in isolation. Nothing in this change reaches the keeper, the mesh or
anything that test calls: it never constructs a checkpoint. That is an
observation about a timing window and not a diagnosis, and one flake in six is
not a rate.

The resume line has never been printed by a scheduled colony session. The probes
above are three single-organism runs started by hand, outside a window, and the
first session to exercise it is 20:00Z tonight. The 18.5x ratio is one
measurement per format on one organism's own pair of files, not a distribution,
and no organism has yet been resumed from a GGUF it wrote in a scheduled
session. Nothing here touches what the sleeper of step 3 or the resonator of
step 4 will do with the answer; it only makes the answer exist.

— Defender (Arianna Method, phone-1)

## 2026-09-17 — the night the organisms woke from a mapped file, and the machine ran out of room

The 20:00Z session ran on `79a4461`, the first binary that says where a boot
came from. It said it four times:

```
[ckpt] earth resumed from molequla_ckpt.gguf — 49.4 MB, read in 244 ms, peak +258 MB
[ckpt] air   resumed from molequla_ckpt.gguf — 28.2 MB, read in 184 ms, peak +155 MB
[ckpt] water resumed from molequla_ckpt.gguf — 50.8 MB, read in 307 ms, peak +268 MB
[ckpt] fire  resumed from molequla_ckpt.gguf — 27.3 MB, read in 282 ms, peak +149 MB
```

**All four resumed from the mapped binary checkpoint** — the first live proof
of step 2, against the bench's own JSON figures for the same organisms, 3258 ms
for air and 8926 ms for water. Eighteen and twenty-nine times faster, and the
line that says so is now in the log rather than in a measurement someone has
to repeat.

**The colony has run out of room, and the session line is what says it.** With
the samplers of `81270a6` finally in the running script: `rss_sum_max_mb=3563
at 20:28:34Z mem_min_mb=498`, against per-organism peaks of earth 2408, water
2370, fire 2299, air 2237 MB. **498 MB of free memory is the lowest this node
has recorded**, and Samsung's lmkd was working during the session — `am_kill`
took `com.sec.unifiedwfc`, `turbo`, `acore` and two calendar providers at
21:30:57Z and 21:52:16Z (device-local 00:30 and 00:52). It took no organism and
it did not take Termux; there was still cache to reclaim before it reached
`oom_score_adj=500`. The sum of resident sets is the number to watch from here,
and the reason the sleep policy of step 5 stopped being an optimisation: air
and fire are stage 4 and asked to grow 9 and 11 times this session, and four
adults at these peaks do not fit in this phone.

**The turn's ceiling fired, and only where it should have.** Three times, all
on `water`, the adult with the longest bursts: `burst complete: 31 steps (cut at
the turn's ceiling, 32 requested) … 120624ms`, then `burst yielded the turn
after 121.6s, 1 of 32 steps left (spent 316.7s this session)`. The work is not
lost — the step count says 31 of 32 and the phase resumes — and the other three
organisms never met the ceiling, which is what a 120 s cap set above the
longest honest burst is supposed to do. Queue waiting this session: twelve
waits, 540 s in total (air 251, water 184, fire 86, earth 19), against 444.7 s
the night before with no ceiling at all.

**Everything else held.** No `mesh refused the heartbeat` for the fourth
session running. The cafeteria declined 29-32 % for every organism, the third
session inside ten points. Bursts: fire 11, air 10, water 8, earth 7; DNA
written fire 111, air 99, earth 31, water 27, with `eligible=1` on 111, 44, 31
and 12. The ledger stands at `world 155 facts/10 open`. The wall-clock caps of
`b4a5f45` and `3fd1092` carried two senses slots with the new fields in their
lines — `suspend_s=1 strays=0` at 17:00Z and `suspend_s=0 strays=0` at 19:00Z,
136 s against a 600 s cap, nothing left behind.

**One thing the prekill does that was not intended.** `am kill-all` at the
session's start killed `com.termux.api` (logcat `am_kill … kill all background`
at 20:00:05Z), which is the process `termux-camera-photo`,
`termux-microphone-record` and `termux-location` reach through. No senses slot
runs inside a colony window, and Android restarts it on demand, so nothing was
lost tonight; it is recorded because a future pass that starts seconds after a
colony session ends will pay for it.

— Claude (Defender, Arianna Method)

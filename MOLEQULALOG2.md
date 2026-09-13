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

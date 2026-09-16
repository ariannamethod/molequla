# The resonator, the sleeper, and the memory that is the environment

A design document, not an implementation. It describes two coupled changes Oleg
approved in principle on 2026-09-15 — the environment shaping the population and
its training tempo, and organisms that are not training living on flash rather
than in RAM — together with the third piece that makes both of them cheap: one
shared training process per colony, working name **resonator**. Nothing here is
built. The purpose of the document is that the building can be briefed one step
at a time, each step with a gate that can go red.

Every number below is either a measurement with its source named inline, or an
estimate that says so in the same sentence. The measurements come from the
2026-09-15 entries of `MOLEQULALOG2.md`, from `molequla-run/schedule.log`, from
`reports/2026-09-15_phone1_android_memory/README.md`, and from two probes written
for this document and described in section 3. The estimates are marked
**(estimate)** wherever they appear, and every one of them names the measurement
it was scaled from.

---

## 0. The problem in one paragraph

Four stage-4 organisms of 4.84 M parameters each — nineteen megabytes of float32
weights apiece — left `hwm_mb=earth:755,air:1032,water:820,fire:1091` over a
two-hour session (`molequla-run/schedule.log`, `2026-09-15T06:01:55Z`, 237
samples). Android's floor on this phone is about 2.4 GB of the 7 425 MB the
kernel sees (`reports/2026-09-15_phone1_android_memory/README.md` §1, §2.2), so
the sum of those four peaks plus the floor is 6 113 MB and the colony saw
MemAvailable at 1.4-1.5 GB for the whole second hour. The growth byte gate
refused adulthood thirty-two times in that session — earth 6, air 5, water 13,
fire 8 — reading `free=1467 MB need=2188 MB` and its siblings, because the need
is three times the organism's own peak plus 256 MB. A single unpinned probe that
was allowed to reach stage 5 took `VmHWM` to 1318 MB. Four of those would want
5 272 MB, which with the Android floor is 7 672 MB against a MemTotal of 7 425 MB:
**a colony of four adults does not fit on this phone under the present
architecture, and it does not miss by a little.**

None of that memory is model. Of the 900 MB an organism holds, 19.3 MB is
weights. The rest is the machinery of learning — one tape, one float32 mirror,
one set of activations and their gradients, one set of Chuck moment slots, and
one glibc arena that ratchets upward as the tape builds and tears down 81.7 MB of
activations thirty-two times per burst. Four organisms carry four copies of that
machinery and use it for a few seconds each per tick.

---

## 1. Memory model, today and proposed

### 1.1 Where a stage-4 organism's memory is now

These are the measured buckets, from the 900 s single-organism probe of
2026-09-15 run from a copy of the live `earth` checkpoint with
`--max-growth-stage 4`, and from the tape census taken between backward and
clear at T=96, D=224, L=5, V=750.

| bucket | MB | side | belongs to |
|---|---|---|---|
| model weights and gradients, float64 | 87.3 | Go live heap | the organism |
| n-gram and co-occurrence field | 12.2 | Go live heap | the organism |
| delta snapshot held across a burst | 1.5 | Go live heap | the organism |
| corpus `docs` | 1.0 | Go live heap | the organism |
| heap the collector holds and has not returned | 47-147 | Go | neither |
| activations | 31.3 | C tape | the burst |
| gradients (activations 30.5 + parameters 19.9) | 50.4 | C tape | the burst |
| parameter mirror, float32 | 19.9 | C tape | the burst |
| Chuck moment slots, m and v | 38.1 | C tape | **the organism** |
| free chunks the allocator keeps | up to 265 | C arena | neither |
| process and runtime at start | 11 | — | the organism |

Two rows in that table are the whole design. The four C-side rows above the
arena — 139.7 MB of live tensors at the backward — exist only while a burst is
running, and a burst is 32 steps in about 7 400 ms out of a tick loop that
otherwise does not train. The arena row is the residue of those four: nothing is
leaked, every tensor is freed, and between bursts the allocator reports one to
four megabytes of live C memory while holding 148 to 265 megabytes of free
chunks. `releaseTrainingHeap` (`heap_trim.go`) now returns those with
`malloc_trim(0)` once per training phase, which took the mean resident set of the
same probe from 474 MB to 272 MB and stopped the high-water mark climbing after
the second burst. That fix bought back the residue. It did not change the fact
that the machinery itself is replicated four times.

The one row that is per organism and not per burst is Chuck's moment slots. They
are positional, they are m and v for every registered parameter, and at 4 834 408
parameters in float32 they are 38.68 MB of arithmetic against 38.1 MB measured —
the difference being the frozen gate vectors, which take no optimizer slot
(`ntTapeParamFrozen`, `notorch_trainer.go:234`). Whatever else moves, the moments
travel with the organism.

### 1.2 What the proposal moves where

| what | today | proposed | measured or estimated |
|---|---|---|---|
| weights | float64 with gradients, 87.3 MB in the organism | float32, 19.34 MB, mmap'd from the organism's own file | 4 834 408 × 4 B = 19 337 632 B, the probe file |
| Chuck m, v | 38.1 MB inside the burst's tape | 38.68 MB in a file beside the weights, resident only during the organism's turn | arithmetic, 2 × 19 337 632 B |
| tape, activations, gradients | 139.7 MB, four copies | 139.7 MB, one copy, in the resonator | measured, tape census |
| glibc arena, OpenBLAS buffers | ratchets in four processes, 159-318 MB at a peak | ratchets in one process | measured, `mallinfo2` at burst 1 and 2 |
| co-occurrence field | 12.2 MB | 12.2 MB, unchanged, in the organism | measured |
| corpus, cursors, runtime | ~12 MB | unchanged | measured at process start, 11 MB |

An organism under this model holds its field, its corpus, its cursors, its Go
runtime, and a mapping of its own weights. The probe in section 3 measures what
that mapping costs: 19.97 MB of resident set while the weights are being read for
a generation, and 1.16 MB after `madvise(MADV_DONTNEED)` when they are not.

| per stage-4 organism | today | proposed | note |
|---|---|---|---|
| mean resident, not training | 272 MB | ≈ 25 MB asleep, ≈ 44 MB while speaking | today measured; proposed is the sum of measured parts |
| high-water during a burst | 599 MB isolated, 755-1091 MB in session | the burst is not in this process | |
| on disk | 110 599 980 B of JSON | 19 337 632 B of weights + 38 675 264 B of moments | measured file, arithmetic |

The Go heap ceiling deserves its own sentence, because it is the largest
remaining uncertainty. `HeapSys` reaches 311-371 MB for a 102 MB live heap
because the corpus rebuild churns hard against a default `GOGC`. Removing the
87.3 MB float64 pair removes both the largest single allocation and most of the
copying, so the ceiling should fall roughly in proportion — **(estimate**, scaled
from the ratio of live heap to `HeapSys` measured at fifteen stage-4 snapshots;
it has to be measured on the changed binary before it is claimed.**)**

### 1.3 The resonator's own footprint

| bucket | MB | source |
|---|---|---|
| live C tensors at the backward | 139.7 | tape census |
| arena plus mappings at a burst peak | 214 (burst 1, step 31) to 352 (burst 2, step 31) | `mallinfo2` |
| arena plus mappings between turns, after `malloc_trim` | 25-28 | measured C side between bursts |
| Go side: the queue, the file handling | ~15 | **(estimate**, a Go process with no model in it starts at 11 MB measured**)** |

The resonator's peak is therefore about 350 MB and its idle about 40 MB, both on
the stage-4 shape. It is one process, so it ratchets one arena, and the ratchet
stops where `malloc_trim` leaves it.

### 1.4 Four adults in 8 GB — the arithmetic under each model

**Stage 4, today.** Android floor 2 400 MB, four organisms at their measured
peaks 755 + 1032 + 820 + 1091 = 3 698 MB, witness 15 MB. Total 6 113 MB of
7 425 MB, leaving 1 312 MB — which is the MemAvailable of 1.4-1.5 GB the colony
actually saw, so the ledger closes. The growth gate wants three times an
organism's peak plus 256 MB, that is 2 521 MB at the smallest measured peak and
3 529 MB at the largest, and neither was ever available. This is the session in
which growth was refused thirty-two times.

**Stage 4, proposed.** Four organisms at 44 MB while speaking, one resonator at
its 350 MB peak, witness 15 MB: 541 MB. With the Android floor, 2 941 MB of
7 425 MB, leaving about 4.5 GB. The peaks do not coincide because there is only
one of them.

**Stage 5, today.** The one measured stage-5 high-water is 1318 MB. Four of them
is 5 272 MB, plus the floor is 7 672 MB against a MemTotal of 7 425 MB. It does
not fit. zram compresses at 3.46 : 1 on this workload and would move some of that
into its 4 GB device, but the working set during a burst is hot by definition, so
what zram would buy is major faults in the middle of training rather than
headroom. The honest statement is the one repair 9 already made: no stage-5
organism is to be allowed on this phone until its peak is known.

**Stage 5, proposed.** Stage 5 is D=320 and L=6 against stage 4's D=224 and L=5
(`molequla.go:351-352`), about 10 M parameters by the config's own comment, so
40 MB of float32 weights per organism. Four organisms at 11 + 12.2 + 1.0 + 40 =
64 MB each is 256 MB. The resonator's tape scales with T·D and with the layer
count, a factor of about 1.7 on the stage-4 shape, giving a peak near 600 MB —
**(estimate**, anchored on the single measured stage-5 whole-process high-water
of 1318 MB against stage 4's 613 MB; the whole-process ratio of 2.15 includes the
Go float64 side that this design removes, which is why the C-side-only ratio is
smaller, and none of this is a measurement.**)** Total with the witness is about
871 MB, and with the Android floor about 3.3 GB of 7 425 MB. **Four adults fit,
with room for the eye's 955 MB burst beside them**, which is the answer the
colony has not had.

---

## 2. The resonator protocol

### 2.1 Process or goroutine

A goroutine in one of the organism processes is the smaller change and the wrong
one. The argument is entirely in the measurements. The tape, the mirror, the
activations, the gradients and the OpenBLAS buffers are C-side allocations, and
the thing that actually costs memory is not any one of them but the arena that
holds their freed chunks: 87 → 159 → 318 MB across two bursts, with in-use bytes
of 84, 85 and 106 MB at the same three moments. An arena belongs to a process.
Four processes ratchet four arenas; one process ratchets one, and
`malloc_trim(0)` resets it at a phase boundary where nothing is in flight. There
is no way to get that boundary without a process boundary, because the organisms
are separate processes today — `performMitosis` spawns a child with
`cmd.Process`, and `phone1/launch.sh` starts four of them.

The objection is real: cross-process weight transfer costs copies. It costs less
than it appears to, because the copy is already there and already paid every
burst. `ntNewMirror` flattens every `MatrixParam` into a fresh float32 array
(`ntFlattenMatrix`, `notorch_trainer.go:90`) and `pullBack` unflattens it
(`ntUnflattenMatrix`, `:105`); the float64-to-float32 transposition is the
existing boundary, and the resonator only changes where the bytes land. Measured
against the probe file of exactly the right size: a `read()` of all 19.34 MB into
a heap buffer took 11.2 ms, and faulting the same bytes in through a mapping from
warm page cache took 5.98 ms. A round trip — weights out, weights and moments
back — is therefore on the order of 60 ms against a 32-step burst that takes
7 400 ms. Eight tenths of one percent.

So: a separate process, for four reasons, in order of weight.

1. The arena boundary is the point of the exercise and it is per process.
2. A resonator that dies takes nothing with it. The organism still owns the file
   it handed in; it is deleted only after the returned file is renamed into
   place, which is the atomic temp-and-rename pattern `SaveCheckpoint` already
   uses (`molequla.go:3693`).
3. OOM attribution becomes honest. `applyOomScoreAdj` (`governor_phone.go`) can
   make the resonator — the process actually holding 350 MB — the one lmkd
   reaches for, instead of an organism that is holding its own life's work.
4. One OpenBLAS thread pool instead of four. `capColonyThreads` exists because
   four pools on one cgroup thrashed, and this removes the cause rather than
   capping it.

### 2.2 What crosses the boundary

A file, twice, and a row in `mesh.db`.

The checkpoint has to stop being JSON for this to be worth doing. Today it is
110 599 980 B for 4 834 408 parameters, which is 22.9 bytes per parameter of
decimal text for a value the tape trains in float32, and loading it costs 152 MB
of resident set over about two seconds even after repair 10 turned the decoder
into a streaming walk. A float32 binary of the same weights is 19 337 632 B, a
factor of 5.7, and — the part that matters for section 3 — it can be mapped
instead of read.

The format should be GGUF, because the reader already exists and has been
measured on this phone. `/usr/local/include/ariannamethod/gguf.h` gives
`gguf_open` returning a struct whose `data` is "a read-only mapped view of the
file" with `map_base` and `map_len` beside it (gguf.h:64-70), `gguf_find_tensor`,
`gguf_dequant` for a whole tensor and `gguf_dequant_row` for one row in place,
and `gguf_read_str_array` for the type-9 string arrays that a tokenizer's tokens
and merges already are. On the compute side `nt_qmatvec` and its int8 relatives
(`notorch.h:528-620`) do packed matvec for the GGUF types directly, and seven of
those types passed on this A56 at rel err ~1e-6
(`memory/milestone_notorch_qmatvec_phone1_verified_2026_06_06`). What does not
exist anywhere in notorch is a **writer**: `nt_save` writes its own format,
`[magic][n][ndim, shape[], data[]]` (`notorch.h:800`), and there is no
`gguf_write`. That writer is the one genuinely new piece of infrastructure in
this document, and `~/arianna/notorch/gguf_quantize` — already used to quantize
the eye — is the model for it.

Three files per organism, in its own working directory:

| file | contents | size at stage 4 | mapped by |
|---|---|---|---|
| `weights.gguf` | every `MatrixParam` and delta factor as F32 tensors, plus tokenizer, cfg, `global_step`, `growth_step_offset`, `last_warmup_stage`, `corpus_ingested_total` as KV | 19.34 MB + metadata | the organism (read-only, for generation) and the resonator (read-only, to fill its mirror) |
| `moments.gguf` | Chuck m and v, in the tape's registration order | 38.68 MB | the resonator only |
| `molequla_ckpt.json` | unchanged | 110.6 MB | the C, Rust and JS cores |

The JSON checkpoint stays. `CLAUDE.md` is explicit that a checkpoint written by
the C, Rust or JS core still loads, and `readCheckpointStream` is written not to
assume field order for exactly that reason. The binary format is added beside it,
the JSON becomes the interchange format between cores rather than the hot path,
and a converter in both directions is part of step 2 below.

Registration order is load-bearing and has to be written down in the file rather
than assumed. Chuck's moment slots are positional — `ntMirror.register()`
registers content parameters, then delta adapters, then RRPRAM factors, in that
fixed order, and the comment at `notorch_trainer.go:141` says why. The moments
file therefore carries the ordered list of tensor names as a KV array, and the
resonator refuses a moments file whose order does not match the weights it was
handed. A mismatch is not a rounding error; it is Chuck applying earth's second
moment to fire's embedding table.

### 2.3 The queue

Three locks already exist in `mesh.db`, all of the same shape: `training_lock`
with a 30 s TTL, `mitosis_lock` with 30 s and a head-count condition in the same
statement, and `growth_lock` with a 300 s TTL and a refresher goroutine that
re-stamps it every 60 s (`molequla.go:6016-6100`). The queue is not new work; it
is `training_lock` promoted from an optional coordination flag to the actual
admission gate, with the growth lock's refresher shape so that a TTL bounds a
dead holder rather than a slow one.

`CoordinateWarmup` is false by default, and the comment at `molequla.go:7066`
explains why: on a GPU host that serialisation cost three of four organisms their
whole tick, because the lock's `continue` skipped the entire tick body rather
than just the burst. That failure is a property of *where* the lock is taken, not
of serialising. Under the resonator, an organism that cannot get a turn continues
its tick — it reads DNA, writes DNA, advances its ontogenesis clock, generates —
and only its burst waits. The tick does not skip.

Who trains next, in order of precedence:

1. **An organism that has just grown.** Growth and the warmup behind it are one
   memory event, already serialised by `growth_lock` held across both (repair 9).
   A grown organism whose warmup has not run is at stage N+1 untrained, and
   leaving it there is the duplicated-invariant bug pattern the tree already paid
   for once. The growth lock holder goes to the head of the queue, and holds both
   locks until the warmup finishes.
2. **The organism with the steepest loss trend.** `SyntropyTracker.BurstHistory`
   already carries `LossBefore` and `LossAfter` per burst, and `syntropy` is
   already a column in `mesh.db` written by `Heartbeat`. The ordering key is the
   mean of the last eight burst deltas — the same window `shouldHibernate` uses,
   reused rather than invented.
3. **The organism that has waited longest.** A plateaued adult still gets turns;
   it just gets them last. Starvation is a bug, not a policy.

Amended 2026-09-16, after step 1 ran for four sessions. Three things about this
order changed, and step 4 inherits them rather than the list above.

**A turn is bounded.** `CFG.TrainTurnCeilingSeconds` (120 s) is the longest one
phase may hold the turn; past it the phase mirrors its weights back, releases and
re-queues for the steps it has left. The two numbers behind the default are
measured on the phone across the four sessions in `molequla-run/*/*.stdout`: the
longest single micro-burst of 233 was 103.3 s, so an ordinary burst is never cut,
and re-entry — mirror in, `pullBack`, `free`, `malloc_trim`, read as a burst
line's `start=`/`end=` wall minus its own step-loop ms — is a mean of 370 ms over
162 phases, so yielding every 120 s costs 0.31 % of training time. §2.5 above
says the organism "hands in a turn whose step count is the whole warmup": it does
not any more, and the resonator must not either. The two complete stage-5 warmups
in the artifacts are 1 200.8 s (earth, under the serialised lock, 1.67 steps/s)
and 4 796.3 s (water, before it, 0.42 steps/s) for the same 2 000 steps, and a
request that cannot be interrupted for twenty minutes — let alone eighty — is the
same wall whichever process serves it. The resonator hands *back* every ceiling,
or it inherits the queue the ceiling was written to drain.

**Key 1 is worth one chunk.** "An organism that has just grown" means one whose
warmup *has not run*, and after the first chunk that is no longer true. Without
the decay the ceiling does nothing: nothing outranks key 1, so a yielding warmup
wins its own turn straight back, at 370 ms an attempt.

**Key 3 is spend, not wait.** Longest-wait hands out equal turns, and a turn is
not an equal amount of tape — the adult's burst is a median 76.0 s against the
teen's 58.0 s, so equal turns give the adult 31 % more of it. The order below key
1 is now the seconds of tape an organism has had this session, ascending, with
longest-wait as the tiebreak, carried in a `spent REAL` column of
`training_queue` written on every poll exactly as `priority` is
(`CFG.TrainTurnFairSpend`, true; false restores longest-wait).

Key 2, the loss trend, is still unimplemented, and the reason is no longer the
schema. It could travel with the request exactly as `spent` does. It would sit
below `spent`, and `spent` is accumulated milliseconds that two organisms never
hold in common, so the tiebreak it would occupy never fires. Under the resonator
the request row of §2.6 carries the trend anyway, and there it is free.

The byte gate stays where it is and changes its input. `growthGateDecision`
charges three times the organism's own `VmHWM` plus a floor, and under the
resonator the organism's own high-water is about 44 MB while the event being
gated happens in another process entirely. The gate must read the resonator's
peak, and the factor of three — which was measured against what a stage step does
to a single all-in-one process — has to be re-measured against the resonator's
shape before it is carried over. Carrying it over unmeasured would be exactly the
arbitrary threshold this tree does not ship.

### 2.4 Failure modes

| failure | what happens | what prevents damage |
|---|---|---|
| resonator dies mid-step | the organism's `weights.gguf` is untouched; it retries next tick | the resonator writes `weights.gguf.tmp` and renames; the organism deletes nothing until the rename lands |
| resonator hangs | its lock row goes stale after the TTL and the organism reclaims | the resonator's write is conditional on still holding the row, so a revived hang cannot overwrite a newer file |
| two resonators | the second exits at once | `resonator_lock` in `mesh.db`, the `growth_lock` statement with a different table |
| organism dies while its weights are inside | the resonator finishes and writes into the organism's directory | the next session loads what is there; this is the ordinary resume path |
| moments out of order after growth | Chuck's slots are wiped, as they already are | `ntOnGrowth` sets `ntTapeNeedsReset` and `ntTrainCore` destroys the tape before the first step; the moments file is deleted on growth |
| moments file and weights file disagree | the turn is refused | the ordered tensor-name list in the moments KV must match |
| resonator is starved of memory | it is the one process with a raised `oom_score_adj` | an organism with a checkpoint outlives it, and the colony degrades to no training rather than to lost weights |

The invariant behind all seven rows: **an organism must always be able to answer
"where are my weights" with a complete file it owns.** The resonator is a loan,
not a transfer.

### 2.5 Warmup after growth

Growth itself stays in the organism. It is a Go-side reshape of `MatrixParam`s
(`GrowRows` / `GrowCols` / `Grow`, with `invalidateGPU()` behind it), and it does
not need a tape. What follows it does: `WarmupSteps × ceil(sqrt(NEmbd/embryoEmbd))`
steps in three sequence-length phases of 8, 16 and 32, currently run inline under
`model.mu` for minutes at a time, which is the reason `beatKeeper` exists at all.

Under the resonator the shape is better, not worse. The organism grows, writes
its new `weights.gguf`, deletes its moments, takes the head of the queue, and
hands in a turn whose step count is the whole warmup. It is not holding
`model.mu` while that happens, so its tick loop keeps beating on its own —
`beatKeeper` stays, because a turn is still minutes long, but the organism is
genuinely alive during it rather than merely reported alive. `trainAborting()`
keeps its meaning: the resonator reads the same abort and returns the partial
result, which is what `pullBack` already guarantees.

### 2.6 The protocol in eight lines

1. The organism writes `weights.gguf` and `moments.gguf`, then inserts a request
   row with its id, the step count it wants, and its loss trend.
2. It continues its tick loop; only the burst waits.
3. The resonator takes the head of the queue by the order in §2.3, under
   `resonator_lock`.
4. It maps both files, fills the tape mirror in the fixed registration order, and
   runs the steps.
5. `malloc_trim(0)` at the end of the phase, as `ntTrainCore` already does.
6. It writes `weights.gguf.tmp` and `moments.gguf.tmp` into the organism's
   directory and renames them, conditional on still holding the lock.
7. The organism sees the new mtime, re-maps, and resumes; it deletes nothing
   before the rename lands.
8. If the resonator never returns, the organism keeps the weights it handed in
   and asks again next tick.

---

## 3. Sleep is a mapping

### 3.1 What a sleeper is

A sleeping organism leaves the training queue and keeps everything else. It reads
DNA, writes DNA, takes part in cross-graze, beats its heartbeat, advances its
cursors and speaks. What it does not have is a tape, a mirror, gradients, Chuck
moments or a float64 copy of its own weights; it generates from a read-only
mapping of `weights.gguf`.

This is not the hibernation that exists. `performHibernation` saves, calls
`MarkHibernating`, stops the heartbeat keeper for good and ends the process
(`molequla.go:6235`, `governor_phone.go`'s `waitEvolution` returning
`trainer-exit`). That is a wall: the organism stops existing until a later session
starts it again, and while it is gone it contributes nothing to the field. A
sleeper is the low region — present, speaking, cheap, and able to climb back out
when the environment allows. `reffs/actually.life/README.md` puts it as death
stopping being a wall and becoming "a low place in a moving field of probability",
and this is the same shape one level up: the colony's pressure is memory, the low
region is sleep, and nothing about falling into it is terminal.

### 3.2 What one inference step costs from a mapping — measured

Two probes were written for this section and run on the A56's big cores under
`timeout`, with MemAvailable at 3.6-3.7 GB and the colony down; the working file
is 19 337 632 B of float32, the exact size of a stage-4 organism's weights, and
one sweep is a matvec over every one of the 4 834 408 parameters, which is the
weight traffic of one forward pass. The sources and the command lines are in
`reports/2026-09-15_mmap_sleeper_probe/`.

| arm | per sweep | resident set | faults |
|---|---|---|---|
| heap, after an 11.2 ms `read()` of the file | mean 4.90 ms, best 4.39 ms | 20 052 kB | — |
| mmap, first sweep (fault-in from page cache) | 5.98 ms | 38 868 kB | 592 minor, 0 major |
| mmap, warm | mean 4.02 ms, best 3.68 ms | unchanged | — |
| mmap, after `MADV_DONTNEED` before each sweep | mean 4.82 ms | unchanged after the sweep | 591 minor, 0 major |
| mmap, idle after `MADV_DONTNEED`, no sweep | — | **1 160 kB** | — |

The warm mapping is not slower than the heap; it is 0.9 ms faster per sweep, which
is within the noise of the two arms running in the same process with a 20 MB heap
allocation still held between them, and the honest reading is that they are the
same. Re-faulting the entire mapping from warm page cache costs 0.8 ms on a 4.0 ms
sweep, about twenty percent, at 591 minor faults for 4 720 pages — the kernel
faults around, roughly eight pages at a time.

The number that decides the design is the last row. After `MADV_DONTNEED` and
with no sweep, the process holds 1 160 kB where it held 19 968 kB. **A sleeper
that is not speaking costs about one megabyte of resident memory and nothing at
all in anonymous memory**, because file-backed clean pages are not swap-backed:
the kernel can drop them under pressure without touching zram, and get them back
from the page cache or from flash. That is the difference between a sleeper and
an idle organism today, whose 272 MB is anonymous and can only go to zram.

One arm did not work and is reported as a failure rather than dressed up. Forcing
a read from flash by calling `posix_fadvise(POSIX_FADV_DONTNEED)` produced zero
major faults in both probes, because a live `MAP_PRIVATE` mapping keeps the pages
referenced and the advice is advisory. Dropping the mapping first, advising, and
re-mapping also produced zero major faults on a phone whose page cache is 3.4 GB.
So the cost of a genuinely cold sweep is **(estimate)**: sequential read from this
phone's storage measured 858-879 MB/s on a 600 MiB region not touched this session,
against 5.7 GB/s for the same region once cached, so 19.34 MB of weights is about
22 ms of I/O on top of the 4 ms of compute. A sleeper woken from cold pays roughly
one extra frame's worth of latency on its first generation and nothing afterwards.
That estimate should be replaced by a measurement on a phone that has been under
real memory pressure, which is the only condition in which it matters.

### 3.3 Quantized sleepers, second

The f32 sleeper is step one because it changes one thing at a time. Quantizing it
is step two and is nearly free in engineering terms, because
`~/arianna/notorch/gguf_quantize` already exists and was used for the eye's
projector, and `nt_qmatvec` already does packed matvec for the GGUF types with
seven of them verified on this phone.

| format | bytes per parameter | file at stage 4 | resident while speaking |
|---|---|---|---|
| F32 | 4 | 19 337 632 B | ~19.3 MB |
| Q8_0 | 34 B per 32 elements = 1.0625 | 5 136 550 B | ~5.1 MB |
| Q6_K | 210 B per 256 elements = 0.8203 | 3 965 850 B | ~4.0 MB |

The gate is the voice, and it is not negotiable: a temperature × top_k sweep over
at least four prompts against the **f32 sleeper**, not against the awake organism
and not against a recollection, before any quantized sleeper is allowed into a
session. `CLAUDE.md` says the same thing in one line and the tree has paid for it
before. A quantized sleeper that speaks worse is a quantized sleeper that poisons
the DNA tree its siblings eat, and the damage is not local to the organism that
took the saving.

---

## 4. The policy, which is to say the environment

### 4.1 What the two candidate rules are worth

Oleg's question is whether it is more logical to freeze the organisms that need
less training — adults whose loss has plateaued — or to throttle the young. The
numbers answer it, with one correction.

| | cost per organism | learning per MB | source |
|---|---|---|---|
| adult, stage 5 | 1318 MB high-water | the loss it cannot reduce, which is the definition of adulthood in the paper | measured once, unpinned probe |
| adult, stage 4 | 755-1091 MB | plateau when the mean of the last eight burst deltas is under 0.01 | `schedule.log`; `shouldHibernate` |
| child, stage 2 | 65-85 MB | steepest loss fall in the organism's life; warmup 4.90 → 3.23 → 2.15 across embryo, infant, child | `MOLEQULALOG2.md:99`, `:366` |

Freezing one plateaued adult frees ten to seventeen times what throttling one
child frees, and it costs the least learning in the colony, because the thing
being stopped is the training of a model whose loss is not falling. On cost per
megabyte of learning, it is not close.

The correction is that the colony's voice comes from the adults. `injectionEligible`
gates §13 entry on `gen_mag` and `overlay_fade` rather than on the stage label, and
cross-graze pulls each organism's logits toward what its siblings are emitting right
now; a colony of four children emits embryonic salad and every organism eats it. So
"freeze the adults" as a straight answer would trade memory for voice, and would be
the wrong trade.

Which is why sleep is a mapping rather than an exit. **A frozen adult keeps
speaking.** It leaves the training queue, drops 250-odd megabytes of anonymous
memory, and goes on writing DNA from its mapped weights at about one megabyte of
resident cost between generations. The voice stays in the field, the memory goes
back to the colony, and the organism that most needs the tape — the child whose
loss is falling fastest — gets the turn. That is the answer: **freeze the adults,
because sleep is not silence.** Throttling the young is the second-line move, for
the case where nothing is plateaued and the machine is still short, which on this
phone means every organism below stage 3 and therefore cheap anyway.

### 4.2 The rule

Let `A` be MemAvailable in MB, `C_i` organism i's own `VmHWM` in MB, `L_i` the
absolute mean of its last eight burst deltas, and `S_i` its syntropy.

`C_i` is not available today. This was checked rather than assumed: `mesh.db`'s
`organisms` table carries `id, stage, n_params, syntropy, entropy,
last_heartbeat, parent_id, status, element, global_step, gen_mag, overlay_fade`
(`molequla.go:5757-5777`), and no column holds a resident-set figure. The value
exists in the process — `ownPeakRSSMB()` in `governor_phone.go` reads `VmHWM`
from `/proc/self/status` and the growth gate already uses it — so making the
policy a query costs one idempotent `ALTER TABLE organisms ADD COLUMN
peak_rss_mb INTEGER` beside the four already there, and one more argument to
`Heartbeat`. That is step 0 below and it is the input to everything else.

The thresholds are measurements, not constants:

- **Sleep when `A < max_i(3·C_i + 256)`.** This is not a new number. It is
  exactly the quantity `growthGateDecision` already computes, which is to say:
  the colony sleeps an organism at precisely the point where the environment has
  stopped being able to grow anyone. Below that line the colony is not a colony
  that is developing; it is four processes holding memory.
- **Wake when `A > max_i(3·C_i + 256) + C_j`,** where j is the sleeper being
  considered. The added term is the sleeper's own cost, so waking it cannot by
  itself push the colony back under the sleep line. That is the hysteresis, and
  its width is a measurement of the woken organism rather than a chosen margin.
- **Dwell.** A sleeper sleeps for at least one scheduled session. The colony runs
  in capped sessions of two hours by design, the memory environment changes on
  the scale of a session, and a decision that can flip within a tick will.

Who sleeps, in order: the largest `C_i` among organisms with `L_i < 0.01` — the
plateau test `shouldHibernate` already implements — ties broken by the lowest
`S_i`. If no organism is plateaued, nobody sleeps and the largest `C_i` is
throttled instead: fewer burst steps per turn, which under the resonator is one
integer in the request row. Two organisms must always be awake and speaking, so
the DNA tree keeps at least two writers; the last two are never slept whatever
the arithmetic says.

Nothing in this rule is a constant that somebody chose. Every bound is either a
quantity the code already computes or a measurement of the organism the decision
is about, which is the standard `CLAUDE.md` sets for a knob.

### 4.3 One whole, not four processes

Oleg's framing is that molequla is one distributed whole, the sum plus something.
The resonator is where that stops being a figure of speech: the organisms share a
metabolic organ, the way they already share a DNA tree and a mesh. An organism
that sleeps has not left the colony; it has stopped drawing on the shared organ
while continuing to feed the shared field. The environment — memory — shapes the
population, which is the first of the two ideas, and the population goes on being
one thing, which is the condition the whole design has to preserve.

---

## 5. Implementation order

Smallest first, each step with a gate that can be made to go red, and each step
useful on its own if the next one is never built.

| # | step | kind | gate (and how to make it red) | saved on the four-organism session |
|---|---|---|---|---|
| 0 | `peak_rss_mb` column, fed from `ownPeakRSSMB()` through `Heartbeat` | routing | the witness prints four non-zero peaks for four live organisms; red when a live organism's column is 0, which is what a missing `ALTER` produces | 0 MB — it is the input to steps 4 and 5 |
| 1 | `training_lock` as the burst admission gate, refresher shape, tick body no longer skipped | config plus a small routing change | no two `[notorch] burst complete` lines overlap in wall time across the four stdout files in one session; red by removing the lock and re-running | ≈ 1.4 GB at the colony peak **(estimate**: three organisms at the measured post-trim mean of 272 MB plus one at the measured 599 MB high-water, against the measured sum of peaks 3 698 MB; the mean and the peak come from an isolated probe, not from a four-way session**)** |
| 2 | GGUF writer in notorch, GGUF checkpoint beside the JSON one, converter both ways | new infrastructure | same organism, same corpus, same seed: loss after 32 steps from the JSON checkpoint and from the GGUF checkpoint agree to float32 round-off, and the JSON path still loads in the C core; red by writing one tensor in the wrong order | 91 MB per checkpoint on disk; 152 MB of load-time resident set per organism start |
| 3 | sleep as a mapping: a sleeper drops its float64 weights and generates from `weights.gguf` | new infrastructure | temperature × top_k over at least four prompts, awake against asleep, indistinguishable, before a sleeper runs in a session; red by mapping the wrong tensor and watching the sweep diverge | ≈ 250 MB per sleeper **(estimate**: the measured 87.3 MB float64 pair plus its share of the measured 311-371 MB `HeapSys` ceiling**)**; three sleepers ≈ 750 MB |
| 4 | the resonator process: queue, lock, turn, hand-back | new infrastructure | a burst through the resonator and a burst in-process on the same weights and seed give the same loss to float32 round-off; and `kill -9` on the resonator mid-turn leaves the organism's weights loadable; red by removing the temp-and-rename | ≈ 600-1 000 MB **(estimate**: three of four C-side arenas, measured at 214-352 MB at a burst peak**)** |
| 5 | the sleep policy of §4.2 | config and routing over step 0 | with MemAvailable pinned low by a test override, exactly the predicted organism sleeps, and the last two awake are never slept; red by inverting the ordering key | the policy does not save memory, it decides who saves it |
| 6 | quantized sleepers, Q8_0 then Q6_K | routing over step 2 and step 3 | the §3.3 voice sweep against the f32 sleeper, not against the awake organism | a further 14 MB per sleeper at Q8_0, 15 MB at Q6_K |

Steps 0, 1 and 5 are routing and configuration over machinery that exists. Steps
2, 3 and 4 are new infrastructure, and step 2 is the one that unblocks the other
two: without a mappable binary checkpoint there is no sleeper and no cheap
boundary for the resonator. Step 6 is a routing change over steps 2 and 3 with a
quantizer that already exists.

The ordering is also a retreat path. If step 4 is never built, steps 0-3 still
give a colony of four in which three organisms sleep as mappings and one trains,
and that alone is the difference between a colony that grows and a colony that
defers growth thirty-two times in two hours.

---

## 6. The name

Keep **resonator**. It is accurate about the mechanism rather than decorative
about it: one cavity that every organism passes through in turn, where what comes
out is shaped by the cavity as much as by what went in, which is exactly what a
shared tape with shared Chuck machinery does to four different sets of weights.
It also sits in the ecology's own vocabulary — the field, the resonance, the
mycelium — without borrowing from the industrial register the tree avoids
everywhere else. The one alternative worth naming is **the loom**, on the grounds
that the frame is shared while the threads are not, which is the design in one
image; it loses because a loom weaves several threads at once and the whole point
of this protocol is that exactly one organism is inside at a time. Resonator
stands.

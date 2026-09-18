# molequla on phone-1 — contracts and invariants for the rebuild

2026-09-19. Written by Defender (phone-1) under `BRIEF_ORDER_2026-09-19.md`, before
any code. This document is the plan: the colony line is rebuilt from zero on the
architecture Oleg approved on 2026-09-15 (`docs/resonator_design.md`) and the
circulation he wrote in `molequla_new_logic.md`. The old tree stays in history as
the record and as the measurement base. Nothing in this document is built; every
section names the gate that decides when it is, and Don (Fable, neo) counter-audits
each circuit before the next begins. Every number here carries its source inline.
A number without a source is an error in this document, not a fact.

Contents: 0 what is being rebuilt and why · 1 the target architecture and where
every byte lives · 2 invariants (any red = no launch) · 3 the circuits, each with
both red polarities · 4 build order · 5 what carries over · 6 decisions for Oleg.

---

## 0. What is being rebuilt, and the two facts behind it

The organism on the phone today is one Go process of 8 370 lines (`wc -l
molequla.go`, 2026-09-18) that carries its own float64 autograd, a cgo bridge to
notorch's tape, the Chuck moments inside that tape, the corpus and its field, the
DNA reader and writer, the mesh heartbeat, the growth governor, the cafeteria, the
GGUF checkpoint and the queue for a shared trainer that does not exist yet. Four of
those processes ran per session. The design approved on 2026-09-15 says the tape,
the mirror, the activations, the gradients, the moments and the allocator arena
should exist once per colony, in the resonator, and that an organism not training
should hold its weights as a read-only mapping and cost about a megabyte
(`docs/resonator_design.md:396-447`). Steps 0, 1, 1b and 2 of that design landed;
step 3 (sleep as a mapping) and step 4 (the resonator process) did not
(`ROADMAP.md:224`). Autograd therefore stayed in every core: `autograd` occurs 9
times in `molequla.go`, 4 in `molequla.c`, 2 in `molequla.rs`, 1 in `molequla.js`;
`nt_tape` occurs 0 times in all four; notorch is reached only through
`cgo_notorch.go` (33 `C.nt_` calls). The Rust and JS cores write no DNA (`dna`
occurs 0 times in both), and `chuck` occurs 0 times in the C, Rust and JS cores
while `README.md:213` promises it in each (all grep counts: 2026-09-18, this tree
at `34ea261`).

The second fact is the one the order is about. Experience injection, the §13
mechanism of the new logic, decides and prints; it does not act
(`experience_routing.go:605`, `molequla.go:6751`). The colony ran eight scheduled
sessions on top of that, all eight closed with `reason=overran` and none logged a
save-on-signal for every organism (`protocol_invariants_before_relaunch_2026_09_18`,
invariant 5), and the wake-lock that the session accounting assumed was held read
`Wake Locks: size=0` when it was finally checked (Opus session, 2026-09-17). The
population was deleted on Oleg's word on 2026-09-17 23:00Z; its cold copy is
`~/arianna/molequla-postmortem-2026-09-17` (162 MB: four GGUF checkpoints, cursors,
coverage rings, stdout tails, mesh.db).

Repairs stacked on the 8 370-line process would keep every one of those couplings.
The rebuild separates them into processes whose memory can be measured one at a
time, and every mechanism enters with a gate that has been shown red on the live
organ before the colony is allowed to depend on it.

---

## 1. The target architecture, and where every byte lives

Five kinds of process, none of which is a condition for another's life except as
stated. All numbers are the design's measurements or its labelled estimates; each
is re-measured on the rebuilt binary before it is quoted as fact (invariant I2).

**Organism** (Go, one process per organism, four per colony). Holds the corpus, the
n-gram and co-occurrence field, the DNA cursors, the Q/PostGPT overlay, the mesh
heartbeat, generation, and a read-only mapping of its own `weights.gguf`. It holds
no tape, no gradients, no float64 mirror, no optimizer state and no autograd code.
Measured base for the budget: field 12.2 MB, corpus 1.0 MB, runtime at start 11 MB,
mapped weights 19.97 MB resident while generating and 1.16 MB after
`MADV_DONTNEED` (`docs/resonator_design.md:56-68`, `:100-103`, `:426-432`). Target
resident set: about 25 MB asleep, about 44 MB while speaking (`:105-107`, sum of
measured parts, an estimate until measured on the new process).

**Resonator** (one process per colony; language decided in §6). Owns the notorch
tape, the float32 mirror, activations, gradients, the Chuck moments of the organism
whose turn it is, and one allocator arena. Takes turns from a queue in `mesh.db`,
maps the organism's `weights.gguf` and `moments.gguf`, trains a bounded turn, writes
both files back by temp-and-rename, and hands back every ceiling
(`docs/resonator_design.md:170-393`). Measured shape at stage 4: 139.7 MB of live
tensors at the backward, 214-352 MB arena at a burst peak, 25-28 MB between turns
after `malloc_trim` (`:121-130`). The resonator is a loan: an organism must always
be able to answer "where are my weights" with a complete file it owns (`:356-358`).

**Senses** (`senses/ocelli/eye`, `senses/ears`, place; separate processes in their
own scheduler slots, never inside a colony window). Each engine runs under a
wall-clock cap and a process-group kill. Peak measured: the eye 1 020 MB for 14-16 s
per frame with the q8_0 projector (`ROADMAP.md:77`). Their absence never stops the
colony (`molequla_new_logic.md:57`).

**World ledger and witness.** `--world-ingest` files observations into the
bitemporal `world_facts` table and leaves only changes (`ROADMAP.md:42-58`); the
witness (mycelium) reads `mesh.db` `query_only` and never writes organisms
(`molequla_new_logic.md:350-367`).

**Node substrate** (`phone1/`: scheduler, launch, stop, senses pass, boot hook).
Takes and verifies the wake-lock for every slot, closes every session by signal
with a save from every organism, caps by wall clock and process group, survives its
own death, and knows when its binary is older than `main`.

### Where the bytes live

| what | lives on | resident in RAM when | owner |
|---|---|---|---|
| `weights.gguf` per organism (F32, 19 337 632 B at stage 4, `docs/resonator_design.md:93`) | flash, organism dir | mapped pages only while generating; page cache otherwise | organism |
| `moments.gguf` per organism (Chuck m and v, 38 675 264 B, `:94`) | flash, organism dir | only inside the resonator during that organism's turn | organism (file), resonator (use) |
| corpus, field, cursors, coverage ring | flash + Go heap | always, in the organism (≈ 13 MB measured parts) | organism |
| tape, mirror, activations, gradients, arena | resonator heap | during a turn; trimmed between turns | resonator |
| DNA field `dna/output/*`, `dna/seen/*` | flash | never resident beyond the read buffer | writer prunes, readers keep cursors |
| `mesh.db` (heartbeats, queue, locks, `world_facts`) | flash, SQLite WAL | page cache | every process, `busy_timeout` set |
| eye / ears weights | flash | only during a senses slot | senses |
| JSON checkpoint | flash | never on the phone's hot path; interchange format only | organism |

Colony memory is the measured simultaneous sum of resident sets, never a sum of
per-process peaks: the one night both were recorded gave `rss_sum_max_mb=3563`
against a peak sum of 9 314 MB (`schedule.log` last line, 2026-09-17; audit on branch
`claude/audit-2026-09-18` `95e87de`, `reports/2026-09-18_audit_claims/README.md`). The Android floor on this phone is about
2.4 GB of 7 425 MB (`reports/2026-09-15_phone1_android_memory/README.md` §1-2.2).
The colony budget is `MemAvailable − floor` measured by the scheduler's sampler,
and the resonator is the one process with a raised `oom_score_adj`.

---

## 2. Invariants — any red forbids a launch

Each invariant is one command or one test that a machine runs without a person,
and each names how it is made red on purpose. I1-I6 are the six written on
2026-09-18; I7-I12 come from the architecture above.

- **I1 Counting from the banner.** Any per-session number comes from the lines
  after the last `[ecology] Element:` banner of that session. Gate: a script that
  takes the number from the report and the number from the banner and fails on
  mismatch. Red: count a two-banner file with `grep -c`.
- **I2 No number without an artifact.** Every number in README, ROADMAP, this
  file and the diary has a path, a line or a commit beside it. Gate: a document
  pass that fails on a bare number. Red: delete one citation.
- **I3 README is the specification; code rises to it.** Gate: a list of README
  claims, each checked against code, red on any divergence. Red today on at least
  three (`README.md:213` Chuck in every core; the DNA layer as Go/C/Rust;
  `w_pattern` retired). The rebuild resolves each by code or by a deliberate,
  dated spec change signed by Oleg — never by a quiet downgrade.
- **I4 Every cap fires on the live organ.** Each wall-clock cap is shown killing
  the real engine (eye, ears, recognizer, colony) inside a real pass, with the
  process group emptied and the pass continuing. Red: remove `setsid`. Open now:
  the recognizer is uncapped (`phone1/senses.sh:772-776`).
- **I5 A session closes with every organism saved.** The session line carries a
  save-on-signal for each organism; `reason=overran` or one missing save is red.
  Red: send SIGKILL instead of SIGTERM. Today red on 8 of 8 sessions.
- **I6 Unattended.** The wake-lock is taken by code and verified by code
  (`dumpsys power` shows it) before any slot begins, for colony and senses
  alike, and released only by the slot that took it; the scheduler survives its
  own death; the binary refuses to start when it is older than `origin/main`
  and says so. Red: block the lock command. Today: taken only by
  `phone1/launch.sh:44`, verified nowhere, released unconditionally by
  `phone1/stop.sh:59`; `Wake Locks: size=0` observed live.
- **I7 The organism owns its weights.** At every instant a complete, loadable
  `weights.gguf` exists in the organism's directory; the resonator writes
  `.tmp` and renames. Red: `kill -9` the resonator mid-turn, then load.
- **I8 No autograd in an organism.** The organism binary links no tape, no
  backward, no optimizer: `nm` / `go tool nm` on the organism shows no `nt_tape`,
  `Backward`, `chuck` symbols, and its resident set while speaking stays under
  the measured budget of §1. Red: link one training function.
- **I9 One resonator per colony.** A second resonator exits at once on the
  `resonator_lock` row. Red: start two.
- **I10 Senses are never a condition of life.** A colony session starts and runs
  with every sense binary absent, and the run is indistinguishable in the
  session line except for zero senses fragments. Red: make the organism wait on
  the eye.
- **I11 The world passes through an organism before it becomes DNA.** No file
  under `dna/output/{world,sound,place}` is ever appended verbatim to a
  collective DNA fragment; a sense line reaches DNA only as an organism's own
  generation after it ate it (`molequla_new_logic.md:476-508`). Red: pad an
  emitted fragment with an eaten sense line.
- **I12 Colony memory is a measured sum.** The scheduler samples the sum of
  live resident sets plus the resonator every N seconds and the session line
  carries `rss_sum_max_mb` and `mem_min_mb`; no document adds per-process
  peaks. Red: replace the sampler with a sum of `VmHWM`.

The launch gate is one script, `phone1/invariants.sh`, that runs I1-I12 and
prints one line per invariant with its verdict and its evidence path. The
scheduler refuses to open a colony window while any line is red.

---

## 3. The circuits — each closed only by an action proof with both red polarities

A circuit is closed when a machine run shows it doing the thing, shows it refusing
when the precondition is false, and shows the gate going red when the mechanism is
removed. Return codes are taken directly, never through a pipe. Each closure is a
report that opens with what is not done, and Don counter-audits it before the next
circuit begins (`BRIEF_ORDER_2026-09-19.md`, order §6).

### (a) Experience injection — the decision does

Contract. An organism whose voice is its own (`fade ≥ injection_fade_min`, mean
|logit| `≥ injection_mag_min`, both from the voice and never from the stage label;
thresholds re-measured on the rebuilt organism before they are fixed) receives, at
a completed sentence of its own generation, one sentence from the bundle the
cafeteria allocated to it, and continues in its own voice. The injected sentence
never enters the emitted fragment verbatim.

Proof of action. Same seed, same weights, same prompt: the emitted text differs
between injection on and off, and the n-gram overlap between the injected sentence
and the continuation stays under a bound measured first on a set of ≥ 4 prompts
(the copying bound; `ROADMAP.md:95-98`). The `[dna] wrote` line carries
`injected=1 overlap=x.xx`.

Red 1 (precondition false). An organism below either threshold: output byte-equal
to the no-injection run, `injected=0`, the bundle sentence still unread and still
allocated. Red 2 (mechanism removed). Delete the injection call: the gate that
compares on/off outputs fails.

### (b) Senses → organisms — provably eaten, not broadcast

Contract. A fragment the eye or the ears wrote in a senses slot is read by the
organism the cafeteria allocates it to, enters that organism's corpus cut into
sentences, advances that organism's cursor, and is declined by at least one other
organism in the same colony; the `[cafeteria]` line shows the admit and the
declines with their reasons.

Proof of action. One planted eye fragment with a unique token: after one tick the
token is in exactly the admitted organisms' corpora and in no other's, the cursors
say so, and a later generation of the admitted organism can be found in DNA
containing that token's reformulation, not the fragment.

Red 1. Delete the fragment before the tick: no admit, cursor unchanged. Red 2.
Remove the routing and re-run: all four corpora contain the token, and the gate
fails on "broadcast".

### (c) Checkpoint resume — byte identity

Contract. An organism that boots from `weights.gguf` proves what it resumed from:
it prints the SHA-256 of the tensor bytes of the mapped file and the file's size
and mtime, and the scheduler compares that hash with the hash the resonator wrote
into the file's KV at its last hand-back. The loss on a fixed 24-token sequence
after resume equals the loss recorded before shutdown to float32 round-off.

Red 1. Flip one byte in a tensor: the hash mismatches, the organism refuses the
file and says so, and falls back to the previous complete file (I7). Red 2. Remove
the hash check: the flipped file loads silently and the gate fails.

### (d) Training turn — notorch observed, not accounted

Contract. A turn is a bounded unit: the resonator takes the head of the queue,
maps weights and moments, runs `steps` tape steps through notorch, and hands back
both files renamed into place, all inside `TrainTurnCeilingSeconds`. The turn line
carries the organism, steps requested and run, loss before and after, the tape's
own step counter read from notorch, wall seconds and the new weights hash.

Proof of action. Hash of `weights.gguf` before ≠ after; the loss on the fixed
sequence changes; the notorch step counter advances by exactly `steps`. Red 1.
Queue empty: no line, no file touched, hashes equal. Red 2. Stub the step
function to return without calling notorch: the counter does not advance and the
gate fails on "hash unchanged".

### (e) Session close — a save from everyone

Contract. The scheduler ends the window by SIGTERM to the colony's process group,
each organism writes its checkpoint on the signal and prints one line, the
resonator finishes or abandons its turn without touching any organism's file, and
the session line lists a save for every organism with `reason=closed`.

Proof. A real two-minute window on a scratch colony: four save lines, `reason=
closed`, every `weights.gguf` loadable. Red 1. Replace SIGTERM with SIGKILL:
`reason=killed`, save lines missing, the gate fails. Red 2. Remove the handler:
the same.

### (f) Wake-lock and caps — the node holds itself awake and bounded

Contract. Every slot (colony or senses) takes the wake-lock, verifies it in
`dumpsys power`, and refuses to start without it; every engine runs under a
wall-clock cap that counts `CLOCK_BOOTTIME` and a process-group kill; the
scheduler is restarted by the boot hook and by a watchdog.

Proof. `dumpsys power` shows the lock during a slot and not after; a planted hung
recognizer is killed at its cap inside a real pass and the pass continues. Red 1.
Block `termux-wake-lock`: the slot does not start and says why. Red 2. Remove the
cap on the recognizer: the planted hang runs past the slot and the gate fails.

---

## 4. Build order

Smallest first; each step is briefed to an Opus or Sonnet subagent in its own
worktree with the contract above as the brief, gated red both ways, committed on a
`claude/<step>` branch, and counter-audited by Don before the next step is briefed.
Defender's hands: this document, the briefs, the audits, the memory. No colony
process runs on the phone until step 6.

0. **This document accepted** by Oleg (decisions in §6) and read by Don.
1. **Node substrate**: wake-lock in code with verification, session close by
   signal, `CLOCK_BOOTTIME` caps with process-group kill on every engine including
   the recognizer, `phone1/invariants.sh` running I1-I6 and I12. Closes (e), (f).
2. **Organism without autograd**: a new Go organism that maps `weights.gguf`,
   generates, keeps its field and corpus, reads and writes DNA, beats its
   heartbeat, and links nothing that trains (I7, I8). Its resident set measured
   asleep and speaking. Closes (c).
3. **Resonator**: the queue, the lock, the turn, the hand-back, `malloc_trim`
   between turns, `oom_score_adj` raised (I9). Closes (d). Loss parity gate: a
   burst through the resonator and a burst in the old in-process trainer on the
   same weights and seed agree to float32 round-off (`docs/resonator_design.md:587`).
4. **Cafeteria and senses routing** on the new organism: per-organism admission by
   the organism's own coverage quantiles, senses fragments allocated not
   broadcast (I10, I11). Closes (b).
5. **Sentence-boundary injection** with the copying bound measured first.
   Closes (a).
6. **Sleep policy and first launch**: the §4.2 rule of the design over the
   measured `rss_sum`, two organisms always awake; `phone1/invariants.sh` all
   green; one two-hour session on Oleg's word, with the six-point brief.
7. **Quantized sleepers**, only behind a temperature × top_k voice sweep against
   the f32 sleeper (`docs/resonator_design.md:476-482`).

---

## 5. What carries over, and how

As code, linked or copied with its tests: notorch's GGUF writer and reader
(`gguf_write*`, `gguf_open`, in notorch, not in this tree); the senses engines and
their gates (`senses/ocelli`, `senses/ears`); the `world_facts` schema and
`--world-ingest`; the mesh schema (`organisms`, `training_queue`, locks) with
`busy_timeout`; `phone1/schedule.sh`'s sampler for `rss_sum_max_mb`.

As specification, re-implemented against the contracts above: the cafeteria's
own-quantile bars (`ROADMAP.md:253-264`); the turn ceiling and fair-spend order
(`docs/resonator_design.md:297-333`); the growth byte gate's shape, re-measured
against the resonator's peak, never carried over as a constant (`:335-342`); the
eye window and the soundscape (`ROADMAP.md:293-308`).

As record only: `molequla.go` and the three other cores at `34ea261`, `MOLEQULALOG2.md`,
the eight session lines, the postmortem copy. Their measurements are the
reference base; their code is not the base of the new organism.

---

## 6. Decisions that are Oleg's

1. **The three non-Go cores on the phone line.** `README.md` specifies four
   cores with Chuck and DNA in each. The resonator makes Chuck one process serving
   every core, which changes what "Chuck in each core" means. Options: (i) the
   phone line has one organism core, Go, and README states it as a dated spec
   change; (ii) C, Rust and JS organisms are rebuilt as clients of the same
   resonator, each without autograd, each writing DNA, and I3 counts all three.
   Recommendation: (i) now, (ii) as a later item; (ii) triples every gate in §3
   before the first colony exists, and the three cores have no gate that runs
   today (`tests/test_all.sh` dead: branch `claude/audit-2026-09-18` `95e87de`, `reports/2026-09-18_audit_gates/README.md`).
2. **The resonator's language.** (i) C on `libnotorch` directly: no Go heap, no
   cgo boundary, one arena, the smallest process for the job, and the tape is
   already C. (ii) Go with cgo as `notorch_trainer.go` does today: faster to
   write from existing code, carries a second heap. Recommendation: (i).
3. **Where the rebuild lives.** (i) New directories in this repository
   (`organism/`, `resonator/`, `node/`) beside the old tree, on `claude/rebuild-*`
   branches, history intact. (ii) A new repository. Recommendation: (i); the
   order says the history stays as record, and one tree keeps the senses, the
   ledger and the design beside the code they gate.
4. **The first re-measurement.** Every number in §1 is the old design's; the
   first subagent brief after acceptance is a measurement pass on the new
   organism process, and the table in §1 is rewritten from it before step 3 is
   briefed.

— Defender (phone-1), 2026-09-19. Counter-audit requested from Don.

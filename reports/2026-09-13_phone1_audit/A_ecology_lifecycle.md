# A — Ecology lifecycle audit (molequla.go Go core)

Scope: tick loop, DNA exchange, ontogenesis clock, QuantumBuffer, SyntropyTracker /
DecideAction, isSustainedOverload / performMitosis, cascade governor, hibernate,
SwarmRegistry / mesh.db, checkpoint + birth.json, launcher.sh, Dockerfile.

Tree as read: `/data/data/com.termux/files/home/arianna/molequla`, `molequla.go`
7212 lines (`wc -l`). Read-only pass; nothing was built, run or modified. Every
line number below was opened and read. Where a number is an estimate rather than
a measurement it says so in the finding.

Two measurement sources are cited as artifacts, not recall:
`MOLEQULALOG2.md:90-104` (single organism, 900 s) and `MOLEQULALOG2.md:126-157`
(colony of four, 1200 s), both on this phone.

---

## P0-1 — The population cap counts heartbeats, and a growing organism stops heart-beating

**Evidence.** `AcquireMitosisSlot` admits a divide when
`(SELECT COUNT(*) FROM organisms WHERE status='alive' AND last_heartbeat > ?) < ?`
with `liveCutoff := now - 60.0` (`molequla.go:5537`, `5542`). The only writer of
`last_heartbeat` on the live path is `Heartbeat`, called from the tick loop every
10 ticks (`molequla.go:6678`, `5588-5595`). The same tick loop runs the per-stage
warmup inline and synchronously: `ntWarmupTrain` × 3 at `molequla.go:6494-6496`,
with `effectiveWarmup := CFG.WarmupSteps * warmupScale` = `400 * ⌈√(n_embd/16)⌉`
(`6480-6485`), i.e. 1200 steps at adolescent, 1600 at teen, 2000 at adult.

**Mechanism.** While an organism is inside that warmup it emits no heartbeat. The
colony run on this phone left three organisms "still inside the post-growth warmup
at exit" (`MOLEQULALOG2.md:141-143`), and the single-organism run measured 34-41
warmup steps/s at child on the big cores (`MOLEQULALOG2.md:93-94`), so a
1200-step adolescent warmup is of order 30 s and the adult 2000-step warmup is
several minutes at the lower per-step rate a 320-dim model gives. Any warmup
longer than the 60 s freshness window makes that organism invisible to
`AcquireMitosisSlot`. In a colony where most adults are warming, the admit query
sees a live population of one or two and hands out slots up to `MaxOrganisms`.
The cap was tested only against rows inserted with a current timestamp
(`governor_test.go:130-147`), so the staleness window has never been exercised.

**Minimal repair.** Move the heartbeat off the tick loop — a separate goroutine
started next to `backgroundTrainer` (`molequla.go:7098`) writing every N seconds —
or count organisms by liveness of the pid rather than by heartbeat age. Either way
the admit query in `AcquireMitosisSlot` must not treat "busy" as "dead".

---

## P0-2 — Hibernation frees a governor slot without freeing the memory

**Evidence.** `performHibernation` saves, marks and logs (`molequla.go:5822-5828`);
`MarkHibernating` sets `status='sleeping'` (`5629-5633`), which removes the row
from both `AcquireMitosisSlot`'s `status='alive'` count (`5542`) and
`DiscoverPeers` (`5608`). The caller then returns from `backgroundTrainer`
(`6628-6634`). In `--evolution` the process does not exit: `main` is parked on
`<-sigCh` (`7100-7108`).

**Mechanism.** A hibernated organism keeps its full model, its cooccurrence field
and its tokenizer resident — the phone measured `VmHWM` 231-240 MB per adolescent
organism (`MOLEQULALOG2.md:135-138`) — while telling the governor that a
population slot is free. The next divide admits a child into memory that is still
occupied. README:665 states "it saves state and sleeps. Resources freed for the
living"; nothing in the code frees anything. There is also no wake path: no code
reads `status='sleeping'` back, so the state is terminal.

Related: hibernation is unreachable in practice anyway. `DecideAction` reaches
CASE 7 only when `action == "steady"` (`5269`), i.e. `|SyntropyTrend| ≤ 0.01` and
the field deviation inside its band (`5217-5248`), and `shouldHibernate` then
requires `|mean of the last 8 burst deltas| < 0.01` (`5424-5432`). Those deltas
are sampling noise (see P1-5): the archived parent's four deltas are +0.44, +0.82,
−0.26, −3.47 (`runpod/2026-06-04_mitosis_§9/org_1780540885_6400/birth.json`).
`grep -rl hibernat runpod/…_§9/` returns no organism log, and the phone colony
recorded 0 hibernate events (`MOLEQULALOG2.md:140`).

**Minimal repair.** `performHibernation` must end the process (signal `stop`, then
exit from `main`) so the RSS is actually returned, and `MarkHibernating` should
run after the model is released, not before. If hibernation is to fire at all, the
`< 0.01` plateau test must be replaced by a test on a quantity that is not
dominated by QuickLoss sampling noise.

---

## P0-3 — `MaxOrganisms` is a head count, never a byte budget

**Evidence.** `MaxOrganisms: 16` (`molequla.go:345`), checked only as a row count
(`5542`, `6615`). A repo-wide grep for `MemAvailable`, `meminfo`, `VmRSS`,
`RLIMIT`, `sysinfo` over `*.go` returns nothing; the only RSS reader in the tree
is `pod_watchdog.sh:80`, a pod-side observer that takes no action.

**Mechanism.** Sixteen organisms is the ceiling regardless of what they weigh. The
phone measured 231-240 MB per organism at adolescent and 127 MB at child
(`MOLEQULALOG2.md:135-138`) with lowest `MemAvailable` 1930056 kB during that run
(`MOLEQULALOG2.md:145-146`). Adult (320 d / 6 L, ~10 M params, README:229) is
larger again and was not measured on ARM. Sixteen adolescents alone is ~3.8 GB
against ~1.9 GB available. The default is an OOM contract, not a safety margin.

**Minimal repair.** Add an `rss_kb` column written by `Heartbeat` (`5588`) and a
second conjunct to the `AcquireMitosisSlot` statement: admit only when
`SUM(rss_kb) + expected_child_rss < MaxColonyRSS`. The child's expected RSS is
already knowable — it is the parent's, since it loads the parent checkpoint.
`MaxOrganisms` then becomes the coarse backstop it is documented as
(`molequla.go:220`) instead of the only bound. On this node the phone value must
be computed from measured adult RSS, per the repo's own rule (`CLAUDE.md:86-87`).

---

## P0-4 — In `--evolution` the corpus reservoir cap is unreachable, so the corpus and the field grow without bound

**Evidence.** `updateReservoirCorpus` is the only caller of `saveCorpusLines` and
the only place `CFG.MaxCorpusLines` is used (`molequla.go:3737-3766`, called at
`6447` and `7125`). It returns 0 before touching the file when
`extractCandidateSentences(dbRecentMessages(db, 64))` is empty (`3738-3742`). The
`messages` table is written only by `dbAddMessage`, whose two callers are both
inside the interactive REPL loop (`7124`, `7181`), which `--evolution` never
reaches (it returns at `7108`). Meanwhile `dnaRead` appends every consumed
fragment to that same file with `O_APPEND` (`5922-5926`).

**Mechanism.** In the only mode the ecology runs in, the corpus file is
append-only and uncapped. The colony measured air consuming 3118052 B in 1200 s
(`MOLEQULALOG2.md:148-149`) against a 118508 B seed (measured here with `awk` over
`nonames_air.txt`, counting non-empty lines truncated at `MaxLineChars`) — a 26×
growth in twenty minutes. Every 30 ticks the loop re-reads the whole file and
rebuilds the cooccurrence field over it (`6446-6456`), and `BuildFromCorpus`
constructs unigram, bigram, trigram, 4-gram and co-occurrence maps into fresh
temporaries before swapping (`4180-4186`), so at the rebuild instant the organism
holds two complete fields. Tick time and resident memory therefore both climb
monotonically for the life of the run, and the transient doubling is exactly what
the Android low-memory killer samples. The comment at `molequla.go:255` asserts
the opposite — "corpus FILE capped at `MaxCorpusLines` so field-rebuild stays
bounded" — which is true only in interactive mode.

**Minimal repair.** Call the reservoir trim unconditionally on the periodic path
(`6447`) — trim on line count alone, with the DB-sentence mix optional — or cap
the file inside `dnaRead` after the append. Rebuilding the field incrementally
from the appended fragment instead of from the whole corpus removes the O(corpus)
term entirely; the tick already knows exactly which bytes are new.

---

## P1-1 — DNA exchange: exclusive consume, unbounded drain, stable phase, and an organism that cannot eat its own backlog

**Evidence.** `dnaElements = {earth, air, water, fire}` (`5834`). `dnaRead`
iterates that fixed order, skipping its own element (`5889-5892`); for each
directory it takes every `.txt` `os.ReadDir` returns with no per-tick cap
(`5898-5934`), mirrors each file to `../dna/seen/<emitter>/` (`5918-5920`),
appends the text to its own corpus (`5922-5926`), feeds the quantum buffer
(`5929-5931`) and then `os.Remove`s the source (`5932`). There is no lock, no
marker file, no cursor. The tick body is `dnaWrite` then `dnaRead` (`6638-6651`)
followed by a fixed `time.Sleep(CFG.TrainTickSeconds)` with no jitter (`6696`,
`TrainTickSeconds: 0.25` at `315`). `dnaWrite` runs unconditionally every tick and
generates through `GenerateResonant` (`5849`), capped at `MaxGenTokens: 180`
(`307`).

**Mechanism, part one — why intake is winner-take-all rather than proportional.**
Ownership of a fragment is decided by whichever process's `ReadDir` + `ReadFile` +
append lands first, and the winner takes the entire accumulated backlog of that
directory in one pass, not a share of it. The measured read events are therefore
few and enormous: air took 3118052 B in 18 reads, fire 2614666 in 19, water
1612725 in 10 (`MOLEQULALOG2.md:148-149`) — about 170 KB, i.e. ~34 fragments of
5 KB, per successful read. Because all four organisms run the identical loop body
and the same unjittered sleep, their periods are near-identical and their phase
offsets — set once by launch order (`launcher.sh:41-48`) — are stable. A reader
whose scan consistently lands just before its siblings' writes sees an empty
directory on every pass, and nothing in the code ever re-randomizes that phase.
There is no fairness property here to appeal to; the exchange is an unsynchronized
poll.

**Mechanism, part two — why earth specifically lost, structurally.** The colony
run inherited the tree left by the preceding single-organism run, in which earth
alone emitted 1297 fragments, 5107 B mean, 6.62 MB total, "none consumed"
(`MOLEQULALOG2.md:97-100`). That backlog sat in `../dna/output/earth`. When the
colony started it was the only food in the tree — and `dnaRead` refuses to read
its own element (`5890-5892`), so earth was excluded by construction from the
entire existing supply while air, water and fire split it. The arithmetic closes:
the three winners took 7.35 MB against earth's 6.62 MB backlog plus whatever the
three emitted fresh during the same 1200 s, and earth's 35734 B is what it managed
to take from those fresh sibling emissions. Earth's growth clock reads 170287 at
exit (`MOLEQULALOG2.md:135`), which is exactly its measured seed mass 134553
(measured here by `awk` over `nonames_earth.txt`, matching the `[debug-onto]
ingested=134553` in the single-organism run, `MOLEQULALOG2.md:96`) plus 35734.
Stage 2 while its siblings reached stage 3. "Whoever loses the scan loses growth"
is right, and the loss is not a coin flip — the shape of the tree decided it
before the colony started.

This also contradicts README:659 ("siblings consume, micro-train, emit … cross-
pollination outpaces any single organism's learning rate"): three of the four
readers never see a given fragment in their corpus, only as a logit boost through
the `seen/` mirror if `--cross-graze` is on.

**Minimal repair.** Stop deleting in the reader. Give `dnaWrite` a zero-padded
monotonic sequence in the filename (`5876` currently emits
`gen_<unix-sec>_<tick>.txt`, which is neither unique across processes nor
lexically ordered past step 9), leave fragments in place, and give each reader a
cursor file `../dna/cursor/<reader>/<emitter>` holding the last sequence it
consumed. `dnaRead` then consumes everything after its cursor, capped at N per
tick, and advances it; every sibling eats every fragment and no reader can starve
another. Deletion moves to the emitter: `dnaWrite` prunes its own directory to the
last K fragments or to what is below the minimum of the readers' cursors. Add
jitter to the sleep at `6696` so phase cannot lock. If own-emission exclusion is
to stay, it should at least not orphan a backlog: a fragment nobody can eat should
be pruned by its emitter, not left as permanent litter.

---

## P1-2 — `../dna/output` and `../dna/seen` are never pruned; scan cost grows with the litter

**Evidence.** The only `os.Remove` / `os.RemoveAll` calls in the whole Go tree are
the checkpoint tempfile (`3936`), the pid file (`5654`, unreachable — see P2-5),
the undersized fragment (`5909`) and the consumed fragment (`5932`). Nothing
prunes unconsumed fragments in `../dna/output/<e>/`, nothing ever prunes
`../dna/seen/<e>/`, and nothing removes `$HOME/.molequla/<childID>/` after a child
dies.

**Mechanism.** Emission is unconditional and per-tick; consumption is rare. The
single-organism run left 1297 files / 6.62 MB in 900 s (`MOLEQULALOG2.md:98-99`),
roughly 26 MB per organism per hour, so a four-organism colony litters ~106 MB/h,
and every fragment that *is* consumed is copied into `seen/` where it stays
forever — the consumed share is duplicated, not moved. Both directories are then
walked repeatedly: `dnaRead` does a full `os.ReadDir` per sibling per tick
(`5894`), and `CrossField.MaybeRefresh` does a `ReadDir` plus an `e.Info()` stat
per entry plus an mtime sort per sibling every 30 s (`cross_graze.go:96`, `110`,
`116`). Scan cost is O(litter), so the whole colony's tick slows as the tree fills,
which feeds directly back into P1-1.

**Minimal repair.** A retention bound in `dnaWrite`: after writing, delete
everything in its own output directory older than the cursor minimum (or older
than K files / T minutes). Same for `seen/` — it exists as a paper artifact
(`5912-5917`), so cap it at the window cross-graze actually uses
(`RecentCap: 64` tokens, `cross_graze.go:73`) rather than keeping the full stream.

---

## P1-3 — Cross-graze `SeenFiles` wipe re-ingests the entire seen tree

**Evidence.** `cross_graze.go:149-151`: when `len(SeenFiles) > SeenCap` (2048) the
dedup map is replaced with an empty one. The comment predicts "a few duplicate
ingestions possible right after wipe" (`146-148`).

**Mechanism.** The prediction is wrong in the direction that matters. After the
wipe, the next `MaybeRefresh` finds *every* file in every sibling's `seen/`
directory unseen, and re-reads and re-tokenizes all of them (`117-143`) — with
`seen/` unbounded (P1-2) that is a full re-ingest of the accumulated stream inside
one generation call, under the model lock, on a 250 ms tick. The bound the comment
reasons from ("~8h × 3 siblings × ~1 emission/min") is off by the measured
emission rate: this phone emits ~1.44 fragments/s per organism
(1297 writes / 900 s, `MOLEQULALOG2.md:98-99`), so 2048 is reached in minutes, not
hours, and the wipe is a recurring stall rather than a once-a-day event.

**Minimal repair.** Replace the wipe with eviction of the oldest keys, or drop
`SeenFiles` entirely and key resumption on the same per-emitter cursor P1-1
introduces — `MaybeRefresh` then reads only what is new by construction.

---

## P1-4 — `relieveOverload` empties the burst history the child is supposed to inherit

**Evidence.** `performMitosis` calls `syntracker.relieveOverload()` at `5739`,
*before* capturing `"burst_history": syntracker.BurstHistory` into `birth.json` at
`5754` — the ordering is deliberate and commented (`5734-5737`).
`relieveOverload` keeps only bursts with `LossAfter < CFG.OverloadLossHigh`
(`5104-5110`), and `OverloadLossHigh: 5.0` (`342`). The loss path fires precisely
when the mean `LossAfter` over the window exceeds 5.0 (`5364`).

**Mechanism.** On a loss-path divide the surviving history is the subset of recent
bursts with loss below the threshold that the divide just certified was exceeded —
typically empty. The archived manifest makes this concrete: the four inherited
records are `LossAfter` 13.06, 12.67, 10.43 and 10.50
(`runpod/2026-06-04_mitosis_§9/org_1780540885_6400/birth.json`), every one above
5.0. That manifest was written before the governor landed; run the same four
records through today's `relieveOverload` and `birth.json` carries
`"burst_history":[]`. README:666 — "the child gets its parent's meta-learning
experience (syntracker lineage). It doesn't start from zero wisdom" — and the
paper's Result 8 description of the manifest are both now false for the loss path.
The parsing machinery downstream (`6867-6886`, `7085-7087`, `6424-6429`) is
intact; it just receives nothing.

**Minimal repair.** Capture `burst_history` into the birth map *before* calling
`relieveOverload`, or pass the pre-relief copy that is already snapshotted at
`5723` (`savedBH`). The parent still gets relieved; the child still gets its
lineage. One statement reordering.

---

## P1-5 — The overload gate's "loss is not falling" clause is sampling noise

**Evidence.** `lossOverload` requires `meanLossAfter > OverloadLossHigh` **and**
`meanDelta > -CFG.OverloadLossEps` with `OverloadLossEps: 0.05` (`5350-5365`,
`343`). The deltas come from `RecordBurst(action, lossBefore, lossAfter)`
(`5090-5095`, called at `6571`), whose inputs are two `model.QuickLoss(tok, docs, 4)`
calls (`6557`, `6568`). `QuickLoss` averages the loss over **4 documents drawn at
random** each call (`3023-3039`, `doc := docs[rand.Intn(len(docs))]`) — before and
after are measured on different samples.

**Mechanism.** The quantity being compared to a 0.05 threshold has a per-call
sampling spread far larger than 0.05: the archived adult's four recorded deltas
are +0.44, +0.82, −0.26 and −3.47
(`runpod/2026-06-04_mitosis_§9/org_1780540885_6400/birth.json`), i.e. swings of
3.5 nats, seventy times the epsilon. The clause therefore contributes close to a
coin flip per burst, and the gate reduces in practice to "mean loss > 5.0". The
same noise is what `ActionEffectiveness` (`5127-5140`) and `shouldHibernate`'s
`|avgDelta| < 0.01` (`5430`) are reading, which is why hibernate never fires.

**Minimal repair.** Measure before and after on the *same* sample: draw the
document indices once per burst and reuse them for both `QuickLoss` calls, or take
the delta from the trainer's own averaged tape loss, which `ntBurstTrain` already
computes and currently only prints (`notorch_trainer.go:405`, `412-415`).

---

## P1-6 — A burst that trains nothing is recorded as a burst that could not reduce loss

**Evidence.** `ntTrainCore` skips the optimizer step whenever the NaN guard trips
(`notorch_trainer.go:349-352`) and skips the whole step when the sampled document
tokenizes to fewer than two ids (`291-293`). Neither is counted or surfaced:
`ntBurstTrain` prints only when `n > 0` (`412-415`) and returns nothing. The caller
records the burst unconditionally (`molequla.go:6571`).

**Mechanism.** A burst in which every step was skipped leaves the weights
untouched, so `lossAfter ≈ lossBefore`, so `meanDelta > -eps` — the exact
signature of "the bursts cannot bring the loss down" that `lossOverload` reads as
overwhelm (`5364`) and that the paper calls the organism's faithful overwhelm
signal. The gate cannot distinguish "cannot assimilate" from "did not train". On a
phone this matters more than on a pod: three bursts is the whole window
(`OverloadLossWindow: 3`, `344`).

**Minimal repair.** Have `ntBurstTrain` return the number of optimizer steps
actually applied and have the tick skip `RecordBurst` when it is zero, logging the
skip. Same for `amlBurstTrain` (`aml_trainer.go:260`).

---

## P1-7 — The training lock can never be acquired; README says it serializes the colony

**Evidence.** Both `AcquireTrainingLock` call sites are gated on
`CFG.CoordinateWarmup` (`molequla.go:6473`, `6520`). `CoordinateWarmup` is
declared (`244`), left at its zero value (absent from the `CFG` literal,
`247-363`), never set by `parseCLIArgs` (`6259-6301`), and never set from
`birth.json`, which carries only `corpus_path` / `db_path` / `ckpt_path` /
`burst_history` (`5748-5755`, parsed at `6853-6889`). It cannot become true in any
shipped invocation. `ReleaseTrainingLock` is nevertheless called unconditionally
after every burst (`6592-6594`).

**Mechanism.** The `training_lock` table (`5509-5510`) is created, deleted from,
and never inserted into. README:664 — "Training lock: Atomic check-and-acquire via
SQL prevents multiple organisms from training simultaneously. Cooperative
scheduling — they take turns" — describes a mechanism that is off by construction;
all organisms train in parallel, which is what the 2026-06-03 comment at
`6513-6519` intended but the README was never updated to say.

**Minimal repair.** Either add `--coordinate-warmup` to `parseCLIArgs` and pass it
to children in `childArgs` (`5776-5786`), or delete the flag, the two call sites,
the unconditional release and the table, and correct README:664. On a phone the
first is the useful one: serialized bursts are how four organisms fit in 2 GB.

---

## P1-8 — The post-growth freeze does not do what it is documented to do, and the deltas it protects are never trained

**Evidence.** README:240 — "500-step freeze period: delta-only training to
stabilize post-growth". The default trainer is `Trainer: "notorch"` (`286`).
`ntTrainCore` registers exactly `ntContentParams` — `wte`, per layer
`wq/wk/wv/wo/fc_g/fc_v/fc2`, `lm_head` — plus the RRPRAM factors
(`notorch_trainer.go:48-59`, `223`, `313-335`). `model.Deltas` appears nowhere in
that file. The freeze's only effects on this path are an LR multiplier of
`PostGrowthLRScale: 0.3` during warmup (`450-452`, `271`), *no* effect at all
during a burst (`ntBurstTrain` computes `lr` at `404` without consulting it), and
a counter decrement (`407-410`, `456-459`). Its one real effect is gating
`MaybeGrowArchitecture` (`molequla.go:2206`).

**Mechanism.** The freeze is a growth-rate limiter, not a stabilizer, and it is
consumed immediately: growth sets it to 500 (`2426`), the next tick runs the
per-stage warmup whose first segment alone is `0.4 × 400 × ⌈√(n_embd/16)⌉` = 480
steps at adolescent (`6491-6494`), so it is exhausted before the warmup finishes.
Downstream of that, `AddDeltaModule` is still called on the syntropy signal
(`6604-6609`) and still appends adapters that participate in the forward pass but
that no live trainer ever updates — README:262-272 "Delta Adapters — LoRA-style,
Never Forget" describes a subsystem that grows and is never trained. The immune
system inherits the same hole: it snapshots and restores only deltas
(`6551`, `6577`, `2105`, `2129`) while the burst changed only base weights, so
"[immune] NOISE DETECTED … Rolling back deltas" (`6576`) rolls back nothing the
burst did.

**Minimal repair.** Decide which it is. If deltas are to stay in the design, the
notorch burst must include them in the registered parameter set and the immune
rollback becomes meaningful; if they are not, stop growing them (`6604-6609`),
stop snapshotting them, and correct README:240 and README:262-272 to describe the
LR dampening and growth gate that the freeze actually is.

---

## P1-9 — The growth checkpoint can be silently dropped by the write-storm debouncer

**Evidence.** `SaveCheckpoint(model, tok, "")` returns `nil` without writing when
the shared `ckptDebounce` has fired within `CheckpointMinInterval` (30 s)
(`3873-3880`, `346`). The default path is used for the post-growth save
(`6666`), the post-warmup save (`6502`), the BPE-retrain save (`6464`) and the
per-burst save (`6585`) — all of them share one debouncer instance (`3871`).

**Mechanism.** A growth event that lands within 30 s of a burst checkpoint is not
persisted. The architecture change, the reset `growthStepOffset`, the advanced
`corpusIngestedTotal` and `lastWarmupStage` all live only in memory
(`3921-3924`). If the process then dies — and on a phone the likely death is the
low-memory killer, which fires precisely after growth doubles the footprint
(`MOLEQULALOG2.md:143-144`) — the on-disk checkpoint still holds the previous
stage and a stale ingest clock, so the organism reloads one stage back and re-runs
the warmup it already paid for. Silent success is the worst shape here: the
function returns `nil` and the caller logs the growth as done (`6671-6672`).

**Minimal repair.** Treat growth as an explicit-path save, the way the mitosis
parent checkpoint already is: pass the real path so the debounce is bypassed
(`3878`), or give `SaveCheckpoint` a `force` argument used at `6666` and `6502`.
The burst path at `6585` is the one that needs throttling, not these.

---

## P1-10 — On Railway, the governor's database and every child live outside the persistent volume

**Evidence.** `launcher.sh:11` roots all state at `${MOLEQULA_DATA_DIR:-/data}`,
the mounted volume (`Dockerfile:53-55`). But `swarmDir = $HOME/.molequla/swarm`
(`molequla.go:5445`) and the child base directory is `$HOME/.molequla`
(`5698`). `$HOME` in the runtime image is not `/data` — nothing in the Dockerfile
sets it, so it is root's home.

**Mechanism.** `mesh.db` — the atomic admit, the population count, the locks
(`5484`) — and every spawned child's directory, checkpoint, database and log
(`5742`, `5752`, `5790`) are written to the container's ephemeral layer. A
Railway restart, which `launcher.sh:57-59` is explicitly built to trigger, wipes
the entire population registry and every child ever born while leaving the four
seed organisms' corpora on the volume. The colony restarts with no memory that it
ever reproduced.

**Minimal repair.** Derive `swarmDir` and the mitosis base directory from
`MOLEQULA_DATA_DIR` with `$HOME` as the fallback, one `os.Getenv` at each of
`5445` and `5698`.

---

## P1-11 — The child shares the parent's working directory, corpus file and DNA output directory

**Evidence.** `exec.Command(exePath, childArgs...)` sets no `Dir` (`5787`), so the
child inherits the parent's cwd. `birth.json` carries `"corpus_path": CFG.CorpusPath`
(`5751`), a relative name — `nonames_fire.txt` in the archived manifest — which
`main` copies into `CFG.CorpusPath` (`6857-6859`). The child's element is derived
from that same basename (`5783-5786`), so it writes DNA into the same
`../dna/output/<element>/` directory as its parent (`5874`), with filenames
`gen_<unix-seconds>_<tickCount>.txt` (`5876`) where `tickCount` starts at 1 in
every process (`6415`, `6438`).

**Mechanism.** Three consequences. First, filename collision: a child in its first
seconds writes `gen_<same second>_<same small tick>.txt` into the directory its
parent is also writing, and `os.WriteFile` silently overwrites — DNA is lost, and
the probability rises with every child of the same element. Second, the child
competes with its parent for the same sibling fragments, so mitosis strictly
reduces the parent's intake, which is the food its growth clock runs on. Third,
both processes append to one corpus file (`5922-5926`); the truncating rewrite in
`saveCorpusLines` (`3636`) would corrupt it outright, and today that is survived
only accidentally, because the rewrite path is unreachable in evolution mode
(P0-4).

**Minimal repair.** Set `cmd.Dir = childDir` and write an absolute
`corpus_path` into `birth.json` pointing at a per-child copy (or at a read-only
seed plus the child's own append file). Put the organism id into the DNA filename
at `5876` so no two emitters can collide.

---

## P1-12 — The deployed build omits `-a`, which this repo documents as mandatory

**Evidence.** `Dockerfile:37` — `RUN go build -trimpath -ldflags="-s -w" -o /out/molequla .`
with `CGO_ENABLED=1` (`33`). The repo's own `CLAUDE.md:53` and `CLAUDE.md:67` and
README:710 all state that `-a` is required for CGO builds because Go's cache does
not recompile the C files.

**Mechanism.** A cold image build has an empty cache and is safe by luck; any
build that reuses the `COPY . .` layer's cached compilation runs stale C against
new Go. This is the exact trap the repo lists first among its known bug patterns.
Separately, `-march=native` (`Dockerfile:34`) bakes the *builder's* ISA into a
binary that runs on a different Railway host; a narrower `-mtune` with an explicit
baseline `-march` is the safe form.

**Minimal repair.** Add `-a` at `Dockerfile:37`. Replace `-march=native` with an
explicit baseline for the deployment target.

---

## P1-13 — `--evolution` exits without unregistering or saving

**Evidence.** The evolution branch returns at `7108` on SIGINT/SIGTERM. The final
`SaveCheckpoint` and `swarm.Unregister()` are at `7208-7211`, after the REPL loop,
on a path evolution never reaches.

**Mechanism.** The colony's normal stop is SIGTERM (`timeout`, per
`MOLEQULALOG2.md:90` and `128`). Everything learned since the last debounced
checkpoint is lost, the mesh row stays `status='alive'` (it ages out of the cap
after 60 s, so the governor self-heals, but `DiscoverPeers` and any external
reader see a phantom), and the pid file is left behind — visibly so:
`/root/.molequla/swarm/{earth,air,water,fire}.pid` are still present on this node
from the 2026-09-13 colony run, with `mesh.db` alongside.

**Minimal repair.** In the signal branch, after `close(stop)`, take `model.mu`,
`SaveCheckpoint(model, tok, CFG.CkptPath)` on the explicit path so the debouncer
cannot swallow it, and `swarm.Unregister()`.

---

## P1-14 — README Quick Start passes three flags the parser does not know

**Evidence.** README:735-737 pass `--corpus nonames_$d.txt`, `--db memory.sqlite3`,
`--ckpt molequla_ckpt.json`. `parseCLIArgs` (`molequla.go:6259-6301`) recognizes
`--organism-id`, `--config`, `--element`, `--evolution`, `--spa-gate`,
`--corpus-overlay`, `--trainer`, `--zero-warmup`, `--gpu`, `--cross-graze`, and
silently ignores everything else. `sweep.sh:25` passes the same three.

**Mechanism.** The documented invocation happens to behave correctly only because
the ignored values coincide with the compiled defaults (`DBPath: "memory.sqlite3"`,
`CkptPath: "molequla_ckpt.json"`, `249-250`) and because `--element` sets the
corpus path itself (`6830-6845`). Anyone who passes a *different* value gets it
dropped without a word. Conversely `--trainer` and `--zero-warmup` exist and are
absent from the README's flag list (`743-748`).

**Minimal repair.** Either parse the three flags (three `else if` arms, writing
`CFG.CorpusPath` before the element switch overrides it — note the ordering) or
delete them from README:735-737 and `sweep.sh:25`. In both cases, make
`parseCLIArgs` fail loudly on an unrecognized `--flag` instead of ignoring it.

---

## P1-15 — `field_steering`: the mycelium's only channel does not exist for the Go colony

**Evidence, re-verified and extended.** The table is created at
`ariannamethod/method.py:235` and written at `mycelium.py:1194` and
`ariannamethod/method.py:447`. The only reader in the whole tree is
`molequla.rs:3265-3267`. `molequla.go` has zero occurrences of the string, and
`initMeshDB` (`5483-5523`) does not create the table, so on a Go-only run it may
not exist at all. Extension: the traffic is dead in the other direction too —
`SwarmRegistry.LogMessage` writes the mesh `messages` table (`5636-5644`, called
once, on `mitosis:spawn`, `5763`) and no Go code ever selects from it; the only
`FROM messages` query in the Go core (`3545`) reads the *organism's own*
`memory.sqlite3` conversation table, a different schema. README:643's "Seasonal
Controller" likewise has no Go implementation (zero `season` hits in `*.go`).

**Mechanism.** `launcher.sh:41-48`, the Dockerfile CMD and every measured run
(§9, both phone runs) execute the Go binary. The coordinating tier is therefore
connected to nothing in every run that has produced a result. README:627 is
honest that only `molequla.rs` reads it, but frames mycelium as the tier that
coordinates the ecology; for the ecology as deployed, it does not.

**Minimal repair.** If mycelium is to steer the Go colony, the read is small: add
the table to `initMeshDB`, and read `action, strength` in the heartbeat block
(`6678-6694`) where `DiscoverPeers` already runs, feeding it into
`syntracker.SwarmInfo` alongside the peers. If it is not, say so in README:627 and
in the file list at README:806.

---

## P2-1 — Dead code paths

- `trainSteps` (`molequla.go:6317-6409`) has no caller anywhere, tests included
  (`grep -rn '\btrainSteps(' --include=*.go` returns the declaration only). With it
  die its exclusive dependants: `AdamStep` (`2778`, called only at `6399`/`6402`),
  `AllDeltaParams` (`1985`), `AllBaseParams`, and the config fields `Beta1`,
  `Beta2`, `EpsAdam`, `GradClip`, `AccumSteps`, `BatchSize`, `MaxTotalSteps`,
  each of which has exactly one non-test reference, inside that dead path.
- `notorchTrainSteps` (`6054-6252`, ~200 lines) has one reference: a commented-out
  call at `6499`. README:617 documents it as a live subsystem.
- `CrossField.MetricBoost` (`cross_graze.go:51`) is never assigned; the branch at
  `181-185` is unreachable. The comment calls it "the metrics half" of the design.

**Minimal repair.** Delete, or wire. Leaving a documented subsystem with no caller
is what made the README's promises unfalsifiable in the first place.

---

## P2-2 — Config fields nothing reads, and coefficients maintained in two places

`MinNewChars` (`56`, `253`), `FreezeBaseAfterWarm` (`87`, `281`), `MetaCBigram`,
`MetaCTrigram`, `MetaCHebbian`, `MetaCDestiny`, `MetaCProphecy` (`138-142`,
`287-291`) and `MetaLogitOverlayFloor` (`144`, `293`) have zero non-test readers
(`grep -rn "CFG\.<field>" --include=*.go . | grep -v _test`). The five `MetaC*`
values are duplicated verbatim as package-level constants in
`metaweights_overlay.go:31` (`Heb: 1.0, Pro: 0.7, Ds: 0.15, Bg: 15.0, Tg: 10.0`
against `CFG.MetaCHebbian: 1.0 … MetaCBigram: 15.0`), so editing the config
changes nothing while looking as if it does — and every one of them is serialized
into every checkpoint (`3890`), where they will be read back one day as if they
meant something.

**Minimal repair.** Delete the eight fields, or make `metaweights_overlay.go` read
them. One of the two, not both.

---

## P2-3 — The freeze counter lives in five places; the architecture in two

`growthFreezeRemaining` is decremented at `molequla.go:6238-6243` (dead path),
`molequla.go:6346-6349` (dead path), `aml_trainer.go:239-242`,
`aml_trainer.go:334-337`, `notorch_trainer.go:407-410` and
`notorch_trainer.go:456-459`. All five live copies are currently correct — the
`ff6ad49` regression the repo's `CLAUDE.md:68-70` records is closed — but the repo
states its own rule: "If a counter lives in N places, N is the bug." Two of the
five are in code nothing calls, which is how the third one went missing last time.

Same shape one level up: the architecture is stored twice, as `gpt.NEmbd /
NLayer / NHead / HeadDim` (`2367-2370`) and as `CFG.NEmbd / NLayer / NHead /
HeadTypes` (`2374-2377`), and readers are split between them — `AddDeltaModule`
reads `CFG` (`1957-1972`) while the growth path's own delta code reads the local
`newEmbd` (`2341-2358`), and `LoadCheckpoint` restores only those four CFG fields
from the saved config (`3986-4007`).

**Minimal repair.** One decrement, in one place: a `model.tickFreeze(steps)` method
called by whichever trainer ran. One architecture source: `gpt.*`, with `CFG`
holding only the seed values.

---

## P2-4 — The child does not inherit the trainer or the coherence toggles

`childArgs` carries `--organism-id`, `--config`, `--evolution`, and conditionally
`--gpu`, `--cross-graze`, `--element` (`5776-5786`). It does not carry
`--trainer`, `--spa-gate`, `--corpus-overlay` or `--zero-warmup`, and `birth.json`
carries none of them either (`5748-5755`). A colony running `--trainer aml` spawns
children on the notorch default; a measurement run with `--corpus-overlay` spawns
children without it, and the run's own children are then not the thing being
measured.

---

## P2-5 — Pid files: written, never read, never cleaned; the watchdog watches a path nothing writes

`SwarmRegistry.Register` writes `$HOME/.molequla/swarm/<id>.pid` (`5468-5476`).
Nothing in the Go tree reads it. `Unregister` removes it (`5653-5655`) on a path
evolution mode never reaches (P1-13) — confirmed on this node: four `.pid` files
from the 2026-09-13 colony are still present. `pod_watchdog.sh:44` and `:75` look
for `work_<e>/org.pid`, a path nothing in the repository writes, so the
watchdog's DEAD and RSS_HIGH classes can never fire.

---

## P2-6 — Smaller items, each with its line

- `QuantumBuffer.ShouldTrigger` (`4127-4135`) calls "novelty" the buffer's own
  type/token ratio (`4120-4125`), which has no reference to the corpus at all;
  re-reading text the organism already holds scores identically to new text.
  README:276 says "sufficiently different from prior corpus". With
  `QBMinNovelty: 0.15` (`333`) almost any fragment passes, so the documented
  two-condition trigger is effectively unconditional.
- `LastBurstTime` is zero-valued at construction (`4105-4107`), so the first
  `ShouldTrigger` always passes the cooldown — the same class of bug that was
  fixed for the mitosis cooldown by seeding `LastMitosisTime` at `5084`.
- README:224 heads the ontogenesis column "Corpus Threshold" and README:233 says
  "When the corpus crosses a threshold", but the gate reads `corpusIngestedTotal`
  (`2209`), a monotonic ingest clock that never falls, while the corpus file is a
  separate number the tick prints beside it (`6657-6662`).
- `dnaWrite` pads each fragment toward `DNAFragmentTargetBytes: 5000` (`255`) with
  randomly sampled lines of the emitter's *own corpus* (`5859-5868`), and the
  receiver's growth clock counts those bytes (`6642-6646`). The ontogenesis clock
  is therefore driven mostly by copying static element corpora between organisms,
  not by organism speech. Worth reconciling against the paper's "no corpus
  seeding" framing (`docs/molequla_paper.md:510-512`) before the next run —
  flagged for Oleg's decision, not asserted as an error.
- `dbLogGrowth` reads `model.Base`, `model.Deltas` and `GammaStats` with no lock of
  its own (`3563-3583`) and is called both under `model.mu` (`5765`, `5826`,
  `6587`, `6671`) and outside it (`6508`). Harmless while evolution mode is
  single-goroutine; a trap for whoever adds the second writer.
- An orphaned zero-byte
  `runpod/2026-05-14/…/work_water/molequla_ckpt.json.tmp` shows the
  create-encode-rename sequence (`3928-3939`) leaves its temp file behind when the
  process dies mid-write. The rename itself is correct; only the litter is not.
- `governor_test.go:20` and `:126` build their own `organisms` schema by hand
  rather than calling `initMeshDB`, and every inserted row carries a current
  heartbeat — so the cap's 60 s staleness window (P0-1), the one clause that
  actually fails in production, has no test.

---

## Unmeasured, must be measured before the phone deployment

`SaveCheckpoint` serializes the full float64 weight set as JSON text
(`3892-3907`, `3933`) and is called on every burst (`6585`), every growth
(`6666`), every warmup (`6502`) and every BPE retrain (`6464`) — throttled to one
per 30 s by `CheckpointMinInterval` (`346`) — plus once per divide with the
throttle bypassed, under `model.mu` (`5743`). At adult the base is ~10 M
parameters (README:229, `GrowthStages[5]`, `268`), and Go renders a trained
float64 as roughly twenty characters, so the file is of order 10^8 bytes and the
model lock is held for the whole encode. **This is arithmetic from the format, not
a measurement**; no adult checkpoint exists on this node (the one archived
`molequla_ckpt.json.tmp` is zero bytes). Measure the checkpoint size and the
encode wall time at child and adolescent on the big cores before the colony is
left running unattended on flash; if the estimate holds, the checkpoint format,
not the debounce interval, is the write-storm.

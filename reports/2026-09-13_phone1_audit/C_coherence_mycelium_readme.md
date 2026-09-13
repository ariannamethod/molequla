# Audit C — coherence layer, mycelium, README-vs-code contract

Scope: `metaweights_overlay.go`, `metaweights_seeding.go`, `spa_coherence.go`, `cross_graze.go`,
the generation path in `molequla.go`, `mycelium.py`, `ariannamethod/{method.py,sentinel.py}`,
`ariannamethod/ariannamethod.c` (`am_harmonic_*` / `am_method_*`), the `mesh.db` schema, and the
whole `README.md`.

Read-only pass. Nothing built, nothing run, no Python executed, no git state touched. Every number
below is a line I read; every claim is reproducible with `grep`. Repo root
`/data/data/com.termux/files/home/arianna/molequla`, all paths relative to it.

Severity: **P0** breaks a shipped feature silently · **P1** wrong behaviour or wrong documentation
that will mislead the next change · **P2** drift, dead surface, latent hazard.

---

## 1. OVERLAY — what `--corpus-overlay` actually changes

### 1.1 The flag and what it switches on

`--corpus-overlay` sets exactly one bit: `CFG.CorpusLogitOverlay = true` (`molequla.go:6274-6275`;
default `false` at `molequla.go:285`, JSON key `corpus_logit_overlay` at `molequla.go:108`). That
bit is read in four places, and they are not equivalent:

| site | what it gates | reads |
|---|---|---|
| `molequla.go:6950-6952` | `SeedEmbeddingsFromMetaweights(model, tmpCooccur, 0.15)` at birth | the flag |
| `molequla.go:4550-4553` | allocation of the per-call `OverlayScratch` + `PrepareStatic` | the flag |
| `molequla.go:4606` | `overlayActive`, the per-step overlay | the flag |
| `molequla.go:4831` | **skips the prob-space corpus blend** | the flag |

The asymmetry between the third and the fourth is a defect in its own right — see P1 `C-OVL-05`.

### 1.2 Order of application inside one generation step

`generateResonantLocked` (`molequla.go:4483-5034`), per token:

1. `logits := model.ForwardStep(...)` — `molequla.go:4560`.
2. frequency + presence penalty, in place on `logits.Data` — `molequla.go:4563-4571`.
3. `overlaidLogits := logits.Data` (an **alias**, not a copy) — `molequla.go:4605`.
4. if the flag is on: `overlaidLogits = make(...)` + `copy(...)` — `molequla.go:4613-4614`. From
   here the two slices are detached.
5. `tmag = mean(|logits.Data|)` measured on the **pre-overlay** model output —
   `molequla.go:4618-4628`; `untrainedRegime = tmag <= 1.0` — `molequla.go:4634`.
6. if `untrainedRegime`: `MetaweightsOverlay(...)` then `MetaweightsRepetitionPenalty(...)` —
   `molequla.go:4647-4648`. **else `overlayActive = false`** — `molequla.go:4650`.
7. cross-graze `Apply` — `molequla.go:4662-4668`.
8. untrained + `step < 10`: greedy `argmax(overlaidLogits)` excluding EOS, `continue` —
   `molequla.go:4680-4711`. Skips everything below.
9. untrained: hard top-15 mask excluding EOS, else plain `/temp` — `molequla.go:4728-4772`.
10. softmax, per-token entropy, dissonance temperature rescale that preserves the mask —
    `molequla.go:4773-4821`.
11. prob-space corpus blend — **skipped whenever the flag is on** — `molequla.go:4831`.
12. anti-field bypass with probability `CFG.AntiFieldProb` (0.05, `molequla.go:355`) —
    `molequla.go:4886`.
13. `TopKTopPSample`, then `MetaweightsOverlayCollapse` — `molequla.go:4890-4895`.
14. post-loop SPA reseed — `molequla.go:4948-5031`.

Inside `MetaweightsOverlay` (`metaweights_overlay.go:182-378`) the order is: magnitude gate
(204-217) → transformer gate `logits[i] *= tg` (227-237) → bigram/trigram (242-274) → Hebbian
(276-302) → prophecy seed/age/normalise (306-350) → the additive sum with the five coefficients
(361-366) → unigram damping (368-374).

### 1.3 The coefficient switch at `mag > 1.0` is unreachable from the generation path

`metaweights_overlay.go:214-217`:

```go
coeffs := metaCoeffsWeightless
if tmag > metaTFGateThreshold { coeffs = metaCoeffsTrained }
```

`metaTFGateThreshold = 1.0` (`metaweights_overlay.go:40`), `metaCoeffsTrained` at
`metaweights_overlay.go:32`. The caller only enters the function when `tmag <= 1.0`
(`molequla.go:4634` + `4646`), and it hands over `overlaidLogits`, a byte-for-byte copy of
`logits.Data` at that moment (`molequla.go:4613-4614`) — so the `tmag` recomputed at
`metaweights_overlay.go:205-213` is the same number the caller already tested. `metaCoeffsTrained`
therefore never binds in production. The only path that could reach it is a direct call to
`MetaweightsOverlay`; there is none besides `molequla.go:4647` (`grep -n "MetaweightsOverlay("` →
`metaweights_overlay.go:182`, `molequla.go:4647`).

For the same reason the transformer gate at `metaweights_overlay.go:227` (`if tmag <=
metaTFGateThreshold`) **always** fires when the overlay runs: `logits[i] *= tmag/1.0`.

### 1.4 Where the overlay's authority actually declines

Two mechanisms, one continuous and one a cliff:

* **Continuous, 0 → 1.0:** the transformer gate `tg = tmag/1.0` clamped to `[0,1]`
  (`metaweights_overlay.go:228-237`). At `tmag ≈ 0.25` (the post-seeding magnitude the code's own
  comment records, `metaweights_overlay.go:37-39`) the model contributes a quarter of its logit
  scale and the overlay's fixed weightless coefficients — `Heb 1.0, Pro 0.7, Ds 0.15, Bg 15.0,
  Tg 10.0` (`metaweights_overlay.go:31`) — dominate. As training lifts `tmag` toward 1.0 the model
  recovers full scale while the overlay's additive contribution stays constant, so the overlay's
  *relative* authority decays. That is the whole of the "innate coherence decays as the weights
  mature" behaviour.
* **Cliff at `tmag = 1.0`:** `molequla.go:4650` sets `overlayActive = false` and the overlay, the
  repetition penalty, the greedy bootstrap and the hard top-15 mask all disappear in the same step.
  The discontinuity is the full additive stack — up to `15·p_bigram + 10·p_trigram + 1.0·heb +
  0.7·pro + 0.15·ds` minus the damping term, i.e. tens of logits — vanishing between one token and
  the next. The code comments at `molequla.go:4635-4645` name the sigmoid-blend fix and mark it
  deferred.

**`C-OVL-02` (P1).** Mechanism: binary regime switch at the caller makes the documented dynamic
coefficient bundle dead code and turns a fade into a step discontinuity.
Minimal repair: call `MetaweightsOverlay` in both regimes and let its internal switch pick the
trained bundle, or replace `untrainedRegime` with the sigmoid blend already sketched at
`molequla.go:4645`. Do not do both at once without re-running the sweep.

### 1.5 Overlay + cross-graze + SPA all on together

**`C-OVL-01` (P0) — cross-graze is silently discarded when the overlay flag is on and the organism
is warmed.**

`molequla.go:4662-4668`:

```go
if model.crossField != nil {
    target := overlaidLogits
    if !overlayActive { target = logits.Data }
    model.crossField.Apply(target, CFG.CrossGrazeCoef, CFG.CrossGrazeTopN)
}
```

`overlayActive` was cleared at `molequla.go:4650` for a warmed organism, but `overlaidLogits` is
still the detached copy allocated at `molequla.go:4613`. Sampling reads **`overlaidLogits`**
(`molequla.go:4727` sizes `scaled` from it; `molequla.go:4769-4771` scales from it). So the boost
lands in `logits.Data`, which nothing downstream reads. Result:

| `--corpus-overlay` | regime | cross-graze in generation |
|---|---|---|
| off | any | works (`overlaidLogits` aliases `logits.Data`, `molequla.go:4605`) |
| on | `tmag ≤ 1.0` | works (writes into the copy that is sampled) |
| on | `tmag > 1.0` | **dropped** |

This matters exactly because the overlay is going on by default at deployment: every organism past
warmup loses the cross-organism channel in its own voice while `ComputeModelEntropy`
(`molequla.go:2629-2631`) keeps injecting graze into the entropy the overload gate reads. The
mirror at `molequla.go:2623-2628` was added precisely so the gate would feel the same stress as
generation; with the overlay on, generation no longer feels it.

Minimal repair (`molequla.go:4664`): delete the `if !overlayActive { target = logits.Data }` branch.
`overlaidLogits` is the correct target in all three rows of the table above.

**SPA.** `--spa-gate` sets `CFG.SPACoherenceGate` (`molequla.go:6272-6273`, default false at
`molequla.go:283`). It runs strictly after the token loop on the decoded string
(`molequla.go:4948-5031`), so it neither reads nor writes logits and does not double-count with the
overlay. Its recursion re-enters `generateResonantLocked` (`molequla.go:5016`) with the gate
temporarily false, which means the regenerated sentence is produced with the overlay and cross-graze
active again but with a fresh `OverlayScratch` and a fresh `prophecyField` — a second, independent
overlay pass over three seed tokens.

**Double counting that is real, in order of size:**

1. *Seeding vs overlay.* `SeedEmbeddingsFromMetaweights` writes bigram-weighted neighbour
   embeddings into `wte` (`metaweights_seeding.go:45-87`) and `unigram × wte` into `lm_head`
   (`metaweights_seeding.go:89-123`). The overlay then adds the same bigram and unigram statistics
   again at `metaweights_overlay.go:361-374`. This is deliberate (both are ported from postgpt) and
   the code acknowledges the consequence: seeding lifts `mag` to ~0.25, which is why the threshold
   was moved from 0.1 to 1.0 (`metaweights_overlay.go:35-40`).
2. *Cross-graze vs overlay.* `dnaRead` appends a sibling's fragment to the host's own corpus
   (`molequla.go:5922-5927`) **and** mirrors it to `../dna/seen/<e>/` (`molequla.go:5918-5920`)
   which is what cross-graze ingests (`cross_graze.go:82-152`). For whichever organism wins the
   scan race, the same sibling text enters the logits twice: once as corpus statistics through the
   overlay's bigram/trigram/Hebbian terms, once as a raw token-id boost of `coef/(1+rank)` with
   `coef = 2.0` (`molequla.go:273`). Every other organism gets only the second.
3. *Prob-blend.* Not double-counted — it is switched off entirely (`molequla.go:4831`).

**`C-OVL-05` (P1).** `molequla.go:4831` tests `CFG.CorpusLogitOverlay`, not `overlayActive`. For a
warmed organism with the flag on, the overlay does not run *and* the prob-space corpus blend is
skipped: both corpus paths are off, which is not what any of the three comments at
`molequla.go:4826-4829` describe. Flipping the default to on makes this the steady state of the
whole colony.
Minimal repair: change the condition to `overlayActive` and hoist that variable above the loop, or
gate on `CFG.CorpusLogitOverlay && untrainedRegime`.

**`C-OVL-06` (P1).** The overload gate reads `ComputeModelEntropy` (`molequla.go:2581-2665` →
`SyntropyTracker.Measure`, `molequla.go:5153-5156` → `isSustainedOverload`, `molequla.go:5320`),
which applies cross-graze (`molequla.go:2629-2631`) but never the overlay. With the overlay on by
default the gate measures a distribution generation never produces. The §9 result keys on the loss
path (`README.md:678`), so the entropy path is the exposed one.
Minimal repair: either apply the overlay in `ComputeModelEntropy` behind the same flag, or state in
the log that the entropy path is defined on the raw transformer and leave it.

### 1.6 Other overlay findings

**`C-OVL-03` (P1) — five configuration knobs that do nothing.** `CFG.MetaCBigram`, `MetaCTrigram`,
`MetaCHebbian`, `MetaCDestiny`, `MetaCProphecy` (`molequla.go:138-142`, defaulted
`molequla.go:287-291`) and `CFG.MetaLogitOverlayFloor` (`molequla.go:144`, `293`) are declared,
JSON-serialised, and never read: `grep -n "MetaCBigram\|MetaLogitOverlayFloor" *.go` returns only
the declaration and the default. The overlay uses the package-level constants at
`metaweights_overlay.go:31-32`. A `--config` file that tunes the overlay changes nothing, silently.
Minimal repair: build `MetaCoeffs` from `CFG` inside `MetaweightsOverlay`, or delete the six fields.

**`C-OVL-04` (P1) — the repetition penalty inverts on negative logits.**
`MetaweightsRepetitionPenalty` multiplies: `logits[t] *= 0.5` (`metaweights_overlay.go:425`) and
`logits[ids[ri+1]] *= 0.2` (`metaweights_overlay.go:435`). It runs immediately after the overlay
(`molequla.go:4648`), and the overlay guarantees negative values: `logits[i] -= 2.0` for every token
with unigram probability below 1e-6 (`metaweights_overlay.go:371-372`), plus the graded damping at
`373`. For any repeated token whose post-overlay logit is negative, `×0.5` moves it *toward zero*,
i.e. **up** the ranking — the penalty rewards repetition exactly where the overlay has already
declared the token unlikely. The shape is inherited from the reference (`reffs/q/postgpt_q.c:1399-1409`
multiplies `raw` the same way), so this is a faithful port of a hazard, not a transcription error.
Minimal repair: subtract instead of multiply (`logits[t] -= penalty`), or clamp to
`logits[t] = min(logits[t]*0.5, logits[t])`.

**`C-OVL-07` (P2) — seeding runs on one path only.** `SeedEmbeddingsFromMetaweights` is called at
`molequla.go:6951`, inside the `if err != nil || model == nil` branch that only executes when the
checkpoint failed to load (`molequla.go:6911-6920`). A normal restart from
`molequla_ckpt.json` never seeds. That is correct — the seeding is baked into the saved `wte` — but
`README.md:391` ("Runs once at init when `--corpus-overlay` is on") does not say so, and a reader
turning the flag on for an existing colony will expect a change that cannot happen.

**`C-OVL-08` (P2) — seed scale documented twice, differently.** `metaweights_seeding.go:27` says the
default is 0.1; the only call site passes 0.15 (`molequla.go:6951`), the comment above it says
"scale=0.15 verbatim from postgpt.c:542" (`molequla.go:6948`), and `README.md:391` says 0.15. Fix
the docstring.

**`C-OVL-09` (P2) — `--zero-warmup` is four behaviours under one flag.** `molequla.go:6280-6284`
sets `CFG.WarmupSteps = 0`, which then (a) skips the checkpoint load (`molequla.go:6911-6915`),
(b) skips `SaveCheckpoint` and the `lastWarmupStage` marker (`molequla.go:6990-6997`), (c) breaks
out of the ontogenesis loop after embryo (`molequla.go:7013-7015`), (d) returns before the REPL and
the ecology (`molequla.go:7052-7055`). Every one of those is commented and deliberate; none is
discoverable from the flag name or from `README.md:408`.

---

## 2. MYCELIUM as it exists

### 2.1 What it reads and writes

**Reads**, `mesh.db` table `organisms`, two SELECTs:

* `ariannamethod/method.py:271-277` — `id, pid, stage, n_params, syntropy, entropy,
  gamma_direction, gamma_magnitude, last_heartbeat` `WHERE status='alive' AND last_heartbeat >
  now-120`. Nine columns; `Organism.__init__` (`method.py:194-204`) reads `element` at index 9, so
  `o.element` is always `None` on this path.
* `mycelium.py:1221-1227` — the same plus `element` (ten columns). **Never executed**: it lives in
  `async_read_field`, and `grep -n "async_read_field" mycelium.py` finds only the definition at
  1214. `async_step` calls the sync `self.method.read_field()` at `mycelium.py:1088`.

**Writes**, one row, `field_steering` id=1:

* sync: `method.py:441-465` (`INSERT OR REPLACE`, ten columns, `updated_at = time.time()`).
* async: `mycelium.py:1184-1212` via `aiosqlite`, identical statement, falling back to the sync one
  on any exception (`mycelium.py:1210-1212`).

Tables created by `method.py:229-263`: `field_steering` and `field_deltas`. **`field_deltas` is
written by nothing** — `write_deltas` (`method.py:420-439`) has no caller anywhere in the repo, and
neither do `apply_to_logits` (`method.py:494-501`) or `notorch_update` (`method.py:503-523`).

**Cadence.** `--interval` default 1.0 s (`mycelium.py:1626-1627`). Daemon loop: `async_step` then
`asyncio.sleep(interval)` (`mycelium.py:1501-1509`). REPL runs the same stepper on a background
thread with its own event loop (`mycelium.py:1516-1550`). `--once` does a single sync `step()` and
prints JSON (`mycelium.py:1643-1646`). Sentinel scans every 5 steps (`mycelium.py:959`,
`1051-1055`); drift check every 4 steps (`mycelium.py:1045-1047`).

Per tick the field is read **twice** — `mycelium.py:964` (`read_field`) and again inside
`method.step()` at `method.py:475` — and each `read_field` opens and closes its own
`sqlite3.connect` (`method.py:269-280`), as does `write_steering` (`method.py:444-463`). Three
connection open/close cycles per second per mycelium process.

### 2.2 `C-MYC-01` (P0) — the mycelium reads columns the Go colony never creates

Go's `organisms` table (`molequla.go:5490-5497`):

```
id, pid, stage, n_params, syntropy, entropy, last_heartbeat, parent_id, status, element
```

No `gamma_direction`, no `gamma_magnitude`. Only `molequla.rs:2599` declares them (plus
`rrpram_signature`). `method.py:271-277` selects them; on a Go-created `mesh.db` the statement
raises `no such column: gamma_direction`, which is swallowed by the bare `except Exception: pass` at
`method.py:281-282`, leaving `self.organisms == []`. From there `am_method_step` returns
`AM_METHOD_WAIT` with every field zero (`ariannamethod.c:7898-7901`), `FieldMonitor` reports
"no organisms alive" (`mycelium.py:72-74`), and the `field_steering` row written is a constant.

And even the Rust core never fills those columns: its only `heartbeat` call passes `None` for the
gamma blob, and `0` for stage, n_params and entropy (`molequla.rs:3016`, signature at
`molequla.rs:2612-2613`). So `gamma_direction` is NULL in every real run of every core.

The test suite hides this. `tests/test_all.sh:291-310` builds its own `mesh.db` with the Rust-shaped
schema (`gamma_direction BLOB, gamma_magnitude REAL, rrpram_signature BLOB`, no `element`) and fills
`gamma_direction` with `np.random.randn(32).astype(np.float64).tobytes()` — exactly the 256 bytes
`HarmonicNet` wants (`mycelium.py:611`, `dim*8 = 256`). The mycelium block of the integration suite
is green against a shape the ecology has never produced.

Consequences that follow automatically: `Method.field_coherence()` returns 1.0 for n<2 and for
`gamma_mag <= 1e-6` (`ariannamethod.c:7874-7887`); `HarmonicNet` layer 2 correlates all-zero gammas
(`mycelium.py:618-619` / `720-724`), so `resonance` is all zeros and `mean_res < 0.3` always fires
the `explore` bias (`mycelium.py:752-754`); `DriftTracker` compares entropies only
(`method.py:374-384`), which does work.

Minimal repair for the Go witness: read only what Go writes —
`id, pid, stage, n_params, syntropy, entropy, last_heartbeat, parent_id, status, element` — treat
gamma as unavailable, and never swallow a schema error. If gamma is wanted later, it has to be
written first: add the columns to `molequla.go:5490` and fill them from
`GammaContrastiveProjection()` in `SwarmRegistry.Heartbeat` (`molequla.go:5588-5595`).

### 2.3 `C-MYC-02` (P1) — the intent/outcome loop compares a snapshot with itself

`Mycelium.step()` (`mycelium.py:961-1079`):

* `pre_h/pre_s/pre_c` from `read_field()` + `field_entropy()` — `mycelium.py:964-967`.
* `steering = self.method.step(...)` — `mycelium.py:971` — which re-reads the same rows
  (`method.py:475`) microseconds later and returns `s.entropy` computed from them
  (`ariannamethod.c:7903-7909`).
* `post_h = steering.get("entropy", pre_h)` — `mycelium.py:1000-1002`.
* `dis = self.dissonance.update(action, post_h - pre_h, post_c - pre_c)` — `mycelium.py:1004`.

Nothing happened to the organisms between the two reads — the steering row is not even written until
`mycelium.py:1009`. So `post_h - pre_h ≈ 0` by construction. Everything downstream of that delta is
structurally null: `SteeringDissonance.dissonance` stays near zero and
`strength_multiplier()` returns ~0.5 forever (`mycelium.py:365-366`); `OrganismAttention` decays
every organism by 0.99 every tick and boosts none (`mycelium.py:406-412`);
`MyceliumSyntropy.record_decision` gets `delta_h = delta_s = delta_c ≈ 0` because `snapshot_before`
is called with this tick's pre-values (`mycelium.py:968`, consumed at `mycelium.py:812-823`), so
`effectiveness` ≈ 0, `purpose_magnitude` ≈ 0 and `purpose_alignment` ≈ 1.0
(`mycelium.py:866-888`). `async_step` has the identical structure (`mycelium.py:1088-1092`,
`1118-1122`).

What *is* real in `MyceliumSyntropy`: `entropy_history` accumulates one true field entropy per tick,
so `syntropy_trend` (`mycelium.py:847-853`) and `decision_entropy` (`mycelium.py:856-864`) are
meaningful.

Minimal repair: keep the previous tick's post-state and compare across ticks, i.e. move
`snapshot_before` to the end of the tick, or drop the three classes that depend on the intra-tick
delta.

### 2.4 Logic vs bookkeeping, class by class

| class | verdict |
|---|---|
| `FieldMonitor` (`mycelium.py:55-106`) | bookkeeping + 4 threshold alerts; the dampen/realign loop detectors (83-88) are real logic |
| `DriftTracker` (`113-129`) | thin wrapper over `Method.field_drift` (`method.py:374-384`); entropy-only, works |
| `MyceliumVoice` (`136-231`) | presentation only |
| `FieldPulse` (`240-297`) | real: novelty from set difference (264-271), arousal from cross-tick entropy delta (274-278 — this one *does* span ticks and is valid), Shannon over organism entropies (281-290) |
| `SteeringDissonance` (`300-370`) | logic, fed a null signal — see `C-MYC-02` |
| `OrganismAttention` (`373-432`) | logic, fed a null signal — see `C-MYC-02` |
| `MyceliumGamma` (`438-536`) | real computation, zero consumers: `as_blob()` (521) has no caller, the vector never leaves the process |
| `HarmonicNet` (`543-768`) | the substantive class; two implementations of the same three layers, C at `_forward_c` (690-768), Python at `forward` (589-687) |
| `MyceliumSyntropy` (`775-921`) | trend + decision entropy real; effectiveness/purpose null |
| `Mycelium` (`928-1618`) | orchestration + 14 REPL commands |
| `Sentinel` (`ariannamethod/sentinel.py:81-356`) | real: mtime+size diffing over `dna/`, FNV-1a fallback hash (159-177), AML reactions (276-307) |

### 2.5 What `am_harmonic_*` and `am_method_*` compute

Declarations `ariannamethod/ariannamethod.h:942-946` and `1005-1013`; structs at `934-940`,
`961-989`, `992-1002`. Implementation `ariannamethod/ariannamethod.c:7708-8000`.

**HarmonicNet**, state in the file-static `HN` (`ariannamethod.c:7708-7718`): 64-slot circular
entropy history, 64×32 float gamma matrix, per-organism entropy.

* `am_harmonic_init` (7720) zeroes everything; `am_harmonic_clear` (7724) only resets
  `n_organisms` — the entropy history survives, which is the intended cadence.
* `am_harmonic_push_entropy` (7728) advances the circular buffer.
* `am_harmonic_push_gamma` (7735) copies `min(dim, 32)` floats, zero-pads, **ignores `id`**
  (7736) — organism identity is positional only.
* `am_harmonic_forward` (7747): layer 1 is a sine-only DFT over the history,
  `r.harmonics[k] = (1/T)·Σ_t h[t]·sin(2π(k+1)t/T)`, for k<8, only when T≥4 (7759-7769); layer 2
  computes L2 norms (7776-7782), the mean organism entropy as phase reference (7785-7787), and
  `resonance[i] = (1/(n-1))·Σ_{j≠i} cos(γ_i,γ_j)·exp(-|phase_i-phase_j|)` (7789-7808); layer 3 picks
  the dominant harmonic by absolute amplitude (7812-7819) and sets
  `strength_mod = 0.3 + 0.7·min(1,T/16)·min(1,n/4)` (7822-7824).

`mycelium.py:589-687` is a line-for-line Python twin of the same three layers, with identical
confidence formula at 678-679. The action-bias mapping that turns this into a decision lives in
Python on **both** paths (`mycelium.py:647-676` and `739-760`) — the C returns numbers only.

**METHOD**, state in the file-static `M` (`ariannamethod.c:7837`):

* `am_method_push_organism` (7847) stores `{id, entropy, syntropy, gamma_mag, gamma_cos}`; the host
  computes `gamma_cos` against the field mean, C does not (see the comment at 7878).
* `am_method_field_entropy` / `_syntropy` (7858, 7866) are plain means; `_field_coherence` (7874)
  is the mean of host-supplied `gamma_cos` over organisms with `gamma_mag > 1e-6`, returning 1.0
  when there are none — which is the value every real run gets, per `C-MYC-01`.
* `am_method_step` (7890): pushes entropy and coherence into a 16-slot circular history
  (7911-7917), computes `trend = mean(older 4) - mean(recent 4)` once history ≥ 4 (7919-7932),
  picks the lowest-entropy organism as `target_id` (7934-7943), then a 6-way decision ladder
  (7946-7964): `coherence < 0.3 → REALIGN`, `trend > 0.05 → AMPLIFY`, `trend < -0.05 → DAMPEN`,
  `entropy > 2.0 → GROUND`, `entropy < 0.5 → EXPLORE`, else `SUSTAIN(0.1)`.
* **It is not a read-only call.** `am_method_step` advances AML field physics with `am_step(dt)`
  (7967) and executes AML statements per action — `PAIN`, `VELOCITY`, `DESTINY`, `ATTEND_FOCUS`,
  `TUNNEL_CHANCE` (7970-7993).

### 2.6 Dependencies beyond the standard library

* `mycelium.py:25-37` — stdlib only: `argparse asyncio json math os random signal sqlite3 struct
  sys threading time pathlib`.
* `mycelium.py:42-45` — `aiosqlite`, guarded, `None` on ImportError, every use falls back
  (`mycelium.py:1186-1189`, `1216-1218`).
* `mycelium.py:47-48` — `from ariannamethod import Method, Sentinel`, and
  **`ariannamethod/method.py:25` is an unguarded top-level `import numpy as np`**. `requirements.txt`
  contains exactly `numpy`. So `mycelium.py` has a hard numpy dependency through its import chain.
* `ariannamethod/sentinel.py:18-22` — stdlib only (`ctypes hashlib os time pathlib`).
* `libaml.so` / `libaml.dylib`, looked up next to `method.py` (`method.py:29-36`). **Not present in
  `ariannamethod/`.** Build recipe: `ariannamethod/Makefile` (`SRCS = ariannamethod.c notorch.c`,
  `-shared`, OpenBLAS via pkg-config).
* `tests/test_all.sh` additionally needs `python3` + `numpy` for its mesh fixtures
  (`tests/test_all.sh:239, 264, 292, 351`).

**`C-MYC-03` (P1) — without `libaml`, the "Python fallback" is a constant.** `_load_libaml` returns
`None` when the shared object is missing (`method.py:88-90`), and `Method.compute_steering`'s
fallback returns a hardcoded `{"action": "sustain", "strength": 0.1}` with no trend, no step
counter, no decision ladder (`method.py:405-418`). `Method.step` routes to it at `method.py:492`.
The C is not an accelerator; it is the entire decision logic.

### 2.7 The minimal contract a Go witness needs

Everything below is verified against what the cores actually write, not what the mycelium hopes to
read.

**Inputs — read-only, both already public traces:**

1. `mesh.db`, table `organisms`, columns that a Go core actually populates:
   `id TEXT, pid INTEGER, stage INTEGER, n_params INTEGER, syntropy REAL, entropy REAL,
   last_heartbeat REAL, parent_id TEXT, status TEXT, element TEXT`
   (`molequla.go:5490-5497`; written by `registerInMesh` 5560-5569, `ReserveChildSlot` 5577-5585,
   `Heartbeat` 5588-5595, `MarkHibernating` 5629-5633). Freshness: heartbeat every 10 ticks of
   `CFG.TrainTickSeconds = 0.25` (`molequla.go:6677`, `molequla.go:325`) ≈ 2.5 s; the mycelium's
   existing liveness window is 120 s (`method.py:277`) and `DiscoverPeers` uses 60 s
   (`molequla.go:5605`). Pick one and write it down.
   `syntropy` is `SyntropyTracker.SyntropyTrend`, `entropy` is the last `ComputeModelEntropy` sample
   (`molequla.go:6685-6690`) — i.e. corpus-sample entropy with cross-graze applied, not generation
   entropy. Note that the Rust core writes zeros into `stage`, `n_params` and `entropy`
   (`molequla.rs:3016`); a witness that mixes cores must tolerate that.
2. `../dna/seen/<element>/gen_*.txt` — the non-destructive mirror (`molequla.go:5918-5920`). This is
   the only complete record of what each organism said: `dna/output/` entries are deleted on
   consumption (`molequla.go:5932`). Filename convention `gen_<unix>_<step>.txt`
   (`molequla.go:5876`); `cross_graze.go:107` already relies on the `gen_` prefix and `.txt` suffix.
3. Optional: `memory.sqlite3` per organism — `growth` and `syntropy_log`
   (`molequla.go:3503`, `3521`) for developmental history.

**Compute:** call the same C through cgo. `cgo_aml.go:10-11` already does `#include
"ariannamethod.h"` and `#include "ariannamethod.c"`, so the molequla Go binary **already links**
`am_method_*` and `am_harmonic_*`; `grep -rn "am_method\|am_harmonic" *.go` returns nothing, i.e.
there are no Go callers today and no new build wiring is required. Two constraints:

* `HN` (`ariannamethod.c:7708`) and `M` (`7837`) are file-static process globals. Serialise every
  `clear/push/step/forward` sequence behind one mutex; they are not reentrant and not per-instance.
* `am_method_step` executes AML statements and advances field physics (`ariannamethod.c:7967-7993`).
  A witness that must not steer should use `am_method_field_entropy/_syntropy/_coherence` (7858,
  7866, 7874) plus `am_harmonic_forward` (7747) and implement the decision ladder in Go, or accept
  that the AML side-effect is internal to the witness process and never reaches an organism.
* Gamma cosines are host-computed (`method.py:286-329`). Without `gamma_direction` in the schema the
  correlation layer has no input; do not port that half until the column is written.

**Outputs — outward only, no arrow back:** stdout / a log / the mesh transport. Do **not** write
`field_steering` (see §3), do **not** write into `dna/output/` (the four cores consume and delete
from there, `molequla.go:5889-5934`), do **not** delete from `dna/seen/`.

**Cadence to match:** field read ≈ 1 s, sentinel scan ≈ every 5 reads, drift ≈ every 4 reads
(`mycelium.py:959, 1045, 1051`). One SQLite handle held open in WAL mode instead of three
connect/close cycles per tick.

**Dead surface not worth porting:** `field_deltas` and `Method.write_deltas/apply_to_logits/
notorch_update` (`method.py:249-259, 420-523`, no callers); `MyceliumGamma.as_blob`
(`mycelium.py:521`, no callers); `async_read_field` (`mycelium.py:1214`, no callers);
`SteeringDissonance` / `OrganismAttention` until `C-MYC-02` is fixed.

---

## 3. `field_steering` — proof of the single reader

**Exhaustive grep**, whole tree minus `.git/` and `reffs/`:

```
README.md:627          prose
README.md:806          prose
PROJECT_LOG.md:2617    prose
MOLEQULALOG2.md:110    prose
MOLEQULALOG2.md:157    prose
mycelium.py:1194       INSERT OR REPLACE  (async writer)
ariannamethod/method.py:235   CREATE TABLE IF NOT EXISTS
ariannamethod/method.py:447   INSERT OR REPLACE  (sync writer)
molequla.rs:3265       comment
molequla.rs:3267       SELECT action, strength, entropy, coherence ... WHERE id=1 AND updated_at > ?1
```

Two writers, both Python. **One reader, `molequla.rs`.** Corroborating counts:
`grep -c steering molequla.{go,c,js,rs}` → `0, 0, 0, 5`; `grep -n "field_steering\|field_deltas"
molequla.c molequla.js molequla.go` → empty; `grep -n mycelium molequla.go molequla.c molequla.js` →
empty.

**What Rust does with it** (`molequla.rs:3265-3285`): a 120-second freshness cutoff (3269-3270),
then four fields copied into the model under its mutex — `field_action`, `field_strength`,
`field_entropy`, `field_coherence` (3276-3282; declared `molequla.rs:1044-1047`, initialised to
`"wait"/0.0/0.0/1.0` at `1097-1098`). The only consumer is a temperature multiplier during sampling
(`molequla.rs:1889-1911`):

```
dampen  → 1.0 - 0.3·strength      amplify → 1.0 + 0.2·strength
ground  → 1.0 - 0.5·strength      explore → 1.0 + 0.4·strength
realign → 1.0 - 0.2·strength      sustain / wait → 1.0
```

applied as `temp = base_temp · final_mul · field_mul` with a resample (`molequla.rs:1913-1916`).
`field_entropy` and `field_coherence` are stored and never read again. There is no action
modulation beyond temperature — no training, no growth, no mitosis path consumes the steering row.

**Does removing the table break anything?**

* Go: no references at all. No.
* C: no references. No.
* JS: no references. No.
* Rust: `db.prepare(...)` is inside `if let Ok(mut stmt)` (`molequla.rs:3266`). A missing table makes
  `prepare` return `Err`, the block is skipped, `field_action` stays `"wait"`, `field_mul` stays 1.0
  (`molequla.rs:1910`) and no resample happens (`molequla.rs:1913`). Silent, correct degradation.
* Python: `method.py:_init_db` recreates it on construction (`method.py:229-247`), wrapped in
  `try/except: pass` (262-263).

So `field_steering` can be dropped with zero effect on Go, C and JS, and with Rust falling back to
its own temperature schedule. It is exactly the "arrow back" the witness design removes, and it
costs one `if let Ok` branch in one core to remove it.

---

## 4. A fifth DNA source — every place that enumerates the four elements

### 4.1 Complete enumeration sites

| # | site | shape | breaks on a 5th source? |
|---|---|---|---|
| 1 | `molequla.go:5834` `var dnaElements = []string{"earth","air","water","fire"}` | slice, length implicit | the list to extend; see 4.2 |
| 2 | `cross_graze.go:60` `all := []string{"earth","air","water","fire"}` | **second copy** | yes — world never grazed unless also edited |
| 3 | `molequla.c:4958` `dna_elements[]` + `:4959` `dna_n_elements = 4` | array **and a separate count** | yes — the classic duplicated invariant (`CLAUDE.md:68-70`) |
| 4 | `molequla.go:6832-6845` element→`nonames_<e>.txt` switch, `default: os.Exit(1)` | hard exit | only if `--element world` is ever passed |
| 5 | `molequla.c:5317-5321` same, `exit(1)` | hard exit | same |
| 6 | `molequla.rs:3323-3328` same, `std::process::exit(1)` | hard exit | same |
| 7 | `molequla.js:3816-3822` `elementCorpus` map + unknown-element log | soft | no |
| 8 | `ariannamethod/sentinel.py:67` `if elem in ("earth","air","water","fire")` | label filter | soft — world files get `element=None` |
| 9 | `launcher.sh:13-20` mkdir list, `:25` corpus copy loop, `:41-48` four launches | shell | needs `dna/output/world` created |
| 10 | `pod_watchdog.sh:23` `ORGANISMS=(earth air water fire)` | shell | monitoring only |
| 11 | `sweep.sh:18, 33` `for e in earth air water fire` | shell | harness only |
| 12 | `runpod/2026-06-03_criterion9/criterion9_run.sh:46, 51` | shell | archive, do not touch (`CLAUDE.md:100-101`) |
| 13 | `molequla_test.go:619` `for _, elem := range []string{...}` in `TestDnaReadWriteFilesystem` | test fixture | builds 4 dirs; passes unchanged |
| 14 | `governor_test.go:140` `ReserveChildSlot(..., "earth")` | single literal | no |
| 15 | `tests/test_all.sh:294-309` mesh fixture | no element column at all | no |

There is **no element enum in `mesh.db`**: the column is free `element TEXT` with no CHECK, in both
schemas (`molequla.go:5495`, `molequla.rs:2599`). Corpus lookup is by filename convention
`nonames_<e>.txt` only; a `world` source needs no corpus file because nothing would launch an
organism for it. No fixed-size array is indexed by element count anywhere in Go or Rust; the only
count constant is C's `dna_n_elements` (row 3).

`molequla.js` has no DNA exchange at all — `grep -ci dna molequla.js` → 0. The JS core is outside
this loop entirely.

### 4.2 The two hidden assumptions that will actually bite

**`C-DNA-01` (P1) — `dnaRead` is destructive, so a shared source feeds exactly one organism.**
`molequla.go:5889-5935`: for each non-self element it reads every `*.txt`, mirrors the bytes to
`../dna/seen/<e>/` (5918-5920), appends the text to its own corpus (5922-5926), feeds the quantum
buffer (5929-5931), and then `os.Remove(fpath)` (5932). Whichever organism scans first takes the
file; the rest never see it in their corpus. This is already measured on this node —
`MOLEQULALOG2.md:148-157`: air 3 118 052 bytes in 18 reads against earth 35 734 in 5, and earth
stayed a child. Adding `"world"` to `dnaElements` inherits that race: each vision observation would
reach exactly one of the four, at random, which is the opposite of "the four elements eat it like a
sibling".
Minimal repair for the world source: do not route it through `dnaRead`'s consume-and-delete path.
Either give each organism a per-element cursor file under its own working directory and never delete
from `dna/output/world/`, or have the writer fan out one copy per element
(`dna/output/world/<element>/gen_*.txt`) so each organism deletes only its own copy. The second is
smaller and keeps `dnaRead` untouched.

**`C-DNA-02` (P1) — the element list is duplicated three times with three shapes.** `molequla.go:5834`
(slice), `cross_graze.go:60` (slice, independent), `molequla.c:4958-4959` (array + count). If the
world source is added to the first and not the second, cross-graze never boosts world tokens and the
new source is invisible to the logit path; if added to the C array without bumping
`dna_n_elements`, the C core silently ignores it; if the count is bumped without the array, the C
core reads past the end.
Minimal repair: one exported list in Go, consumed by both `dnaRead` and `NewCrossField`; in C, drop
`dna_n_elements` and derive it as `sizeof(dna_elements)/sizeof(dna_elements[0])`.

**`C-DNA-03` (P2) — filename convention.** `cross_graze.go:107` ingests only
`strings.HasPrefix(n,"gen_") && strings.HasSuffix(n,".txt")`. World observations must be written as
`gen_<unix>_<n>.txt` or cross-graze will skip them while `dnaRead` (which accepts any `*.txt`,
`molequla.go:5899`) eats them — an asymmetry that would be very hard to see from the outside.

**`C-DNA-04` (P2) — sentinel labelling.** `ariannamethod/sentinel.py:61-69` returns `None` for any
directory under `output/` that is not one of the four, so world file events would be reported
unlabelled (`sentinel.py:347`). Harmless; one tuple to extend if the Python sentinel survives the
rewrite.

**`C-DNA-05` (P2) — `DNAMinFragmentBytes = 5` and the delete-on-short path.** `molequla.go:5908-5910`
deletes any file shorter than the threshold **before** mirroring it to `seen/`. A one-sentence
world observation is comfortably above 5 bytes, but note the asymmetry: short files are destroyed
without ever reaching the mirror, so a witness reading only `dna/seen/` would never know they
existed.

---

## 5. README vs code — every contradiction found

Read in full, 872 lines. Ordered by severity.

### P1

**`C-RDM-01` — the documented launch command passes three flags the binary does not know.**
`README.md:734-739` (and the same in `sweep.sh:25`) passes `--corpus`, `--db`, `--ckpt`.
`parseCLIArgs` (`molequla.go:6259-6301`) handles `--organism-id --config --element --evolution
--spa-gate --corpus-overlay --trainer --zero-warmup --gpu --cross-graze` and nothing else; unknown
arguments fall through the `if/else if` chain with no error. Every run started from the README's
Quick Start silently uses `CFG.CorpusPath` / `CFG.DBPath` / `CFG.CkptPath` defaults (`molequla.go:249-250`),
saved by the `--element` switch (`molequla.go:6831-6845`) which does set the corpus path.
*Already recorded at `MOLEQULALOG2.md:106-109`; repeated here because it is also in `sweep.sh`,
which the log does not mention.*
Repair: delete the three flags from `README.md:735-737` and `sweep.sh:25`, or implement them.

**`C-RDM-02` — the sampling pipeline is documented backwards.** `README.md:398`: "**Step ≥ 10 or
trained regime** — top-15 raw-logit mask". Code: the hard mask runs only when `overlayActive &&
untrainedRegime` (`molequla.go:4728`), and the comment at `molequla.go:4718-4726` states the trained
regime deliberately skips it because the mask on a BPE subword vocab produced
"«,iieriying the isa?yenanan?»". The trained regime uses the ordinary soft `TopKTopPSample`
(`molequla.go:4769-4771`, `4890`).
Repair: `README.md:398` → "Step ≥ 10, untrained regime only".

**`C-RDM-03` — "Composes with Q-style overlay regardless of regime" is false.** `README.md:545`,
contradicted by `C-OVL-01` above (`molequla.go:4650` + `4664` + `4727`). The same sentence appears
as a code comment at `molequla.go:4659-4660`.
Repair: fix the code (one line, §1.5) rather than the sentence.

**`C-RDM-04` — "numpy-free, pure stdlib" is false.** `README.md:806`. `mycelium.py:47` imports
`ariannamethod.Method`; `ariannamethod/method.py:25` is a top-level unguarded `import numpy as np`;
`requirements.txt` is the single line `numpy`. `mycelium.py` itself is stdlib-only
(`mycelium.py:25-45`), which is presumably what the line meant.
Repair: "`mycelium.py` is stdlib-only; its `ariannamethod.Method` dependency pulls numpy".

**`C-RDM-05` — test counts, three different numbers, none of them right.**
`README.md:758` and `:823` say 132, itemised as `molequla_test.go 122 + molequla_rrpram_test.go 4 +
governor_test.go 5 + mitosis_cooldown_test.go 1`. `governor_test.go` has **6** `func Test`
(lines 14, 49, 64, 82, 99, 120), so package `main` has **133**. Package `tests` adds **8**
(`tests/molequla_test.go`), so `go test ./...` covers **141**, which is what the phone counted
today and what `MOLEQULALOG2.md:86-88` records. `PROJECT_LOG.md:2677` says 140; `CLAUDE.md:55` says
140.
Repair: `README.md:758` → "133 in package main + 8 in ./tests = 141"; `README.md:822` → "governor_test.go
149 lines … (6 tests)"; `README.md:823` → "(1 test; 133 in package main)". `PROJECT_LOG.md:2677`
is history and stays; `CLAUDE.md:55` should be refreshed when the count is next measured.

**`C-RDM-06` — the SPA measurement harness cannot measure SPA.** `README.md:349`: "What the
measurement run captures is the signal: how often the gate fires and reseeds". `sweep.sh:36` counts
`grep -c "\[spa-gate\]"`. The code emits `[spa]` — `molequla.go:4999` and `:5023`; `grep -rn
"\[spa-gate\]"` over all `.go/.c/.rs/.js` returns nothing. The `spa-gate=` column of every sweep cell
is 0 by construction, including `cell_3_full_coherence` (`sweep.sh:44`).
Repair: `sweep.sh:36` → `grep -c "^\[spa\]"`.

### P2

**`C-RDM-07` — `GenerateSentence` is documented as a live path and has no caller.**
`README.md:473` describes `gpuRefreshWeights` being called "symmetrically at the top of
`GenerateSentence` (`molequla.go:3055`)" — the line reference is exact (`molequla.go:3054-3055`) but
`grep -rn "GenerateSentence" --include=*.go` finds only the definition at `molequla.go:3043`. The
chat path is `GenerateResonant` (`molequla.go:7167`), the DNA path is `dnaWrite`
(`molequla.go:5849`), the warmup probes are `molequla.go:7003` and `7040`. ~330 lines of duplicated
generation loop (`molequla.go:3043-3380`) that nothing executes, including its own corpus blend and
`lastGenEntropy` write (`molequla.go:3372`).

**`C-RDM-08` — "Go-native training" runs through an unreachable function.** `README.md:181`
("`AdamStep()` updates parameters … Handles inference, loss computation, Go-native training"). The
diagonal per-parameter step at `molequla.go:2778` is called only from `trainSteps`
(`molequla.go:6399`, `6402`), and `grep -n "trainSteps(" *.go *_test.go tests/*.go` finds only the
definition at `molequla.go:6317`. Warmup goes through `ntWarmupTrain`
(`notorch_trainer.go:430`, called `molequla.go:6494-6496` and `6985-6987`); the trainer default is
`"notorch"` (`molequla.go:286`). The pre-Chuck optimizer survives exclusively inside dead code —
worth saying in the log, since the README currently presents it as the Go training path.
Repair: either delete `trainSteps` + `AdamStep` + `ensureAdam`, or mark them retired in
`README.md:181`.

**`C-RDM-09` — line numbers and line counts have drifted.** `molequla.go` grew 66 lines past the
README's snapshot and the references above ~6600 all shifted by that amount:

| README | claim | actual |
|---|---|---|
| 203 | `molequla.go` 218K, 7,146 lines | 225 526 B (220K), **7212** |
| 205 | `molequla.rs` 148K | 147 527 B (**144K**), 3544 ✓ |
| 206 | `molequla.js` 152K | 153 924 B (**150K**), 3971 ✓ |
| 473 | `GenerateResonant` ... `molequla.go:4486` | `gpuRefreshWeights` at **4492** |
| 524 | `CrossField` built at `molequla.go:7001` | **7066-7067** |
| 544 | `MaybeRefresh` at `molequla.go:4493` | **4498-4499** |
| 490 | GPU init at `molequla.go:6745` | **6811-6812** |
| 498 | `effectiveCPUs` 6680 / `colonyThreadsFor` 6711 / `capColonyThreads` 6722 / `main()` 6735 | **6746 / 6777 / 6788 / 6800-6801** |
| 822 | `governor_test.go` 114 lines | **149** |
| 779 | `cgo_notorch.go` 186 lines | **187** |
| 780 | `cgo_notorch_cpu.go` 13 lines | **14** |

Correct as read: `molequla.go:840, 908, 966, 980, 999` (`README.md:458, 461, 475`),
`molequla.go:273` (`README.md:540`), `molequla.go:3055` (`README.md:473`), `molequla.c` 5583,
`mycelium.py` 1660, `ariannamethod.c` 8000, `method.py` 527, `sentinel.py` 356,
`metaweights_overlay.go` 439, `metaweights_seeding.go` 124, `spa_coherence.go` 164,
`cross_graze.go` 216, `tests/test_all.sh` 711, `tests/molequla_test.go` 262,
`molequla_test.go` 2623, `PROJECT_LOG.md` 2682 ("≈2600").

**`C-RDM-10` — "Runs on CPU" carries no build caveat for non-x86.** `README.md:44` and the build
recipe at `README.md:706` (`CGO_ENABLED=1 go build -a -o molequla_cgo .`). `cgo_aml.go:8-9` hardcodes
`-I/usr/include/x86_64-linux-gnu/openblas-pthread/` and
`-L/usr/lib/x86_64-linux-gnu/openblas-pthread/`, paths that do not exist on aarch64; the build
actually recorded on this node overrides them
(`MOLEQULALOG2.md:76-78`: `CGO_CFLAGS="-O3 -march=native -mtune=native -DUSE_BLAS"
CGO_LDFLAGS="-lopenblas -lm -lpthread" ... -buildvcs=false`). I did not build, so I do not claim the
unqualified command fails — only that the paths it relies on are x86-specific and that the working
recipe on ARM is different and undocumented in the README.
Repair: add the ARM recipe next to `README.md:706`, or make `cgo_aml.go:8-9` pkg-config-driven the
way `ariannamethod/Makefile:19-21` already is.

**`C-RDM-11` — mycelium claims.** `README.md:627` "writes a `field_steering` row to `mesh.db`; the
organisms read it back" — one organism, the Rust one (§3); the Go colony of §9 never heard it, as
`README.md:41` itself half-admits ("post-§9 layer"). `README.md:641` "The mycelium reads mesh.db to
see the entire ecology and makes decisions that individual organisms cannot: when to spawn, when to
hibernate" — nothing in the mycelium writes a spawn or hibernate decision anywhere; `field_steering`
carries `action/strength/target_id/entropy/syntropy/coherence/trend/n_organisms`
(`method.py:447-450`) and the only consumer multiplies a temperature (`molequla.rs:1889-1911`).
Spawn and hibernate are decided inside each organism (`molequla.go:5261`, `5822`).
`README.md:633` "HarmonicNet … Output: action biases" — the C returns harmonics, resonance and a
strength modulation only; the action-bias mapping is Python (`mycelium.py:647-676`, `739-760`).
Repair: `README.md:641` should say the mycelium observes and publishes one advisory row; spawn and
hibernate are organism-local.

**`C-RDM-12` — mitosis claims.** `README.md:670-678` is accurate against the code —
`performMitosis` writes the parent checkpoint into the child directory, writes `birth.json`, spawns
with `--gpu`/`--cross-graze`/`--element` inherited (`molequla.go:5774-5781`), reserves the slot
(`molequla.go:5813`), and the 300 s cooldown is seeded at birth (`molequla.go:5080-5085`). Two gaps:
`README.md:674` lists the inherited flags but not `--corpus-overlay`, and
`molequla.go:5774-5781` does not pass it either — so **a child of an overlay-running parent is born
without the overlay**. Once the flag is on by default (in the config, not on the command line) this
resolves itself; if it is turned on with the CLI flag it does not. Worth deciding deliberately
before deployment.
Repair: if the overlay is switched on via `CFG` default, nothing to do; if via the flag, add it to
the child argv at `molequla.go:5780`.

**`C-RDM-13` — `AdamStep` / "Adam state resets".** `README.md:181` and `README.md:239` name the
classical diagonal baseline. `CLAUDE.md:102-103` grandfathers the README's historical mentions, so
these stay as history — but `README.md:239` is written as a *current* ontogenesis step, not history,
and it describes state that only the dead `trainSteps` path maintains (`C-RDM-08`).
Repair: reword `README.md:239` to name the notorch tape state that `ntOnGrowth()`
(`molequla.go:7020`) actually resets.

**`C-RDM-14` — JS in the ecology.** `README.md:659` describes DNA exchange as an ecology-wide
property and `README.md:206` lists JS as a full implementation with the same feature set
("Each: autograd, forward/backward, Chuck optimizer, ontogenesis, … corpus field, immune system,
consciousness features, sampling", `README.md:208` — DNA is not in that list, which is correct).
`grep -ci dna molequla.js` → 0. Worth one clause at `README.md:659` saying the DNA layer is
Go/C/Rust.

### Other code-level findings in scope, not README-related

**`C-SPA-02` (P2) — the SPA recursion mutates two globals without `defer`.**
`molequla.go:5014-5017` saves and restores `CFG.SPACoherenceGate` around the recursive call; a panic
inside `generateResonantLocked` leaves the gate off for the process. Worse, the inner call's
`defer gradEnabled.Store(true)` (`molequla.go:4485`) fires on return while the outer frame is still
executing, so `gradEnabled` is `true` for the remainder of the outer SPA block. Today that block
performs no forward pass, so nothing breaks — but `MatrixParam.Matvec` gates the GPU path on
`!gradEnabled.Load()` (`molequla.go:908`), so any future forward added after the SPA splice would
silently drop to CPU.
Repair: `defer func() { CFG.SPACoherenceGate = savedSPA }()`, and make the grad flag a counter or
save/restore it the same way.

**`C-TST-01` (P2) — 8 of the 141 tests test a copy.** `tests/molequla_test.go:1-11` is
`package tests` and re-implements `SoftmaxProbs` with the comment "Copied from molequla.go for
testing since Go cannot import main packages". A change to the real `SoftmaxProbs` cannot turn that
suite red. Same class as `feedback_circular_tests_lying`.
Repair: move the shared numerics into an importable package, or drop the duplicated tests and keep
the ones that exercise real files.

---

## Repair order, if only three things get done before deployment

1. `molequla.go:4664` — one line, restores cross-graze under the default-on overlay (`C-OVL-01`).
2. `molequla.go:4831` — one condition, stops the overlay default from silently killing the corpus
   blend for every warmed organism (`C-OVL-05`).
3. Decide the overlay fade: either call `MetaweightsOverlay` in both regimes so the trained bundle
   binds, or land the sigmoid blend (`C-OVL-02`). Shipping a default-on overlay with a step
   discontinuity of ~25 logits at `mag = 1.0` is the largest open risk in this slice.

For the Go witness, the single load-bearing fact is `C-MYC-01`: the contract it inherits from
`mycelium.py` is written against a schema the Go colony does not have, and the test that would have
caught it builds the schema itself.

— Defender (Arianna Method, phone-1), 2026-09-13

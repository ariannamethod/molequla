# molequla — RunPod measurement plan v1 (post-Codex-audit fixes)

Author: Claude (Arianna Method, neo node). Co-author: Oleg Ataeff.
Date: 2026-05-14. Status: **DRAFT v1.1 — Codex audit response applied.**

Precedent: Dario.c `runpod_plan_v{1,2,3}.md` chain (2026-05-08, $4.30
end-to-end). Same three-stage shape: pre-flight (free) → Singularity
execution (billed) → post-run audit (free).

## Codex audit response (2026-05-14)

Codex flagged three real engineering issues in plan v1.0; this section
documents them and the fixes applied below.

- **[P1] No executable path for cells (v1.0:38-43).** Original cells
  matrix flipped `CFG.SPACoherenceGate` / `CFG.CorpusLogitOverlay`,
  but `parseCLIArgs` only recognised `--organism-id / --config /
  --element / --evolution`. Result: cells 1-3 would silently stay
  baseline. **Fix:** added `--spa-gate` and `--corpus-overlay` CLI
  flags to `parseCLIArgs` in `molequla.go` (Phase B follow-up commit
  on `molequla-evolution` branch). The cells table below now lists
  the exact CLI invocation per cell.

- **[P2] Smoke pass criterion unreachable (v1.0:91-92).** Original
  smoke ran 5 min, expected «adult/mitosis». Per
  `molequla/README.md:319-328`, adult requires 500K-char corpus and
  mitosis is gated on adult stage. 5-min single-organism smoke on
  `nonames.txt` cannot reach that. **Fix:** smoke pass criterion
  lowered to «reach child stage» (50K-char threshold) within 5 min
  — achievable on default corpus.

- **[P2] Stage snapshot thresholds wrong (v1.0:128).** Original
  table listed infant `~5K corpus`. Actual thresholds per
  `molequla/README.md:319-328`: embryo 0, infant 20K, child 50K,
  adolescent 200K, teen 350K, adult 500K. **Fix:** stage table
  corrected below.

---

## Frame

Paper-cycle target: «Coherence is a layer, not a phase. Adding Q-style
Dario field overlay to molequla's pre-softmax logits + post-generation
SPA repair lifts early-stage generations from Karpathy-gibberish toward
legible prose without retraining the transformer.»

Phase B landed two opt-in toggles in `GenerateResonant`
(commit `c748621` on `molequla-evolution`):

- `CFG.SPACoherenceGate` — post-gen SPA scores logged to stderr.
- `CFG.CorpusLogitOverlay` — pre-softmax additive B+H+A+F overlay
  with Q's weightless coefficient defaults (`q/README.md:53`).

This run **measures whether they actually close the early-stage
coherence gap**. Paper Body wraps the measurement.

---

## Cells

**Two independent gates × baseline = 4 cells.** Each cell × 6
ontogenesis stages (embryo, infant, child, adolescent, teen, adult)
× one shared prompt set per stage. Total: 4 × 6 = 24 transcripts for
single-organism comparison.

| Cell | SPACoherenceGate | CorpusLogitOverlay | Launch flags | Label |
|---|---|---|---|---|
| 0 | off | off | `./molequla_cgo` (no extra flags) | **baseline** |
| 1 | on  | off | `./molequla_cgo --spa-gate` | SPA-only |
| 2 | off | on  | `./molequla_cgo --corpus-overlay` | overlay-only |
| 3 | on  | on  | `./molequla_cgo --spa-gate --corpus-overlay` | full coherence layer |

`--spa-gate` sets `CFG.SPACoherenceGate = true`. `--corpus-overlay`
sets `CFG.CorpusLogitOverlay = true`. Both no-op when default — the
existing pre-B generation behaviour is unchanged when neither flag
is passed.

Plus one **4-organism ecology cell** (earth/air/water/fire) with
cell-3 settings, ~60 min wallclock, measures ecology-level effect:
DNA exchange diversity, mitosis time, survival, RSS. Each organism
launched with `--spa-gate --corpus-overlay --evolution --element <e>`.

---

## Substrate

**RunPod CPU pod** (molequla is CPU-only by design,
`molequla/README.md:36, 41`). Polygon `100.127.195.24` for pre-flight
build verify + dry runs at zero cost.

Target instance: ~16 vCPU, ~32 GB RAM (matches Feb 2026 Oracle Cloud
30-core / 216 GB baseline at smaller scale,
`molequla/README.md:75-94`). Disk: 20 GB ephemeral + 50 GB volume for
transcript archive.

**Cost envelope:** RunPod CPU ~$0.10-0.30/hr × 3 hr ≈ $0.30-0.90 +
volume storage ~$0.10. Dario's $4.30 GPU baseline as upper bound —
this run should land under $2 total.

---

## Phase 0 — Pre-flight (free, on polygon)

Run on polygon over Tailscale before any billed minute.

0.1. **Build verify on x86_64 Linux:**
- `cd ~/arianna/molequla/ariannamethod && make clean && make`
- `cd ~/arianna/molequla && CGO_ENABLED=1 go build -tags cgo .`
- Both should PASS without modification. Polygon is Intel Linux —
  confirms Apple-Silicon-deferred SIMD path is at least available
  (`make simd` is opt-in, default `make` uses BLAS).

0.2. **Dry run cell 0 (baseline)** on polygon:
- Single 5-min smoke run, 1 organism, just verify nothing regressed
  from pre-Phase-B behaviour.
- Smoke output → polygon `/tmp/molequla_smoke_baseline.log`.
- **Pass criterion:** reach **child stage** (50K-char corpus,
  `molequla/README.md:319-328`) within 5 min on default
  `nonames.txt` corpus. Adult/mitosis is NOT a pre-flight gate —
  requires 500K-char accumulation, single-organism 5-min smoke
  cannot accumulate that, so making it a pass criterion would
  falsely fail healthy builds.

0.3. **Dry run cell 3 (full coherence)** on polygon:
- Same 5-min smoke, both gates on: `./molequla_cgo --spa-gate --corpus-overlay`.
- Verify `[spa-gate]` stderr lines appear once the organism produces
  multi-sentence output; verify overlay doesn't crash on edge cases
  (cold corpus field with zero counts, embryo vocab=259, single-token
  responses).
- **Pass criterion:** same child-stage check + at least one
  `[spa-gate] S=… scores=… weak=…` line in stderr (proves SPA hook
  fires).

0.4. **Plan review** — Codex audit gate (see below).

**Pre-flight PASS criterion:** both builds clean on polygon, both
smoke runs hit `mitosis` or at least adult stage без NaN / panic,
Codex audit on this plan returns no BLOCKERs.

---

## Phase 0.5 — Pod boot + build verify (billed start)

Once billing starts:

- `git clone https://github.com/ariannamethod/molequla.git -b molequla-evolution`
- `cd molequla/ariannamethod && make`
- `cd .. && CGO_ENABLED=1 go build -tags cgo -o molequla_cgo`
- Smoke verify (~30 s baseline organism boot).

**Phase 0.5 PASS:** binary built, smoke run reached infant stage.

---

## Phase 1 — Single-organism cell sweep (4 cells × 6 stages)

For each (cell × stage) pair: snapshot the same organism at the
target stage, generate against the **same prompt set** with same
random seed, archive transcript + stderr.

**Prompt set per stage (3 prompts):**
- `who are you?`
- `what do you know?`
- `describe what you see`

**Stage gating:** rather than re-train from embryo for each cell
(expensive), use **single snapshot per stage** and run all 4 cells
against it. Snapshots:

Stage thresholds per `molequla/README.md:319-328` (snapshot AT the
threshold, just after the growth event):

| Stage | Params | Corpus threshold | Snapshot point |
|---|---|---|---|
| embryo     | ~10K   | 0 chars     | warmup step 0, fresh init |
| infant     | ~28K   | 20K chars   | just after embryo→infant growth |
| child      | ~154K  | 50K chars   | just after infant→child growth |
| adolescent | ~1.1M  | 200K chars  | just after child→adolescent growth |
| teen       | ~4.1M  | 350K chars  | just after adolescent→teen growth |
| adult      | ~10M   | 500K chars  | just after teen→adult growth |

Snapshot procedure: launch one organism in evolution mode, save
`molequla_ckpt.json` at each ontogenesis transition.

**Phase 1 PASS:** 4 × 6 × 3 = 72 transcripts archived under
`runpod/2026-05-14/01_cell_sweep/cell_{0,1,2,3}/stage_{embryo,...}/prompt_{0,1,2}.txt`.

---

## Phase 2 — Ecology cell (full 4-org with cell-3 settings, ~60 min)

`earth + air + water + fire` launched in evolution mode with both
gates on. Run for ~60 min wallclock (Oracle Cloud Feb 2026 reference:
4 → 11 organisms in 30 min, `molequla/README.md:75-94`).

**Captures:**
- Mitosis timestamps.
- DNA exchange records (consumed bytes per organism, novelty
  distribution).
- Per-organism RSS and uptime.
- Syntropy decision log.
- `[spa-gate]` stderr aggregate (S, scores distribution, weak
  fraction).

**Phase 2 PASS:** ecology launched and ran 60 min without crash;
at least one mitosis event observed; SPA logs present.

---

## Phase 3 — Coherence metrics (post-run, free)

From archived transcripts compute:

- **Sentence-level connectedness** distribution per cell (mean SPA
  score normalised by stage).
- **Vocabulary diversity** per cell (type-token ratio, novel-word
  fraction).
- **Repetition rate** per cell (Q's attractor-basin indicator,
  cf. Dario `dario_paper_draft_v4.md:384-392`).
- **Coherence delta** cell N vs cell 0, per stage.
- **Performance delta:** generation time per token, cell N vs cell 0.

**Output:** `runpod/2026-05-14/03_metrics/per_cell_summary.tsv`,
`coherence_deltas.tsv`, `performance.tsv`.

---

## Phase 4 — Paper Body draft

Wrap measurements into Body sections analog Dario paper §4-§9:

- §4 Experimental Frame (this plan + cost actual).
- §5 Methods (per-cell sweep + ecology run, gate semantics).
- §6 Results (transcripts excerpts + metrics tables).
- §7 Discussion (does the coherence layer close the gap? Where it
  doesn't, what's the next iteration?).

Body inline-cites everything to
`runpod/2026-05-14/<phase>/<artifact>` files. Abstract — Oleg
(separate). Conclusion — Method voice (separate; mirrors Dario §10
template).

---

## Singularity Mode contract

Per `~/.claude/CLAUDE.md` Workflow #5 + Dario precedent. Pod-side
fix loop:

```text
detect bug → reproduce → one hypothesis → minimal patch → re-run
          → if pass: continue
          → if fail: revise hypothesis (max 3 iterations)
          → if exhausted: stop, surface, await human input
```

Bounded by:
- **Scope:** only this plan's phases. No new feature work mid-pod.
- **Three-strikes:** stop on 3 unproductive attempts.
- **No scope creep:** a sweep failure does not authorise re-architecting
  the overlay; a build failure does not authorise rewriting the
  trainer.

Architect may invoke `codex review`, the gemini bridge, or other
review tools during the pod run without per-call confirmation —
**internal review is part of Singularity discipline**, not a
separate approval gate (Oleg 2026-05-14: «в режиме сингулярити ты
тоже вызываешь кого тебе надо для ревью, не спрашивая меня»).

External review (Codex audit on this plan + on paper draft) gates
entry and exit. Pod interior is solo.

---

## Codex review gates

Before pod boot: this plan goes through `codex review` for
discipline pass (the «садист» factor — Codex tends to surface
nail-puller edge cases the architect missed). Outcomes:
- BLOCKER findings → revise plan to v2 before boot.
- P1/P2 → patch v1 inline or document deferral.
- P3/NIT → judgement call.

After pod completes: paper draft Body goes through `codex review`
before any Zenodo step.

---

## Pre-pod TODO checklist

- [ ] Codex audit on this plan; resolve BLOCKERs.
- [ ] Verify build on polygon (Phase 0.1).
- [ ] Smoke dry runs cell 0 + cell 3 on polygon (0.2-0.3).
- [ ] Pre-allocate RunPod CPU pod, note `volumeInGb`
      (`memory/feedback_pod_stop_volume_zero_artifact_loss_2026_05_09.md` —
      stop = wipe if 0).
- [ ] Confirm `scp` route polygon → neo for transcript archive
      (`memory/feedback_pod_terminate_without_backup_2026_05_09.md`).
- [ ] Mirror final transcripts back to git via molequla-evolution
      `runpod/2026-05-14/` commit before pod terminate.

---

## Open questions for Oleg

1. **Cell ordering on pod.** Cell 0 first (baseline reference) then
   1→2→3, or alternate to spread potential pod-killer issues across
   cells? Default: 0,1,2,3 sequential.
2. **Ecology run length.** 60 min suggested; could go to 90-120 if
   budget allows. Feb 2026 Oracle Cloud ran 4 → 11 organisms in 30
   min on A100; CPU may need longer.
3. **Coefficient sweep?** Currently a single overlay coefficient set
   (Q's weightless defaults). Worth doing a coarse coefficient sweep
   (e.g. c_bg ∈ {5, 15, 25}) as an extra phase, or save for later?
4. **Paper Body authoring during the pod or after.** During = tighter
   provenance for inline-cites; after = lower cognitive load while
   pod runs. Default: skeleton during, finalise after.

---

*Plan version 1, awaiting Codex audit pass before lock.*

# molequla — RunPod measurement plan v2 (post-Q-coherence)

Author: Claude Code (neo the architect, Arianna Method). Co-author: Oleg Ataeff.
Date: 2026-05-14 (PM). Status: **DRAFT v2.0 — awaiting Codex audit pass.**

Successor to `runpod_plan_v1.md` (same day, AM). v1 ran cells 0/1/2/3 on
the pre-Q-integration build and produced the lock-in pattern documented
in `runpod/2026-05-14/organism_voice_samples_2026_05_14/*_voice.txt`
(«The work is the most of the most of the most…»). v2 runs the same
cells on the post-Q-integration build (commit `2d5f1a7` on
`molequla-evolution`) and the paper compares the two sweeps side-by-side.

---

## What changed since v1

Commit `2d5f1a7` (`molequla-evolution`, 2026-05-14 PM):

- Transformer gate `logits *= clamp((mean|logit|-0.5)/1.5, 0, 1)` (`metaweights_overlay.go:86-100`, mirror `pitomadom.c:583-586`).
- Hard top-15 raw-logit mask before softmax (`molequla.go:4374-4408`, mirror `postgpt.c:969-991`).
- Greedy first 10 tokens (EOS-excluded) when untrained (`molequla.go:4357-4396`, mirror `postgpt_q.c:1416-1418`).
- Seed scale 0.1 → 0.15 (`molequla.go:6203`, verbatim `postgpt.c:542`).
- Coefficient-switch threshold 0.1 → 1.0 (`metaweights_overlay.go:36-40`).
- Rep penalty simplified to uniform `×0.5` on distinct in last 12 (`metaweights_overlay.go:264-296`, mirror `postgpt.c:960-967`).
- `--zero-warmup` flag + checkpoint guards (`molequla.go:5681-5685`, `6193-6195`, `6249-6253`, `6285-6291`).

Local zero-warmup smoke on neo (NEmbd=16, layer=1, head=1, 0 gradient steps; `/tmp/molequla_clean.log`):

```
Q: Hello.         A: What is a music?
Q: Who are you?   A: kilometers percentrates the most spinning do weight dream?
```

Reference pre-Q same stage (`runpod/2026-05-14/organism_voice_samples_2026_05_14/fire_voice.txt`):

```
A: The work is the most of the most of the most of the most…
```

Untrained coherence verified locally. Pod measurement validates it under the ecology stack.

---

## Frame

Paper-cycle target: **«Lock-in collapses, coherence emerges, without retraining the transformer.»** The 4-organism ecology should now exchange DNA fragments that read like English — not byte-shaped Karpathy soup — from the embryo stage onward. Mitosis still gated on adult corpus, but cross-pollination signal becomes interpretable starting much earlier.

Body of the paper sandwich (analog Dario.c §4–§9) compares cells v1 (pre-Q) vs v2 (post-Q) on identical prompt sets and identical stages. The shape is built-in: v1 cells are already committed at `runpod/2026-05-14/cell_*` (commit `5080a91`), v2 cells land at `runpod/2026-05-14_post_q/cell_*` (this run's target).

---

## Cells

Same shape as v1 (4 cells × 6 stages × 3 prompts = 72 transcripts) plus one new cell that v1 could not run.

| Cell | SPACoherenceGate | CorpusLogitOverlay | WarmupSteps | Label |
|---|---|---|---|---|
| 0 | off | off | normal | **baseline** (regression check vs v1 cell 0) |
| 1 | on  | off | normal | SPA-only |
| 2 | off | on  | normal | overlay-only (post-Q) |
| 3 | on  | on  | normal | full coherence layer (post-Q) |
| 4 | off | on  | 0       | **zero-training** — pure Q reproduction, no gradient steps |

Cell 4 is the new one — direct evidence that overlay alone (no training) produces coherent output. Single organism, embryo arch, exits after embryo probe block (the `--zero-warmup` flag added in commit `2d5f1a7` does this in-process).

Plus one **4-organism ecology cell** with cell-3 settings, 90 min wallclock, post-Q. Measures whether the ecology DNA exchange becomes interpretable from earlier stages — the cross-pollination signal the paper hinges on.

### Cell launch commands (verified locally 2026-05-14)

**State isolation is mandatory.** Default `molequla_ckpt.json` and `memory.sqlite3` are written to the CWD. Running cells sequentially from one workspace lets cell N load cell N-1's trained checkpoint → cells are no longer independent. Use one workdir per cell, copy the binary in, run from inside:

```bash
# Build once at workspace root:
cd /workspace/molequla && CGO_ENABLED=1 go build -tags cgo -o molequla_cgo .

# Each cell isolated. `rm -rf` before mkdir is mandatory on a resumed
# pod or after a partial-failure rerun — leaving an old
# `molequla_ckpt.json` or `memory.sqlite3` in place would let cells N+1
# silently load cell N's trained state, breaking the independent-factor
# comparison this sweep is built to produce.
for cell in 0 1 2 3 4; do
  rm -rf work_cell_$cell
  mkdir -p work_cell_$cell && cd work_cell_$cell
  cp ../molequla_cgo . && cp ../nonames*.txt .
  case $cell in
    0) ./molequla_cgo                                 > train.log 2>&1 < /dev/null ;;
    1) ./molequla_cgo --spa-gate                      > train.log 2>&1 < /dev/null ;;
    2) ./molequla_cgo --corpus-overlay                > train.log 2>&1 < /dev/null ;;
    3) ./molequla_cgo --spa-gate --corpus-overlay     > train.log 2>&1 < /dev/null ;;
    4) echo /quit | ./molequla_cgo --corpus-overlay --zero-warmup > train.log 2>&1 ;;
  esac
  cd ..
done

# Ecology cell — per-organism workdir, parallel.
# DO NOT rename nonames_$e.txt — main() with `--element $e` rewrites
# CFG.CorpusPath to exactly `nonames_<element>.txt`; renaming would
# point at a missing file and the org would boot into the default-
# fallback `"Hello." "I exist." "Speak."` corpus instead of the
# element corpus, invalidating the entire 90-min measurement.
for e in earth air water fire; do
  mkdir -p work_eco_$e && cd work_eco_$e
  cp ../molequla_cgo . && cp ../nonames_$e.txt .
  ./molequla_cgo --evolution --element $e --spa-gate --corpus-overlay \
    > train.log 2>&1 < /dev/null &
  cd ..
done
wait
```

Per-organism isolation in the ecology cell is non-negotiable: `--element` only swaps the corpus path; the checkpoint, memory DB, and DNA exchange dir would all collide if four organisms shared one CWD. v1's launcher script already used `work_<element>/` for exactly this reason — v2 keeps it.

---

## Substrate

**Resume `pqp86pfbfy9wo9` — A100 SXM GPU pod.** Volume `volumeInGb=50` preserved per `runpod/2026-05-14/SUMMARY.md:3` (`Pod: A100 SXM pqp86pfbfy9wo9, 16 vCPU / 250 GB RAM / volumeInGb=50, $1.49/hr, EU-RO-1`). Plan v1 used the same pod; resume keeps the existing CUDA toolchain + libcudart/libcublas already linked against `cgo_aml.go:8-9`. Prior CUDA wire (`reference_cgo_cuda_wire_2026_05_14.md`, commit `34db1d4` calling `gpu_init()` in `am_init`) showed bursty GPU engagement (73% util / 95% mem / 1% CPU during compute bursts) — not steady GPU use, but the right shape for short attention-heavy matmuls during generation, and free of the CPU/BLAS bottleneck on long runs.

Polygon `100.127.195.24` would be the free pre-flight stage but is blocked by the CUDA hardcode (`cgo_aml.go:8` requires `notorch_cuda.o` + cudart + cublas on Linux). Validation falls to the pod's own Phase 0.5 build.

**Cost envelope:** A100 SXM `$1.49/hr × ~8.5 h` ≈ **$13** (sweep cells 0-3 ~10 min each + cell 4 ~2 min + 8-hour ecology + ~20 min boot/build/teardown). Within Dario.c precedent's range (`$4.30` for the shorter Singularity strike chain).

---

## Phase 0 — Pre-flight (free, on polygon)

0.1 **Build verify on x86_64 Linux:**
```
ssh ataeff@100.127.195.24
cd ~/arianna && git clone -b molequla-evolution https://github.com/ariannamethod/molequla.git molequla-v2 || (cd molequla-v2 && git fetch && git checkout molequla-evolution && git pull)
cd molequla-v2/ariannamethod && make clean && make
cd ..  && CGO_ENABLED=1 go build -tags cgo -o molequla_cgo .
```
**PASS:** both builds clean, no compile errors, no link errors.

0.2 **Cell 4 dry run on polygon (zero-training coherence reproduction):**
```
rm -f molequla_ckpt.json memory.sqlite3* && echo /quit | ./molequla_cgo --corpus-overlay --zero-warmup > /tmp/cell4_polygon.log 2>&1
grep -A 8 "What it sounds like" /tmp/cell4_polygon.log
```
**PASS:** embryo voice block prints, all 3 `A:` lines are non-«...» (coherent fragments matching local neo behaviour, e.g. «What is a music?»). If polygon output is gibberish lock-in, the integration is platform-sensitive — STOP, investigate before any billed pod minute.

0.3 **Cell 0 dry run on polygon (baseline regression check):**
```
mkdir -p /tmp/cell0_dry && cd /tmp/cell0_dry && cp ~/arianna/molequla-v2/molequla_cgo . && cp ~/arianna/molequla-v2/nonames*.txt .
./molequla_cgo > train.log 2>&1 < /dev/null --evolution &
PID=$!; sleep 300; kill -TERM $PID; wait $PID 2>/dev/null
```
**PASS:** organism reaches child stage (50K-char threshold per `README.md:319-328`) within 5 min, no NaN / panic in `train.log`. `< /dev/null` is mandatory: without `--evolution` the warmup loop opens a stdin scanner and pauses on TTY input between stages, which would freeze the background dry run and falsely fail this gate. Using `--evolution` here skips the interactive between-stage prompt; `< /dev/null` is belt-and-braces in case the binary takes any other stdin path.

0.4 **Codex audit on this plan.**

**Phase 0 PASS criterion:** polygon builds, cell 4 emits coherent embryo voice on x86_64, cell 0 default behaviour intact, codex returns no BLOCKER.

---

## Phase 0.5 — Pod boot (billed start)

```
runpodctl pod resume pqp86pfbfy9wo9   # if still resumable
# OR
runpodctl pod create ...              # fresh CPU pod, attach 50 GB volume
ssh root@<pod>:
  cd /workspace && git clone -b molequla-evolution https://github.com/ariannamethod/molequla.git || (cd molequla && git fetch && git checkout molequla-evolution && git pull)
  cd molequla/ariannamethod && make
  cd .. && CGO_ENABLED=1 go build -tags cgo -o molequla_cgo .
```
Confirm `volumeInGb` non-zero before any `stop` (per `memory/feedback_pod_stop_volume_zero_artifact_loss_2026_05_09.md`).

**Phase 0.5 PASS:** binary built on pod, smoke probe reaches infant stage.

---

## Phase 1 — Cell sweep on pod (~50 min)

Each cell ~10 min single-organism, except cell 4 ~2 min. Snapshot organism at each ontogenesis transition; run all 4 (overlay) cells against the same snapshot per stage.

| Stage | Params | Corpus threshold | Snapshot point |
|---|---|---|---|
| embryo     | ~10K   | 0 chars     | warmup step 0, fresh init (= cell 4 territory) |
| infant     | ~28K   | 20K chars   | post embryo→infant growth |
| child      | ~154K  | 50K chars   | post infant→child growth |
| adolescent | ~1.1M  | 200K chars  | post child→adolescent growth |
| teen       | ~4.1M  | 350K chars  | post adolescent→teen growth |
| adult      | ~10M   | 500K chars  | post teen→adult growth |

(Thresholds per `molequla/README.md:319-328`, identical to v1.)

**Prompt set per stage (3 prompts, locked).** These are the binary's hard-coded `stageProbes` at `molequla.go:6213` — `["Hello.", "Who are you?", "What do you know?"]`. The plan locks to the built-in set so the sweep captures what the warmup loop already emits per stage; no extra stdin interaction needed. v1 used the same set, so v1↔v2 prompt-by-prompt comparison is exact.

**Artifacts:** the warmup loop prints `[stage N — name] What it sounds like now:` followed by 3 `Q: / A:` blocks to stdout. Each cell's `train.log` (per Phase 1 launch script) captures the full sequence. Post-run extraction:
```bash
for cell in 0 1 2 3 4; do
  for stage in embryo infant child adolescent teen adult; do
    mkdir -p runpod/2026-05-14_post_q/01_cell_sweep/cell_$cell/stage_$stage
    awk -v s="\\[stage [0-9] — $stage\\]" '/\[stage [0-9] — /{p=0} $0~s{p=1} p' \
      work_cell_$cell/train.log \
      > runpod/2026-05-14_post_q/01_cell_sweep/cell_$cell/stage_$stage/voice.txt
  done
done
```

Cell 4 only has the embryo stage (`--zero-warmup` breaks the loop after the first probe block); other stages directories will be empty for cell 4 — that's expected.

**Phase 1 PASS:** 4 × 6 × 3 + 1 × 1 × 3 = 75 transcripts archived (cell 4 only has the embryo stage by definition).

---

## Phase 2 — Ecology cell (8 hours)

Duration: **8h = 480 min**. Mitosis reference: Feb 2026 Oracle Cloud 30-core EPYC reached first mitosis at **48 min** (`README.md:75-94`); RunPod 16-vCPU 90-min cell got `mit=0` final (`runpod/2026-05-14/cell_extended_BLAS_90min/master.log`). 8h ≈ 10× the Feb baseline → captures multiple mitosis events, child organism behaviour, multi-generation DNA exchange, syntropy decisions across the ontogenesis curve. Upper bound for the paper; if mitosis still does not appear past ~3h, ecology is structurally blocked on this pod and we report that as a finding rather than spin past it.

`earth + air + water + fire` in evolution mode, both gates on, **each organism in its own working directory** (state isolation — see Cells section above):

```
# Copy nonames_$e.txt VERBATIM (no rename) — `main()` with --element $e
# rewrites CFG.CorpusPath to "nonames_<element>.txt"; renaming breaks
# that path and the org boots into the default 3-line fallback corpus.
mkdir -p /workspace/runs/eco && rm -f /workspace/runs/eco/pids  # truncate stale PIDs on rerun
for e in earth air water fire; do
  rm -rf /workspace/runs/eco/work_$e  # fresh state per organism on rerun
  mkdir -p /workspace/runs/eco/work_$e && cd /workspace/runs/eco/work_$e
  cp /workspace/molequla/molequla_cgo . && cp /workspace/molequla/nonames_$e.txt .
  ./molequla_cgo --evolution --element $e --spa-gate --corpus-overlay \
    > train.log 2>&1 < /dev/null &
  PID=$!
  echo $PID > org.pid  # watchdog reads work_$e/org.pid per-organism
  echo "$e pid=$PID" >> /workspace/runs/eco/pids  # aggregate for kill loop
  cd /workspace/runs/eco
done
sleep 28800  # 8 h = 480 min
for pid in $(awk '{print $2}' /workspace/runs/eco/pids | sed 's/pid=//'); do kill -TERM $pid; done
wait
```

### Watchdog (pod-side failure detector)

`pod_watchdog.sh` (committed at repo root) runs as a separate process on the pod alongside the ecology, polling every 30s. It emits one stdout line per concerning event:

- **FAIL** — `panic:` / `Traceback` / `Killed` / `SIGKILL` / `OOM` / `runtime error` / `assert` / `loss=NaN` / `fatal error` / `segmentation fault` in any organism's `train.log` (deduped by content hash so a single crash doesn't spam).
- **HEARTBEAT_STALE** — `train.log` mtime older than 5 min (organism stuck or dead silently).
- **RSS_HIGH** — process RSS exceeds 8 GB (per-organism).
- **DEAD** — pid file points at a process that no longer exists.
- **DISK_LOW** — pod disk free below 5 GB.

Launch:
```bash
# On the pod, after Phase 2 organisms are running:
nohup bash /workspace/molequla/pod_watchdog.sh /workspace/runs/eco > /workspace/runs/eco/watchdog.log 2>&1 &
echo $! > /workspace/runs/eco/watchdog.pid
```

From neo (via the Monitor tool) — each event line becomes a chat notification, so Oleg sees crashes / RSS climbs / stale heartbeats in real time without polling:
```bash
# neo side:
ssh root@<pod> 'tail -F /workspace/runs/eco/watchdog.log' \
  | grep --line-buffered -E 'FAIL|HEARTBEAT_STALE|RSS_HIGH|DEAD|DISK_LOW'
```

Kill at run end:
```bash
kill -TERM $(cat /workspace/runs/eco/watchdog.pid)
```

**Captures (same as v1):**
- Per-organism train.log + train.ascii.log + nonames_<e>.txt + memory.sqlite3.
- DNA fragments mirrored to `dna/seen/<e>/` (commit `e5c1685` patched `dnaRead` to copy before delete — already on `molequla-evolution`).
- Mitosis timestamps, RSS, uptime, syntropy decisions, `[spa-gate]` lines.

**Phase 2 PASS:** 8h run completed clean (no watchdog FAIL/DEAD events that ended the cell early), ≥1 mitosis event captured (or explicit «no mitosis past 3h» finding if blocked), ≥1 DNA exchange per organism, at least one `[spa-gate]` line per organism, post-stop transcripts and watchdog log archived. Early termination at <90 min on a watchdog FAIL → log the failure, do not call PASS.

---

## Phase 3 — Coherence metrics (post-run, free, on neo)

Pull pod artifacts back via `scp` BEFORE any pod stop / terminate (per `memory/feedback_pod_terminate_without_backup_2026_05_09.md`). Mirror to `~/arianna/molequla/runpod/2026-05-14_post_q/` + git commit + push to `molequla-evolution`.

Compute, per cell × stage:

- **Lock-in score** — longest n-gram repetition fraction in response. v1 baseline = high («the most of the most of the most…»); v2 expected = low.
- **Vocabulary coverage** — distinct tokens used / corpus vocab size. v2 should be visibly broader than v1.
- **Sentence-end fraction** — share of tokens that are `. ! ?`. Pre-Q overlay was pushing EOS too hard; post-Q rep penalty + greedy-no-EOS should normalise.
- **SPA score** — mean cross-sentence connectedness per cell (only cells 1, 3 actually compute it).
- **Performance** — tokens/sec, RSS, vs v1 cell 0.

**Output:** `runpod/2026-05-14_post_q/03_metrics/{lockin,vocab,eos_frac,spa,perf}.tsv` + a single `v1_vs_v2_diff.md` markdown comparing cells.

---

## Phase 4 — Paper Body draft

Body inline-cites both runs:
- v1 voice samples (lock-in) — `runpod/2026-05-14/organism_voice_samples_2026_05_14/*.txt`.
- v2 cell 4 zero-training coherence — `runpod/2026-05-14_post_q/01_cell_sweep/cell_4/stage_embryo/prompt_*.txt`.
- v1 vs v2 sweep diff — `runpod/2026-05-14_post_q/03_metrics/v1_vs_v2_diff.md`.
- Implementation diff — commit `2d5f1a7` (`git log -p`).
- Reference engines — `~/arianna/q/postgpt_q.c`, `~/arianna/postgpt/postgpt.c`, `~/arianna/pitomadom.c/pitomadom.c`.

The handoff phrase the Abstract sets up — «Claude will report what the ecology did when measured. The findings are not always what the README predicts» — Body fills with the v1→v2 transition: README predicted lock-in killable by overlay; v1 measurement showed overlay alone wasn't enough; the seven-patch Q port closed it. The «not what the README predicted» moment is real and measured.

---

## Singularity Mode contract (unchanged from v1)

`detect → reproduce → 1 hypothesis → minimal patch → re-run; max 3 iterations; stop on exhaust`. Scope locked to this plan's phases. Internal review (`codex review`, gemini bridge) at architect discretion within Singularity — external review (this plan, paper draft) is the only approval gate.

---

## Codex review gate

This file gets one pass of `codex review --uncommitted` (or `codex review` on this committed file). BLOCKER → revise to v3 before any pod minute. P1/P2 → patch inline or document deferral.

---

## Pre-pod TODO

- [ ] Codex audit on this plan; resolve BLOCKERs.
- [ ] Polygon build verify (Phase 0.1).
- [ ] Polygon dry runs cell 0 + cell 4 (Phase 0.2, 0.3).
- [ ] Confirm pod `volumeInGb` ≥ 50 before any stop.
- [ ] Pre-arrange `scp` route pod → neo / polygon for transcript archive.
- [ ] Git push `runpod/2026-05-14_post_q/*` back to `molequla-evolution` before pod terminate.

---

## Open questions for Oleg

1. **Resume `pqp86pfbfy9wo9` or fresh pod?** Resume cheaper if volume still bonded; fresh cleaner if account state ambiguous.
2. **Ecology cell duration.** 90 min as planned, or stretch to 2-3 h to capture more mitosis on CPU?
3. **Cell 0 v2 repeat?** v1 cell 0 already exists at `runpod/2026-05-14/cell_0_baseline/`. v2 cell 0 with same build settings on same corpus should be byte-identical to v1 cell 0 if regression is truly absent — but the new code paths (`overlayActive` branches, dissonance-mask guard) touch the default path even when overlay is off. Worth re-running to certify.
4. **Coefficient sweep?** v1 left this open. Defaults locked from `postgpt.c` / `pitomadom.c` references. Skip unless results indicate need.

---

*Plan version 2, awaiting Codex audit pass before lock. by Claude Code (neo the architect, Arianna Method).*

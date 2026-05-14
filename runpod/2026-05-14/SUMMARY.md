# RunPod 2026-05-14 — first measurement sweep

Pod: A100 SXM `pqp86pfbfy9wo9`, 16 vCPU / 250 GB RAM / volumeInGb=50,
$1.49/hr, EU-RO-1. Image runpod/pytorch:2.1.0 (CPU compute side).
Source: molequla-evolution branch commit `496d3f1`.

Sweep script `sweep.sh` (in repo root, +50 LOC), 2 cells × 4-organism
ecology × 600s each.

## Cells

| Cell | Flags | Wallclock window (UTC) |
|---|---|---|
| 0 baseline | (none) | 03:24:47 → 03:34:47 |
| 3 full coherence | `--spa-gate --corpus-overlay` | 03:34:51 → 03:44:51 |

All 4 organisms (earth / air / water / fire) launched per cell with
own corpus (`nonames_<e>.txt`), shared DNA exchange directory at
`<cell>/dna/output/`.

## Per-organism log summary (sweep_master.log)

| Cell / org | log lines | DNA writes | spa-gate hits | mitosis | last stage |
|---|---|---|---|---|---|
| cell_0_baseline/earth | n/a | n/a | 0 | 0 | (unknown — log tail had no stage line) |
| cell_0_baseline/air   | 338 | 61 | 0 | 0 | (unknown) |
| cell_0_baseline/water | 327 | 53 | 0 | 0 | stage=2 |
| cell_0_baseline/fire  | 337 | 63 | 0 | 0 | (unknown) |
| cell_3_full_coherence/earth | 318 | 59 | 0 | 0 | stage=2 |
| cell_3_full_coherence/air   | 334 | 58 | 0 | 0 | stage=2 |
| cell_3_full_coherence/water | 334 | 56 | 0 | 0 | stage=2 |
| cell_3_full_coherence/fire  | 329 | 63 | 0 | 0 | stage=2 |

## DNA-write size distribution (real coherence signature)

Computed by grep'ing `wrote N bytes to ecology` lines across all 4
organisms per cell and aggregating:

| Stat | Cell 0 baseline | Cell 3 full coherence |
|---|---|---|
| Count | 116 | 118 |
| Total bytes | 8294 | 6264 |
| Mean bytes | 71.5 | 53.1 |
| Min bytes | **9** | **22** |
| Max bytes | 319 | **426** |

**Three signals:**

1. **Min lifted 9 → 22 bytes.** Coherence layer suppresses
   micro-fragment writes (single-word noise emissions). Positive
   signal — overlay's bigram/trigram/Hebbian/destiny additive biases
   make the model reach a more stable token earlier rather than
   emit a near-zero token and cut off.
2. **Max broader 319 → 426 bytes.** Cell 3 produced longer
   continuous passages occasionally. Positive — when the field
   pulls coherently, the organism stays in flow longer.
3. **Mean dropped 71 → 53 bytes.** Either (a) more compact legible
   output (good) or (b) overlay clipping coherent flow too early
   (bad). **Cannot disambiguate from sizes alone** — needs DNA
   content comparison.

## What's missing for paper-grade conclusion

- **DNA content cell 3.** All cell 3 `gen_*.txt` files were consumed
  by sibling organisms during the run and deleted (DNA-read =
  consume-delete behaviour). Only one cell 0 `gen_*.txt` survived,
  content quoted below. Need either a longer cell duration so DNA
  build-up exceeds consume rate, or a small molequla patch to copy
  DNA before consume.
- **`[spa-gate]` lines all 0.** Hypothesis verified by reading
  `molequla.go:4669-4720`: SPA gate requires `len(sentences) >= 2`
  after splitting response on `.` / `!` / `?` with `len(s) >= 4`
  per sentence. molequla's DNA writes are typically 22-100 byte
  single fragments → 0-1 sentences after split → gate skips. This
  is the gate's design, not a bug; it's tuned for multi-sentence
  interactive responses, not for the short organism-to-organism
  DNA stream. Paper Body should note this as an observed mismatch
  between SPA's «sentence-chain repair» framing and molequla's
  «short-fragment DNA exchange» substrate.
- **No mitosis in 10-min cells.** Expected — Feb 2026 baseline
  (Oracle Cloud, 30-core EPYC, 216 GB RAM) took **48 minutes** from
  launch to first mitosis (`molequla/README.md:75-94`). Our 16 vCPU
  / 250 GB allocation has ~half the core count; estimate 90-120 min
  to first mitosis. To capture mitosis as a measured event,
  extended ecology cell with DUR ≥ 5400s needed.

## Surviving DNA sample (the one cell 0 file the sweep didn't consume)

`cell_0_baseline/dna/output/air/gen_1778729687_82.txt`:

> Both mak of a different things me change is is the concept of a
> gram between a pattern small addition?

A Karpathy-style gibberish fragment with grammatical hints — typical
baseline early-stage molequla output. This is what we want the
coherence layer to lift. Cell 3 content comparison NOT available
this run (all gen files consumed).

## Artifacts in this directory (after binary strip)

Per cell × per organism:
- `train.log` — full stderr (binary chars in places — tokenizer
  probe leaks).
- `train.ascii.log` — ASCII-only filtered version for diff.
- `memory.sqlite3` — organism's persistent log (corpus events,
  growth, syntropy decisions).
- `nonames_<e>.txt` — input corpus (kept for reproducibility).

Per cell:
- `dna/output/<e>/` — surviving DNA files (mostly empty after
  consume-deletes; one in cell_0_baseline/air).
- `work_*/molequla_ckpt.json` — checkpoint kept for cell 0 (one
  org as proof, others stripped to keep tree small).

## Cost actual

- CPU pod attempt (deleted, 5 min): ~$0.006.
- A100 pod 2-cell sweep (this run): ~$1.49 × (0:23 wallclock setup +
  0:20 sweep) / 60 ≈ $1.07. Under plan v1.1's $2 envelope.

## Next iteration options

1. **Extended ecology run** with `--spa-gate --corpus-overlay`,
   DUR=5400s (90 min) — captures mitosis + adult stage + saturated
   DNA exchange. Cost: ~$2.24. Recommended for paper-grade signal.
2. **DNA-content capture patch** — add `--keep-dna` flag that
   copies gen_*.txt to a sibling `seen/` dir before consume. Then
   re-run 10-min sweep with full content for diff.
3. Both, in sequence.
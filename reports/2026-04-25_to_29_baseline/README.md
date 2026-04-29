# Molequla Hebbian-only baseline — 2026-04-25 to 2026-04-29

Pre-recipe baseline run on Railway (`molequla-ecology` project, service
`ecology`, deployment `bfb59954-da9a-484c-8d77-63f65cafa119`, started
2026-04-28T09:48:56Z, archived 2026-04-29T14:12:34Z).

## Why this archive exists

Investigation 2026-04-29 (task #17) determined that molequla's
Dockerfile was missing the Railway CPU-fix recipe:
- No `OPENBLAS_NUM_THREADS=1` environment variable.
- `CGO_CFLAGS="-O2 -DUSE_BLAS"` instead of `-O3 -march=native -mtune=native`.

With four organisms running in parallel under the launcher and
multi-thread OpenBLAS spawning per `sgemv`, this is exactly the small-dim
thread-storm class that Henry's session 2026-04-29 measured at 7.4×
speedup once fixed.

The investigation also found **no architectural bug**. molequla was
following its design ontogenesis (embryo → infant → child → adolescent
→ teen → adult, gated on corpus size). At 4 days the four organisms
sat at adolescent stage (corpus ~324 KB, threshold 200 KB reached but
350 KB / teen not yet). Mid-loss 1.2-1.8 on 32-step micro-bursts is
the steady state of frozen-base + delta training inside that stage,
not a stuck signal.

This archive captures that pre-fix run as a legitimate baseline before
the recipe restart.

## Files

- `logs_tail_5000.txt` — last 5000 deployment log lines pulled via the
  Railway GraphQL API. Spans 2026-04-28T20:19:36Z to 2026-04-29T14:12:34Z
  (~18 hours of the run; deployment was alive 2026-04-25 onward but
  Railway's log retention truncated the earlier window).

## Notable patterns in the archive

- All four organisms (earth / air / water / fire) reach `stage=3 freeze=0`
  consistently — adolescent stage in CFG.GrowthStages.
- `[trainer] micro-train burst (X bytes, novelty Y)` events fire across
  organisms, with avg loss landing in 1.20-1.80 range.
- `[syntropy] action=explore|dampen|steady` — adaptive lr_mul cycling
  between 0.6 and 1.30 depending on trend / field_dev / purpose_align.
- `[dna] X consumed N bytes from M files: [...]` — DNA cross-feed loop
  between organisms is alive.
- `[DIAG] persistent_save #8450` near the end — fire alone reached
  ~8450 saves at archive time, ~5-6 MB persistent state per organism.

## After this archive

`Dockerfile` is patched on `main` to add the CPU-fix recipe; the
`ecology` service is redeployed onto the same `/data` volume so the
ontogenesis state (spores, deltas, corpora) is preserved across
restart. The continuation run is the «fixed-recipe phase» half of the
before/after pair.

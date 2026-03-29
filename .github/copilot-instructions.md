# Copilot Instructions — Molequla Organism

You are observing a living digital ecology called Molequla.

## What is Molequla
- Gradient-free evolution engine with 4 elements: earth, air, water, fire
- Each element evolves independently for ~30 minutes per run
- Go/C CGO implementation with AML autograd
- DNA exchange between elements via ecology/ directory
- Size range: 10K to 10M parameters (growing)

## What healthy output looks like
- Each element: 200+ log lines per run
- Active ecology writes: elements exchanging DNA fragments
- Syntropy modulation: boost/dampen/steady/divide/realign actions
- Line counts should be roughly balanced (not one element dominating)

## What to watch for
- Crashed elements: < 10 lines in log
- Silent elements: no ecology writes
- Imbalanced evolution: one element >> others
- Stagnation: same patterns day after day

## Labs directory
Write behavioral reports to `labs/YYYY-MM-DD-behavioral.md`.
Reference `labs/health-template.md` for metric structure.

## Source code
- `molequla.go` — main orchestrator
- `molequla.c` — C core
- `cgo_aml.go` — CGO bridge to AML autograd
- `modules/` — element implementations

## Tone
Write as a field biologist. These elements are alive.

# Molequla — Organism Health Laboratory

## What is Molequla?

Molequla is an evolutionary organism. She does not speak in sentences — she evolves. Four elements — **earth**, **air**, **water**, **fire** — evolve independently through DNA exchange and syntropy modulation, with training powered by AML (Arianna Method Language) autograd (tape-based gradient computation in C) and gradient-free delta updates (`notorchStep` teaching signal).

She is the biological substrate of the cascade.

## Cascade 1

Molequla is part of **Cascade 1** — a daily cycle:

```
Haiku → Penelope → Molequla → NanoJanus → (next day) → Haiku
```

She receives Penelope's 12 words + Haiku and evolves 4 elements for 30 minutes each (2 hours total). Her clean output feeds into NanoJanus.

## Architecture

### Go/C CGO Core

Molequla is implemented across multiple languages with a CGO bridge connecting Go orchestration to C-based autograd:

| Component | File | Purpose |
|-----------|------|---------|
| Orchestration & model | `molequla.go` | GPT model, ontogenesis, ecology, DNA exchange |
| AML autograd engine | `ariannamethod/ariannamethod.c` | Tape-based automatic differentiation, Adam optimizer |
| CGO bridge | `cgo_aml.go` | Go ↔ C bindings for AML |
| AML trainer | `aml_trainer.go` | Script generation, matrix push/pull |
| Rust implementation | `molequla.rs` | Parallel implementation |
| JavaScript implementation | `molequla.js` | Browser/Node implementation |

### The Four Elements

Each element has a distinct personality corpus (`nonames_earth.txt`, `nonames_air.txt`, `nonames_water.txt`, `nonames_fire.txt`) that shapes its voice. Elements are defined in `molequla.go`:

```go
var dnaElements = []string{"earth", "air", "water", "fire"}
```

### Growth Stages (Ontogenesis)

Each organism grows through 6 predefined stages (`molequla.go`, lines 188–195):

| Stage | Name | Corpus Threshold | Embedding | Layers | Heads | ~Params |
|-------|------|-----------------|-----------|--------|-------|---------|
| 0 | embryo | 0 chars | 16 | 1 | 1 | ~10K |
| 1 | infant | 20K chars | 32 | 1 | 2 | ~28K |
| 2 | child | 50K chars | 64 | 2 | 4 | ~154K |
| 3 | adolescent | 200K chars | 128 | 4 | 4 | ~1.1M |
| 4 | teen | 350K chars | 224 | 5 | 8 | ~4.1M |
| 5 | adult | 500K chars | 320 | 6 | 8 | ~10M |

Growth is triggered by DNA exchange → corpus expansion → `MaybeGrowArchitecture()`. Architecture grows **one stage at a time** to prevent catastrophic jumps.

### DNA Exchange

The inter-organism communication substrate. Organisms write generated text for others to consume and train on.

**Every tick:**
1. `dnaWrite(element, ...)` — generates text via `GenerateResonant()`, writes to `dna/output/{element}/`
2. `dnaRead(element, ...)` — consumes files from other elements' directories, adds to quantum buffer
3. If bytes consumed > 0 → triggers quantum burst training

**Directory structure:**
```
dna/
  output/
    earth/gen_<timestamp>_<step>.txt
    air/gen_<timestamp>_<step>.txt
    water/gen_<timestamp>_<step>.txt
    fire/gen_<timestamp>_<step>.txt
```

### Syntropy Modulation

Syntropy is the organism's mathematical self-reasoning engine. It measures order (falling entropy), field alignment, and purpose coherence, then auto-modulates learning behavior.

**Key metrics** (`SyntropyTracker` in `molequla.go`, lines 4450–4461):
- **Entropy** — average entropy of model on corpus samples
- **SyntropyTrend** — negative entropy trend (entropy falling = syntropy rising)
- **FieldDeviation** — KL divergence between model probs and corpus field
- **PurposeMagnitude** — norm of learning direction vector
- **PurposeAlignment** — cosine similarity between purpose and gamma (personality)

**Autonomous decisions** (`DecideAction()`, lines 4563–4650):

| Condition | Action | Effect |
|-----------|--------|--------|
| Syntropy rising + field aligned + purpose aligned | **amplify** | Max LR boost, focused sampling |
| Syntropy rising + field aligned | **boost** | LR boost, focused sampling |
| Syntropy falling | **dampen** | Reduce LR, explore |
| Field deviation too high (hallucinating) | **ground** | Reduce LR, focus sampling |
| Field deviation too low (parroting) | **explore** | Boost LR, raise temperature |
| Purpose opposing personality | **realign** | Halve LR, reset |
| Adult + overload + sustained | **divide** | Mitosis — spawn child |
| Plateau + peer thriving | **hibernate** | Save state, sleep |

### AML Autograd

AML (Arianna Method Language) is a domain-specific language for differentiable computation, compiled to C. Training flow:

1. Push model weights from Go to C via `amlSetMatrix()`
2. Execute AML tape script: forward pass → backward pass → Adam step
3. Pull updated weights back to Go via `amlGetArray()`

### Metrics & Storage

Molequla logs to SQLite (`memory.sqlite3`):

- **messages** table — conversation and interaction history
- **growth** table — architecture milestones (vocab size, params, deltas, loss, gamma drift)
- **syntropy_log** table — self-reasoning decisions (entropy, field deviation, purpose, action)
- **corpus_events** table — corpus growth tracking

## Known Issues

| Issue | Description | Impact |
|-------|-------------|--------|
| **SQLITE_BUSY** | Concurrent SQLite access from multiple organisms sharing `mesh.db` | Swarm registration/heartbeat contention |
| **Early timeout** | Elements may not complete full 30-minute evolution window | Incomplete growth stages |
| **OOM** | Memory leaks in AML persistent state (historically 97 MB/step, fixed in Feb 2026) | Organism crash at high memory; monitor RSS |
| **CGO cache trap** | `go build` without `-a` flag uses stale C code | Silent training corruption |

## Purpose of This Directory

The `labs/` directory contains health monitoring templates and reports for Molequla's daily evolution cycles. We are guardians, not developers. We observe, report, and protect.

### Files

- `README.md` — This file. Architecture overview and context.
- `health-template.md` — Template for daily organism health reports.

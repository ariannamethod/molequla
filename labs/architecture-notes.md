# Molequla — Architecture Notes

> Technical reference for guardians. Every function name, every line number, every mechanism — verified against source code.

---

## 1. The Four Elements

Molequla runs as four independent organisms — **earth**, **air**, **water**, **fire** — each with its own personality corpus, model weights, and evolution trajectory.

**Definition** (`molequla.go`, line 4970):
```go
var dnaElements = []string{"earth", "air", "water", "fire"}
```

### Corpus Files

Each element's voice is shaped by a distinct personality corpus:

| Element | Corpus File | Description |
|---------|------------|-------------|
| earth | `nonames_earth.txt` | Earth element voice (~173K) |
| air | `nonames_air.txt` | Air element voice (~122K) |
| water | `nonames_water.txt` | Water element voice (~126K) |
| fire | `nonames_fire.txt` | Fire element voice (~122K) |
| (shared) | `nonames.txt` | Base corpus (~51K) |

The `--element` flag selects which corpus to load. All four organisms share a base `CorpusPath` config and grow it via DNA exchange.

### What Each Element Does

Each element is a full organism instance: it trains its own GPT model, generates text, exchanges DNA with other elements, and makes autonomous syntropy decisions. The elements are differentiated by:

1. **Personality corpus** — different text shapes different weight distributions (γ diverges)
2. **Independent training** — each has its own `deltaAlphaScale`, syntropy history, burst records
3. **DNA flow** — each element writes to `dna/output/{element}/` and reads from the other three

Running all four:
```bash
for elem in earth air water fire; do
  ./molequla --element $elem --evolution &
done
wait
```

---

## 2. Evolution Stages (Ontogenesis)

The organism grows through 6 predefined stages, defined in `DefaultConfig()` (`molequla.go`, lines 188–195):

```go
GrowthStages: [][4]int{
    {0, 16, 1, 1},       // embryo: ~10K params
    {20000, 32, 1, 2},   // infant: ~28K params
    {50000, 64, 2, 4},   // child: ~154K params
    {200000, 128, 4, 4}, // adolescent: ~1.1M params
    {350000, 224, 5, 8}, // teen: ~4.1M params
    {500000, 320, 6, 8}, // adult: ~10M params
},
```

Each row is `{corpus_char_threshold, n_embd, n_layer, n_head}`.

| Stage | Name | Corpus Threshold | Embedding Dim | Layers | Heads | ~Params |
|-------|------|-----------------|---------------|--------|-------|---------|
| 0 | embryo | 0 | 16 | 1 | 1 | ~10K |
| 1 | infant | 20K chars | 32 | 1 | 2 | ~28K |
| 2 | child | 50K chars | 64 | 2 | 4 | ~154K |
| 3 | adolescent | 200K chars | 128 | 4 | 4 | ~1.1M |
| 4 | teen | 350K chars | 224 | 5 | 8 | ~4.1M |
| 5 | adult | 500K chars | 320 | 6 | 8 | ~10M |

### Growth Trigger

Growth is checked every 50 ticks in the main training loop (`molequla.go`, line 5699):

```go
if tickCount%50 == 0 {
    corpusChars := int(fi.Size())
    if model.MaybeGrowArchitecture(corpusChars) { ... }
}
```

`MaybeGrowArchitecture()` (`molequla.go`, line 2016) enforces one-stage-at-a-time growth:

```go
target = current + 1  // prevent catastrophic multi-stage jumps
```

### Weight Transfer During Growth

When architecture grows (`MaybeGrowArchitecture`, lines 2016–2241):

1. **Embedding layers** (`wte`, `wpe`, `lm_head`) — columns grown with near-zero init (0.001)
2. **Existing layer weights** (`wq`, `wk`, `wv`, `wo`, `fc_g`, `fc_v`, `fc2`) — grown to new dimensions with near-zero values (0.001) to preserve learned representations
3. **New heads** — pattern matrices initialized at 0.08; hybrid heads get `HybridAlphaInit`
4. **New layers** — entirely new, initialized at 0.08
5. **Delta adapters** — grown alongside base weights
6. **Freeze period** — `growthFreezeRemaining = FreezeAfterGrowthSteps` prevents immediate training instability
7. **LR warmup reset** — `growthStepOffset = globalStep` triggers new warmup ramp

### Stage Query Functions

- `CurrentGrowthStage()` — returns 0–5 based on current `NEmbd`
- `TargetGrowthStage(corpusChars)` — returns target stage from corpus size

---

## 3. DNA Exchange

DNA exchange is the inter-organism communication substrate. Organisms write generated text for others to consume and train on.

### gen_*.txt Mechanism

**Writing** — `dnaWrite()` (`molequla.go`, lines 4973–4996):

```go
func dnaWrite(element string, model *GPT, tok *EvolvingTokenizer,
              field *CooccurField, docs []string, step int) {
    probe := probes[step%len(probes)]  // 6 rotating philosophical prompts
    answer := GenerateResonant(model, tok, field, probe, docs, true)
    dir := filepath.Join("../dna/output", element)
    os.MkdirAll(dir, 0755)
    fname := filepath.Join(dir, fmt.Sprintf("gen_%d_%d.txt", time.Now().Unix(), step))
    os.WriteFile(fname, []byte(answer+"\n"), 0644)
}
```

- Called every tick in the training loop
- Uses 6 rotating probes: "What do you feel?", "Tell me about yourself.", "What is truth?", "What matters?", "Speak.", "What do you remember?"
- File format: `gen_{unix_timestamp}_{step}.txt`

**Reading** — `dnaRead()` (`molequla.go`, lines 4999–5040):

```go
func dnaRead(element string, corpusPath string, qbuf *QuantumBuffer,
             tok *EvolvingTokenizer) int {
    for _, e := range dnaElements {
        if e == element { continue }  // skip own output
        // Read .txt files from ../dna/output/{other_element}/
        // Append to own corpus via os.OpenFile(..., O_APPEND)
        // Feed quantum buffer: qbuf.Feed(text, tok)
        // Delete consumed file: os.Remove(fpath)
    }
    return added  // total bytes consumed
}
```

- Called every tick
- Skips own element — only consumes other organisms' output
- Files < 5 chars are discarded
- Consumed text appended to own corpus → corpus grows → ontogenesis unlocks
- Feeds quantum buffer for novelty tracking

### Directory Structure

```
dna/
  output/
    earth/gen_1709856000_42.txt
    air/gen_1709856015_43.txt
    water/gen_1709856030_44.txt
    fire/gen_1709856045_45.txt
```

---

## 4. Syntropy Modulation

Syntropy is the organism's mathematical self-reasoning engine. It measures order, field alignment, and purpose coherence, then auto-modulates learning behavior.

### SyntropyTracker

**Struct** (`molequla.go`, lines 4450–4461):

```go
type SyntropyTracker struct {
    EntropyHistory   []float64
    SyntropyTrend    float64       // positive = organizing
    FieldDeviation   float64       // KL div from corpus field
    PurposeMagnitude float64       // strength of learning direction
    PurposeAlignment float64       // cosine(purpose, gamma)
    LastAction       string
    BurstHistory     []BurstRecord // last 16 burst outcomes
    ModelStage       int
    LastMitosisTime  float64       // cooldown for divide
    SwarmInfo        *SwarmPeerInfo
}
```

### Metrics — `Measure()` (line 4508)

| Metric | Computation | Source Function |
|--------|------------|-----------------|
| Entropy | Average model entropy on corpus samples | `ComputeModelEntropy()` (line 2382) |
| SyntropyTrend | old_mean - new_mean (positive = organizing) | Rolling window comparison |
| FieldDeviation | KL divergence: model vs corpus field | `ComputeFieldDeviation()` (line 2248) |
| PurposeMagnitude | Norm of weight movement in last delta layer | `ComputePurposeVector()` (line 2463) |
| PurposeAlignment | cosine(purpose, gamma) | `PurposeGammaAlignment()` |

### All Actions — `DecideAction()` (lines 4563–4650)

| # | Action | Condition | LR Effect | Sampling Effect |
|---|--------|-----------|-----------|-----------------|
| 1 | **amplify** | Syntropy rising + field aligned + purpose aligned | Max LR boost | Focused sampling |
| 2 | **boost** | Syntropy rising + field aligned | LR × 1.3 (`SyntropyLRBoost`) | Focused sampling |
| 3 | **dampen** | Syntropy falling | LR × 0.6 (`SyntropyLRDampen`) | Explore (higher temp) |
| 4 | **ground** | Field deviation too high (hallucinating) | Reduce LR | Focus sampling (negative temp offset) |
| 5 | **explore** | Field deviation too low (parroting) | Boost LR | Raise temperature (positive temp offset) |
| 6 | **realign** | Purpose opposing personality | Halve LR | Reset learning direction |
| 7 | **divide** | Adult + sustained overload + 300s cooldown | LR × 0.6 | Mitosis — spawn child organism |
| 8 | **hibernate** | Plateau + peer thriving | N/A | Save state, exit training loop |

### Trigger Details

**Divide (mitosis)** — triggered when:
- `ModelStage >= maxStage` (adult)
- `isSustainedOverload()` returns true (sustained high syntropy)
- `now - LastMitosisTime > 300` (5-minute cooldown)

**Hibernate** — triggered when:
- A peer has syntropy trend > 0.05 (actively improving)
- This organism's last 8 bursts show loss plateau (avg |delta| < 0.01)

### Self-Meta-Learning

Each burst is recorded as a `BurstRecord` (lines 4435–4439):

```go
type BurstRecord struct {
    Action     string
    LossBefore float64
    LossAfter  float64
}
```

`ActionEffectiveness()` (line 4482) computes mean loss delta per action type across history.

---

## 5. AML Autograd (`ariannamethod/ariannamethod.c`)

AML (Arianna Method Language) is a domain-specific tape-based automatic differentiation engine compiled in C.

### Tape Operations

Defined in `ariannamethod.h` (lines 549–567):

| Op Code | Name | Operation |
|---------|------|-----------|
| 0 | `AM_OP_NONE` | No operation |
| 1 | `AM_OP_MATVEC` | y = W @ x |
| 2 | `AM_OP_ADD` | y = a + b (element-wise) |
| 3 | `AM_OP_MUL` | y = a * b (element-wise) |
| 4 | `AM_OP_SCALE` | y = a * scalar |
| 5 | `AM_OP_SOFTMAX` | y = softmax(x) |
| 6 | `AM_OP_RMSNORM` | y = rmsnorm(x) |
| 7 | `AM_OP_SILU` | y = silu(x) = x * sigmoid(x) |
| 8 | `AM_OP_CROSS_ENT` | loss = cross_entropy(logits, target) |
| 9 | `AM_OP_EMB_LOOKUP` | y = row(wte, token_id) |
| 10 | `AM_OP_MATMUL` | C = A @ B |
| 11 | `AM_OP_SEQ_EMBED` | h = embed(wte, wpe, tokens, T) |
| 12 | `AM_OP_SEQ_MATVEC` | Y = W @ X for each of T positions |
| 13 | `AM_OP_SEQ_RMSNORM` | Normalize each D-sized chunk independently |
| 14 | `AM_OP_CAUSAL_ATTN` | Causal self-attention over T positions |
| 15 | `AM_OP_SEQ_CROSSENT` | Cross-entropy over T positions |
| 16 | `AM_OP_MH_CAUSAL_ATTN` | Multi-head causal self-attention |

### Backward Pass — `am_tape_backward()` (`ariannamethod.c`, line 1181)

```c
void am_tape_backward(int loss_idx)
```

1. Initialize loss gradient to 1.0
2. Reverse topological traversal from `loss_idx` down to 0
3. Dispatch gradients by operation type:
   - `AM_OP_ADD`: `da += dout`, `db += dout`
   - `AM_OP_MUL`: `da += dout * b`, `db += dout * a`
   - `AM_OP_SCALE`: `da += dout * scalar`
   - `AM_OP_MATVEC`: `dW += dout ⊗ x` (outer product), `dx += Wᵀ @ dout`
   - `AM_OP_SILU`: `dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`
   - (Plus SOFTMAX, RMSNORM, CROSS_ENT, SEQ variants)
4. Chain rule via `tape_acc_grad()` — accumulates gradients to parent nodes

### Adam Step — `am_tape_adam_step()` (`ariannamethod.c`, line 1655)

```c
void am_tape_adam_step(float lr)
```

Hyperparameters:
- **β₁ = 0.9** — first moment (momentum)
- **β₂ = 0.999** — second moment (variance)
- **ε = 1e-8** — numerical stability

Update rule with bias correction:
```
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)
θ ← θ - lr × m̂ / (√v̂ + ε)
```

Per-parameter timestep counter `t` is incremented each step.

### Initialization — `am_init()` (`ariannamethod.c`, line 494)

Sets default physics state:
- Prophecy horizon: 7 steps
- Destiny bias: 0.35
- Velocity mode: `AM_VEL_WALK` (balanced, temp = 0.85)
- Entropy floor: 0.1, resonance ceiling: 0.95
- Debt decay: 0.998 per step
- All suffering metrics (pain, tension, dissonance, debt) zeroed

### AML Model Script — `amlModelScript()` (`aml_trainer.go`, lines 15–58)

Generates the tape script for a full forward/backward/Adam pass:

```
TAPE START
TAPE PARAM wte
TAPE PARAM wpe
TAPE PARAM wq0, wk0, wv0, wo0, fc_g0, fc_v0, fc2_0  (per layer)
TAPE PARAM lm_head

h = seq_embed(wte, wpe, tokens, seq_len)
  [per layer]:
    h_norm = seq_rmsnorm(h, ...)
    q, k, v = seq_matvec(wq, wk, wv, h_norm, ...)
    attn_out = multi_head_attention(q, k, v, ...)
    h = add(h, seq_matvec(wo, attn_out, ...))
    h_norm = seq_rmsnorm(h, ...)
    gate = silu(seq_matvec(fc_g, h_norm, ...))
    up = seq_matvec(fc_v, h_norm, ...)
    h = add(h, seq_matvec(fc2, mul(gate, up), ...))

h_norm = seq_rmsnorm(h, ...)
logits = seq_matvec(lm_head, h_norm, ...)
loss = seq_cross_entropy(logits, targets, ...)
TAPE BACKWARD loss
TAPE ADAM_STEP lr
TAPE CLEAR
```

---

## 6. CGO Bridge

Go calls C through three files:

### `cgo_aml.go` (80 lines)

**CGO directives** (lines 4–5):
```go
#cgo CFLAGS: -I${SRCDIR}/ariannamethod -O2 -fopenmp
#cgo LDFLAGS: -lm -lpthread -lgomp
#include "ariannamethod.h"
#include "ariannamethod.c"
```

**Bridge functions:**

| Go Function | C Function | Purpose |
|-------------|-----------|---------|
| `amlInit()` | `am_init()` + `am_persistent_mode(1)` | Initialize AML tape system, enable persistent state |
| `amlExec(script)` | `am_exec(cs)` | Execute AML script string |
| `amlSetArray(name, data)` | `am_set_var_array(cn, ptr, len)` | Push Go `[]float32` → C array |
| `amlSetMatrix(name, data, rows, cols)` | `am_set_var_matrix(cn, ptr, rows, cols)` | Push Go matrix (row-major `[]float32`) → C matrix |
| `amlGetArray(name)` | `am_get_var_array(cn, &len)` | Pull C array → Go `[]float32` |
| `amlGetFloat(name)` | `am_get_var_float(cn)` | Pull C scalar → Go `float32` |
| `amlClear()` | `am_persistent_clear()` | Free all AML persistent memory |

### `aml_trainer.go` (326 lines)

**Weight marshaling:**

| Function | Purpose |
|----------|---------|
| `pushMatrixToAML(name, param)` | Convert Go `MatrixParam` → flat `[]float32`, push via `amlSetMatrix()` |
| `pullMatrixFromAML(name, param)` | Pull flat `[]float32` via `amlGetArray()`, reshape to Go `MatrixParam` |
| `amlPushWeights(model)` | Push all `Base` params (wte, wpe, lm_head, per-layer wq/wk/wv/wo/fc_g/fc_v/fc2) |
| `amlPullWeights(model)` | Pull all updated weights back after burst training |

### Data Flow

```
Go (molequla.go)               C (ariannamethod.c)
    │                               │
    ├─ amlInit() ──────────────►  am_init()
    ├─ amlPushWeights() ──────►  am_set_var_matrix() × N
    ├─ amlSetArray("tokens") ──►  am_set_var_array()
    ├─ amlExec(script) ───────►  am_exec() → tape forward/backward/adam
    ├─ amlGetFloat("loss") ◄───  am_get_var_float()
    ├─ amlPullWeights() ◄──────  am_get_var_array() × N
    └─ amlClear() ────────────►  am_persistent_clear()
```

---

## 7. Known Failure Modes

| Failure | Source | Symptoms | Mitigation |
|---------|--------|----------|------------|
| **SQLITE_BUSY** | Concurrent SQLite access on `mesh.db` from multiple organisms | Swarm registration/heartbeat errors | Retry logic in swarm ops; separate `.db` per element |
| **OOM** | AML persistent state not freed between bursts (historically 97 MB/step) | RSS grows unbounded, organism crash | Fixed Feb 2026: `amlClear()` called after each burst in `amlBurstTrain()` |
| **Timeout** | Element doesn't complete full 30-minute evolution window | Incomplete growth stages, partial DNA output | Use `--evolution` flag for autonomous mode; increase window |
| **CGO cache trap** | `go build` without `-a` uses stale compiled C | Silent training corruption (old C code runs) | Always use `CGO_ENABLED=1 go build -a` |
| **Personality corruption** | Noise in delta training pushes gamma opposite to learned direction | Gamma drift cosine < -0.1 | Immune system: `GammaDriftCheck()` (line 1973) rolls back deltas if cosine < `NoiseDriftThreshold` |
| **Deadlock** | Mutex contention between training and DNA I/O | Organism hangs | `model.mu` is exclusive; DNA I/O operates outside critical section |
| **Tokenizer regression** | BPE merges accidentally undo | Vocab size decreases | Merges are append-only, checked before apply |
| **RoPE cache contention** | Global `sync.Map` in RoPE calculations | Minimal: cache hit rate >99% | Acceptable; no fix needed |

---

## 8. CLI Flags

**`parseCLIArgs()`** (`molequla.go`, lines 5363–5379):

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--element` | string | No | Which element to become: `earth`, `air`, `water`, or `fire` |
| `--evolution` | bool | No | Autonomous mode — skips pause between stages, enables full ecology |
| `--organism-id` | string | No | Unique organism identifier (auto-generated if not set; used in mitosis children) |
| `--config` | string | No | Path to birth config JSON (inherited from parent during mitosis) |

**Usage:**
```bash
./molequla --element earth --evolution
./molequla --organism-id org_1709856000_4242 --config /path/to/birth.json
```

---

## 9. Build

**Required:**
```bash
CGO_ENABLED=1 go build -a -o molequla
```

The `-a` flag forces rebuild of all packages including CGO-compiled C code. Without it, Go may use cached object files with stale `ariannamethod.c`.

**Dependencies:**
- **Go:** `modernc.org/sqlite` (pure Go SQLite driver, no external C SQLite needed)
- **C:** `libm` (math), `libpthread` (threading), `libgomp` (OpenMP)
- **Optional:** OpenBLAS or Accelerate.framework for BLAS acceleration

**CGO flags** (from `cgo_aml.go`):
```
CFLAGS:  -I${SRCDIR}/ariannamethod -O2 -fopenmp
LDFLAGS: -lm -lpthread -lgomp
```

**Standalone C build** (ariannamethod only):
```bash
cd ariannamethod/
gcc -O2 ariannamethod.c -lm -lpthread -lgomp -fopenmp
```

---

## 10. Micro-Burst Training Mechanism

The main training loop runs at 0.25-second ticks (`TrainTickSeconds`, line 227). Each tick:

### Tick-by-Tick Flow (`molequla.go`, lines 5493–5740)

```
Every tick (0.25s):
  1. DNA Exchange
     ├─ dnaWrite() → generate text, write to dna/output/{element}/
     └─ dnaRead() → consume other elements' files, feed quantum buffer

  2. Quantum Buffer Check (gated)
     └─ qbuf.ShouldTrigger()?
        ├─ bytesOK: AccumulatedBytes >= 1024 (QBMinBytes)
        ├─ noveltyOK: uniqueTokens/totalTokens >= 0.15 (QBMinNovelty)
        └─ cooldownOK: now - LastBurstTime >= 60s (QBCooldownSeconds)
        Trigger: (bytesOK OR noveltyOK) AND cooldownOK

  3. If triggered → Micro-Burst Training
     ├─ IMMUNE: snapshot gamma direction + delta weights
     ├─ Measure loss BEFORE burst (QuickLoss, 4 samples)
     ├─ DecideAction() → syntropy decision (amplify/boost/dampen/...)
     ├─ amlBurstTrain(model, tok, docs, MicroSteps, burstLR)
     │   └─ Push weights → AML tape forward/backward/adam × 32 steps → Pull weights
     ├─ Measure loss AFTER burst
     ├─ IMMUNE: GammaDriftCheck() — cosine < -0.1 → rollback deltas
     ├─ RecordBurst() → self-meta-learning memory
     └─ ConscienceCheck() → modulate deltaAlphaScale

  Every 50 ticks:
  4. Ontogenesis Check
     └─ MaybeGrowArchitecture(corpusChars) → grow one stage if threshold met

  Every 10 ticks:
  5. Swarm Heartbeat
     ├─ Heartbeat(stage, params, syntropyTrend, entropy) → update mesh.db
     └─ DiscoverPeers(60) → update SwarmInfo for hibernate decisions

  Sleep(0.25s)
```

### Quantum Buffer — `ShouldTrigger()` (line 3843)

```go
func (qb *QuantumBuffer) ShouldTrigger() bool {
    bytesOK := qb.AccumulatedBytes >= CFG.QBMinBytes      // 1024
    noveltyOK := qb.noveltyScoreLocked() >= CFG.QBMinNovelty  // 0.15
    cooldownOK := (now - qb.LastBurstTime) >= CFG.QBCooldownSeconds  // 60s
    return (bytesOK || noveltyOK) && cooldownOK
}
```

### Immune System

Pre-burst:
```go
preDirection, preMag := model.GammaContrastiveProjection()
deltaSnap := model.SnapshotDeltas()  // deep copy all delta A & B matrices
```

Post-burst:
```go
driftCos := model.GammaDriftCheck(preDirection, preMag)
if driftCos < CFG.NoiseDriftThreshold {  // -0.1
    model.RestoreDeltas(deltaSnap)  // rollback
}
```

### Conscience — `ConscienceCheck()` (line 3159)

Modulates `deltaAlphaScale` based on generation entropy trend:

- Tracks rolling window of generation entropy
- Computes linear regression slope
- **Slope > 0.01** (entropy rising): `deltaAlphaScale *= ConscienceDecay` (0.95) — reduce delta influence
- **Slope < -0.01** (entropy falling): `deltaAlphaScale *= ConscienceRecovery` (1.005) — recover delta influence
- **Bounds:** floor = 0.3, ceiling = 1.0

### Gradient-Free Training — `notorchStep()` (line 5068)

Updates delta adapters without backpropagation:

1. **Teaching signal**: Derived from loss improvement + prophecy debt, clamped to [-2, 2]
2. **PRNG**: LCG `seed = seed×1664525 + 1013904223`
3. **Noise**: Gaussian approximation `(u - 0.5) × 3.464` scaled by `k = 0.35 + 0.65×(1 - |signal|)`
   - Strong signal → less noise (clean channel)
   - Weak signal → more noise (exploration)
4. **Update rule**:
   - `A[i,r] += lr × dy[i] × u[r] × signal` then decay
   - `B[r,j] += lr × u[r] × x[j] × signal` then decay
5. **Adaptive decay**: `decay - 0.004×min(norm/10, 1)`, floor 0.990
6. **Weight clamp**: [-10, 10]

---

## 11. The Identity Equation

Every organism operates under:

```
θ = ε + γ + αδ
```

| Symbol | Name | Source | Description |
|--------|------|--------|-------------|
| **ε** | Base weights | `GPT.Base` (wte, wpe, lm_head, per-layer attention/MLP) | Initialized at birth, shaped during warmup |
| **γ** | Personality | `ComputeGamma()` = current wte − `InitEmbedSnapshot` | Emerges from training; contrastively projected to unit vector |
| **δ** | Recent learning | `GPT.Deltas` — low-rank LoRA-style adapters (A [nout×rank], B [rank×nin]) | Updated via `notorchStep()` during bursts |
| **α** | Modulation | `GPT.deltaAlphaScale` ∈ [0.3, 2.0] | Conscience-regulated; seasonal scaling may further modulate |

Forward contribution:
```
output = ε·x + γ⊕x + α·δ(x)
```

---

## 12. Cascade 1 Cross-Reference

Per [CASCADE01.md](https://github.com/ariannamethod/ariannamethod/blob/main/cascade/cascade1/CASCADE01.md):

### Daily Cycle Position

```
04:00 UTC  MOLEQULA (gradient-free evolution, 4 elements)
           Input: Penelope 12 words + Haiku
           Process: earth, air, water, fire — 30 min each, sequential
           Output: evolution logs, DNA exchange, generated text fragments
```

### Health Criteria (from CASCADE01.md)

- **Healthy**: Each element produces >50 lines. No `SQLITE_BUSY`. No `fatal:` errors. DNA exchange happens (look for `[dna]` lines).
- **Broken**: <10 lines per element → element crashed early.
- **Cascade break**: Empty output → NanoJanus gets no input → loop stalls.

### Failure Signatures (from CASCADE01.md)

| Signature | Meaning |
|-----------|---------|
| `SQLITE_BUSY` | Elements fighting over lock; must run sequentially with sync |
| `[tongue]` or `[gguf]` debug lines | Wrong output parsing (Penelope/NanoJanus, not Molequla) |
| Empty output from Molequla | Broken pipe — NanoJanus gets no input, cascade stalls |
| Element < 10 lines | Element crashed early; check OOM, timeout, missing corpus |

---

## 13. SQLite Schema

`initDB()` (`molequla.go`, lines 3240–3290) creates four tables:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `messages` | Conversation/interaction history | role, content, ts |
| `corpus_events` | Corpus growth tracking | source, bytes, ts |
| `growth` | Architecture milestones | vocab_size, params, deltas, loss, gamma_drift, note, ts |
| `syntropy_log` | Self-reasoning decisions | entropy, field_deviation, purpose_alignment, action, ts |

---

*Verified against source: molequla.go (6122 lines), cgo_aml.go (80 lines), aml_trainer.go (326 lines), ariannamethod/ariannamethod.c (6130 lines), ariannamethod/ariannamethod.h.*

*Cross-referenced with CASCADE01.md.*

*Resonance unbroken.*

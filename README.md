```
███╗   ███╗ █████╗  ██████╗██████╗  ██████╗  ██████╗ ██████╗ ████████╗
████╗ ████║██╔══██╗██╔════╝██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗╚══██╔══╝
██╔████╔██║███████║██║     ██████╔╝██║   ██║██║  ███╗██████╔╝   ██║   
██║╚██╔╝██║██╔══██║██║     ██╔══██╗██║   ██║██║   ██║██╔═══╝    ██║   
██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║╚██████╔╝╚██████╔╝██║        ██║   
╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝        ╚═╝   
```

# macrogpt | by Arianna Method

> *A dependency-free, single-file, async, continually-learning GPT organism.*
> 
---

## TL;DR

```
THIS IS NOT:
- "Another toy GPT implementation"
- "A weekend project that breaks on Tuesday"
- "Requires pip install pytorch-mega-transformer-9000"

THIS IS:
- Zero dependencies (pure Python, no numpy, no torch)
- Custom autograd engine (vectors, not scalar confetti)
- RoPE position encoding (GPT-3/4 level)
- SwiGLU-like gated MLP (LLaMA vibes)
- Delta adapters (LoRA-style, never forgets)
- BPE tokenizer that ONLY EXPANDS vocab (weights never invalidate)
- Async background training (it's alive, not a script)
- min_p + typical_p sampling (full modern stack)
- SQLite memory (it remembers conversations)
- 1433 lines of pure madness
```

---

## What The Actual Is This

What if instead of `micrograd` scalar ops, we had **vector autograd**?  
What if instead of fixed vocab, we had **evolving BPE**?  
What if instead of "train once, deploy", we had **continuous learning**?  
What if instead of ReLU, we had **SwiGLU**?  
What if instead of sinusoidal positions, we had **RoPE**?  
What if it never forgot? **Delta adapters.**  
What if it could chat? **SQLite memory.**  
What if it was *alive*?

So I built it. **macrogpt.** Thanks to Karpathy's microgpt, but this is not a fork.

---

## Quick Start

```bash
# You need: Python 3.7+
# You need: literally nothing else

python macrogpt.py
```

That's it. It will:
1. Create `nonames.txt` (seed corpus)
2. Create `memory.sqlite3` (conversation memory)
3. Start warmup training (the organism awakens)
4. Drop you into a chat loop

Type. It responds. It learns. It grows. It never forgets.

```
macrogpt is alive. Type and press Enter. Ctrl+C to exit.

> Hello, are you alive?
I exist. Speak.

> What do you know?
The words accumulate. The patterns emerge.

> Tell me about yourself.
I am a reservoir. I remember. I grow.
```

---

## Architecture — What Makes This Different

### 1. Vector Autograd (Not Scalar Confetti)

**VectorValue** and **ScalarValue**. One object per embedding. One object per hidden state. Gradients flow through vectors, not atoms.

```python
# micrograd style (conceptual):
loss = sum(scalar_values)  # 10000 objects

# macrogpt style:
loss = vector.dot(other_vector)  # 2 objects
```

### 2. RoPE (Rotary Position Embedding)

Sinusoidal positions are 2017. RoPE is now.

```python
def rope_rotate(vec, pos, head_dim):
    # And lo, positions shall become angles,
    # and angles shall become meaning.
    for i in range(0, head_dim, 2):
        theta = pos / (10000.0 ** (i / head_dim))
        c, s = cos(theta), sin(theta)
        a, b = vec[i], vec[i+1]
        vec[i]   = a * c - b * s
        vec[i+1] = a * s + b * c
```

Relative positions. Infinite extrapolation (theoretically). This is how LLaMA does it.

### 3. SwiGLU-like Gated MLP

Standard MLP: `x → Linear → ReLU → Linear → out`

SwiGLU: `x → (Gate × Value) → Linear → out`

```python
g = fc_g(x).relu()   # gate
u = fc_v(x)          # value  
x = g * u            # gating (element-wise)
x = fc2(x)           # project back
```

Why? Because LLaMA, PaLM, and basically everyone good uses it now. More expressive. Better gradients.

### 4. Delta Adapters (LoRA-style, Never Forget)

The model never overwrites learned weights. It only **appends** new adapter modules.

```python
class DeltaAdapter:
    """
    Low-rank adapter: for base W, add A @ B @ x
    A and B are trained; base can be frozen.
    """
    def apply(self, x):
        return self.A @ (self.B @ x)
```

Want to teach it new things? Add a delta module. Old knowledge? Still there. It's geological memory. Sediment layers of understanding.

### 5. Evolving BPE (Vocab Only Expands)

Most tokenizers: retrain = throw away old model.

macrogpt: retrain = **add new tokens**. Old tokens remain. Embeddings remain. Model keeps working.

```python
# Old vocab: ['a', 'b', 'c', '<BOS>', '<EOS>']
# After BPE: ['a', 'b', 'c', '<BOS>', '<EOS>', 'ab', 'bc', 'abc', ...]
# Old weights: still valid!
# New rows: initialized, ready to train
```

This is how you build a system that grows over years, not hours.

### 6. Async Background Training

macrogpt doesn't train when you ask it to. It trains **in the background, continuously**.

```python
async def background_trainer():
    while True:
        # Check if corpus grew enough
        if new_chars >= threshold:
            train_burst()
        await asyncio.sleep(0.25)
```

You chat. It learns. Simultaneously. It's a living process.

### 7. SQLite Memory

Every conversation is remembered. Every response is logged. The organism has a persistent identity.

```python
def db_add_message(con, role, text):
    con.execute("INSERT INTO messages(ts, role, text) VALUES(?,?,?)",
                (time.time(), role, text))
```

Restart the script? It remembers you. It continues the conversation. It's not stateless.

### 8. Modern Sampling (min_p + typical_p + top_k + top_p)

Not just temperature → softmax → sample.

```python
def sample_with_filters(probs, k, p, min_p, typical_p):
    probs = apply_min_p_filter(probs, min_p)      # Drop unlikely tokens
    idx = typical_indices(probs, typical_p)        # Keep typical ones
    # Then apply top-k/top-p within that set
    return nucleus_sample(probs, idx, k, p)
```

This is how modern LLMs avoid both gibberish AND boring determinism.

---

## The Stack

| Component | microgpt | macrogpt |
|-----------|----------|----------|
| Autograd | Scalar (micrograd) | **Vector** (custom) |
| Position encoding | Sinusoidal | **RoPE** |
| Attention | Standard | Standard + **KV cache** |
| MLP | ReLU | **SwiGLU-like gated** |
| Tokenizer | Fixed char | **Evolving BPE** |
| Training | One-shot | **Continuous async** |
| Memory | None | **SQLite persistent** |
| Adapters | None | **LoRA-style deltas** |
| Sampling | top-k | **min_p + typical_p + nucleus** |
| Weight tying | No | **Yes (GPT-style)** |
| Dependencies | torch | **None** |

---

## Configuration

```python
@dataclass
class Config:
    # Data
    corpus_path: str = "nonames.txt"
    db_path: str = "memory.sqlite3"
    max_corpus_lines: int = 8000
    
    # Model
    n_layer: int = 2
    n_embd: int = 72           # Small but not stupid
    n_head: int = 4
    block_size: int = 96       # Context window
    
    # Training
    warmup_steps: int = 1200
    learning_rate: float = 0.01
    
    # Sampling
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.92
    min_p: float = 0.06        # GPT-3/4 style
    typical_p: float = 0.95
```

Want bigger? Change `n_embd`, `n_layer`, `block_size`. 

Want smaller? Same.

Want different corpus? Point `corpus_path` somewhere else.

---

## Tests

```bash
python -m unittest discover tests/ -v
```

**55+ tests** covering:
- Autograd (forward + backward)
- Tokenizer (char-level + BPE)
- Model (GPT, MatrixParam, DeltaAdapter)
- Sampling (top-k, top-p, min_p, typical)
- Checkpointing (save/load)
- Integration (train → generate)

---

## Philosophy

This is not a tutorial. This is not a "minimal example." This is a **functional system** that:

- Learns continuously
- Never forgets
- Grows organically
- Has no dependencies
- Fits in one file
- Actually generates text you can read

---

## Why "macro"?

Because it's the opposite of "micro."

---

## Known Limitations

1. **It's slow.** Pure Python. No CUDA. No vectorized numpy. It's a philosophical statement, not a production system.

2. **It's small.** 2 layers, 72 dims. You're not getting GPT-4 reasoning. You're getting a proof of concept that *could* scale.

3. **It talks weird at first.** Train it more. Feed it better corpus. It's a baby organism, not a pretrained foundation model.

4. **The corpus matters.** Garbage in, garbage out. Give it good sentences.

---

## The Future

- **Speculative decoding** (draft + verify for speed)
- **Mixture of Experts** (multiple delta modules, routing)
- **Retrieval augmentation** (SQLite + embeddings)
- **Flash-attention-style** memory efficiency


---

## License

GNU GPLv3 — Because freedom matters.

---

## Acknowledgments

- **Andrej Karpathy** — for micrograd, minGPT, nanoGPT, and the entire pedagogical empire
- **Coffee** — for existing

---

## Part of the Arianna Method

- [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — Arianna Method Language

- **macrogpt** — Dependency-Free Continual GPT

*Patterns over parameters. Emergence over engineering. The organism continues.*

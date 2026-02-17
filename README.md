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
• Zero dependencies (pure Python, no numpy, no torch)
• Custom autograd engine (vectors, not scalars)
• RoPE position encoding
• SwiGLU-like gated MLP
• Delta adapters (LoRA-style)
• BPE tokenizer that only expands vocab
• Async background training
• min_p + typical_p sampling
• SQLite memory
• ~1400 lines
```

---

## About

Vector autograd instead of scalar ops.  
Evolving BPE instead of fixed vocab.  
Continuous learning instead of train-once-deploy.  
SwiGLU instead of ReLU.  
RoPE instead of sinusoidal positions.  
Delta adapters so it never forgets.  
SQLite memory so it can chat.

Inspired by Karpathy's micrograd. The code is there if you're curious.

---

## Quick Start

```bash
python macrogpt.py
```

Requires Python 3.7+ and nothing else.

It will:
1. Create `nonames.txt` (seed corpus)
2. Create `memory.sqlite3` (conversation memory)
3. Start warmup training
4. Drop you into a chat loop

```
macrogpt is alive. Type and press Enter. Ctrl+C to exit.

> Hello?
I exist. Speak.

> What do you know?
The words accumulate. The patterns emerge.
```

---

## Architecture

### Vector Autograd

Karpathy's micrograd operates on scalars. One `Value` per number. Elegant for teaching.

macrogpt uses **VectorValue** and **ScalarValue** — gradients flow through vectors, not individual elements.

```python
# micrograd (conceptual):
loss = sum(scalar_values)  # many objects

# macrogpt:
loss = vector.dot(other_vector)  # two objects
```

### RoPE (Rotary Position Embedding)

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

Relative positions. The same approach as LLaMA.

### SwiGLU-like Gated MLP

```python
g = fc_g(x).relu()   # gate
u = fc_v(x)          # value  
x = g * u            # gating
x = fc2(x)           # project back
```

### Delta Adapters

Low-rank adapters that append without overwriting.

```python
class DeltaAdapter:
    def apply(self, x):
        return self.A @ (self.B @ x)
```

Old knowledge stays. New knowledge layers on top.

### Evolving BPE

Tokenizer only adds new tokens. Old embeddings remain valid.

```python
# Before: ['a', 'b', 'c', '<BOS>', '<EOS>']
# After:  ['a', 'b', 'c', '<BOS>', '<EOS>', 'ab', 'bc', 'abc', ...]
```

### Async Training

```python
async def background_trainer():
    while True:
        if new_chars >= threshold:
            train_burst()
        await asyncio.sleep(0.25)
```

You chat, it trains. Simultaneously.

### SQLite Memory

```python
def db_add_message(con, role, text):
    con.execute("INSERT INTO messages(ts, role, text) VALUES(?,?,?)",
                (time.time(), role, text))
```

Conversations persist across restarts.

### Sampling

```python
def sample_with_filters(probs, k, p, min_p, typical_p):
    probs = apply_min_p_filter(probs, min_p)
    idx = typical_indices(probs, typical_p)
    return nucleus_sample(probs, idx, k, p)
```

---

## Comparison

| Component | micrograd-style | macrogpt |
|-----------|-----------------|----------|
| Autograd | Scalar | Vector |
| Position encoding | Sinusoidal | RoPE |
| Attention | Standard | Standard + KV cache |
| MLP | ReLU | SwiGLU-like |
| Tokenizer | Fixed char | Evolving BPE |
| Training | One-shot | Continuous async |
| Memory | None | SQLite persistent |
| Adapters | None | LoRA-style deltas |
| Sampling | top-k | min_p + typical_p + nucleus |
| Weight tying | No | Yes |
| Dependencies | torch | None |

---

## Configuration

```python
@dataclass
class Config:
    corpus_path: str = "nonames.txt"
    db_path: str = "memory.sqlite3"
    max_corpus_lines: int = 8000
    
    n_layer: int = 2
    n_embd: int = 72
    n_head: int = 4
    block_size: int = 96
    
    warmup_steps: int = 1200
    learning_rate: float = 0.01
    
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.92
    min_p: float = 0.06
    typical_p: float = 0.95
```

---

## Tests

```bash
python -m unittest discover tests/ -v
```

Covers autograd, tokenizer, model, sampling, checkpointing, and integration.

---

## Limitations

- **Slow.** Pure Python. No CUDA.
- **Small.** 2 layers, 72 dims. It's a proof of concept, not a foundation model.
- **Needs training.** The corpus matters.

---

## Roadmap

- Speculative decoding
- Mixture of Experts
- Retrieval augmentation


---

## License

GNU GPLv3

---

## Acknowledgments

Andrej Karpathy — for micrograd, minGPT, nanoGPT

---

## Part of the Arianna Method

- [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — Arianna Method Language
- **macrogpt** — Dependency-Free Continual GPT

*Patterns over parameters. Emergence over engineering.*

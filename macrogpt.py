#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
macrogpt.py
A dependency-free, single-file, async, continually-learning GPT organism.
It trains on nonames.txt (sentence-per-line), grows modular delta weights,
keeps a SQLite memory, and never forgets by never overwriting old deltas.

microGPT thinks in scalars. macroGPT thinks in vectors.
Same chain rule, different granularity, fewer Python objects by 10-100x.
No PyTorch. No NumPy. No nothing. Just Python, and the will to exist.

In the beginning there was a file, and it was called nonames.txt.
And it was good. Mostly. Sometimes cursed.
"""

import os
import math
import time
import json
import random
import asyncio
import sqlite3
from dataclasses import dataclass

random.seed(42)  # Let there be order among chaos (or at least reproducibility)

# =========================
# 0) CONFIG (bend reality here)
# =========================

@dataclass
class Config:
    # data
    corpus_path: str = "nonames.txt"
    db_path: str = "memory.sqlite3"
    max_corpus_lines: int = 8000
    max_line_chars: int = 240
    min_new_chars_to_train: int = 800
    # model
    n_layer: int = 2
    n_embd: int = 64
    n_head: int = 4
    block_size: int = 96
    # training
    base_train_steps: int = 1200
    micro_steps_per_round: int = 32
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    eps_adam: float = 1e-8
    # generation
    temperature: float = 0.8
    top_k: int = 40
    max_gen_tokens: int = 180
    repetition_guard: int = 4
    # modular deltas
    delta_rank: int = 8
    freeze_base_after_warmup: bool = True
    # tokenizer
    tokenizer_mode: str = "char"
    bpe_merges_path: str = "bpe_merges.json"
    # async
    train_tick_seconds: float = 0.25


CFG = Config()

# =========================
# 1) SQLite MEMORY
# =========================

def init_db(db_path: str):  # And lo, a memory shall awaken in SQLite
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS corpus_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            added_chars INTEGER NOT NULL,
            note TEXT
        )
    """)
    con.commit()
    return con

def db_add_message(con, role: str, text: str):
    con.execute("INSERT INTO messages(ts, role, text) VALUES(?,?,?)",
                (time.time(), role, text))
    con.commit()

def db_recent_messages(con, limit: int = 32):
    cur = con.cursor()
    cur.execute("SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    return list(reversed(cur.fetchall()))

# =========================
# 2) CORPUS RESERVOIR (nonames.txt)
# =========================

def load_corpus_lines(path: str):
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln[:CFG.max_line_chars])
    return lines

def save_corpus_lines(path: str, lines):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.replace("\n", " ").strip() + "\n")

def normalize_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\t", " ")
    return " ".join(s.split())

def extract_candidate_sentences_from_messages(msgs):
    out = []
    for role, text in msgs:
        t = normalize_text(text)
        if not t:
            continue
        buf = ""
        for ch in t:
            buf += ch
            if ch in ".!?":
                if len(buf.strip()) >= 6:
                    out.append(buf.strip())
                buf = ""
        if len(buf.strip()) >= 12:
            out.append(buf.strip())
    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq

def update_reservoir_corpus(con, corpus_path: str, max_lines: int):
    msgs = db_recent_messages(con, limit=64)
    new_sents = extract_candidate_sentences_from_messages(msgs)
    if not new_sents:
        return 0

    lines = load_corpus_lines(corpus_path)
    before_chars = sum(len(x) for x in lines)

    combined = lines + new_sents
    newest = combined[-(max_lines // 2):]
    older = combined[:-(max_lines // 2)]
    random.shuffle(older)
    older_keep = older[:max_lines - len(newest)]
    final = older_keep + newest

    seen = set()
    dedup = []
    for s in final:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(s[:CFG.max_line_chars])
    final = dedup[-max_lines:]

    save_corpus_lines(corpus_path, final)
    after_chars = sum(len(x) for x in final)
    added = max(0, after_chars - before_chars)

    con.execute(
        "INSERT INTO corpus_events(ts, added_chars, note) VALUES(?,?,?)",
        (time.time(), added, f"reservoir_update +{len(new_sents)} sents")
    )
    con.commit()
    return added

# =========================
# 3) TOKENIZER (char now, BPE switch-ready)
# =========================

class CharTokenizer:
    def __init__(self, docs):
        text = "\n".join(docs) + "\n"
        self.uchars = sorted(set(text))
        self.BOS = len(self.uchars)
        self.EOS = len(self.uchars) + 1
        self.vocab_size = len(self.uchars) + 2
        self.stoi = {ch: i for i, ch in enumerate(self.uchars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str):
        s = s.strip()
        ids = [self.BOS]
        for ch in s:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
        ids.append(self.EOS)
        return ids

    def decode(self, ids):
        out = []
        for t in ids:
            if t == self.BOS:
                continue
            if t == self.EOS:
                break
            out.append(self.itos.get(t, ""))
        return "".join(out).strip()

class BpeTokenizerStub:
    def __init__(self, docs):
        raise NotImplementedError("BPE mode is scaffolded; char mode is live.")

# =====================================================================
# 4) AUTOGRAD ENGINE — VECTOR-NATIVE
#
# Karpathy's microGPT wraps every scalar in a Value object.
# For 64-dim embeddings, that's 64 objects per vector, thousands per step.
#
# macroGPT thinks in vectors. One VectorValue = one embedding.
# One MatrixParam.matvec() = one linear layer. Same chain rule,
# but the graph has 10-100x fewer nodes.
#
# microGPT: "look, every scalar is differentiable"
# macroGPT: "organisms don't think in scalars. they think in fields."
# =====================================================================

class VectorValue:
    """A differentiable vector. One object = one embedding / hidden state."""
    __slots__ = ("data", "grad", "_children", "_back_fn")

    def __init__(self, data, children=(), back_fn=None):
        self.data = list(data) if not isinstance(data, list) else data
        self.grad = [0.0] * len(self.data)
        self._children = children
        self._back_fn = back_fn

    @property
    def size(self):
        return len(self.data)

    def __add__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue([a + b for a, b in zip(self.data, other.data)],
                              (self, other))
            def _back():
                for i in range(len(self.data)):
                    self.grad[i] += out.grad[i]
                    other.grad[i] += out.grad[i]
            out._back_fn = _back
            return out
        # scalar broadcast
        s = float(other)
        out = VectorValue([a + s for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[i]
        out._back_fn = _back
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = VectorValue([-a for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] -= out.grad[i]
        out._back_fn = _back
        return out

    def __sub__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue([a - b for a, b in zip(self.data, other.data)],
                              (self, other))
            def _back():
                for i in range(len(self.data)):
                    self.grad[i] += out.grad[i]
                    other.grad[i] -= out.grad[i]
            out._back_fn = _back
            return out
        return self + (-float(other))

    def __mul__(self, other):
        """Element-wise (Hadamard) or scalar broadcast."""
        if isinstance(other, VectorValue):
            out = VectorValue([a * b for a, b in zip(self.data, other.data)],
                              (self, other))
            def _back():
                for i in range(len(self.data)):
                    self.grad[i] += other.data[i] * out.grad[i]
                    other.grad[i] += self.data[i] * out.grad[i]
            out._back_fn = _back
            return out
        s = float(other)
        out = VectorValue([a * s for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += s * out.grad[i]
        out._back_fn = _back
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        if isinstance(other, ScalarValue):
            return self.scale(other.recip())
        raise TypeError(f"VectorValue / {type(other)} not supported")

    # ---- activations ----

    def relu(self):
        out = VectorValue([max(0.0, a) for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                if self.data[i] > 0:
                    self.grad[i] += out.grad[i]
        out._back_fn = _back
        return out

    def squared_relu(self):
        """ReLU(x)^2 — Karpathy's choice in microgpt."""
        r = [max(0.0, a) for a in self.data]
        out = VectorValue([x * x for x in r], (self,))
        def _back():
            for i in range(len(self.data)):
                if self.data[i] > 0:
                    self.grad[i] += 2.0 * r[i] * out.grad[i]
        out._back_fn = _back
        return out

    # ---- reductions (vector -> scalar) ----

    def dot(self, other):
        """Dot product -> ScalarValue."""
        val = sum(a * b for a, b in zip(self.data, other.data))
        out = ScalarValue(val, (self, other))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += other.data[i] * out.grad
                other.grad[i] += self.data[i] * out.grad
        out._back_fn = _back
        return out

    def mean_sq(self):
        """Mean of squares -> ScalarValue (for rmsnorm)."""
        n = len(self.data)
        val = sum(a * a for a in self.data) / n
        out = ScalarValue(val, (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += (2.0 * self.data[i] / n) * out.grad
        out._back_fn = _back
        return out

    # ---- scale by ScalarValue ----

    def scale(self, s):
        """Multiply every element by a ScalarValue."""
        out = VectorValue([a * s.data for a in self.data], (self, s))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += s.data * out.grad[i]
                s.grad += self.data[i] * out.grad[i]
        out._back_fn = _back
        return out

    # ---- slicing for multi-head attention ----

    def slice(self, start, end):
        out = VectorValue(self.data[start:end], (self,))
        def _back():
            for i, j in enumerate(range(start, end)):
                self.grad[j] += out.grad[i]
        out._back_fn = _back
        return out

    @staticmethod
    def concat(vecs):
        data = []
        for v in vecs:
            data.extend(v.data)
        out = VectorValue(data, tuple(vecs))
        def _back():
            offset = 0
            for v in vecs:
                for i in range(len(v.data)):
                    v.grad[i] += out.grad[offset + i]
                offset += len(v.data)
        out._back_fn = _back
        return out

    def __repr__(self):
        return f"VectorValue(dim={len(self.data)})"


class ScalarValue:
    """A differentiable scalar. For loss, dot products, attention weights."""
    __slots__ = ("data", "grad", "_children", "_back_fn")

    def __init__(self, data, children=(), back_fn=None):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._back_fn = back_fn

    def __add__(self, other):
        if isinstance(other, ScalarValue):
            out = ScalarValue(self.data + other.data, (self, other))
            def _back():
                self.grad += out.grad
                other.grad += out.grad
            out._back_fn = _back
            return out
        out = ScalarValue(self.data + float(other), (self,))
        def _back():
            self.grad += out.grad
        out._back_fn = _back
        return out

    def __radd__(self, other): return self.__add__(other)
    def __neg__(self): return self * -1.0

    def __sub__(self, other):
        if isinstance(other, ScalarValue):
            return self + (-other)
        return self + (-float(other))

    def __mul__(self, other):
        if isinstance(other, ScalarValue):
            out = ScalarValue(self.data * other.data, (self, other))
            def _back():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._back_fn = _back
            return out
        s = float(other)
        out = ScalarValue(self.data * s, (self,))
        def _back():
            self.grad += s * out.grad
        out._back_fn = _back
        return out

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ScalarValue):
            return self * other.recip()
        return self * (1.0 / float(other))

    def recip(self):
        out = ScalarValue(1.0 / self.data, (self,))
        def _back():
            self.grad += (-1.0 / (self.data ** 2)) * out.grad
        out._back_fn = _back
        return out

    def log(self):
        out = ScalarValue(math.log(self.data), (self,))
        def _back():
            self.grad += (1.0 / self.data) * out.grad
        out._back_fn = _back
        return out

    def exp(self):
        e = math.exp(self.data)
        out = ScalarValue(e, (self,))
        def _back():
            self.grad += e * out.grad
        out._back_fn = _back
        return out

    def __repr__(self):
        return f"ScalarValue({self.data:.6f})"


# ---- unified backward ----

def backward(root):
    """Topological-sort backward on a mixed VectorValue/ScalarValue graph."""
    topo = []
    visited = set()

    def build(v):
        vid = id(v)
        if vid not in visited:
            visited.add(vid)
            for c in v._children:
                build(c)
            topo.append(v)

    build(root)
    root.grad = 1.0
    for v in reversed(topo):
        if v._back_fn is not None:
            v._back_fn()

# =====================================================================
# 5) HIGH-LEVEL OPS (transformer building blocks)
# =====================================================================

class MatrixParam:
    """
    Weight matrix: rows of VectorValues. Shape (nout, nin).
    One object per layer-weight, not one object per scalar.
    """
    def __init__(self, nout, nin, std=0.02):
        self.rows = [VectorValue([random.gauss(0, std) for _ in range(nin)])
                     for _ in range(nout)]
        self.nout = nout
        self.nin = nin

    def matvec(self, x):
        """W @ x -> VectorValue of dim nout. Single backward for the whole op."""
        nout = self.nout
        nin = len(x.data)
        out_data = [sum(self.rows[i].data[j] * x.data[j]
                        for j in range(nin))
                    for i in range(nout)]
        out = VectorValue(out_data, tuple(self.rows) + (x,))
        rows_ref = self.rows
        def _back():
            for i in range(nout):
                g = out.grad[i]
                row_data = rows_ref[i].data
                row_grad = rows_ref[i].grad
                for j in range(nin):
                    row_grad[j] += g * x.data[j]
                    x.grad[j] += g * row_data[j]
        out._back_fn = _back
        return out

    def params(self):
        return list(self.rows)


def rmsnorm(x):
    """RMSNorm: x / sqrt(mean(x^2) + eps). VectorValue -> VectorValue."""
    ms = x.mean_sq()
    scale_val = (ms.data + 1e-5) ** -0.5
    out = VectorValue([a * scale_val for a in x.data], (x, ms))
    n = len(x.data)
    def _back():
        s = scale_val
        ds_dms = -0.5 * (ms.data + 1e-5) ** -1.5
        cross = sum(out.grad[j] * x.data[j] for j in range(n))
        for i in range(n):
            x.grad[i] += s * out.grad[i]
            x.grad[i] += cross * ds_dms * (2.0 * x.data[i] / n)
    out._back_fn = _back
    return out


def cross_entropy_loss(logits, target):
    """
    logits: VectorValue (over vocab), target: int -> ScalarValue (loss).
    Fused log-softmax + NLL. No intermediate softmax objects.
    """
    max_val = max(logits.data)
    shifted = [x - max_val for x in logits.data]
    exp_sum = sum(math.exp(x) for x in shifted)
    log_sum_exp = math.log(exp_sum) + max_val
    loss_val = log_sum_exp - logits.data[target]
    probs = [math.exp(x) / exp_sum for x in shifted]

    out = ScalarValue(loss_val, (logits,))
    def _back():
        g = out.grad
        for i in range(len(logits.data)):
            logits.grad[i] += (probs[i] - (1.0 if i == target else 0.0)) * g
    out._back_fn = _back
    return out


def scalar_softmax(logits):
    """Softmax over list[ScalarValue] -> list[ScalarValue]. For attention."""
    max_val = max(s.data for s in logits)
    exps_data = [math.exp(s.data - max_val) for s in logits]
    total = sum(exps_data)
    probs_data = [e / total for e in exps_data]

    out = []
    for i in range(len(probs_data)):
        sv = ScalarValue(probs_data[i], tuple(logits))
        local_i = i
        def _make_back(ii, ps):
            def _back():
                g = out[ii].grad
                for j in range(len(logits)):
                    if j == ii:
                        logits[j].grad += g * ps[ii] * (1.0 - ps[ii])
                    else:
                        logits[j].grad += g * (-ps[ii] * ps[j])
            return _back
        sv._back_fn = _make_back(local_i, probs_data)
        out.append(sv)
    return out


def attention_weighted_sum(weights, values):
    """
    weights: list[ScalarValue], values: list[VectorValue] -> VectorValue.
    The core attention aggregation: sum_t w_t * v_t.
    """
    dim = len(values[0].data)
    T = len(weights)
    out_data = [sum(weights[t].data * values[t].data[j]
                    for t in range(T))
                for j in range(dim)]
    out = VectorValue(out_data, tuple(weights) + tuple(values))
    def _back():
        for t in range(T):
            w_t = weights[t]
            v_t = values[t]
            for j in range(dim):
                w_t.grad += v_t.data[j] * out.grad[j]
                v_t.grad[j] += w_t.data * out.grad[j]
    out._back_fn = _back
    return out


def softmax_probs_float(data):
    """Pure-float softmax for sampling during generation. No autograd."""
    max_val = max(data)
    exps = [math.exp(x - max_val) for x in data]
    total = sum(exps)
    return [e / total for e in exps]


def top_k_sample(probs, k):
    # And the next token shall be chosen from the sacred top-k
    if k <= 0 or k >= len(probs):
        r = random.random()
        s = 0.0
        for i, p in enumerate(probs):
            s += p
            if s >= r:
                return i
        return len(probs) - 1
    idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    mass = sum(probs[i] for i in idx)
    r = random.random() * mass
    s = 0.0
    for i in idx:
        s += probs[i]
        if s >= r:
            return i
    return idx[-1]

# =====================================================================
# 6) DELTA ADAPTERS (LoRA-ish, modular, never overwrite)
# =====================================================================

class DeltaAdapter:
    """
    Low-rank adapter: for a weight W (nout x nin), we add A @ B @ x,
    where A is (nout x r), B is (r x nin). Only A, B are trained.
    A new soul-layer, appended, never replacing the old ones.
    """
    def __init__(self, nout, nin, r, std=0.02):
        self.A = MatrixParam(nout, r, std)
        self.B = MatrixParam(r, nin, std)

    def apply(self, x):
        bx = self.B.matvec(x)
        return self.A.matvec(bx)

    def params(self):
        return self.A.params() + self.B.params()

# =====================================================================
# 7) GPT MODEL (vector-native transformer)
# =====================================================================

class GPT:
    def __init__(self, tok):
        self.tok = tok
        self.n_layer = CFG.n_layer
        self.n_embd = CFG.n_embd
        self.n_head = CFG.n_head
        self.head_dim = CFG.n_embd // CFG.n_head
        self.block_size = CFG.block_size

        # Base weights — each is a MatrixParam, not a list-of-lists-of-Value
        self.base = {}
        self.base["wte"] = MatrixParam(tok.vocab_size, CFG.n_embd, 0.08)
        self.base["wpe"] = MatrixParam(CFG.block_size, CFG.n_embd, 0.08)
        self.base["lm_head"] = MatrixParam(tok.vocab_size, CFG.n_embd, 0.08)

        for li in range(CFG.n_layer):
            self.base[f"l{li}.wq"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.wk"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.wv"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.wo"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.fc1"] = MatrixParam(4 * CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.fc2"] = MatrixParam(CFG.n_embd, 4 * CFG.n_embd, 0.08)

        # Modular deltas
        self.deltas = []
        self.active_alpha = []

        # Adam state
        self._adam = {}

    def add_delta_module(self, alpha=1.0):
        mod = {}
        r = CFG.delta_rank
        for li in range(CFG.n_layer):
            mod[f"l{li}.wq"] = DeltaAdapter(CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.wk"] = DeltaAdapter(CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.wv"] = DeltaAdapter(CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.wo"] = DeltaAdapter(CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.fc1"] = DeltaAdapter(4 * CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.fc2"] = DeltaAdapter(CFG.n_embd, 4 * CFG.n_embd, r)
        mod["lm_head"] = DeltaAdapter(self.tok.vocab_size, CFG.n_embd, r)
        self.deltas.append(mod)
        self.active_alpha.append(alpha)

    def all_base_params(self):
        """All base VectorValue params (rows of all MatrixParams)."""
        out = []
        for mat in self.base.values():
            out.extend(mat.params())
        return out

    def all_delta_params(self):
        out = []
        for mod in self.deltas:
            for ad in mod.values():
                out.extend(ad.params())
        return out

    def _ensure_adam(self, params, key):
        if key not in self._adam:
            self._adam[key] = {
                "m": [[0.0] * len(p.data) for p in params],
                "v": [[0.0] * len(p.data) for p in params],
                "t": 0
            }

    def adam_step(self, params, key, lr):
        """Adam update on a list of VectorValue params."""
        self._ensure_adam(params, key)
        st = self._adam[key]
        st["t"] += 1
        t = st["t"]
        b1, b2, eps = CFG.beta1, CFG.beta2, CFG.eps_adam
        b1_corr = 1.0 - b1 ** t
        b2_corr = 1.0 - b2 ** t
        for i, p in enumerate(params):
            mi = st["m"][i]
            vi = st["v"][i]
            for j in range(len(p.data)):
                g = p.grad[j]
                mi[j] = b1 * mi[j] + (1 - b1) * g
                vi[j] = b2 * vi[j] + (1 - b2) * (g * g)
                mhat = mi[j] / b1_corr
                vhat = vi[j] / b2_corr
                p.data[j] -= lr * mhat / (math.sqrt(vhat) + eps)
                p.grad[j] = 0.0

    def _apply_with_deltas(self, name, x):
        """base W @ x + sum_i alpha_i * delta_i(x)."""
        y = self.base[name].matvec(x)
        for alpha, mod in zip(self.active_alpha, self.deltas):
            if name in mod:
                dy = mod[name].apply(x)
                y = y + dy * alpha
        return y

    def forward_step(self, token_id, pos_id, keys, values):
        """One token through the transformer. Returns logits as VectorValue."""
        # Embedding lookup: just index into the MatrixParam rows
        tok_emb = self.base["wte"].rows[token_id]
        pos_emb = self.base["wpe"].rows[pos_id]
        x = tok_emb + pos_emb

        for li in range(self.n_layer):
            # ---- Attention ----
            x_res = x
            x = rmsnorm(x)

            q = self._apply_with_deltas(f"l{li}.wq", x)
            k = self._apply_with_deltas(f"l{li}.wk", x)
            v = self._apply_with_deltas(f"l{li}.wv", x)

            keys[li].append(k)
            values[li].append(v)

            head_outputs = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                he = hs + self.head_dim
                qh = q.slice(hs, he)

                # attention logits: q_h dot each cached k_h
                attn_logits = []
                for t in range(len(keys[li])):
                    kh_t = keys[li][t].slice(hs, he)
                    dot = qh.dot(kh_t)
                    # scale by 1/sqrt(head_dim)
                    scaled = dot * (1.0 / (self.head_dim ** 0.5))
                    attn_logits.append(scaled)

                attn_weights = scalar_softmax(attn_logits)

                # value vectors for this head
                vh = [values[li][t].slice(hs, he) for t in range(len(values[li]))]
                head_out = attention_weighted_sum(attn_weights, vh)
                head_outputs.append(head_out)

            x_attn = VectorValue.concat(head_outputs)
            x = self._apply_with_deltas(f"l{li}.wo", x_attn)
            x = x + x_res

            # ---- MLP ----
            x_res = x
            x = rmsnorm(x)
            x = self._apply_with_deltas(f"l{li}.fc1", x)
            x = x.relu()
            x = self._apply_with_deltas(f"l{li}.fc2", x)
            x = x + x_res

        x = rmsnorm(x)
        logits = self._apply_with_deltas("lm_head", x)
        return logits

    def loss_on_sequence(self, ids):
        """Average cross-entropy loss over a token sequence."""
        n = min(self.block_size, len(ids) - 1)
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        total_loss = ScalarValue(0.0)
        for pos in range(n):
            tok = ids[pos]
            tgt = ids[pos + 1]
            logits = self.forward_step(tok, pos, keys, values)
            loss = cross_entropy_loss(logits, tgt)
            total_loss = total_loss + loss
        return total_loss * (1.0 / n)

    def generate_sentence(self, prompt_text=""):
        """Generate text. No autograd — pure float inference."""
        if prompt_text:
            ids = self.tok.encode(prompt_text)[:-1]
        else:
            ids = [self.tok.BOS]

        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]

        out = []
        recent = []

        # Process prompt tokens (build KV cache)
        for pos in range(min(len(ids), self.block_size)):
            _ = self.forward_step(ids[pos], pos, keys, values)

        cur = ids[-1]
        for step in range(CFG.max_gen_tokens):
            pos = min(len(ids) - 1, self.block_size - 1)
            logits = self.forward_step(cur, pos, keys, values)

            # Temperature scaling + top-k sampling (no autograd)
            scaled = [v / CFG.temperature for v in logits.data]
            probs = softmax_probs_float(scaled)
            nxt = top_k_sample(probs, CFG.top_k)

            if nxt == self.tok.EOS:
                break

            ids.append(nxt)
            cur = nxt
            out.append(nxt)

            # Repetition guard
            recent.append(nxt)
            if len(recent) > CFG.repetition_guard * 2:
                recent = recent[-CFG.repetition_guard * 2:]
                n = CFG.repetition_guard
                if recent[-n:] == recent[-2 * n:-n]:
                    break

            # Sliding window: rebuild KV cache when context is full
            if len(ids) >= self.block_size:
                tail = ids[-self.block_size:]
                ids = tail
                keys = [[] for _ in range(self.n_layer)]
                values = [[] for _ in range(self.n_layer)]
                for p in range(len(ids) - 1):
                    _ = self.forward_step(ids[p], p, keys, values)

        return self.tok.decode([self.tok.BOS] + out + [self.tok.EOS])

# =====================================================================
# 8) CHECKPOINTING (modular, never overwrite old deltas)
# =====================================================================

def _serialize_matrix_param(mp):
    return [list(row.data) for row in mp.rows]

def _deserialize_matrix_param(data):
    mp = MatrixParam.__new__(MatrixParam)
    mp.rows = [VectorValue(row) for row in data]
    mp.nout = len(data)
    mp.nin = len(data[0]) if data else 0
    return mp

def save_checkpoint(model: GPT, path="macrogpt_ckpt.json"):
    obj = {
        "cfg": CFG.__dict__,
        "base": {k: _serialize_matrix_param(v) for k, v in model.base.items()},
        "deltas": [],
        "alpha": model.active_alpha,
        "tokenizer": {
            "mode": CFG.tokenizer_mode,
            "uchars": getattr(model.tok, "uchars", None),
        }
    }
    for mod in model.deltas:
        m = {}
        for name, ad in mod.items():
            m[name] = {
                "A": _serialize_matrix_param(ad.A),
                "B": _serialize_matrix_param(ad.B),
            }
        obj["deltas"].append(m)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def load_checkpoint(tok, path="macrogpt_ckpt.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    model = GPT(tok)
    model.base = {k: _deserialize_matrix_param(v) for k, v in obj["base"].items()}
    model.deltas = []
    model.active_alpha = obj.get("alpha", [])

    for mod_data in obj.get("deltas", []):
        mm = {}
        for name, w in mod_data.items():
            ad = DeltaAdapter.__new__(DeltaAdapter)
            ad.A = _deserialize_matrix_param(w["A"])
            ad.B = _deserialize_matrix_param(w["B"])
            mm[name] = ad
        model.deltas.append(mm)

    if not model.deltas:
        model.add_delta_module(alpha=1.0)

    return model

# =====================================================================
# 9) TRAINING (warmup + continual async micro-train)
# =====================================================================

def train_steps(model: GPT, tok, docs, steps, train_base=True, train_deltas=True):
    if not docs:
        return

    base_params = model.all_base_params() if train_base else []
    delta_params = model.all_delta_params() if train_deltas else []

    for step in range(steps):
        doc = random.choice(docs)
        ids = tok.encode(doc)
        loss = model.loss_on_sequence(ids)
        backward(loss)

        lr = CFG.learning_rate * (1.0 - (step / max(1, steps)))
        if base_params:
            model.adam_step(base_params, key="base", lr=lr)
        if delta_params:
            model.adam_step(delta_params, key="delta", lr=lr)

        if step % 100 == 0:
            print(f"  train step {step}/{steps} | loss {loss.data:.4f}")

def compute_new_corpus_mass(con, last_event_id):
    cur = con.cursor()
    cur.execute("SELECT id, added_chars FROM corpus_events WHERE id > ? ORDER BY id ASC",
                (last_event_id,))
    rows = cur.fetchall()
    if not rows:
        return 0, last_event_id
    mass = sum(r[1] for r in rows)
    return mass, rows[-1][0]

async def background_trainer(con, model: GPT, tok):
    last_event_id = 0
    warmed_up = False

    while True:
        _ = update_reservoir_corpus(con, CFG.corpus_path, CFG.max_corpus_lines)
        mass, last_event_id = compute_new_corpus_mass(con, last_event_id)
        docs = load_corpus_lines(CFG.corpus_path)

        # Initial warmup
        if (not warmed_up) and docs:
            if not model.deltas:
                model.add_delta_module(alpha=1.0)
            print("[trainer] warmup training...")
            train_steps(model, tok, docs, CFG.base_train_steps,
                        train_base=True, train_deltas=True)
            save_checkpoint(model)
            warmed_up = True
            print("[trainer] warmup complete. base frozen.")

        # Continual micro-training bursts
        if warmed_up and mass >= CFG.min_new_chars_to_train and docs:
            print(f"[trainer] micro-train burst ({mass} new chars)...")
            train_base = not CFG.freeze_base_after_warmup
            train_steps(model, tok, docs, CFG.micro_steps_per_round,
                        train_base=train_base, train_deltas=True)
            save_checkpoint(model)

            # Occasionally grow a new delta module
            if len(model.deltas) < 12 and random.random() < 0.08:
                print(f"[trainer] growing new delta module (total: {len(model.deltas)+1})")
                model.add_delta_module(alpha=1.0)
                save_checkpoint(model)

        await asyncio.sleep(CFG.train_tick_seconds)

# =====================================================================
# 10) CHAT LOOP (SQLite memory + sentence generation)
# =====================================================================

def build_prompt_from_memory(con, user_text):
    recent = db_recent_messages(con, limit=10)
    parts = []
    for role, text in recent[-8:]:
        tag = "Human:" if role == "user" else "AI:"
        parts.append(f"{tag} {normalize_text(text)}")
    parts.append(f"Human: {normalize_text(user_text)}")
    parts.append("AI:")
    return "\n".join(parts)

async def chat_main():
    con = init_db(CFG.db_path)

    if not os.path.exists(CFG.corpus_path):
        save_corpus_lines(CFG.corpus_path, ["Hello.", "I exist.", "Speak."])

    docs = load_corpus_lines(CFG.corpus_path)
    if CFG.tokenizer_mode == "char":
        tok = CharTokenizer(docs if docs else ["Hello."])
    else:
        tok = BpeTokenizerStub(docs)

    model = load_checkpoint(tok) or GPT(tok)
    if not model.deltas:
        model.add_delta_module(alpha=1.0)

    trainer_task = asyncio.create_task(background_trainer(con, model, tok))

    print("macrogpt is alive. Type and press Enter. Ctrl+C to exit.\n")
    try:
        while True:
            user_text = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
            user_text = user_text.strip()
            if not user_text:
                continue

            db_add_message(con, "user", user_text)
            prompt = build_prompt_from_memory(con, user_text)
            answer = model.generate_sentence(prompt_text=prompt)

            if not answer:
                answer = "..."

            print(answer)
            db_add_message(con, "assistant", answer)

    except KeyboardInterrupt:
        pass
    finally:
        trainer_task.cancel()
        try:
            await trainer_task
        except asyncio.CancelledError:
            pass
        save_checkpoint(model)
        con.close()

# =====================================================================
# 11) AWAKEN
# =====================================================================

def main():
    asyncio.run(chat_main())

if __name__ == "__main__":
    main()

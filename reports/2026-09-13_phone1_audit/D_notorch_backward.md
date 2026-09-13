# D — notorch training backward path, audit

Target: `/data/data/com.termux/files/home/arianna/notorch`, branch `claude/frozen-slot`,
HEAD `42ff2f3` (`git -C … log --oneline -1`). Consumer:
`/data/data/com.termux/files/home/arianna/molequla/notorch_trainer.go`.
Read-only pass. CPU path only; `USE_CUDA` blocks are noted where a CPU branch
would read a stale mirror, never audited for GPU correctness. The optimizer is
Chuck (`nt_tape_chuck_step`, notorch.c:2523).

All line numbers are from the files as read today. Every claim below is
reproducible with `grep -n` on the cited file.

---

## 0. Headline

The per-op gradient math in the molequla chain is **correct against its own
forward**. Every op in
`nt_seq_embedding → nt_seq_rmsnorm → nt_seq_linear → nt_rope →
nt_mh_causal_attention → nt_rrpram_lowrank_attention → nt_mul/nt_add →
nt_silu → nt_seq_cross_entropy` was checked term by term (§1) and no
unconditional sign, scale, transpose or pairing error was found. Parent
accumulation is uniformly `+=` through one funnel (`tape_acc_grad`,
notorch.c:495-515) and no op in the chain drops a parent.

What is wrong is one level up: the graph notorch differentiates is **not the
graph molequla's Go inference runs** (D1), and the loss is computed over
padding as if padding were content (D2). Both are P0/P1 by impact and neither
is visible from inside notorch.

Findings, by severity:

| id | sev | one line |
|---|---|---|
| D1 | **P0** | `residualAlpha = 1/√NLayer` scales every residual in Go inference and is absent from the training graph |
| D2 | **P1** | `nt_seq_cross_entropy` has no ignore/pad path — padding is trained as token 0, denominator is always `T` |
| D3 | **P1** | `nt_mul`/`nt_add` backward drop the `% pb->len` broadcast the forward applies (OOB read in MUL) |
| D4 | **P1** | RRPRAM-LR rank is derived from `T`; `T < T_max` silently reinterprets the packed `Wr` layout |
| D5 | **P1** | Chuck's per-param auto-freeze is permanent and survives `nt_tape_clear` |
| D6 | **P1** | `nt_tape_destroy()` after `nt_tape_clear()` frees no optimizer moments — confirmed leak |
| D7 | **P1** | RMSNorm eps 1e-6 in training vs 1e-5 in Go inference |
| D8 | **P2** | params past `NT_TAPE_MAX_PARAMS` get grads, inflate the clip norm, and are never stepped |
| D9 | **P2** | `nt_tape_record`/`record3` do not reset `aux3`/`aux4` |
| D10 | **P2** | `NT_OP_SEQ_CROSSENT` backward forms `&entries[parent2]` without a bound check |
| D11 | **P2** | Chuck clamps `n` by `m->len` but never by `grad->len` |
| D12 | **P2** | `nt_nan_guard`'s `loss_scale` is computed and never consumed |
| D13 | **P2** | `NT_OP_SEQ_EMBED` backward builds a dense `V×D` gradient every step |
| D14 | **P2** | RoPE leaves the unpaired tail (odd `head_dim`, `D % head_dim ≠ 0`) at zero grad |
| D15 | **P2** | aarch64/x86 divergence: `USE_BLAS` branch + per-arch `-march`, plus x86-hardcoded cgo paths |
| D16 | **P2** | Chuck slot identity depends on the wr-layer set being stable; the len-mismatch branch reinterprets moments row-major |
| D17 | **P2** | a full tape returns `-1`, which propagates into a silent no-op step |

---

## 1. Per-op: forward, backward, accumulation, parents

Shared machinery. `tape_acc_grad` (notorch.c:495) is the only writer of parent
gradients on the CPU path:

```
495  static void tape_acc_grad(int idx, const float* grad, int len) {
498      if (e->frozen) return;
499      if (!e->grad) { e->grad = nt_tensor_new(len); ... }
508      int n = e->grad->len < len ? e->grad->len : len;
509      for (int i = 0; i < n; i++) e->grad->data[i] += grad[i];
```

`+=` always — **accumulation, never overwrite**. Every backward case builds a
`calloc`'d local buffer and hands it to this function, so a parameter consumed
by several ops (e.g. `v`, consumed by both `nt_mh_causal_attention` and
`nt_rrpram_lowrank_attention` in notorch_trainer.go:118,126) sums correctly.

Traversal order: `nt_tape_backward` walks `for (int idx = loss_idx; idx >= 0;
idx--)` (notorch.c:565). Every recorder appends at `g_tape.count`
(notorch.c:343, 364, 385) and parents are indices obtained earlier, so
`parent < child` holds by construction and the descending walk is a valid
topological order. Entries with `!e->grad` are skipped (notorch.c:567).

### 1.1 `nt_seq_embedding` → `NT_OP_SEQ_EMBED`

Forward notorch.c:3047-3096. `h[t,d] = wte[tok_t, d] + wpe[min(t, wpe_rows-1), d]`,
with `tok` clamped to `[0, wte_rows-1]` (3075-3076).

Backward notorch.c:950-1003.
- `dwte[tok_t, d] += dout[t,d]` (981) — correct: the embedding row is selected,
  so ∂h[t,d]/∂wte[r,d] = δ(r, tok_t); repeated tokens accumulate, which the
  `+=` gives.
- `dwpe[pos, d] += dout[t,d]` (995) — same, with the identical clamp
  (993 vs forward 3088). ✓
- Token clamp identical between passes (978-979 vs 3075-3076). ✓

Parents: `parent1 = wte`, `parent2 = wpe`, `parent3 = tokens`. The guard is
`if (e->parent1 >= 0 && e->parent3 >= 0)` (951) and wpe is handled inside
`if (e->parent2 >= 0)` (987). Tokens correctly receive no gradient (they are a
`NT_OP_NONE` input, cgo_notorch.go:101-103). No missing parent.

CUDA note: the CPU branch reads `ptok->output->data` and is preceded by
`nt_tensor_ensure_cpu(ptok->output)` (972) only inside `#ifdef USE_CUDA`. It
does **not** sync `pwte->output`, but it does not read it either (only its
`len`/`shape`). Clean.

See D13 for the cost of the dense `dwte` buffer.

### 1.2 `nt_seq_rmsnorm`, gamma = -1 → `NT_OP_SEQ_RMSNORM`

Forward notorch.c:3256-3302, CPU branch 3281-3296:

```
3287   float rms = sqrtf(ss / D + 1e-6f);
3288   o_t[d] = x_t[d] / rms;
```

so per position `y_d = x_d / r`, `r = sqrt(mean(x²) + ε)`, `ε = 1e-6`. With
`gamma_idx = -1` the gamma multiply (3290-3295) is skipped and
`nt_tape_record3` stores `parent2 = -1` (3298-3299), so `has_gamma = 0` in the
backward (1102).

Math. `∂r/∂x_i = (1/2r)·(2x_i/D) = x_i/(D·r)`, therefore

```
∂y_d/∂x_i = δ_{di}/r − x_d·x_i/(D·r³)
dL/dx_i   = g_i/r − x_i·(Σ_d g_d x_d)/(D·r³)
```

Backward notorch.c:1137-1159:

```
1142   float rms = sqrtf(ss / D + 1e-6f);
1143   float rms3 = rms * rms * rms;
1149   sum_dx += de * x_t[d];
1153   gx[t*D+d] = (de / rms) - (x_t[d] * sum_dx / (D * rms3));
```

Exact match, ε included (ε shifts `r` but not `∂(ss/D+ε)/∂x_i = 2x_i/D`, so no
ε term appears in the derivative — correct). Per-position statistics, no
cross-position leakage. ✓

With gamma present, `de = dout·γ_d` (1148, 1152) and
`dγ_d = Σ_t dout[t,d]·x[t,d]/r_t` (1158) — both correct; molequla does not use
this branch.

The single-tensor sibling `NT_OP_RMSNORM` (841-904) uses the same formula with
`n = out_len` over the whole tensor and the same `1e-6` (857). Not on
molequla's path (`ntSeqRMSNorm` only).

Train ≡ infer: see D7 — the Go side uses `1e-5`.

### 1.3 `nt_seq_linear` → `NT_OP_SEQ_MATVEC`

Forward notorch.c:3122-3173. `Y[T,out] = X[T,in] · Wᵀ`, i.e.
`Y[t,i] = Σ_j W[i,j]·X[t,j]` (scalar 3157-3166; BLAS `NoTrans,Trans` 3152).

Backward notorch.c:1005-1093. `dX = dY·W`, `dW = dYᵀ·X`.

BLAS path (1052-1064):
- `dx[T,in] = dout[T,out] @ W[out,in]`, `NoTrans/NoTrans`, `M=T N=in_d K=out_d`,
  `lda=out_d` (dout row stride ✓), `ldb=in_d` (W row stride ✓), `ldc=in_d` ✓.
- `dw[out,in] = doutᵀ[out,T] @ X[T,in]`, `Trans/NoTrans`, `M=out_d N=in_d K=T`,
  `lda=out_d` (A is stored `T×out_d`, correct leading dim under `CblasTrans`),
  `ldb=in_d`, `ldc=in_d` ✓.

Scalar fallback (1066-1082) computes the same sums with `+=` into `calloc`'d
buffers. `beta = 0` in the BLAS calls overwrites those buffers — safe, they are
fresh locals; the accumulation into the tape happens in `tape_acc_grad`
(1086-1087).

`w_frozen`/`x_frozen` short-circuits (1012-1013) come from the entry flag, not
from a slot, consistent with the HEAD fix. Parents: `parent1 = W`,
`parent2 = X`, `parent3 = -1` unused. ✓

### 1.4 `nt_rope` → `NT_OP_ROPE`

Forward notorch.c:4360-4410 (`nt_rope` → `nt_rope_freq` with base 10000,
4412-4414). Even/odd pairing:

```
4394   float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
4395   float angle = t * freq;
4400   out[base+2i]   = x0*cos − x1*sin
4401   out[base+2i+1] = x0*sin + x1*cos
```

Recorded with `nt_tape_record4(... aux=T, aux2=head_dim, aux3=freq_base,
aux4=0.0)` (4407), so the backward recovers `head_dim` from `aux2` and the base
from `aux3`.

Backward notorch.c:2232-2297. Jacobian of the forward pair is
`J = [[c, −s],[s, c]]` (a rotation by +θ), so `dx = Jᵀ·dy`, i.e. rotation by
−θ:

```
2287   gx[o0] =  dx0*cos + dx1*sin
2288   gx[o1] = −dx0*sin + dx1*cos
```

**Yes — the backward rotates by the inverse angle.** ✓

Pairing and geometry match the forward exactly:
- `head_dim` from `aux2`, fallback `D` (2242-2243); `n_heads = D/head_dim`
  (2244) vs forward 4366; `D = total/T` (2240) vs forward 4365. ✓
- indices `base + 2i` / `base + 2i + 1` (2275-2276, `split_half = 0`) vs forward
  4398-4399. ✓
- `freq = 1/powf(fb, 2.0f*i/head_dim)`, `angle = t*freq` (2271-2272) —
  character-for-character the forward's expression (4394-4395), so the same
  `powf`/`cosf`/`sinf` rounding on the same machine. ✓
- `half = head_dim/2` (2266) vs forward's `head_dim/2` bound (4393). ✓

The split-half variant (`nt_rope_split_half_freq`, 4416-4464, `aux4 = 1.0`) has
forward `n0 = x0·c + x1·s`, `n1 = −x0·s + x1·c` (4452-4453), `J = [[c,s],[−s,c]]`,
`Jᵀ = [[c,−s],[s,c]]`, and the backward writes exactly that (2282-2283). ✓
Not used by molequla (`ntRope` → `nt_rope`).

`gx` is written with `=` not `+=` (2282-2288) — safe: `gx` is `calloc`'d (2264)
and, for even `head_dim` and `D % head_dim == 0`, every index is written exactly
once. See D14 for the case where it is not.

### 1.5 `nt_mh_causal_attention` → `NT_OP_MH_CAUSAL_ATTN`

Forward notorch.c:3502-3572. Per head `h`, per query `i`:
`s_j = (q_i·k_j)/√hd` for `j ≤ i`, `a = softmax(s)`, `o_i = Σ_{j≤i} a_j·v_j`.
`sc = 1/sqrtf(head_dim)` (3508), `D = pq->len/T`, `n_heads = D/head_dim` with an
explicit `D % head_dim != 0` reject (3507).

Backward notorch.c:1232-1351, CPU branch 1300-1347.

- **1/√hd scale.** `sc = 1.0f/sqrtf((float)head_dim)` (1241) — same
  `head_dim` (from `aux2`, 1238), same expression as the forward (3508).
  Applied once, on `ds` (1333), which is where the chain rule puts it:
  `∂s_j/∂(q·k) = sc`. ✓ `head_dim` is used for the per-head dot product
  (1313, 1324, 1328, 1335) while `D` is the row stride (1304, 1311, 1336) —
  the two are not conflated.
- **Softmax backward.** `d_attn_j = Σ_d dout_i[d]·v_j[d]` (1324);
  `dot_da = Σ_j d_attn_j·a_j` (1331);
  `ds_j = a_j·(d_attn_j − dot_da)·sc` (1333). This is the standard
  `da/ds = diag(a) − aaᵀ` contraction, `(∂L/∂s)_j = a_j(g_j − Σ_k g_k a_k)`. ✓
- **Causal mask on the backward.** Every `j` loop is bounded `j <= i`
  (1310, 1318, 1322, 1326, 1331, 1332). Positions `j > i` are never read and
  never written, so `dk`/`dv` receive no contribution from queries that could
  not see them. The mask is therefore respected in the backward, structurally
  rather than by a mask tensor. ✓
- `dq[i] += Σ_j ds_j·k_j`, `dk[j] += Σ_j ds_j·q_i`, `dv[j] += Σ_j a_j·dout_i`
  (1328, 1336-1337) — correct transposes of the forward contractions,
  accumulating across `i` and across heads. ✓
- Scores are **recomputed** in the backward (1310-1319) rather than cached; the
  recomputation is byte-identical to the forward loop (3549-3558) including the
  `mx` max-subtraction and the `if (sm > 0)` guard, so the recomputed `attn`
  equals the forward's. ✓
- Parents: `parent1=q, parent2=k, parent3=v`, all three required (1233) and all
  three written (1344-1346). No missing parent. ✓

**GQA aux fields.** `NT_OP_MH_CAUSAL_ATTN` is recorded via `nt_tape_record3`
(3531, 3569), which sets only `aux`/`aux2`. The MH backward reads only
`e->aux` and `e->aux2` (1237-1238) — it never touches `aux3`/`aux4`, so the
absence of GQA fields here is correct, not a latent read. GQA has its own op,
`NT_OP_GQA_ATTN` (1353-1422), recorded with `record4` carrying
`aux3 = n_heads`, `aux4 = n_kv_heads` (3615-3616) and mapping `kv_h = h/gqa_ratio`
(1371) with separate `Q_D`/`KV_D` strides — that backward is self-consistent and
correctly folds all `gqa_ratio` query heads into one kv head via `+=` on
`dk`/`dv` (1408, 1399). Not on molequla's path. The general hazard that
`record`/`record3` leave `aux3`/`aux4` stale is D9.

CUDA note: the CPU fallback syncs `pq`, `pk`, `pv` before reading them
(1297-1299) — this is the documented 6th instance of the stale-mirror class and
it is handled.

### 1.6 `nt_rrpram_lowrank_attention` (op 33) → `NT_OP_RRPRAM_LR`

Forward notorch.c:3679-3779. Layout (documented 3662-3678):
`Wr` is one tensor holding `Wr_a[H,E,R]` then `Wr_b[H,R,T_r]`,
`len = H·R·(E + T_r)`. Per head `h`, query `i`:

```
u[r]      = Σ_d x_i[d]·Wr_a[h,d,r]
scores[j] = Σ_r u[r]·Wr_b[h,r,j]      (j ≤ i, no 1/√ scale)
attn      = softmax(scores[0..i])
out_i[d]  = Σ_{j≤i} attn[j]·v[j, h·hd + d]
```

**Backward for both factors packed in one tensor.** notorch.c:1424-1584. One
`dwr` buffer of `combined_len` (1448) receives both factor gradients, at the
same offsets the forward reads from:

- `wr_a_base = h·E·R` (1506) — identical to forward 3738.
- `wr_b_base = wra_total + h·R·T_r`, `wra_total = H·E·R` (1507, 1446) —
  identical to forward 3739, 3693.
- `d_Wr_b[h,r,j] += d_score_j · u[r]` at `dwr[wr_b_base + r·T_r + j]` (1557),
  same index expression as the forward's read (1524 / 3755). ✓
- `d_Wr_a[h,d,r] += d_u[r] · x_i[d]` at `dwr + wr_a_base + d·R` (1565, 1570),
  same as the forward's `wa_row` (1517 / 3747). ✓

Chain, term by term: `d_attn_j = Σ_d dout_i[d]·v_j[d]` (1540);
`dv[j,h·hd+d] += attn_j·dout_i[d]` (1541);
`d_score_j = attn_j(d_attn_j − Σ_k d_attn_k·attn_k)` (1547-1548) — softmax
backward, again with every `j` loop bounded `j <= i` (1535-1536, 1547-1548,
1553) so the causal mask holds in the backward;
`d_u[r] = Σ_{j≤i} d_score_j·Wr_b[h,r,j]` (1556);
`d_x_i[d] = Σ_r d_u[r]·Wr_a[h,d,r]` (1569, 1572). All correct transposes.
`u` and `attn` are recomputed (1514-1531) with the forward's exact loop. ✓

Parents: `parent1 = Wr` (both factors), `parent2 = x`, `parent3 = v`, all
required (1436), all three written (1576-1578). No missing parent. Because
both factors live in one tape entry they share one gradient, one Chuck slot and
one `no_decay` flag — which is what notorch_trainer.go:336-337 intends.

**The `T == T_max` assumption.** Forward:

```
3689   int T_r = T;   /* assumption */
3691   int rank = (int)(combined_len / ((long)nr_heads * (n_embd + T_r)));
```

Backward, identically:

```
1443   int T_r = T;   /* same assumption as forward */
1445   int rank = (int)(combined_len / ((long)nr * (n_embd + T_r)));
```

Forward and backward agree with each other, so the gradient is the true
gradient *of whatever layout that formula implies*. The assumption is about the
producer of the buffer. molequla packs `Wr_b` at width `BlockSize`
(`ensureRRPRAMFactors`, molequla.go:894: `NewMatrixParam(gpt.NHead*R,
gpt.BlockSize, 0.02)`; `ntPackWr`, notorch_trainer.go:144-154), and pins the
training sequence length to match:

```
224   hasRRPRAM := layerHasHybrid()
225   if hasRRPRAM {
226       seqLen = model.BlockSize
227   }
```

so today `T == T_max` holds and `rank` resolves to the true `R`. **What happens
when `T < BlockSize`:** `rank' = floor(R·(E+T_max)/(E+T)) > R`. The forward and
the backward then both read `Wr_a` with row stride `rank'` instead of `R`, and
`Wr_b` at base `H·E·rank'` with row stride `T` instead of `T_max` — a complete
reinterpretation of the buffer. It is **memory-safe**: the largest index touched
is `nr·rank'·(E+T) − 1 ≤ combined_len − 1` because `rank'` is a floor, so there
is no out-of-bounds read and no crash. It is **silently wrong**: every forward
score and every `dwr` element lands on the wrong factor entries, and `ntUnpackWr`
(notorch_trainer.go:158-170) writes the scrambled result back into
`model.Base`. There is no assertion anywhere. See D4.

`T > T_max` is worse but also silent: `rank'` floors to a smaller value or to 0,
and `rank < 1` is rejected only in the forward (3692), not in the backward.

### 1.7 `nt_mul`, `nt_add` → `NT_OP_MUL`, `NT_OP_ADD`

Forward `nt_add` notorch.c:4260-4294, `nt_mul` 4296-4328. Both size the output
from `a` and **broadcast `b` by wrapping**:

```
4289   out->data[i] = pa->output->data[i] + pb->output->data[i % pb->output->len];
4323   out->data[i] = pa->output->data[i] * pb->output->data[i % pb->output->len];
```

Backward `NT_OP_ADD` notorch.c:581-603:

```
600   if (e->parent1 >= 0) tape_acc_grad(e->parent1, dout, out_len);
601   if (e->parent2 >= 0) tape_acc_grad(e->parent2, dout, out_len);
```

Backward `NT_OP_MUL` notorch.c:605-650:

```
641   ga[i] = dout[i] * pb->output->data[i];
642   gb[i] = dout[i] * pa->output->data[i];
```

**Broadcasting assumption: both backwards assume `len(a) == len(b) == out_len`.**
Neither reproduces the forward's `% pb->len`. Consequences when `len(b) < len(a)`
are in D3. On molequla's path the assumption holds: `ntMul(attn, gateCIdx[l])`
and `ntMul(rAttn, gateRIdx[l])` (notorch_trainer.go:127) pair a `T·D` activation
with a `T·D` gate built at the same `seqLen` (`ntBuildGateVectors`,
notorch_trainer.go:195-205, called at 257 with the same `seqLen`); the three
residual/gate `ntAdd`s (127, 129, 134) all pair equal-length tensors.

Math, given equal lengths: `∂(a+b)/∂a = ∂(a+b)/∂b = 1` ✓;
`∂(a⊙b)/∂a = b`, `∂(a⊙b)/∂b = a` ✓.

**Frozen parents.** `gateCIdx`/`gateRIdx` are registered through
`nt_tape_param_frozen` (notorch_trainer.go:338-339 → notorch.c:468-492), which
sets `frozen = 1` and `slot = -1` and does **not** increment `n_params`. The MUL
backward still computes `gb` into a temporary (642) and calls `tape_acc_grad`
(645), where `if (e->frozen) return;` (498) drops it before allocating. So the
frozen gates cost one wasted `calloc` + one multiply pass per gated MUL per step
but never allocate a grad, never enter the clip norm (2776 requires `e->grad`),
never enter the NaN guard (2954), and never consume a Chuck slot. Correct, if
slightly wasteful.

### 1.8 `nt_silu` → `NT_OP_SILU`

Forward notorch.c:3304-3332: `out[i] = x/(1+e^{−x}) = x·σ(x)` (3326).

Backward notorch.c:703-737:

```
729   float sig = 1.0f / (1.0f + expf(-x));
730   gx[i] = dout[i] * sig * (1.0f + x * (1.0f - sig));
```

`d/dx [x·σ] = σ + x·σ(1−σ) = σ·(1 + x(1−σ))`. ✓ Recomputed from the parent's
input `x`, not from the output — correct choice, since `σ` is not recoverable
from `x·σ` alone. Parent1 only; `parent2 = -1` (3329) and unused.

CUDA note: `nt_tensor_sync_cpu(px->output)` at 724 guards the documented stale
mirror. The sibling `NT_OP_SWIGLU` (2300-2348) has the same derivative in fused
form (2338) — molequla uses the unfused `ntSilu` + `ntMul` pair instead
(notorch_trainer.go:132-134), which is equivalent and costs one extra tape entry.

### 1.9 `nt_seq_cross_entropy` → `NT_OP_SEQ_CROSSENT`

Forward notorch.c:4178-4220, CPU branch 4203-4216:

```
4207   int target = (int)pt->output->data[t];
4208   if (target < 0 || target >= V) target = 0;
4213   total_loss += -(logits_t[target] - mx - logf(sum));
4215   out->data[0] = total_loss / T;
```

Backward notorch.c:1885-1939, CPU branch 1916-1934:

```
1919   int target = (int)pt->output->data[t];
1920   if (target < 0 || target >= V) target = 0;
1929   dl[t*V+j] = softmax_j
1930   dl[t*V+target] -= 1.0f;
1931   float s = dout[0] / T;
```

- **Mean vs sum over T.** Forward divides by `T` (4215); backward multiplies by
  `dout[0]/T` (1931). Consistent: `∂L/∂logits[t,j] = (p_{t,j} − y_{t,j})/T`. ✓
  The `T` is the tape's `aux` (1889), the same `T` the forward was called with.
- **Targets as float indices.** Targets arrive as a `float` tensor
  (`ntTapeInput(tgtT)`, notorch_trainer.go:347) and are truncated with `(int)`
  in both passes. `float32` represents integers exactly to 2²⁴, and molequla
  writes `float32(ids[idx])` (notorch_trainer.go:311), so the round-trip is
  exact for any realistic vocabulary. The truncation is toward zero, which is
  identity on exact non-negative integers. ✓
- **Ignore / pad handling: none.** Out-of-range targets are *clamped to 0*, not
  masked, in both passes. Token 0 therefore receives a full `−1` in the gradient
  row and the position counts toward the `1/T` denominator. notorch does have a
  masked variant — `nt_seq_cross_entropy_masked` / `NT_OP_SEQ_CROSSENT_MASKED`
  (4222-4258 forward, 1941-1991 backward), which skips `m == 0` rows entirely
  (1969) and normalizes by `n_active` (1983) — and molequla does not use it.
  See D2.

Parents: `parent1 = logits` (differentiated), `parent2 = targets` (constant,
correctly no gradient), `parent3 = -1`. No missing parent; see D10 for the
unguarded `parent2` deref.

Softmax is recomputed in the backward from the logits with the same
max-subtraction as the forward (1921-1929 vs 4209-4212). ✓

---

## 2. `nt_tape_clip_grads` and `nt_nan_guard*`

### 2.1 `nt_tape_clip_grads` (notorch.c:2744-2814)

**Is the norm global over all params?** Yes, and only over params. Both the
norm loop (2774-2783) and the scaling loop (2788-2811) filter on
`if (!e->is_param || !e->grad) continue;` (2776, 2790). Activation gradients
are excluded — correct, they are not being clipped. Every `is_param` entry
carrying a grad contributes `Σ g²` to one accumulator (2781) and the result is
one `sqrtf` (2785) — a single global L2 norm, matching
`torch.nn.utils.clip_grad_norm_` semantics.

**Does it include frozen / no-slot entries?**
- *Frozen* params: excluded, but indirectly. `nt_tape_param_frozen` sets
  `frozen = 1` (485) and `tape_acc_grad` returns before allocating (498), so
  `e->grad` stays `NULL` and the `!e->grad` filter drops them. The same holds
  for a param frozen after registration via `nt_tape_freeze_param` (457-466).
  Correct outcome, but note it depends on `tape_acc_grad`'s early return rather
  than on a `frozen` check inside `clip_grads` — a future backward that writes a
  grad by another route would silently pull a frozen param into the norm.
- *No-slot* params (registered past `NT_TAPE_MAX_PARAMS`): **included**, and
  that is a bug — see D8.

**Does it scale in place?** Yes: `e->grad->data[j] *= scale` (2806), scale
`= max_norm / (total_norm + 1e-6f)` (2787), applied only when
`total_norm > max_norm` (2786). The return value is the *pre-clip* norm (2813),
which is the useful one for logging. The `+1e-6` in the denominator makes the
post-clip norm marginally below `max_norm` — cosmetic.

`n` is clamped by `min(e->output->len, e->grad->len)` (2777-2778, 2791-2792), so
if a grad were ever shorter than its param only the leading elements would be
normed and scaled. See D11.

Ordering in molequla (notorch_trainer.go:353-357): backward → `guard.check()` →
clip → Chuck step. Correct — the NaN guard runs *before* the clip, so a NaN
grad never reaches `total_norm_sq` (where it would make `scale` NaN and poison
every parameter's gradient). Had the order been reversed this would be a P0.

### 2.2 `nt_nan_guard_new` / `nt_nan_guard_check` (notorch.c:2940-3004)

**What it actually checks:** every tape entry with `is_param && grad` (2954),
scanning `n = e->grad->len` floats (2955) for `g != g` (NaN) or `g == ±inf`
(2973). It does **not** inspect activations, does **not** inspect parameter
values, and does **not** inspect the loss. It breaks on the first hit (2975,
2978). Frozen and grad-less params are skipped by the same filter as the clip.

**What it does on a hit:** zeroes *all* param grads with `memset`
(2983-2987) — note `e->grad->len` bytes, the full grad, not clamped to the
param length — then halves `loss_scale`, floors it at 1e-8, resets
`stable_steps`, increments `total_nan_count` and `skipped_steps`, returns 0
(2988-2993). On a clean step it increments `stable_steps` and, every
`scale_window` clean steps, doubles `loss_scale` up to 65536 (2997-3002), and
returns 1.

This is sound as a *detector*: molequla's `guard.check()` (cgo_notorch.go:130)
returns false and the step is skipped entirely (notorch_trainer.go:354-357), so
Chuck never sees a NaN, `as->t` does not advance and the moments are not
polluted. The dynamic-loss-scale half of the mechanism is inert — see D12.

One gap worth naming: because the guard only reads *parameter* gradients, a NaN
confined to an activation gradient that happens to multiply out to a finite
parameter gradient would pass. In practice NaN propagates, so this is
theoretical.

---

## 3. `nt_tape_clear` vs `nt_tape_destroy` — the moment leak

**Confirmed. `nt_tape_destroy()` called after `nt_tape_clear()` frees nothing,
and the Chuck moments leak.**

`nt_tape_clear` (notorch.c:300-316):

```
301   for (int i = 0; i < g_tape.count; i++) {
302       if (g_tape.entries[i].output) nt_tensor_free(...);
304       if (g_tape.entries[i].grad)   { nt_tensor_free(...); ... = NULL; }
310       g_tape.entries[i].frozen = 0;
311       g_tape.entries[i].slot   = -1;
312   }
313   g_tape.count    = 0;
314   g_tape.active   = 0;
315   g_tape.n_params = 0;
```

It frees outputs and grads and **zeroes both loop counters**, including
`n_params`. It deliberately does not touch `g_tape.adam[]` — that is how
moments survive from one step's tape to the next.

`nt_tape_destroy` (notorch.c:318-336):

```
319   for (int i = 0; i < g_tape.count; i++) { ... free output, free grad ... }
329   for (int i = 0; i < g_tape.n_params; i++) {
330       if (g_tape.adam[i].m)        { nt_tensor_free(...); ... }
331       if (g_tape.adam[i].v)        { nt_tensor_free(...); ... }
332       if (g_tape.adam[i].acc_grad) { nt_tensor_free(...); ... }
333       g_tape.adam[i].t = 0;
334   }
335   memset(&g_tape, 0, sizeof(g_tape));
```

**Both loop bounds are the counters `clear` just set to zero.** After a
`clear`, `count == 0` and `n_params == 0`, so loop 319 runs zero times (harmless
— those tensors were already freed) and loop 329 runs **zero times**. Then
`memset` overwrites `g_tape.adam[i].m` / `.v` / `.acc_grad` with NULL. Every
moment buffer that was allocated in `nt_tape_param` (notorch.c:431-432, or the
resize at 435-436) becomes unreachable. Leak per destroy-after-clear:

```
Σ over registered params of 2 · param->len · sizeof(float)
```

(three tensors if `nt_tape_accum_grads` was ever used, notorch.c:2824). For
molequla that is `2 × (total content params + Σ combined Wr)` floats, i.e.
roughly twice the model's parameter bytes, lost on **every growth event**.

This is exactly the call pattern in the trainer. `ntTrainCore` ends its step
loop with `ntTapeClear()` (notorch_trainer.go:358), and the *next* burst opens
with:

```
281   if ntTapeNeedsReset {
282       ntTapeDestroy()
283       ntTapeNeedsReset = false
284   }
```

`ntTapeNeedsReset` is set by `ntOnGrowth()` (notorch_trainer.go:34), called from
molequla.go:6664 and molequla.go:7020 after `MaybeGrowArchitecture()`. So every
architecture growth leaks the full moment set. See D6.

**What `ntTapeDestroy`-then-`nt_tape_start` leaves in `g_tape`:** the `memset`
at 335 is a full-struct wipe — `entries[]`, `count`, `active`, `adam[]`,
`n_params`, `chuck` and `chuck_params[]` all become zero. `nt_tape_start`
(295-298) then calls `nt_tape_clear` (a no-op on a zeroed tape) and sets
`active = 1`. So after a destroy the tape is genuinely fresh: `adam[i].m == NULL`,
and the next `nt_tape_param` takes the **allocate** branch (430-433), producing
zero-initialised moments of the correct new length. **No stale m/v shapes are
reused after growth** — the intended S1 semantics hold. The leak is the only
defect.

**The len-mismatch branch matters when destroy is *not* called** (the normal
step-to-step path, and any shape change that did not go through `ntOnGrowth`):

```
434   } else if (g_tape.adam[pi].m->len != param->len) {
435       nt_tensor* new_m = nt_tensor_new(param->len);
437       int copy_len = min(old->len, param->len);
438       memcpy(new_m->data, old->data, copy_len * sizeof(float));
```

This is a flat prefix copy. For a 2-D parameter that grew from `Nout×Nin` to
`Nout'×Nin'`, the copied prefix lands at the wrong `(row, col)` for every row
after the first — the moments are reinterpreted row-major rather than
re-embedded. It is also the branch that silently accepts a shape change as
routine. See D16.

Slot identity across bursts otherwise holds: `clear` resets `n_params` to 0 and
each burst re-registers the same tensors in the same fixed order
(`ntContentParams`, notorch_trainer.go:50-62, explicitly documented as
order-critical at 36-39), so slot *k* receives the same parameter and keeps its
moments. That is the design and it works — conditional on the wr-layer set not
changing (D16).

---

## 4. `nt_seq_rmsnorm` gamma = -1: train ≡ infer

notorch training (notorch.c:3287, and the backward's identical `rms` at 1142):

```
rms = sqrtf(ss / D + 1e-6f);      y = x / rms
```

molequla Go inference, `RMSNorm` (molequla.go:1015-1021):

```
1016   ms := x.MeanSq()                              // Σx²/n  (molequla.go:617-624)
1017   scaleVal := math.Pow(ms.Data+1e-5, -0.5)
1020   d[i] = x.Data[i] * scaleVal
```

Formula parity: **identical** — `y = x / sqrt(mean(x²) + ε)`, non-parametric,
per-vector, no mean subtraction on either side.

Epsilon parity: **broken. 1e-6 (train) vs 1e-5 (infer), a factor of 10.**
The relative divergence is `½·ε_diff/mean(x²) = 4.5e-6/mean(x²)`. For a settled
hidden state with `mean(x²) ~ 1` it is below float32 noise. For molequla
specifically it is not automatically negligible: organisms are born as ~10K-param
embryos with small-magnitude initialisation, and when `mean(x²)` falls toward or
below `1e-5` the two functions differ materially — in the limit `x → 0`, the
trained normaliser has gain `1/√1e-6 = 1000` where inference has `1/√1e-5 ≈ 316`.
See D7.

Precision parity is a separate matter and out of scope here: training runs in
`float32` and Go inference in `float64`, and `MeanSq` sums in `float64`.

**A larger train ≢ infer break found in the same comparison — D1.** Go inference
scales both residual branches:

```
molequla.go:1859   gpt.residualAlpha = 1.0 / math.Sqrt(math.Max(1, float64(CFG.NLayer)))
molequla.go:2958   x = xRes.Add(attnOut.Scale(gpt.residualAlpha))
molequla.go:2975   x = xRes.Add(mlpOut.Scale(gpt.residualAlpha))
```

The notorch training graph does not:

```
notorch_trainer.go:129   h = ntAdd(h, ntSeqLinear(wo, attn, T))
notorch_trainer.go:134   h = ntAdd(h, ntSeqLinear(fc2, ntMul(gate, up), T))
```

For `NLayer = 4`, `residualAlpha = 0.5`: training optimises weights under a
residual branch that is twice as strong as the one inference evaluates. Every
gradient notorch computes is the exact gradient of a *different* function than
the one the organism speaks with.

---

## 5. aarch64 vs x86

**Inside the backward: nothing.** Every `#if defined(__aarch64__)` /
`__ARM_NEON` / `__AVX2__` block in notorch.c is in the quantized-inference
kernels, not in `nt_tape_backward` or any forward on the training path:
5021 (`nt_f16_to_f32`), 5148/5329 (Q4_K / f16 row kernels), 5911
(`nt_f32_to_f16_round`), 6302-7931 (SDOT / i8mm packed matvec and matmul), 6886
(`NT_QMV_PAUSE` spin hint), 8650 (`nt_f16_rows_n`). The backward is plain
scalar C plus, optionally, `cblas_sgemm`. Alignment is uniform: every tensor is
`calloc`'d (notorch.c:171) with no over-alignment requested and no aligned-load
intrinsic anywhere in the backward, so there is no alignment-dependent path.

Four things *do* diverge, none of them a correctness bug:

1. **`USE_BLAS` selects a different reduction.** `NT_OP_SEQ_MATVEC` backward has
   two implementations (notorch.c:1052-1064 BLAS vs 1065-1082 triple loop) that
   are mathematically equal and numerically different — OpenBLAS blocks and
   vectorises the `K` reduction, the fallback sums in index order. Whether a
   given build takes it is decided by the Makefile's per-OS `BLAS_FLAGS`
   (Makefile:108-113 Linux/pkg-config, 17 Darwin/Accelerate), so two nodes can
   run different reductions from the same source. Within a machine it is
   deterministic.
2. **OpenBLAS kernel selection differs by architecture.** The aarch64 and x86_64
   `sgemm` micro-kernels use different blocking and accumulator counts, so
   `dw`/`dx` differ in the low bits between phone and polygon even with BLAS on
   both.
3. **The Makefile adds different `-march` flags globally.** aarch64 gets
   `-march=armv8.2-a+dotprod+i8mm` when the host supports it (Makefile:40-58),
   x86_64 gets `-mavx2 -mfma -mf16c` (Makefile:78-95). These land in `CFLAGS`
   for the whole translation unit, so they also change FMA contraction in the
   *scalar backward loops* (`dw[i*cols+j] += dout_t[i]*x_t[j]`, notorch.c:1080;
   `dq[...] += ds*kj[d]`, notorch.c:1336). Both targets contract, but into
   different instruction shapes with different intermediate rounding. Bit-exact
   cross-node reproduction of a training run is therefore not available.
4. **libm.** `expf`, `logf`, `powf`, `cosf`, `sinf`, `tanhf` are used in both
   the forward and the backward (e.g. `powf` for the RoPE frequency at
   notorch.c:4394 *and* 2271). Within one machine the two calls receive
   identical arguments and cancel exactly, so the RoPE inverse rotation is exact
   regardless of libm quality. Across machines the angle can differ by an ulp.

**One concrete x86 assumption in the consumer.** molequla hardcodes x86_64
multiarch paths in its cgo directives:

```
cgo_notorch.go:5        #cgo linux CFLAGS: -DUSE_BLAS -I/usr/include/x86_64-linux-gnu/openblas-pthread/
cgo_notorch_cpu.go:11   #cgo linux LDFLAGS: -L/usr/local/lib -lnotorch -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lm
```

On aarch64 those directories do not exist. The include path is silently ignored
(the preamble only needs `notorch.h`), and the link falls back to whatever
`-lopenblas` resolves to on the default search path — or fails. `-DUSE_BLAS`
here affects only the cgo preamble, not the prebuilt `libnotorch.a`, so the
library's own BLAS decision was made by the Makefile at library build time and
this flag can disagree with it without any diagnostic. See D15.

---

## 6. Findings

### D1 — P0 — the training graph omits `residualAlpha`

`notorch_trainer.go:129`, `notorch_trainer.go:134` vs `molequla.go:1859`,
`molequla.go:2958`, `molequla.go:2975`; growth path `molequla.go:2371`.

Inference computes `x ← x_res + α·f(x)` with `α = 1/√NLayer`; training computes
`x ← x_res + f(x)`. The gradient notorch delivers to `model.Base` is the exact
gradient of a model whose residual branches are `√NLayer` times stronger than
the deployed one. Wrong gradient with respect to the function that runs.

**Minimal repair.** Multiply the two residual branches in `ntBuildForward` by
`residualAlpha`. notorch already has the op — `nt_scale(idx, s)` /
`NT_OP_SCALE`, forward notorch.c:4330-4357, backward notorch.c:652-673
(`ga[i] = dout[i]·aux`, correct and cheap). Add an `ntScale` binding and write:

```go
a := float32(model.residualAlpha)
h = ntAdd(h, ntScale(ntSeqLinear(wo, attn, T), a))
...
h = ntAdd(h, ntScale(ntSeqLinear(fc2, ntMul(gate, up), T), a))
```

Gate: a forward-parity check comparing `ntEntryData` on the final hidden against
the Go `ForwardStep` for the same weights and tokens — it must go red on the
current code.

### D2 — P1 — cross-entropy trains padding as token 0

Condition: any sampled document shorter than `seqLen + 1`.

`notorch_trainer.go:302-313` fills the tail of `tokBuf`/`tgtBuf` with `0` when
the document runs out. `nt_seq_cross_entropy` has no ignore index: both the
forward (notorch.c:4208) and the backward (notorch.c:1920) clamp an
out-of-range target to `0`, and `0` is a perfectly in-range token that receives
the full `−1` in `dl[t·V + target]` (notorch.c:1930). The denominator is always
`T` (notorch.c:4215, 1931), never the number of real positions. So every short
document teaches the model to emit token 0 for the rest of the block, and
dilutes the real loss by `n_real/T`.

Reachability: `start` is randomised only when `len(ids) > seqLen+1`
(notorch_trainer.go:299-301); otherwise the sequence starts at 0 and everything
past `len(ids)` is pad. With `BlockSize = 96` (molequla.go:260) and an ecology
of short utterances this fires constantly.

**Minimal repair.** notorch already has the correct op:
`nt_seq_cross_entropy_masked` / `NT_OP_SEQ_CROSSENT_MASKED` (forward
notorch.c:4222-4258, backward 1941-1991) skips `m == 0` rows entirely and
normalises by `n_active`. Bind it, build a `[]float32` mask alongside `tgtBuf`
(1 where `idx+1 < len(ids)`, else 0), register it with `ntTapeInput`, and swap
`ntSeqCrossEntropy` for the masked call. Gate: a two-document batch where one
document is half-length — masked loss must equal the loss of the unpadded
prefix computed alone.

### D3 — P1 — MUL/ADD backward drop the forward's broadcast

Condition: `len(parent2) != len(output)`.

Forward wraps (`i % pb->output->len`, notorch.c:4289 and 4323); backward does
not. For `NT_OP_MUL`, `ga[i] = dout[i] * pb->output->data[i]` (notorch.c:641)
reads **past the end of `pb->output->data`** for `i ≥ pb->len` — heap
out-of-bounds read, and a wrong `ga` even in the part that is in bounds is not
the issue; the whole tail is garbage. For `NT_OP_ADD`, `tape_acc_grad(parent2,
dout, out_len)` (notorch.c:601) either writes only the first tile (if a grad of
the parent's length already exists, via the `min` at notorch.c:508) or allocates
an oversized grad that the optimizer then reads only the head of
(notorch.c:2638-2639) — in both cases the true `db[k] = Σ_{i≡k mod L} dout[i]`
is never formed.

Not currently triggered: molequla pairs equal-length tensors everywhere
(notorch_trainer.go:127, 129, 134; gates built at the same `seqLen`,
notorch_trainer.go:195-205, 257). This is a loaded trap for the next bias vector
or scalar gate someone adds.

**Minimal repair.** In `NT_OP_MUL` backward apply the same modulo the forward
uses and reduce into the shorter buffer:

```c
int bl = pb->output->len, al = pa->output->len;
float* gb = calloc(bl, sizeof(float));
for (int i = 0; i < out_len; i++) {
    ga[i]        = dout[i] * pb->output->data[i % bl];
    gb[i % bl]  += dout[i] * pa->output->data[i % al];
}
tape_acc_grad(e->parent2, gb, bl);
```

and in `NT_OP_ADD`, when `pb->output->len != out_len`, fold `dout` into a
`pb->len` buffer before accumulating. Alternatively reject unequal lengths in
`nt_add`/`nt_mul` outright — simpler, and nothing in the tree relies on the
broadcast. Gate: a `[T·D] ⊙ [D]` product whose analytic `db` is compared against
a finite-difference estimate.

### D4 — P1 — RRPRAM-LR rank derived from `T`; no `T == T_max` assertion

Condition: `T != BlockSize` on any call to `nt_rrpram_lowrank_attention`.

`int T_r = T;` (forward notorch.c:3689, backward notorch.c:1443) and
`rank = combined_len / (nr·(n_embd + T_r))` (3691, 1445). The packed buffer is
produced at `T_max = BlockSize` (molequla.go:894, notorch_trainer.go:144-154).
When `T < T_max`, `rank` resolves too large and both passes reinterpret the
whole `Wr_a | Wr_b` layout; when `T > T_max`, too small or zero. No bounds are
violated — `rank` is a floor, so the largest index is
`nr·rank·(n_embd+T) − 1 ≤ combined_len − 1` — so there is no crash, only wrong
numbers written back through `ntUnpackWr` (notorch_trainer.go:158-170).

Today it is safe only because `ntTrainCore` pins `seqLen = model.BlockSize`
when any hybrid head exists (notorch_trainer.go:224-226). That pin is one
`if` away from being lost, and `model.BlockSize` is mutable at runtime
(molequla.go:6327-6330 caps it inside `trainSteps`, with a deferred restore) —
currently on a different trainer and under the same mutex as growth, so the
windows do not overlap, but nothing enforces that.

**Minimal repair.** Make the assumption checkable in notorch rather than only in
the caller. Either take `T_r` as an explicit argument, or assert the layout:

```c
if (combined_len % ((long)nr_heads * (n_embd + T_r)) != 0) return -1;
```

in `nt_rrpram_lowrank_attention` (after 3691) and mirror the same rejection in
the backward before the head loop (after notorch.c:1445), leaving `dwr` zero.
Gate: call the op with `T = BlockSize/2` on a buffer packed at `BlockSize` and
require `-1`.

### D5 — P1 — Chuck's per-param auto-freeze is permanent

Condition: a parameter whose gradient norm stays below `NT_CHUCK_FREEZE_THRESH`
(0.01, notorch.h:178) for `NT_CHUCK_STAG_STEPS` (8, notorch.h:175) consecutive
steps, once its ring buffer has filled to 8.

```
2675   if (gnorm < NT_CHUCK_FREEZE_THRESH) {
2676       cp->stag++;
2677       if (cp->stag >= NT_CHUCK_STAG_STEPS) cp->frozen = 1;
```

and at the top of every subsequent step:

```
2635   if (cp->frozen) continue;
```

`cp->frozen` is written to 1 at notorch.c:2677 and notorch.c:465 and is
**never written back to 0** anywhere in the file (`grep -n 'chuck_params\|
cp->frozen' notorch.c` returns exactly 465, 2633, 2635, 2677). `nt_tape_clear`
does not touch `g_tape.chuck_params[]` (notorch.c:300-316), so the flag survives
every step and every burst. Only the `memset` in `nt_tape_destroy`
(notorch.c:335) clears it — i.e. only on a growth event.

The threshold is absolute, not relative to the parameter's scale or count, so a
small or well-converged tensor — `wpe`, a low-rank `Wr` under `no_decay`, a
narrow embryo matrix — crosses it easily and then stops learning for the rest of
the process while the loss and the logs look normal.

**Minimal repair.** Make the freeze recoverable: clear it when the gradient
returns.

```c
if (gnorm < NT_CHUCK_FREEZE_THRESH) { cp->stag++; if (cp->stag >= NT_CHUCK_STAG_STEPS) cp->frozen = 1; }
else { cp->stag = 0; cp->frozen = 0; }
```

(the `else` at notorch.c:2678-2680 already resets `stag`; add the `frozen`
reset there). Guard it so it cannot un-freeze a param frozen deliberately via
`nt_tape_freeze_param` — those have `e->frozen = 1` and never reach this loop
with a grad anyway, but an explicit `if (!e->frozen)` makes it safe. Gate: feed
one param nine steps of zero gradient then a large one, and require the param
to move.

### D6 — P1 — `destroy` after `clear` frees no moments (resource, not gradient)

Confirmed, with loop bounds cited in §3. `nt_tape_clear` zeroes `count` and
`n_params` (notorch.c:313, 315); `nt_tape_destroy` bounds both its loops by
those same counters (notorch.c:319, 329) and then `memset`s the struct
(notorch.c:335). Every `adam[i].m` / `.v` / `.acc_grad` allocated at
notorch.c:431-432 / 435-436 / 2824 is leaked. Triggered on every growth event
via `ntTapeNeedsReset` → `ntTapeDestroy()` (notorch_trainer.go:281-284), which
always follows the previous burst's `ntTapeClear()` (notorch_trainer.go:358).

Severity note: this is a leak, not a wrong gradient. It is P1 because on an 8 GB
phone running a growing colony it is unbounded in the number of growth events.

**Minimal repair.** Do not derive the free bounds from counters that `clear`
resets. Iterate the fixed array:

```c
void nt_tape_destroy(void) {
    for (int i = 0; i < NT_TAPE_MAX_ENTRIES; i++) { ...free output, grad... }
    for (int i = 0; i < NT_TAPE_MAX_PARAMS; i++) {
        if (g_tape.adam[i].m)        { nt_tensor_free(g_tape.adam[i].m);        g_tape.adam[i].m = NULL; }
        if (g_tape.adam[i].v)        { nt_tensor_free(g_tape.adam[i].v);        g_tape.adam[i].v = NULL; }
        if (g_tape.adam[i].acc_grad) { nt_tensor_free(g_tape.adam[i].acc_grad); g_tape.adam[i].acc_grad = NULL; }
        g_tape.adam[i].t = 0;
    }
    memset(&g_tape, 0, sizeof(g_tape));
}
```

The entries loop is safe over the full array because `clear` NULLs the pointers
it frees (notorch.c:306 for grad; note it does **not** NULL `output`, so that
loop must keep the `count` bound or `clear` must also NULL `output` — the
latter is the cleaner fix, one line at notorch.c:303). Gate: a
start/param/clear/destroy cycle under a leak checker, or simply assert that
`adam[0].m` was freed by tracking the allocation count.

### D7 — P1 — RMSNorm epsilon 1e-6 (train) vs 1e-5 (infer)

notorch.c:3287 and notorch.c:1142 vs molequla.go:1017. Formula is otherwise
identical (§4). Diverges materially when `mean(x²) ≲ 1e-5`, which embryo-stage
organisms reach.

**Minimal repair.** One of the two moves. The Go side is the deployed function
and the C side is shared by other organisms, so the lower-risk edit is to make
notorch's epsilon a parameter of the op rather than a literal — `nt_seq_rmsnorm`
already has an unused `aux3`/`aux4` pair available through `nt_tape_record4` —
and have molequla pass `1e-5`. Failing that, change molequla.go:1017 to `1e-6`
and note it in the log; nothing in the Go tree depends on the value.
Gate: a parity test comparing `ntEntryData` of the first `ntSeqRMSNorm` against
Go `RMSNorm` on a deliberately small-magnitude vector (`mean(x²) ≈ 1e-6`),
tolerance tight enough to fail today.

### D8 — P2 — params beyond `NT_TAPE_MAX_PARAMS` get grads but no update

`nt_tape_param` (notorch.c:427): the slot allocation is inside
`if (g_tape.n_params < NT_TAPE_MAX_PARAMS)`. When the limit (512, notorch.h:88)
is reached the entry keeps `is_param = 1`, `frozen = 0`, `slot = -1`. It then:
accumulates gradients normally (`tape_acc_grad` has no slot check,
notorch.c:495); **enters the global clip norm** (notorch.c:2776 filters on
`is_param && grad`, not on `slot`); and is skipped by every optimizer loop
(notorch.c:2630, 2442, 2470, 2819). So it shrinks everyone else's update while
never being trained, silently.

molequla registers `3 + 8·NLayer` params (notorch_trainer.go:50-62 plus the
per-layer `wr`, 336), so `NLayer > 63` is required — not reachable today.

**Minimal repair.** Either return `-1` from `nt_tape_param` when the pool is
exhausted (callers already handle `-1` by collapsing the chain), or exclude
`slot < 0` entries from `nt_tape_clip_grads` so a no-slot param cannot distort
the norm. The first is honest; the second is defensive. Gate: register 513
params and require the 513th to be rejected.

### D9 — P2 — `nt_tape_record` / `record3` leave `aux3` / `aux4` stale

notorch.c:341-360 and 362-381 set `aux` and `aux2` and reset `is_param`,
`no_decay`, `frozen`, `slot`, but never `aux3`/`aux4`. `g_tape.entries[]` is
reused across tape sessions, so an entry index previously written by
`nt_tape_record4` (RoPE at 4407/4460, RRPRAM at 3720/3775/3656/3889, GQA at
3615, SEQ_GATE at 3394) leaves its `aux3`/`aux4` behind.

Currently benign: every op that reads `aux3`/`aux4` in the backward
(notorch.c:776, 1360-1361, 1441, 1605-1606, 1734, 2246-2247) is recorded with
`record4`. It is a trap for the next op added with `record3` that grows a third
parameter.

**Minimal repair.** Add `e->aux3 = 0; e->aux4 = 0;` to `nt_tape_record` (after
353) and `nt_tape_record3` (after 374).

### D10 — P2 — `NT_OP_SEQ_CROSSENT` backward dereferences an unchecked parent2

notorch.c:1886-1888:

```c
if (e->parent1 >= 0) {
    nt_tape_entry* pl = &g_tape.entries[e->parent1];
    nt_tape_entry* pt = &g_tape.entries[e->parent2];
```

`parent2 == -1` forms `&entries[-1]` and the `if (dl && pt)` guard at 1916 never
catches it (`pt` is an address, never NULL); `pt->output->data[t]` at 1919 then
reads before the array. Not reachable through `nt_seq_cross_entropy`, which
rejects `targets_idx < 0` (notorch.c:4179).

**Minimal repair.** Change the case guard at 1886 to
`if (e->parent1 >= 0 && e->parent2 >= 0)`, matching the masked variant
(notorch.c:1942).

### D11 — P2 — Chuck clamps by `m->len`, never by `grad->len`

notorch.c:2638-2639 sets `n = min(e->output->len, as->m->len)` and then reads
`e->grad->data[j]` for `j < n` (2715, and 2654 in the norm loop) without
consulting `e->grad->len`. Because `as->m` is allocated at `param->len`
(notorch.c:431), `n == param->len`, so any grad shorter than its parameter is
read out of bounds. `nt_tape_clip_grads` does clamp correctly (2777-2778,
2791-2792), which makes the inconsistency easy to miss.

A short grad is produced whenever a backward calls `tape_acc_grad(param_idx,
buf, len)` with `len < param->len` on the *first* accumulation of the step —
e.g. `NT_OP_SEQ_RMSNORM` passes `D` for gamma (notorch.c:1163) where `D` is
`aux2`, not `gamma->len`. Not reachable from molequla (gamma is `-1`).

**Minimal repair.** `if (e->grad->len < n) n = e->grad->len;` after
notorch.c:2639, and the same in `nt_tape_adam_step` (2448) and
`nt_tape_adamw_step` (2476).

### D12 — P2 — the NaN guard's loss scale is computed and never consumed

`guard->loss_scale` is initialised (notorch.c:2942), halved on NaN (2988) and
doubled on stability (2999), and **nothing reads it** —
`grep -n loss_scale notorch.c notorch.h` returns only those writes plus the
field declaration (notorch.h:305) and two comment lines. molequla's binding
discards the whole struct except the return code (cgo_notorch.go:130). The
mixed-precision machinery the field implies does not exist on this path.

**Minimal repair.** Either delete the scaling half of `nt_nan_guard_check` and
document it as a detector, or expose `loss_scale` through the cgo bridge and
apply it. Do not leave a knob that reads as working.

### D13 — P2 — `NT_OP_SEQ_EMBED` backward allocates a dense `V × D` gradient per step

notorch.c:974 `calloc(pwte->output->len)` then notorch.c:983
`tape_acc_grad(parent1, dwte, pwte->output->len)` — a full vocabulary-sized zero
buffer, written at `T ≤ 96` rows, then added element-wise into an equally large
tape grad. Two full passes over `V·D` floats every step to deposit at most
`T·D` non-zeros. For a 10K-token vocabulary at `D = 128` that is 5 MB of
`calloc` + `memset` + add per step. Same shape of waste in `nt_tape_clip_grads`
and Chuck, which then norm and update the whole dense `wte` grad.

**Minimal repair.** Accumulate the touched rows directly into `e->grad` instead
of staging a dense buffer — add a `tape_acc_grad_rows(idx, row, vec, D)` helper
that allocates `e->grad` at `pwte->len` once and adds `D` floats at
`row·D`. Measure before and after on the A56 big cores and put the number in
`NOTORCHLOG.md`.

### D14 — P2 — RoPE drops the gradient of the unpaired tail

notorch.c:2264-2292: `gx` is `calloc`'d and written only at `base + 2i` and
`base + 2i + 1` for `i < head_dim/2`. When `head_dim` is odd, index
`base + head_dim - 1` is never written; when `D % head_dim != 0`, the trailing
`D − n_heads·head_dim` columns of every position are never written. The forward
passes exactly those elements through unchanged (`memcpy` at notorch.c:4389 with
no rotation applied), so their true gradient is `dout`, and the backward
delivers `0`.

Not reachable from molequla: `headDim = D / model.NHead` (notorch_trainer.go:104)
makes `D % head_dim == 0` by construction, and `nt_mh_causal_attention` rejects
`D % head_dim != 0` anyway (notorch.c:3507). `nt_rope_freq` has no such check.

**Minimal repair.** Initialise the pass-through: `memcpy(gx, dout, total *
sizeof(float))` instead of `calloc` at notorch.c:2264, then overwrite the
rotated pairs as now. That makes the backward the exact transpose of a forward
that is identity outside the pairs. Add the `D % head_dim` reject to
`nt_rope_freq` for good measure.

### D15 — P2 — build-time divergence, and x86 paths hardcoded in the consumer

Detailed in §5. Nothing in the backward is architecture-conditional; the
divergence is `USE_BLAS` selection (notorch.c:1052 vs 1065), OpenBLAS kernels,
the per-arch `-march` in Makefile:40-58 / 78-95, and libm.

The actionable half is molequla's cgo directives: `cgo_notorch.go:5` and
`cgo_notorch_cpu.go:11` name `x86_64-linux-gnu` explicitly, which on aarch64
resolves to nothing and leaves the link to chance.

**Minimal repair.** Replace both with `pkg-config`, matching what the notorch
Makefile already does (Makefile:107-112):

```go
#cgo linux pkg-config: openblas
```

or gate the multiarch directory on `GOARCH` with build-tagged files, as the
CUDA split already does (`cgo_notorch_cpu.go` / `cgo_notorch_cuda.go`). And
record in the log which BLAS the phone's `libnotorch.a` was actually built
with — the cgo `-DUSE_BLAS` does not tell you.

### D16 — P2 — slot identity depends on the wr-layer set; the resize branch reinterprets moments

Two coupled hazards.

1. `ntTrainCore` registers a Chuck slot for layer `l` only when
   `wrTensors[l] != nil` (notorch_trainer.go:331-336). If the set of layers with
   `wr` factors changes between bursts without a `ntTapeDestroy` in between,
   slot *k* silently becomes a different parameter and inherits the previous
   one's moments. Growth deletes and reallocates all `wr` matrices
   (molequla.go:2411-2413) and does call `ntOnGrowth` (molequla.go:6664, 7020),
   so the current paths are covered — by convention, not by construction.
2. `nt_tape_param`'s len-mismatch branch (notorch.c:434-444) resizes `m`/`v`
   and `memcpy`s `min(old, new)` floats. That is a flat prefix copy, so for a 2-D
   parameter that changed either dimension the surviving moments land at the
   wrong `(row, col)`. It also means a shape change is accepted silently rather
   than flagged.

**Minimal repair.** Store the registered tensor pointer (or a caller-supplied
stable id) in `nt_adam_state` and reset `m`/`v` to zero whenever slot *k* is
re-registered with a different one. Zeroing is the honest answer to a shape
change anyway — the prefix copy has no defensible semantics.

### D17 — P2 — a full tape degrades into a silent no-op step

`nt_tape_record*` and `nt_tape_param` return `-1` when
`g_tape.count >= NT_TAPE_MAX_ENTRIES` (notorch.c:342, 363, 384, 407). Every
forward op returns `-1` on a negative input index (e.g. notorch.c:3123, 3257,
4261), so the whole chain collapses to `-1`, `nt_seq_cross_entropy` returns
`-1`, and `nt_tape_backward(-1)` returns immediately (notorch.c:548). molequla
then reads `ntEntryScalar(-1)` → `0.0` (cgo_notorch.go:20-21), counts it as a
clean step with loss 0 (notorch_trainer.go:352, 360-363), and drags the reported
average down while training nothing.

molequla's per-step entry count is roughly `3 + 8·NLayer` params +
`2·NLayer` frozen gates + 2 inputs + ~20 ops per layer + 3, so `NLayer ≈ 270`
would be needed — not reachable today.

**Minimal repair.** Have `ntTrainCore` treat `lossIdx < 0` as a hard error
(skip the step, count it, print once), and make `nt_tape_record*` log the
overflow the first time it happens rather than returning `-1` mutely.

---

## 7. What was checked and found clean

Recorded so the next pass does not re-derive it.

- Gradient accumulation is `+=` everywhere on the CPU path; no backward case
  overwrites a parent's gradient (single funnel, notorch.c:509).
- Reverse-index traversal is a valid topological order (parents always have
  lower indices; notorch.c:343, 364, 385, 408).
- No op in the molequla chain ignores a parent it should differentiate. `v` is
  correctly shared between `nt_mh_causal_attention` and
  `nt_rrpram_lowrank_attention` and receives the sum of both gradients.
- Frozen gate vectors are correctly excluded from grad allocation
  (notorch.c:498), from the clip norm (2776), from the NaN guard (2954) and from
  the optimizer (2630), and consume no Chuck slot (notorch.c:486-489).
- The HEAD fix holds: `e->slot` is the only path from a tape entry to its
  optimizer state (notorch.c:429, 2442, 2470, 2630, 2819), and
  `nt_tape_param_frozen` does not advance `n_params` (notorch.c:488).
- `nt_tape_clear` correctly preserves `adam[]` across steps so moments survive
  the per-step tape rebuild, and re-registration in fixed order restores slot
  identity (notorch_trainer.go:36-39, 50-62).
- RoPE backward is the exact inverse rotation for both conventions; pairing,
  `head_dim`, `n_heads`, `freq_base` and the angle expression are recovered
  identically from `aux`/`aux2`/`aux3`/`aux4`.
- Attention backward applies `1/√head_dim` exactly once, in the right place, and
  honours causality structurally in every loop bound.
- Cross-entropy backward is consistent with its forward on the `1/T` mean and
  on integer-target truncation.
- `nt_nan_guard_check` runs before `nt_tape_clip_grads` in molequla, which is
  the order that prevents a NaN from becoming a NaN scale factor.
- Nothing in the backward path is SIMD, aligned-load or architecture-guarded.

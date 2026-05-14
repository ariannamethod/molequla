// notorch.h — PyTorch replacement in pure C
// Train and run neural networks without Python.
//
// Extracted from ariannamethod.ai/core/ (Arianna Method)
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// "fuck torch"

#ifndef NOTORCH_H
#define NOTORCH_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR — multi-dimensional array with refcounting + optional GPU
// ═══════════════════════════════════════════════════════════════════════════════

#define NT_MAX_DIMS     8
#define NT_MAX_ELEMENTS (1 << 28)  // 268M floats max per tensor (Qwen-0.5B vocab×embed = 136M)

typedef struct {
    float*   data;              // CPU data (heap-allocated)
    int      ndim;              // number of dimensions (1..NT_MAX_DIMS)
    int      shape[NT_MAX_DIMS];// shape[0] = outermost, shape[ndim-1] = innermost
    int      stride[NT_MAX_DIMS];
    int      len;               // total number of elements (product of shape)
    int      refcount;
#ifdef USE_CUDA
    float*   d_data;            // GPU device pointer
    int      gpu_valid;         // 1 = GPU copy is up to date with last write
    int      cpu_dirty;         // 1 = GPU was last writer, CPU mirror stale, needs ensure_cpu
#endif
} nt_tensor;

// Create a 1D tensor of given length, zeroed
nt_tensor* nt_tensor_new(int len);

// Create a 2D tensor (rows × cols), zeroed
nt_tensor* nt_tensor_new2d(int rows, int cols);

// Create a tensor from shape array
nt_tensor* nt_tensor_new_shape(const int* shape, int ndim);

// Free tensor (decrements refcount, frees at 0)
void nt_tensor_free(nt_tensor* t);

// Increment refcount (for shared references)
nt_tensor* nt_tensor_ref(nt_tensor* t);

// Deep copy
nt_tensor* nt_tensor_clone(const nt_tensor* src);

// Fill with value
void nt_tensor_fill(nt_tensor* t, float val);

// Fill with random uniform [-scale, scale]
void nt_tensor_rand(nt_tensor* t, float scale);

// Fill with Xavier/Kaiming init
void nt_tensor_xavier(nt_tensor* t, int fan_in, int fan_out);

// Kaiming uniform initialization (variance = 1 / fan_in).
// Uniform in [-sqrt(3/fan_in), +sqrt(3/fan_in)] — variance a²/3 = 1/fan_in.
// Used for LoRA A matrix init (per-fan_in scale, NOT per-rank).
void nt_kaiming_uniform_init(nt_tensor* t, int fan_in);

// Reshape in-place (total elements must match). Returns 0 on success.
int nt_tensor_reshape(nt_tensor* t, const int* new_shape, int new_ndim);

// Sync GPU mirror to CPU. No-op on non-USE_CUDA builds. Use before reading
// param->data after a Chuck/Adam step on GPU (e.g. saving LoRA adapter).
void nt_tensor_sync_cpu(nt_tensor* t);

// Print tensor info (shape, first/last few values)
void nt_tensor_print(const nt_tensor* t, const char* name);

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOGRAD TAPE — reverse-mode automatic differentiation
// ═══════════════════════════════════════════════════════════════════════════════

#define NT_TAPE_MAX_ENTRIES  8192
#define NT_TAPE_MAX_PARAMS    512

// Tape operation types
#define NT_OP_NONE            0
#define NT_OP_MATVEC          1   // y = W @ x
#define NT_OP_ADD             2   // y = a + b
#define NT_OP_MUL             3   // y = a * b (element-wise)
#define NT_OP_SCALE           4   // y = a * scalar
#define NT_OP_SOFTMAX         5   // y = softmax(x)
#define NT_OP_RMSNORM         6   // y = rmsnorm(x, gamma)
#define NT_OP_SILU            7   // y = silu(x) = x * sigmoid(x)
#define NT_OP_CROSS_ENT       8   // loss = -log(softmax(logits)[target])
#define NT_OP_EMB_LOOKUP      9   // y = wte[token_id, :]
#define NT_OP_MATMUL         10   // C = A @ B
#define NT_OP_SEQ_EMBED      11   // h = wte[tokens] + wpe[positions]
#define NT_OP_SEQ_MATVEC     12   // Y[t] = W @ X[t] for T positions
#define NT_OP_SEQ_RMSNORM    13   // rmsnorm each position independently
#define NT_OP_CAUSAL_ATTN    14   // causal self-attention over T positions
#define NT_OP_SEQ_CROSSENT   15   // cross-entropy over T positions
#define NT_OP_MH_CAUSAL_ATTN 16   // multi-head causal self-attention
#define NT_OP_GEGLU          17   // y = GELU(x @ W1) * (x @ W2) — Gemma-3 FFN
#define NT_OP_ROPE           18   // rotary position embedding
#define NT_OP_DROPOUT        19   // zero mask with probability p
#define NT_OP_LAYERNORM      20   // (x - mean) / sqrt(var + eps) * gamma + beta
#define NT_OP_SEQ_LAYERNORM  21   // layernorm per position
#define NT_OP_GELU           22   // GELU activation
#define NT_OP_GQA_ATTN       23   // grouped-query causal attention
#define NT_OP_RRPRAM_ATTN    24   // RRPRAM positional attention (x @ Wr, causal)
#define NT_OP_CONCAT         25   // concatenate two tensors per position
#define NT_OP_SEQ_MATVEC_T  26   // Y[t] = W^T @ X[t] — transposed seq_linear for Janus Echo
#define NT_OP_SIGMOID       27   // y = 1 / (1 + exp(-x)) — logistic activation
#define NT_OP_SCALE_BY_T    28   // y[i] = a[0] * x[i], a is scalar tensor [1]
#define NT_OP_SWIGLU        29   // y = SiLU(gate) * up (element-wise, pre-computed tensors)
#define NT_OP_BIT_LINEAR    30   // y = bitquant(W) @ x — BitNet 1.58, STE backward
#define NT_OP_BIT_SEQ_LINEAR 31  // Y[t] = bitquant(W) @ X[t] for T positions (BitNet seq)
#define NT_OP_SEQ_CROSSENT_MASKED 32  // masked sequence cross-entropy (parent3 = mask)
#define NT_OP_RRPRAM_LR     33   // low-rank RRPRAM (Wr = Wr_a × Wr_b packed in one tensor)
#define NT_OP_RRPRAM_BCAST  34   // broadcast RRPRAM — mid[h,r] = Σ_t x[t]·Wr_a[h] (canonical Janus pattern, sc=1/sqrt(D))

typedef struct {
    nt_tensor* output;          // forward result
    nt_tensor* grad;            // gradient (allocated on backward)
    int        op;              // NT_OP_* type
    int        parent1;         // index into tape (-1 = none)
    int        parent2;
    int        parent3;
    float      aux;             // auxiliary scalar (target for CE, scale for SCALE, T for seq)
    float      aux2;            // second auxiliary (D for seq ops, V for seq_crossent)
    float      aux3;            // third auxiliary (n_heads for GQA)
    float      aux4;            // fourth auxiliary (n_kv_heads for GQA)
    int        is_param;        // 1 = trainable parameter
    int        no_decay;        // 1 = skip weight decay (embeddings)
    int        frozen;          // 1 = skip backward computation (frozen base in LoRA)
} nt_tape_entry;

// Adam optimizer state per parameter
typedef struct {
    nt_tensor* m;               // first moment
    nt_tensor* v;               // second moment
    nt_tensor* acc_grad;        // gradient accumulation buffer
    int        t;               // timestep counter
} nt_adam_state;

// ── Chuck optimizer — self-aware Adam ──
// θ -= (α × λ × λ_l) × m̂/(√v̂ + ε) + η
// github.com/iamolegataeff/chuck.optimizer

// Synced with PyTorch chuck.py (iamolegataeff/chuck.optimizer) 2026-04-06
#define NT_CHUCK_WINDOW      16
#define NT_CHUCK_DAMP_LO     0.3f
#define NT_CHUCK_DAMP_HI     2.0f
#define NT_CHUCK_DAMP_DOWN   0.97f    // was 0.95, PyTorch = 0.97 (less aggressive)
#define NT_CHUCK_DAMP_UP     1.03f    // was 1.05, PyTorch = 1.03 (less aggressive)
#define NT_CHUCK_TREND_BRAKE  0.02f   // loss rising > 2% → brake
#define NT_CHUCK_TREND_PUSH  -0.02f   // loss falling > 2% → push (symmetric)
#define NT_CHUCK_STAG_THRESH 0.001f
#define NT_CHUCK_STAG_STEPS  8
#define NT_CHUCK_NOISE_MAG   0.001f
#define NT_CHUCK_NOISE_DECAY 0.9f     // exponential noise decay per step
#define NT_CHUCK_FREEZE_THRESH 0.01f
#define NT_CHUCK_MACRO_INT   1000     // was 500, PyTorch = 1000
#define NT_CHUCK_MACRO_PAT   3
#define NT_CHUCK_MACRO_DECAY 0.5f
#define NT_CHUCK_MEAN_REVERT 0.999f   // dampen → 1.0 EMA (prevents drift)

typedef struct {
    float grad_hist[NT_CHUCK_WINDOW];
    float dampen;
    int   frozen;
    int   pos;
    int   full;
    int   stag;
} nt_chuck_param_state;

typedef struct {
    float loss_hist[NT_CHUCK_WINDOW];
    float dampen;
    float noise;
    float loss_ema;
    float macro_ema;
    float best_macro;
    float lr_scale;
    int   macro_stag;
    int   global_step;
    int   pos;
    int   full;
    int   stag;
    int   initialized;
} nt_chuck_state;

// The tape itself
typedef struct {
    nt_tape_entry entries[NT_TAPE_MAX_ENTRIES];
    int           count;
    int           active;

    nt_adam_state  adam[NT_TAPE_MAX_PARAMS];
    int           n_params;

    nt_chuck_state       chuck;
    nt_chuck_param_state chuck_params[NT_TAPE_MAX_PARAMS];
} nt_tape;

// ── Tape API ──

void     nt_tape_start(void);
void     nt_tape_clear(void);
void     nt_tape_destroy(void);
int      nt_tape_is_active(void);
nt_tape* nt_tape_get(void);

// Record operations on tape (returns entry index)
int  nt_tape_record(nt_tensor* output, int op, int p1, int p2, float aux);
int  nt_tape_record3(nt_tensor* output, int op, int p1, int p2, int p3, float aux, float aux2);
int  nt_tape_record4(nt_tensor* output, int op, int p1, int p2, int p3, float aux, float aux2, float aux3, float aux4);
int  nt_tape_param(nt_tensor* param);
void nt_tape_no_decay(int idx);   // mark param as no-decay (embeddings)
void nt_tape_freeze_param(int param_idx);  // post-registration freeze: entry + Chuck slot

// Register a tape entry for `param` WITHOUT allocating a Chuck optimizer slot.
// Sets entry->frozen=1 so backward skips dw accumulation. Use this for base
// weights when downstream LoRA A,B (or other adapters) are the actual trainable
// params: keeps Chuck's chuck_params[] array 1:1 with TRULY trainable entries
// only, so optimizer slot indexing stays correct.
//
// Returns the tape entry index (mirrors nt_tape_param's return contract).
// Contrast with nt_tape_param() + nt_tape_freeze_param() which still consume
// a Chuck slot and can mis-route LoRA grads in current Chuck step indexing.
int  nt_tape_param_frozen(nt_tensor* param);

// Backward pass
void nt_tape_backward(int loss_idx);

// Optimizers
void  nt_tape_adam_step(float lr);
void  nt_tape_adamw_step(float lr, float weight_decay, float beta1, float beta2);
void  nt_tape_chuck_step(float lr, float loss_val);

// Gradient utilities
float nt_tape_clip_grads(float max_norm);
void  nt_tape_accum_grads(void);
void  nt_tape_apply_accum(int n_accum);

// ── GPU mode toggle ──
// When on (1), hot tape ops (seq_linear / seq_linear_t / seq_rmsnorm / silu /
// swiglu / add / mh_causal_attention / seq_cross_entropy) dispatch to CUDA via
// notorch_cuda.{h,cu}. Default = off (CPU path). Compiled out when USE_CUDA
// is undefined. Caller is responsible for gpu_init() / gpu_shutdown().
void nt_set_gpu_mode(int on_off);
int  nt_get_gpu_mode(void);

// ═══════════════════════════════════════════════════════════════════════════════
// LR SCHEDULE — warmup + cosine annealing + step decay
// ═══════════════════════════════════════════════════════════════════════════════

#define NT_SCHED_NONE     0
#define NT_SCHED_COSINE   1   // cosine annealing to min_lr
#define NT_SCHED_STEP     2   // multiply by gamma every step_size steps
#define NT_SCHED_LINEAR   3   // linear decay to min_lr

typedef struct {
    int   type;               // NT_SCHED_*
    float base_lr;            // starting learning rate
    float min_lr;             // floor (default 0)
    int   warmup_steps;       // linear warmup from min_lr to base_lr
    int   total_steps;        // for cosine/linear: total training steps
    // Step decay params
    int   step_size;          // decay every N steps (NT_SCHED_STEP)
    float step_gamma;         // multiply factor (NT_SCHED_STEP, default 0.1)
    // State
    int   current_step;
} nt_schedule;

// Create schedule
nt_schedule nt_schedule_cosine(float base_lr, int warmup_steps, int total_steps, float min_lr);
nt_schedule nt_schedule_step(float base_lr, int warmup_steps, int step_size, float gamma);
nt_schedule nt_schedule_linear(float base_lr, int warmup_steps, int total_steps, float min_lr);

// Get current LR and advance step
float nt_schedule_get_lr(nt_schedule* s);

// ═══════════════════════════════════════════════════════════════════════════════
// NaN/Inf GUARD — detect divergence, auto loss scaling
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    float loss_scale;         // dynamic loss scale (starts at 1.0)
    float scale_factor;       // multiply/divide by this (default 2.0)
    int   stable_steps;       // consecutive clean steps
    int   scale_window;       // increase scale after this many clean steps (default 100)
    int   total_nan_count;    // lifetime NaN detections
    int   skipped_steps;      // steps skipped due to NaN
} nt_nan_guard;

// Initialize guard
nt_nan_guard nt_nan_guard_new(void);

// Check gradients for NaN/Inf. Returns 1 if clean, 0 if NaN detected.
// On NaN: zeros grads, halves loss_scale, increments skipped_steps.
// On clean: increments stable_steps, doubles loss_scale if stable enough.
int nt_nan_guard_check(nt_nan_guard* guard);

// ═══════════════════════════════════════════════════════════════════════════════
// TRAINING MODE — dropout needs this
// ═══════════════════════════════════════════════════════════════════════════════

void nt_train_mode(int training);   // 1 = training, 0 = eval
int  nt_is_training(void);

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD OPS — record on tape and compute forward pass
// All return tape entry index.
// ═══════════════════════════════════════════════════════════════════════════════

// Embedding lookup: y = wte[token_id, :]
int nt_embedding(int wte_idx, int token_id);

// Sequence embedding: h[t] = wte[tokens[t]] + wpe[t]
int nt_seq_embedding(int wte_idx, int wpe_idx, int tokens_idx, int T, int D);

// Linear: y = W @ x (+ bias if bias_idx >= 0)
int nt_linear(int w_idx, int x_idx, int bias_idx);

// Sequence linear: Y[t] = W @ X[t] for t=0..T-1
int nt_seq_linear(int w_idx, int x_idx, int T);

// Transposed sequence linear: Y[t] = W^T @ X[t] — Janus Echo W^T·W
// W is [rows, cols]. Computes Y[t] = W^T @ X[t] where X[t] has rows elements,
// output Y[t] has cols elements. Same W, gradient flows through both passes.
int nt_seq_linear_t(int w_idx, int x_idx, int T);

// RMSNorm: y = x / rms(x) * gamma
int nt_rmsnorm(int x_idx, int gamma_idx);

// Sequence RMSNorm: normalize each of T positions independently
int nt_seq_rmsnorm(int x_idx, int gamma_idx, int T, int D);

// SiLU activation: y = x * sigmoid(x)
int nt_silu(int x_idx);

// Sigmoid activation: y = 1 / (1 + exp(-x))
int nt_sigmoid(int x_idx);

// Broadcast scale: y[i] = a[0] * x[i], where a is a scalar tensor (shape [1]).
// Grad flows to both x (gx = a*gy) and a (ga = sum(gy*x)).
int nt_scale_by_t(int x_idx, int a_idx);

// GEGLU: y = GELU(x @ W1) * (x @ W2) — Gemma-3 style FFN
int nt_geglu(int x_idx, int w1_idx, int w2_idx, int T, int D_in, int D_out);

// Softmax
int nt_softmax(int x_idx);

// Causal self-attention (single head)
int nt_causal_attention(int q_idx, int k_idx, int v_idx, int T, int D);

// Multi-head causal self-attention
int nt_mh_causal_attention(int q_idx, int k_idx, int v_idx, int T, int head_dim);

// Grouped-Query Attention (GQA): Q has n_heads, K/V have n_kv_heads
// Q: [T, n_heads * head_dim], K/V: [T, n_kv_heads * head_dim]
int nt_gqa_causal_attention(int q_idx, int k_idx, int v_idx, int T, int head_dim, int n_heads, int n_kv_heads);

// Cross-entropy loss (single position)
int nt_cross_entropy(int logits_idx, int target);

// Sequence cross-entropy loss (T positions)
int nt_seq_cross_entropy(int logits_idx, int targets_idx, int T, int V);

// Masked sequence cross-entropy: loss only on positions where mask[t] == 1.
// mask tensor must have T float elements; gradient zeroed on positions with mask=0.
int nt_seq_cross_entropy_masked(int logits_idx, int targets_idx, int mask_idx, int T, int V);

// Element-wise add
int nt_add(int a_idx, int b_idx);

// Element-wise multiply
int nt_mul(int a_idx, int b_idx);

// Scale by scalar
int nt_scale(int x_idx, float s);

// RoPE: apply rotary position embeddings in-place (default freq_base = 10000)
int nt_rope(int x_idx, int T, int head_dim);

// RoPE with explicit freq_base — Qwen2 uses 1000000, Llama uses 10000
int nt_rope_freq(int x_idx, int T, int head_dim, float freq_base);

// Split-half RoPE — pairs (i, i+head_dim/2) instead of even/odd (2i, 2i+1).
// Sign convention matches canonical Janus rope_pos (infer_v4.c:35-49):
//   q[i]      =  q0*cos + q1*sin
//   q[i+half] = -q0*sin + q1*cos
// Used by nanochat / Janus v4 / similar split-half-trained bases.
// CPU-only forward; GPU backward dispatches only for even/odd, split-half
// falls back to CPU.
int nt_rope_split_half_freq(int x_idx, int T, int head_dim, float freq_base);

// Dropout: zero random elements with probability p (training only)
int nt_dropout(int x_idx, float p);

// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
int nt_layernorm(int x_idx, int gamma_idx, int beta_idx);

// Sequence LayerNorm: normalize each of T positions independently
int nt_seq_layernorm(int x_idx, int gamma_idx, int beta_idx, int T, int D);

// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
int nt_gelu(int x_idx);

// RRPRAM attention: positional pattern recognition via x @ Wr
// wr: [nr_heads * n_embd, ctx], x: [T, n_embd], v: [T, nr_heads * head_dim]
// output: [T, nr_heads * head_dim]
int nt_rrpram_attention(int wr_idx, int x_idx, int v_idx, int T, int n_embd, int nr_heads, int head_dim);

// Low-rank RRPRAM: same as nt_rrpram_attention but Wr = Wr_a × Wr_b factorized.
// wr_combined holds Wr_a (size H*E*R) followed by Wr_b (size H*R*T_r), so total
// length = H*R*(E+T_r). Assumption: T_r == T (positional dim equals current ctx).
// Rank derived from tensor length: R = len / (H * (E + T)).
// Per head: scores[i,j] = (xi @ Wr_a[h]) @ Wr_b[h] [:, j] for j ≤ i (causal).
// Backward propagates gradients to Wr_a, Wr_b (stored in same combined buffer
// at proper offsets), x, v separately. Same V interpretation as nt_rrpram_attention.
//
// Saves params relative to full-rank when R << min(E, T_r). Plan #5.1: R=128
// at E=1024, T_r=2048 reduces RRPRAM weights from H·E·T_r per layer down to
// H·R·(E+T_r) — a 5× reduction at this configuration.
int nt_rrpram_lowrank_attention(int wr_combined_idx, int x_idx, int v_idx,
                                 int T, int n_embd, int nr_heads, int head_dim);

// Broadcast RRPRAM low-rank attention — canonical Janus pattern (per dario/infer_v4.c:218-249).
// mid[h,r] = Σ_t Σ_e x[t,e] · Wr_a[h,e,r] computed once per layer (broadcast over t).
// score[h,j] = Σ_r mid[h,r] · Wr_b[h,r,j] * sc with sc = 1/sqrt(D) (canonical scale).
// attn[h,i,j] = softmax_causal(scores[h])[i,j] for j ≤ i.
// out[i, h_off+d] = Σ_{j≤i} attn[h,i,j] · v[j, h_off+d].
int nt_rrpram_broadcast_attention(int wr_combined_idx, int x_idx, int v_idx,
                                   int T, int n_embd, int nr_heads, int head_dim);

// Concatenate per-position: out[t] = [a[t], b[t]]. a: [T, D_a], b: [T, D_b] → out: [T, D_a+D_b]
int nt_concat(int a_idx, int b_idx, int T);

// SwiGLU: y = SiLU(gate) * up (element-wise)
// gate and up must have same length (typically pre-computed via nt_seq_linear or nt_bit_seq_linear).
// Used in LLaMA/Qwen/BitNet FFN: gate = W_gate @ x, up = W_up @ x, h = swiglu(gate, up), out = W_down @ h.
int nt_swiglu(int gate_idx, int up_idx);

// BitLinear (BitNet b1.58): y = bitquant(W) @ x
// W quantized to ternary {-1, 0, +1} via absmean (γ_W = mean|W|).
// x quantized to int8 via absmax (γ_x = max|x|). Output rescaled: y = (γ_W γ_x / 127) × int_matmul.
// Backward uses Straight-Through Estimator: gradient flows through quantization as identity,
// so dW = dout ⊗ x, dx = W^T @ dout (using full-precision W).
int nt_bit_linear(int w_idx, int x_idx);

// Sequence BitLinear: Y[t] = bitquant(W) @ X[t] for t = 0..T-1.
// W is quantized once per forward (shared across positions), x is per-position absmax quantized.
int nt_bit_seq_linear(int w_idx, int x_idx, int T);

// SPA — Sentence Phonon Attention (inference-time helpers; no tape, no gradient).
// Compute sentence embedding via exponentially-weighted mean of token embeddings.
// tokens: token IDs in sentence (len n_tokens). W_embed: pointer to [vocab_size × dim] matrix.
// alpha: recency bias (0.85 typical — larger α = more recent tokens weighted higher).
// out_emb: caller-provided buffer of length `dim`.
void nt_spa_embed_sentence(const int* tokens, int n_tokens,
                           const float* W_embed, int vocab_size, int dim,
                           float alpha, float* out_emb);

// SPA connectedness: max softmax attention score between query_emb and history of sentence embeddings.
// Returns value in [0, 1]. Larger = current sentence more connected to history.
float nt_spa_connectedness(const float* query_emb, int dim,
                           const float* sentence_embeddings, int n_sentences);

// Modulate logits by SPA connectedness (higher conn → sharper distribution via effective-temperature drop).
// logits: buffer of V values modified in place. strength ∈ [0, 1] (0.3 typical).
void nt_spa_modulate_logits(float* logits, int V, float connectedness, float strength);

// ═══════════════════════════════════════════════════════════════════════════════
// BLAS — direct matmul API for inference engines
// ═══════════════════════════════════════════════════════════════════════════════

// C[m,n] = A[m,k] @ B[n,k]^T  (B stored row-major [n,k])
void nt_blas_mmT(float *C, const float *A, const float *BT, int m, int k, int n);

// C[m,n] = A[m,k] @ B[k,n]
void nt_blas_mm(float *C, const float *A, const float *B, int m, int k, int n);

// out[m] = W[m,n] @ x[n]  — matrix–vector path for inference engines that
// call matvec per-token inside a hot loop. Under USE_BLAS uses cblas_sgemv
// (Accelerate / OpenBLAS); without BLAS falls back to the naive nested loop.
void nt_blas_matvec(float *out, const float *W, const float *x, int m, int n);

// ═══════════════════════════════════════════════════════════════════════════════
// PROFILER — op timing + memory tracking
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    double forward_ms;        // total forward time
    double backward_ms;       // total backward time
    double optimizer_ms;      // total optimizer time
    long   peak_memory;       // peak bytes allocated
    long   current_memory;    // current bytes allocated
    int    n_ops;             // number of ops recorded
    int    n_params;          // number of params
    long   total_param_elems; // total parameter elements
    int    enabled;           // 1 = profiling active
} nt_profiler;

void         nt_profiler_enable(void);
void         nt_profiler_disable(void);
void         nt_profiler_reset(void);
nt_profiler* nt_profiler_get(void);
void         nt_profiler_print(void);

// ═══════════════════════════════════════════════════════════════════════════════
// BPE TOKENIZER — byte-pair encoding for training and inference
// ═══════════════════════════════════════════════════════════════════════════════

#define NT_BPE_MAX_MERGES 32768
#define NT_BPE_MAX_VOCAB  (256 + NT_BPE_MAX_MERGES)
#define NT_BPE_MAX_TOKEN_LEN 64

typedef struct {
    int merges[NT_BPE_MAX_MERGES][2];  // merge pairs: (a, b) → 256 + merge_idx
    int n_merges;
    int vocab_size;                     // 256 + n_merges
    // Decode table: token_id → byte sequence
    unsigned char tokens[NT_BPE_MAX_VOCAB][NT_BPE_MAX_TOKEN_LEN];
    int token_len[NT_BPE_MAX_VOCAB];
} nt_bpe;

// Load merges from text file: one "a b\n" pair per line
int nt_bpe_load(nt_bpe* bpe, const char* path);

// Load merges from C array (for embedded merges)
void nt_bpe_init(nt_bpe* bpe, const int merges[][2], int n_merges);

// Encode text → token IDs. Returns number of tokens written.
int nt_bpe_encode(const nt_bpe* bpe, const char* text, int text_len, int* out, int max_tokens);

// Decode token IDs → text. Returns number of bytes written.
int nt_bpe_decode(const nt_bpe* bpe, const int* tokens, int n_tokens, char* out, int max_bytes);

// ═══════════════════════════════════════════════════════════════════════════════
// DATALOADER — batch iterator for training
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int*   tokens;              // all tokenized data
    int    n_tokens;            // total tokens
    int    seq_len;             // sequence length per sample
    int    batch_size;
    int    pos;                 // current position in token stream
    int    epoch;               // current epoch counter
    // Shuffle state
    int*   shuffle_indices;     // shuffled start positions
    int    n_batches;           // total batches per epoch
    int    batch_idx;           // current batch within epoch
} nt_dataloader;

// Create dataloader from text file + BPE tokenizer
nt_dataloader* nt_dataloader_create(const char* text_file, nt_bpe* bpe,
                                     int seq_len, int batch_size);

// Create dataloader from pre-tokenized file (one int per token, binary)
nt_dataloader* nt_dataloader_from_tokens(const char* token_file,
                                          int seq_len, int batch_size);

// Get next batch. Writes input[batch_size * seq_len] and target[batch_size * seq_len].
// Returns 0 on success, -1 on epoch end (auto-resets, increments epoch).
int nt_dataloader_next(nt_dataloader* dl, int* input, int* target);

// Reset to beginning
void nt_dataloader_reset(nt_dataloader* dl);

// Shuffle for new epoch
void nt_dataloader_shuffle(nt_dataloader* dl);

// Free
void nt_dataloader_free(nt_dataloader* dl);

// ═══════════════════════════════════════════════════════════════════════════════
// SAVE / LOAD — binary weight format
// ═══════════════════════════════════════════════════════════════════════════════

// Save N tensors to binary file. Format: [magic][n][for each: ndim, shape[], data[]]
int nt_save(const char* path, nt_tensor** params, int n_params);

// Load N tensors from binary file. Returns array of tensors (caller frees each).
// Sets *n_params to number loaded. Returns NULL on failure.
nt_tensor** nt_load(const char* path, int* n_params);

// ═══════════════════════════════════════════════════════════════════════════════
// NOTORCH HEBBIAN — runtime microlearning without backward pass
// ═══════════════════════════════════════════════════════════════════════════════

// Update low-rank delta matrices from experience (Hebbian-style)
// A: [in_dim × rank], B: [rank × out_dim]
// x: input, dy: output gradient proxy, signal: teaching signal
void nt_hebbian_step(float* A, float* B, int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal,
                     float lr, float decay);

// ═══════════════════════════════════════════════════════════════════════════════
// NOTORCH LoRA — low-rank adapters on a frozen base weight
// ═══════════════════════════════════════════════════════════════════════════════
//
// Classic LoRA: y = W_frozen @ x + (alpha/rank) * B @ (A @ x).
// A : [rank, in_dim] — kaiming_uniform_(fan_in=in_dim) init, trainable
// B : [out_dim, rank] — zeros init, trainable
// W : [out_dim, in_dim] — frozen base weight, registered separately via
//                         nt_tape_param_frozen() so it does not consume a
//                         Chuck optimizer slot.
//
// LoRA pair owns persistent A,B tensors on the heap. The trainer's per-step
// loop calls nt_lora_forward() which registers A,B into THAT step's tape via
// nt_tape_param() (allocating Chuck slots ONLY for A,B — clean indexing).
// Chuck step then advances A,B's optimizer state correctly while base entries
// are skipped (they have no Chuck slots).

typedef struct {
    nt_tensor* A;          // [rank, in_dim] persistent, trainable
    nt_tensor* B;          // [out_dim, rank] persistent, trainable
    int        rank;
    float      alpha;
    float      scaling;    // alpha / rank
    int        in_dim;
    int        out_dim;
} nt_lora_pair;

// Allocate persistent A, B; init A ~ kaiming_uniform_(fan_in=in_dim), B = 0.
// Returns 0 on success, -1 on alloc fail.
int  nt_lora_init(nt_lora_pair* lora, int in_dim, int out_dim, int rank, float alpha);

// Free persistent A, B. Call once at trainer shutdown.
void nt_lora_free(nt_lora_pair* lora);

// Per-step forward: registers persistent A,B as trainable into the active tape
// (via nt_tape_param), then composes
//     y = nt_seq_linear(w_idx, x_idx, T) + scaling * nt_seq_linear(b_tape_idx,
//                                              nt_seq_linear(a_tape_idx, x_idx, T), T)
// Returns the tape entry index of y (final sum). Must be called inside an
// active nt_tape_start()/nt_tape_clear() pair. -1 on failure.
int  nt_lora_forward(int w_idx, nt_lora_pair* lora, int x_idx, int T);

// Single-artifact save: writes ALL targets × all layers into one file.
// `pairs` is flat-indexed [layer * num_targets + target_idx].
// Calls nt_tensor_ensure_cpu(A,B) before reading host buffers (GPU-safe).
// Returns 0 on success, -1 on I/O / format fail.
//
// File format:
//   [u32 magic 0x4C4F5241 'LORA'][u32 version=1]
//   [u32 num_targets][per-target: u8 namelen, name bytes]
//   [u32 num_layers][u32 rank][u32 alpha_int][u32 in_dim][u32 out_dim]
//   [for each L in [0,num_layers): for each T in [0,num_targets):
//       A floats (rank × in_dim), B floats (out_dim × rank)]
int  nt_lora_save(const nt_lora_pair* pairs, int num_layers, int num_targets,
                  const char* const* target_names, const char* path);

// Single-artifact load. Caller pre-allocates pairs (init'd via nt_lora_init).
// Verifies magic/version/dims/rank/alpha and target_names against caller's,
// then reads A,B into caller's persistent tensors. -1 on mismatch / I/O fail.
int  nt_lora_load(nt_lora_pair* pairs, int num_layers, int num_targets,
                  const char* const* target_names, const char* path);

// Merge LoRA delta into a CPU float buffer:
//     W_dst[i,j] = W_frozen[i,j] + scaling * sum_k B[i,k] * A[k,j]
// Calls nt_tensor_ensure_cpu(lora->A,B) internally. W_frozen and W_dst may
// alias for in-place merge. Layout: row-major [out_dim, in_dim].
void nt_lora_merge_into(float* W_dst, const float* W_frozen,
                        const nt_lora_pair* lora, int in_dim, int out_dim);

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═════���═════════════════════════════════════════════════════════════════════════

// Count total parameters across N tensors
long nt_count_params(nt_tensor** params, int n);

// Print parameter summary
void nt_print_params(nt_tensor** params, int n, const char** names);

// Seed RNG
void nt_seed(uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif // NOTORCH_H

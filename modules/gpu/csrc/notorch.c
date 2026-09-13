// notorch.c — PyTorch replacement in pure C
// Extracted from ariannamethod.ai/core/ (Arianna Method)
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// fuck torch

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>

// ═══════════════════════════════════════════════════════════════════════════════
// BLAS BACKEND
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#ifdef USE_SIMD
  #ifdef USE_BLAS
    #error "USE_SIMD and USE_BLAS are mutually exclusive — pick one matmul backend."
  #endif
  // In-house AVX2 + FMA shim for cblas_sgemm / sgemv / sger.
  // Lets every existing cblas_* call site stay unchanged.
  #ifdef NOTORCH_SIMD_DEBUG_SCALAR
    #include "notorch_simd_scalar.h"
  #else
    #include "notorch_simd.h"
  #endif
  // Also satisfy the original `#ifdef USE_BLAS` guards in this file by aliasing
  // them on. The shim defines the same CBLAS_* enums and functions.
  #define USE_BLAS 1
#endif

#ifdef USE_CUDA
  #include "notorch_cuda.h"
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// GPU MODE — runtime flag + per-tensor lazy CPU↔GPU mirror helpers
// All compiled out when USE_CUDA is undefined.
// ═══════════════════════════════════════════════════════════════════════════════

static int g_use_gpu = 0;

void nt_set_gpu_mode(int on_off) {
#ifdef USE_CUDA
    g_use_gpu = on_off ? 1 : 0;
#else
    (void)on_off;
    g_use_gpu = 0;
#endif
}

int nt_get_gpu_mode(void) { return g_use_gpu; }

#ifdef USE_CUDA
// Lazy upload: ensure t->d_data is allocated and contains current CPU values.
// If gpu_valid == 1 the GPU buffer is up to date and no transfer happens.
// If cpu_dirty == 1 the GPU is the source of truth — caller already wrote
// there. Do not overwrite it with stale CPU data.
static float* nt_tensor_ensure_gpu(nt_tensor* t) {
    if (!t || t->len <= 0) return NULL;
    if (!t->d_data) {
        t->d_data = gpu_alloc(t->len);
        t->gpu_valid = 0;
    }
    if (!t->gpu_valid && !t->cpu_dirty && t->d_data) {
        gpu_upload(t->d_data, t->data, t->len);
        t->gpu_valid = 1;
    }
    return t->d_data;
}

// Lazy download: pull GPU data into CPU mirror only if a CPU op needs it.
// Called at the start of any CPU-only op that reads tensor data.
static void nt_tensor_ensure_cpu(nt_tensor* t) {
    if (!t || !t->d_data || !t->cpu_dirty) return;
    gpu_download(t->data, t->d_data, t->len);
    t->cpu_dirty = 0;
}

// Mark a tensor as freshly written by a GPU kernel: GPU is source of truth,
// CPU mirror is stale. Avoids the eager D2H copy of v1 dispatch (one transfer
// per op was killing throughput more than the kernels saved).
static void nt_tensor_mark_gpu_fresh(nt_tensor* t) {
    if (!t) return;
    t->gpu_valid = 1;
    t->cpu_dirty = 1;
}

// Mark CPU as authoritative (e.g. after Chuck step on CPU writes weights).
static void nt_tensor_mark_cpu_dirty(nt_tensor* t) {
    if (!t) return;
    t->gpu_valid = 0;  /* next ensure_gpu re-uploads */
    t->cpu_dirty = 0;  /* CPU is now the source of truth */
}
#endif

/* Public sync wrapper for external callers (e.g. nanoarianna LoRA trainer that
 * needs to read a parameter's CPU data after Chuck step). On non-USE_CUDA build
 * this is a no-op since CPU is always authoritative. */
void nt_tensor_sync_cpu(nt_tensor* t) {
#ifdef USE_CUDA
    nt_tensor_ensure_cpu(t);
#else
    (void)t;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// RNG
// ═══════════════════════════════════════════════════════════════════════════════

static uint64_t g_rng_state = 2463534242ULL;

void nt_seed(uint64_t seed) {
    g_rng_state = seed ? seed : 2463534242ULL;
}

static uint32_t xorshift32(void) {
    uint64_t s = g_rng_state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    g_rng_state = s;
    return (uint32_t)s;
}

static float rand_uniform(void) {
    return (float)xorshift32() / 4294967296.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR
// ═══════════════════════════════════════════════════════════════════════════════

static void compute_strides(nt_tensor* t) {
    if (t->ndim <= 0) return;
    t->stride[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--)
        t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
}

nt_tensor* nt_tensor_new(int len) {
    if (len <= 0 || len > NT_MAX_ELEMENTS) return NULL;
    nt_tensor* t = (nt_tensor*)calloc(1, sizeof(nt_tensor));
    if (!t) return NULL;
    t->data = (float*)calloc(len, sizeof(float));
    if (!t->data) { free(t); return NULL; }
    t->len = len;
    t->ndim = 1;
    t->shape[0] = len;
    t->stride[0] = 1;
    t->refcount = 1;
    return t;
}

nt_tensor* nt_tensor_new2d(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    int total = rows * cols;
    if (total > NT_MAX_ELEMENTS) return NULL;
    nt_tensor* t = nt_tensor_new(total);
    if (!t) return NULL;
    t->ndim = 2;
    t->shape[0] = rows;
    t->shape[1] = cols;
    compute_strides(t);
    return t;
}

nt_tensor* nt_tensor_new_shape(const int* shape, int ndim) {
    if (ndim <= 0 || ndim > NT_MAX_DIMS) return NULL;
    int total = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return NULL;
        total *= shape[i];
        if (total > NT_MAX_ELEMENTS) return NULL;
    }
    nt_tensor* t = nt_tensor_new(total);
    if (!t) return NULL;
    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    compute_strides(t);
    return t;
}

void nt_tensor_free(nt_tensor* t) {
    if (!t) return;
    t->refcount--;
    if (t->refcount <= 0) {
        free(t->data);
#ifdef USE_CUDA
        if (t->d_data) { gpu_free(t->d_data); t->d_data = NULL; }
#endif
        free(t);
    }
}

nt_tensor* nt_tensor_ref(nt_tensor* t) {
    if (t) t->refcount++;
    return t;
}

nt_tensor* nt_tensor_clone(const nt_tensor* src) {
    if (!src) return NULL;
    nt_tensor* dst = nt_tensor_new(src->len);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->len * sizeof(float));
    dst->ndim = src->ndim;
    for (int i = 0; i < src->ndim; i++) {
        dst->shape[i] = src->shape[i];
        dst->stride[i] = src->stride[i];
    }
    return dst;
}

void nt_tensor_fill(nt_tensor* t, float val) {
    if (!t) return;
    for (int i = 0; i < t->len; i++) t->data[i] = val;
}

void nt_tensor_rand(nt_tensor* t, float scale) {
    if (!t) return;
    for (int i = 0; i < t->len; i++)
        t->data[i] = (2.0f * rand_uniform() - 1.0f) * scale;
}

void nt_tensor_xavier(nt_tensor* t, int fan_in, int fan_out) {
    if (!t || fan_in <= 0 || fan_out <= 0) return;
    float scale = sqrtf(6.0f / (float)(fan_in + fan_out));
    nt_tensor_rand(t, scale);
}

void nt_kaiming_uniform_init(nt_tensor* t, int fan_in) {
    // Uniform in [-sqrt(3/fan_in), +sqrt(3/fan_in)] → variance a²/3 = 1/fan_in.
    if (!t || fan_in <= 0) return;
    float scale = sqrtf(3.0f / (float)fan_in);
    nt_tensor_rand(t, scale);
}

int nt_tensor_reshape(nt_tensor* t, const int* new_shape, int new_ndim) {
    if (!t || new_ndim <= 0 || new_ndim > NT_MAX_DIMS) return -1;
    int total = 1;
    for (int i = 0; i < new_ndim; i++) total *= new_shape[i];
    if (total != t->len) return -1;
    t->ndim = new_ndim;
    for (int i = 0; i < new_ndim; i++) t->shape[i] = new_shape[i];
    compute_strides(t);
    return 0;
}

void nt_tensor_print(const nt_tensor* t, const char* name) {
    if (!t) { printf("%s: NULL\n", name ? name : "tensor"); return; }
    printf("%s: [", name ? name : "tensor");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? "×" : "");
    }
    printf("] (%d params)", t->len);
    if (t->len > 0) {
        printf(" first=%.4f", t->data[0]);
        if (t->len > 1) printf(" last=%.4f", t->data[t->len - 1]);
    }
    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOGRAD TAPE
// ═══════════════════════════════════════════════════════════════════════════════

static nt_tape g_tape = {0};

void nt_tape_start(void) {
    nt_tape_clear();
    g_tape.active = 1;
}

void nt_tape_clear(void) {
    for (int i = 0; i < g_tape.count; i++) {
        if (g_tape.entries[i].output)
            nt_tensor_free(g_tape.entries[i].output);
        if (g_tape.entries[i].grad) {
            nt_tensor_free(g_tape.entries[i].grad);
            g_tape.entries[i].grad = NULL;
        }
    }
    g_tape.count = 0;
    g_tape.active = 0;
    g_tape.n_params = 0;
}

void nt_tape_destroy(void) {
    for (int i = 0; i < g_tape.count; i++) {
        if (g_tape.entries[i].output) {
            nt_tensor_free(g_tape.entries[i].output);
            g_tape.entries[i].output = NULL;
        }
        if (g_tape.entries[i].grad) {
            nt_tensor_free(g_tape.entries[i].grad);
            g_tape.entries[i].grad = NULL;
        }
    }
    for (int i = 0; i < g_tape.n_params; i++) {
        if (g_tape.adam[i].m) { nt_tensor_free(g_tape.adam[i].m); g_tape.adam[i].m = NULL; }
        if (g_tape.adam[i].v) { nt_tensor_free(g_tape.adam[i].v); g_tape.adam[i].v = NULL; }
        if (g_tape.adam[i].acc_grad) { nt_tensor_free(g_tape.adam[i].acc_grad); g_tape.adam[i].acc_grad = NULL; }
        g_tape.adam[i].t = 0;
    }
    memset(&g_tape, 0, sizeof(g_tape));
}

int nt_tape_is_active(void) { return g_tape.active; }
nt_tape* nt_tape_get(void) { return &g_tape; }

int nt_tape_record(nt_tensor* output, int op, int p1, int p2, float aux) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = output;
    nt_tensor_ref(output);
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = -1;
    e->aux = aux;
    e->aux2 = 0;
    e->is_param = 0;
    e->no_decay = 0;
    g_tape.count++;
    return idx;
}

int nt_tape_record3(nt_tensor* output, int op, int p1, int p2, int p3, float aux, float aux2) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = output;
    nt_tensor_ref(output);
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = p3;
    e->aux = aux;
    e->aux2 = aux2;
    e->is_param = 0;
    e->no_decay = 0;
    g_tape.count++;
    return idx;
}

int nt_tape_record4(nt_tensor* output, int op, int p1, int p2, int p3, float aux, float aux2, float aux3, float aux4) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = output;
    nt_tensor_ref(output);
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = p3;
    e->aux = aux;
    e->aux2 = aux2;
    e->aux3 = aux3;
    e->aux4 = aux4;
    e->is_param = 0;
    e->no_decay = 0;
    g_tape.count++;
    return idx;
}

int nt_tape_param(nt_tensor* param) {
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = param;
    nt_tensor_ref(param);
    e->grad = NULL;
    e->op = NT_OP_NONE;
    e->parent1 = -1;
    e->parent2 = -1;
    e->parent3 = -1;
    e->aux = 0;
    e->aux2 = 0;
    e->is_param = 1;
    e->no_decay = 0;
    e->frozen = 0;       // explicit reset — prevents sticky frozen flag from
                         // a previous nt_tape_param_frozen() that reused this slot.
                         // Per Codex notorch-pass-1 P2 #1.

    if (g_tape.n_params < NT_TAPE_MAX_PARAMS) {
        int pi = g_tape.n_params;
        if (!g_tape.adam[pi].m) {
            g_tape.adam[pi].m = nt_tensor_new(param->len);
            g_tape.adam[pi].v = nt_tensor_new(param->len);
            g_tape.adam[pi].t = 0;
        } else if (g_tape.adam[pi].m->len != param->len) {
            nt_tensor* new_m = nt_tensor_new(param->len);
            nt_tensor* new_v = nt_tensor_new(param->len);
            int copy_len = g_tape.adam[pi].m->len < param->len ? g_tape.adam[pi].m->len : param->len;
            memcpy(new_m->data, g_tape.adam[pi].m->data, copy_len * sizeof(float));
            memcpy(new_v->data, g_tape.adam[pi].v->data, copy_len * sizeof(float));
            nt_tensor_free(g_tape.adam[pi].m);
            nt_tensor_free(g_tape.adam[pi].v);
            g_tape.adam[pi].m = new_m;
            g_tape.adam[pi].v = new_v;
        }
        g_tape.n_params++;
    }

    g_tape.count++;
    return idx;
}

void nt_tape_no_decay(int idx) {
    if (idx >= 0 && idx < g_tape.count)
        g_tape.entries[idx].no_decay = 1;
}

void nt_tape_freeze_param(int param_idx) {
    if (param_idx >= 0 && param_idx < g_tape.n_params)
        g_tape.chuck_params[param_idx].frozen = 1;
    // Also set the per-entry frozen flag so backward can skip computation.
    // Note: param_idx in this API is the *tape entry index*, returned by nt_tape_param().
    if (param_idx >= 0 && param_idx < g_tape.count)
        g_tape.entries[param_idx].frozen = 1;
}

int nt_tape_param_frozen(nt_tensor* param) {
    // Mirror nt_tape_param body, but DO NOT allocate a Chuck optimizer slot.
    // Set entry->frozen=1 so backward skips dw accumulation.
    if (!g_tape.active || g_tape.count >= NT_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    nt_tape_entry* e = &g_tape.entries[idx];
    e->output = param;
    nt_tensor_ref(param);
    e->grad = NULL;
    e->op = NT_OP_NONE;
    e->parent1 = -1;
    e->parent2 = -1;
    e->parent3 = -1;
    e->aux = 0;
    e->aux2 = 0;
    e->is_param = 1;
    e->no_decay = 0;
    e->frozen = 1;            // backward skips dw via this flag (notorch.c:845 path)
    // INTENTIONAL: do NOT increment g_tape.n_params, do NOT touch g_tape.adam[].
    // Chuck slots stay 1:1 with truly trainable params registered via nt_tape_param().
    g_tape.count++;
    return idx;
}

// Find tape entry by tensor pointer
static int tape_find(nt_tensor* t) {
    if (!t) return -1;
    for (int i = g_tape.count - 1; i >= 0; i--)
        if (g_tape.entries[i].output && g_tape.entries[i].output->data == t->data)
            return i;
    return -1;
}

// Ensure tensor is on tape (record as leaf if not)
static int tape_ensure(nt_tensor* t) {
    if (!t || !g_tape.active) return -1;
    int idx = tape_find(t);
    if (idx >= 0) return idx;
    return nt_tape_record(t, NT_OP_NONE, -1, -1, 0);
}

// Accumulate gradient into a tape entry
static void tape_acc_grad(int idx, const float* grad, int len) {
    if (idx < 0 || idx >= g_tape.count) return;
    nt_tape_entry* e = &g_tape.entries[idx];
    if (e->frozen) return;   // skip allocation + accumulation for frozen params
    if (!e->grad) {
        e->grad = nt_tensor_new(len);
        if (!e->grad) return;
    }
#ifdef USE_CUDA
    /* If GPU is the source of truth for e->grad, sync to CPU first so this
     * CPU contribution lands on the latest accumulated value. */
    nt_tensor_ensure_cpu(e->grad);
#endif
    int n = e->grad->len < len ? e->grad->len : len;
    for (int i = 0; i < n; i++) e->grad->data[i] += grad[i];
#ifdef USE_CUDA
    /* CPU just modified — invalidate GPU mirror. */
    e->grad->gpu_valid = 0;
    e->grad->cpu_dirty = 0;
#endif
}

#ifdef USE_CUDA
/* Accumulate a GPU-resident contribution into e->grad's GPU buffer.
 * If e->grad doesn't have GPU storage yet, allocate + zero. If e->grad
 * is currently CPU-fresh, upload the existing CPU values first so the
 * GPU buffer sees full accumulated state, then axpy. */
static void tape_acc_grad_gpu(int idx, const float* d_grad, int len) {
    if (idx < 0 || idx >= g_tape.count) return;
    nt_tape_entry* e = &g_tape.entries[idx];
    if (e->frozen) return;
    if (!e->grad) {
        e->grad = nt_tensor_new(len);
        if (!e->grad) return;
    }
    int n = e->grad->len < len ? e->grad->len : len;
    /* Ensure GPU buffer exists and contains current CPU state. */
    float* d_dst = nt_tensor_ensure_gpu(e->grad);
    if (!d_dst) return;
    /* Use cuBLAS axpy: dst += d_grad. */
    extern void gpu_axpy(float* d_y, const float* d_x, int n, float alpha);
    gpu_axpy(d_dst, d_grad, n, 1.0f);
    /* GPU is now fresh. */
    e->grad->gpu_valid = 1;
    e->grad->cpu_dirty = 1;
}
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// BACKWARD PASS
// ═══════════════════════════════════════════════════════════════════════════════

void nt_tape_backward(int loss_idx) {
    if (loss_idx < 0 || loss_idx >= g_tape.count) return;

    /* Lazy GPU/CPU mirror model — no eager D2H prelude. Each bw op-case is
     * responsible for either staying GPU-resident (GPU branch) or pulling
     * the specific parents/grads it consumes via nt_tensor_ensure_cpu().
     * Avoids the avg ~18% GPU-utilization ceiling caused by syncing all
     * activations at the start of backward. */

    nt_tape_entry* loss = &g_tape.entries[loss_idx];
    if (!loss->grad) loss->grad = nt_tensor_new(loss->output->len);
    for (int i = 0; i < loss->grad->len; i++) loss->grad->data[i] = 1.0f;
#ifdef USE_CUDA
    /* Loss grad is a CPU-authored fresh value — invalidate any stale GPU mirror. */
    loss->grad->gpu_valid = 0;
    loss->grad->cpu_dirty = 0;
#endif

    for (int idx = loss_idx; idx >= 0; idx--) {
        nt_tape_entry* e = &g_tape.entries[idx];
        if (!e->grad) continue;
#ifdef USE_CUDA
        /* CPU bw cases read e->grad->data (`dout`) directly. If a downstream
         * GPU bw kernel deposited the grad via tape_acc_grad_gpu, the CPU
         * mirror is stale — pull it down now. Cost: one D2H per active grad.
         * GPU bw cases that ensure_gpu(e->grad) below will see cpu_dirty=0,
         * gpu_valid=1 and skip the upload. */
        nt_tensor_ensure_cpu(e->grad);
#endif
        float* dout = e->grad->data;
        int out_len = e->output->len;

        switch (e->op) {

        case NT_OP_ADD: {
#ifdef USE_CUDA
            if (g_use_gpu) {
                int p1_match = e->parent1 >= 0 &&
                    g_tape.entries[e->parent1].output &&
                    g_tape.entries[e->parent1].output->len == out_len;
                int p2_match = e->parent2 >= 0 &&
                    g_tape.entries[e->parent2].output &&
                    g_tape.entries[e->parent2].output->len == out_len;
                if (p1_match && p2_match) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    if (d_dout) {
                        tape_acc_grad_gpu(e->parent1, d_dout, out_len);
                        tape_acc_grad_gpu(e->parent2, d_dout, out_len);
                        break;
                    }
                }
            }
#endif
            if (e->parent1 >= 0) tape_acc_grad(e->parent1, dout, out_len);
            if (e->parent2 >= 0) tape_acc_grad(e->parent2, dout, out_len);
            break;
        }

        case NT_OP_MUL: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pa = &g_tape.entries[e->parent1];
                nt_tape_entry* pb = &g_tape.entries[e->parent2];
                /* SwiGLU / gate-blend FIX 2026-05-11: forward output of both
                 * parents may live on GPU; CPU mirror is stale calloc-zero.
                 * Without sync, ga=gb=0 — masks all LoRA gradients on the
                 * mlp_gate + mlp_up SwiGLU branch. */
                nt_tensor_sync_cpu(pa->output);
                nt_tensor_sync_cpu(pb->output);
                float* ga = (float*)calloc(out_len, sizeof(float));
                float* gb = (float*)calloc(out_len, sizeof(float));
                if (ga && gb) {
                    for (int i = 0; i < out_len; i++) {
                        ga[i] = dout[i] * pb->output->data[i];
                        gb[i] = dout[i] * pa->output->data[i];
                    }
                    tape_acc_grad(e->parent1, ga, out_len);
                    tape_acc_grad(e->parent2, gb, out_len);
                }
                free(ga); free(gb);
            }
            break;
        }

        case NT_OP_SCALE: {
            if (e->parent1 >= 0) {
#ifdef USE_CUDA
                if (g_use_gpu) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_ga   = gpu_scratch(3, out_len);
                    if (d_dout && d_ga) {
                        gpu_scale(d_ga, d_dout, out_len, e->aux);
                        tape_acc_grad_gpu(e->parent1, d_ga, out_len);
                        break;
                    }
                }
#endif
                float* ga = (float*)calloc(out_len, sizeof(float));
                if (ga) {
                    for (int i = 0; i < out_len; i++) ga[i] = dout[i] * e->aux;
                    tape_acc_grad(e->parent1, ga, out_len);
                }
                free(ga);
            }
            break;
        }

        case NT_OP_MATVEC: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int rows = pw->output->shape[0];
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
                if (rows > 0 && cols > 0) {
                    float* dw = (float*)calloc(rows * cols, sizeof(float));
                    if (dw) {
                        for (int i = 0; i < rows; i++)
                            for (int j = 0; j < cols; j++)
                                dw[i * cols + j] = dout[i] * px->output->data[j];
                        tape_acc_grad(e->parent1, dw, rows * cols);
                    }
                    free(dw);
                    float* dx = (float*)calloc(cols, sizeof(float));
                    if (dx) {
                        for (int j = 0; j < cols; j++)
                            for (int i = 0; i < rows; i++)
                                dx[j] += pw->output->data[i * cols + j] * dout[i];
                        tape_acc_grad(e->parent2, dx, cols);
                    }
                    free(dx);
                }
            }
            break;
        }

        case NT_OP_SILU: {
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                /* FIX 2026-05-11: parent output may be GPU-resident; CPU stale
                 * gives sigmoid(0)=0.5 partial grad — still corrupts the SiLU
                 * derivative used in SwiGLU mlp_gate path. */
                nt_tensor_sync_cpu(px->output);
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        float x = px->output->data[i];
                        float sig = 1.0f / (1.0f + expf(-x));
                        gx[i] = dout[i] * sig * (1.0f + x * (1.0f - sig));
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_SIGMOID: {
            /* y = sigmoid(x); dy/dx = y * (1 - y) */
            if (e->parent1 >= 0) {
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        float y = e->output->data[i];
                        gx[i] = dout[i] * y * (1.0f - y);
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_SCALE_BY_T: {
            /* y = a[0] * x; gx = a[0] * dout; ga = sum(dout * x) */
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                nt_tape_entry* pa = &g_tape.entries[e->parent2];
                float a_val = pa->output->data[0];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) gx[i] = a_val * dout[i];
                    tape_acc_grad(e->parent1, gx, out_len);
                    free(gx);
                }
                float ga = 0;
                for (int i = 0; i < out_len; i++) ga += dout[i] * px->output->data[i];
                float ga_buf[1] = { ga };
                tape_acc_grad(e->parent2, ga_buf, 1);
            }
            break;
        }

        case NT_OP_SOFTMAX: {
            if (e->parent1 >= 0) {
                float dot_dy = 0;
                for (int i = 0; i < out_len; i++)
                    dot_dy += dout[i] * e->output->data[i];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++)
                        gx[i] = e->output->data[i] * (dout[i] - dot_dy);
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_RMSNORM: {
            // y = (x / rms) * gamma (if gamma provided)
            // parent1 = x, parent2 = gamma (-1 if none)
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int n = out_len;
                float ss = 0;
                for (int i = 0; i < n; i++) ss += px->output->data[i] * px->output->data[i];
                float rms = sqrtf(ss / n + 1e-6f);
                float rms3 = rms * rms * rms;

                // If gamma exists, dout_eff = dout * gamma for x-gradient
                float* dout_eff = dout;
                float* gamma_data = NULL;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                if (has_gamma) {
                    nt_tape_entry* pg = &g_tape.entries[e->parent2];
                    gamma_data = pg->output->data;
                    dout_eff = (float*)calloc(n, sizeof(float));
                    if (dout_eff) {
                        for (int i = 0; i < n; i++)
                            dout_eff[i] = dout[i] * gamma_data[i % pg->output->len];
                    } else {
                        dout_eff = dout;
                        has_gamma = 0;
                    }
                }

                float sum_dout_x = 0;
                for (int i = 0; i < n; i++)
                    sum_dout_x += dout_eff[i] * px->output->data[i];
                float* gx = (float*)calloc(n, sizeof(float));
                if (gx) {
                    for (int i = 0; i < n; i++)
                        gx[i] = (dout_eff[i] / rms) - (px->output->data[i] * sum_dout_x / (n * rms3));
                    tape_acc_grad(e->parent1, gx, n);
                }
                free(gx);

                // Gamma gradient: d_gamma[i] = dout[i] * (x[i] / rms)
                if (has_gamma && e->parent2 >= 0) {
                    nt_tape_entry* pg = &g_tape.entries[e->parent2];
                    float* gg = (float*)calloc(pg->output->len, sizeof(float));
                    if (gg) {
                        for (int i = 0; i < n; i++)
                            gg[i % pg->output->len] += dout[i] * (px->output->data[i] / rms);
                        tape_acc_grad(e->parent2, gg, pg->output->len);
                    }
                    free(gg);
                }

                if (has_gamma && dout_eff != dout) free(dout_eff);
            }
            break;
        }

        case NT_OP_CROSS_ENT: {
            if (e->parent1 >= 0) {
                nt_tape_entry* pl = &g_tape.entries[e->parent1];
                int n = pl->output->len;
                int target = (int)e->aux;
                float mx = pl->output->data[0];
                for (int i = 1; i < n; i++)
                    if (pl->output->data[i] > mx) mx = pl->output->data[i];
                float* sm = (float*)calloc(n, sizeof(float));
                if (sm) {
                    float sum = 0;
                    for (int i = 0; i < n; i++) {
                        sm[i] = expf(pl->output->data[i] - mx);
                        sum += sm[i];
                    }
                    for (int i = 0; i < n; i++) sm[i] /= sum;
                    if (target >= 0 && target < n) sm[target] -= 1.0f;
                    for (int i = 0; i < n; i++) sm[i] *= dout[0];
                    tape_acc_grad(e->parent1, sm, n);
                }
                free(sm);
            }
            break;
        }

        case NT_OP_EMB_LOOKUP: {
            if (e->parent1 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                int token_id = (int)e->aux;
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : out_len;
                int rows = pw->output->len / cols;
                if (cols > 0 && token_id >= 0 && token_id < rows) {
                    float* gw = (float*)calloc(pw->output->len, sizeof(float));
                    if (gw) {
                        for (int i = 0; i < cols && i < out_len; i++)
                            gw[token_id * cols + i] = dout[i];
                        tape_acc_grad(e->parent1, gw, pw->output->len);
                    }
                    free(gw);
                }
            }
            break;
        }

        case NT_OP_SEQ_EMBED: {
            if (e->parent1 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pwte = &g_tape.entries[e->parent1];
                nt_tape_entry* ptok = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                int wte_rows = pwte->output->ndim >= 2 ? pwte->output->shape[0] : pwte->output->len / D;
                int seqemb_done_gpu = 0;
#ifdef USE_CUDA
                /* GPU bw — only when no WPE branch (parent2 < 0). WPE handled CPU. */
                if (g_use_gpu && e->parent2 < 0) {
                    float* d_dwte = gpu_scratch(3, pwte->output->len);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_tok  = nt_tensor_ensure_gpu(ptok->output);
                    if (d_dwte && d_dout && d_tok) {
                        gpu_zero(d_dwte, pwte->output->len);
                        gpu_seq_embedding_backward(d_dwte, d_dout, d_tok, T, D, wte_rows);
                        tape_acc_grad_gpu(e->parent1, d_dwte, pwte->output->len);
                        seqemb_done_gpu = 1;
                    }
                }
                if (seqemb_done_gpu) break;
                nt_tensor_ensure_cpu(ptok->output);
#endif
                float* dwte = (float*)calloc(pwte->output->len, sizeof(float));
                if (dwte) {
                    for (int t = 0; t < T; t++) {
                        int tok = (int)ptok->output->data[t];
                        if (tok < 0) tok = 0;
                        if (tok >= wte_rows) tok = wte_rows - 1;
                        for (int d = 0; d < D; d++)
                            dwte[tok * D + d] += dout[t * D + d];
                    }
                    tape_acc_grad(e->parent1, dwte, pwte->output->len);
                }
                free(dwte);
                /* Position embedding gradients (if present) */
                if (e->parent2 >= 0) {
                    nt_tape_entry* pwpe = &g_tape.entries[e->parent2];
                    float* dwpe = (float*)calloc(pwpe->output->len, sizeof(float));
                    if (dwpe) {
                        int wpe_rows = pwpe->output->ndim >= 2 ? pwpe->output->shape[0] : pwpe->output->len / D;
                        for (int t = 0; t < T; t++) {
                            int pos = t < wpe_rows ? t : wpe_rows - 1;
                            for (int d = 0; d < D; d++)
                                dwpe[pos * D + d] += dout[t * D + d];
                        }
                        tape_acc_grad(e->parent2, dwpe, pwpe->output->len);
                    }
                    free(dwpe);
                }
            }
            break;
        }

        case NT_OP_SEQ_MATVEC: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int out_d = pw->output->shape[0];
                int in_d = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / out_d;
                int w_frozen = pw->frozen;     // skip dw if W is frozen (LoRA on frozen base)
                int x_frozen = px->frozen;     // also skip dx if X chain is frozen (rare)
                float* dw = NULL;
                float* dx = NULL;
                int bw_done_gpu = 0;
#ifdef USE_CUDA
                /* GPU backward path: stays GPU-resident.
                 * dW grad accumulates directly on pw->grad->d_data via cuBLAS
                 * axpy. Same for dX → px->grad->d_data. Saves the full
                 * download → calloc → CPU-add chain of v1. */
                if (g_use_gpu && (!w_frozen || !x_frozen)) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_W = nt_tensor_ensure_gpu(pw->output);
                    float* d_X = nt_tensor_ensure_gpu(px->output);
                    float* d_dx = !x_frozen ? gpu_scratch(3, px->output->len) : NULL;
                    float* d_dw = !w_frozen ? gpu_scratch(4, pw->output->len) : NULL;
                    if (d_dout && d_W && d_X &&
                        ((x_frozen) || d_dx) && ((w_frozen) || d_dw)) {
                        if (!x_frozen)
                            gpu_sgemm_nn(T, in_d, out_d, d_dout, d_W, d_dx);
                        if (!w_frozen)
                            gpu_sgemm_tn(out_d, in_d, T, d_dout, d_X, d_dw);
                        if (!w_frozen)
                            tape_acc_grad_gpu(e->parent1, d_dw, pw->output->len);
                        if (!x_frozen)
                            tape_acc_grad_gpu(e->parent2, d_dx, px->output->len);
                        bw_done_gpu = 1;
                    }
                }
                if (!bw_done_gpu) {
                    dw = w_frozen ? NULL : (float*)calloc(pw->output->len, sizeof(float));
                    dx = x_frozen ? NULL : (float*)calloc(px->output->len, sizeof(float));
                }
#else
                dw = w_frozen ? NULL : (float*)calloc(pw->output->len, sizeof(float));
                dx = x_frozen ? NULL : (float*)calloc(px->output->len, sizeof(float));
#endif
                if (!bw_done_gpu && ((dw || w_frozen) && (dx || x_frozen))) {
                    float* Wd = pw->output->data;
                    float* Xd = px->output->data;
#ifdef USE_BLAS
                    if (!x_frozen) {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    T, in_d, out_d,
                                    1.0f, dout, out_d, Wd, in_d,
                                    0.0f, dx, in_d);
                    }
                    if (!w_frozen) {
                        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                    out_d, in_d, T,
                                    1.0f, dout, out_d, Xd, in_d,
                                    0.0f, dw, in_d);
                    }
#else
                    if (!x_frozen) {
                        for (int t = 0; t < T; t++) {
                            float* dout_t = dout + t * out_d;
                            for (int j = 0; j < in_d; j++)
                                for (int i = 0; i < out_d; i++)
                                    dx[t * in_d + j] += Wd[i * in_d + j] * dout_t[i];
                        }
                    }
                    if (!w_frozen) {
                        for (int t = 0; t < T; t++) {
                            float* dout_t = dout + t * out_d;
                            float* x_t = Xd + t * in_d;
                            for (int i = 0; i < out_d; i++)
                                for (int j = 0; j < in_d; j++)
                                    dw[i * in_d + j] += dout_t[i] * x_t[j];
                        }
                    }
#endif
                }
                if (!bw_done_gpu && ((dw || w_frozen) && (dx || x_frozen))) {
                    if (!w_frozen) tape_acc_grad(e->parent1, dw, pw->output->len);
                    if (!x_frozen) tape_acc_grad(e->parent2, dx, px->output->len);
                }
                if (dw) free(dw);
                if (dx) free(dx);
            }
            break;
        }

        case NT_OP_SEQ_RMSNORM: {
            // y[t] = (x[t] / rms[t]) * gamma (if gamma provided)
            // parent1 = x, parent2 = gamma (-1 if none)
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                int srn_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu) {
                    float* d_X    = nt_tensor_ensure_gpu(px->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_gamma = NULL;
                    if (has_gamma) {
                        nt_tape_entry* pg = &g_tape.entries[e->parent2];
                        d_gamma = nt_tensor_ensure_gpu(pg->output);
                    }
                    float* d_gx = gpu_scratch(3, T * D);
                    float* d_gg = has_gamma ? gpu_scratch(4, D) : NULL;
                    if (d_X && d_dout && d_gx && (!has_gamma || (d_gamma && d_gg))) {
                        if (d_gg) gpu_zero(d_gg, D);
                        gpu_seq_rmsnorm_backward(d_gx, d_gg, d_dout, d_X, d_gamma, T, D);
                        tape_acc_grad_gpu(e->parent1, d_gx, T * D);
                        if (has_gamma && d_gg)
                            tape_acc_grad_gpu(e->parent2, d_gg, D);
                        srn_done_gpu = 1;
                    }
                }
#endif
                if (srn_done_gpu) break;
#ifdef USE_CUDA
                nt_tensor_ensure_cpu(px->output);
                if (has_gamma) nt_tensor_ensure_cpu(g_tape.entries[e->parent2].output);
#endif
                float* gamma_data = NULL;
                if (has_gamma) gamma_data = g_tape.entries[e->parent2].output->data;

                float* gx = (float*)calloc(T * D, sizeof(float));
                float* gg = has_gamma ? (float*)calloc(D, sizeof(float)) : NULL;
                if (gx) {
                    float* Xrn = px->output->data;
                    for (int t = 0; t < T; t++) {
                        float* x_t = Xrn + t * D;
                        float* dout_t = dout + t * D;
                        float ss = 0;
                        for (int d = 0; d < D; d++) ss += x_t[d] * x_t[d];
                        float rms = sqrtf(ss / D + 1e-6f);
                        float rms3 = rms * rms * rms;

                        // dout_eff = dout * gamma for x-gradient
                        float sum_dx = 0;
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            sum_dx += de * x_t[d];
                        }
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            gx[t * D + d] = (de / rms) - (x_t[d] * sum_dx / (D * rms3));
                        }
                        // gamma gradient: d_gamma[d] += dout[t,d] * (x[t,d] / rms[t])
                        if (gg) {
                            for (int d = 0; d < D; d++)
                                gg[d] += dout_t[d] * (x_t[d] / rms);
                        }
                    }
                    tape_acc_grad(e->parent1, gx, T * D);
                    if (gg && has_gamma)
                        tape_acc_grad(e->parent2, gg, D);
                }
                free(gx);
                free(gg);
            }
            break;
        }

        case NT_OP_CAUSAL_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pq = &g_tape.entries[e->parent1];
                nt_tape_entry* pk = &g_tape.entries[e->parent2];
                nt_tape_entry* pv = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                float sc = 1.0f / sqrtf((float)D);
                float* dq = (float*)calloc(T * D, sizeof(float));
                float* dk = (float*)calloc(T * D, sizeof(float));
                float* dv = (float*)calloc(T * D, sizeof(float));
                if (dq && dk && dv) {
                    for (int i = 0; i < T; i++) {
                        float* qi = pq->output->data + i * D;
                        float* dout_i = dout + i * D;
                        float* scores = (float*)calloc(i + 1, sizeof(float));
                        float* attn = (float*)calloc(i + 1, sizeof(float));
                        if (!scores || !attn) { free(scores); free(attn); continue; }
                        float mx = -1e30f;
                        for (int j = 0; j <= i; j++) {
                            float* kj = pk->output->data + j * D;
                            float dot = 0;
                            for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                            scores[j] = dot * sc;
                            if (scores[j] > mx) mx = scores[j];
                        }
                        float sm = 0;
                        for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                        if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                        float* d_attn = (float*)calloc(i + 1, sizeof(float));
                        if (d_attn) {
                            for (int j = 0; j <= i; j++) {
                                float* vj = pv->output->data + j * D;
                                for (int d = 0; d < D; d++) d_attn[j] += dout_i[d] * vj[d];
                            }
                            for (int j = 0; j <= i; j++) {
                                float* dvj = dv + j * D;
                                for (int d = 0; d < D; d++) dvj[d] += attn[j] * dout_i[d];
                            }
                            float dot_da = 0;
                            for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                            for (int j = 0; j <= i; j++) {
                                float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                float* kj = pk->output->data + j * D;
                                for (int d = 0; d < D; d++) {
                                    dq[i * D + d] += ds * kj[d];
                                    dk[j * D + d] += ds * qi[d];
                                }
                            }
                        }
                        free(scores); free(attn); free(d_attn);
                    }
                    tape_acc_grad(e->parent1, dq, T * D);
                    tape_acc_grad(e->parent2, dk, T * D);
                    tape_acc_grad(e->parent3, dv, T * D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case NT_OP_MH_CAUSAL_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pq = &g_tape.entries[e->parent1];
                nt_tape_entry* pk = &g_tape.entries[e->parent2];
                nt_tape_entry* pv = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int head_dim = (int)e->aux2;
                int D = e->output->len / T;
                int n_heads = D / head_dim;
                float sc = 1.0f / sqrtf((float)head_dim);
                float* dq = (float*)calloc(T * D, sizeof(float));
                float* dk = (float*)calloc(T * D, sizeof(float));
                float* dv = (float*)calloc(T * D, sizeof(float));
                int mh_done_gpu = 0;
#ifdef USE_CUDA
                /* GPU backward: kernel needs softmaxed scores. Forward did not
                 * persist them, so re-run forward into scratch first.
                 * Slot map (GPU_SCRATCH_SLOTS=16):
                 *   0 silu, 1 mh-attn scores, 2 cross_ent losses,
                 *   3,4 seq_matvec_bw d_dx/d_dw,
                 *   5,6 mh_bw scratch_TT/scratch_TT2, 7 mh recompute out,
                 *   8,9,10 mh_bw d_dQ/d_dK/d_dV. */
                if (g_use_gpu && dq && dk && dv) {
                    float* d_Q = nt_tensor_ensure_gpu(pq->output);
                    float* d_K = nt_tensor_ensure_gpu(pk->output);
                    float* d_V = nt_tensor_ensure_gpu(pv->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_scores = gpu_scratch(1, n_heads * T * T);
                    float* d_scratch_TT  = gpu_scratch(5, n_heads * T * T);
                    float* d_scratch_TT2 = gpu_scratch(6, n_heads * T * T);
                    float* d_out_tmp = gpu_scratch(7, T * D);
                    float* d_dQ_buf  = gpu_scratch(8, T * D);
                    float* d_dK_buf  = gpu_scratch(9, T * D);
                    float* d_dV_buf  = gpu_scratch(10, T * D);
                    if (d_Q && d_K && d_V && d_dout && d_scores && d_scratch_TT &&
                        d_scratch_TT2 && d_out_tmp && d_dQ_buf && d_dK_buf && d_dV_buf) {
                        /* Recompute softmaxed scores (kernel writes them). */
                        gpu_multi_head_attention(d_Q, d_K, d_V, d_out_tmp, d_scores, T, D, n_heads);
                        gpu_multi_head_attention_backward(d_Q, d_K, d_V, d_scores, d_dout,
                                                          d_dQ_buf, d_dK_buf, d_dV_buf,
                                                          d_scratch_TT, d_scratch_TT2,
                                                          T, D, n_heads);
                        /* GPU-resident grad accumulation — no D2H. */
                        tape_acc_grad_gpu(e->parent1, d_dQ_buf, T * D);
                        tape_acc_grad_gpu(e->parent2, d_dK_buf, T * D);
                        tape_acc_grad_gpu(e->parent3, d_dV_buf, T * D);
                        mh_done_gpu = 1;
                    }
                }
#endif
                if (mh_done_gpu) {
                    free(dq); free(dk); free(dv);
                    break;
                }
                if (dq && dk && dv) {
                    for (int h = 0; h < n_heads; h++) {
                        int ho = h * head_dim;
                        for (int i = 0; i < T; i++) {
                            float* qi = pq->output->data + i * D + ho;
                            float* dout_i = dout + i * D + ho;
                            float* scores = (float*)calloc(i + 1, sizeof(float));
                            float* attn = (float*)calloc(i + 1, sizeof(float));
                            if (!scores || !attn) { free(scores); free(attn); continue; }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float* kj = pk->output->data + j * D + ho;
                                float dot = 0;
                                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                                scores[j] = dot * sc;
                                if (scores[j] > mx) mx = scores[j];
                            }
                            float sm = 0;
                            for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                            if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                            float* d_attn = (float*)calloc(i + 1, sizeof(float));
                            if (d_attn) {
                                for (int j = 0; j <= i; j++) {
                                    float* vj = pv->output->data + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) d_attn[j] += dout_i[d] * vj[d];
                                }
                                for (int j = 0; j <= i; j++) {
                                    float* dvj = dv + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) dvj[d] += attn[j] * dout_i[d];
                                }
                                float dot_da = 0;
                                for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                                for (int j = 0; j <= i; j++) {
                                    float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                    float* kj = pk->output->data + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) {
                                        dq[i * D + ho + d] += ds * kj[d];
                                        dk[j * D + ho + d] += ds * qi[d];
                                    }
                                }
                            }
                            free(scores); free(attn); free(d_attn);
                        }
                    }
                    tape_acc_grad(e->parent1, dq, T * D);
                    tape_acc_grad(e->parent2, dk, T * D);
                    tape_acc_grad(e->parent3, dv, T * D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case NT_OP_GQA_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pq = &g_tape.entries[e->parent1];
                nt_tape_entry* pk = &g_tape.entries[e->parent2];
                nt_tape_entry* pv = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int head_dim = (int)e->aux2;
                int n_heads = (int)e->aux3;
                int n_kv_heads = (int)e->aux4;
                int Q_D = n_heads * head_dim;
                int KV_D = n_kv_heads * head_dim;
                int gqa_ratio = n_heads / n_kv_heads;
                float sc = 1.0f / sqrtf((float)head_dim);
                float* dq = (float*)calloc(T * Q_D, sizeof(float));
                float* dk = (float*)calloc(T * KV_D, sizeof(float));
                float* dv = (float*)calloc(T * KV_D, sizeof(float));
                if (dq && dk && dv) {
                    for (int h = 0; h < n_heads; h++) {
                        int kv_h = h / gqa_ratio;
                        int q_off = h * head_dim;
                        int kv_off = kv_h * head_dim;
                        for (int i = 0; i < T; i++) {
                            float* qi = pq->output->data + i * Q_D + q_off;
                            float* dout_i = dout + i * Q_D + q_off;
                            float* scores = (float*)calloc(i + 1, sizeof(float));
                            float* attn = (float*)calloc(i + 1, sizeof(float));
                            if (!scores || !attn) { free(scores); free(attn); continue; }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float* kj = pk->output->data + j * KV_D + kv_off;
                                float dot = 0;
                                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                                scores[j] = dot * sc;
                                if (scores[j] > mx) mx = scores[j];
                            }
                            float sm = 0;
                            for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                            if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                            float* d_attn = (float*)calloc(i + 1, sizeof(float));
                            if (d_attn) {
                                for (int j = 0; j <= i; j++) {
                                    float* vj = pv->output->data + j * KV_D + kv_off;
                                    for (int d = 0; d < head_dim; d++) d_attn[j] += dout_i[d] * vj[d];
                                }
                                for (int j = 0; j <= i; j++) {
                                    float* dvj = dv + j * KV_D + kv_off;
                                    for (int d = 0; d < head_dim; d++) dvj[d] += attn[j] * dout_i[d];
                                }
                                float dot_da = 0;
                                for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                                for (int j = 0; j <= i; j++) {
                                    float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                    float* kj = pk->output->data + j * KV_D + kv_off;
                                    for (int d = 0; d < head_dim; d++) {
                                        dq[i * Q_D + q_off + d] += ds * kj[d];
                                        dk[j * KV_D + kv_off + d] += ds * qi[d];
                                    }
                                }
                            }
                            free(scores); free(attn); free(d_attn);
                        }
                    }
                    tape_acc_grad(e->parent1, dq, T * Q_D);
                    tape_acc_grad(e->parent2, dk, T * KV_D);
                    tape_acc_grad(e->parent3, dv, T * KV_D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case NT_OP_RRPRAM_LR: {
            /* Low-rank RRPRAM backward.
             * Forward: u = X @ Wr_a[h]; scores = u @ Wr_b[h]; attn = softmax(causal); out = Σ attn·V.
             * dout flows back through:
             *   d_attn  = dout · V               (per i, h, j)
             *   d_v     = attn · dout            (per j, h)
             *   d_score = softmax_bwd(d_attn, attn)
             *   d_u     = d_score @ Wr_b[h]^T
             *   d_Wr_b  = u^T @ d_score (causal-masked outer-product)
             *   d_x     = Σ_h d_u · Wr_a[h]^T
             *   d_Wr_a  = Σ_h x^T @ d_u
             */
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pwr = &g_tape.entries[e->parent1];
                nt_tape_entry* px  = &g_tape.entries[e->parent2];
                nt_tape_entry* pv  = &g_tape.entries[e->parent3];
                int T = (int)e->aux; int n_embd = (int)e->aux2;
                int nr = (int)e->aux3; int hd = (int)e->aux4;
                int out_dim = nr * hd;
                int T_r = T;   /* same assumption as forward */
                long combined_len = pwr->output->len;
                int rank = (int)(combined_len / ((long)nr * (n_embd + T_r)));
                long wra_total = (long)nr * n_embd * rank;

                float* dwr = (float*)calloc(combined_len, sizeof(float));
                float* dx  = (float*)calloc((long)T * n_embd, sizeof(float));
                float* dv  = (float*)calloc((long)T * out_dim, sizeof(float));

                int rrlr_bw_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu && dwr && dx && dv) {
                    /* Recompute U and scores on GPU (forward did not persist
                     * across tape boundary cleanly — this is cheap: H·T·R + H·T·T floats). */
                    float* d_X  = nt_tensor_ensure_gpu(px->output);
                    float* d_Wr = nt_tensor_ensure_gpu(pwr->output);
                    float* d_V  = nt_tensor_ensure_gpu(pv->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_U      = gpu_scratch(12, nr * T * rank);
                    float* d_scores = gpu_scratch(1,  nr * T * T);
                    float* d_O_tmp  = gpu_scratch(7,  T * out_dim);
                    float* d_d_attn  = gpu_scratch(13, nr * T * T);
                    float* d_d_score = gpu_scratch(14, nr * T * T);
                    /* All scratch via persistent slots — avoid per-call cudaMalloc. */
                    float* d_dX  = gpu_scratch(15, T * n_embd);
                    /* Slots 11..14 already used by other backward paths above — but
                     * those paths run sequentially per backward pass (different ops),
                     * so slot reuse across distinct op-cases in the same backward
                     * call is safe. d_dWr fits in slot 11 (CE backward path scratch),
                     * d_dV in slot 0 (forward silu, not running here). */
                    float* d_dWr = gpu_scratch(11, combined_len);
                    float* d_dV  = gpu_scratch(0, T * out_dim);
                    if (d_X && d_Wr && d_V && d_dout && d_U && d_scores && d_O_tmp &&
                        d_d_attn && d_d_score && d_dX && d_dWr && d_dV) {
                        /* Recompute forward (writes U and scores). */
                        gpu_rrpram_lr_forward(d_X, d_Wr, d_V, d_O_tmp, d_U, d_scores,
                                              T, n_embd, nr, rank, hd);
                        gpu_rrpram_lr_backward(d_X, d_Wr, d_V, d_U, d_scores, d_dout,
                                               d_dWr, d_dX, d_dV,
                                               d_d_attn, d_d_score,
                                               T, n_embd, nr, rank, hd);
                        /* GPU-resident grad accumulation. */
                        tape_acc_grad_gpu(e->parent1, d_dWr, combined_len);
                        tape_acc_grad_gpu(e->parent2, d_dX,  (long)T * n_embd);
                        tape_acc_grad_gpu(e->parent3, d_dV,  (long)T * out_dim);
                        rrlr_bw_gpu = 1;
                    }
                }
                if (rrlr_bw_gpu) {
                    free(dwr); free(dx); free(dv);
                    break;
                }
#endif

                float* u_buf      = (float*)malloc(rank * sizeof(float));
                float* du_buf     = (float*)malloc(rank * sizeof(float));
                float* scores_buf = (float*)malloc(T_r  * sizeof(float));
                float* attn_buf   = (float*)malloc(T_r  * sizeof(float));
                float* d_attn_buf = (float*)malloc(T_r  * sizeof(float));
                float* d_score_buf= (float*)malloc(T_r  * sizeof(float));

                if (dwr && dx && dv && u_buf && du_buf && scores_buf && attn_buf && d_attn_buf && d_score_buf) {
                    for (int h = 0; h < nr; h++) {
                        long wr_a_base = (long)h * n_embd * rank;
                        long wr_b_base = wra_total + (long)h * rank * T_r;
                        int  v_off     = h * hd;
                        for (int i = 0; i < T; i++) {
                            float* xi = px->output->data + i * n_embd;
                            float* dout_i = dout + i * out_dim + v_off;

                            /* recompute forward: u, scores, attn */
                            for (int r = 0; r < rank; r++) u_buf[r] = 0.0f;
                            for (int d = 0; d < n_embd; d++) {
                                float xd = xi[d];
                                const float* wa_row = pwr->output->data + wr_a_base + (long)d * rank;
                                for (int r = 0; r < rank; r++) u_buf[r] += xd * wa_row[r];
                            }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float s = 0.0f;
                                for (int r = 0; r < rank; r++) {
                                    s += u_buf[r] * pwr->output->data[wr_b_base + (long)r * T_r + j];
                                }
                                scores_buf[j] = s;
                                if (s > mx) mx = s;
                            }
                            float sm = 0.0f;
                            for (int j = 0; j <= i; j++) { attn_buf[j] = expf(scores_buf[j] - mx); sm += attn_buf[j]; }
                            if (sm > 0.0f) for (int j = 0; j <= i; j++) attn_buf[j] /= sm;

                            /* d_attn[j] = Σ_d dout_i[d] · v[j, h_off+d]
                             * d_v [j, h_off+d] += attn[j] · dout_i[d] */
                            for (int j = 0; j <= i; j++) d_attn_buf[j] = 0.0f;
                            for (int j = 0; j <= i; j++) {
                                const float* vj = pv->output->data + j * out_dim + v_off;
                                float* dvj      = dv + j * out_dim + v_off;
                                for (int d = 0; d < hd; d++) {
                                    d_attn_buf[j] += dout_i[d] * vj[d];
                                    dvj[d]        += attn_buf[j] * dout_i[d];
                                }
                            }

                            /* softmax backward → d_score */
                            float dot_da = 0.0f;
                            for (int j = 0; j <= i; j++) dot_da += d_attn_buf[j] * attn_buf[j];
                            for (int j = 0; j <= i; j++) d_score_buf[j] = attn_buf[j] * (d_attn_buf[j] - dot_da);

                            /* d_u[r] = Σ_j d_score[j] · Wr_b[h, r, j] (j ≤ i)
                             * d_Wr_b[h, r, j] += d_score[j] · u[r]   (j ≤ i) */
                            for (int r = 0; r < rank; r++) du_buf[r] = 0.0f;
                            for (int j = 0; j <= i; j++) {
                                float ds = d_score_buf[j];
                                for (int r = 0; r < rank; r++) {
                                    du_buf[r] += ds * pwr->output->data[wr_b_base + (long)r * T_r + j];
                                    dwr[wr_b_base + (long)r * T_r + j] += ds * u_buf[r];
                                }
                            }

                            /* d_xi[d] += Σ_r d_u[r] · Wr_a[h, d, r]
                             * d_Wr_a[h, d, r] += d_u[r] · xi[d] */
                            for (int d = 0; d < n_embd; d++) {
                                const float* wa_row = pwr->output->data + wr_a_base + (long)d * rank;
                                float* dwa_row     = dwr + wr_a_base + (long)d * rank;
                                float dxd = 0.0f;
                                float xd = xi[d];
                                for (int r = 0; r < rank; r++) {
                                    dxd        += du_buf[r] * wa_row[r];
                                    dwa_row[r] += du_buf[r] * xd;
                                }
                                dx[i * n_embd + d] += dxd;
                            }
                        }
                    }
                    tape_acc_grad(e->parent1, dwr, combined_len);
                    tape_acc_grad(e->parent2, dx,  (long)T * n_embd);
                    tape_acc_grad(e->parent3, dv,  (long)T * out_dim);
                }
                free(dwr); free(dx); free(dv);
                free(u_buf); free(du_buf); free(scores_buf); free(attn_buf); free(d_attn_buf); free(d_score_buf);
            }
            break;
        }

        case NT_OP_RRPRAM_ATTN: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pwr = &g_tape.entries[e->parent1];
                nt_tape_entry* px  = &g_tape.entries[e->parent2];
                nt_tape_entry* pv  = &g_tape.entries[e->parent3];
                int T = (int)e->aux; int n_embd = (int)e->aux2;
                int nr = (int)e->aux3; int hd = (int)e->aux4;
                int out_dim = nr * hd;
                int ctx = pwr->output->len / (nr * n_embd);
                float* dwr = (float*)calloc(pwr->output->len, sizeof(float));
                float* dx  = (float*)calloc(T * n_embd, sizeof(float));
                float* dv  = (float*)calloc(T * out_dim, sizeof(float));
                if (dwr && dx && dv) {
                    for (int h = 0; h < nr; h++) {
                        int wr_base = h * n_embd * ctx; int v_off = h * hd;
                        for (int i = 0; i < T; i++) {
                            float* xi = px->output->data + i * n_embd;
                            float* dout_i = dout + i * out_dim + v_off;
                            float* scores = (float*)calloc(i + 1, sizeof(float));
                            float* attn = (float*)calloc(i + 1, sizeof(float));
                            if (!scores || !attn) { free(scores); free(attn); continue; }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float dot = 0;
                                for (int d = 0; d < n_embd; d++)
                                    dot += xi[d] * pwr->output->data[wr_base + d * ctx + j];
                                scores[j] = dot; if (dot > mx) mx = dot;
                            }
                            float sm = 0;
                            for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                            if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                            float* d_attn = (float*)calloc(i + 1, sizeof(float));
                            if (d_attn) {
                                for (int j = 0; j <= i; j++) {
                                    float* vj = pv->output->data + j * out_dim + v_off;
                                    for (int d = 0; d < hd; d++) d_attn[j] += dout_i[d] * vj[d];
                                }
                                for (int j = 0; j <= i; j++) {
                                    float* dvj = dv + j * out_dim + v_off;
                                    for (int d = 0; d < hd; d++) dvj[d] += attn[j] * dout_i[d];
                                }
                                float dot_da = 0;
                                for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                                for (int j = 0; j <= i; j++) {
                                    float ds = attn[j] * (d_attn[j] - dot_da);
                                    for (int d = 0; d < n_embd; d++)
                                        dx[i * n_embd + d] += ds * pwr->output->data[wr_base + d * ctx + j];
                                    for (int d = 0; d < n_embd; d++)
                                        dwr[wr_base + d * ctx + j] += ds * xi[d];
                                }
                            }
                            free(scores); free(attn); free(d_attn);
                        }
                    }
                    tape_acc_grad(e->parent1, dwr, pwr->output->len);
                    tape_acc_grad(e->parent2, dx, T * n_embd);
                    tape_acc_grad(e->parent3, dv, T * out_dim);
                }
                free(dwr); free(dx); free(dv);
            }
            break;
        }

        case NT_OP_CONCAT: {
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pa = &g_tape.entries[e->parent1];
                nt_tape_entry* pb = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int Da = pa->output->len / T; int Db = pb->output->len / T; int Dc = Da + Db;
                float* da = (float*)calloc(T * Da, sizeof(float));
                float* db = (float*)calloc(T * Db, sizeof(float));
                if (da && db) {
                    for (int t = 0; t < T; t++) {
                        for (int d = 0; d < Da; d++) da[t * Da + d] = dout[t * Dc + d];
                        for (int d = 0; d < Db; d++) db[t * Db + d] = dout[t * Dc + Da + d];
                    }
                    tape_acc_grad(e->parent1, da, T * Da);
                    tape_acc_grad(e->parent2, db, T * Db);
                }
                free(da); free(db);
            }
            break;
        }

        case NT_OP_SEQ_MATVEC_T: {
            /* Y[t] = W^T @ X[t]. W[W_rows, W_cols], X[t] has W_rows elems, Y[t] has W_cols elems.
             * dX[t][i] = sum_j dout[t][j] * W[i][j]  → dX[t] = W @ dout[t]
             * dW[i][j] = sum_t dout[t][j] * X[t][i]  → dW = X^T @ dout
             */
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int W_rows = pw->output->shape[0];
                int W_cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / W_rows;
                float* dw = (float*)calloc(pw->output->len, sizeof(float));
                float* dx = (float*)calloc(px->output->len, sizeof(float));
                int bw_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu && dw && dx) {
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_W = nt_tensor_ensure_gpu(pw->output);
                    float* d_X = nt_tensor_ensure_gpu(px->output);
                    float* d_dx = gpu_scratch(3, px->output->len);
                    float* d_dw = gpu_scratch(4, pw->output->len);
                    if (d_dout && d_W && d_X && d_dx && d_dw) {
                        /* dX[T, W_rows] = dout[T, W_cols] @ W^T[W_cols, W_rows] — NT gemm
                         *   M=T, N=W_rows, K=W_cols, A=dout, B=W */
                        gpu_sgemm_nt(T, W_rows, W_cols, d_dout, d_W, d_dx);
                        /* dW[W_rows, W_cols] = X^T[W_rows, T] @ dout[T, W_cols] — TN gemm
                         *   M=W_rows, N=W_cols, K=T, A=X(T,W_rows), B=dout(T,W_cols) */
                        gpu_sgemm_tn(W_rows, W_cols, T, d_X, d_dout, d_dw);
                        gpu_download(dx, d_dx, px->output->len);
                        gpu_download(dw, d_dw, pw->output->len);
                        bw_done_gpu = 1;
                    }
                }
#endif
                if (!bw_done_gpu && dw && dx) {
                    float* Wd = pw->output->data;
                    float* Xd = px->output->data;
#ifdef USE_BLAS
                    /* dX[T, W_rows] = dout[T, W_cols] @ W^T[W_cols, W_rows] */
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                T, W_rows, W_cols,
                                1.0f, dout, W_cols, Wd, W_cols,
                                0.0f, dx, W_rows);
                    /* dW[W_rows, W_cols] = X^T[W_rows, T] @ dout[T, W_cols] */
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                W_rows, W_cols, T,
                                1.0f, Xd, W_rows, dout, W_cols,
                                0.0f, dw, W_cols);
#else
                    for (int t = 0; t < T; t++) {
                        float* dout_t = dout + t * W_cols;
                        for (int i = 0; i < W_rows; i++)
                            for (int j = 0; j < W_cols; j++)
                                dx[t * W_rows + i] += Wd[i * W_cols + j] * dout_t[j];
                    }
                    for (int t = 0; t < T; t++) {
                        float* dout_t = dout + t * W_cols;
                        float* x_t = Xd + t * W_rows;
                        for (int i = 0; i < W_rows; i++)
                            for (int j = 0; j < W_cols; j++)
                                dw[i * W_cols + j] += x_t[i] * dout_t[j];
                    }
#endif
                }
                if (dw && dx) {
                    tape_acc_grad(e->parent1, dw, pw->output->len);
                    tape_acc_grad(e->parent2, dx, px->output->len);
                }
                free(dw); free(dx);
            }
            break;
        }

        case NT_OP_SEQ_CROSSENT: {
            if (e->parent1 >= 0) {
                nt_tape_entry* pl = &g_tape.entries[e->parent1];
                nt_tape_entry* pt = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int V = (int)e->aux2;
                int ce_done_gpu = 0;
#ifdef USE_CUDA
                /* Pure GPU backward: write straight into pl->grad GPU buffer.
                 * Kernel produces (softmax - one_hot) / T. The loss tape entry
                 * carries dout[0] via e->grad which we read on CPU (single
                 * scalar — cheap). Bake (dout[0] / T) into per-T kernel scale. */
                if (g_use_gpu) {
                    float* d_logits  = nt_tensor_ensure_gpu(pl->output);
                    float* d_targets = nt_tensor_ensure_gpu(pt->output);
                    float* d_grad_logits = gpu_scratch(11, T * V);
                    if (d_logits && d_targets && d_grad_logits) {
                        gpu_cross_entropy_backward(d_grad_logits, d_logits, d_targets, T, V);
                        if (dout[0] != 1.0f) {
                            extern void gpu_axpy(float*, const float*, int, float);
                            /* Multiply in-place: scratch *= dout[0]; do it via
                             * a brief CPU read of dout[0] (already on CPU) and
                             * a kernel-side scale. Reuse gpu_scale for in-place. */
                            gpu_scale(d_grad_logits, d_grad_logits, T * V, dout[0]);
                        }
                        tape_acc_grad_gpu(e->parent1, d_grad_logits, T * V);
                        ce_done_gpu = 1;
                    }
                }
#endif
                float* dl = ce_done_gpu ? NULL : (float*)calloc(T * V, sizeof(float));
                if (!ce_done_gpu && dl && pt) {
                    for (int t = 0; t < T; t++) {
                        float* logits_t = pl->output->data + t * V;
                        int target = (int)pt->output->data[t];
                        if (target < 0 || target >= V) target = 0;
                        float mx = logits_t[0];
                        for (int j = 1; j < V; j++)
                            if (logits_t[j] > mx) mx = logits_t[j];
                        float sum = 0;
                        for (int j = 0; j < V; j++) {
                            dl[t * V + j] = expf(logits_t[j] - mx);
                            sum += dl[t * V + j];
                        }
                        for (int j = 0; j < V; j++) dl[t * V + j] /= sum;
                        dl[t * V + target] -= 1.0f;
                        float s = dout[0] / T;
                        for (int j = 0; j < V; j++) dl[t * V + j] *= s;
                    }
                }
                if (dl) tape_acc_grad(e->parent1, dl, T * V);
                free(dl);
            }
            break;
        }

        case NT_OP_SEQ_CROSSENT_MASKED: {
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* pl = &g_tape.entries[e->parent1];
                nt_tape_entry* pt = &g_tape.entries[e->parent2];
                nt_tape_entry* pm = &g_tape.entries[e->parent3];
                int T = (int)e->aux;
                int V = (int)e->aux2;
                float n_active = 0;
                for (int t = 0; t < T; t++) n_active += pm->output->data[t];
                if (n_active <= 0) break;
                float* dl = (float*)calloc(T * V, sizeof(float));
                if (dl) {
                    for (int t = 0; t < T; t++) {
                        float m = pm->output->data[t];
                        if (m == 0.0f) continue;   // dl row stays zero
                        float* logits_t = pl->output->data + t * V;
                        int target = (int)pt->output->data[t];
                        if (target < 0 || target >= V) target = 0;
                        float mx = logits_t[0];
                        for (int j = 1; j < V; j++)
                            if (logits_t[j] > mx) mx = logits_t[j];
                        float sum = 0;
                        for (int j = 0; j < V; j++) {
                            dl[t * V + j] = expf(logits_t[j] - mx);
                            sum += dl[t * V + j];
                        }
                        for (int j = 0; j < V; j++) dl[t * V + j] /= sum;
                        dl[t * V + target] -= 1.0f;
                        float s = m * dout[0] / n_active;
                        for (int j = 0; j < V; j++) dl[t * V + j] *= s;
                    }
                    tape_acc_grad(e->parent1, dl, T * V);
                }
                free(dl);
            }
            break;
        }

        case NT_OP_GEGLU: {
            // y = GELU(x @ W1) * (x @ W2)
            // Stored: parent1 = x, parent2 = W1, parent3 = W2
            // aux = T*D_out (output total), aux2 encodes T and D_in
            // For backward: we need the intermediate values, recompute from parents
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                nt_tape_entry* pw1 = &g_tape.entries[e->parent2];
                nt_tape_entry* pw2 = &g_tape.entries[e->parent3];
                int D_out = pw1->output->shape[0];
                int D_in = pw1->output->ndim >= 2 ? pw1->output->shape[1] : pw1->output->len / D_out;
                int T = px->output->len / D_in;

                // Recompute gate and value
                float* gate = (float*)calloc(T * D_out, sizeof(float));
                float* val = (float*)calloc(T * D_out, sizeof(float));
                float* gelu_gate = (float*)calloc(T * D_out, sizeof(float));
                float* dx = (float*)calloc(px->output->len, sizeof(float));
                float* dw1 = (float*)calloc(pw1->output->len, sizeof(float));
                float* dw2 = (float*)calloc(pw2->output->len, sizeof(float));

                if (gate && val && gelu_gate && dx && dw1 && dw2) {
                    // Forward recompute: gate = x @ W1^T, val = x @ W2^T
                    for (int t = 0; t < T; t++) {
                        float* x_t = px->output->data + t * D_in;
                        for (int i = 0; i < D_out; i++) {
                            float g = 0, v = 0;
                            for (int j = 0; j < D_in; j++) {
                                g += pw1->output->data[i * D_in + j] * x_t[j];
                                v += pw2->output->data[i * D_in + j] * x_t[j];
                            }
                            gate[t * D_out + i] = g;
                            val[t * D_out + i] = v;
                            // GELU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
                            float x3 = g * g * g;
                            float inner = 0.7978845608f * (g + 0.044715f * x3);
                            float th = tanhf(inner);
                            gelu_gate[t * D_out + i] = 0.5f * g * (1.0f + th);
                        }
                    }

                    // Backward: dy = dout, y = gelu(gate) * val
                    // d_val = dout * gelu(gate)
                    // d_gelu_gate = dout * val
                    // d_gate = d_gelu_gate * gelu'(gate)
                    for (int t = 0; t < T; t++) {
                        for (int i = 0; i < D_out; i++) {
                            int ti = t * D_out + i;
                            float d_val = dout[ti] * gelu_gate[ti];
                            float g = gate[ti];
                            float x3 = g * g * g;
                            float inner = 0.7978845608f * (g + 0.044715f * x3);
                            float th = tanhf(inner);
                            float gelu_grad = 0.5f * (1.0f + th) +
                                0.5f * g * (1.0f - th * th) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * g * g);
                            float d_gate = dout[ti] * val[ti] * gelu_grad;

                            // Accumulate into weight and input grads
                            float* x_t = px->output->data + t * D_in;
                            for (int j = 0; j < D_in; j++) {
                                dw1[i * D_in + j] += d_gate * x_t[j];
                                dw2[i * D_in + j] += d_val * x_t[j];
                                dx[t * D_in + j] += d_gate * pw1->output->data[i * D_in + j];
                                dx[t * D_in + j] += d_val * pw2->output->data[i * D_in + j];
                            }
                        }
                    }
                    tape_acc_grad(e->parent1, dx, px->output->len);
                    tape_acc_grad(e->parent2, dw1, pw1->output->len);
                    tape_acc_grad(e->parent3, dw2, pw2->output->len);
                }
                free(gate); free(val); free(gelu_gate);
                free(dx); free(dw1); free(dw2);
            }
            break;
        }

        case NT_OP_DROPOUT: {
            // y = x * mask (mask encoded in output: 0 = dropped, scale = kept)
            if (e->parent1 >= 0) {
                float p = e->aux;
                float scale = (p > 0.0f && p < 1.0f) ? 1.0f / (1.0f - p) : 1.0f;
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        // If output was zero, the mask dropped it
                        gx[i] = (e->output->data[i] != 0.0f) ? dout[i] * scale : 0.0f;
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_GELU: {
            // y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        float x = px->output->data[i];
                        float x3 = x * x * x;
                        float inner = 0.7978845608f * (x + 0.044715f * x3);
                        float th = tanhf(inner);
                        float gelu_grad = 0.5f * (1.0f + th) +
                            0.5f * x * (1.0f - th * th) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
                        gx[i] = dout[i] * gelu_grad;
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_LAYERNORM: {
            // y = gamma * (x - mean) / sqrt(var + eps) + beta
            // parent1 = x, parent2 = gamma, parent3 = beta
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int n = out_len;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                int has_beta = (e->parent3 >= 0 && e->parent3 < g_tape.count);
                float* gamma_data = has_gamma ? g_tape.entries[e->parent2].output->data : NULL;

                // Recompute stats
                float mean = 0;
                for (int i = 0; i < n; i++) mean += px->output->data[i];
                mean /= n;
                float var = 0;
                for (int i = 0; i < n; i++) { float d = px->output->data[i] - mean; var += d * d; }
                var /= n;
                float inv_std = 1.0f / sqrtf(var + 1e-5f);

                // dout_eff = dout * gamma for x-gradient
                float* dout_eff = (float*)calloc(n, sizeof(float));
                if (dout_eff) {
                    for (int i = 0; i < n; i++)
                        dout_eff[i] = has_gamma ? dout[i] * gamma_data[i] : dout[i];

                    // x gradient (standard layernorm backward)
                    float sum_dout = 0, sum_dout_xhat = 0;
                    for (int i = 0; i < n; i++) {
                        float xhat = (px->output->data[i] - mean) * inv_std;
                        sum_dout += dout_eff[i];
                        sum_dout_xhat += dout_eff[i] * xhat;
                    }
                    float* gx = (float*)calloc(n, sizeof(float));
                    if (gx) {
                        for (int i = 0; i < n; i++) {
                            float xhat = (px->output->data[i] - mean) * inv_std;
                            gx[i] = inv_std * (dout_eff[i] - sum_dout / n - xhat * sum_dout_xhat / n);
                        }
                        tape_acc_grad(e->parent1, gx, n);
                    }
                    free(gx);
                    free(dout_eff);
                }

                // Gamma gradient: d_gamma[i] = dout[i] * xhat[i]
                if (has_gamma) {
                    int gn = g_tape.entries[e->parent2].output->len;
                    float* gg = (float*)calloc(gn, sizeof(float));
                    if (gg) {
                        for (int i = 0; i < n && i < gn; i++)
                            gg[i] += dout[i] * (px->output->data[i] - mean) * inv_std;
                        tape_acc_grad(e->parent2, gg, gn);
                    }
                    free(gg);
                }
                // Beta gradient: d_beta[i] = dout[i]
                if (has_beta) {
                    int bn = g_tape.entries[e->parent3].output->len;
                    float* gb = (float*)calloc(bn, sizeof(float));
                    if (gb) {
                        for (int i = 0; i < n && i < bn; i++)
                            gb[i] += dout[i];
                        tape_acc_grad(e->parent3, gb, bn);
                    }
                    free(gb);
                }
            }
            break;
        }

        case NT_OP_SEQ_LAYERNORM: {
            // Same as LAYERNORM but per-position
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                int has_gamma = (e->parent2 >= 0 && e->parent2 < g_tape.count);
                int has_beta = (e->parent3 >= 0 && e->parent3 < g_tape.count);
                float* gamma_data = has_gamma ? g_tape.entries[e->parent2].output->data : NULL;

                float* gx = (float*)calloc(T * D, sizeof(float));
                float* gg = has_gamma ? (float*)calloc(D, sizeof(float)) : NULL;
                float* gb = has_beta ? (float*)calloc(D, sizeof(float)) : NULL;

                if (gx) {
                    for (int t = 0; t < T; t++) {
                        float* x_t = px->output->data + t * D;
                        float* dout_t = dout + t * D;
                        float mean = 0;
                        for (int d = 0; d < D; d++) mean += x_t[d];
                        mean /= D;
                        float var = 0;
                        for (int d = 0; d < D; d++) { float dd = x_t[d] - mean; var += dd * dd; }
                        var /= D;
                        float inv_std = 1.0f / sqrtf(var + 1e-5f);

                        float sum_de = 0, sum_de_xhat = 0;
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            float xhat = (x_t[d] - mean) * inv_std;
                            sum_de += de;
                            sum_de_xhat += de * xhat;
                        }
                        for (int d = 0; d < D; d++) {
                            float de = has_gamma ? dout_t[d] * gamma_data[d] : dout_t[d];
                            float xhat = (x_t[d] - mean) * inv_std;
                            gx[t * D + d] = inv_std * (de - sum_de / D - xhat * sum_de_xhat / D);
                        }
                        if (gg) for (int d = 0; d < D; d++)
                            gg[d] += dout_t[d] * (x_t[d] - mean) * inv_std;
                        if (gb) for (int d = 0; d < D; d++)
                            gb[d] += dout_t[d];
                    }
                    tape_acc_grad(e->parent1, gx, T * D);
                    if (gg && has_gamma) tape_acc_grad(e->parent2, gg, D);
                    if (gb && has_beta) tape_acc_grad(e->parent3, gb, D);
                }
                free(gx); free(gg); free(gb);
            }
            break;
        }

        case NT_OP_ROPE: {
            // RoPE: rotation is orthogonal, backward = inverse rotation (transpose)
            // forward: x' = x*cos - y*sin, y' = x*sin + y*cos
            // backward: dx = dx'*cos + dy'*sin, dy = -dx'*sin + dy'*cos
            if (e->parent1 >= 0) {
                nt_tape_entry* px = &g_tape.entries[e->parent1];
                int total = px->output->len;
                int T = (int)e->aux;
                int D = total / T;
                // Recover head_dim from aux2 (stored when we fix forward)
                int head_dim = (int)e->aux2;
                if (head_dim <= 0) head_dim = D; // fallback: single head
                int n_heads = D / head_dim;

                float fb = (e->aux3 > 0.0f) ? e->aux3 : 10000.0f;
                int split_half = (e->aux4 > 0.5f) ? 1 : 0;
                int rope_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu && !split_half) {  /* GPU only handles even/odd */
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_gx   = gpu_scratch(3, total);
                    if (d_dout && d_gx) {
                        gpu_rope_backward(d_gx, d_dout, T, D, n_heads, head_dim, fb);
                        tape_acc_grad_gpu(e->parent1, d_gx, total);
                        rope_done_gpu = 1;
                    }
                }
#endif
                if (rope_done_gpu) break;
#ifdef USE_CUDA
                nt_tensor_ensure_cpu(e->grad);
#endif
                float* gx = (float*)calloc(total, sizeof(float));
                if (gx) {
                    int half = head_dim / 2;
                    for (int t = 0; t < T; t++) {
                        for (int h = 0; h < n_heads; h++) {
                            int base = t * D + h * head_dim;
                            for (int i = 0; i < half; i++) {
                                float freq = 1.0f / powf(fb, 2.0f * i / head_dim);
                                float angle = t * freq;
                                float cos_a = cosf(angle);
                                float sin_a = sinf(angle);
                                int o0 = split_half ? (base + i)        : (base + 2 * i);
                                int o1 = split_half ? (base + half + i) : (base + 2 * i + 1);
                                float dx0 = dout[o0];
                                float dx1 = dout[o1];
                                if (split_half) {
                                    /* Forward (Janus): n0 = x0*c + x1*s; n1 = -x0*s + x1*c
                                     * Backward (transpose): dx0 = c*dn0 - s*dn1; dx1 = s*dn0 + c*dn1 */
                                    gx[o0] =  dx0 * cos_a - dx1 * sin_a;
                                    gx[o1] =  dx0 * sin_a + dx1 * cos_a;
                                } else {
                                    /* Forward (notorch even/odd): n0 = x0*c - x1*s; n1 = x0*s + x1*c
                                     * Backward (transpose): dx0 = c*dn0 + s*dn1; dx1 = -s*dn0 + c*dn1 */
                                    gx[o0] = dx0 * cos_a + dx1 * sin_a;
                                    gx[o1] = -dx0 * sin_a + dx1 * cos_a;
                                }
                            }
                        }
                    }
                    tape_acc_grad(e->parent1, gx, total);
                }
                free(gx);
            }
            break;
        }

        case NT_OP_SWIGLU: {
            // y = SiLU(gate) * up, silu(g) = g * σ(g)
            // d/dg silu(g) = σ(g) + g*σ(g)*(1-σ(g)) = σ(g) * (1 + g*(1-σ(g)))
            // dgate = dout * up * silu'(gate)
            // dup   = dout * silu(gate)
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pg = &g_tape.entries[e->parent1];
                nt_tape_entry* pu = &g_tape.entries[e->parent2];
                int n = out_len;
                int swi_done_gpu = 0;
#ifdef USE_CUDA
                if (g_use_gpu) {
                    float* d_G    = nt_tensor_ensure_gpu(pg->output);
                    float* d_U    = nt_tensor_ensure_gpu(pu->output);
                    float* d_dout = nt_tensor_ensure_gpu(e->grad);
                    float* d_dg = gpu_scratch(3, n);
                    float* d_du = gpu_scratch(4, n);
                    if (d_G && d_U && d_dout && d_dg && d_du) {
                        gpu_swiglu_backward(d_dg, d_du, d_dout, d_G, d_U, n);
                        tape_acc_grad_gpu(e->parent1, d_dg, n);
                        tape_acc_grad_gpu(e->parent2, d_du, n);
                        swi_done_gpu = 1;
                    }
                }
#endif
                if (swi_done_gpu) break;
#ifdef USE_CUDA
                nt_tensor_ensure_cpu(pg->output);
                nt_tensor_ensure_cpu(pu->output);
#endif
                float* dg = (float*)calloc(n, sizeof(float));
                float* du = (float*)calloc(n, sizeof(float));
                if (dg && du) {
                    for (int i = 0; i < n; i++) {
                        float g = pg->output->data[i];
                        float u = pu->output->data[i];
                        float s = 1.0f / (1.0f + expf(-g));
                        float silu = g * s;
                        float dsilu_dg = s * (1.0f + g * (1.0f - s));
                        dg[i] = dout[i] * u * dsilu_dg;
                        du[i] = dout[i] * silu;
                    }
                    tape_acc_grad(e->parent1, dg, n);
                    tape_acc_grad(e->parent2, du, n);
                }
                free(dg); free(du);
            }
            break;
        }

        case NT_OP_BIT_LINEAR: {
            // STE: treat quantization as identity, so backward = standard matvec
            // dW[i,j] = dout[i] * x[j]
            // dx[j]   = Σ_i W[i,j] * dout[i]   (using full-precision W, per BitNet paper)
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int rows = pw->output->shape[0];
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
                if (rows > 0 && cols > 0) {
                    float* dw = (float*)calloc(rows * cols, sizeof(float));
                    if (dw) {
                        for (int i = 0; i < rows; i++)
                            for (int j = 0; j < cols; j++)
                                dw[i * cols + j] = dout[i] * px->output->data[j];
                        tape_acc_grad(e->parent1, dw, rows * cols);
                    }
                    free(dw);
                    float* dx = (float*)calloc(cols, sizeof(float));
                    if (dx) {
                        for (int j = 0; j < cols; j++) {
                            float acc = 0;
                            for (int i = 0; i < rows; i++)
                                acc += pw->output->data[i * cols + j] * dout[i];
                            dx[j] = acc;
                        }
                        tape_acc_grad(e->parent2, dx, cols);
                    }
                    free(dx);
                }
            }
            break;
        }

        case NT_OP_BIT_SEQ_LINEAR: {
            // STE backward over T positions: dW = Σ_t dout[t] ⊗ x[t]; dx[t] = W^T @ dout[t]
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                nt_tape_entry* pw = &g_tape.entries[e->parent1];
                nt_tape_entry* px = &g_tape.entries[e->parent2];
                int T = (int)e->aux;
                int rows = pw->output->shape[0];
                int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
                if (rows > 0 && cols > 0 && T > 0) {
                    float* dw = (float*)calloc(rows * cols, sizeof(float));
                    if (dw) {
                        for (int t = 0; t < T; t++) {
                            const float* dout_t = dout + t * rows;
                            const float* x_t = px->output->data + t * cols;
                            for (int i = 0; i < rows; i++) {
                                float dot_i = dout_t[i];
                                float* dw_row = dw + i * cols;
                                for (int j = 0; j < cols; j++)
                                    dw_row[j] += dot_i * x_t[j];
                            }
                        }
                        tape_acc_grad(e->parent1, dw, rows * cols);
                    }
                    free(dw);
                    float* dx = (float*)calloc(T * cols, sizeof(float));
                    if (dx) {
                        for (int t = 0; t < T; t++) {
                            const float* dout_t = dout + t * rows;
                            float* dx_t = dx + t * cols;
                            for (int j = 0; j < cols; j++) {
                                float acc = 0;
                                for (int i = 0; i < rows; i++)
                                    acc += pw->output->data[i * cols + j] * dout_t[i];
                                dx_t[j] = acc;
                            }
                        }
                        tape_acc_grad(e->parent2, dx, T * cols);
                    }
                    free(dx);
                }
            }
            break;
        }

        default:
            break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPTIMIZERS
// ═══════════════════════════════════════════════════════════════════════════════

void nt_tape_adam_step(float lr) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;
        nt_adam_state* as = &g_tape.adam[param_idx];
        if (!as->m || !as->v) { param_idx++; continue; }
        as->t++;
        int n = e->output->len;
        if (as->m->len < n) n = as->m->len;
        for (int j = 0; j < n; j++) {
            float g = e->grad->data[j];
            as->m->data[j] = beta1 * as->m->data[j] + (1.0f - beta1) * g;
            as->v->data[j] = beta2 * as->v->data[j] + (1.0f - beta2) * g * g;
            float m_hat = as->m->data[j] / (1.0f - powf(beta1, (float)as->t));
            float v_hat = as->v->data[j] / (1.0f - powf(beta2, (float)as->t));
            e->output->data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
#ifdef USE_CUDA
        nt_tensor_mark_cpu_dirty(e->output);
#endif
        param_idx++;
    }
#ifdef USE_CUDA
    if (g_use_gpu) gpu_mark_all_dirty();
#endif
}

void nt_tape_adamw_step(float lr, float weight_decay, float beta1, float beta2) {
    float eps = 1e-8f;
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;
        nt_adam_state* as = &g_tape.adam[param_idx];
        if (!as->m || !as->v) { param_idx++; continue; }
        as->t++;
        int n = e->output->len;
        if (as->m->len < n) n = as->m->len;
        float bc1 = 1.0f - powf(beta1, (float)as->t);
        float bc2 = 1.0f - powf(beta2, (float)as->t);
        float wd = (e->no_decay) ? 0.0f : weight_decay;
        for (int j = 0; j < n; j++) {
            if (wd > 0.0f)
                e->output->data[j] -= lr * wd * e->output->data[j];
            float g = e->grad->data[j];
            as->m->data[j] = beta1 * as->m->data[j] + (1.0f - beta1) * g;
            as->v->data[j] = beta2 * as->v->data[j] + (1.0f - beta2) * g * g;
            float m_hat = as->m->data[j] / bc1;
            float v_hat = as->v->data[j] / bc2;
            e->output->data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
#ifdef USE_CUDA
        nt_tensor_mark_cpu_dirty(e->output);
#endif
        param_idx++;
    }
#ifdef USE_CUDA
    if (g_use_gpu) gpu_mark_all_dirty();
#endif
}

// ── Chuck optimizer ──────────────────────────────────────────────────────────

static float chuck_ring_avg(const float* buf, int pos, int full, int start, int count) {
    int len = full ? NT_CHUCK_WINDOW : pos;
    if (len == 0 || count == 0) return 0.0f;
    float sum = 0.0f;
    int actual = 0;
    for (int i = 0; i < count && i < len; i++) {
        int idx = (start + i) % NT_CHUCK_WINDOW;
        if (idx < len || full) { sum += buf[idx]; actual++; }
    }
    return actual > 0 ? sum / actual : 0.0f;
}

static uint32_t chuck_rng = 2463534242u;
static float chuck_randn(void) {
    chuck_rng ^= chuck_rng << 13;
    chuck_rng ^= chuck_rng >> 17;
    chuck_rng ^= chuck_rng << 5;
    return 2.0f * (float)(chuck_rng) / 4294967296.0f - 1.0f;
}

// Synced with PyTorch chuck.py (iamolegataeff/chuck.optimizer) 2026-04-06
// θ -= (α × S × λ × λ_l) × m̂/(√v̂ + ε) + η
void nt_tape_chuck_step(float lr, float loss_val) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    // ── Level 1: Global loss trend → λ (dampen) ──
    nt_chuck_state* cs = &g_tape.chuck;
    if (!cs->initialized) {
        cs->dampen = 1.0f;
        cs->noise = 0.0f;
        cs->lr_scale = 1.0f;
        cs->best_macro = 1e9f;
        cs->initialized = 1;
    }
    if (cs->loss_ema == 0.0f) cs->loss_ema = loss_val;
    else cs->loss_ema = 0.99f * cs->loss_ema + 0.01f * loss_val;
    cs->loss_hist[cs->pos] = cs->loss_ema;
    cs->pos = (cs->pos + 1) % NT_CHUCK_WINDOW;
    if (cs->pos == 0) cs->full = 1;

    int len = cs->full ? NT_CHUCK_WINDOW : cs->pos;
    if (len >= 8) {
        int q = len / 4;
        if (q < 1) q = 1;
        int old_start = cs->full ? ((cs->pos) % NT_CHUCK_WINDOW) : 0;
        int recent_start = cs->full ? ((cs->pos - q + NT_CHUCK_WINDOW) % NT_CHUCK_WINDOW) : (cs->pos - q);
        float old_avg = chuck_ring_avg(cs->loss_hist, cs->pos, cs->full, old_start, q);
        float recent_avg = chuck_ring_avg(cs->loss_hist, cs->pos, cs->full, recent_start, q);
        if (old_avg > eps) {
            float trend = (recent_avg - old_avg) / old_avg;
            // Symmetric thresholds (synced with PyTorch: 0.02 / -0.02)
            if (trend > NT_CHUCK_TREND_BRAKE) cs->dampen *= NT_CHUCK_DAMP_DOWN;
            if (trend < NT_CHUCK_TREND_PUSH)  cs->dampen *= NT_CHUCK_DAMP_UP;

            // ── Level 3: Stagnation escape ──
            if (fabsf(trend) < NT_CHUCK_STAG_THRESH) {
                cs->stag++;
                if (cs->stag >= NT_CHUCK_STAG_STEPS) {
                    cs->noise = NT_CHUCK_NOISE_MAG;
                    cs->stag = 0;  // reset counter (PyTorch behavior)
                }
            } else {
                cs->stag = 0;
                cs->noise *= NT_CHUCK_NOISE_DECAY;  // exponential decay (was: reset to 0)
            }
        }
    }
    // Mean reversion: pull dampen toward 1.0 (prevents drift)
    cs->dampen = NT_CHUCK_MEAN_REVERT * cs->dampen + (1.0f - NT_CHUCK_MEAN_REVERT) * 1.0f;
    if (cs->dampen < NT_CHUCK_DAMP_LO) cs->dampen = NT_CHUCK_DAMP_LO;
    if (cs->dampen > NT_CHUCK_DAMP_HI) cs->dampen = NT_CHUCK_DAMP_HI;

    // ── Level 9: Multi-scale awareness (macro patience) ──
    cs->global_step++;
    if (cs->macro_ema == 0.0f) cs->macro_ema = loss_val;
    else cs->macro_ema = 0.999f * cs->macro_ema + 0.001f * loss_val;
    if (cs->global_step % NT_CHUCK_MACRO_INT == 0 && cs->global_step > NT_CHUCK_WINDOW) {
        if (cs->macro_ema > cs->best_macro * 0.999f) {
            cs->macro_stag++;
            if (cs->macro_stag >= NT_CHUCK_MACRO_PAT) {
                cs->lr_scale *= NT_CHUCK_MACRO_DECAY;
                if (cs->lr_scale < 0.05f) cs->lr_scale = 0.05f;
                cs->macro_stag = 0;
            }
        } else {
            cs->best_macro = cs->macro_ema;
            cs->macro_stag = 0;
            // LR recovery when improving (PyTorch: lr_scale *= 1.2)
            if (cs->lr_scale < 1.0f) {
                cs->lr_scale *= 1.2f;
                if (cs->lr_scale > 1.0f) cs->lr_scale = 1.0f;
            }
        }
    }

    float global_lambda = cs->dampen;
    float noise_mag = cs->noise;

    // ── Level 2: Per-param gradient norm + Adam update ──
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;
        nt_adam_state* as = &g_tape.adam[param_idx];
        nt_chuck_param_state* cp = &g_tape.chuck_params[param_idx];
        if (cp->dampen == 0.0f) cp->dampen = 1.0f;
        if (cp->frozen) { param_idx++; continue; }
        if (!as->m || !as->v) { param_idx++; continue; }

        int n = e->output->len;
        if (as->m->len < n) n = as->m->len;
        float gnorm = 0.0f;
#ifdef USE_CUDA
        if (g_use_gpu) {
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            if (d_g) {
                gnorm = gpu_nrm2(d_g, n);
            } else {
                nt_tensor_ensure_cpu(e->grad);
                for (int j = 0; j < n; j++) gnorm += e->grad->data[j] * e->grad->data[j];
                gnorm = sqrtf(gnorm);
            }
        } else
#endif
        {
            for (int j = 0; j < n; j++) gnorm += e->grad->data[j] * e->grad->data[j];
            gnorm = sqrtf(gnorm);
        }

        cp->grad_hist[cp->pos] = gnorm;
        cp->pos = (cp->pos + 1) % NT_CHUCK_WINDOW;
        if (cp->pos == 0) cp->full = 1;

        int plen = cp->full ? NT_CHUCK_WINDOW : cp->pos;
        if (plen >= 8) {
            int q = plen / 4; if (q < 1) q = 1;
            int old_start = cp->full ? ((cp->pos) % NT_CHUCK_WINDOW) : 0;
            int recent_start = cp->full ? ((cp->pos - q + NT_CHUCK_WINDOW) % NT_CHUCK_WINDOW) : (cp->pos - q);
            float old_gn = chuck_ring_avg(cp->grad_hist, cp->pos, cp->full, old_start, q);
            float recent_gn = chuck_ring_avg(cp->grad_hist, cp->pos, cp->full, recent_start, q);
            if (old_gn > eps) {
                float gtrend = (recent_gn - old_gn) / old_gn;
                // Per-param: 0.05 thresholds (symmetric, PyTorch)
                if (gtrend > 0.05f)  cp->dampen *= NT_CHUCK_DAMP_UP;   // grad rising → boost
                if (gtrend < -0.05f) cp->dampen *= NT_CHUCK_DAMP_DOWN;  // grad settling → ease
            }
            if (gnorm < NT_CHUCK_FREEZE_THRESH) {
                cp->stag++;
                if (cp->stag >= NT_CHUCK_STAG_STEPS) cp->frozen = 1;
            } else {
                cp->stag = 0;
            }
            // Per-param mean reversion
            cp->dampen = NT_CHUCK_MEAN_REVERT * cp->dampen + (1.0f - NT_CHUCK_MEAN_REVERT) * 1.0f;
            if (cp->dampen < NT_CHUCK_DAMP_LO) cp->dampen = NT_CHUCK_DAMP_LO;
            if (cp->dampen > NT_CHUCK_DAMP_HI) cp->dampen = NT_CHUCK_DAMP_HI;
        }

        float param_lambda = cp->dampen;
        float effective_lr = lr * global_lambda * param_lambda * cs->lr_scale;
        as->t++;
        float bc1 = 1.0f - powf(beta1, (float)as->t);
        float bc2 = 1.0f - powf(beta2, (float)as->t);
        int chuck_done_gpu = 0;
#ifdef USE_CUDA
        /* GPU path: trivially parallel m,v update + param step. Skip when
         * Chuck noise injection is active (rare stagnation escape) since
         * CPU RNG is harder to port deterministically. */
        if (g_use_gpu && noise_mag == 0.0f) {
            float* d_p = nt_tensor_ensure_gpu(e->output);
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            float* d_m = nt_tensor_ensure_gpu(as->m);
            float* d_v = nt_tensor_ensure_gpu(as->v);
            if (d_p && d_g && d_m && d_v) {
                gpu_chuck_inner(d_p, d_m, d_v, d_g, n, beta1, beta2, bc1, bc2, effective_lr, eps);
                /* GPU is now source of truth for param, m, v. Mark CPU stale —
                 * next forward will read GPU directly without re-upload. */
                e->output->cpu_dirty = 1; e->output->gpu_valid = 1;
                as->m->cpu_dirty = 1;     as->m->gpu_valid = 1;
                as->v->cpu_dirty = 1;     as->v->gpu_valid = 1;
                chuck_done_gpu = 1;
            }
        }
#endif
        if (!chuck_done_gpu) {
            for (int j = 0; j < n; j++) {
                float g = e->grad->data[j];
                as->m->data[j] = beta1 * as->m->data[j] + (1.0f - beta1) * g;
                as->v->data[j] = beta2 * as->v->data[j] + (1.0f - beta2) * g * g;
                float m_hat = as->m->data[j] / bc1;
                float v_hat = as->v->data[j] / bc2;
                float update = effective_lr * m_hat / (sqrtf(v_hat) + eps);
                if (noise_mag > 0.0f) update += noise_mag * chuck_randn();
                e->output->data[j] -= update;
            }
#ifdef USE_CUDA
            /* CPU just mutated param weights — invalidate GPU mirror so next
             * forward re-uploads. */
            nt_tensor_mark_cpu_dirty(e->output);
#endif
        }
        param_idx++;
    }
#ifdef USE_CUDA
    /* Conservative belt-and-braces: mark global weight cache dirty too.
     * Cached entries (gpu_cache_weight) are not the same as per-tensor
     * d_data, but coa_v1_janus does not use the named weight cache today;
     * harmless if empty. */
    if (g_use_gpu) gpu_mark_all_dirty();
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRADIENT UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

float nt_tape_clip_grads(float max_norm) {
    float total_norm_sq = 0.0f;
    for (int i = 0; i < g_tape.count; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;
        int n = e->output->len;
        if (e->grad->len < n) n = e->grad->len;
#ifdef USE_CUDA
        if (g_use_gpu) {
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            if (d_g) {
                float nrm = gpu_nrm2(d_g, n);
                total_norm_sq += nrm * nrm;
                continue;
            }
        }
        nt_tensor_ensure_cpu(e->grad);
#endif
        for (int j = 0; j < n; j++) {
            float g = e->grad->data[j];
            total_norm_sq += g * g;
        }
    }
    float total_norm = sqrtf(total_norm_sq);
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (int i = 0; i < g_tape.count; i++) {
            nt_tape_entry* e = &g_tape.entries[i];
            if (!e->is_param || !e->grad) continue;
            int n = e->output->len;
            if (e->grad->len < n) n = e->grad->len;
#ifdef USE_CUDA
            if (g_use_gpu) {
                float* d_g = nt_tensor_ensure_gpu(e->grad);
                if (d_g) {
                    gpu_sscal(d_g, n, scale);
                    /* GPU is now source of truth — mark CPU stale so later
                     * reads pull fresh values. */
                    e->grad->gpu_valid = 1;
                    e->grad->cpu_dirty = 1;
                    continue;
                }
            }
#endif
            for (int j = 0; j < n; j++) e->grad->data[j] *= scale;
#ifdef USE_CUDA
            e->grad->gpu_valid = 0;
            e->grad->cpu_dirty = 0;
#endif
        }
    }
    return total_norm;
}

void nt_tape_accum_grads(void) {
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;
        nt_adam_state* as = &g_tape.adam[param_idx];
        int n = e->output->len;
        if (!as->acc_grad) {
            as->acc_grad = nt_tensor_new(n);
        } else if (as->acc_grad->len < n) {
            nt_tensor_free(as->acc_grad);
            as->acc_grad = nt_tensor_new(n);
        }
        for (int j = 0; j < n && j < as->acc_grad->len; j++)
            as->acc_grad->data[j] += e->grad->data[j];
        param_idx++;
    }
}

void nt_tape_apply_accum(int n_accum) {
    float scale = (n_accum > 1) ? 1.0f / (float)n_accum : 1.0f;
    int param_idx = 0;
    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param) continue;
        nt_adam_state* as = &g_tape.adam[param_idx];
        if (as->acc_grad) {
            int n = e->output->len;
            if (as->acc_grad->len < n) n = as->acc_grad->len;
            if (!e->grad) e->grad = nt_tensor_new(n);
            for (int j = 0; j < n; j++) {
                e->grad->data[j] = as->acc_grad->data[j] * scale;
                as->acc_grad->data[j] = 0.0f;
            }
        }
        param_idx++;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRAINING MODE
// ═══════════════════════════════════════════════════════════════════════════════

static int g_training_mode = 1;

void nt_train_mode(int training) { g_training_mode = training; }
int  nt_is_training(void) { return g_training_mode; }

// ═══════════════════════════════════════════════════════════════════════════════
// LR SCHEDULE
// ═══════════════════════════════════════════════════════════════════════════════

nt_schedule nt_schedule_cosine(float base_lr, int warmup_steps, int total_steps, float min_lr) {
    nt_schedule s = {0};
    s.type = NT_SCHED_COSINE;
    s.base_lr = base_lr;
    s.min_lr = min_lr;
    s.warmup_steps = warmup_steps;
    s.total_steps = total_steps > 0 ? total_steps : 1;
    return s;
}

nt_schedule nt_schedule_step(float base_lr, int warmup_steps, int step_size, float gamma) {
    nt_schedule s = {0};
    s.type = NT_SCHED_STEP;
    s.base_lr = base_lr;
    s.warmup_steps = warmup_steps;
    s.step_size = step_size > 0 ? step_size : 1;
    s.step_gamma = gamma > 0 ? gamma : 0.1f;
    return s;
}

nt_schedule nt_schedule_linear(float base_lr, int warmup_steps, int total_steps, float min_lr) {
    nt_schedule s = {0};
    s.type = NT_SCHED_LINEAR;
    s.base_lr = base_lr;
    s.min_lr = min_lr;
    s.warmup_steps = warmup_steps;
    s.total_steps = total_steps > 0 ? total_steps : 1;
    return s;
}

float nt_schedule_get_lr(nt_schedule* s) {
    if (!s) return 0.001f;
    int step = s->current_step++;
    float lr = s->base_lr;

    // Warmup phase: linear ramp from min_lr to base_lr
    if (step < s->warmup_steps && s->warmup_steps > 0) {
        float t = (float)step / (float)s->warmup_steps;
        return s->min_lr + t * (s->base_lr - s->min_lr);
    }

    int decay_step = step - s->warmup_steps;

    switch (s->type) {
    case NT_SCHED_COSINE: {
        int decay_total = s->total_steps - s->warmup_steps;
        if (decay_total <= 0) return lr;
        float progress = (float)decay_step / (float)decay_total;
        if (progress > 1.0f) progress = 1.0f;
        lr = s->min_lr + 0.5f * (s->base_lr - s->min_lr) * (1.0f + cosf(3.14159265f * progress));
        break;
    }
    case NT_SCHED_STEP: {
        int n_decays = decay_step / s->step_size;
        lr = s->base_lr * powf(s->step_gamma, (float)n_decays);
        break;
    }
    case NT_SCHED_LINEAR: {
        int decay_total = s->total_steps - s->warmup_steps;
        if (decay_total <= 0) return lr;
        float progress = (float)decay_step / (float)decay_total;
        if (progress > 1.0f) progress = 1.0f;
        lr = s->base_lr - progress * (s->base_lr - s->min_lr);
        break;
    }
    default:
        break;
    }
    return lr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// NaN/Inf GUARD
// ═══════════════════════════════════════════════════════════════════════════════

nt_nan_guard nt_nan_guard_new(void) {
    nt_nan_guard g = {0};
    g.loss_scale = 1.0f;
    g.scale_factor = 2.0f;
    g.scale_window = 100;
    return g;
}

int nt_nan_guard_check(nt_nan_guard* guard) {
    if (!guard) return 1;
    int has_nan = 0;

    for (int i = 0; i < g_tape.count; i++) {
        nt_tape_entry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;
        int n = e->grad->len;
#ifdef USE_CUDA
        if (g_use_gpu) {
            float* d_g = nt_tensor_ensure_gpu(e->grad);
            if (d_g) {
                /* NaN/Inf propagate through Snrm2: result = NaN if any input is NaN,
                 * Inf if any input is Inf. Cheap O(n) GPU reduction vs CPU loop. */
                float nrm = gpu_nrm2(d_g, n);
                if (nrm != nrm || nrm == 1.0f/0.0f || nrm == -1.0f/0.0f) {
                    has_nan = 1;
                }
                if (has_nan) break;
                continue;
            }
        }
#endif
        for (int j = 0; j < n; j++) {
            float g = e->grad->data[j];
            if (g != g || g == 1.0f/0.0f || g == -1.0f/0.0f) {  // NaN or Inf
                has_nan = 1;
                break;
            }
        }
        if (has_nan) break;
    }

    if (has_nan) {
        // Zero all gradients — don't apply this step
        for (int i = 0; i < g_tape.count; i++) {
            nt_tape_entry* e = &g_tape.entries[i];
            if (!e->is_param || !e->grad) continue;
            memset(e->grad->data, 0, e->grad->len * sizeof(float));
        }
        guard->loss_scale /= guard->scale_factor;
        if (guard->loss_scale < 1e-8f) guard->loss_scale = 1e-8f;
        guard->stable_steps = 0;
        guard->total_nan_count++;
        guard->skipped_steps++;
        return 0;
    }

    // Clean step
    guard->stable_steps++;
    if (guard->stable_steps >= guard->scale_window) {
        guard->loss_scale *= guard->scale_factor;
        if (guard->loss_scale > 65536.0f) guard->loss_scale = 65536.0f;
        guard->stable_steps = 0;
    }
    return 1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROFILER
// ═══════════════════════════════════════════════════════════════════════════════

#include <sys/time.h>

static nt_profiler g_profiler = {0};
static long g_alloc_bytes = 0;

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void nt_profiler_enable(void)  { g_profiler.enabled = 1; }
void nt_profiler_disable(void) { g_profiler.enabled = 0; }
void nt_profiler_reset(void)   { memset(&g_profiler, 0, sizeof(g_profiler)); }
nt_profiler* nt_profiler_get(void) { return &g_profiler; }

void nt_profiler_print(void) {
    printf("── notorch profiler ──\n");
    printf("  ops: %d, params: %d (%ld elements, %.2f MB)\n",
           g_profiler.n_ops, g_profiler.n_params,
           g_profiler.total_param_elems,
           (float)g_profiler.total_param_elems * 4.0f / 1048576.0f);
    printf("  forward:   %.2f ms\n", g_profiler.forward_ms);
    printf("  backward:  %.2f ms\n", g_profiler.backward_ms);
    printf("  optimizer: %.2f ms\n", g_profiler.optimizer_ms);
    printf("  peak mem:  %.2f MB\n", (float)g_profiler.peak_memory / 1048576.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD OPS
// ═══════════════════════════════════════════════════════════════════════════════

int nt_embedding(int wte_idx, int token_id) {
    if (wte_idx < 0 || wte_idx >= g_tape.count) return -1;
    nt_tape_entry* wte = &g_tape.entries[wte_idx];
    int cols = wte->output->ndim >= 2 ? wte->output->shape[1] : wte->output->len;
    int rows = wte->output->len / cols;
    if (token_id < 0 || token_id >= rows) return -1;
    nt_tensor* out = nt_tensor_new(cols);
    if (!out) return -1;
    memcpy(out->data, wte->output->data + token_id * cols, cols * sizeof(float));
    int idx = nt_tape_record(out, NT_OP_EMB_LOOKUP, wte_idx, -1, (float)token_id);
    nt_tensor_free(out); // tape holds ref
    return idx;
}

int nt_seq_embedding(int wte_idx, int wpe_idx, int tokens_idx, int T, int D) {
    if (wte_idx < 0 || tokens_idx < 0) return -1;
    nt_tape_entry* wte = &g_tape.entries[wte_idx];
    nt_tape_entry* tok = &g_tape.entries[tokens_idx];
    int wte_rows = wte->output->ndim >= 2 ? wte->output->shape[0] : wte->output->len / D;

    nt_tensor* out = nt_tensor_new(T * D);
    if (!out) return -1;

#ifdef USE_CUDA
    if (g_use_gpu && wpe_idx < 0) {
        /* Pure WTE lookup on GPU. Skip WPE branch to keep kernel simple. */
        float* d_wte = nt_tensor_ensure_gpu(wte->output);
        float* d_tok = nt_tensor_ensure_gpu(tok->output);
        float* d_out = nt_tensor_ensure_gpu(out);
        if (d_wte && d_tok && d_out) {
            gpu_seq_embedding_forward(d_out, d_wte, d_tok, T, D, wte_rows);
            nt_tensor_mark_gpu_fresh(out);
            int idx = nt_tape_record3(out, NT_OP_SEQ_EMBED, wte_idx, wpe_idx, tokens_idx, (float)T, (float)D);
            nt_tensor_free(out);
            return idx;
        }
    }
    nt_tensor_ensure_cpu(wte->output);
    nt_tensor_ensure_cpu(tok->output);
#endif
    for (int t = 0; t < T; t++) {
        int tid = (int)tok->output->data[t];
        if (tid < 0) tid = 0;
        if (tid >= wte_rows) tid = wte_rows - 1;
        for (int d = 0; d < D; d++)
            out->data[t * D + d] = wte->output->data[tid * D + d];
    }
    /* Add position embeddings if provided */
    if (wpe_idx >= 0) {
        nt_tape_entry* wpe = &g_tape.entries[wpe_idx];
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(wpe->output);
#endif
        int wpe_rows = wpe->output->ndim >= 2 ? wpe->output->shape[0] : wpe->output->len / D;
        for (int t = 0; t < T; t++) {
            int pos = t < wpe_rows ? t : wpe_rows - 1;
            for (int d = 0; d < D; d++)
                out->data[t * D + d] += wpe->output->data[pos * D + d];
        }
    }
    int idx = nt_tape_record3(out, NT_OP_SEQ_EMBED, wte_idx, wpe_idx, tokens_idx, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

int nt_linear(int w_idx, int x_idx, int bias_idx) {
    if (w_idx < 0 || x_idx < 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int rows = pw->output->shape[0];
    int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;

    nt_tensor* out = nt_tensor_new(rows);
    if (!out) return -1;
    for (int i = 0; i < rows; i++) {
        float s = 0;
        for (int j = 0; j < cols; j++)
            s += pw->output->data[i * cols + j] * px->output->data[j];
        out->data[i] = s;
    }
    int idx = nt_tape_record(out, NT_OP_MATVEC, w_idx, x_idx, 0);
    nt_tensor_free(out);

    if (bias_idx >= 0) {
        idx = nt_add(idx, bias_idx);
    }
    return idx;
}

int nt_seq_linear(int w_idx, int x_idx, int T) {
    if (w_idx < 0 || x_idx < 0 || T <= 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int out_dim = pw->output->shape[0];
    int in_dim = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / out_dim;

    nt_tensor* out = nt_tensor_new(T * out_dim);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        /* Y(T, out_dim) = X(T, in_dim) @ W^T(in_dim, out_dim)
         * gpu_sgemm_nt: C(M,N) = A(M,K) × B^T(N,K), so M=T, N=out_dim, K=in_dim. */
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_W = nt_tensor_ensure_gpu(pw->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_W && d_Y) {
            gpu_sgemm_nt(T, out_dim, in_dim, d_X, d_W, d_Y);
            nt_tensor_mark_gpu_fresh(out);  /* keep CPU mirror coherent for non-GPU ops */
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        float* W = pw->output->data;
        float* X = px->output->data;
        float* Y = out->data;
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, out_dim, in_dim,
                    1.0f, X, in_dim, W, in_dim,
                    0.0f, Y, out_dim);
#else
        for (int t = 0; t < T; t++) {
            float* x_t = X + t * in_dim;
            float* y_t = Y + t * out_dim;
            for (int i = 0; i < out_dim; i++) {
                float s = 0;
                for (int j = 0; j < in_dim; j++)
                    s += W[i * in_dim + j] * x_t[j];
                y_t[i] = s;
            }
        }
#endif
    }

    int idx = nt_tape_record3(out, NT_OP_SEQ_MATVEC, w_idx, x_idx, -1, (float)T, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_linear_t(int w_idx, int x_idx, int T) {
    if (w_idx < 0 || x_idx < 0 || T <= 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int W_rows = pw->output->shape[0];
    int W_cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / W_rows;

    /* W^T @ X[t]: input dim = W_rows, output dim = W_cols */
    nt_tensor* out = nt_tensor_new(T * W_cols);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        /* Y[T, W_cols] = X[T, W_rows] @ W[W_rows, W_cols] — NN gemm.
         * gpu_sgemm_nn(M, N, K, A, B, C):  C(M,N) = A(M,K) × B(K,N)
         *   M = T, N = W_cols, K = W_rows. */
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_W = nt_tensor_ensure_gpu(pw->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_W && d_Y) {
            gpu_sgemm_nn(T, W_cols, W_rows, d_X, d_W, d_Y);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        float* W = pw->output->data;
        float* X = px->output->data;
        float* Y = out->data;
#ifdef USE_BLAS
        /* Y[T, W_cols] = X[T, W_rows] @ W[W_rows, W_cols] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, W_cols, W_rows,
                    1.0f, X, W_rows, W, W_cols,
                    0.0f, Y, W_cols);
#else
        for (int t = 0; t < T; t++) {
            float* x_t = X + t * W_rows;
            float* y_t = Y + t * W_cols;
            for (int j = 0; j < W_cols; j++) {
                float s = 0;
                for (int i = 0; i < W_rows; i++)
                    s += W[i * W_cols + j] * x_t[i];
                y_t[j] = s;
            }
        }
#endif
    }

    int idx = nt_tape_record3(out, NT_OP_SEQ_MATVEC_T, w_idx, x_idx, -1, (float)T, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_rmsnorm(int x_idx, int gamma_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;

    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    float ss = 0;
    for (int i = 0; i < n; i++) ss += px->output->data[i] * px->output->data[i];
    float rms = sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out->data[i] = px->output->data[i] / rms;

    // Apply gamma scale if provided
    if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
        nt_tape_entry* pg = &g_tape.entries[gamma_idx];
        for (int i = 0; i < n && i < pg->output->len; i++)
            out->data[i] *= pg->output->data[i];
    }

    int g_idx = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int idx = nt_tape_record(out, NT_OP_RMSNORM, x_idx, g_idx, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_rmsnorm(int x_idx, int gamma_idx, int T, int D) {
    if (x_idx < 0 || T <= 0 || D <= 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];

    nt_tensor* out = nt_tensor_new(T * D);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        /* GPU forward: y = (x / rms) * gamma (single dispatch, gamma optional). */
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        float* d_gamma = NULL;
        if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
            nt_tape_entry* pg = &g_tape.entries[gamma_idx];
            d_gamma = nt_tensor_ensure_gpu(pg->output);
        }
        if (d_X && d_Y) {
            gpu_seq_rmsnorm_gamma(d_Y, d_X, d_gamma, T, D);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        for (int t = 0; t < T; t++) {
            float* x_t = px->output->data + t * D;
            float* o_t = out->data + t * D;
            float ss = 0;
            for (int d = 0; d < D; d++) ss += x_t[d] * x_t[d];
            float rms = sqrtf(ss / D + 1e-6f);
            for (int d = 0; d < D; d++) o_t[d] = x_t[d] / rms;
        }
        if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
            nt_tape_entry* pg = &g_tape.entries[gamma_idx];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D && d < pg->output->len; d++)
                    out->data[t * D + d] *= pg->output->data[d];
        }
    }

    int g_idx2 = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int idx = nt_tape_record3(out, NT_OP_SEQ_RMSNORM, x_idx, g_idx2, -1, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

int nt_silu(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_Y) {
            gpu_silu(d_Y, d_X, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        for (int i = 0; i < n; i++) {
            float x = px->output->data[i];
            out->data[i] = x / (1.0f + expf(-x));
        }
    }
    int idx = nt_tape_record(out, NT_OP_SILU, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_sigmoid(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    for (int i = 0; i < n; i++) {
        float x = px->output->data[i];
        /* numerically stable */
        out->data[i] = (x >= 0) ? 1.0f / (1.0f + expf(-x))
                                : expf(x) / (1.0f + expf(x));
    }
    int idx = nt_tape_record(out, NT_OP_SIGMOID, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_scale_by_t(int x_idx, int a_idx) {
    if (x_idx < 0 || a_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    if (pa->output->len != 1) return -1;  /* scalar required */
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    float a_val = pa->output->data[0];
    for (int i = 0; i < n; i++) out->data[i] = a_val * px->output->data[i];
    int idx = nt_tape_record3(out, NT_OP_SCALE_BY_T, x_idx, a_idx, -1, 0, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_geglu(int x_idx, int w1_idx, int w2_idx, int T, int D_in, int D_out) {
    if (x_idx < 0 || w1_idx < 0 || w2_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    nt_tape_entry* pw1 = &g_tape.entries[w1_idx];
    nt_tape_entry* pw2 = &g_tape.entries[w2_idx];

    nt_tensor* out = nt_tensor_new(T * D_out);
    if (!out) return -1;

    for (int t = 0; t < T; t++) {
        float* x_t = px->output->data + t * D_in;
        for (int i = 0; i < D_out; i++) {
            float gate = 0, val = 0;
            for (int j = 0; j < D_in; j++) {
                gate += pw1->output->data[i * D_in + j] * x_t[j];
                val += pw2->output->data[i * D_in + j] * x_t[j];
            }
            // GELU approximation
            float x3 = gate * gate * gate;
            float inner = 0.7978845608f * (gate + 0.044715f * x3);
            float gelu = 0.5f * gate * (1.0f + tanhf(inner));
            out->data[t * D_out + i] = gelu * val;
        }
    }

    int idx = nt_tape_record3(out, NT_OP_GEGLU, x_idx, w1_idx, w2_idx, (float)(T * D_out), 0);
    nt_tensor_free(out);
    return idx;
}

int nt_softmax(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    float mx = px->output->data[0];
    for (int i = 1; i < n; i++) if (px->output->data[i] > mx) mx = px->output->data[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out->data[i] = expf(px->output->data[i] - mx); sum += out->data[i]; }
    for (int i = 0; i < n; i++) out->data[i] /= sum;
    int idx = nt_tape_record(out, NT_OP_SOFTMAX, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_causal_attention(int q_idx, int k_idx, int v_idx, int T, int D) {
    if (q_idx < 0 || k_idx < 0 || v_idx < 0) return -1;
    nt_tape_entry* pq = &g_tape.entries[q_idx];
    nt_tape_entry* pk = &g_tape.entries[k_idx];
    nt_tape_entry* pv = &g_tape.entries[v_idx];
    float scale = 1.0f / sqrtf((float)D);
    nt_tensor* out = nt_tensor_new(T * D);
    if (!out) return -1;
    for (int i = 0; i < T; i++) {
        float* qi = pq->output->data + i * D;
        float* scores = (float*)calloc(i + 1, sizeof(float));
        if (!scores) { nt_tensor_free(out); return -1; }
        float mx = -1e30f;
        for (int j = 0; j <= i; j++) {
            float* kj = pk->output->data + j * D;
            float dot = 0;
            for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
            scores[j] = dot * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        float sum = 0;
        for (int j = 0; j <= i; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        if (sum > 0) for (int j = 0; j <= i; j++) scores[j] /= sum;
        float* oi = out->data + i * D;
        for (int d = 0; d < D; d++) oi[d] = 0;
        for (int j = 0; j <= i; j++) {
            float* vj = pv->output->data + j * D;
            for (int d = 0; d < D; d++) oi[d] += scores[j] * vj[d];
        }
        free(scores);
    }
    int idx = nt_tape_record3(out, NT_OP_CAUSAL_ATTN, q_idx, k_idx, v_idx, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

int nt_mh_causal_attention(int q_idx, int k_idx, int v_idx, int T, int head_dim) {
    if (q_idx < 0 || k_idx < 0 || v_idx < 0) return -1;
    nt_tape_entry* pq = &g_tape.entries[q_idx];
    int D = pq->output->len / T;
    int n_heads = D / head_dim;
    if (n_heads <= 0 || D % head_dim != 0) return -1;
    float scale = 1.0f / sqrtf((float)head_dim);

    nt_tensor* out = nt_tensor_new(T * D);
    if (!out) return -1;
    nt_tape_entry* pk = &g_tape.entries[k_idx];
    nt_tape_entry* pv = &g_tape.entries[v_idx];

#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_Q = nt_tensor_ensure_gpu(pq->output);
        float* d_K = nt_tensor_ensure_gpu(pk->output);
        float* d_V = nt_tensor_ensure_gpu(pv->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        /* Scratch buffer for attention scores: n_heads * T * T floats. */
        float* d_scores = gpu_scratch(1, n_heads * T * T);
        if (d_Q && d_K && d_V && d_Y && d_scores) {
            gpu_multi_head_attention(d_Q, d_K, d_V, d_Y, d_scores, T, D, n_heads);
            nt_tensor_mark_gpu_fresh(out);
            int idx = nt_tape_record3(out, NT_OP_MH_CAUSAL_ATTN, q_idx, k_idx, v_idx, (float)T, (float)head_dim);
            nt_tensor_free(out);
            return idx;
        }
    }
#endif

    float* scores_buf = (float*)malloc(T * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        int ho = h * head_dim;
        for (int i = 0; i < T; i++) {
            float* qi = pq->output->data + i * D + ho;
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float* kj = pk->output->data + j * D + ho;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                scores_buf[j] = dot * scale;
                if (scores_buf[j] > mx) mx = scores_buf[j];
            }
            float sum = 0;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sum += scores_buf[j]; }
            if (sum > 0) for (int j = 0; j <= i; j++) scores_buf[j] /= sum;
            float* oi = out->data + i * D + ho;
            for (int d = 0; d < head_dim; d++) oi[d] = 0;
            for (int j = 0; j <= i; j++) {
                float* vj = pv->output->data + j * D + ho;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(scores_buf);

    int idx = nt_tape_record3(out, NT_OP_MH_CAUSAL_ATTN, q_idx, k_idx, v_idx, (float)T, (float)head_dim);
    nt_tensor_free(out);
    return idx;
}

int nt_gqa_causal_attention(int q_idx, int k_idx, int v_idx, int T, int head_dim, int n_heads, int n_kv_heads) {
    if (q_idx < 0 || k_idx < 0 || v_idx < 0) return -1;
    int Q_D = n_heads * head_dim;
    int KV_D = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    nt_tensor* out = nt_tensor_new(T * Q_D);
    if (!out) return -1;
    nt_tape_entry* pq = &g_tape.entries[q_idx];
    nt_tape_entry* pk = &g_tape.entries[k_idx];
    nt_tape_entry* pv = &g_tape.entries[v_idx];

    float* scores_buf = (float*)malloc(T * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        int q_off = h * head_dim;
        int kv_off = kv_h * head_dim;
        for (int i = 0; i < T; i++) {
            float* qi = pq->output->data + i * Q_D + q_off;
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float* kj = pk->output->data + j * KV_D + kv_off;
                float dot = 0;
                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                scores_buf[j] = dot * scale;
                if (scores_buf[j] > mx) mx = scores_buf[j];
            }
            float sum = 0;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sum += scores_buf[j]; }
            if (sum > 0) for (int j = 0; j <= i; j++) scores_buf[j] /= sum;
            float* oi = out->data + i * Q_D + q_off;
            for (int d = 0; d < head_dim; d++) oi[d] = 0;
            for (int j = 0; j <= i; j++) {
                float* vj = pv->output->data + j * KV_D + kv_off;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(scores_buf);

    int idx = nt_tape_record4(out, NT_OP_GQA_ATTN, q_idx, k_idx, v_idx,
                              (float)T, (float)head_dim, (float)n_heads, (float)n_kv_heads);
    nt_tensor_free(out);
    return idx;
}

int nt_rrpram_attention(int wr_idx, int x_idx, int v_idx, int T, int n_embd, int nr_heads, int head_dim) {
    if (wr_idx < 0 || x_idx < 0 || v_idx < 0) return -1;
    int out_dim = nr_heads * head_dim;
    nt_tensor* out = nt_tensor_new(T * out_dim);
    if (!out) return -1;
    nt_tape_entry* pwr = &g_tape.entries[wr_idx];
    nt_tape_entry* px  = &g_tape.entries[x_idx];
    nt_tape_entry* pv  = &g_tape.entries[v_idx];
    int ctx = pwr->output->len / (nr_heads * n_embd);
    float* scores_buf = (float*)malloc(T * sizeof(float));
    for (int h = 0; h < nr_heads; h++) {
        int wr_base = h * n_embd * ctx;
        int v_off = h * head_dim;
        for (int i = 0; i < T; i++) {
            float* xi = px->output->data + i * n_embd;
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float dot = 0;
                for (int d = 0; d < n_embd; d++)
                    dot += xi[d] * pwr->output->data[wr_base + d * ctx + j];
                scores_buf[j] = dot;
                if (dot > mx) mx = dot;
            }
            float sm = 0;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sm += scores_buf[j]; }
            if (sm > 0) for (int j = 0; j <= i; j++) scores_buf[j] /= sm;
            float* oi = out->data + i * out_dim + v_off;
            for (int d = 0; d < head_dim; d++) oi[d] = 0;
            for (int j = 0; j <= i; j++) {
                float* vj = pv->output->data + j * out_dim + v_off;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(scores_buf);
    int idx = nt_tape_record4(out, NT_OP_RRPRAM_ATTN, wr_idx, x_idx, v_idx,
                              (float)T, (float)n_embd, (float)nr_heads, (float)head_dim);
    nt_tensor_free(out);
    return idx;
}

/* ════════════════════════════════════════════════════════════════════════
 * Low-rank RRPRAM: Wr = Wr_a × Wr_b factorized.
 *
 * wr_combined layout: [Wr_a flat | Wr_b flat]
 *   Wr_a: H*E*R floats — head h offset = h*E*R, indexed [d, r] = h*E*R + d*R + r
 *   Wr_b: H*R*T_r floats — head h offset = H*E*R + h*R*T_r, indexed [r, j] = ... + r*T_r + j
 *   Total length = H*R*(E + T_r)
 *
 * Assumption: T_r == T (positional dim equals current ctx).
 * Rank derived: R = wr_combined->len / (H * (E + T))
 *
 * Per head h, position i (causal: j ≤ i):
 *   u[r]      = Σ_d xi[d] · Wr_a[h, d, r]              (matmul X[i,:] @ Wr_a[h])
 *   scores[j] = Σ_r u[r]   · Wr_b[h, r, j]              (matmul u @ Wr_b[h])
 *   attn[j]   = softmax(scores[0..i])
 *   out[d]    = Σ_j attn[j] · v[j, h_off+d]              (weighted sum of V)
 * ════════════════════════════════════════════════════════════════════════ */
int nt_rrpram_lowrank_attention(int wr_combined_idx, int x_idx, int v_idx,
                                 int T, int n_embd, int nr_heads, int head_dim) {
    if (wr_combined_idx < 0 || x_idx < 0 || v_idx < 0) return -1;
    int out_dim = nr_heads * head_dim;
    nt_tensor* out = nt_tensor_new(T * out_dim);
    if (!out) return -1;
    nt_tape_entry* pwr = &g_tape.entries[wr_combined_idx];
    nt_tape_entry* px  = &g_tape.entries[x_idx];
    nt_tape_entry* pv  = &g_tape.entries[v_idx];

    int T_r = T;   /* assumption */
    long combined_len = pwr->output->len;
    int rank = (int)(combined_len / ((long)nr_heads * (n_embd + T_r)));
    if (rank < 1) { nt_tensor_free(out); return -1; }
    long wra_total = (long)nr_heads * n_embd * rank;          /* offset of Wr_b section */

    int rrlr_done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_X  = nt_tensor_ensure_gpu(px->output);
        float* d_Wr = nt_tensor_ensure_gpu(pwr->output);
        float* d_V  = nt_tensor_ensure_gpu(pv->output);
        float* d_O  = nt_tensor_ensure_gpu(out);
        /* Slot map (re-using free slots beyond MH/CE backward use):
         *   slot 1: forward-only, used by mh_attn forward — rrpram_lr never coexists.
         *           Reuse slot 1 for d_scores [H, T, T] of rrpram.
         *   slot 12: rrpram U buffer [H, T, R] — persisted to backward via tape.
         *   slot 13/14: rrpram backward d_attn / d_score scratch [H, T, T].
         * NOTE: forward U/scores must live in DEVICE buffers persisted across
         * forward→backward boundary. tape_clear frees activation tensor d_data.
         * Approach: cudaMalloc per-call into nt_tape entry's grad ptr is dirty.
         * Cleaner: alloc dedicated GPU scratch and snapshot it into a dedicated
         * tape slot. For now: backward will RECOMPUTE U and scores on GPU since
         * they are O(T·R·H) + O(T·T·H) ≈ 8·512·512 = 2M floats — cheap recompute. */
        int n_h = nr_heads;
        float* d_U      = gpu_scratch(12, n_h * T * rank);
        float* d_scores = gpu_scratch(1,  n_h * T * T);
        if (d_X && d_Wr && d_V && d_O && d_U && d_scores) {
            gpu_rrpram_lr_forward(d_X, d_Wr, d_V, d_O, d_U, d_scores,
                                  T, n_embd, n_h, rank, head_dim);
            nt_tensor_mark_gpu_fresh(out);
            int idx = nt_tape_record4(out, NT_OP_RRPRAM_LR, wr_combined_idx, x_idx, v_idx,
                                      (float)T, (float)n_embd, (float)nr_heads, (float)head_dim);
            nt_tensor_free(out);
            return idx;
        }
    }
    /* CPU fallback — ensure inputs synced. */
    nt_tensor_ensure_cpu(pwr->output);
    nt_tensor_ensure_cpu(px->output);
    nt_tensor_ensure_cpu(pv->output);
#endif
    (void)rrlr_done_gpu;

    float* u_buf      = (float*)malloc(rank * sizeof(float));
    float* scores_buf = (float*)malloc(T_r  * sizeof(float));
    if (!u_buf || !scores_buf) { free(u_buf); free(scores_buf); nt_tensor_free(out); return -1; }

    for (int h = 0; h < nr_heads; h++) {
        long wr_a_base = (long)h * n_embd * rank;             /* Wr_a[h] */
        long wr_b_base = wra_total + (long)h * rank * T_r;     /* Wr_b[h] inside same buffer */
        int  v_off     = h * head_dim;
        for (int i = 0; i < T; i++) {
            float* xi = px->output->data + i * n_embd;
            /* u[r] = Σ_d xi[d] · Wr_a[h, d, r] */
            for (int r = 0; r < rank; r++) u_buf[r] = 0.0f;
            for (int d = 0; d < n_embd; d++) {
                float xd = xi[d];
                const float* wa_row = pwr->output->data + wr_a_base + (long)d * rank;
                for (int r = 0; r < rank; r++) u_buf[r] += xd * wa_row[r];
            }
            /* scores[j] = Σ_r u[r] · Wr_b[h, r, j] for j ≤ i */
            float mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float s = 0.0f;
                for (int r = 0; r < rank; r++) {
                    s += u_buf[r] * pwr->output->data[wr_b_base + (long)r * T_r + j];
                }
                scores_buf[j] = s;
                if (s > mx) mx = s;
            }
            /* softmax */
            float sm = 0.0f;
            for (int j = 0; j <= i; j++) { scores_buf[j] = expf(scores_buf[j] - mx); sm += scores_buf[j]; }
            if (sm > 0.0f) for (int j = 0; j <= i; j++) scores_buf[j] /= sm;
            /* out[d] = Σ_j attn[j] · v[j, h_off+d] */
            float* oi = out->data + i * out_dim + v_off;
            for (int d = 0; d < head_dim; d++) oi[d] = 0.0f;
            for (int j = 0; j <= i; j++) {
                const float* vj = pv->output->data + j * out_dim + v_off;
                for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
            }
        }
    }
    free(u_buf); free(scores_buf);

    int idx = nt_tape_record4(out, NT_OP_RRPRAM_LR, wr_combined_idx, x_idx, v_idx,
                              (float)T, (float)n_embd, (float)nr_heads, (float)head_dim);
    nt_tensor_free(out);
    return idx;
}

int nt_concat(int a_idx, int b_idx, int T) {
    if (a_idx < 0 || b_idx < 0) return -1;
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    nt_tape_entry* pb = &g_tape.entries[b_idx];
    int Da = pa->output->len / T;
    int Db = pb->output->len / T;
    int Dc = Da + Db;
    nt_tensor* out = nt_tensor_new(T * Dc);
    if (!out) return -1;
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < Da; d++) out->data[t * Dc + d] = pa->output->data[t * Da + d];
        for (int d = 0; d < Db; d++) out->data[t * Dc + Da + d] = pb->output->data[t * Db + d];
    }
    int idx = nt_tape_record(out, NT_OP_CONCAT, a_idx, b_idx, (float)T);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SWIGLU — y = SiLU(gate) * up (element-wise, pre-computed tensors)
// ═══════════════════════════════════════════════════════════════════════════════
int nt_swiglu(int gate_idx, int up_idx) {
    if (gate_idx < 0 || up_idx < 0) return -1;
    nt_tape_entry* pg = &g_tape.entries[gate_idx];
    nt_tape_entry* pu = &g_tape.entries[up_idx];
    int n = pg->output->len;
    if (pu->output->len != n) return -1;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    if (pg->output->ndim > 0)
        nt_tensor_reshape(out, pg->output->shape, pg->output->ndim);

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_G = nt_tensor_ensure_gpu(pg->output);
        float* d_U = nt_tensor_ensure_gpu(pu->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_G && d_U && d_Y) {
            gpu_swiglu(d_Y, d_G, d_U, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        for (int i = 0; i < n; i++) {
            float g = pg->output->data[i];
            float s = 1.0f / (1.0f + expf(-g));
            out->data[i] = (g * s) * pu->output->data[i];  // silu(g) * u
        }
    }
    int idx = nt_tape_record(out, NT_OP_SWIGLU, gate_idx, up_idx, 0);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BITLINEAR — BitNet b1.58 (ternary W, int8 x, STE backward)
// Forward: Wq = clamp(round(W/γ_W), -1, +1), γ_W = mean|W|
//          xq = clamp(round(x * 127/γ_x), -128, +127), γ_x = max|x|
//          y = (γ_W γ_x / 127) × (Wq @ xq)
// Backward: STE — treats quant as identity, dW = dout ⊗ x, dx = W^T @ dout (full-precision W)
// ═══════════════════════════════════════════════════════════════════════════════
static inline float nt_bit_absmean(const float* w, int n) {
    if (n <= 0) return 1.0f;
    float s = 0; for (int i = 0; i < n; i++) s += fabsf(w[i]);
    float g = s / n;
    return g > 1e-8f ? g : 1e-8f;
}

static inline signed char nt_bit_ternary(float w, float inv_gamma) {
    int q = (int)lrintf(w * inv_gamma);
    if (q > 1) q = 1; else if (q < -1) q = -1;
    return (signed char)q;
}

static inline float nt_bit_int8_absmax(const float* x, int n) {
    float xmax = 0;
    for (int j = 0; j < n; j++) { float v = fabsf(x[j]); if (v > xmax) xmax = v; }
    return xmax > 1e-8f ? xmax : 1e-8f;
}

int nt_bit_linear(int w_idx, int x_idx) {
    if (w_idx < 0 || x_idx < 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int rows = pw->output->shape[0];
    int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
    if (rows <= 0 || cols <= 0) return -1;
    nt_tensor* out = nt_tensor_new(rows);
    if (!out) return -1;

    float gamma_w = nt_bit_absmean(pw->output->data, rows * cols);
    float inv_gw = 1.0f / gamma_w;
    float gamma_x = nt_bit_int8_absmax(px->output->data, cols);
    float inv_sx = 127.0f / gamma_x;
    float output_scale = gamma_w * gamma_x / 127.0f;

    signed char* x_q = (signed char*)calloc(cols, sizeof(signed char));
    if (!x_q) { nt_tensor_free(out); return -1; }
    for (int j = 0; j < cols; j++) {
        int q = (int)lrintf(px->output->data[j] * inv_sx);
        if (q > 127) q = 127; else if (q < -128) q = -128;
        x_q[j] = (signed char)q;
    }

    const float* W = pw->output->data;
    for (int i = 0; i < rows; i++) {
        long long acc = 0;
        const float* W_row = W + i * cols;
        for (int j = 0; j < cols; j++)
            acc += (long long)nt_bit_ternary(W_row[j], inv_gw) * x_q[j];
        out->data[i] = output_scale * (float)acc;
    }
    free(x_q);

    int idx = nt_tape_record(out, NT_OP_BIT_LINEAR, w_idx, x_idx, gamma_w);
    nt_tensor_free(out);
    return idx;
}

int nt_bit_seq_linear(int w_idx, int x_idx, int T) {
    if (w_idx < 0 || x_idx < 0 || T <= 0) return -1;
    nt_tape_entry* pw = &g_tape.entries[w_idx];
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int rows = pw->output->shape[0];
    int cols = pw->output->ndim >= 2 ? pw->output->shape[1] : pw->output->len / rows;
    if (rows <= 0 || cols <= 0) return -1;

    nt_tensor* out = nt_tensor_new(T * rows);
    if (!out) return -1;

    float gamma_w = nt_bit_absmean(pw->output->data, rows * cols);
    float inv_gw = 1.0f / gamma_w;

    /* Pre-quantize W to ternary stored as FLOAT (so cblas_sgemm can consume it) */
    float* Wq_f = (float*)malloc((size_t)rows * cols * sizeof(float));
    if (!Wq_f) { nt_tensor_free(out); return -1; }
    for (int i = 0; i < rows * cols; i++) {
        int q = (int)lrintf(pw->output->data[i] * inv_gw);
        if (q > 1) q = 1; else if (q < -1) q = -1;
        Wq_f[i] = (float)q;
    }

    /* Pre-quantize full X per-position to int8-range FLOAT, store per-position scale */
    float* Xq_f = (float*)malloc((size_t)T * cols * sizeof(float));
    float* gamma_x_per_t = (float*)malloc(T * sizeof(float));
    if (!Xq_f || !gamma_x_per_t) {
        free(Wq_f); free(Xq_f); free(gamma_x_per_t); nt_tensor_free(out); return -1;
    }
    for (int t = 0; t < T; t++) {
        const float* x_row = px->output->data + t * cols;
        float gamma_x = nt_bit_int8_absmax(x_row, cols);
        gamma_x_per_t[t] = gamma_x;
        float inv_sx = 127.0f / gamma_x;
        float* xq_row = Xq_f + t * cols;
        for (int j = 0; j < cols; j++) {
            float q = lrintf(x_row[j] * inv_sx);
            if (q > 127.0f) q = 127.0f; else if (q < -128.0f) q = -128.0f;
            xq_row[j] = q;
        }
    }

#ifdef USE_BLAS
    /* Single BLAS matmul: Y[T,rows] = Xq[T,cols] @ Wq^T[cols,rows]
     * Wq stored row-major as [rows, cols] so CblasTrans gives Wq^T. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, rows, cols,
                1.0f, Xq_f, cols, Wq_f, cols,
                0.0f, out->data, rows);
    /* Apply per-position output scale (gamma_w * gamma_x / 127) */
    float base = gamma_w / 127.0f;
    for (int t = 0; t < T; t++) {
        float s = base * gamma_x_per_t[t];
        float* y_row = out->data + t * rows;
        for (int i = 0; i < rows; i++) y_row[i] *= s;
    }
#else
    for (int t = 0; t < T; t++) {
        float output_scale = gamma_w * gamma_x_per_t[t] / 127.0f;
        const float* xq_row = Xq_f + t * cols;
        float* y_row = out->data + t * rows;
        for (int i = 0; i < rows; i++) {
            float acc = 0;
            const float* Wq_row = Wq_f + i * cols;
            for (int j = 0; j < cols; j++) acc += Wq_row[j] * xq_row[j];
            y_row[i] = output_scale * acc;
        }
    }
#endif

    free(Wq_f);
    free(Xq_f);
    free(gamma_x_per_t);

    int idx = nt_tape_record3(out, NT_OP_BIT_SEQ_LINEAR, w_idx, x_idx, -1, (float)T, gamma_w);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPA — Sentence Phonon Attention (inference-time; pure helpers, no tape)
// ═══════════════════════════════════════════════════════════════════════════════
void nt_spa_embed_sentence(const int* tokens, int n_tokens,
                           const float* W_embed, int vocab_size, int dim,
                           float alpha, float* out_emb) {
    if (!tokens || !W_embed || !out_emb || n_tokens <= 0 || dim <= 0 || vocab_size <= 0) return;
    if (alpha < 0 || alpha > 1) alpha = 0.85f;

    for (int d = 0; d < dim; d++) out_emb[d] = 0;

    float total_weight = 0;
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= vocab_size) continue;
        float w = powf(alpha, (float)(n_tokens - 1 - i));
        total_weight += w;
        const float* row = W_embed + (size_t)tok * dim;
        for (int d = 0; d < dim; d++) out_emb[d] += w * row[d];
    }
    if (total_weight > 0)
        for (int d = 0; d < dim; d++) out_emb[d] /= total_weight;
}

float nt_spa_connectedness(const float* query_emb, int dim,
                           const float* sentence_embeddings, int n_sentences) {
    if (!query_emb || !sentence_embeddings || dim <= 0 || n_sentences <= 0) return 0;
    float scale = 1.0f / sqrtf((float)dim);

    float* scores = (float*)calloc(n_sentences, sizeof(float));
    if (!scores) return 0;

    float max_s = -1e30f;
    for (int i = 0; i < n_sentences; i++) {
        float s = 0;
        const float* emb = sentence_embeddings + (size_t)i * dim;
        for (int d = 0; d < dim; d++) s += query_emb[d] * emb[d];
        s *= scale;
        scores[i] = s;
        if (s > max_s) max_s = s;
    }
    float sum = 0;
    for (int i = 0; i < n_sentences; i++) { scores[i] = expf(scores[i] - max_s); sum += scores[i]; }
    float max_attn = 0;
    if (sum > 0) {
        for (int i = 0; i < n_sentences; i++) {
            float w = scores[i] / sum;
            if (w > max_attn) max_attn = w;
        }
    }
    free(scores);
    return max_attn;
}

void nt_spa_modulate_logits(float* logits, int V, float connectedness, float strength) {
    if (!logits || V <= 0) return;
    if (connectedness < 0) connectedness = 0;
    if (connectedness > 1) connectedness = 1;
    float spa_temp = 1.0f - strength * connectedness;
    if (spa_temp < 1e-3f) spa_temp = 1e-3f;
    float inv = 1.0f / spa_temp;
    for (int i = 0; i < V; i++) logits[i] *= inv;
}

int nt_cross_entropy(int logits_idx, int target) {
    if (logits_idx < 0) return -1;
    nt_tape_entry* pl = &g_tape.entries[logits_idx];
    int n = pl->output->len;
    if (target < 0 || target >= n) return -1;
    float mx = pl->output->data[0];
    for (int i = 1; i < n; i++) if (pl->output->data[i] > mx) mx = pl->output->data[i];
    float sum = 0;
    for (int i = 0; i < n; i++) sum += expf(pl->output->data[i] - mx);
    float log_sm = pl->output->data[target] - mx - logf(sum);
    nt_tensor* out = nt_tensor_new(1);
    if (!out) return -1;
    out->data[0] = -log_sm;
    int idx = nt_tape_record(out, NT_OP_CROSS_ENT, logits_idx, -1, (float)target);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_cross_entropy(int logits_idx, int targets_idx, int T, int V) {
    if (logits_idx < 0 || targets_idx < 0) return -1;
    nt_tape_entry* pl = &g_tape.entries[logits_idx];
    nt_tape_entry* pt = &g_tape.entries[targets_idx];
    nt_tensor* out = nt_tensor_new(1);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_L = nt_tensor_ensure_gpu(pl->output);
        float* d_T = nt_tensor_ensure_gpu(pt->output);
        /* per-position losses scratch. gpu_cross_entropy reads it back to
         * compute the mean — the value is a host float. */
        float* d_losses = gpu_scratch(2, T);
        if (d_L && d_T && d_losses) {
            float mean = gpu_cross_entropy(d_L, d_T, d_losses, T, V);
            out->data[0] = mean;
            /* loss is a 1-element CPU value — mark GPU mirror invalid in case
             * something later tries to consume it on GPU. */
            out->gpu_valid = 0;
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
        float total_loss = 0;
        for (int t = 0; t < T; t++) {
            float* logits_t = pl->output->data + t * V;
            int target = (int)pt->output->data[t];
            if (target < 0 || target >= V) target = 0;
            float mx = logits_t[0];
            for (int j = 1; j < V; j++) if (logits_t[j] > mx) mx = logits_t[j];
            float sum = 0;
            for (int j = 0; j < V; j++) sum += expf(logits_t[j] - mx);
            total_loss += -(logits_t[target] - mx - logf(sum));
        }
        out->data[0] = total_loss / T;
    }
    int idx = nt_tape_record3(out, NT_OP_SEQ_CROSSENT, logits_idx, targets_idx, -1, (float)T, (float)V);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_cross_entropy_masked(int logits_idx, int targets_idx, int mask_idx, int T, int V) {
    if (logits_idx < 0 || targets_idx < 0 || mask_idx < 0) return -1;
    nt_tape_entry* pl = &g_tape.entries[logits_idx];
    nt_tape_entry* pt = &g_tape.entries[targets_idx];
    nt_tape_entry* pm = &g_tape.entries[mask_idx];
#ifdef USE_CUDA
    /* CPU op — pull GPU mirrors back. Without these calls, when callers
     * use GPU mode, logits arrive GPU-fresh / CPU-stale (zeros) →
     * softmax(zeros) = uniform → loss = ln(V) every step regardless of
     * input. Caught during heart.c phase 1 cal: loss exactly 9.7041 =
     * ln(16384) on Resonance until this fix landed. */
    nt_tensor_ensure_cpu(pl->output);
    nt_tensor_ensure_cpu(pt->output);
    nt_tensor_ensure_cpu(pm->output);
#endif
    nt_tensor* out = nt_tensor_new(1);
    if (!out) return -1;
    float total_loss = 0;
    float n_active = 0;
    for (int t = 0; t < T; t++) {
        float m = pm->output->data[t];
        if (m == 0.0f) continue;
        float* logits_t = pl->output->data + t * V;
        int target = (int)pt->output->data[t];
        if (target < 0 || target >= V) target = 0;
        float mx = logits_t[0];
        for (int j = 1; j < V; j++) if (logits_t[j] > mx) mx = logits_t[j];
        float sum = 0;
        for (int j = 0; j < V; j++) sum += expf(logits_t[j] - mx);
        total_loss += m * -(logits_t[target] - mx - logf(sum));
        n_active += m;
    }
    out->data[0] = (n_active > 0) ? total_loss / n_active : 0.0f;
    int idx = nt_tape_record3(out, NT_OP_SEQ_CROSSENT_MASKED, logits_idx, targets_idx, mask_idx, (float)T, (float)V);
    nt_tensor_free(out);
    return idx;
}

int nt_add(int a_idx, int b_idx) {
    if (a_idx < 0 || b_idx < 0) return -1;
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    nt_tape_entry* pb = &g_tape.entries[b_idx];
    int n = pa->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    /* GPU add requires equal-length operands (no broadcast). Skip when
     * shapes mismatch — fall back to CPU broadcast loop. */
    if (g_use_gpu && pb->output->len == n) {
        float* d_A = nt_tensor_ensure_gpu(pa->output);
        float* d_B = nt_tensor_ensure_gpu(pb->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_A && d_B && d_Y) {
            gpu_add(d_Y, d_A, d_B, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(pa->output);
        nt_tensor_ensure_cpu(pb->output);
#endif
        for (int i = 0; i < n; i++)
            out->data[i] = pa->output->data[i] + pb->output->data[i % pb->output->len];
    }
    int idx = nt_tape_record(out, NT_OP_ADD, a_idx, b_idx, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_mul(int a_idx, int b_idx) {
    if (a_idx < 0 || b_idx < 0) return -1;
    nt_tape_entry* pa = &g_tape.entries[a_idx];
    nt_tape_entry* pb = &g_tape.entries[b_idx];
    int n = pa->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu && pb->output->len == n) {
        float* d_A = nt_tensor_ensure_gpu(pa->output);
        float* d_B = nt_tensor_ensure_gpu(pb->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_A && d_B && d_Y) {
            gpu_mul(d_Y, d_A, d_B, n);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(pa->output);
        nt_tensor_ensure_cpu(pb->output);
#endif
        for (int i = 0; i < n; i++)
            out->data[i] = pa->output->data[i] * pb->output->data[i % pb->output->len];
    }
    int idx = nt_tape_record(out, NT_OP_MUL, a_idx, b_idx, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_scale(int x_idx, float s) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_Y) {
            gpu_scale(d_Y, d_X, n, s);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
#endif
    if (!done_gpu) {
#ifdef USE_CUDA
        nt_tensor_ensure_cpu(px->output);
#endif
        for (int i = 0; i < n; i++) out->data[i] = px->output->data[i] * s;
    }
    int idx = nt_tape_record(out, NT_OP_SCALE, x_idx, -1, s);
    nt_tensor_free(out);
    return idx;
}

int nt_rope_freq(int x_idx, int T, int head_dim, float freq_base) {
    if (x_idx < 0 || T <= 0 || head_dim <= 0) return -1;
    if (freq_base <= 0.0f) freq_base = 10000.0f;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int total = px->output->len;
    int D = total / T;
    int n_heads = D / head_dim;
    if (n_heads <= 0) return -1;

    nt_tensor* out = nt_tensor_new(total);
    if (!out) return -1;
    if (px->output->ndim > 0) nt_tensor_reshape(out, px->output->shape, px->output->ndim);

    int done_gpu = 0;
#ifdef USE_CUDA
    if (g_use_gpu) {
        float* d_X = nt_tensor_ensure_gpu(px->output);
        float* d_Y = nt_tensor_ensure_gpu(out);
        if (d_X && d_Y) {
            gpu_rope_forward(d_Y, d_X, T, D, n_heads, head_dim, freq_base);
            nt_tensor_mark_gpu_fresh(out);
            done_gpu = 1;
        }
    }
    if (!done_gpu) nt_tensor_ensure_cpu(px->output);
#endif

    if (!done_gpu) {
        memcpy(out->data, px->output->data, total * sizeof(float));
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                int base = t * D + h * head_dim;
                for (int i = 0; i < head_dim / 2; i++) {
                    float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
                    float angle = t * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float x0 = out->data[base + 2 * i];
                    float x1 = out->data[base + 2 * i + 1];
                    out->data[base + 2 * i] = x0 * cos_a - x1 * sin_a;
                    out->data[base + 2 * i + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
    }

    int idx = nt_tape_record4(out, NT_OP_ROPE, x_idx, -1, -1, (float)T, (float)head_dim, freq_base, 0.0f);
    nt_tensor_free(out);
    return idx;
}

int nt_rope(int x_idx, int T, int head_dim) {
    return nt_rope_freq(x_idx, T, head_dim, 10000.0f);
}

int nt_rope_split_half_freq(int x_idx, int T, int head_dim, float freq_base) {
    /* Split-half RoPE: pairs (i, i+head_dim/2) instead of even/odd
     * (2i, 2i+1). Used by nanochat / Janus v4 (infer_v4.c:35-49).
     * Sign convention matches canonical Janus rope_pos:
     *   q[i]      =  q0*cos + q1*sin
     *   q[i+half] = -q0*sin + q1*cos
     * (notorch's even/odd nt_rope_freq uses the inverse rotation.)
     * CPU-only forward; dispatches via NT_OP_ROPE with aux4=1.0
     * — backward case branches on aux4 for split-half formulas. */
    if (x_idx < 0 || T <= 0 || head_dim <= 0) return -1;
    if (freq_base <= 0.0f) freq_base = 10000.0f;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int total = px->output->len;
    int D = total / T;
    int n_heads = D / head_dim;
    int half = head_dim / 2;
    if (n_heads <= 0 || half <= 0) return -1;

    nt_tensor* out = nt_tensor_new(total);
    if (!out) return -1;
    if (px->output->ndim > 0) nt_tensor_reshape(out, px->output->shape, px->output->ndim);

#ifdef USE_CUDA
    nt_tensor_ensure_cpu(px->output);
#endif
    memcpy(out->data, px->output->data, total * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < n_heads; h++) {
            int base = t * D + h * head_dim;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
                float angle = t * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x0 = out->data[base + i];
                float x1 = out->data[base + half + i];
                out->data[base + i]        =  x0 * cos_a + x1 * sin_a;
                out->data[base + half + i] = -x0 * sin_a + x1 * cos_a;
            }
        }
    }
#ifdef USE_CUDA
    out->cpu_dirty = 0; out->gpu_valid = 0;
#endif
    int idx = nt_tape_record4(out, NT_OP_ROPE, x_idx, -1, -1,
                              (float)T, (float)head_dim, freq_base, 1.0f /* split-half */);
    nt_tensor_free(out);
    return idx;
}

int nt_dropout(int x_idx, float p) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    if (g_training_mode && p > 0.0f && p < 1.0f) {
        float scale = 1.0f / (1.0f - p);  // inverted dropout
        for (int i = 0; i < n; i++) {
            float r = rand_uniform();
            out->data[i] = (r >= p) ? px->output->data[i] * scale : 0.0f;
        }
    } else {
        memcpy(out->data, px->output->data, n * sizeof(float));
    }

    // Store the dropout mask in output for backward (mask encoded as: 0 = dropped, scale = kept)
    int idx = nt_tape_record(out, NT_OP_DROPOUT, x_idx, -1, p);
    nt_tensor_free(out);
    return idx;
}

int nt_gelu(int x_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;
    for (int i = 0; i < n; i++) {
        float x = px->output->data[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    int idx = nt_tape_record(out, NT_OP_GELU, x_idx, -1, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_layernorm(int x_idx, int gamma_idx, int beta_idx) {
    if (x_idx < 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    int n = px->output->len;
    nt_tensor* out = nt_tensor_new(n);
    if (!out) return -1;

    // Compute mean and variance
    float mean = 0;
    for (int i = 0; i < n; i++) mean += px->output->data[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) {
        float d = px->output->data[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    for (int i = 0; i < n; i++)
        out->data[i] = (px->output->data[i] - mean) * inv_std;

    // Apply affine: gamma * normalized + beta
    if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
        nt_tape_entry* pg = &g_tape.entries[gamma_idx];
        for (int i = 0; i < n && i < pg->output->len; i++)
            out->data[i] *= pg->output->data[i];
    }
    if (beta_idx >= 0 && beta_idx < g_tape.count) {
        nt_tape_entry* pb = &g_tape.entries[beta_idx];
        for (int i = 0; i < n && i < pb->output->len; i++)
            out->data[i] += pb->output->data[i];
    }

    int g_idx = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int b_idx = (beta_idx >= 0 && beta_idx < g_tape.count) ? beta_idx : -1;
    int idx = nt_tape_record3(out, NT_OP_LAYERNORM, x_idx, g_idx, b_idx, 0, 0);
    nt_tensor_free(out);
    return idx;
}

int nt_seq_layernorm(int x_idx, int gamma_idx, int beta_idx, int T, int D) {
    if (x_idx < 0 || T <= 0 || D <= 0) return -1;
    nt_tape_entry* px = &g_tape.entries[x_idx];
    nt_tensor* out = nt_tensor_new(T * D);
    if (!out) return -1;

    for (int t = 0; t < T; t++) {
        float* x_t = px->output->data + t * D;
        float* o_t = out->data + t * D;
        float mean = 0;
        for (int d = 0; d < D; d++) mean += x_t[d];
        mean /= D;
        float var = 0;
        for (int d = 0; d < D; d++) { float dd = x_t[d] - mean; var += dd * dd; }
        var /= D;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < D; d++) o_t[d] = (x_t[d] - mean) * inv_std;
    }

    if (gamma_idx >= 0 && gamma_idx < g_tape.count) {
        nt_tape_entry* pg = &g_tape.entries[gamma_idx];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D && d < pg->output->len; d++)
                out->data[t * D + d] *= pg->output->data[d];
    }
    if (beta_idx >= 0 && beta_idx < g_tape.count) {
        nt_tape_entry* pb = &g_tape.entries[beta_idx];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D && d < pb->output->len; d++)
                out->data[t * D + d] += pb->output->data[d];
    }

    int g_idx = (gamma_idx >= 0 && gamma_idx < g_tape.count) ? gamma_idx : -1;
    int b_idx = (beta_idx >= 0 && beta_idx < g_tape.count) ? beta_idx : -1;
    int idx = nt_tape_record3(out, NT_OP_SEQ_LAYERNORM, x_idx, g_idx, b_idx, (float)T, (float)D);
    nt_tensor_free(out);
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BPE TOKENIZER
// ═══════════════════════════════════════════════════════════════════════════════

static void bpe_build_decode_table(nt_bpe* bpe) {
    for (int i = 0; i < 256; i++) {
        bpe->tokens[i][0] = (unsigned char)i;
        bpe->token_len[i] = 1;
    }
    for (int m = 0; m < bpe->n_merges; m++) {
        int new_id = 256 + m;
        int a = bpe->merges[m][0];
        int b = bpe->merges[m][1];
        int la = bpe->token_len[a];
        int lb = bpe->token_len[b];
        if (la + lb < NT_BPE_MAX_TOKEN_LEN) {
            memcpy(bpe->tokens[new_id], bpe->tokens[a], la);
            memcpy(bpe->tokens[new_id] + la, bpe->tokens[b], lb);
            bpe->token_len[new_id] = la + lb;
        }
    }
}

void nt_bpe_init(nt_bpe* bpe, const int merges[][2], int n_merges) {
    memset(bpe, 0, sizeof(nt_bpe));
    bpe->n_merges = n_merges;
    bpe->vocab_size = 256 + n_merges;
    for (int i = 0; i < n_merges; i++) {
        bpe->merges[i][0] = merges[i][0];
        bpe->merges[i][1] = merges[i][1];
    }
    bpe_build_decode_table(bpe);
}

int nt_bpe_load(nt_bpe* bpe, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    memset(bpe, 0, sizeof(nt_bpe));
    int a, b, n = 0;
    while (fscanf(f, "%d %d", &a, &b) == 2 && n < NT_BPE_MAX_MERGES) {
        bpe->merges[n][0] = a;
        bpe->merges[n][1] = b;
        n++;
    }
    fclose(f);
    bpe->n_merges = n;
    bpe->vocab_size = 256 + n;
    bpe_build_decode_table(bpe);
    return n;
}

int nt_bpe_encode(const nt_bpe* bpe, const char* text, int text_len, int* out, int max_tokens) {
    if (!text || text_len <= 0 || !out || max_tokens <= 0) return 0;
    int n = 0;
    for (int i = 0; i < text_len && n < max_tokens; i++)
        out[n++] = (unsigned char)text[i];
    /* Two-pointer write — O(n) per merge instead of O(n²).
     * Old shift-on-match was catastrophic on multi-MB corpora. */
    for (int m = 0; m < bpe->n_merges; m++) {
        int a = bpe->merges[m][0];
        int b = bpe->merges[m][1];
        int new_id = 256 + m;
        int w = 0, r = 0;
        while (r < n) {
            if (r + 1 < n && out[r] == a && out[r + 1] == b) {
                out[w++] = new_id;
                r += 2;
            } else {
                out[w++] = out[r++];
            }
        }
        n = w;
    }
    return n;
}

int nt_bpe_decode(const nt_bpe* bpe, const int* tokens, int n_tokens, char* out, int max_bytes) {
    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        int id = tokens[i];
        if (id < 0 || id >= bpe->vocab_size) continue;
        int len = bpe->token_len[id];
        if (pos + len >= max_bytes) break;
        memcpy(out + pos, bpe->tokens[id], len);
        pos += len;
    }
    out[pos] = '\0';
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATALOADER
// ═══════════════════════════════════════════════════════════════════════════════

nt_dataloader* nt_dataloader_create(const char* text_file, nt_bpe* bpe,
                                     int seq_len, int batch_size) {
    if (!text_file || !bpe || seq_len <= 0 || batch_size <= 0) return NULL;

    // Read entire file
    FILE* f = fopen(text_file, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* text = (char*)malloc(fsize + 1);
    if (!text) { fclose(f); return NULL; }
    fread(text, 1, fsize, f);
    text[fsize] = 0;
    fclose(f);

    // Tokenize
    int* tokens = (int*)malloc(fsize * sizeof(int)); // worst case: 1 token per char
    if (!tokens) { free(text); return NULL; }
    int n_tokens = nt_bpe_encode(bpe, text, (int)fsize, tokens, (int)fsize);
    free(text);

    if (n_tokens < seq_len + 1) { free(tokens); return NULL; }

    // Shrink tokens array
    int* shrunk = (int*)realloc(tokens, n_tokens * sizeof(int));
    if (shrunk) tokens = shrunk;

    nt_dataloader* dl = (nt_dataloader*)calloc(1, sizeof(nt_dataloader));
    if (!dl) { free(tokens); return NULL; }
    dl->tokens = tokens;
    dl->n_tokens = n_tokens;
    dl->seq_len = seq_len;
    dl->batch_size = batch_size;
    dl->n_batches = (n_tokens - 1) / (seq_len * batch_size);
    if (dl->n_batches <= 0) dl->n_batches = 1;

    // Create shuffle indices
    dl->shuffle_indices = (int*)malloc(dl->n_batches * sizeof(int));
    for (int i = 0; i < dl->n_batches; i++) dl->shuffle_indices[i] = i;

    return dl;
}

nt_dataloader* nt_dataloader_from_tokens(const char* token_file,
                                          int seq_len, int batch_size) {
    if (!token_file || seq_len <= 0 || batch_size <= 0) return NULL;
    FILE* f = fopen(token_file, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n_tokens = (int)(fsize / sizeof(int));
    if (n_tokens < seq_len + 1) { fclose(f); return NULL; }
    int* tokens = (int*)malloc(n_tokens * sizeof(int));
    if (!tokens) { fclose(f); return NULL; }
    fread(tokens, sizeof(int), n_tokens, f);
    fclose(f);

    nt_dataloader* dl = (nt_dataloader*)calloc(1, sizeof(nt_dataloader));
    if (!dl) { free(tokens); return NULL; }
    dl->tokens = tokens;
    dl->n_tokens = n_tokens;
    dl->seq_len = seq_len;
    dl->batch_size = batch_size;
    dl->n_batches = (n_tokens - 1) / (seq_len * batch_size);
    if (dl->n_batches <= 0) dl->n_batches = 1;
    dl->shuffle_indices = (int*)malloc(dl->n_batches * sizeof(int));
    for (int i = 0; i < dl->n_batches; i++) dl->shuffle_indices[i] = i;
    return dl;
}

int nt_dataloader_next(nt_dataloader* dl, int* input, int* target) {
    if (!dl || !input || !target) return -1;
    if (dl->batch_idx >= dl->n_batches) {
        dl->epoch++;
        dl->batch_idx = 0;
        nt_dataloader_shuffle(dl);
        return -1;
    }

    int batch_start = dl->shuffle_indices[dl->batch_idx] * dl->seq_len * dl->batch_size;
    for (int b = 0; b < dl->batch_size; b++) {
        int offset = batch_start + b * dl->seq_len;
        for (int s = 0; s < dl->seq_len; s++) {
            int pos = offset + s;
            if (pos + 1 >= dl->n_tokens) pos = dl->n_tokens - 2;
            input[b * dl->seq_len + s] = dl->tokens[pos];
            target[b * dl->seq_len + s] = dl->tokens[pos + 1];
        }
    }
    dl->batch_idx++;
    return 0;
}

void nt_dataloader_reset(nt_dataloader* dl) {
    if (!dl) return;
    dl->batch_idx = 0;
    dl->pos = 0;
}

void nt_dataloader_shuffle(nt_dataloader* dl) {
    if (!dl || !dl->shuffle_indices) return;
    for (int i = dl->n_batches - 1; i > 0; i--) {
        int j = xorshift32() % (i + 1);
        int tmp = dl->shuffle_indices[i];
        dl->shuffle_indices[i] = dl->shuffle_indices[j];
        dl->shuffle_indices[j] = tmp;
    }
}

void nt_dataloader_free(nt_dataloader* dl) {
    if (!dl) return;
    free(dl->tokens);
    free(dl->shuffle_indices);
    free(dl);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAVE / LOAD
// ═══════════════════════════════════════════════════════════════════════════════

#define NT_MAGIC 0x4E544F52  // "NTOR"

int nt_save(const char* path, nt_tensor** params, int n_params) {
    if (!path || !params || n_params <= 0) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = NT_MAGIC;
    int32_t n = n_params;
    fwrite(&magic, 4, 1, f);
    fwrite(&n, 4, 1, f);
    for (int i = 0; i < n_params; i++) {
        nt_tensor* t = params[i];
        int32_t ndim = t->ndim;
        fwrite(&ndim, 4, 1, f);
        for (int d = 0; d < ndim; d++) {
            int32_t s = t->shape[d];
            fwrite(&s, 4, 1, f);
        }
        fwrite(t->data, sizeof(float), t->len, f);
    }
    fclose(f);
    return 0;
}

nt_tensor** nt_load(const char* path, int* n_params) {
    if (!path || !n_params) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic;
    int32_t n;
    fread(&magic, 4, 1, f);
    if (magic != NT_MAGIC) { fclose(f); return NULL; }
    fread(&n, 4, 1, f);
    if (n <= 0 || n > NT_TAPE_MAX_PARAMS) { fclose(f); return NULL; }

    nt_tensor** params = (nt_tensor**)calloc(n, sizeof(nt_tensor*));
    if (!params) { fclose(f); return NULL; }

    for (int i = 0; i < n; i++) {
        int32_t ndim;
        fread(&ndim, 4, 1, f);
        int shape[NT_MAX_DIMS];
        for (int d = 0; d < ndim; d++) {
            int32_t s;
            fread(&s, 4, 1, f);
            shape[d] = s;
        }
        params[i] = nt_tensor_new_shape(shape, ndim);
        if (!params[i]) { fclose(f); *n_params = i; return params; }
        fread(params[i]->data, sizeof(float), params[i]->len, f);
    }
    fclose(f);
    *n_params = n;
    return params;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEBBIAN MICROLEARNING
// ═══════════════════════════════════════════════════════════════════════════════

void nt_hebbian_step(float* A, float* B, int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal,
                     float lr, float decay) {
    if (!A || !B || !x || !dy) return;
    // A: [in_dim × rank], B: [rank × out_dim]
    // Hebbian: A += lr * signal * x ⊗ (B^T @ dy), B += lr * signal * (A^T @ x) ⊗ dy
    float* proj = (float*)calloc(rank, sizeof(float));
    if (!proj) return;

    // proj = B^T @ dy (rank vector)
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rank, out_dim,
                1.0f, B, out_dim, dy, 1, 0.0f, proj, 1);
#else
    for (int r = 0; r < rank; r++) {
        float s = 0;
        for (int j = 0; j < out_dim; j++) s += B[r * out_dim + j] * dy[j];
        proj[r] = s;
    }
#endif

    // A update: A[i*rank+r] += lr * signal * x[i] * proj[r]
    float alpha = lr * signal;
#ifdef USE_BLAS
    cblas_sger(CblasRowMajor, in_dim, rank,
               alpha, x, 1, proj, 1, A, rank);
#else
    for (int i = 0; i < in_dim; i++)
        for (int r = 0; r < rank; r++)
            A[i * rank + r] += alpha * x[i] * proj[r];
#endif

    // proj2 = A^T @ x (rank vector)
    float* proj2 = (float*)calloc(rank, sizeof(float));
    if (proj2) {
#ifdef USE_BLAS
        cblas_sgemv(CblasRowMajor, CblasTrans, in_dim, rank,
                    1.0f, A, rank, x, 1, 0.0f, proj2, 1);
#else
        for (int r = 0; r < rank; r++) {
            float s = 0;
            for (int i = 0; i < in_dim; i++) s += A[i * rank + r] * x[i];
            proj2[r] = s;
        }
#endif
        // B update: B[r*out_dim+j] += lr * signal * proj2[r] * dy[j]
#ifdef USE_BLAS
        cblas_sger(CblasRowMajor, rank, out_dim,
                   alpha, proj2, 1, dy, 1, B, out_dim);
#else
        for (int r = 0; r < rank; r++)
            for (int j = 0; j < out_dim; j++)
                B[r * out_dim + j] += alpha * proj2[r] * dy[j];
#endif
        free(proj2);
    }

    // Weight decay
    if (decay > 0.0f && decay < 1.0f) {
        for (int i = 0; i < in_dim * rank; i++) A[i] *= decay;
        for (int i = 0; i < rank * out_dim; i++) B[i] *= decay;
    }
    free(proj);
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

long nt_count_params(nt_tensor** params, int n) {
    long total = 0;
    for (int i = 0; i < n; i++)
        if (params[i]) total += params[i]->len;
    return total;
}

void nt_print_params(nt_tensor** params, int n, const char** names) {
    long total = 0;
    for (int i = 0; i < n; i++) {
        if (!params[i]) continue;
        const char* name = (names && names[i]) ? names[i] : "param";
        nt_tensor_print(params[i], name);
        total += params[i]->len;
    }
    printf("Total: %ld parameters (%.2f MB)\n", total, (float)total * 4.0f / 1048576.0f);
}

/* BPE implementation is above, near dataloader */

// ═══════════════════════════════════════════════════════════════════════════════
// BLAS — direct matmul API for inference engines
// ═══════════════════════════════════════════════════════════════════════════════

void nt_blas_mmT(float *C, const float *A, const float *BT, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, BT, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * BT[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

void nt_blas_mm(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
#endif
}

void nt_blas_matvec(float *out, const float *W, const float *x, int m, int n) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                m, n, 1.0f, W, n, x, 1, 0.0f, out, 1);
#else
    for (int i = 0; i < m; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) s += W[i*n + j] * x[j];
        out[i] = s;
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOTORCH LoRA — low-rank adapter implementation
// ═══════════════════════════════════════════════════════════════════════════════
//
// Forward:  y = W @ x + (alpha/rank) * B @ (A @ x)
// A: [rank, in_dim]  → kaiming_uniform_(fan_in=in_dim) init
// B: [out_dim, rank] → zeros init  (so initial Δ output = 0)
// W is supplied externally (via existing tape entry), frozen via nt_tape_param_frozen().

#define NT_LORA_MAGIC   0x4C4F5241u  // 'LORA'
#define NT_LORA_VERSION 1u

int nt_lora_init(nt_lora_pair* lora, int in_dim, int out_dim, int rank, float alpha) {
    if (!lora || in_dim <= 0 || out_dim <= 0 || rank <= 0) return -1;
    int a_shape[2] = { rank, in_dim };
    int b_shape[2] = { out_dim, rank };
    lora->A = nt_tensor_new_shape(a_shape, 2);
    lora->B = nt_tensor_new_shape(b_shape, 2);
    if (!lora->A || !lora->B) {
        if (lora->A) nt_tensor_free(lora->A);
        if (lora->B) nt_tensor_free(lora->B);
        lora->A = NULL; lora->B = NULL;
        return -1;
    }
    nt_kaiming_uniform_init(lora->A, in_dim);
    // B already zero from nt_tensor_new_shape (calloc-backed), but be explicit:
    for (int i = 0; i < lora->B->len; i++) lora->B->data[i] = 0.0f;
    lora->rank    = rank;
    lora->alpha   = alpha;
    lora->scaling = alpha / (float)rank;
    lora->in_dim  = in_dim;
    lora->out_dim = out_dim;
    return 0;
}

void nt_lora_free(nt_lora_pair* lora) {
    if (!lora) return;
    if (lora->A) { nt_tensor_free(lora->A); lora->A = NULL; }
    if (lora->B) { nt_tensor_free(lora->B); lora->B = NULL; }
}

int nt_lora_forward(int w_idx, nt_lora_pair* lora, int x_idx, int T) {
    if (!lora || !lora->A || !lora->B) return -1;
    if (w_idx < 0 || x_idx < 0) return -1;
    // Register persistent A,B as trainable in this step's tape — Chuck slot
    // allocation for them happens here, AFTER any base nt_tape_param_frozen()
    // calls, so Chuck slot indices stay clean for the optimizer.
    int a_idx = nt_tape_param(lora->A);
    int b_idx = nt_tape_param(lora->B);
    if (a_idx < 0 || b_idx < 0) return -1;

    // Compose y = nt_seq_linear(w, x, T) + scaling * nt_seq_linear(b, nt_seq_linear(a, x, T), T)
    int wx_idx = nt_seq_linear(w_idx, x_idx, T);     // base: W @ x
    if (wx_idx < 0) return -1;
    int ax_idx = nt_seq_linear(a_idx, x_idx, T);     // A @ x  → [T, rank]
    if (ax_idx < 0) return -1;
    int bax_idx = nt_seq_linear(b_idx, ax_idx, T);   // B @ (A @ x)  → [T, out_dim]
    if (bax_idx < 0) return -1;
    int scaled_idx = nt_scale(bax_idx, lora->scaling);
    if (scaled_idx < 0) return -1;
    int y_idx = nt_add(wx_idx, scaled_idx);
    return y_idx;
}

// ── LoRA file I/O ──
//
// Format (little-endian, packed):
//   [u32 magic 0x4C4F5241 'LORA']
//   [u32 version=1]
//   [u32 num_targets]
//   [for each target T in [0,num_targets): u8 namelen, namelen × ascii bytes]
//   [u32 num_layers][u32 rank][u32 alpha_int (= (uint32_t)(alpha*1000))]
//   [u32 in_dim][u32 out_dim]
//   [for each layer L in [0,num_layers), for each target T in [0,num_targets):
//       A floats (rank × in_dim), B floats (out_dim × rank)]

int nt_lora_save(const nt_lora_pair* pairs, int num_layers, int num_targets,
                 const char* const* target_names, const char* path) {
    if (!pairs || !target_names || !path || num_layers <= 0 || num_targets <= 0) return -1;

    // Validate ALL pairs FIRST (per Codex notorch-pass-1 P2 #2 — fopen 'wb' truncates,
    // so checking dimensions before opening keeps any pre-existing checkpoint safe).
    int rank = pairs[0].rank;
    int in_dim = pairs[0].in_dim;
    int out_dim = pairs[0].out_dim;
    float alpha = pairs[0].alpha;
    for (int i = 0; i < num_layers * num_targets; i++) {
        if (pairs[i].rank != rank || pairs[i].in_dim != in_dim ||
            pairs[i].out_dim != out_dim || pairs[i].alpha != alpha) {
            return -1;  // fail before touching the destination file
        }
        if (!pairs[i].A || !pairs[i].B) return -1;
    }

    // Write to a temp file first, then atomically rename — guards against partial
    // writes leaving a corrupt checkpoint if the process is killed mid-save.
    char tmp_path[2048];
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (n < 0 || n >= (int)sizeof(tmp_path)) return -1;
    FILE* f = fopen(tmp_path, "wb");
    if (!f) return -1;

    uint32_t magic = NT_LORA_MAGIC, version = NT_LORA_VERSION;
    uint32_t nt_targets = (uint32_t)num_targets;
    uint32_t nt_layers = (uint32_t)num_layers;
    uint32_t nt_rank = (uint32_t)rank;
    // Store alpha as raw float bytes (per Codex notorch-pass-1 P3 #4 — int-milli
    // is lossy for non-rounded alpha values like 16.5 / 13.7 / etc).
    union { float f; uint32_t u; } alpha_bits = { .f = alpha };
    uint32_t nt_alpha_bits = alpha_bits.u;
    uint32_t nt_in = (uint32_t)in_dim;
    uint32_t nt_out = (uint32_t)out_dim;

    if (fwrite(&magic, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&version, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_targets, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }

    for (int t = 0; t < num_targets; t++) {
        const char* name = target_names[t];
        size_t nl = name ? strlen(name) : 0;
        if (nl > 255) nl = 255;
        uint8_t bnl = (uint8_t)nl;
        if (fwrite(&bnl, 1, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
        if (nl > 0 && fwrite(name, 1, nl, f) != nl) { fclose(f); remove(tmp_path); return -1; }
    }

    if (fwrite(&nt_layers, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_rank, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_alpha_bits, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_in, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }
    if (fwrite(&nt_out, 4, 1, f) != 1) { fclose(f); remove(tmp_path); return -1; }

    int a_n = rank * in_dim;
    int b_n = out_dim * rank;
    for (int L = 0; L < num_layers; L++) {
        for (int T = 0; T < num_targets; T++) {
            const nt_lora_pair* p = &pairs[L * num_targets + T];
            // Sync GPU mirror to CPU before reading host buffer (per Codex Pass 3 P1 #5)
#ifdef USE_CUDA
            nt_tensor_ensure_cpu(p->A);
            nt_tensor_ensure_cpu(p->B);
#endif
            if ((int)fwrite(p->A->data, sizeof(float), a_n, f) != a_n) { fclose(f); remove(tmp_path); return -1; }
            if ((int)fwrite(p->B->data, sizeof(float), b_n, f) != b_n) { fclose(f); remove(tmp_path); return -1; }
        }
    }
    if (fflush(f) != 0) { fclose(f); remove(tmp_path); return -1; }
    fclose(f);
    // Atomic rename: only on Unix; on Windows rename(2) fails if dest exists.
    // For our pod-side flow this is Linux/macOS, so rename(2) is atomic.
    if (rename(tmp_path, path) != 0) { remove(tmp_path); return -1; }
    return 0;
}

int nt_lora_load(nt_lora_pair* pairs, int num_layers, int num_targets,
                 const char* const* target_names, const char* path) {
    if (!pairs || !target_names || !path || num_layers <= 0 || num_targets <= 0) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic, version, nt_targets, nt_layers, nt_rank, nt_in, nt_out;
    if (fread(&magic, 4, 1, f) != 1 || magic != NT_LORA_MAGIC) { fclose(f); return -1; }
    if (fread(&version, 4, 1, f) != 1 || version != NT_LORA_VERSION) { fclose(f); return -1; }
    if (fread(&nt_targets, 4, 1, f) != 1 || (int)nt_targets != num_targets) { fclose(f); return -1; }

    char buf[256];
    for (int t = 0; t < num_targets; t++) {
        uint8_t bnl;
        if (fread(&bnl, 1, 1, f) != 1) { fclose(f); return -1; }
        if (bnl > 0 && fread(buf, 1, bnl, f) != bnl) { fclose(f); return -1; }
        buf[bnl] = '\0';
        if (target_names[t] && strcmp(buf, target_names[t]) != 0) { fclose(f); return -1; }
    }

    if (fread(&nt_layers, 4, 1, f) != 1 || (int)nt_layers != num_layers) { fclose(f); return -1; }
    if (fread(&nt_rank, 4, 1, f) != 1) { fclose(f); return -1; }
    uint32_t nt_alpha_bits;
    if (fread(&nt_alpha_bits, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&nt_in, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&nt_out, 4, 1, f) != 1) { fclose(f); return -1; }

    int rank = (int)nt_rank, in_dim = (int)nt_in, out_dim = (int)nt_out;
    union { uint32_t u; float f; } alpha_bits = { .u = nt_alpha_bits };
    float alpha = alpha_bits.f;
    // Compare alpha with tolerance — float exact equality is brittle even when
    // load round-trips raw bytes (compiler / fp env may diverge).
    for (int i = 0; i < num_layers * num_targets; i++) {
        if (pairs[i].rank != rank || pairs[i].in_dim != in_dim ||
            pairs[i].out_dim != out_dim) {
            fclose(f); return -1;
        }
        float diff = pairs[i].alpha - alpha;
        if (diff < 0) diff = -diff;
        if (diff > 1e-4f) { fclose(f); return -1; }
    }

    int a_n = rank * in_dim, b_n = out_dim * rank;
    for (int L = 0; L < num_layers; L++) {
        for (int T = 0; T < num_targets; T++) {
            nt_lora_pair* p = &pairs[L * num_targets + T];
            if ((int)fread(p->A->data, sizeof(float), a_n, f) != a_n) { fclose(f); return -1; }
            if ((int)fread(p->B->data, sizeof(float), b_n, f) != b_n) { fclose(f); return -1; }
#ifdef USE_CUDA
            // Mark CPU mirror authoritative; next nt_tensor_ensure_gpu uploads.
            p->A->gpu_valid = 0;
            p->B->gpu_valid = 0;
#endif
        }
    }
    fclose(f);
    return 0;
}

void nt_lora_merge_into(float* W_dst, const float* W_frozen,
                        const nt_lora_pair* lora, int in_dim, int out_dim) {
    if (!W_dst || !W_frozen || !lora || !lora->A || !lora->B) return;
    if (in_dim <= 0 || out_dim <= 0) return;
    if (lora->in_dim != in_dim || lora->out_dim != out_dim) return;
#ifdef USE_CUDA
    nt_tensor_ensure_cpu(lora->A);
    nt_tensor_ensure_cpu(lora->B);
#endif
    int rank = lora->rank;
    float scale = lora->scaling;
    // W_dst[i,j] = W_frozen[i,j] + scale * sum_k B[i,k] * A[k,j]
    // Compute Δ = B @ A first (out × in), then add to W_frozen → W_dst.
    // Use existing nt_blas_mm: C[m,n] = A[m,k] @ B[k,n].
    float* delta = (float*)malloc((size_t)out_dim * in_dim * sizeof(float));
    if (!delta) return;
    nt_blas_mm(delta, lora->B->data, lora->A->data, out_dim, rank, in_dim);
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            W_dst[i * in_dim + j] = W_frozen[i * in_dim + j] + scale * delta[i * in_dim + j];
        }
    }
    free(delta);
}

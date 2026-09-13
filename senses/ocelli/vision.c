/* vision.c — SmolVLM / idefics3 image preprocessing + SigLIP vision tower. See vision.h. */
#include "vision.h"
#include "notorch_vision.h"   /* nt_image*, stb_image impl lives in THIS TU only */
#include "gguf.h"             /* mmproj weight loading */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#define TILE     512   /* per-tile / global square size (= vision image_size) */
#define LONGEST 2048   /* outer cap on longest edge before tiling */

static void img_free(nt_image* im) { if (im) { free(im->data); free(im); } }

/* Copy [r0:r0+TILE, c0:c0+TILE] sub-rect of a CHW [0,1] image into dst[3*TILE*TILE].
 * Out-of-bounds (image not a perfect multiple) is zero-padded. */
static void crop_tile(const nt_image* src, int r0, int c0, float* dst) {
    int W = src->width, H = src->height;
    for (int ch = 0; ch < 3; ch++)
        for (int r = 0; r < TILE; r++)
            for (int c = 0; c < TILE; c++) {
                int sr = r0 + r, sc = c0 + c;
                float v = (sr < H && sc < W)
                        ? src->data[(long)ch * H * W + (long)sr * W + sc] : 0.0f;
                dst[(long)ch * TILE * TILE + (long)r * TILE + c] = v;
            }
}

/* In-place normalize a [3,TILE,TILE] frame: (x-0.5)/0.5  (x already in [0,1]). */
static void norm_frame(float* f) {
    long n = 3L * TILE * TILE;
    for (long i = 0; i < n; i++) f[i] = (f[i] - 0.5f) / 0.5f;
}

float* smolvlm_preprocess_grid(const char* path, int* out_n_frames, int* out_S,
                               int* out_rows, int* out_cols) {
    nt_image* img = nt_image_load(path, 3);          /* CHW float in [0,1] */
    if (!img) return NULL;

    /* idefics3 splitting, geometry taken from the reference runtime rather than
     * from the HF config: scale the LONGEST edge to 2048 — up or down, this is
     * not a cap — then cut a ceil(W/512) x ceil(H/512) grid of 512 tiles and
     * append one global frame. Verified against llama-mtmd-cli slice counts:
     * 1024x1024 -> 4x4+1 = 17, 640x448 -> 4x3+1 = 13, 896x727 -> 4x4+1 = 17.
     * SMOLVLM_NOSPLIT=1 falls back to a single global frame (64 tokens): far
     * cheaper, far blinder — small text is lost at that resolution. */
    int W = img->width, H = img->height;
    {
        float s = (float)LONGEST / (W > H ? W : H);
        int nw = (int)(W * s + 0.5f), nh = (int)(H * s + 0.5f);
        if (nw < 1) nw = 1;
        if (nh < 1) nh = 1;
        if (nw != W || nh != H) {
            nt_image* r = nt_image_resize(img, nw, nh);
            img_free(img);
            if (!r) return NULL;
            img = r; W = img->width; H = img->height;
        }
    }

    int n_cols = (W + TILE - 1) / TILE;
    int n_rows = (H + TILE - 1) / TILE;
    int n_tiles = n_rows * n_cols;
    int splitting = (getenv("SMOLVLM_NOSPLIT") == NULL) && (n_tiles > 1);
    int n_frames = splitting ? n_tiles + 1 : 1;      /* tiles + one global, or just global */

    float* out = (float*)malloc((long)n_frames * 3 * TILE * TILE * sizeof(float));
    if (!out) { img_free(img); return NULL; }

    int fi = 0;
    if (splitting) {
        /* resize so the image divides evenly into n_rows x n_cols tiles of TILE */
        nt_image* grid = nt_image_resize(img, n_cols * TILE, n_rows * TILE);
        if (!grid) { free(out); img_free(img); return NULL; }
        for (int rr = 0; rr < n_rows; rr++)
            for (int cc = 0; cc < n_cols; cc++)
                crop_tile(grid, rr * TILE, cc * TILE, out + (long)(fi++) * 3 * TILE * TILE);
        img_free(grid);
    }

    /* global image: full (capped) frame resized to TILE x TILE, appended last */
    nt_image* g = nt_image_resize(img, TILE, TILE);
    if (!g) { free(out); img_free(img); return NULL; }
    memcpy(out + (long)fi * 3 * TILE * TILE, g->data, (long)3 * TILE * TILE * sizeof(float));
    fi++;
    img_free(g);
    img_free(img);

    for (int i = 0; i < n_frames; i++) norm_frame(out + (long)i * 3 * TILE * TILE);

    *out_n_frames = n_frames;
    *out_S = TILE;
    if (out_rows) *out_rows = splitting ? n_rows : 1;
    if (out_cols) *out_cols = splitting ? n_cols : 1;
    return out;
}

float* smolvlm_preprocess(const char* path, int* out_n_frames, int* out_S) {
    return smolvlm_preprocess_grid(path, out_n_frames, out_S, NULL, NULL);
}

/* ─────────────────────────────────────────────────────────────────────────
 * SigLIP vision tower (PHASE 3). Forward verified vs llama.cpp clip.cpp + HF.
 * Raw-pointer f32 (notorch tape ops don't fit a forward loop); BLAS for matmuls.
 * ───────────────────────────────────────────────────────────────────────── */

/* y[m,n] = x[m,k] @ W[n,k]^T   (W is GGUF [out=n, in=k] row-major; PyTorch Linear) */
static void mmT(float* C, const float* A, const float* W, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, A, k, W, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
        float s = 0; for (int p = 0; p < k; p++) s += A[(long)i*k+p] * W[(long)j*k+p]; C[(long)i*n+j] = s; }
#endif
}
/* C[m,n] = A[m,k] @ B[k,n] */
static void mm(float* C, const float* A, const float* B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
        float s = 0; for (int p = 0; p < k; p++) s += A[(long)i*k+p] * B[(long)p*n+j]; C[(long)i*n+j] = s; }
#endif
}

/* ── f16 weights (half RAM): a matmul weight is kept as f16 (exact GGUF value) or
 *    f32. Lazy-dequant f16 -> a reused f32 scratch right before the portable cblas.
 *    gguf_f16_to_f32_n is bit-identical to gguf_dequant's F16 path, so the f32 view
 *    equals the previous gd() weight exactly -> vision output is unchanged. ── */
typedef struct { const uint16_t *f16; const float *f32; } wt;   /* exactly one set */

static wt load_wt(gguf_file *gf, const char *name) {
    wt w = {NULL, NULL};
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return w;
    w.f16 = gguf_load_f16(gf, ti);          /* non-NULL iff tensor is F16 */
    if (!w.f16) w.f32 = gguf_dequant(gf, ti);   /* genuinely-f32 weights stay f32 */
    return w;
}
static int  wt_ok(wt w)   { return w.f16 || w.f32; }
static void wt_free(wt w) { free((void*)w.f16); free((void*)w.f32); }

static float *g_vscratch = NULL; static long g_vscap = 0;
static const float *materialize(wt w, long n) {     /* -> f32 view (reused scratch if f16) */
    if (w.f32) return w.f32;
    if (!w.f16) return NULL;
    if (n > g_vscap) { free(g_vscratch); g_vscratch = (float*)malloc(n * sizeof(float));
                       g_vscap = g_vscratch ? n : 0; }   /* cap only on success -> retry works */
    if (!g_vscratch) return NULL;                        /* OOM: no NULL write */
    gguf_f16_to_f32_n(w.f16, g_vscratch, n);
    return g_vscratch;
}
/* C[m,n] = A[m,k] @ W[n,k]^T with a wt weight (lazy-dequant), via mmT. */
static void mmT_w(float* C, const float* A, wt W, int m, int k, int n) {
    mmT(C, A, materialize(W, (long)n * k), m, k, n);
}

static void add_bias_rows(float* x, const float* b, int m, int n) {
    if (!b) return;
    for (int i = 0; i < m; i++) { float* r = x + (long)i*n; for (int j = 0; j < n; j++) r[j] += b[j]; }
}
/* LayerNorm (mean/var) per row, eps; ggml NORM_TYPE_NORMAL == HF nn.LayerNorm */
static void layernorm_rows(float* x, const float* g, const float* b, int m, int n, float eps) {
    for (int i = 0; i < m; i++) {
        float* r = x + (long)i*n;
        double mu = 0; for (int j = 0; j < n; j++) mu += r[j]; mu /= n;
        double var = 0; for (int j = 0; j < n; j++) { double d = r[j]-mu; var += d*d; } var /= n;
        float inv = 1.0f / sqrtf((float)var + eps);
        for (int j = 0; j < n; j++) r[j] = ((r[j]-(float)mu)*inv)*g[j] + b[j];
    }
}
/* gelu_pytorch_tanh, matching ggml's GGML_GELU_FP16 path bit-for-bit: clamp on
 * f32 input, else round input to f16, exact tanh-gelu, round result to f16.
 * (llama.cpp clip uses this f16-LUT; exact-f32 gelu diverged ~1e-3 -> greedy flips.) */
static void gelu_tanh_inplace(float* x, long n) {
    const float c = 0.79788456080286535588f, a = 0.044715f;
    for (long i = 0; i < n; i++) {
        float v = x[i];
        if (v <= -10.0f) { x[i] = 0.0f; continue; }   /* ggml clamp */
        if (v >=  10.0f) { x[i] = v;    continue; }    /* ggml clamp (gelu(x)~x) */
        float xr = (float)(__fp16)v;                   /* LUT index: input rounded to f16 */
        float g  = 0.5f*xr*(1.0f + tanhf(c*(xr + a*xr*xr*xr)));
        x[i] = (float)(__fp16)g;                       /* LUT stores result as f16 */
    }
}
static void softmax_row(float* x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

struct siglip_model {
    int n_layers, hidden, n_heads, head_dim, ffn, patches, patch, img;
    int scale, text_dim;           /* pixel-shuffle scale (4), projector out dim (576) */
    float eps;
    float *patch_w, *patch_b;      /* [hidden, 3*patch*patch], [hidden] */
    float *pos_w;                  /* [patches, hidden] */
    float *post_ln_w, *post_ln_b;  /* [hidden] */
    wt fc_w;                       /* mm.model.fc.weight [text_dim, hidden*scale^2] f16 (no bias) */
    struct {
        float *ln1_w,*ln1_b,*ln2_w,*ln2_b;
        wt q_w, k_w, v_w, o_w;     /* attn matmul weights (f16 in GGUF) */
        float *q_b,*k_b,*v_b,*o_b;
        wt fc1_w, fc2_w;           /* ffn matmul weights (f16 in GGUF), keyed by ROLE */
        float *fc1_b,*fc2_b;
    } *L;
};

static float* gd(gguf_file* gf, const char* name) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return NULL;
    return gguf_dequant(gf, ti);
}
/* element count of a tensor as recorded in the file, or -1 if absent */
static long tensor_nelem(gguf_file* gf, const char* name) {
    int ti = gguf_find_tensor(gf, name);
    return ti < 0 ? -1 : (long)gf->tensors[ti].n_elements;
}
static int kv_u32(gguf_file* gf, const char* key, int def) {
    const gguf_kv* kv = gguf_get_kv(gf, key);
    return kv ? (int)kv->val.u32 : def;
}
static float kv_f32(gguf_file* gf, const char* key, float def) {
    const gguf_kv* kv = gguf_get_kv(gf, key);
    return kv ? kv->val.f32 : def;
}

siglip_model* siglip_load(const char* mmproj_path) {
    gguf_file* gf = gguf_open(mmproj_path);
    if (!gf) return NULL;
    siglip_model* m = (siglip_model*)calloc(1, sizeof(*m));
    if (!m) { gguf_close(gf); return NULL; }

    m->hidden   = kv_u32(gf, "clip.vision.embedding_length", 768);
    m->n_layers = kv_u32(gf, "clip.vision.block_count", 12);
    m->n_heads  = kv_u32(gf, "clip.vision.attention.head_count", 12);
    m->ffn      = kv_u32(gf, "clip.vision.feed_forward_length", 3072);
    m->img      = kv_u32(gf, "clip.vision.image_size", 512);
    m->patch    = kv_u32(gf, "clip.vision.patch_size", 16);
    m->eps      = kv_f32(gf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    m->head_dim = m->hidden / m->n_heads;
    m->scale    = kv_u32(gf, "clip.vision.projector.scale_factor", 4);
    m->text_dim = kv_u32(gf, "clip.vision.projection_dim", 576);
    int grid    = m->img / m->patch;
    m->patches  = grid * grid;

    printf("siglip: L=%d D=%d heads=%d hd=%d ffn=%d img=%d patch=%d patches=%d eps=%.0e\n",
           m->n_layers, m->hidden, m->n_heads, m->head_dim, m->ffn, m->img, m->patch, m->patches, m->eps);

    m->patch_w = gd(gf, "v.patch_embd.weight");
    m->patch_b = gd(gf, "v.patch_embd.bias");
    m->pos_w   = gd(gf, "v.position_embd.weight");
    m->post_ln_w = gd(gf, "v.post_ln.weight");
    m->post_ln_b = gd(gf, "v.post_ln.bias");
    m->fc_w      = load_wt(gf, "mm.model.fc.weight");   /* connector projector (f16) */

    m->L = (typeof(m->L))calloc(m->n_layers, sizeof(*m->L));
    char nm[128];
    for (int l = 0; l < m->n_layers; l++) {
        #define LD(field, fmt) do { snprintf(nm,sizeof(nm),fmt,l); m->L[l].field = gd(gf, nm); } while(0)
        #define LW(field, fmt) do { snprintf(nm,sizeof(nm),fmt,l); m->L[l].field = load_wt(gf, nm); } while(0)
        LD(ln1_w,"v.blk.%d.ln1.weight"); LD(ln1_b,"v.blk.%d.ln1.bias");
        LD(ln2_w,"v.blk.%d.ln2.weight"); LD(ln2_b,"v.blk.%d.ln2.bias");
        LW(q_w,"v.blk.%d.attn_q.weight"); LD(q_b,"v.blk.%d.attn_q.bias");
        LW(k_w,"v.blk.%d.attn_k.weight"); LD(k_b,"v.blk.%d.attn_k.bias");
        LW(v_w,"v.blk.%d.attn_v.weight"); LD(v_b,"v.blk.%d.attn_v.bias");
        LW(o_w,"v.blk.%d.attn_out.weight"); LD(o_b,"v.blk.%d.attn_out.bias");
        #undef LD
        #undef LW
        /* MLP by ROLE, not by name: converters disagree about which of ffn_up /
         * ffn_down holds fc1. Ground truth is the bias length in the file —
         * fc1 is hidden->ffn (bias == ffn), fc2 is ffn->hidden (bias == hidden).
         * SmolVLM-256M's mmproj names fc1 "ffn_down"; SmolVLM2-500M's names it
         * "ffn_up" (same SigLIP weights, mirrored names). */
        snprintf(nm, sizeof(nm), "v.blk.%d.ffn_up.bias", l);
        long ub = tensor_nelem(gf, nm);
        if (ub != m->ffn && ub != m->hidden) {
            fprintf(stderr, "siglip: layer %d ffn_up.bias has %ld elements, expected %d (fc1) or %d (fc2)\n",
                    l, ub, m->ffn, m->hidden);
            gguf_close(gf); siglip_free(m); return NULL;
        }
        const char *f1 = (ub == m->ffn) ? "ffn_up" : "ffn_down";
        const char *f2 = (ub == m->ffn) ? "ffn_down" : "ffn_up";
        snprintf(nm, sizeof(nm), "v.blk.%d.%s.weight", l, f1); m->L[l].fc1_w = load_wt(gf, nm);
        snprintf(nm, sizeof(nm), "v.blk.%d.%s.bias",   l, f1); m->L[l].fc1_b = gd(gf, nm);
        snprintf(nm, sizeof(nm), "v.blk.%d.%s.weight", l, f2); m->L[l].fc2_w = load_wt(gf, nm);
        snprintf(nm, sizeof(nm), "v.blk.%d.%s.bias",   l, f2); m->L[l].fc2_b = gd(gf, nm);
    }
    gguf_close(gf);   /* dequant copied to float; raw gguf no longer needed */

    if (!m->patch_w || !m->pos_w || !m->post_ln_w || !wt_ok(m->L[0].q_w) || !wt_ok(m->fc_w) ||
        !wt_ok(m->L[0].fc1_w) || !wt_ok(m->L[0].fc2_w)) {
        fprintf(stderr, "siglip: missing critical vision weights\n");
        siglip_free(m); return NULL;
    }
    return m;
}

void siglip_free(siglip_model* m) {
    if (!m) return;
    free(m->patch_w); free(m->patch_b); free(m->pos_w); free(m->post_ln_w); free(m->post_ln_b); wt_free(m->fc_w);
    if (m->L) for (int l = 0; l < m->n_layers; l++) {
        free(m->L[l].ln1_w); free(m->L[l].ln1_b); free(m->L[l].ln2_w); free(m->L[l].ln2_b);
        wt_free(m->L[l].q_w); free(m->L[l].q_b); wt_free(m->L[l].k_w); free(m->L[l].k_b);
        wt_free(m->L[l].v_w); free(m->L[l].v_b); wt_free(m->L[l].o_w); free(m->L[l].o_b);
        wt_free(m->L[l].fc1_w); free(m->L[l].fc1_b); wt_free(m->L[l].fc2_w); free(m->L[l].fc2_b);
    }
    free(m->L); free(m);
    free(g_vscratch); g_vscratch = NULL; g_vscap = 0;   /* reusable dequant scratch */
}

int siglip_n_patches(const siglip_model* m) { return m ? m->patches : 0; }
int siglip_hidden(const siglip_model* m)    { return m ? m->hidden  : 0; }

int siglip_encode(const siglip_model* m, const float* frame, float* out) {
    int P = m->patches, D = m->hidden, H = m->n_heads, hd = m->head_dim;
    int grid = m->img / m->patch, ps = m->patch, img = m->img, in = 3*ps*ps;
    float scale = 1.0f / sqrtf((float)hd);

    /* 1) patch embed: build X[P, in] in (c,kh,kw) order, then X @ patch_w^T + bias */
    float* X = (float*)malloc((long)P*in*sizeof(float));
    if (!X) return -1;
    for (int ph = 0; ph < grid; ph++) for (int pw = 0; pw < grid; pw++) {
        int p = ph*grid + pw;             /* row-major patch index (h outer, w inner) */
        float* xr = X + (long)p*in;
        int t = 0;
        for (int c = 0; c < 3; c++) for (int kh = 0; kh < ps; kh++) for (int kw = 0; kw < ps; kw++)
            xr[t++] = frame[(long)c*img*img + (long)(ph*ps+kh)*img + (pw*ps+kw)];
    }
    mmT(out, X, m->patch_w, P, in, D);     /* out = hidden [P, D] */
    free(X);
    add_bias_rows(out, m->patch_b, P, D);

    /* 2) + position embedding */
    for (long i = 0; i < (long)P*D; i++) out[i] += m->pos_w[i];

    /* scratch */
    float* tmp = (float*)malloc((long)P*D*sizeof(float));
    float* q   = (float*)malloc((long)P*D*sizeof(float));
    float* k   = (float*)malloc((long)P*D*sizeof(float));
    float* v   = (float*)malloc((long)P*D*sizeof(float));
    float* att = (float*)malloc((long)P*D*sizeof(float));
    float* proj= (float*)malloc((long)P*D*sizeof(float));
    float* up  = (float*)malloc((long)P*m->ffn*sizeof(float));
    float* qh  = (float*)malloc((long)P*hd*sizeof(float));
    float* kh_ = (float*)malloc((long)P*hd*sizeof(float));
    float* vh  = (float*)malloc((long)P*hd*sizeof(float));
    float* sc  = (float*)malloc((long)P*P*sizeof(float));
    float* oh  = (float*)malloc((long)P*hd*sizeof(float));
    if (!tmp||!q||!k||!v||!att||!proj||!up||!qh||!kh_||!vh||!sc||!oh) {
        free(tmp);free(q);free(k);free(v);free(att);free(proj);free(up);free(qh);free(kh_);free(vh);free(sc);free(oh);
        return -1;
    }

    for (int l = 0; l < m->n_layers; l++) {
        /* ── attention sublayer: x = x + Attn(LN1(x)) ── */
        memcpy(tmp, out, (long)P*D*sizeof(float));
        layernorm_rows(tmp, m->L[l].ln1_w, m->L[l].ln1_b, P, D, m->eps);
        mmT_w(q, tmp, m->L[l].q_w, P, D, D); add_bias_rows(q, m->L[l].q_b, P, D);
        mmT_w(k, tmp, m->L[l].k_w, P, D, D); add_bias_rows(k, m->L[l].k_b, P, D);
        mmT_w(v, tmp, m->L[l].v_w, P, D, D); add_bias_rows(v, m->L[l].v_b, P, D);
        for (int h = 0; h < H; h++) {
            int off = h*hd;
            for (int t = 0; t < P; t++) {
                memcpy(qh +(long)t*hd, q +(long)t*D+off, hd*sizeof(float));
                memcpy(kh_+(long)t*hd, k +(long)t*D+off, hd*sizeof(float));
                memcpy(vh +(long)t*hd, v +(long)t*D+off, hd*sizeof(float));
            }
            mmT(sc, qh, kh_, P, hd, P);                 /* scores[P,P] = qh @ kh^T */
            for (long i = 0; i < (long)P*P; i++) sc[i] *= scale;
            for (int t = 0; t < P; t++) softmax_row(sc + (long)t*P, P);   /* non-causal */
            mm(oh, sc, vh, P, P, hd);                   /* oh[P,hd] = scores @ vh */
            for (int t = 0; t < P; t++) memcpy(att +(long)t*D+off, oh +(long)t*hd, hd*sizeof(float));
        }
        mmT_w(proj, att, m->L[l].o_w, P, D, D); add_bias_rows(proj, m->L[l].o_b, P, D);
        for (long i = 0; i < (long)P*D; i++) out[i] += proj[i];

        /* ── mlp sublayer: x = x + MLP(LN2(x)) ──
         * fc1/fc2 were resolved by bias length at load time (see siglip_load),
         * so both the mirrored 256M naming and the plain 500M one land here right. */
        memcpy(tmp, out, (long)P*D*sizeof(float));
        layernorm_rows(tmp, m->L[l].ln2_w, m->L[l].ln2_b, P, D, m->eps);
        mmT_w(up, tmp, m->L[l].fc1_w, P, D, m->ffn); add_bias_rows(up, m->L[l].fc1_b, P, m->ffn);  /* fc1: D->ffn */
        gelu_tanh_inplace(up, (long)P*m->ffn);
        mmT_w(proj, up, m->L[l].fc2_w, P, m->ffn, D); add_bias_rows(proj, m->L[l].fc2_b, P, D);    /* fc2: ffn->D */
        for (long i = 0; i < (long)P*D; i++) out[i] += proj[i];
    }

    /* 3) final post-layernorm over all tokens */
    layernorm_rows(out, m->post_ln_w, m->post_ln_b, P, D, m->eps);

    free(tmp);free(q);free(k);free(v);free(att);free(proj);free(up);free(qh);free(kh_);free(vh);free(sc);free(oh);
    return 0;
}

/* ── pixel-shuffle connector (PHASE 4) ──────────────────────────────────────
 * Index map verified vs clip.cpp build_patch_merge_permute (oracle):
 *   patches row-major p = h*grid + w (w inner). scale s=4, grid 32 -> 8x8=64 tokens.
 *   output token o = hg*(grid/s) + wg     (hg outer, wg inner)
 *   channel D = hin*(s*hidden) + win*hidden + c   (slot k = hin*s+win; hin outer, win inner)
 *   pulls from patch (h = hg*s+hin, w = wg*s+win), channel c.
 * Then projector mm.model.fc [text_dim, hidden*s^2], no bias, no activation. */
int siglip_n_vis_tokens(const siglip_model* m) {
    if (!m) return 0;
    int g = (m->img / m->patch) / m->scale;
    return g * g;
}
int siglip_text_dim(const siglip_model* m) { return m ? m->text_dim : 0; }

int siglip_connect(const siglip_model* m, const float* hidden, float* out) {
    int D = m->hidden, grid = m->img / m->patch, s = m->scale;
    int g = grid / s;                 /* 8 */
    int n_vis = g * g;                /* 64 */
    int vis_dim = D * s * s;          /* 12288 */
    float* sh = (float*)malloc((long)n_vis * vis_dim * sizeof(float));
    if (!sh) return -1;
    for (int hg = 0; hg < g; hg++)
        for (int wg = 0; wg < g; wg++) {
            int o = hg * g + wg;                       /* token col: hg outer, wg inner */
            for (int hin = 0; hin < s; hin++)
                for (int win = 0; win < s; win++) {
                    int k = hin * s + win;             /* block slot: hin outer, win inner */
                    int src = (hg * s + hin) * grid + (wg * s + win);   /* patch p = h*grid+w */
                    memcpy(sh + (long)o * vis_dim + (long)k * D,
                           hidden + (long)src * D, D * sizeof(float));
                }
        }
    mmT_w(out, sh, m->fc_w, n_vis, vis_dim, m->text_dim);   /* [64, text_dim] = sh @ fc^T */
    free(sh);
    return 0;
}

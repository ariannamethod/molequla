/* ggml_bin.c — see ggml_bin.h. */

#include "ggml_bin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cursor over the blob. Every read is bounds-checked against blob_len, because a
 * truncated or foreign file otherwise walks off the end while still looking like
 * a plausible tensor table. */
typedef struct { const unsigned char *p; size_t off, len; int bad; } cur;

static int32_t rd_i32(cur *c) {
    if (c->bad || c->off + 4 > c->len) { c->bad = 1; return 0; }
    int32_t v;
    memcpy(&v, c->p + c->off, 4);
    c->off += 4;
    return v;
}

static const void *rd_bytes(cur *c, size_t n) {
    if (c->bad || n > c->len - c->off) { c->bad = 1; return NULL; }
    const void *r = c->p + c->off;
    c->off += n;
    return r;
}

void gb_f16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint16_t h = src[i];
        /* __fp16 is a native aarch64 type and gives the exact IEEE value,
         * subnormals included; the compiler emits a single fcvt. */
        __fp16 v;
        memcpy(&v, &h, sizeof(v));
        dst[i] = (float) v;
    }
}

float *gb_dequant(const gb_tensor *t) {
    if (!t) return NULL;
    float *out = (float *) malloc((size_t) t->nelem * sizeof(float));
    if (!out) return NULL;
    if (t->ttype == GGML_BIN_F32) memcpy(out, t->data, (size_t) t->nelem * sizeof(float));
    else                          gb_f16_to_f32((const uint16_t *) t->data, out, t->nelem);
    return out;
}

const gb_tensor *gb_find(const gb_file *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++)
        if (!strcmp(f->tensors[i].name, name)) return &f->tensors[i];
    return NULL;
}

const gb_tensor *gb_need(const gb_file *f, const char *name, int64_t nelem) {
    const gb_tensor *t = gb_find(f, name);
    if (!t) { fprintf(stderr, "ears: model has no tensor '%s'\n", name); return NULL; }
    if (t->nelem != nelem) {
        fprintf(stderr, "ears: tensor '%s' has %lld elements, expected %lld\n",
                name, (long long) t->nelem, (long long) nelem);
        return NULL;
    }
    return t;
}

void gb_close(gb_file *f) {
    if (!f) return;
    free(f->blob);
    free(f->tok);
    free(f->tok_len);
    free(f->tensors);
    free(f);
}

gb_file *gb_open(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "ears: cannot open model '%s'\n", path); return NULL; }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long sz = ftell(fp);
    if (sz <= 0) { fprintf(stderr, "ears: model '%s' is empty\n", path); fclose(fp); return NULL; }
    rewind(fp);

    gb_file *f = (gb_file *) calloc(1, sizeof(*f));
    if (!f) { fclose(fp); return NULL; }
    f->blob_len = (size_t) sz;
    f->blob = (unsigned char *) malloc(f->blob_len);
    if (!f->blob || fread(f->blob, 1, f->blob_len, fp) != f->blob_len) {
        fprintf(stderr, "ears: short read on '%s'\n", path);
        fclose(fp); gb_close(f); return NULL;
    }
    fclose(fp);

    cur c = { f->blob, 0, f->blob_len, 0 };

    uint32_t magic = (uint32_t) rd_i32(&c);
    if (c.bad || magic != GGML_BIN_MAGIC) {
        fprintf(stderr, "ears: '%s' is not a flat ggml model (magic 0x%08x, want 0x%08x)\n",
                path, magic, GGML_BIN_MAGIC);
        gb_close(f); return NULL;
    }

    gb_hparams *h = &f->hparams;
    h->n_vocab       = rd_i32(&c);
    h->n_audio_ctx   = rd_i32(&c);
    h->n_audio_state = rd_i32(&c);
    h->n_audio_head  = rd_i32(&c);
    h->n_audio_layer = rd_i32(&c);
    h->n_text_ctx    = rd_i32(&c);
    h->n_text_state  = rd_i32(&c);
    h->n_text_head   = rd_i32(&c);
    h->n_text_layer  = rd_i32(&c);
    h->n_mels        = rd_i32(&c);
    h->ftype         = rd_i32(&c);

    f->n_mel = rd_i32(&c);
    f->n_fft = rd_i32(&c);
    if (c.bad || f->n_mel <= 0 || f->n_fft <= 0) {
        fprintf(stderr, "ears: bad mel filter geometry in '%s'\n", path);
        gb_close(f); return NULL;
    }
    f->filters = (const float *) rd_bytes(&c, (size_t) f->n_mel * f->n_fft * sizeof(float));

    f->n_tok = rd_i32(&c);
    if (c.bad || f->n_tok <= 0 || f->n_tok > h->n_vocab) {
        fprintf(stderr, "ears: bad vocab count in '%s'\n", path);
        gb_close(f); return NULL;
    }
    f->tok     = (const char **) calloc((size_t) f->n_tok, sizeof(char *));
    f->tok_len = (int *)         calloc((size_t) f->n_tok, sizeof(int));
    if (!f->tok || !f->tok_len) { gb_close(f); return NULL; }
    for (int i = 0; i < f->n_tok; i++) {
        int len = rd_i32(&c);
        if (c.bad || len < 0) { fprintf(stderr, "ears: bad token %d in '%s'\n", i, path); gb_close(f); return NULL; }
        f->tok_len[i] = len;
        f->tok[i] = (const char *) (len ? rd_bytes(&c, (size_t) len) : (const void *) "");
    }

    /* Tensor table to EOF. Grown by doubling; tiny has 167 entries, base 245. */
    int cap = 256;
    f->tensors = (gb_tensor *) malloc((size_t) cap * sizeof(gb_tensor));
    if (!f->tensors) { gb_close(f); return NULL; }

    while (!c.bad && c.off + 12 <= c.len) {
        int ndim = rd_i32(&c);
        int nlen = rd_i32(&c);
        int tt   = rd_i32(&c);
        if (c.bad) break;
        if (ndim < 1 || ndim > 4 || nlen < 1 || nlen > 63 ||
            (tt != GGML_BIN_F32 && tt != GGML_BIN_F16)) {
            fprintf(stderr, "ears: corrupt tensor header at offset %zu in '%s'\n", c.off - 12, path);
            gb_close(f); return NULL;
        }
        if (f->n_tensors == cap) {
            cap *= 2;
            gb_tensor *g = (gb_tensor *) realloc(f->tensors, (size_t) cap * sizeof(gb_tensor));
            if (!g) { gb_close(f); return NULL; }
            f->tensors = g;
        }
        gb_tensor *t = &f->tensors[f->n_tensors];
        memset(t, 0, sizeof(*t));
        t->ndim  = ndim;
        t->ttype = tt;
        t->nelem = 1;
        for (int d = 0; d < ndim; d++) {
            t->ne[d] = rd_i32(&c);
            if (t->ne[d] <= 0) { c.bad = 1; break; }
            t->nelem *= t->ne[d];
        }
        const char *nm = (const char *) rd_bytes(&c, (size_t) nlen);
        if (c.bad) break;
        memcpy(t->name, nm, (size_t) nlen);
        t->name[nlen] = '\0';

        size_t esz = (tt == GGML_BIN_F32) ? 4 : 2;
        t->data = rd_bytes(&c, (size_t) t->nelem * esz);
        if (c.bad) break;
        f->n_tensors++;
    }

    if (c.bad) {
        fprintf(stderr, "ears: truncated model '%s' (%d tensors read)\n", path, f->n_tensors);
        gb_close(f); return NULL;
    }
    return f;
}

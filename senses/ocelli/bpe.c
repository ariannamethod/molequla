/* bpe.c — byte-level BPE (GPT-2 / Tekken style) over a GGUF tokenizer. See bpe.h. */
#include "bpe.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* ── string -> int open-addressing hashmap ─────────────────────────────────── */
typedef struct { char **keys; int *vals; int cap; int n; } smap;

static unsigned long fnv1a(const char *s) {
    unsigned long h = 1469598103934665603UL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211UL; }
    return h;
}
static void smap_init(smap *m, int cap) {
    if (cap < 8) cap = 8;
    m->cap = cap; m->n = 0;
    m->keys = (char**)calloc(cap, sizeof(char*));
    m->vals = (int*)calloc(cap, sizeof(int));
}
static void smap_put(smap *m, const char *k, int v) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], k) == 0) { m->vals[h] = v; return; }
        h = (h + 1) % m->cap;
    }
    m->keys[h] = strdup(k); m->vals[h] = v; m->n++;
}
static int smap_get(const smap *m, const char *k) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], k) == 0) return m->vals[h];
        h = (h + 1) % m->cap;
    }
    return -1;
}
static void smap_free(smap *m) {
    for (int i = 0; i < m->cap; i++) free(m->keys[i]);
    free(m->keys); free(m->vals);
}

/* ── GPT-2 bytes<->unicode ─────────────────────────────────────────────────── */
static void build_byte_table(int cp[256], int cp2byte[512]) {
    for (int i = 0; i < 512; i++) cp2byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        cp[b] = printable ? b : (256 + n);
        if (!printable) n++;
    }
    for (int b = 0; b < 256; b++) if (cp[b] < 512) cp2byte[cp[b]] = b;
}
static int utf8_enc(int cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3;
}
static int utf8_dec(const char *s, int *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c >> 5) == 0x6) { *cp = ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F); return 2; }
    if ((c >> 4) == 0xE) { *cp = ((c & 0xF) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F); return 3; }
    *cp = c; return 1;
}

/* ── tokenizer ─────────────────────────────────────────────────────────────── */
struct bpe_tokenizer {
    char **tokens; int n_tokens;   /* id -> token string */
    smap vocab;                    /* token string -> id */
    smap merges;                   /* "A B" -> rank */
    int byte_cp[256];
    int cp2byte[512];
    /* special/control tokens (token_type CONTROL=3 / USER_DEFINED=4): emitted
     * atomically by id, never byte-BPE'd — needed for chat (<|im_start|>) and
     * idefics3 image tokens (<image>, <fake_token_around_image>, ...). */
    char **special; int *special_id; int n_special;
};

bpe_tokenizer *bpe_load(const char *path) {
    int nt = 0;
    char **toks = gguf_read_str_array(path, "tokenizer.ggml.tokens", &nt);
    if (!toks || nt <= 0) return NULL;
    int nm = 0;
    char **mg = gguf_read_str_array(path, "tokenizer.ggml.merges", &nm);

    bpe_tokenizer *t = (bpe_tokenizer*)calloc(1, sizeof(*t));
    t->tokens = toks; t->n_tokens = nt;
    build_byte_table(t->byte_cp, t->cp2byte);
    smap_init(&t->vocab, nt * 2);
    for (int i = 0; i < nt; i++) if (toks[i]) smap_put(&t->vocab, toks[i], i);
    smap_init(&t->merges, (nm > 0 ? nm : 1) * 2);
    for (int i = 0; i < nm; i++) if (mg[i]) smap_put(&t->merges, mg[i], i); /* rank = line index */
    for (int i = 0; i < nm; i++) free(mg[i]);
    free(mg);

    /* special tokens: token_type CONTROL(3) / USER_DEFINED(4) -> emitted atomically */
    int ntt = 0;
    int32_t *tt = gguf_read_i32_array(path, "tokenizer.ggml.token_type", &ntt);
    if (tt) {
        int cap = 0;
        for (int i = 0; i < ntt && i < nt; i++)
            if ((tt[i] == 3 || tt[i] == 4) && toks[i] && toks[i][0]) cap++;
        if (cap > 0) {
            t->special    = (char**)calloc(cap, sizeof(char*));
            t->special_id = (int*)  calloc(cap, sizeof(int));
            if (t->special && t->special_id) {
                for (int i = 0; i < ntt && i < nt; i++)
                    if ((tt[i] == 3 || tt[i] == 4) && toks[i] && toks[i][0]) {
                        t->special[t->n_special]    = toks[i];   /* borrowed; freed via t->tokens */
                        t->special_id[t->n_special] = i;
                        t->n_special++;
                    }
            } else {                       /* OOM: degrade to no special-token awareness */
                free(t->special); free(t->special_id);
                t->special = NULL; t->special_id = NULL;
            }
        }
        free(tt);
    }
    return t;
}

void bpe_free(bpe_tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->n_tokens; i++) free(t->tokens[i]);
    free(t->tokens);
    free(t->special); free(t->special_id);   /* surfaces borrowed from t->tokens */
    smap_free(&t->vocab); smap_free(&t->merges);
    free(t);
}

int bpe_n_vocab(const bpe_tokenizer *t) { return t ? t->n_tokens : 0; }
int bpe_token_id(const bpe_tokenizer *t, const char *token) { return t ? smap_get(&t->vocab, token) : -1; }

/* byte-level BPE of a plain span [0,L) — no special-token awareness */
static int encode_span(const bpe_tokenizer *t, const char *text, int L, int *out, int cap) {
    int no = 0, i = 0;
    while (i < L) {
        /* pre-tok piece: [i, j) — a leading space (if any) stays with the run that follows */
        int j = i + 1;
        while (j < L && text[j] != ' ') j++;
        int nsym = j - i;
        char **sym = (char**)malloc(nsym * sizeof(char*));
        for (int b = 0; b < nsym; b++) {
            char buf[8];
            int cp = t->byte_cp[(unsigned char)text[i + b]];
            int n = utf8_enc(cp, buf); buf[n] = 0;
            sym[b] = strdup(buf);
        }
        int ns = nsym;
        while (ns > 1) {
            int best_rank = INT_MAX, bi = -1;
            char key[512];
            for (int b = 0; b < ns - 1; b++) {
                snprintf(key, sizeof(key), "%s %s", sym[b], sym[b + 1]);
                int r = smap_get(&t->merges, key);
                if (r >= 0 && r < best_rank) { best_rank = r; bi = b; }
            }
            if (bi < 0) break;
            char *merged = (char*)malloc(strlen(sym[bi]) + strlen(sym[bi + 1]) + 1);
            strcpy(merged, sym[bi]); strcat(merged, sym[bi + 1]);
            free(sym[bi]); free(sym[bi + 1]); sym[bi] = merged;
            for (int b = bi + 1; b < ns - 1; b++) sym[b] = sym[b + 1];
            ns--;
        }
        for (int b = 0; b < ns; b++) {
            int id = smap_get(&t->vocab, sym[b]);
            if (id >= 0 && no < cap) out[no++] = id;
            free(sym[b]);
        }
        free(sym);
        i = j;
    }
    return no;
}

/* longest special-token surface matching text at p, or -1; sets *slen */
static int match_special(const bpe_tokenizer *t, const char *p, int avail, int *slen) {
    int best = -1, blen = 0;
    for (int s = 0; s < t->n_special; s++) {
        int sl = (int)strlen(t->special[s]);
        if (sl > 0 && sl <= avail && sl > blen && memcmp(p, t->special[s], sl) == 0) { best = s; blen = sl; }
    }
    *slen = blen;
    return best;
}

/* Encode UTF-8 text -> ids, emitting CONTROL/USER_DEFINED tokens atomically and
 * byte-BPE'ing the plain spans between them (matches llama.cpp special handling). */
int bpe_encode(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    int no = 0, L = (int)strlen(text), i = 0;
    while (i < L && no < cap) {
        int slen = 0, sp = match_special(t, text + i, L - i, &slen);
        if (sp >= 0) {
            out[no++] = t->special_id[sp];
            i += slen;
            continue;
        }
        // plain span until the next special-token start (or end)
        int j = i;
        while (j < L) { int sl; if (match_special(t, text + j, L - j, &sl) >= 0) break; j++; }
        no += encode_span(t, text + i, j - i, out + no, cap - no);
        i = j;
    }
    return no;
}

int bpe_decode_token(const bpe_tokenizer *t, int id, char *buf, int cap) {
    if (!t || id < 0 || id >= t->n_tokens || !t->tokens[id]) return 0;
    const char *s = t->tokens[id];
    int L = (int)strlen(s), i = 0, n = 0;
    while (i < L && n < cap - 1) {
        int cp; int adv = utf8_dec(s + i, &cp); i += adv;
        int byte = (cp >= 0 && cp < 512) ? t->cp2byte[cp] : -1;
        if (byte >= 0) buf[n++] = (char)byte;
    }
    buf[n] = 0;
    return n;
}

#ifdef BPE_TEST
int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <gguf> [text]\n", argv[0]); return 1; }
    bpe_tokenizer *t = bpe_load(argv[1]);
    if (!t) { printf("bpe_load failed\n"); return 1; }
    printf("vocab=%d\n", bpe_n_vocab(t));
    const char *text = argc > 2 ? argv[2] : "Privet, mir! Hello world.";
    int ids[1024];
    int n = bpe_encode(t, text, ids, 1024);
    printf("input '%s' (%d bytes) -> %d tokens\n", text, (int)strlen(text), n);
    printf("ids:"); for (int i = 0; i < n && i < 24; i++) printf(" %d", ids[i]); printf("\n");
    char out[4096]; int on = 0;
    for (int i = 0; i < n; i++) on += bpe_decode_token(t, ids[i], out + on, (int)sizeof(out) - on);
    out[on] = 0;
    printf("decode '%s'\n", out);
    int roundtrip = (strcmp(out, text) == 0);
    int merged = (n < (int)strlen(text));
    printf("%s (roundtrip=%d merges_applied=%d)\n", (roundtrip && merged) ? "BPE_OK" : "BPE_FAIL", roundtrip, merged);
    bpe_free(t);
    return (roundtrip && merged) ? 0 : 1;
}
#endif

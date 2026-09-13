/* ears.c — the CLI, the wav reader, and greedy decoding over sliding 30 s windows.
 *
 *   ears <model.bin> <wav> [-l en|auto] [-t threads] [--no-speech-thold x]
 *                         [--tokens] [--timestamps]
 *
 * Greedy only: one decoder, no beam, no best-of, no temperature fallback. The
 * logit filtering, the timestamp-pair rule, the timestamp-vs-text logsumexp
 * comparison, the seek_delta window advance and the no-speech gate all follow
 * whisper.cpp so that the token stream can be diffed against it rather than only
 * the string.
 */

#include "ears.h"
#include "nnops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK_FRAMES 3000     /* 30 s of mel at 10 ms per frame */
#define DELTA_MIN      10

/* ── logit helpers ────────────────────────────────────────────────────────── */

static void logprobs_of(const float *logits, int n, float *logprobs) {
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float lse = 0.0f;
    for (int i = 0; i < n; i++) if (logits[i] > -INFINITY) lse += expf(logits[i] - mx);
    lse = logf(lse) + mx;
    for (int i = 0; i < n; i++) logprobs[i] = (logits[i] > -INFINITY) ? logits[i] - lse : -INFINITY;
}

static void probs_of(const float *logits, const float *logprobs, int n, float *probs) {
    for (int i = 0; i < n; i++) probs[i] = (logits[i] == -INFINITY) ? 0.0f : expf(logprobs[i]);
}

/* ── decoded sequence ─────────────────────────────────────────────────────── */

typedef struct { int id; float p, plog; int tid; } tok_out;

typedef struct {
    const ears_model *m;
    float *logits, *logprobs, *probs;
    tok_out *seq;
    int n_seq, result_len, cap;
    int seek_delta, has_ts;
} greedy;

/* The logit filters of whisper_process_logits, in the same order, for the
 * greedy / temperature-0 / suppress_nst-off / no-grammar configuration v1 runs.
 * Order matters: the timestamp-pair rule reads the state the earlier -INFINITY
 * writes leave behind. */
static void filter_logits(greedy *g, const float *raw) {
    const ears_model *m = g->m;
    const ears_vocab *v = &m->vocab;
    const int n = v->n_vocab;
    float *L = g->logits;
    memcpy(L, raw, (size_t) n * sizeof(float));

    const int is_initial = (g->n_seq == 0);

    if (is_initial) {
        L[v->eot] = -INFINITY;
        /* " " is token 220 in every Whisper vocab; resolved by lookup, not by
         * constant, so a model with a different table still suppresses the right
         * one or suppresses nothing. */
        for (int i = 0; i < v->n_tok; i++)
            if (v->tok_len[i] == 1 && v->tok[i][0] == ' ') { L[i] = -INFINITY; break; }
    }

    L[v->not_ts]     = -INFINITY;
    L[v->sot]        = -INFINITY;
    L[v->nosp]       = -INFINITY;
    L[v->solm]       = -INFINITY;
    L[v->translate]  = -INFINITY;
    L[v->transcribe] = -INFINITY;
    L[v->prev]       = -INFINITY;
    for (int i = 0; i < ears_n_langs(); i++) {
        const int t = v->sot + 1 + i;
        if (t < n) L[t] = -INFINITY;
    }

    /* timestamps come in pairs, except directly before EOT */
    {
        const int last_ts = g->n_seq > 0 && g->seq[g->n_seq - 1].id >= v->beg;
        const int penult_ts = g->n_seq < 2 || g->seq[g->n_seq - 2].id >= v->beg;
        if (last_ts) {
            if (penult_ts) for (int i = v->beg; i < n; i++) L[i] = -INFINITY;
            else           for (int i = 0; i < v->eot; i++)  L[i] = -INFINITY;
        }
    }

    /* the initial timestamp cannot exceed max_initial_ts = 1.0 s */
    if (is_initial) {
        const float precision = 30.0f / (float) m->h.n_audio_ctx;
        const int tid0 = (int) lroundf(1.0f / precision);
        for (int i = v->beg + tid0 + 1; i < n; i++) L[i] = -INFINITY;
    }

    /* timestamps must not go backwards within the window */
    if (g->has_ts) {
        const int tid0 = g->seek_delta / 2;
        for (int i = v->beg; i < v->beg + tid0 && i < n; i++) L[i] = -INFINITY;
    }

    logprobs_of(L, n, g->logprobs);

    /* if the timestamps together outweigh the best text token, force a timestamp */
    {
        float ts_logprob = -INFINITY;
        float lp_max = -INFINITY;
        for (int i = v->beg; i < n; i++) if (g->logprobs[i] > lp_max) lp_max = g->logprobs[i];
        float lse = 0.0f;
        for (int i = v->beg; i < n; i++)
            if (g->logprobs[i] > -INFINITY) lse += expf(g->logprobs[i] - lp_max);
        if (lse > 0.0f) ts_logprob = logf(lse) + lp_max;

        float text_max = -INFINITY;
        for (int i = 0; i < v->beg; i++) if (g->logprobs[i] > text_max) text_max = g->logprobs[i];

        if (ts_logprob > text_max)
            for (int i = 0; i < v->beg; i++) { L[i] = -INFINITY; g->logprobs[i] = -INFINITY; }
    }

    probs_of(L, g->logprobs, n, g->probs);
}

static tok_out sample_best(const greedy *g) {
    const ears_vocab *v = &g->m->vocab;
    const int n = v->n_vocab;
    tok_out r = { 0, 0.0f, 0.0f, -1 };

    double sum_ts = 0.0, max_ts = 0.0;
    for (int i = v->beg; i < n; i++) {
        if (g->probs[i] == -INFINITY) continue;
        sum_ts += g->probs[i];
        if (max_ts < g->probs[i]) { max_ts = g->probs[i]; r.tid = i; }
    }
    for (int i = 0; i < n; i++)
        if (r.p < g->probs[i]) { r.id = i; r.p = g->probs[i]; r.plog = g->logprobs[i]; }
    if (r.id >= v->beg) r.tid = r.id;
    return r;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

static void usage(void) {
    fprintf(stderr,
        "ears — whisper on notorch. one wav in, one transcript out.\n\n"
        "  ears <model.bin> <wav> [options]\n\n"
        "  -l, --language <en|auto>   language, or auto-detect (default en)\n"
        "  -t, --threads <n>          threads for the mel front end (default 4)\n"
        "      --no-speech-thold <x>  no-speech gate (default 0.60)\n"
        "      --tokens               print token ids alongside the text\n"
        "      --timestamps           print [t0 --> t1] per segment\n");
}

static void print_ts(char *buf, size_t cap, int cs) {
    const int ms = cs * 10;
    snprintf(buf, cap, "%02d:%02d:%02d.%03d",
             ms / 3600000, (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000);
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *wav_path = NULL, *lang = "en";
    int n_threads = 4, want_tokens = 0, want_ts = 0;
    float no_speech_thold = 0.60f;
    const float logprob_thold = -1.0f;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else if ((!strcmp(a, "-l") || !strcmp(a, "--language")) && i + 1 < argc) lang = argv[++i];
        else if ((!strcmp(a, "-t") || !strcmp(a, "--threads"))  && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(a, "--no-speech-thold") && i + 1 < argc) no_speech_thold = (float) atof(argv[++i]);
        else if (!strcmp(a, "--tokens"))     want_tokens = 1;
        else if (!strcmp(a, "--timestamps")) want_ts = 1;
        else if (a[0] == '-')                { fprintf(stderr, "ears: unknown option %s\n", a); usage(); return 2; }
        else if (!model_path)                model_path = a;
        else if (!wav_path)                  wav_path = a;
        else                                 { usage(); return 2; }
    }
    if (!model_path || !wav_path) { usage(); return 2; }
    if (n_threads < 1) n_threads = 1;

    ears_model *m = ears_model_load(model_path);
    if (!m) return 1;
    const ears_vocab *v = &m->vocab;

    float *pcm = NULL; int n_samples = 0;
    if (ears_read_wav(wav_path, &pcm, &n_samples) != 0) { ears_model_free(m); return 1; }

    ears_mel mel;
    if (ears_mel_compute(pcm, n_samples, m->f->filters, m->f->n_mel, m->f->n_fft,
                         n_threads, &mel) != 0) {
        fprintf(stderr, "ears: mel failed\n");
        free(pcm); ears_model_free(m); return 1;
    }
    free(pcm);

    int rc = 1;
    const int NC = m->h.n_audio_ctx, NS = m->h.n_audio_state;
    float *enc = (float *) malloc((size_t) NC * NS * sizeof(float));
    greedy g = { m, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0 };
    g.logits   = (float *) malloc((size_t) v->n_vocab * sizeof(float));
    g.logprobs = (float *) malloc((size_t) v->n_vocab * sizeof(float));
    g.probs    = (float *) malloc((size_t) v->n_vocab * sizeof(float));
    g.cap      = m->h.n_text_ctx;
    g.seq      = (tok_out *) malloc((size_t) g.cap * sizeof(tok_out));
    if (!enc || !g.logits || !g.logprobs || !g.probs || !g.seq) {
        fprintf(stderr, "ears: out of memory\n");
        goto cleanup;
    }

    {
    const int seek_end = mel.n_len_org;
    int seek = 0;
    int lang_id = -1;

    if (seek_end < DELTA_MIN) {
        fprintf(stderr, "ears: input is shorter than 100 ms\n");
        goto cleanup;
    }

    /* Language detection: encode the first window, run a single SOT step, and take
     * the argmax over the language token logits — the raw ones, before any filter.
     * whisper.cpp scans all 100 entries of its table even though the model carries
     * 99 language slots, so the last one lands on the translate token; this does
     * the same, so the two agree on every input rather than on most of them. */
    if (!strcmp(lang, "auto")) {
        if (ears_encode(m, &mel, 0, n_threads, enc) != 0) { fprintf(stderr, "ears: encode failed\n"); goto cleanup; }
        ears_dec_state *ds = ears_dec_begin(m, enc, n_threads);
        if (!ds) { fprintf(stderr, "ears: decoder init failed\n"); goto cleanup; }
        const int sot = v->sot;
        const float *raw = ears_decode(ds, &sot, 1, 0);
        if (!raw) { ears_dec_end(ds); fprintf(stderr, "ears: decode failed\n"); goto cleanup; }
        float best = -INFINITY;
        for (int i = 0; i < ears_n_langs(); i++) {
            const int t = v->sot + 1 + i;
            if (t < v->n_vocab && raw[t] > best) { best = raw[t]; lang_id = i; }
        }
        ears_dec_end(ds);
        fprintf(stderr, "ears: auto-detected language: %s\n", ears_lang_str(lang_id));
    } else {
        lang_id = ears_lang_id(lang);
        if (lang_id < 0) { fprintf(stderr, "ears: unknown language '%s'\n", lang); goto cleanup; }
    }

    while (seek + DELTA_MIN < seek_end) {
        if (ears_encode(m, &mel, seek, n_threads, enc) != 0) {
            fprintf(stderr, "ears: encode failed at seek %d\n", seek);
            goto cleanup;
        }
        ears_dec_state *ds = ears_dec_begin(m, enc, n_threads);
        if (!ds) { fprintf(stderr, "ears: decoder init failed\n"); goto cleanup; }

        /* The prompt is decoded in two parts so the no-speech probability can be
         * read where Whisper defines it: at the SOT position, from the unfiltered
         * logits, before the language and task tokens narrow the distribution. The
         * split costs nothing — the same three tokens go through the same cache.
         *
         * whisper.cpp reaches the same value by accident and only sometimes.
         * `whisper_decode_internal` writes a prompt's logits into slot
         * prompt.size()-1 of state->logits, but whisper_full then reads slot 0
         * (src/whisper.cpp:7295-7299). Under -l auto that slot still holds the
         * single-SOT decode language detection ran, which is exactly the value
         * wanted; under -l en nothing ever wrote it, so the gate scores a zeroed
         * buffer as a uniform distribution and never fires. Measured on
         * ambient_8s.wav with tiny: -l auto suppresses, -l en prints " [Music]",
         * same audio and same tokens. ears computes the defined quantity in both
         * cases, which agrees with the oracle wherever the oracle is defined. */
        const int prompt[3] = { v->sot, v->sot + 1 + lang_id, v->transcribe };
        const float *raw = ears_decode(ds, prompt, 1, 0);
        if (!raw) { ears_dec_end(ds); fprintf(stderr, "ears: decode failed\n"); goto cleanup; }
        logprobs_of(raw, v->n_vocab, g.logprobs);
        probs_of(raw, g.logprobs, v->n_vocab, g.probs);
        const float no_speech_prob = g.probs[v->nosp];

        raw = ears_decode(ds, prompt + 1, 2, 1);
        if (!raw) { ears_dec_end(ds); fprintf(stderr, "ears: decode failed\n"); goto cleanup; }

        g.n_seq = 0; g.result_len = 0; g.has_ts = 0;
        g.seek_delta = 100 * 30;
        filter_logits(&g, raw);

        const int n_max = m->h.n_text_ctx / 2 - 4;
        int completed = 0, failed = 0;
        for (int i = 0; i < n_max && g.n_seq < g.cap; i++) {
            tok_out t = sample_best(&g);
            g.seq[g.n_seq++] = t;

            if (t.id > v->beg) {
                const int sd_new = 2 * (t.id - v->beg);
                if (g.has_ts && g.seek_delta > sd_new && g.result_len < i) { failed = 1; break; }
                g.seek_delta = sd_new;
                g.result_len = i + 1;
                g.has_ts = 1;
            }

            if (t.id == v->eot || (g.has_ts && seek + g.seek_delta + DELTA_MIN >= seek_end)) {
                if (g.result_len == 0) {
                    if (seek + g.seek_delta + DELTA_MIN >= seek_end) g.result_len = i + 1;
                    else { failed = 1; break; }
                }
                completed = 1;
                break;
            }
            if (i == n_max - 1 && (g.result_len == 0 || g.seek_delta < 100 * 30 / 2)) { failed = 1; break; }

            raw = ears_decode(ds, &t.id, 1, 3 + i);
            if (!raw) { ears_dec_end(ds); fprintf(stderr, "ears: decode failed\n"); goto cleanup; }
            filter_logits(&g, raw);
        }
        ears_dec_end(ds);
        if (!completed && !failed) g.result_len = g.n_seq;   /* ran out of room */

        /* score the kept prefix */
        double sum_lp = 0.0;
        for (int i = 0; i < g.result_len; i++) sum_lp += g.seq[i].plog;
        const double avg_lp = g.result_len ? sum_lp / g.result_len : -INFINITY;

        /* whisper.cpp's own gate: both conditions, not either. A confident
         * hallucination out of noise has a high no-speech probability and a high
         * average logprob, and only the pair of them suppresses it. */
        const int is_no_speech = (no_speech_prob > no_speech_thold && avg_lp < logprob_thold);
        fprintf(stderr, "ears: window %d cs  tokens %d  no_speech %.4f  avg_logprob %.4f%s\n",
                seek, g.result_len, no_speech_prob, avg_lp, is_no_speech ? "  [suppressed]" : "");

        if (failed) {
            fprintf(stderr, "ears: window at %d cs failed under pure greedy (no fallback in v1)\n", seek);
            g.result_len = 0;
        }

        /* One line of ids, timestamps included, in decode order — the same stream
         * the oracle's JSON carries, so a divergence can be located at a token
         * rather than guessed at from a string diff. */
        if (want_tokens && !is_no_speech && g.result_len > 0) {
            fprintf(stderr, "ears: tokens");
            for (int i = 0; i < g.result_len; i++) fprintf(stderr, " %d", g.seq[i].id);
            fprintf(stderr, "\n");
        }

        if (!is_no_speech && g.result_len > 0) {
            char text[8192];
            size_t tn = 0;
            int t0 = seek + 2 * (g.seq[0].tid - v->beg);
            for (int i = 0; i < g.result_len; i++) {
                const int id = g.seq[i].id;
                if (id < v->eot && tn < sizeof(text) - 64)
                    tn += (size_t) ears_token_append(v, id, text + tn, sizeof(text) - 1 - tn);
                if (id > v->beg) {
                    const int t1 = seek + 2 * (g.seq[i].tid - v->beg);
                    if (tn > 0) {
                        text[tn] = '\0';
                        if (want_ts) {
                            char a[24], b[24];
                            print_ts(a, sizeof(a), t0); print_ts(b, sizeof(b), t1);
                            printf("[%s --> %s]  %s\n", a, b, text);
                        } else {
                            printf("%s", text);
                        }
                        tn = 0;
                    }
                    t0 = t1;
                    while (i + 1 < g.result_len && g.seq[i + 1].id > v->beg) {
                        i++;
                        t0 = seek + 2 * (g.seq[i].tid - v->beg);
                    }
                }
            }
            if (tn > 0) {
                text[tn] = '\0';
                const int t1 = seek + g.seek_delta;
                if (want_ts) {
                    char a[24], b[24];
                    print_ts(a, sizeof(a), t0); print_ts(b, sizeof(b), t1);
                    printf("[%s --> %s]  %s\n", a, b, text);
                } else {
                    printf("%s", text);
                }
            }
        }

        /* a window whose last two tokens are text-then-timestamp gives no usable
         * boundary; skip the whole chunk rather than re-read the same audio */
        int seek_delta = g.seek_delta;
        if (g.result_len > 1 &&
            g.seq[g.result_len - 2].id < v->beg && g.seq[g.result_len - 1].id > v->beg) {
            seek_delta = seek_end - seek < CHUNK_FRAMES ? seek_end - seek : CHUNK_FRAMES;
        }
        if (seek_delta <= 0) seek_delta = CHUNK_FRAMES;
        seek += seek_delta;
    }
    if (!want_ts) printf("\n");
    rc = 0;

cleanup:
    ears_mel_free(&mel);
    free(enc); free(g.logits); free(g.logprobs); free(g.probs); free(g.seq);
    ears_model_free(m);
    return rc;
    }
}

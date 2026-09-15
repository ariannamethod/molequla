/* test_soundscape — the gate for the sound describer.
 *
 * Seven fixtures, seven labels, one table. Each fixture is synthesised here
 * (there is no sox on this phone and no wav in this repository), written to
 * tests/out/soundscape/<name>.wav so that it can be listened to and fed to the
 * CLI by hand, read back through the organ's own wav reader and analysed. A row
 * passes when the label is the one the fixture was built to produce — which
 * also means it is not one of the other six.
 *
 * Every measured feature is printed beside its row: the thresholds in
 * soundscape.c were read off this table, and when one of them moves this is
 * where the movement has to be visible.
 *
 * Shown red by moving a threshold: SND_QUIET_DB=-80 takes the quiet room row
 * down (the run is in EARSLOG.md), SND_SPEECH_BAND=1.1 takes the speech row
 * down. The environment variables are the same ones the organ reads, so the red
 * run needs no patch.
 */

#include "../soundscape.h"
#include "../mel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SR   16000
#define SECS 8
#define N    (SR * SECS)

/* ── deterministic noise ──────────────────────────────────────────────────── */

static uint32_t rng_state = 0x1234567u;
static void  rng_seed(uint32_t s) { rng_state = s ? s : 1u; }
static float rng_uni(void) {                       /* xorshift32 -> [-1, 1) */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float) ((double) rng_state / 2147483648.0 - 1.0);
}

/* ── the fixtures ─────────────────────────────────────────────────────────── */

/* An empty room at night: the microphone's own floor and nothing else. */
static void gen_quiet(float *x) {
    rng_seed(1);
    for (int i = 0; i < N; i++) x[i] = 0.0008f * rng_uni();
}

/* One bang in eight seconds of that same room. */
static void gen_transient(float *x) {
    rng_seed(2);
    for (int i = 0; i < N; i++) x[i] = 0.0010f * rng_uni();
    const int at = 3 * SR, len = SR / 20;           /* 50 ms */
    for (int i = 0; i < len; i++)
        x[at + i] += 0.6f * expf(-6.0f * (float) i / (float) len) * rng_uni();
}

/* A fan, a ventilation shaft, rain on a roof: loud, flat, and going nowhere. */
static void gen_broadband(float *x) {
    rng_seed(3);
    for (int i = 0; i < N; i++) x[i] = 0.08f * rng_uni();
}

/* Something knocking every 350 ms — a pump, a washing machine, a wiper. */
static void gen_mechanical(float *x) {
    rng_seed(4);
    for (int i = 0; i < N; i++) x[i] = 0.002f * rng_uni();
    const int period = (int) (0.35f * SR), len = SR / 25;   /* 40 ms */
    for (int at = SR / 2; at + len < N; at += period)
        for (int i = 0; i < len; i++)
            x[at + i] += 0.35f * expf(-8.0f * (float) i / (float) len) * rng_uni();
}

/* Voices through a wall: the power in the telephone band, the envelope opening
 * and closing at a syllable rate, the formants where a throat would put them. */
static void gen_speech(float *x) {
    rng_seed(5);
    const float f1 = 400.0f, f2 = 1100.0f, f3 = 2300.0f;
    const float syl[8] = { 0.18f, 0.24f, 0.15f, 0.30f, 0.20f, 0.12f, 0.26f, 0.22f };
    int i = 0, s = 0;
    while (i < N) {
        const int on  = (int) (syl[s % 8] * SR);
        const int off = (int) (syl[(s + 3) % 8] * 0.6f * SR);
        for (int j = 0; j < on && i < N; j++, i++) {
            const float t = (float) i / SR;
            const float env = sinf(3.1415926f * (float) j / (float) on);
            x[i] = env * (0.30f * sinf(6.2831853f * f1 * t)
                        + 0.22f * sinf(6.2831853f * f2 * t)
                        + 0.14f * sinf(6.2831853f * f3 * t)
                        + 0.05f * rng_uni());
        }
        for (int j = 0; j < off && i < N; j++, i++) x[i] = 0.002f * rng_uni();
        s++;
    }
}

/* A melody, five notes, each with its harmonics, at a level that does not move:
 * pitched like a hum, but the pitch does not stay where a hum's would. */
static void gen_music(float *x) {
    rng_seed(6);
    const float note[5] = { 440.0f, 494.0f, 523.0f, 587.0f, 659.0f };
    const int len = (int) (0.8f * SR);
    for (int i = 0; i < N; i++) {
        const float t = (float) i / SR;
        const float f = note[(i / len) % 5];
        x[i] = 0.25f * sinf(6.2831853f * f * t)
             + 0.10f * sinf(6.2831853f * 2.0f * f * t)
             + 0.05f * sinf(6.2831853f * 3.0f * f * t)
             + 0.004f * rng_uni();
    }
}

/* A street: bursts of different lengths at intervals that never repeat. */
static void gen_unsteady(float *x) {
    rng_seed(7);
    for (int i = 0; i < N; i++) x[i] = 0.003f * rng_uni();
    const float gap[6] = { 0.9f, 1.7f, 0.5f, 2.3f, 1.1f, 0.7f };
    const float dur[6] = { 0.30f, 0.12f, 0.45f, 0.20f, 0.35f, 0.15f };
    float at = 0.4f;
    for (int k = 0; k < 6 && at * SR < N; k++) {
        const int start = (int) (at * SR), len = (int) (dur[k] * SR);
        const float amp = 0.10f + 0.06f * (float) (k % 3);
        for (int i = 0; i < len && start + i < N; i++)
            x[start + i] += amp * rng_uni();
        at += dur[k] + gap[k];
    }
}

/* ── wav out ──────────────────────────────────────────────────────────────── */

static void put32(unsigned char *p, uint32_t v) { p[0] = (unsigned char) v; p[1] = (unsigned char) (v >> 8); p[2] = (unsigned char) (v >> 16); p[3] = (unsigned char) (v >> 24); }
static void put16(unsigned char *p, uint16_t v) { p[0] = (unsigned char) v; p[1] = (unsigned char) (v >> 8); }

static int write_wav(const char *path, const float *x, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "test_soundscape: cannot write %s\n", path); return -1; }
    unsigned char h[44];
    memcpy(h, "RIFF", 4);      put32(h + 4, (uint32_t) (36 + 2 * n));
    memcpy(h + 8, "WAVEfmt ", 8); put32(h + 16, 16);
    put16(h + 20, 1); put16(h + 22, 1);
    put32(h + 24, SR); put32(h + 28, SR * 2);
    put16(h + 32, 2); put16(h + 34, 16);
    memcpy(h + 36, "data", 4); put32(h + 40, (uint32_t) (2 * n));
    fwrite(h, 1, 44, f);
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        const int16_t s = (int16_t) lrintf(v * 32767.0f);
        unsigned char b[2];
        put16(b, (uint16_t) s);
        fwrite(b, 1, 2, f);
    }
    fclose(f);
    return 0;
}

/* ── the table ────────────────────────────────────────────────────────────── */

typedef void (*gen_fn)(float *);

static const struct { const char *name; gen_fn gen; const char *want; } ROWS[] = {
    { "quiet",      gen_quiet,      SND_QUIET      },
    { "transient",  gen_transient,  SND_TRANSIENT  },
    { "broadband",  gen_broadband,  SND_BROADBAND  },
    { "mechanical", gen_mechanical, SND_MECHANICAL },
    { "speech",     gen_speech,     SND_SPEECH     },
    { "music",      gen_music,      SND_MUSIC      },
    { "unsteady",   gen_unsteady,   SND_UNSTEADY   },
};

int main(void) {
    const char *dir = getenv("SND_FIXTURE_DIR");
    if (!dir) dir = "tests/out/soundscape";

    float *x = (float *) malloc((size_t) N * sizeof(float));
    if (!x) return 2;

    snd_cfg cfg;
    snd_cfg_defaults(&cfg);
    snd_cfg_from_env(&cfg);

    int pass = 0, fail = 0;
    for (size_t r = 0; r < sizeof(ROWS) / sizeof(ROWS[0]); r++) {
        memset(x, 0, (size_t) N * sizeof(float));
        ROWS[r].gen(x);

        char path[512];
        snprintf(path, sizeof(path), "%s/%s.wav", dir, ROWS[r].name);
        if (write_wav(path, x, N) != 0) { fail++; continue; }

        float *pcm = NULL;
        int n = 0;
        if (ears_read_wav(path, &pcm, &n) != 0) { fail++; continue; }

        snd_features f;
        const int rc = snd_analyse(pcm, n, SR, &cfg, &f);
        free(pcm);
        if (rc != 0) { printf("FAIL %-11s could not be analysed\n", ROWS[r].name); fail++; continue; }

        const char *got = snd_label(&f, &cfg);
        if (!strcmp(got, ROWS[r].want)) { pass++; printf("ok   %-11s %s\n", ROWS[r].name, got); }
        else { fail++; printf("FAIL %-11s got \"%s\", want \"%s\"\n", ROWS[r].name, got, ROWS[r].want); }
        printf("     ");
        snd_print_features(&f, stdout);
    }
    free(x);

    printf("\n%d pass, %d fail\n", pass, fail);
    return fail ? 1 : 0;
}

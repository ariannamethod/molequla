/* soundscape.c — see soundscape.h. */

#include "soundscape.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ariannamethod/notorch.h>

#define N_FFT 512
#define HOP   160          /* 10 ms at 16 kHz: the frame rate of the envelope */
#define EPS   1e-12f
#define SND_PI 3.14159265358979323846

/* ── small numerics ───────────────────────────────────────────────────────── */

static int cmp_f(const void *a, const void *b) {
    const float x = *(const float *) a, y = *(const float *) b;
    return (x > y) - (x < y);
}

/* v is sorted ascending; q in [0,1]. */
static float pct(const float *v, int n, float q) {
    if (n <= 0) return 0.0f;
    int i = (int) (q * (float) (n - 1) + 0.5f);
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;
    return v[i];
}

static float median_of(float *scratch, const float *v, int n) {
    memcpy(scratch, v, (size_t) n * sizeof(float));
    qsort(scratch, (size_t) n, sizeof(float), cmp_f);
    return pct(scratch, n, 0.5f);
}

/* ── configuration ────────────────────────────────────────────────────────── */

void snd_cfg_defaults(snd_cfg *c) {
    /* Every number here was read off a measurement, not chosen for its looks.
     * Two measurements: the seven fixtures of tests/test_soundscape.c, which
     * print every feature beside the label they were built to produce, and the
     * six twelve-second recordings the live field left in
     * molequla-run/senses/audio on 2026-09-13/14 — five of a room at night, one
     * of speech played into the room from the phone's own speaker.
     *
     * quiet_db  -40: room tone on this microphone measures db_p90 -47.6 .. -41.5
     *                across the five night recordings; the one with speech in it
     *                reaches -12.9. Anything under -40 is a room nobody is in.
     * speech_mod 0.25: a flat modulation spectrum puts 6 Hz of the 0.5-20 Hz
     *                range inside 2-8 Hz, so 0.31 is what noise scores and the
     *                share cannot be the discriminator on its own — the speech
     *                fixture scores 0.924, the speech played into the room 0.292,
     *                room tone 0.359 .. 0.489. The band ratio is what separates
     *                them (0.92 for the speech recording against 0.39 for every
     *                broadband fixture); this threshold only keeps out the
     *                extremes, and the quiet gate above takes the empty rooms.
     * tonal_peak 25:  the loudest bin over the median of the 48 bins around it.
     *                Pure tones with harmonics: the whole window is tonal (1.000).
     *                Room tone after the high pass: 0.063 .. 0.127 of frames.
     * periodic  0.35: the knocking fixture reaches 0.947 at its own 0.35 s
     *                period; the street fixture, whose gaps never repeat, 0.044.
     * noise_flat 0.10: white noise measures 0.557 .. 0.559, the tone fixtures
     *                0.000 and 0.009, the room 0.001 .. 0.004.
     * the rest:      each sits in the gap between the two fixtures it separates,
     *                not on either of them. */
    c->quiet_db     = -40.0f;
    c->transient_db =  15.0f;
    c->duty_max     =   0.25f;
    c->onset_db     =  10.0f;
    c->tonal_peak   =  25.0f;
    c->tonal_frac   =   0.50f;
    c->music_change =   0.10f;
    c->speech_band  =   0.55f;
    c->speech_mod   =   0.25f;
    c->periodic     =   0.35f;
    c->noise_flat   =   0.10f;
    c->steady_db    =  10.0f;
    c->env_std      =   1.5f;
    c->hpf_hz       = 100.0f;
}

static void env_f(const char *name, float *slot) {
    const char *v = getenv(name);
    if (v && *v) *slot = (float) atof(v);
}

void snd_cfg_from_env(snd_cfg *c) {
    env_f("SND_QUIET_DB",     &c->quiet_db);
    env_f("SND_TRANSIENT_DB", &c->transient_db);
    env_f("SND_DUTY_MAX",     &c->duty_max);
    env_f("SND_ONSET_DB",     &c->onset_db);
    env_f("SND_TONAL_PEAK",   &c->tonal_peak);
    env_f("SND_TONAL_FRAC",   &c->tonal_frac);
    env_f("SND_MUSIC_CHANGE", &c->music_change);
    env_f("SND_SPEECH_BAND",  &c->speech_band);
    env_f("SND_SPEECH_MOD",   &c->speech_mod);
    env_f("SND_PERIODIC",     &c->periodic);
    env_f("SND_NOISE_FLAT",   &c->noise_flat);
    env_f("SND_STEADY_DB",    &c->steady_db);
    env_f("SND_ENV_STD",      &c->env_std);
    env_f("SND_HPF_HZ",       &c->hpf_hz);
}

/* ── the features ─────────────────────────────────────────────────────────── */

int snd_analyse(const float *raw, int n_raw, int sr, const snd_cfg *cfg, snd_features *out) {
    if (!raw || !out || !cfg || sr <= 0 || n_raw < N_FFT) return -1;

    /* Two things before any statistic is taken, both of them about what this
     * phone's microphone actually delivers.
     *
     * The recorder hands over a third of a second of digital silence before the
     * capture begins — the six live recordings in molequla-run/senses/audio all
     * start with an exact run of zeros — and frames of literal nothing sit at
     * -120 dB, which is not a quiet room but an absent one. They moved the
     * envelope's standard deviation to 13.3 dB in every one of those six files
     * while the 10th-to-90th percentile span was under 6 dB. They are cut off
     * both ends.
     *
     * Then everything below hpf_hz goes: a phone lying on a table has more
     * power under 100 Hz — the table, the charger, mains hum — than in the room
     * above it, and with that in the spectrum every recording has one bin far
     * above the rest and every recording is "tonal". The high pass is one pole,
     * which is enough to stop that bin winning. */
    int lo = 0, hi = n_raw - 1;
    while (lo < n_raw    && fabsf(raw[lo]) <= 1e-5f) lo++;
    while (hi > lo       && fabsf(raw[hi]) <= 1e-5f) hi--;
    const int n = hi - lo + 1;
    if (n < N_FFT) return -1;

    float *pcm = (float *) malloc((size_t) n * sizeof(float));
    if (!pcm) return -1;
    {
        const float a = 1.0f / (1.0f + 2.0f * (float) SND_PI * cfg->hpf_hz / (float) sr);
        float yprev = 0.0f, xprev = raw[lo];
        for (int i = 0; i < n; i++) {
            const float x = raw[lo + i];
            yprev = a * (yprev + x - xprev);
            xprev = x;
            pcm[i] = yprev;
        }
    }

    const int n_bin   = N_FFT / 2 + 1;
    const int frames  = 1 + (n - N_FFT) / HOP;
    const float bin_hz = (float) sr / (float) N_FFT;
    const float frame_hz = (float) sr / (float) HOP;

    float *win   = (float *) malloc((size_t) N_FFT * sizeof(float));
    float *power = (float *) malloc((size_t) frames * (size_t) n_bin * sizeof(float));
    float *db    = (float *) malloc((size_t) frames * sizeof(float));
    float *band  = (float *) malloc((size_t) frames * sizeof(float));
    float *flat  = (float *) malloc((size_t) frames * sizeof(float));
    float *cent  = (float *) malloc((size_t) frames * sizeof(float));
    float *scr   = (float *) malloc((size_t) frames * sizeof(float));
    if (!win || !power || !db || !band || !flat || !cent || !scr) {
        free(pcm); free(win); free(power); free(db); free(band); free(flat); free(cent); free(scr);
        return -1;
    }

    nt_hann_window(win, N_FFT);
    if (nt_stft(power, pcm, n, N_FFT, HOP, frames, win, 1) != 0) {
        free(pcm); free(win); free(power); free(db); free(band); free(flat); free(cent); free(scr);
        return -1;
    }

    /* 300-3400 Hz is the band a telephone keeps and the one speech lives in. */
    const int b_lo = (int) (300.0f / bin_hz + 0.5f);
    const int b_hi = (int) (3400.0f / bin_hz + 0.5f);

    int *peaks = (int *) malloc((size_t) frames * sizeof(int));
    if (!peaks) {
        free(pcm); free(win); free(power); free(db); free(band); free(flat); free(cent); free(scr);
        return -1;
    }
    int tonal = 0;
    for (int f = 0; f < frames; f++) {
        const float *p = power + (size_t) f * (size_t) n_bin;

        /* level from the samples themselves, so the number is the recording's
         * and not the window function's. */
        double e = 0.0;
        const int off = f * HOP;
        for (int i = 0; i < N_FFT; i++) {
            const float x = (off + i < n) ? pcm[off + i] : 0.0f;
            e += (double) x * (double) x;
        }
        db[f] = 10.0f * log10f((float) (e / N_FFT) + EPS);

        double tot = 0.0, mid = 0.0, logsum = 0.0, wsum = 0.0;
        float  pmax = 0.0f;
        int    pmax_bin = 0;
        for (int k = 1; k < n_bin; k++) {
            const float v = p[k];
            tot    += v;
            logsum += log((double) v + EPS);
            wsum   += (double) v * (double) k * bin_hz;
            if (k >= b_lo && k <= b_hi) mid += v;
            if (v > pmax) { pmax = v; pmax_bin = k; }
        }
        /* Tonality as prominence over the neighbourhood, not over the whole
         * spectrum. A room recording is steeply tilted — almost all of its
         * power is under 1 kHz — so the loudest bin beats the average bin by a
         * factor of hundreds while nothing in the room is pitched, and by that
         * measure six of six live night recordings came out as music. Against
         * the median of the 48 bins around it (1.5 kHz wide, the peak's own
         * skirt excluded) a whistle or a note still stands out and the tilt
         * cancels. */
        double local[48];
        int nl = 0;
        for (int k = pmax_bin - 24; k <= pmax_bin + 24; k++) {
            if (k < 1 || k >= n_bin || (k >= pmax_bin - 2 && k <= pmax_bin + 2)) continue;
            local[nl++] = p[k];
        }
        for (int i = 1; i < nl; i++) {
            const double v = local[i];
            int j = i - 1;
            while (j >= 0 && local[j] > v) { local[j + 1] = local[j]; j--; }
            local[j + 1] = v;
        }
        const double skirt = nl ? local[nl / 2] : 0.0;
        const double mean = tot / (double) (n_bin - 1);
        band[f] = (tot > 0.0) ? (float) (mid / tot) : 0.0f;
        flat[f] = (mean > 0.0) ? (float) (exp(logsum / (double) (n_bin - 1)) / (mean + EPS)) : 1.0f;
        cent[f] = (tot > 0.0) ? (float) (wsum / tot) : 0.0f;

        /* A tonal frame is one where a single bin stands far above the average
         * bin: a whistle, a hum, a played note. Noise has no such bin. */
        if (skirt > 0.0 && (double) pmax / skirt >= (double) cfg->tonal_peak) peaks[tonal++] = pmax_bin;
    }

    /* Does the pitch move? A fan, a fridge and a transformer are tonal and sit
     * on one bin for the whole window; a melody spends most of its frames away
     * from its own middle note. So the measure is the share of tonal frames
     * whose loudest bin is more than two bins off the median loudest bin, not
     * how often it changed — five notes in twelve seconds change five times and
     * would look like a hum by the second measure. */
    float pitch_change = 0.0f;
    if (tonal > 0) {
        int *pk = (int *) malloc((size_t) tonal * sizeof(int));
        if (pk) {
            memcpy(pk, peaks, (size_t) tonal * sizeof(int));
            for (int i = 1; i < tonal; i++) {          /* insertion sort: tonal is small */
                const int v = pk[i];
                int j = i - 1;
                while (j >= 0 && pk[j] > v) { pk[j + 1] = pk[j]; j--; }
                pk[j + 1] = v;
            }
            const int mid_bin = pk[tonal / 2];
            int away = 0;
            for (int i = 0; i < tonal; i++) if (abs(peaks[i] - mid_bin) > 2) away++;
            pitch_change = (float) away / (float) tonal;
            free(pk);
        }
    }
    free(peaks);

    memcpy(scr, db, (size_t) frames * sizeof(float));
    qsort(scr, (size_t) frames, sizeof(float), cmp_f);
    out->db_p10 = pct(scr, frames, 0.10f);
    out->db_p50 = pct(scr, frames, 0.50f);
    out->db_p90 = pct(scr, frames, 0.90f);
    out->db_max = pct(scr, frames, 1.00f);
    out->dyn    = out->db_p90 - out->db_p10;

    double m = 0.0;
    for (int f = 0; f < frames; f++) m += db[f];
    m /= (double) frames;
    double var = 0.0;
    for (int f = 0; f < frames; f++) { const double d = db[f] - m; var += d * d; }
    out->env_std = (float) sqrt(var / (double) frames);

    out->speech_band = median_of(scr, band, frames);
    out->flatness    = median_of(scr, flat, frames);
    out->centroid    = median_of(scr, cent, frames);
    out->tonal_frac   = (float) tonal / (float) frames;
    out->pitch_change = pitch_change;

    /* Onsets: a rising edge that crosses the median by onset_db and does not
     * count again until the level has fallen back halfway. */
    int onsets = 0, loud = 0, in = 0;
    for (int f = 0; f < frames; f++) {
        const float hi = out->db_p50 + cfg->onset_db;
        if (db[f] >= hi) {
            loud++;
            if (!in) { onsets++; in = 1; }
        } else if (db[f] < out->db_p50 + cfg->onset_db * 0.5f) {
            in = 0;
        }
    }
    out->onsets = onsets;
    out->duty   = (float) loud / (float) frames;

    /* Repetition: the envelope against itself. Below env_std there is no
     * envelope to speak of and the normalised correlation is noise on noise, so
     * it is reported as zero rather than as a rhythm nobody can hear. */
    out->periodicity = 0.0f;
    out->period_s    = 0.0f;
    if (out->env_std >= cfg->env_std) {
        double denom = 0.0;
        for (int f = 0; f < frames; f++) { const double d = db[f] - m; denom += d * d; }
        const int lag_min = (int) (0.08f * frame_hz);
        const int lag_max = (int) (2.00f * frame_hz);
        /* Only lags past the first dip count. One long burst correlates with
         * itself at every short lag and its autocorrelation simply decays —
         * that is the burst's own length, not a rhythm. A repetition dips first
         * and comes back at its period, so the search starts where the curve
         * turns upward again. Without this the street fixture, whose gaps never
         * repeat, scored 0.699 at the 0.08 s floor and was called mechanical. */
        int start = -1;
        float prev = 2.0f;
        for (int lag = lag_min; lag <= lag_max && lag < frames; lag++) {
            double acc = 0.0;
            for (int f = 0; f + lag < frames; f++) acc += (db[f] - m) * (db[f + lag] - m);
            const float r = (denom > 0.0) ? (float) (acc / denom) : 0.0f;
            if (start < 0) {
                if (r > prev) start = lag;      /* the curve turned: a period may follow */
                prev = r;
                if (start < 0) continue;
            }
            if (r > out->periodicity) { out->periodicity = r; out->period_s = (float) lag / frame_hz; }
        }
    }

    /* Modulation: how much of the envelope's movement between 0.5 and 20 Hz
     * sits in the 2-8 Hz syllable band. A direct transform over a few dozen
     * frequencies is cheaper here than another stft. */
    double in_band = 0.0, all_band = 0.0;
    for (float hz = 0.5f; hz <= 20.0f; hz += 0.25f) {
        double re = 0.0, im = 0.0;
        const double w = 2.0 * SND_PI * (double) hz / (double) frame_hz;
        for (int f = 0; f < frames; f++) {
            const double d = db[f] - m;
            re += d * cos(w * f);
            im -= d * sin(w * f);
        }
        const double p = re * re + im * im;
        all_band += p;
        if (hz >= 2.0f && hz <= 8.0f) in_band += p;
    }
    out->mod_2_8 = (all_band > 0.0) ? (float) (in_band / all_band) : 0.0f;

    out->frames  = frames;
    out->seconds = (float) n / (float) sr;

    free(pcm); free(win); free(power); free(db); free(band); free(flat); free(cent); free(scr);
    return 0;
}

/* ── the cascade ──────────────────────────────────────────────────────────── */

const char *snd_label(const snd_features *f, const snd_cfg *c) {
    /* One bang, before the quiet room and not after it: a door in an empty flat
     * is a quiet window by every percentile — the 90th sits at -64.6 dB in the
     * fixture — and the event is the thing worth saying about it. A single
     * rising edge well above the median, loud for a small part of the window. */
    if (f->onsets == 1 && (f->db_max - f->db_p50) >= c->transient_db && f->duty <= c->duty_max)
        return SND_TRANSIENT;

    if (f->db_p90 < c->quiet_db) return SND_QUIET;

    /* Pitched and moving. A hum is pitched and does not move, which is why the
     * pitch has to change before this is called music. */
    if (f->tonal_frac >= c->tonal_frac && f->pitch_change >= c->music_change)
        return SND_MUSIC;

    /* Speech before repetition: syllables are periodic too, and the thing that
     * tells them from a motor is where the power sits and that the envelope
     * moves at a syllable rate. */
    if (f->speech_band >= c->speech_band && f->mod_2_8 >= c->speech_mod && f->env_std >= c->env_std)
        return SND_SPEECH;

    if (f->periodicity >= c->periodic && f->onsets >= 2) return SND_MECHANICAL;

    if (f->flatness >= c->noise_flat && f->dyn <= c->steady_db) return SND_BROADBAND;

    return SND_UNSTEADY;
}

void snd_print_features(const snd_features *f, void *stream) {
    FILE *s = (FILE *) stream;
    fprintf(s, "frames=%d sec=%.2f db_p10=%.1f db_p50=%.1f db_p90=%.1f db_max=%.1f dyn=%.1f "
               "env_std=%.2f band=%.3f flat=%.3f tonal=%.3f pitch=%.3f cent=%.0f "
               "onsets=%d duty=%.3f period=%.3f@%.2fs mod2_8=%.3f\n",
            f->frames, (double) f->seconds, (double) f->db_p10, (double) f->db_p50,
            (double) f->db_p90, (double) f->db_max, (double) f->dyn, (double) f->env_std,
            (double) f->speech_band, (double) f->flatness, (double) f->tonal_frac,
            (double) f->pitch_change, (double) f->centroid, f->onsets, (double) f->duty,
            (double) f->periodicity, (double) f->period_s, (double) f->mod_2_8);
}

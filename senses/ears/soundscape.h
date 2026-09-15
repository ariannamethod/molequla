/* soundscape.h — the half of hearing that is not language.
 *
 * whisper answers one question about a recording: which words were said. A car
 * passing, a door, a fan, a room with nobody in it are all the same answer to
 * it — nothing — and molequla's field used to receive exactly that nothing
 * (molequla_new_logic.md §3). This is the other question, asked without a
 * model: what kind of sound was that.
 *
 * No weights, no vocabulary beyond the labels below, no learning. The frames go
 * through notorch's own `nt_stft`, and every label is a region of the measured
 * features: level percentiles, band ratios, spectral flatness and peakiness, the
 * envelope's autocorrelation and its 2-8 Hz modulation share. A label that no
 * feature can produce does not exist here; a named sound — "a door closed", "a
 * car passed" — needs the tagger port in PORT_NOTES_SOUND.md, and this file is
 * what runs until then.
 *
 * Every threshold is a field of snd_cfg, defaulted from a measurement and
 * overridable by an environment variable of the same name (SND_*), because a
 * number baked into a cascade is an architecture nobody can re-measure.
 */

#ifndef SOUNDSCAPE_H
#define SOUNDSCAPE_H

/* The vocabulary, in the order the cascade tries it. */
#define SND_QUIET      "quiet room"
#define SND_TRANSIENT  "a single loud transient"
#define SND_MUSIC      "music is audible"
#define SND_SPEECH     "speech-like modulation, words unclear"
#define SND_MECHANICAL "repeated mechanical noise"
#define SND_BROADBAND  "steady broadband noise"
#define SND_UNSTEADY   "an unsteady sound without clear structure"

typedef struct {
    /* level, dB relative to a full-scale square wave (a full-scale sine is -3) */
    float db_p10, db_p50, db_p90, db_max;
    float dyn;            /* db_p90 - db_p10 */
    float env_std;        /* standard deviation of the frame level, dB */
    /* spectrum */
    float speech_band;    /* median share of the power between 300 and 3400 Hz */
    float flatness;       /* median spectral flatness, 0 tonal .. 1 white */
    float tonal_frac;     /* share of frames with one bin far above its neighbourhood */
    float pitch_change;   /* share of tonal frames sitting away from the median pitch */
    float centroid;       /* median spectral centroid, Hz */
    /* time structure */
    int   onsets;         /* rising edges above db_p50 + cfg.onset_db */
    float duty;           /* share of frames above db_p50 + cfg.onset_db/2 */
    float periodicity;    /* best envelope autocorrelation, lag 0.08-2.0 s */
    float period_s;       /* the lag that produced it, seconds */
    float mod_2_8;        /* share of the envelope's 0.5-20 Hz modulation in 2-8 Hz */
    int   frames;
    float seconds;
} snd_features;

typedef struct {
    float quiet_db;       /* SND_QUIET_DB      below this at the 90th percentile: a quiet room */
    float transient_db;   /* SND_TRANSIENT_DB  peak above the median that counts as a bang */
    float duty_max;       /* SND_DUTY_MAX      a bang is loud for a small share of the window */
    float onset_db;       /* SND_ONSET_DB      rising edge above the median that opens an onset */
    float tonal_peak;     /* SND_TONAL_PEAK    loudest bin over its own neighbourhood: a tonal frame */
    float tonal_frac;     /* SND_TONAL_FRAC    share of tonal frames music needs */
    float music_change;   /* SND_MUSIC_CHANGE  share of tonal frames whose pitch moved */
    float speech_band;    /* SND_SPEECH_BAND   share of power in 300-3400 Hz speech needs */
    float speech_mod;     /* SND_SPEECH_MOD    share of envelope modulation in 2-8 Hz */
    float periodic;       /* SND_PERIODIC      envelope autocorrelation that is a repetition */
    float noise_flat;     /* SND_NOISE_FLAT    spectral flatness that is broadband noise */
    float steady_db;      /* SND_STEADY_DB     dynamic range a steady sound stays under */
    float env_std;        /* SND_ENV_STD       envelope movement below which there is no rhythm */
    float hpf_hz;         /* SND_HPF_HZ        one-pole high pass in front of everything, Hz */
} snd_cfg;

/* Defaults measured on this phone; see soundscape.c and MOLEQULALOG2.md. */
void snd_cfg_defaults(snd_cfg *c);
/* Each field overridden by the environment variable named beside it above. */
void snd_cfg_from_env(snd_cfg *c);

/* pcm: n mono f32 in [-1, 1] at sr Hz (16000 here). Returns 0, -1 on bad args
 * or allocation failure. A recording shorter than one 512-sample frame is not
 * analysable and returns -1. */
int snd_analyse(const float *pcm, int n, int sr, const snd_cfg *cfg, snd_features *out);

/* One of the SND_* strings above. Never NULL, never allocates. */
const char *snd_label(const snd_features *f, const snd_cfg *cfg);

/* The features as one line, for -v and for the gate's table. */
void snd_print_features(const snd_features *f, void *stream);

#endif /* SOUNDSCAPE_H */

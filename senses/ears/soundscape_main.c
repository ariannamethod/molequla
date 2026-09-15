/* soundscape_main.c — the describer's command line.
 *
 *   soundscape [-v] <file.wav>
 *
 * One 16 kHz mono wav in, one English line out, on stdout and nothing else on
 * it, so that the caller needs no parser: `senses.sh` takes the line as it is
 * and writes it into the field as an `[ears env …]` fragment. -v adds the
 * measured features to stderr, which is where the thresholds in soundscape.c
 * came from.
 */

#include "mel.h"
#include "soundscape.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
        "usage: soundscape [-v] <file.wav>\n"
        "  16 kHz mono wav in, one line about what kind of sound it was out.\n"
        "  -v  the measured features on stderr\n"
        "  Thresholds: SND_QUIET_DB SND_TRANSIENT_DB SND_DUTY_MAX SND_ONSET_DB\n"
        "              SND_TONAL_PEAK SND_TONAL_FRAC SND_MUSIC_CHANGE SND_SPEECH_BAND\n"
        "              SND_SPEECH_MOD SND_PERIODIC SND_NOISE_FLAT SND_STEADY_DB\n"
        "              SND_ENV_STD\n");
}

int main(int argc, char **argv) {
    const char *wav = NULL;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (argv[i][0] == '-')      { fprintf(stderr, "soundscape: unknown option %s\n", argv[i]); usage(); return 2; }
        else if (!wav)                   wav = argv[i];
        else                             { usage(); return 2; }
    }
    if (!wav) { usage(); return 2; }

    float *pcm = NULL;
    int n = 0;
    if (ears_read_wav(wav, &pcm, &n) != 0) return 1;

    snd_cfg cfg;
    snd_cfg_defaults(&cfg);
    snd_cfg_from_env(&cfg);

    snd_features f;
    if (snd_analyse(pcm, n, 16000, &cfg, &f) != 0) {
        fprintf(stderr, "soundscape: '%s' is too short to describe (%d samples)\n", wav, n);
        free(pcm);
        return 1;
    }
    free(pcm);

    if (verbose) snd_print_features(&f, stderr);
    printf("%s\n", snd_label(&f, &cfg));
    return 0;
}

#!/bin/sh
# gate_mel — ears' log-mel against whisper.cpp's own, for jfk.wav on tiny.
#
# The oracle is harness/oracle_dump, which #includes src/whisper.cpp so it runs
# whisper_pcm_to_mel itself rather than an imitation of it. Both dumps carry the
# same header (n_mel, n_len, n_len_org), so a geometry disagreement fails before
# any float is read.
#
# Tolerance 1e-4. The measured value is 0 — the front end is bit-identical — so
# the tolerance is headroom, not a fit to the result.
#
# Shown red: EARS_MEL_BREAK=<bin> zeroes one column of the filter bank.
set -e

WHISPER=${WHISPER:-$HOME/arianna/whisper.cpp}
REF=${REF:-$HOME/arianna/ears-reference}
OUT=${OUT:-$(dirname "$0")/out}
MODEL=$WHISPER/models/ggml-tiny.bin
WAV=$REF/wav/jfk.wav
CORES=${CORES:-0-3}

mkdir -p "$OUT"
echo "gate_mel: $WAV on $(basename "$MODEL"), cores $CORES"

taskset -c "$CORES" ./harness/oracle_dump "$MODEL" "$WAV" "$OUT/mel.oracle" "$OUT/enc.oracle" 2>&1 \
    | grep -E "^oracle_dump: (mel|enc) "
taskset -c "$CORES" ./tests/dump_mel "$MODEL" "$WAV" "$OUT/mel.ears" 4 2>&1 | grep "^dump_mel:"

./tests/cmp_f32 "$OUT/mel.ears" "$OUT/mel.oracle" 3 1e-4

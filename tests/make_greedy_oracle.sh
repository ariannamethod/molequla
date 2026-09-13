#!/bin/sh
# make_greedy_oracle — regenerate ears-reference/*.greedy.{json,txt}.
#
# The stored ears-reference/*.json were made with whisper-cli's defaults: greedy
# sampling with best-of 5 and the temperature-fallback ladder. That is a different
# search, so diffing a pure-greedy port against it measures the search, not the
# port. These runs pin the oracle to the algorithm ears v1 implements:
#
#   -bo 1   one candidate
#   -bs 1   greedy strategy rather than beam search (cli.cpp:1213)
#   -tp 0   temperature zero
#   -nf     no temperature fallback, so the ladder is the single value
#
# jfk gets -l en, the microphone wavs -l auto, matching REFERENCE.md's table.
set -e

WHISPER=${WHISPER:-$HOME/arianna/whisper.cpp}
REF=${REF:-$HOME/arianna/ears-reference}
CORES=${CORES:-0-3}

mkdir -p "$REF/logs"
for model in tiny base; do
  for wav in jfk speech_air_14s ambient_8s; do
    [ "$wav" = jfk ] && lang=en || lang=auto
    set -- taskset -c "$CORES" "$WHISPER/build-blas/bin/whisper-cli" \
        -m "$WHISPER/models/ggml-$model.bin" -f "$REF/wav/$wav.wav" \
        -t 4 -l $lang -bo 1 -bs 1 -tp 0 -nf -ojf -otxt -of "$REF/$wav.$model.greedy"
    echo "$@" > "$REF/logs/$wav.$model.greedy.cmd"
    /usr/bin/time -v "$@" > "$REF/logs/$wav.$model.greedy.stdout" 2> "$REF/logs/$wav.$model.greedy.stderr"
    printf '%-16s %-5s ' "$wav" "$model"
    grep -E "Elapsed \(wall" "$REF/logs/$wav.$model.greedy.stderr" | sed 's/.*): //' | tr -d '\n'
    printf '  [%s]\n' "$(tr -d '\n' < "$REF/$wav.$model.greedy.txt")"
  done
done

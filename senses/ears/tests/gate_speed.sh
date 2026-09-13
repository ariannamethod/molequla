#!/bin/sh
# gate_speed — wall time and peak RSS for ears against whisper-cli, same wav,
# same model, same cores, same thread count, both under /usr/bin/time -v.
#
# This is a report, not a pass/fail: the gate of this organ is the transcript, and
# a port that matched the oracle's clock while disagreeing with its tokens would
# have failed at the only thing that matters. What is enforced is that a run
# finishes and produces output — a silent organ is a failure whatever it costs.
#
# Both sides run pure greedy: whisper-cli with -bo 1 -bs 1 -tp 0 -nf, which is the
# algorithm ears implements. Comparing against the stored REFERENCE.md numbers
# instead would compare a 1-candidate decoder with a 5-candidate one.
set -e

WHISPER=${WHISPER:-$HOME/arianna/whisper.cpp}
REF=${REF:-$HOME/arianna/ears-reference}
CORES=${CORES:-0-3}
WAVS=${WAVS:-"jfk speech_air_14s"}

t_of()  { grep -E "Elapsed \(wall" "$1" | sed 's/.*): //'; }
rss_of(){ grep -E "Maximum resident" "$1" | sed 's/.*: //'; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

printf '%-16s %-6s %-10s %-12s %-10s %-12s\n' wav model ears_wall ears_rss_kB whisper_wall whisper_rss_kB
for model in tiny base; do
  for wav in $WAVS; do
    [ "$wav" = jfk ] && lang=en || lang=auto

    /usr/bin/time -v taskset -c "$CORES" ./ears \
        "$WHISPER/models/ggml-$model.bin" "$REF/wav/$wav.wav" -l $lang -t 4 \
        > "$tmp/ears.out" 2> "$tmp/ears.err"
    [ -s "$tmp/ears.out" ] || { echo "gate_speed: FAIL — ears produced no output for $wav/$model"; exit 1; }

    /usr/bin/time -v taskset -c "$CORES" "$WHISPER/build-blas/bin/whisper-cli" \
        -m "$WHISPER/models/ggml-$model.bin" -f "$REF/wav/$wav.wav" \
        -t 4 -l $lang -bo 1 -bs 1 -tp 0 -nf \
        > "$tmp/w.out" 2> "$tmp/w.err"

    printf '%-16s %-6s %-10s %-12s %-10s %-12s\n' \
        "$wav" "$model" "$(t_of "$tmp/ears.err")" "$(rss_of "$tmp/ears.err")" \
        "$(t_of "$tmp/w.err")" "$(rss_of "$tmp/w.err")"
  done
done
echo "gate_speed: cores $CORES, 4 threads, pure greedy on both sides"

#!/bin/sh
# gate_transcript — token-for-token equality with whisper.cpp under pure greedy.
#
# The oracle runs stored in ears-reference/*.json used whisper-cli's default
# strategy, which is greedy with best-of 5 and a temperature fallback ladder — a
# different algorithm, not a different implementation of the same one. The gate
# therefore uses *.greedy.json, produced by rerunning whisper-cli with
#   -bo 1 -bs 1 -tp 0 -nf
# (best-of one, no beam, temperature zero, no fallback), which is what ears v1
# implements. tests/make_greedy_oracle.sh regenerates them.
#
# Compared: the decode-order token id stream, from `ears --tokens` against the
# "id" fields of the oracle JSON. The JSON drops one of two adjacent timestamp
# tokens at a segment boundary — whisper.cpp's own emitter skips it rather than
# pushing it into either segment (src/whisper.cpp:7808-7813) — so a doubled
# timestamp id is collapsed on the ears side before the diff, and only there.
#
# Shown red: by patch rather than by a switch, because the thing to break lives in
# the organ and not in the harness — deleting the suppress-blank filter from
# filter_logits() in ears.c. The run and its output are in EARSLOG.md; no break
# hook is left behind in ears.c for it.
set -e

WHISPER=${WHISPER:-$HOME/arianna/whisper.cpp}
REF=${REF:-$HOME/arianna/ears-reference}
CORES=${CORES:-0-3}
fails=0

ids_of_json() {
    grep -o '"id": *[0-9]*' "$1" | grep -o '[0-9]*' | tr '\n' ' ' | sed 's/  */ /g;s/ *$//'
}

# collapse a run of identical adjacent ids >= 50365 (timestamps) to one
collapse_ts() {
    awk '{ prev=""; out="";
           for (i = 1; i <= NF; i++) {
               if ($i == prev && $i+0 >= 50365) continue;
               out = out (out == "" ? "" : " ") $i; prev = $i;
           }
           print out }'
}

for model in tiny base; do
  for wav in jfk speech_air_14s ambient_8s; do
    [ "$wav" = jfk ] && lang=en || lang=auto
    ref_json=$REF/$wav.$model.greedy.json
    [ -f "$ref_json" ] || { echo "gate_transcript: missing $ref_json — run tests/make_greedy_oracle.sh"; exit 2; }

    out=$(taskset -c "$CORES" ./ears "$WHISPER/models/ggml-$model.bin" "$REF/wav/$wav.wav" \
              -l $lang -t 4 --tokens 2>&1)
    mine=$(printf '%s\n' "$out" | sed -n 's/^ears: tokens //p' | tr '\n' ' ' | sed 's/ *$//' | collapse_ts)
    orcl=$(ids_of_json "$ref_json")
    text=$(printf '%s\n' "$out" | grep -v '^ears:' | sed 's/^ *//;s/ *$//')
    rtext=$(tr "\n" " " < "$REF/$wav.$model.greedy.txt" 2>/dev/null | sed "s/  */ /g;s/^ //;s/ $//")

    if [ "$mine" = "$orcl" ]; then
        echo "gate_transcript: PASS $wav/$model  $(printf '%s' "$mine" | wc -w) tokens"
    else
        echo "gate_transcript: FAIL $wav/$model"
        echo "  ears   : $mine"
        echo "  oracle : $orcl"
        fails=$((fails + 1))
    fi
    echo "  ears text   : [$text]"
    echo "  oracle text : [$rtext]"
  done
done

[ "$fails" -eq 0 ] || { echo "gate_transcript: $fails of 6 rows differ"; exit 1; }
echo "gate_transcript: all 6 rows equal token for token"

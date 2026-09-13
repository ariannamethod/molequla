#!/bin/bash
# eye — the field's eye: one image in, one honest sentence out.
#
#   ./eye <image> [prompt]
#
# Runs the pure-C SmolVLM engine on the Yent eye weights (SmolVLM2-500M merged
# by Codex). Weights are looked up next to this script; EYE_MODEL / EYE_MMPROJ
# override them, EYE_VERBOSE=1 keeps the engine's own loading chatter.
set -u

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# packed-matvec threading floor for THIS shape: notorch's 4M default was measured
# on a 360M decoder; ours are 2.46M and would never thread. Measured here: 256K
# takes decode from 3.7 to 7.3 tok/s with byte-identical output.
export NT_QMV_THREAD_MIN=${NT_QMV_THREAD_MIN:-262144}
MODEL=${EYE_MODEL:-$DIR/models/yent_eye_smolvlm2_lora_v2_f16.gguf}
MMPROJ=${EYE_MMPROJ:-$DIR/models/yent_eye_smolvlm2_lora_v2_mmproj_f16.gguf}

if [ $# -lt 1 ]; then
    echo "usage: eye <image> [prompt]" >&2
    exit 2
fi
IMG=$1
PROMPT=${2:-Describe this image in one sentence.}

[ -f "$IMG" ]        || { echo "eye: no such image: $IMG" >&2; exit 3; }
[ -x "$DIR/ocelli" ] || { echo "eye: engine not built: $DIR/ocelli" >&2; exit 4; }
[ -f "$MODEL" ]      || { echo "eye: weights missing: $MODEL (set EYE_MODEL)" >&2; exit 5; }
[ -f "$MMPROJ" ]     || { echo "eye: mmproj missing: $MMPROJ (set EYE_MMPROJ)" >&2; exit 5; }

out=$("$DIR/ocelli" "$MODEL" --image "$IMG" --mmproj "$MMPROJ" -p "$PROMPT" 2>&1)
rc=$?
if [ $rc -ne 0 ]; then
    printf '%s\n' "$out" >&2
    exit $rc
fi

if [ -n "${EYE_VERBOSE:-}" ]; then
    printf '%s\n' "$out"
else
    printf '%s\n' "$out" | grep -E '^(OURS: |-- )'
fi

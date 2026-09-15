#!/bin/bash
# Gate for the structured sidecar of the senses (world_ledger.go, ROADMAP 10).
# Every case drives the real functions in `senses.sh`, sourced with
# SENSES_LIB_ONLY=1, over a fixture — no camera, no microphone, no GPS is woken
# here, and none is needed: what is under test is the line the organ writes, not
# the hardware that fills it. Break fact_emit on purpose and this goes red.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0
ok()   { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf 'FAIL %s: %s\n' "$1" "${2:-}"; }

# The library, over a scratch run directory. MOLEQULA_RUN puts facts.jsonl and
# everything else under $TMP; SENSES_INGEST= keeps the writer out of a unit test
# of the writing side.
export MOLEQULA_RUN="$TMP/run"
export SENSES_INGEST=""
export CPUS="4-7"
mkdir -p "$MOLEQULA_RUN/senses"
# shellcheck disable=SC1090
SENSES_LIB_ONLY=1 . "$HERE/senses.sh"

FACTS="$SENSES_FACTS"

# reset — start every case from an empty file.
reset() { : > "$FACTS"; }

# last_line — the line the organ just wrote.
last_line() { tail -n 1 "$FACTS"; }

# valid <name> — the last line parses as JSON and carries the four required
# fields plus a valid_from and a provenance object.
valid() {
    local name="$1" line
    line="$(last_line)"
    if [ -z "$line" ]; then bad "$name" "nothing was written to facts.jsonl"; return 1; fi
    if ! printf '%s' "$line" | jq -e . >/dev/null 2>&1; then
        bad "$name" "not JSON: $line"; return 1
    fi
    if ! printf '%s' "$line" | jq -e '
            (.source|type=="string" and length>0) and
            (.subject|type=="string" and length>0) and
            (.predicate|type=="string" and length>0) and
            (.object|type=="string") and
            (.valid_from|test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")) and
            (.provenance|type=="object")' >/dev/null 2>&1; then
        bad "$name" "shape is wrong: $line"; return 1
    fi
    ok "$name"
    return 0
}

# field <name> <jq filter> <expected>
field() {
    local name="$1" filter="$2" want="$3" got
    got="$(last_line | jq -r "$filter" 2>/dev/null)"
    if [ "$got" = "$want" ]; then ok "$name"; else bad "$name" "got [$got], want [$want]"; fi
}

# ── the eye ─────────────────────────────────────────────────────────────────
# The sentence the live eye wrote on the first night, quotes and all — the
# reason jq builds these lines instead of printf.
EYE_SAID='A black screen with a small white text that reads "the world is not what it seems".'

reset
fact_emit eye "$(cam_lens 0)" interpreted_as "$EYE_SAID" \
    "$(eye_prov 0 "$TMP/run/senses/frames/20260914T010028Z_cam0.jpg" 31 1020)"
valid "eye: the line is JSON"
field "eye: the subject is the lens"     '.subject'   "rear camera"
field "eye: the predicate is not \`is\`" '.predicate' "interpreted_as"
field "eye: the sentence survives quoting" '.object'  "$EYE_SAID"
field "eye: provenance names the camera" '.provenance.camera' "0"
field "eye: provenance names the weights" '.provenance.model' "yent_eye_ours_q6_k.gguf"
field "eye: provenance names the projector" '.provenance.mmproj' "yent_eye_smolvlm2_lora_v2_mmproj_q8_0.gguf"
field "eye: provenance keeps the cost"   '.provenance.rss_mb' "1020"
field "eye: provenance keeps the conditions" '.provenance.conditions.cpus' "4-7"

reset
fact_emit eye "$(cam_lens 1)" interpreted_as "An empty room has a ceiling fan." \
    "$(eye_prov 1 "$TMP/f.jpg" 28 951)"
valid "eye: the front camera is its own subject"
field "eye: cam1 is the front camera" '.subject' "front camera"

# ── the ears ────────────────────────────────────────────────────────────────
reset
fact_emit ears microphone hearing speech "$(ears_prov "And so my fellow Americans" 0)"
valid "ears: speech is a fact"
field "ears: the subject"        '.subject'  "microphone"
field "ears: the window"         '.provenance.window_s' "12"
field "ears: the recognizer"     '.provenance.asr' "ears"
field "ears: the weights"        '.provenance.model' "ggml-tiny.bin"
field "ears: the transcript rides in the provenance" '.provenance.text' "And so my fellow Americans"

reset
fact_emit ears microphone hearing silence "$(ears_prov "" 0)"
valid "ears: silence is a fact too"
field "ears: silence has no text" '.provenance.text' ""

reset
fact_emit ears microphone interpreted_as "[Music] and a voice" "$(ears_prov "[Music] and a voice" 0)"
valid "ears: a bracketed tag does not break the line"
field "ears: the tag survives" '.object' "[Music] and a voice"

# ── place ───────────────────────────────────────────────────────────────────
# The live fix, with the neighbourhood replaced.
LAT=31.2657873; LON=34.7544114; ACC=13; NAME="<neighbourhood>, Be'er-Sheva, Israel"

reset
fact_emit place phone at_place "$NAME" \
    "$(jq -cn --arg lat "$LAT" --arg lon "$LON" --arg accuracy_m "$ACC" \
        --arg provider network --arg moved_m 2 --arg geocoder nominatim \
        '{lat:$lat,lon:$lon,accuracy_m:$accuracy_m,provider:$provider,
          moved_m:$moved_m,geocoder:$geocoder} | with_entries(select(.value != ""))')"
valid "place: the line is JSON"
field "place: the subject"   '.subject'   "phone"
field "place: the predicate" '.predicate' "at_place"
field "place: the apostrophe survives" '.object' "$NAME"
field "place: the fix accuracy" '.provenance.accuracy_m' "13"
field "place: the provider"     '.provenance.provider'   "network"

reset
fact_emit place phone at_position "$LAT,$LON" \
    "$(jq -cn --arg accuracy_m "$ACC" --arg provider network --arg moved_m "" --arg place "$NAME" \
        '{accuracy_m:$accuracy_m,provider:$provider,moved_m:$moved_m,place:$place}
         | with_entries(select(.value != ""))')"
valid "place: the coordinate is its own predicate"
field "place: the coordinate is lat,lon" '.object' "$LAT,$LON"
field "place: an empty field is dropped, not written as \"\"" \
      '.provenance | has("moved_m")' "false"

reset
fact_emit place sky reported_as "fog" \
    "$(jq -cn --arg weather_code 45 --arg temp_c 22.5 --arg humidity 97 \
        --arg wind_kmh 1.8 --arg source open-meteo --arg place "$NAME" \
        '{weather_code:$weather_code,temp_c:$temp_c,humidity:$humidity,
          wind_kmh:$wind_kmh,source:$source,place:$place}')"
valid "place: the sky is a fact of its own"
field "place: the sky's subject" '.subject' "sky"
field "place: the WMO code rides along" '.provenance.weather_code' "45"

# ── the file as a whole ─────────────────────────────────────────────────────
# One pass of all three organs, appended: every line of facts.jsonl is JSON on
# its own, which is what the ingest's line-by-line reader requires.
reset
fact_emit eye "$(cam_lens 0)" interpreted_as "$EYE_SAID" "$(eye_prov 0 "$TMP/f.jpg" 31 1020)"
fact_emit ears microphone hearing silence "$(ears_prov "" 0)"
fact_emit place phone at_place "$NAME" '{"provider":"network"}'
n="$(wc -l < "$FACTS")"
if [ "$n" -eq 3 ]; then ok "a pass of three organs writes three lines"; else bad "a pass of three organs writes three lines" "got $n"; fi
if jq -e . "$FACTS" >/dev/null 2>&1; then
    ok "every line of facts.jsonl is its own JSON object"
else
    bad "every line of facts.jsonl is its own JSON object" "$(cat "$FACTS")"
fi

# The sidecar is a switch: SENSES_FACTS= writes nothing at all.
reset
saved="$SENSES_FACTS"; SENSES_FACTS=""
fact_emit eye "rear camera" interpreted_as "nothing should be written" '{}'
SENSES_FACTS="$saved"
if [ ! -s "$FACTS" ]; then ok "SENSES_FACTS= writes no facts"; else bad "SENSES_FACTS= writes no facts" "$(cat "$FACTS")"; fi

# Rotation: over the cap the file is moved aside and the next fact starts a
# fresh one. The ingest cursor reads a shrunken file as rotated and restarts.
reset
SENSES_FACTS_MAX_KB=1
head -c 2048 /dev/zero | tr '\0' 'x' >> "$FACTS"
fact_emit place phone at_place "$NAME" '{}'
if [ -f "$FACTS.1" ] && [ "$(wc -l < "$FACTS")" -eq 1 ]; then
    ok "facts.jsonl rotates over SENSES_FACTS_MAX_KB"
else
    bad "facts.jsonl rotates over SENSES_FACTS_MAX_KB" "rotated=$( [ -f "$FACTS.1" ] && echo yes || echo no) lines=$(wc -l < "$FACTS")"
fi
SENSES_FACTS_MAX_KB=4096
rm -f "$FACTS.1"

printf '\n%d pass, %d fail\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

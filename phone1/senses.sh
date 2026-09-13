#!/bin/bash
# senses.sh — one pass of the phone's senses, then exit. The colony sleeps most
# of the day; the senses do not. They run in their own short slots and leave
# what they found in the DNA field, so that when an organism wakes at 04:00 the
# world of the last eight hours is already food in ../dna/output/.
#
#   senses.sh [eye|ears|place|all]      (default: all)
#
# Three organs, three directories under $MOLEQULA_RUN/dna/output/:
#   eye   -> world/   one camera frame each from the back and the front camera,
#                     through reffs/ocelli/eye (SmolVLM2-500M in C), one
#                     fragment per sentence.
#   ears  -> sound/   12 s from the microphone through whisper.cpp tiny, one
#                     fragment, and only when there was speech in it.
#   place -> place/   one fragment: where the phone is, what the sky is doing,
#                     and whether it moved since the last pass.
#
# A fragment is gen_<unix>_<seq>.txt, the name dnaListNew orders by, prefixed
# with a bracketed header the organisms read as plain text. Nothing here reads
# the organisms' cursors and nothing here deletes another writer's fragments:
# these directories are food, pruned by the hand that fills them.
#
# Every value below can be overridden from the environment; the ASR binary and
# model are two variables on purpose, so the native `ears` on notorch replaces
# whisper.cpp without touching the rest of this file.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RUN="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
HOME_TERMUX="${SENSES_TERMUX_HOME:-/data/data/com.termux/files/home}"

SENSES_DIR="$RUN/senses"
FRAMES="$SENSES_DIR/frames"
AUDIO="$SENSES_DIR/audio"
LOGF="$SENSES_DIR/senses.log"
SEQF="$SENSES_DIR/seq"
LASTF="$SENSES_DIR/place.last"
LOCK="$SENSES_DIR/.lock"

# --- the eye ---------------------------------------------------------------
# The eye is not part of this repo and nothing here links against it: `reffs/`
# holds reference clones, never dependencies. SENSES_EYE is the one variable
# that names the wrapper — the checkout beside this one if it is there, the
# ocelli checkout on this phone otherwise.
SENSES_EYE="${SENSES_EYE:-}"
if [ -z "$SENSES_EYE" ]; then
    if [ -x "$REPO/reffs/ocelli/eye" ]; then
        SENSES_EYE="$REPO/reffs/ocelli/eye"
    else
        SENSES_EYE="$HOME_TERMUX/arianna/molequla/reffs/ocelli/eye"
    fi
fi
SENSES_EYE_MODELS="${SENSES_EYE_MODELS:-$HOME_TERMUX/models/ocelli}"
SENSES_EYE_MODEL="${SENSES_EYE_MODEL:-$SENSES_EYE_MODELS/yent_eye_ours_q6_k.gguf}"
SENSES_EYE_MMPROJ="${SENSES_EYE_MMPROJ:-$SENSES_EYE_MODELS/yent_eye_smolvlm2_lora_v2_mmproj_q8_0.gguf}"
SENSES_EYE_PROMPT="${SENSES_EYE_PROMPT:-Describe this image in one sentence.}"
SENSES_EYE_CAMS="${SENSES_EYE_CAMS:-0 1}"
# The eye resizes the longest edge to 2048 before it does anything else
# (reffs/ocelli/vision.c:55-66). A 4080x3060 camera jpeg would be decoded to
# 150 MB of float first; 1024 costs 9 MB and, with one global frame, ends up
# at the same 512x512 the tower sees.
SENSES_EYE_EDGE="${SENSES_EYE_EDGE:-1024}"
# The eye peaks near 1 GB. Below this much MemAvailable it does not start.
SENSES_EYE_MIN_MB="${SENSES_EYE_MIN_MB:-1300}"

# --- the ears --------------------------------------------------------------
SENSES_ASR="${SENSES_ASR:-$HOME_TERMUX/arianna/whisper.cpp/build-blas/bin/whisper-cli}"
SENSES_ASR_MODEL="${SENSES_ASR_MODEL:-$HOME_TERMUX/arianna/whisper.cpp/models/ggml-tiny.bin}"
# tiny, four threads, language auto. -nth is the no-speech threshold and -sns
# suppresses non-speech tokens: base spent 185 s on 8 s of room noise and said
# "[Motor]" (arianna/ears-reference/REFERENCE.md); tiny on the same wav emitted
# nothing, which is the behaviour wanted here.
SENSES_ASR_ARGS="${SENSES_ASR_ARGS:--t 4 -l auto -nth 0.6 -sns -nt -np}"
SENSES_REC_SECONDS="${SENSES_REC_SECONDS:-12}"
# Shorter than this, after the noise tags are stripped, is not speech.
SENSES_SPEECH_MIN_CHARS="${SENSES_SPEECH_MIN_CHARS:-8}"

# --- place -----------------------------------------------------------------
SENSES_MOVE_M="${SENSES_MOVE_M:-50}"
SENSES_UA="${SENSES_UA:-molequla-senses/1.0 (phone-1, Arianna Method)}"
SENSES_HTTP_TIMEOUT="${SENSES_HTTP_TIMEOUT:-25}"

# --- shared ----------------------------------------------------------------
SENSES_KEEP="${SENSES_KEEP:-48}"          # frames and wavs kept on disk
SENSES_FRAG_KEEP="${SENSES_FRAG_KEEP:-64}" # fragments kept per source directory
SENSES_SSH="${SENSES_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=8 -i /root/.ssh/id_ed25519 -p 8022 u0_a327@localhost}"
NAMES="earth air water fire witness"

now_iso() { date -u +%FT%TZ; }
stamp()   { date -u +%Y%m%dT%H%M%SZ; }
say()     { echo "[senses] $*"; }

mem_avail_mb() { awk '/^MemAvailable:/{printf "%.0f", $2/1024}' /proc/meminfo; }

# The colony owns the big cores while it is up; the senses take the little ones
# then, and the big ones when the phone is otherwise asleep.
colony_alive() {
    local n p
    for n in $NAMES; do
        p="$(cat "$RUN/pids/$n.pid" 2>/dev/null)" || continue
        [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return 0
    done
    return 1
}

# --- one pass at a time ----------------------------------------------------
take_lock() {
    local old
    if mkdir "$LOCK" 2>/dev/null; then echo $$ > "$LOCK/pid"; return 0; fi
    old="$(cat "$LOCK/pid" 2>/dev/null)"
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
        say "another pass is running (pid $old) — this one exits"
        return 1
    fi
    say "clearing a stale lock${old:+ (pid $old)}"
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || return 1
    echo $$ > "$LOCK/pid"
    return 0
}

# --- fragments -------------------------------------------------------------
# A sequence that only ever grows, so (unix, seq) is a total order even when two
# fragments land in the same second: dnaNewer compares exactly that pair.
next_seq() {
    local n
    n="$(cat "$SEQF" 2>/dev/null)"
    case "${n:-}" in ''|*[!0-9]*) n=0 ;; esac
    n=$((n + 1))
    printf '%d' "$n" > "$SEQF"
    printf '%d' "$n"
}

# frag_write <source-dir> <text> — one fragment, header already in the text.
frag_write() {
    local src="$1" text="$2" dir name
    dir="$RUN/dna/output/$src"
    mkdir -p "$dir" || return 1
    name="gen_$(date -u +%s)_$(next_seq).txt"
    printf '%s\n' "$text" > "$dir/$name" || return 1
    say "$src/$name (${#text} B)"
    return 0
}

# prune_dir <dir> <keep> — newest `keep` files by mtime, the rest go. The
# organisms keep cursors into these directories and only advance them while
# they run, so nothing is pruned by age: a colony that slept through eight
# hours of senses still finds the fragments where they were left.
prune_dir() {
    local dir="$1" keep="$2" f
    [ -d "$dir" ] || return 0
    [ "$keep" -gt 0 ] || return 0
    ls -1t "$dir" 2>/dev/null | tail -n +$((keep + 1)) | while IFS= read -r f; do
        [ -n "$f" ] && rm -f "$dir/$f"
    done
}

# --- the eye ---------------------------------------------------------------
EYE_RC=0; EYE_WALL=0; EYE_RSS=0; EYE_FRAGS=0; EYE_FRAMES=0; EYE_NOTE=""

# eye_one <camera-id> -> frames, fragments, wall, peak RSS
eye_one() {
    local cam="$1" ts remote local_jpg out err rc t0 t1 rss line
    ts="$(stamp)"
    remote="$HOME_TERMUX/.senses_cam${cam}.jpg"
    local_jpg="$FRAMES/${ts}_cam${cam}.jpg"
    out="$SENSES_DIR/.eye.out"; err="$SENSES_DIR/.eye.err"

    $SENSES_SSH "rm -f '$remote'" >/dev/null 2>&1
    if ! timeout 60 $SENSES_SSH "termux-camera-photo -c $cam '$remote'" >/dev/null 2>&1; then
        say "eye cam$cam: capture failed"
        EYE_NOTE="${EYE_NOTE:+$EYE_NOTE,}cam$cam:capture"
        return 1
    fi
    if [ ! -s "$remote" ]; then
        say "eye cam$cam: no frame at $remote"
        EYE_NOTE="${EYE_NOTE:+$EYE_NOTE,}cam$cam:empty"
        return 1
    fi

    # Scale the longest edge down, then let go of the camera's own file.
    if ! ffmpeg -v error -y -i "$remote" \
         -vf "scale='if(gt(iw,ih),$SENSES_EYE_EDGE,-2)':'if(gt(iw,ih),-2,$SENSES_EYE_EDGE)'" \
         -q:v 3 "$local_jpg" 2>/dev/null; then
        say "eye cam$cam: could not scale the frame"
        rm -f "$remote"
        EYE_NOTE="${EYE_NOTE:+$EYE_NOTE,}cam$cam:scale"
        return 1
    fi
    rm -f "$remote"
    EYE_FRAMES=$((EYE_FRAMES + 1))

    t0="$(date -u +%s)"
    SMOLVLM_NOSPLIT=1 EYE_MODEL="$SENSES_EYE_MODEL" EYE_MMPROJ="$SENSES_EYE_MMPROJ" \
        /usr/bin/time -v taskset -c "$CPUS" bash "$SENSES_EYE" "$local_jpg" "$SENSES_EYE_PROMPT" \
        > "$out" 2> "$err"
    rc=$?
    t1="$(date -u +%s)"
    EYE_WALL=$((EYE_WALL + t1 - t0))
    rss="$(awk '/Maximum resident set size/{print $NF}' "$err")"
    case "${rss:-}" in ''|*[!0-9]*) rss=0 ;; esac
    rss=$(( (rss + 512) / 1024 ))
    [ "$rss" -gt "$EYE_RSS" ] && EYE_RSS="$rss"

    if [ "$rc" -ne 0 ]; then
        say "eye cam$cam: engine rc=$rc"
        sed 's/^/[senses]   /' "$err" | head -5
        EYE_RC="$rc"
        EYE_NOTE="${EYE_NOTE:+$EYE_NOTE,}cam$cam:rc$rc"
        return 1
    fi

    # `OURS: " ... "` is the engine's answer; strip the label and the quotes,
    # then one fragment per sentence.
    local said
    said="$(sed -n 's/^OURS: *//p' "$out" | head -1 | sed 's/^"//; s/"$//')"
    said="$(printf '%s' "$said" | sed 's/^ *//; s/ *$//')"
    if [ -z "$said" ]; then
        say "eye cam$cam: the engine said nothing"
        EYE_NOTE="${EYE_NOTE:+$EYE_NOTE,}cam$cam:silent"
        return 1
    fi
    while IFS= read -r line; do
        line="$(printf '%s' "$line" | sed 's/^ *//; s/ *$//')"
        [ "${#line}" -ge 8 ] || continue
        frag_write world "[eye cam$cam $(now_iso)] $line" && EYE_FRAGS=$((EYE_FRAGS + 1))
    done <<< "$(printf '%s\n' "$said" | sed -E 's/([.!?]) +/\1\n/g')"
    return 0
}

do_eye() {
    local mem cam
    mkdir -p "$FRAMES" || return 1
    if [ ! -x "$SENSES_EYE" ]; then
        say "eye: no wrapper at $SENSES_EYE"
        EYE_RC=127; EYE_NOTE="no-engine"; return 1
    fi
    mem="$(mem_avail_mb)"
    if [ "$mem" -lt "$SENSES_EYE_MIN_MB" ]; then
        say "eye: MemAvailable ${mem} MB < ${SENSES_EYE_MIN_MB} MB — the eye does not open"
        EYE_RC=0; EYE_NOTE="skip-mem:${mem}"
        return 0
    fi
    # Both cameras every pass: cam0 looks at the room the phone lies in, cam1
    # at the ceiling above it. Two frames cost ~40 s together, which the slot
    # affords, and one of them is usually in the dark — taking both means the
    # pass still sees something when one of them is black.
    for cam in $SENSES_EYE_CAMS; do
        eye_one "$cam"
    done
    prune_dir "$FRAMES" "$SENSES_KEEP"
    prune_dir "$RUN/dna/output/world" "$SENSES_FRAG_KEEP"
    return 0
}

# --- the ears --------------------------------------------------------------
EARS_RC=0; EARS_WALL=0; EARS_FRAGS=0; EARS_SPEECH=no; EARS_NOTE=""

do_ears() {
    local ts remote wav txt rc t0 t1 clean
    mkdir -p "$AUDIO" || return 1
    if [ ! -x "$SENSES_ASR" ]; then
        say "ears: no recognizer at $SENSES_ASR"
        EARS_RC=127; EARS_NOTE="no-engine"; return 1
    fi
    t0="$(date -u +%s)"
    ts="$(stamp)"
    remote="$HOME_TERMUX/.senses_mic.aac"
    wav="$AUDIO/${ts}.wav"

    $SENSES_SSH "termux-microphone-record -q; rm -f '$remote'" >/dev/null 2>&1
    if ! timeout 30 $SENSES_SSH \
         "termux-microphone-record -f '$remote' -l $SENSES_REC_SECONDS -e aac -r 16000 -c 1" \
         >/dev/null 2>&1; then
        say "ears: the recorder refused to start"
        EARS_RC=1; EARS_NOTE="record"; EARS_WALL=$(( $(date -u +%s) - t0 )); return 1
    fi
    sleep $((SENSES_REC_SECONDS + 2))
    $SENSES_SSH "termux-microphone-record -q" >/dev/null 2>&1

    if [ ! -s "$remote" ]; then
        say "ears: nothing was recorded"
        EARS_RC=1; EARS_NOTE="empty"; EARS_WALL=$(( $(date -u +%s) - t0 )); return 1
    fi
    if ! ffmpeg -v error -y -i "$remote" -ar 16000 -ac 1 -c:a pcm_s16le "$wav" 2>/dev/null; then
        say "ears: could not convert to 16 kHz mono"
        rm -f "$remote"
        EARS_RC=1; EARS_NOTE="convert"; EARS_WALL=$(( $(date -u +%s) - t0 )); return 1
    fi
    rm -f "$remote"

    txt="$(taskset -c "$CPUS" "$SENSES_ASR" -m "$SENSES_ASR_MODEL" -f "$wav" \
           $SENSES_ASR_ARGS 2>/dev/null)"
    rc=$?
    t1="$(date -u +%s)"
    EARS_WALL=$((t1 - t0))
    if [ "$rc" -ne 0 ]; then
        say "ears: recognizer rc=$rc"
        EARS_RC="$rc"; EARS_NOTE="rc$rc"
        prune_dir "$AUDIO" "$SENSES_KEEP"
        return 1
    fi

    # Room noise does not become a sentence. Bracketed tags ([Motor], (wind),
    # *music*) are what the recognizer emits when it hears something that is
    # not speech; with those gone, what is left must still be long enough to
    # be a sentence and must contain a letter.
    clean="$(printf '%s' "$txt" \
        | sed -E 's/\[[^]]*\]//g; s/\([^)]*\)//g; s/\*[^*]*\*//g' \
        | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
    if [ "${#clean}" -lt "$SENSES_SPEECH_MIN_CHARS" ] || ! printf '%s' "$clean" | grep -q '[[:alpha:]]'; then
        say "ears: no speech in ${SENSES_REC_SECONDS}s"
        EARS_SPEECH=no
        prune_dir "$AUDIO" "$SENSES_KEEP"
        return 0
    fi
    EARS_SPEECH=yes
    frag_write sound "[ears mic $(now_iso)] $clean" && EARS_FRAGS=$((EARS_FRAGS + 1))
    prune_dir "$AUDIO" "$SENSES_KEEP"
    prune_dir "$RUN/dna/output/sound" "$SENSES_FRAG_KEEP"
    return 0
}

# --- place -----------------------------------------------------------------
PLACE_RC=0; PLACE_WALL=0; PLACE_FRAGS=0; PLACE_MOVED=unknown; PLACE_NOTE=""

# The WMO present-weather code as a word. Anything unlisted is named by number
# rather than guessed at.
sky_word() {
    case "$1" in
        0) echo "clear sky" ;;
        1) echo "mainly clear" ;;
        2) echo "partly cloudy" ;;
        3) echo "overcast" ;;
        45|48) echo "fog" ;;
        51|53|55) echo "drizzle" ;;
        56|57) echo "freezing drizzle" ;;
        61|63|65) echo "rain" ;;
        66|67) echo "freezing rain" ;;
        71|73|75|77) echo "snow" ;;
        80|81|82) echo "rain showers" ;;
        85|86) echo "snow showers" ;;
        95) echo "thunderstorm" ;;
        96|99) echo "thunderstorm with hail" ;;
        *) echo "weather code $1" ;;
    esac
}

# metres between two WGS84 points, haversine, rounded.
dist_m() {
    awk -v a1="$1" -v o1="$2" -v a2="$3" -v o2="$4" 'BEGIN{
        r=6371000; p=atan2(0,-1)/180;
        dla=(a2-a1)*p; dlo=(o2-o1)*p;
        s=sin(dla/2)^2 + cos(a1*p)*cos(a2*p)*sin(dlo/2)^2;
        printf "%.0f", 2*r*atan2(sqrt(s), sqrt(1-s));
    }'
}

do_place() {
    local t0 loc lat lon acc meteo geo name tz local_time temp hum wind code sky sunrise sunset
    local plat plon moved d text
    t0="$(date -u +%s)"

    loc="$(timeout 45 $SENSES_SSH 'termux-location -p network -r once' 2>/dev/null)"
    lat="$(printf '%s' "$loc" | jq -r '.latitude // empty' 2>/dev/null)"
    if [ -z "$lat" ]; then
        say "place: no network fix, asking the satellites"
        loc="$(timeout 90 $SENSES_SSH 'termux-location -p gps -r once' 2>/dev/null)"
        lat="$(printf '%s' "$loc" | jq -r '.latitude // empty' 2>/dev/null)"
    fi
    lon="$(printf '%s' "$loc" | jq -r '.longitude // empty' 2>/dev/null)"
    acc="$(printf '%s' "$loc" | jq -r '.accuracy // empty' 2>/dev/null)"
    if [ -z "$lat" ] || [ -z "$lon" ]; then
        say "place: no fix at all"
        PLACE_RC=1; PLACE_NOTE="no-fix"; PLACE_WALL=$(( $(date -u +%s) - t0 )); return 1
    fi

    meteo="$(curl -s --max-time "$SENSES_HTTP_TIMEOUT" \
        "https://api.open-meteo.com/v1/forecast?latitude=$lat&longitude=$lon&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&daily=sunrise,sunset&timezone=auto&forecast_days=1")"
    geo="$(curl -s --max-time "$SENSES_HTTP_TIMEOUT" -A "$SENSES_UA" \
        "https://nominatim.openstreetmap.org/reverse?lat=$lat&lon=$lon&format=json&zoom=14&accept-language=en")"

    tz="$(printf '%s' "$meteo" | jq -r '.timezone // empty' 2>/dev/null)"
    local_time="$(printf '%s' "$meteo" | jq -r '.current.time // empty' 2>/dev/null)"
    temp="$(printf '%s' "$meteo" | jq -r '.current.temperature_2m // empty' 2>/dev/null)"
    hum="$(printf '%s' "$meteo" | jq -r '.current.relative_humidity_2m // empty' 2>/dev/null)"
    wind="$(printf '%s' "$meteo" | jq -r '.current.wind_speed_10m // empty' 2>/dev/null)"
    code="$(printf '%s' "$meteo" | jq -r '.current.weather_code // empty' 2>/dev/null)"
    sunrise="$(printf '%s' "$meteo" | jq -r '.daily.sunrise[0] // empty' 2>/dev/null)"
    sunset="$(printf '%s' "$meteo" | jq -r '.daily.sunset[0] // empty' 2>/dev/null)"
    name="$(printf '%s' "$geo" | jq -r '
        [ (.address.suburb // .address.neighbourhood // empty),
          (.address.city // .address.town // .address.village // empty),
          (.address.country // empty) ] | map(select(. != "")) | join(", ")' 2>/dev/null)"
    [ -n "$name" ] || name="$(printf '%s' "$geo" | jq -r '.display_name // empty' 2>/dev/null)"
    [ -n "$name" ] || { name="an unnamed place"; PLACE_NOTE="${PLACE_NOTE:+$PLACE_NOTE,}no-name"; }
    [ -n "$temp" ] || PLACE_NOTE="${PLACE_NOTE:+$PLACE_NOTE,}no-weather"

    # Did the phone move? The last fix is kept next to the log.
    moved=""
    if [ -r "$LASTF" ]; then
        read -r plat plon _ < "$LASTF" 2>/dev/null || true
        if [ -n "${plat:-}" ] && [ -n "${plon:-}" ]; then
            d="$(dist_m "$plat" "$plon" "$lat" "$lon")"
            if [ "$d" -gt "$SENSES_MOVE_M" ]; then
                PLACE_MOVED=yes
                moved="it has moved ${d} m since the last pass"
            else
                PLACE_MOVED=no
                moved="it has not moved more than ${SENSES_MOVE_M} m since the last pass (${d} m from it)"
            fi
        fi
    fi
    if [ -z "$moved" ]; then
        PLACE_MOVED=first
        moved="this is the first fix the senses have taken here"
    fi
    printf '%s %s %s\n' "$lat" "$lon" "$(date -u +%s)" > "$LASTF"

    sky="$(sky_word "${code:-unknown}")"
    text="[place $(now_iso)] The phone is at ${name}"
    [ -n "$acc" ] && text="$text (fix accurate to $(printf '%.0f' "$acc") m)"
    text="$text. Local time ${local_time:-unknown}${tz:+ ($tz)}"
    if [ -n "$temp" ]; then
        text="$text, ${temp} °C, humidity ${hum}%, wind ${wind} km/h, ${sky}"
    fi
    text="$text. Sunrise ${sunrise:-unknown}, sunset ${sunset:-unknown}. And ${moved}."

    frag_write place "$text" && PLACE_FRAGS=$((PLACE_FRAGS + 1))
    prune_dir "$RUN/dna/output/place" "$SENSES_FRAG_KEEP"
    PLACE_WALL=$(( $(date -u +%s) - t0 ))
    return 0
}

# --- one pass --------------------------------------------------------------
MODE="${1:-all}"
case "$MODE" in eye|ears|place|all) ;; *) echo "usage: senses.sh [eye|ears|place|all]" >&2; exit 2 ;; esac

mkdir -p "$SENSES_DIR" "$FRAMES" "$AUDIO" "$RUN/dna/output" || exit 1
take_lock || exit 0
trap 'rm -rf "$LOCK"' EXIT

if colony_alive; then CPUS="${SENSES_CPUS:-0-3}"; else CPUS="${SENSES_CPUS:-4-7}"; fi
MEM0="$(mem_avail_mb)"
T0="$(date -u +%s)"
say "pass $MODE at $(now_iso): cpu $CPUS, MemAvailable ${MEM0} MB"

case "$MODE" in
    eye)   do_eye ;;
    ears)  do_ears ;;
    place) do_place ;;
    all)   do_eye; do_ears; do_place ;;
esac

T1="$(date -u +%s)"
FRAGS=$((EYE_FRAGS + EARS_FRAGS + PLACE_FRAGS))
LINE="$(now_iso) pass=$MODE cpu=$CPUS mem_mb=${MEM0}->$(mem_avail_mb)"
LINE="$LINE eye=rc${EYE_RC},${EYE_WALL}s,rss${EYE_RSS}mb,frames${EYE_FRAMES},frags${EYE_FRAGS}${EYE_NOTE:+,$EYE_NOTE}"
LINE="$LINE ears=rc${EARS_RC},${EARS_WALL}s,speech${EARS_SPEECH},frags${EARS_FRAGS}${EARS_NOTE:+,$EARS_NOTE}"
LINE="$LINE place=rc${PLACE_RC},${PLACE_WALL}s,moved${PLACE_MOVED},frags${PLACE_FRAGS}${PLACE_NOTE:+,$PLACE_NOTE}"
LINE="$LINE frags=$FRAGS total=$((T1 - T0))s"
printf '%s\n' "$LINE" >> "$LOGF"
say "$LINE"

# The pass itself does not fail the slot: an organ that could not run is a
# field in the line above, and the next slot tries again.
exit 0

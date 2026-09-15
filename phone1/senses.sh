#!/bin/bash
# senses.sh — one pass of the phone's senses, then exit. The colony sleeps most
# of the day; the senses do not. They run in their own short slots and leave
# what they found in the DNA field, so that when an organism wakes at 04:00 the
# world of the last eight hours is already food in ../dna/output/.
#
#   senses.sh [eye|ears|place|all]      (default: all)
#   senses.sh overlap "<a>" "<b>"       the text metric below, for the gate
#
# Three organs, three directories under $MOLEQULA_RUN/dna/output/:
#   eye   -> world/   a short trajectory: SENSES_EYE_WINDOW frames taken
#                     SENSES_EYE_SPACING apart, cameras in the order of
#                     SENSES_EYE_PATTERN, each described by senses/ocelli/eye
#                     (SmolVLM2-500M in C), one fragment per sentence and one
#                     summary line per window.
#   ears  -> sound/   12 s from the microphone through senses/ears (whisper on
#                     notorch) on the tiny weights, one fragment, and only when
#                     there was speech in it.
#   place -> place/   one fragment: where the phone is, what the sky is doing,
#                     and whether it moved since the last pass.
#
# A fragment is gen_<unix>_<seq>.txt, the name dnaListNew orders by, prefixed
# with a bracketed header the organisms read as plain text. Nothing here reads
# the organisms' cursors and nothing here deletes another writer's fragments:
# these directories are food, pruned by the hand that fills them.
#
# Beside every fragment each organ also writes one JSON line into
# $MOLEQULA_RUN/senses/facts.jsonl — the same observation as a fact, with its
# provenance, for the world ledger (world_ledger.go, ROADMAP 10). The fragment
# is prose an organism eats; the fact is what lets the ledger tell a change
# from a repetition. At the end of a pass `molequla --world-ingest --once`
# reads what was appended, files it into world_facts in mesh.db and drops one
# more fragment into world/ for each thing that actually changed. The two paths
# are independent: SENSES_FACTS= turns the sidecar off, SENSES_INGEST= leaves
# the facts for the scheduler, and neither moves anything about the fragments.
#
# Every value below can be overridden from the environment; the ASR binary and
# model are two variables on purpose, and that is what let the native `ears` on
# notorch take over from whisper.cpp — the binary and the weights moved, the
# rest of this file did not. whisper-cli is still reachable through the same
# two variables, see the ears block.
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
# The eye lives in this repo now: senses/ocelli, the pure-C SmolVLM engine,
# built by its own Makefile and run through its `eye` wrapper. Nothing links
# against it — it is a separate process with its own notorch copy, and this
# file only ever executes the wrapper. SENSES_EYE is the one variable that
# names it: the tree this script is in, or the checkout on this phone when
# senses.sh is being run from somewhere else.
SENSES_EYE="${SENSES_EYE:-}"
if [ -z "$SENSES_EYE" ]; then
    if [ -x "$REPO/senses/ocelli/eye" ]; then
        SENSES_EYE="$REPO/senses/ocelli/eye"
    else
        SENSES_EYE="$HOME_TERMUX/arianna/molequla/senses/ocelli/eye"
    fi
fi
SENSES_EYE_MODELS="${SENSES_EYE_MODELS:-$HOME_TERMUX/models/ocelli}"
SENSES_EYE_MODEL="${SENSES_EYE_MODEL:-$SENSES_EYE_MODELS/yent_eye_ours_q6_k.gguf}"
SENSES_EYE_MMPROJ="${SENSES_EYE_MMPROJ:-$SENSES_EYE_MODELS/yent_eye_smolvlm2_lora_v2_mmproj_q8_0.gguf}"
SENSES_EYE_PROMPT="${SENSES_EYE_PROMPT:-Describe this image in one sentence.}"
# A sensing episode is a window, not a sample (molequla_new_logic.md §2). The
# pattern is the camera order, cycled to the length of the window; the window is
# how many frames that pass takes; the spacing is the seconds between the starts
# of two consecutive captures, so a frame that runs longer than the spacing does
# not push the next one back.
#
# The defaults are measured on this phone, cores 4-7, 2026-09-15, two runs each
# (MOLEQULALOG2.md, "the sensing window"): n=1 15-18 s, n=2 46-49 s, n=4 104-107 s,
# peak RSS 1020 MB whatever n is — one eye process per frame, nothing accumulates.
# Novelty was 1.000 at n=1 and n=2 and 0.750 at n=4 in both runs. n=4 is the
# default anyway: at n=2 the two frames come from different cameras and can only
# be new, so 1.000 there is arithmetic and not a discovery, while n=4 puts two
# rear frames a minute apart and one of them repeated in both runs — which is the
# window noticing that the scene held still. 107 s is 18 % of the 600 s slot cap.
# Spacing comes out of the eye's own period: a frame occupies 15-18 s of it, so
# under ~20 s there is no spacing at all; 30 s leaves 12-15 s of world between two
# frames and spreads n=4 over 90 s. SENSES_EYE_CAMS is the old name of the
# pattern and still works.
SENSES_EYE_PATTERN="${SENSES_EYE_PATTERN:-${SENSES_EYE_CAMS:-0 1 0 0}}"
SENSES_EYE_WINDOW="${SENSES_EYE_WINDOW:-4}"
SENSES_EYE_SPACING="${SENSES_EYE_SPACING:-30}"
# Two descriptions this close, token for token, are the same observation.
# Measured on the six windows of 2026-09-15: a rear frame repeating an earlier
# rear frame scores 1.000 (the eye says the same sentence word for word), two
# different scenes from the same camera 0.350-0.368, the two cameras of one
# window 0.154. Nothing measured lands between 0.4 and 1.0, so the threshold sits
# in the middle of the empty space.
SENSES_EYE_SAME="${SENSES_EYE_SAME:-0.8}"
# The eye resizes the longest edge to 2048 before it does anything else
# (senses/ocelli/vision.c:55-66). A 4080x3060 camera jpeg would be decoded to
# 150 MB of float first; 1024 costs 9 MB and, with one global frame, ends up
# at the same 512x512 the tower sees.
SENSES_EYE_EDGE="${SENSES_EYE_EDGE:-1024}"
# The eye peaks near 1 GB. Below this much MemAvailable it does not start.
SENSES_EYE_MIN_MB="${SENSES_EYE_MIN_MB:-1300}"

# --- the ears --------------------------------------------------------------
# The recognizer is molequla's own organ: senses/ears, whisper on notorch, in C,
# gated token for token against whisper.cpp on six rows (senses/ears/EARSLOG.md).
# Weights live outside the repo, beside the eye's, and are whisper.cpp's own
# ggml files — the tiny one by default, the base one one variable away.
#
# whisper-cli remains the fallback through the same two variables: point
# SENSES_ASR at it and the command line switches with the binary's name, because
# the two take their model and their wav differently (ears positionally,
# whisper-cli through -m and -f). SENSES_ASR_KIND forces the choice when the
# binary is named something else.
SENSES_ASR="${SENSES_ASR:-$REPO/senses/ears/ears}"
SENSES_ASR_MODEL="${SENSES_ASR_MODEL:-$HOME_TERMUX/models/ears/ggml-tiny.bin}"
SENSES_ASR_KIND="${SENSES_ASR_KIND:-auto}"
if [ "$SENSES_ASR_KIND" = auto ]; then
    case "$(basename "$SENSES_ASR")" in
        ears) SENSES_ASR_KIND=ears ;;
        *)    SENSES_ASR_KIND=whisper-cli ;;
    esac
fi
# Four threads, language auto, and the same no-speech threshold on both sides:
# `ears --no-speech-thold 0.6` is whisper-cli's `-nth 0.6`. On whisper-cli -sns
# also suppresses non-speech tokens, because base spent 185 s on 8 s of room
# noise and said "[Motor]" (arianna/ears-reference/REFERENCE.md); ears drops the
# whole window instead and prints nothing at all, which is the behaviour wanted
# here and what the ambient_8s rows of its transcript gate check.
if [ -z "${SENSES_ASR_ARGS:-}" ]; then
    case "$SENSES_ASR_KIND" in
        ears) SENSES_ASR_ARGS="-l auto -t 4 --no-speech-thold 0.6" ;;
        *)    SENSES_ASR_ARGS="-t 4 -l auto -nth 0.6 -sns -nt -np" ;;
    esac
fi
SENSES_REC_SECONDS="${SENSES_REC_SECONDS:-12}"
# Shorter than this, after the noise tags are stripped, is not speech.
SENSES_SPEECH_MIN_CHARS="${SENSES_SPEECH_MIN_CHARS:-8}"

# --- place -----------------------------------------------------------------
SENSES_MOVE_M="${SENSES_MOVE_M:-50}"
SENSES_UA="${SENSES_UA:-molequla-senses/1.0 (phone-1, Arianna Method)}"
SENSES_HTTP_TIMEOUT="${SENSES_HTTP_TIMEOUT:-25}"

# --- the world ledger ------------------------------------------------------
# The structured sidecar: one JSON line per observation, read by
# `molequla --world-ingest` into the bitemporal world_facts table in mesh.db
# (world_ledger.go). Empty SENSES_FACTS turns the sidecar off and the fragment
# path above is untouched by that. SENSES_INGEST names the binary that reads
# it; it is run once at the end of a pass, from $SENSES_DIR, so that
# ../dna/output and ../senses/facts.jsonl resolve to this run's tree. Empty
# SENSES_INGEST leaves the facts on disk for the scheduler to ingest later.
SENSES_FACTS="${SENSES_FACTS-$SENSES_DIR/facts.jsonl}"
SENSES_FACTS_MAX_KB="${SENSES_FACTS_MAX_KB:-4096}"
if [ -z "${SENSES_INGEST+x}" ]; then
    if [ -x "$RUN/molequla_cgo" ]; then
        SENSES_INGEST="$RUN/molequla_cgo"
    elif [ -x "$REPO/molequla_cgo" ]; then
        SENSES_INGEST="$REPO/molequla_cgo"
    else
        SENSES_INGEST=""
    fi
fi

# --- shared ----------------------------------------------------------------
SENSES_KEEP="${SENSES_KEEP:-48}"          # frames and wavs kept on disk
SENSES_FRAG_KEEP="${SENSES_FRAG_KEEP:-64}" # fragments kept per source directory
SENSES_SSH="${SENSES_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=8 -i /root/.ssh/id_ed25519 -p 8022 u0_a327@localhost}"
NAMES="earth air water fire witness"

now_iso() { date -u +%FT%TZ; }
stamp()   { date -u +%Y%m%dT%H%M%SZ; }
say()     { echo "[senses] $*"; }

mem_avail_mb() { awk '/^MemAvailable:/{printf "%.0f", $2/1024}' /proc/meminfo; }

# Battery, straight off sysfs — the same two files termux-battery-status reads,
# without the round trip into Android. Charge in percent, current in mA (the
# kernel reports µA and signs it by direction of flow).
batt_pct() { cat /sys/class/power_supply/battery/capacity 2>/dev/null || echo ""; }
batt_ma()  { awk '{printf "%.0f", $1/1000}' /sys/class/power_supply/battery/current_now 2>/dev/null || echo ""; }

# text_overlap <a> <b> -> 0.000..1.000, how much of two descriptions is the same
# words. Lowercased, everything that is not a letter or a digit is a separator,
# and the score is the intersection of the distinct tokens over their union — 1
# for the same sentence however it is punctuated, 0 for two sentences with no
# word in common. This is what decides whether a frame of a window repeated an
# earlier frame of the same window; `senses.sh overlap a b` is the same code the
# gate drives.
text_overlap() {
    awk -v a="$1" -v b="$2" '
    function norm(s,   t) {
        t = tolower(s); gsub(/[^a-z0-9]+/, " ", t);
        gsub(/^ +/, "", t); gsub(/ +$/, "", t); return t
    }
    BEGIN {
        na = split(norm(a), A, " "); nb = split(norm(b), B, " ")
        for (i = 1; i <= na; i++) if (A[i] != "") SA[A[i]] = 1
        for (i = 1; i <= nb; i++) if (B[i] != "") SB[B[i]] = 1
        inter = 0; uni = 0
        for (k in SA) { uni++; if (k in SB) inter++ }
        for (k in SB) if (!(k in SA)) uni++
        if (uni == 0) { printf "1.000\n"; exit }   # two silences are one silence
        printf "%.3f\n", inter / uni
    }'
}

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

# --- facts -----------------------------------------------------------------
# The sidecar of every fragment. A fragment is prose for an organism to eat; a
# fact is the same observation with a subject, a predicate and its provenance,
# so that the ledger can tell a change from a repetition. jq builds the line,
# which is the point of using it here: the line is valid JSON or it is nothing,
# and no amount of quoting in a camera sentence can break the file.
#
# rotate_facts keeps the file bounded. Truncating it is safe for the reader:
# the ingest cursor is a byte offset, a file shorter than the offset is read as
# rotated and the cursor returns to zero, and re-reading facts that are already
# in the table writes no row and emits no fragment.
rotate_facts() {
    local bytes kb
    [ -n "$SENSES_FACTS" ] || return 0
    [ -f "$SENSES_FACTS" ] || return 0
    [ "$SENSES_FACTS_MAX_KB" -gt 0 ] || return 0
    bytes="$(stat -c %s "$SENSES_FACTS" 2>/dev/null || echo 0)"
    kb=$((bytes / 1024))
    [ "$kb" -lt "$SENSES_FACTS_MAX_KB" ] && return 0
    mv -f "$SENSES_FACTS" "$SENSES_FACTS.1" && say "facts.jsonl rotated at ${kb} KB"
    return 0
}

# fact_emit <source> <subject> <predicate> <object> [provenance-json]
fact_emit() {
    local src="$1" subj="$2" pred="$3" obj="$4" prov="${5:-}"
    [ -n "$SENSES_FACTS" ] || return 0
    [ -n "$prov" ] || prov='{}'
    mkdir -p "$(dirname "$SENSES_FACTS")" || return 1
    rotate_facts
    jq -cn --arg source "$src" --arg subject "$subj" --arg predicate "$pred" \
       --arg object "$obj" --arg valid_from "$(now_iso)" --argjson provenance "$prov" \
       '{source:$source,subject:$subject,predicate:$predicate,object:$object,
         valid_from:$valid_from,provenance:$provenance}' >> "$SENSES_FACTS"
}

# cam_lens <camera-id> — the subject a camera id is in the ledger. On this
# phone termux-camera-photo -c 0 is the back camera and -c 1 the front one.
cam_lens() {
    case "$1" in
        0) echo "rear camera" ;;
        1) echo "front camera" ;;
        *) echo "camera $1" ;;
    esac
}

# The conditions a pass ran under — §6 stores the interpretation together with
# what made it possible, so that a later correction can see why the first
# reading was reasonable.
conditions_json() {
    jq -cn --arg cpus "${CPUS:-unknown}" \
       --arg colony "$(colony_alive && echo awake || echo asleep)" \
       --argjson mem_mb "$(mem_avail_mb)" \
       '{cpus:$cpus,colony:$colony,mem_mb:$mem_mb}'
}

# eye_prov <camera-id> <frame> <wall-s> <rss-mb>
eye_prov() {
    jq -cn --arg camera "$1" --arg lens "$(cam_lens "$1")" \
       --arg frame "$(basename "$2")" \
       --arg model "$(basename "$SENSES_EYE_MODEL")" \
       --arg mmproj "$(basename "$SENSES_EYE_MMPROJ")" \
       --arg prompt "$SENSES_EYE_PROMPT" --arg engine "$(basename "$SENSES_EYE")" \
       --argjson edge "$SENSES_EYE_EDGE" --argjson wall_s "$3" --argjson rss_mb "$4" \
       --argjson conditions "$(conditions_json)" \
       '{camera:$camera,lens:$lens,frame:$frame,model:$model,mmproj:$mmproj,
         prompt:$prompt,engine:$engine,edge:$edge,wall_s:$wall_s,rss_mb:$rss_mb,
         conditions:$conditions}'
}

# ears_prov <transcript> <rc>
ears_prov() {
    jq -cn --arg asr "$(basename "$SENSES_ASR")" \
       --arg model "$(basename "$SENSES_ASR_MODEL")" \
       --arg kind "$SENSES_ASR_KIND" --arg text "$1" \
       --argjson window_s "$SENSES_REC_SECONDS" --argjson rc "$2" \
       --argjson conditions "$(conditions_json)" \
       '{asr:$asr,model:$model,kind:$kind,window_s:$window_s,text:$text,rc:$rc,
         conditions:$conditions}'
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
EYE_SAID=()          # one description per frame of this window, in order
EYE_REPEAT=0         # frames whose description repeated an earlier one
EYE_NOVEL=""         # share of the descriptions that were new

# eye_one <camera-id> -> frames, fragments, wall, peak RSS
eye_one() {
    local cam="$1" ts remote local_jpg out err rc t0 t1 wall rss line
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
    wall=$((t1 - t0))
    EYE_WALL=$((EYE_WALL + wall))
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
    EYE_SAID+=("$said")
    while IFS= read -r line; do
        line="$(printf '%s' "$line" | sed 's/^ *//; s/ *$//')"
        [ "${#line}" -ge 8 ] || continue
        frag_write world "[eye cam$cam $(now_iso)] $line" && EYE_FRAGS=$((EYE_FRAGS + 1))
    done <<< "$(printf '%s\n' "$said" | sed -E 's/([.!?]) +/\1\n/g')"
    # One fact for the whole answer, not one per sentence: what the engine
    # interpreted is one interpretation. `interpreted_as`, never `is` — §6, the
    # balcony Ocelli read as a bathroom is a true record of a reading and a
    # false record of a room.
    fact_emit eye "$(cam_lens "$cam")" interpreted_as "$said" "$(eye_prov "$cam" "$local_jpg" "$wall" "$rss")"
    return 0
}

# How much of the window was new. A frame repeats when its description overlaps
# an earlier frame of the same window by SENSES_EYE_SAME or more; novelty is the
# share of the descriptions that did not. One frame is trivially all new, which
# is exactly why the number only means something across a window.
eye_novelty() {
    local i j o
    EYE_REPEAT=0; EYE_NOVEL=""
    [ "${#EYE_SAID[@]}" -gt 0 ] || return 0
    for ((i = 1; i < ${#EYE_SAID[@]}; i++)); do
        for ((j = 0; j < i; j++)); do
            o="$(text_overlap "${EYE_SAID[$i]}" "${EYE_SAID[$j]}")"
            if awk -v o="$o" -v t="$SENSES_EYE_SAME" 'BEGIN{exit !(o >= t)}'; then
                EYE_REPEAT=$((EYE_REPEAT + 1)); break
            fi
        done
    done
    EYE_NOVEL="$(awk -v n="${#EYE_SAID[@]}" -v r="$EYE_REPEAT" 'BEGIN{printf "%.3f", (n - r) / n}')"
}

do_eye() {
    local mem cam i n t_start waited b0 b1 c0 c1 t0 t1
    local -a pat
    mkdir -p "$FRAMES" || return 1
    if [ ! -x "$SENSES_EYE" ]; then
        say "eye: no wrapper at $SENSES_EYE"
        EYE_RC=127; EYE_NOTE="no-engine"; return 1
    fi
    read -r -a pat <<< "$SENSES_EYE_PATTERN"
    if [ "${#pat[@]}" -eq 0 ]; then
        say "eye: SENSES_EYE_PATTERN is empty — no camera to open"
        EYE_RC=2; EYE_NOTE="no-pattern"; return 1
    fi
    n="$SENSES_EYE_WINDOW"
    case "$n" in ''|*[!0-9]*) n=${#pat[@]} ;; esac
    [ "$n" -gt 0 ] || { EYE_NOTE="window0"; return 0; }

    # The window: n frames, the pattern cycled, each capture held back to the
    # spacing. cam0 looks at the room the phone lies in, cam1 at the ceiling
    # above it, and the default pattern takes the rear camera three times out of
    # four because that is the one with the scene in it. The memory floor is
    # re-read before every frame and not once before the window: the eye holds
    # about a gigabyte while it runs, and the colony can wake into the gap
    # between two frames.
    t0="$(date -u +%s)"; b0="$(batt_pct)"; c0="$(batt_ma)"
    for ((i = 0; i < n; i++)); do
        mem="$(mem_avail_mb)"
        if [ "$mem" -lt "$SENSES_EYE_MIN_MB" ]; then
            say "eye: MemAvailable ${mem} MB < ${SENSES_EYE_MIN_MB} MB — the window stops at frame $i"
            EYE_NOTE="${EYE_NOTE:+$EYE_NOTE,}skip-mem:${mem}"
            break
        fi
        cam="${pat[$((i % ${#pat[@]}))]}"
        t_start="$(date -u +%s)"
        eye_one "$cam"
        if [ $((i + 1)) -lt "$n" ] && [ "$SENSES_EYE_SPACING" -gt 0 ]; then
            waited=$(( $(date -u +%s) - t_start ))
            [ "$waited" -lt "$SENSES_EYE_SPACING" ] && sleep $((SENSES_EYE_SPACING - waited))
        fi
    done
    t1="$(date -u +%s)"; b1="$(batt_pct)"; c1="$(batt_ma)"
    eye_novelty

    # One line per window, beside the pass line: what the trajectory cost and
    # how much of it was not a repetition of itself.
    printf '%s eyewin pattern=%s n=%s spacing=%ss frames=%s said=%s repeat=%s novel=%s wall=%ss rss=%smb batt=%s%%->%s%%,%s->%smA cpu=%s\n' \
        "$(now_iso)" "$(printf '%s' "$SENSES_EYE_PATTERN" | tr ' ' ',')" "$n" \
        "$SENSES_EYE_SPACING" "$EYE_FRAMES" "${#EYE_SAID[@]}" "$EYE_REPEAT" \
        "${EYE_NOVEL:-none}" "$((t1 - t0))" "$EYE_RSS" \
        "${b0:-?}" "${b1:-?}" "${c0:-?}" "${c1:-?}" "$CPUS" >> "$LOGF"

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

    # Both engines put the transcript, and nothing else, on stdout: ears keeps
    # its per-window no_speech/avg_logprob lines on stderr, whisper-cli is told
    # to with -nt -np. So neither needs a parser, only the right argument order
    # — which is why no --json or -q was added to ears.c for this.
    case "$SENSES_ASR_KIND" in
        ears) txt="$(taskset -c "$CPUS" "$SENSES_ASR" "$SENSES_ASR_MODEL" "$wav" \
                     $SENSES_ASR_ARGS 2>/dev/null)" ;;
        *)    txt="$(taskset -c "$CPUS" "$SENSES_ASR" -m "$SENSES_ASR_MODEL" -f "$wav" \
                     $SENSES_ASR_ARGS 2>/dev/null)" ;;
    esac
    rc=$?
    t1="$(date -u +%s)"
    EARS_WALL=$((t1 - t0))
    if [ "$rc" -ne 0 ]; then
        say "ears: recognizer rc=$rc"
        EARS_RC="$rc"; EARS_NOTE="rc$rc"
        prune_dir "$AUDIO" "$SENSES_KEEP"
        return 1
    fi

    # Room noise does not become a sentence. ears suppresses a whole window on
    # its own no-speech probability and prints an empty line for it, so most of
    # this is the belt to that brace — and the brace whisper-cli does not have.
    # Bracketed tags ([Motor], (wind), *music*) are what a recognizer emits when
    # it hears something that is not speech; with those gone, what is left must
    # still be long enough to be a sentence and must contain a letter.
    clean="$(printf '%s' "$txt" \
        | sed -E 's/\[[^]]*\]//g; s/\([^)]*\)//g; s/\*[^*]*\*//g' \
        | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
    if [ "${#clean}" -lt "$SENSES_SPEECH_MIN_CHARS" ] || ! printf '%s' "$clean" | grep -q '[[:alpha:]]'; then
        say "ears: no speech in ${SENSES_REC_SECONDS}s"
        EARS_SPEECH=no
        # Silence is a fact too, and the only way the ledger can later say that
        # speech stopped. It writes no fragment on its own — a microphone that
        # has always been quiet is not news (world_ledger.go).
        fact_emit ears microphone hearing silence "$(ears_prov "" "$rc")"
        prune_dir "$AUDIO" "$SENSES_KEEP"
        return 0
    fi
    EARS_SPEECH=yes
    frag_write sound "[ears mic $(now_iso)] $clean" && EARS_FRAGS=$((EARS_FRAGS + 1))
    fact_emit ears microphone hearing speech "$(ears_prov "$clean" "$rc")"
    fact_emit ears microphone interpreted_as "$clean" "$(ears_prov "$clean" "$rc")"
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
    local plat plon moved d text provider
    t0="$(date -u +%s)"

    provider=network
    loc="$(timeout 45 $SENSES_SSH 'termux-location -p network -r once' 2>/dev/null)"
    lat="$(printf '%s' "$loc" | jq -r '.latitude // empty' 2>/dev/null)"
    if [ -z "$lat" ]; then
        say "place: no network fix, asking the satellites"
        provider=gps
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

    # Three facts out of one pass, because a moment has more than one kind of
    # truth in it (§15): where the phone is by name, where it is by coordinate,
    # and what the sky over it is doing. The name and the coordinate are
    # separate predicates on purpose — a walk across a neighbourhood changes
    # the coordinate and not the name, and the ledger should be able to say so.
    fact_emit place phone at_place "$name" \
        "$(jq -cn --arg lat "$lat" --arg lon "$lon" --arg accuracy_m "${acc:-}" \
            --arg provider "$provider" --arg moved_m "${d:-}" --arg geocoder nominatim \
            '{lat:$lat,lon:$lon,accuracy_m:$accuracy_m,provider:$provider,
              moved_m:$moved_m,geocoder:$geocoder} | with_entries(select(.value != ""))')"
    fact_emit place phone at_position "$lat,$lon" \
        "$(jq -cn --arg accuracy_m "${acc:-}" --arg provider "$provider" \
            --arg moved_m "${d:-}" --arg place "$name" \
            '{accuracy_m:$accuracy_m,provider:$provider,moved_m:$moved_m,place:$place}
             | with_entries(select(.value != ""))')"
    [ -n "$temp" ] && fact_emit place sky reported_as "$sky" \
        "$(jq -cn --arg weather_code "${code:-}" --arg temp_c "${temp:-}" \
            --arg humidity "${hum:-}" --arg wind_kmh "${wind:-}" \
            --arg local_time "${local_time:-}" --arg tz "${tz:-}" \
            --arg sunrise "${sunrise:-}" --arg sunset "${sunset:-}" \
            --arg place "$name" --arg source open-meteo \
            '{weather_code:$weather_code,temp_c:$temp_c,humidity:$humidity,
              wind_kmh:$wind_kmh,local_time:$local_time,tz:$tz,sunrise:$sunrise,
              sunset:$sunset,place:$place,source:$source}
             | with_entries(select(.value != ""))')"

    prune_dir "$RUN/dna/output/place" "$SENSES_FRAG_KEEP"
    PLACE_WALL=$(( $(date -u +%s) - t0 ))
    return 0
}

# --- the ledger's writer ----------------------------------------------------
# One pass of `molequla --world-ingest --once`, run from $SENSES_DIR so that
# ../dna/output and ../senses/facts.jsonl are this run's. It reads the facts
# written above, files them into world_facts and leaves in dna/output/world/
# only what changed. It is a separate process because the witness's mesh
# handle is query_only and stays that way; failing here costs the pass
# nothing, the facts stay on disk and the next run of the ingest picks them up
# from the cursor.
INGEST_RC=0; INGEST_LINES=0
do_ingest() {
    local out
    [ -n "$SENSES_INGEST" ] || return 0
    [ -n "$SENSES_FACTS" ] || return 0
    [ -x "$SENSES_INGEST" ] || { say "ingest: no binary at $SENSES_INGEST"; INGEST_RC=127; return 0; }
    [ -s "$SENSES_FACTS" ] || return 0
    out="$(cd "$SENSES_DIR" && timeout 120 "$SENSES_INGEST" --world-ingest --once \
           --world-facts "$SENSES_FACTS" 2>&1)"
    INGEST_RC=$?
    printf '%s\n' "$out" | sed 's/^/[senses]   /'
    INGEST_LINES="$(printf '%s\n' "$out" | grep -c '^\[world\] ' || true)"
    return 0
}

# --- sourceable -------------------------------------------------------------
# Everything above is definitions. A caller that wants only the functions —
# phone1/senses_facts_test.sh drives fact_emit over a fixture rather than
# waking the camera, the microphone and the GPS — sources this file with
# SENSES_LIB_ONLY=1 and stops here.
if [ "${SENSES_LIB_ONLY:-0}" = 1 ]; then
    return 0 2>/dev/null || exit 0
fi

# --- one pass --------------------------------------------------------------
MODE="${1:-all}"
# The metric, before the lock and before any directory is made: `overlap` takes
# no camera, writes nothing, and exists so that the gate drives the same awk the
# window does rather than a copy of it.
if [ "$MODE" = overlap ]; then
    [ $# -eq 3 ] || { echo 'usage: senses.sh overlap "<a>" "<b>"' >&2; exit 2; }
    text_overlap "$2" "$3"
    exit 0
fi
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

do_ingest

T1="$(date -u +%s)"
FRAGS=$((EYE_FRAGS + EARS_FRAGS + PLACE_FRAGS))
LINE="$(now_iso) pass=$MODE cpu=$CPUS mem_mb=${MEM0}->$(mem_avail_mb)"
LINE="$LINE eye=rc${EYE_RC},${EYE_WALL}s,rss${EYE_RSS}mb,frames${EYE_FRAMES},frags${EYE_FRAGS}${EYE_NOVEL:+,novel$EYE_NOVEL}${EYE_NOTE:+,$EYE_NOTE}"
LINE="$LINE ears=rc${EARS_RC},${EARS_WALL}s,speech${EARS_SPEECH},frags${EARS_FRAGS}${EARS_NOTE:+,$EARS_NOTE}"
LINE="$LINE place=rc${PLACE_RC},${PLACE_WALL}s,moved${PLACE_MOVED},frags${PLACE_FRAGS}${PLACE_NOTE:+,$PLACE_NOTE}"
LINE="$LINE world=rc${INGEST_RC},changes${INGEST_LINES}"
LINE="$LINE frags=$FRAGS total=$((T1 - T0))s"
printf '%s\n' "$LINE" >> "$LOGF"
say "$LINE"

# The pass itself does not fail the slot: an organ that could not run is a
# field in the line above, and the next slot tries again.
exit 0

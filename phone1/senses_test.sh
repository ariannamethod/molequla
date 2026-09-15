#!/bin/bash
# Gate for the sensing window: the eye's short trajectory, the text-overlap
# metric that decides whether a frame repeated an earlier one, and the two
# fragments hearing leaves — the transcript when somebody spoke and the
# environmental line when nobody did.
#
# Every case drives the real phone1/senses.sh. What is faked is only the
# hardware beyond it: an `ssh` that copies fixture files where termux-camera-photo
# and termux-microphone-record would have written them, an `eye` that prints a
# scripted sentence, a recognizer that prints a scripted transcript, and a sound
# describer that prints a scripted label. ffmpeg is real — the scaling and the
# 16 kHz conversion are part of the path under test.
#
#   bash phone1/senses_test.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSES="$HERE/senses.sh"
# The ledger ingest is the other organ's gate (senses_facts_test.sh): here it is
# off, so a molequla_cgo built beside the checkout cannot add change lines to
# world/ and turn "four frames, four fragments" into six.
export SENSES_INGEST=""
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0

ok()   { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; }

# eq <name> <got> <want>
eq() {
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1: got '$2', want '$3'"; fi
}

command -v ffmpeg >/dev/null 2>&1 || { echo "senses_test: ffmpeg is needed for the frame path" >&2; exit 2; }

# --- fixtures --------------------------------------------------------------
ffmpeg -v error -y -f lavfi -i color=c=gray:s=64x64 -frames:v 1 "$TMP/frame.jpg" 2>/dev/null \
    || { echo "senses_test: could not make a fixture frame" >&2; exit 2; }
ffmpeg -v error -y -f lavfi -i "sine=frequency=440:duration=1" -ar 16000 -ac 1 -c:a pcm_s16le "$TMP/mic.wav" 2>/dev/null \
    || { echo "senses_test: could not make a fixture wav" >&2; exit 2; }

# The phone, faked: one argument, the command line senses.sh would have run in
# Termux. Only the three calls this gate needs are answered.
cat > "$TMP/ssh.sh" <<EOF
#!/bin/bash
cmd="\$*"
path="\$(printf '%s' "\$cmd" | sed -n "s/.*'\([^']*\)'.*/\1/p")"
case "\$cmd" in
    *termux-camera-photo*)
        cam="\$(printf '%s' "\$cmd" | sed -n 's/.*-c \([0-9]*\).*/\1/p')"
        printf '%s\n' "\$cam" >> "$TMP/cams.seen"
        cp "$TMP/frame.jpg" "\$path" ;;
    *termux-microphone-record\ -q*) ;;
    *termux-microphone-record*) cp "$TMP/mic.wav" "\$path" ;;
    rm\ -f*) [ -n "\$path" ] && rm -f "\$path" ;;
    *) exit 0 ;;
esac
exit 0
EOF

# The eye, faked: one scripted sentence per call, in order, cycling.
cat > "$TMP/eye.sh" <<EOF
#!/bin/bash
n=\$(cat "$TMP/eye.n" 2>/dev/null || echo 0)
n=\$((n + 1)); printf '%d' "\$n" > "$TMP/eye.n"
sed -n "\${n}p" "$TMP/eye.lines"
EOF

cat > "$TMP/eye.lines" <<'EOF'
OURS: "A dark room with a chair and a table."
OURS: "A ceiling with a lamp above the bed."
OURS: "A dark room with a chair and a table."
OURS: "A street with cars and a fence."
EOF

chmod +x "$TMP/ssh.sh" "$TMP/eye.sh"

RUN="$TMP/run"
mkdir -p "$RUN"

# run_eye <window> <pattern> <spacing> — one eye pass on a clean field.
run_eye() {
    rm -rf "$RUN"; mkdir -p "$RUN"
    rm -f "$TMP/cams.seen" "$TMP/eye.n"
    MOLEQULA_RUN="$RUN" \
    SENSES_SSH="bash $TMP/ssh.sh" \
    SENSES_EYE="$TMP/eye.sh" \
    SENSES_EYE_WINDOW="$1" SENSES_EYE_PATTERN="$2" SENSES_EYE_SPACING="$3" \
    SENSES_EYE_MIN_MB=0 SENSES_TERMUX_HOME="$TMP/termux" \
    bash "$SENSES" eye > "$TMP/eye.out" 2>&1
}

mkdir -p "$TMP/termux"

# --- the window is N frames, cameras in the pattern's order ------------------
run_eye 4 "0 1 0 0" 0
frags="$(ls -1 "$RUN/dna/output/world" 2>/dev/null | wc -l)"
eq "a window of four leaves four fragments" "$frags" "4"

# The fragments are ordered by (unix, seq) in their names, which is the order
# dnaListNew hands them to an organism.
got_cams="$(for f in $(ls -1 "$RUN/dna/output/world" 2>/dev/null | sort -t_ -k3,3n); do
                sed -n 's/^\[eye cam\([0-9]*\) .*/\1/p' "$RUN/dna/output/world/$f"; done | tr '\n' ' ')"
eq "the cameras alternate as the pattern says" "${got_cams% }" "0 1 0 0"
eq "the capture order matches it too" "$(tr '\n' ' ' < "$TMP/cams.seen" | sed 's/ $//')" "0 1 0 0"

run_eye 2 "1 0" 0
got_cams="$(for f in $(ls -1 "$RUN/dna/output/world" 2>/dev/null | sort -t_ -k3,3n); do
                sed -n 's/^\[eye cam\([0-9]*\) .*/\1/p' "$RUN/dna/output/world/$f"; done | tr '\n' ' ')"
eq "a window of two follows its own pattern" "${got_cams% }" "1 0"

# A pattern shorter than the window cycles rather than stopping.
run_eye 3 "0 1" 0
got_cams="$(for f in $(ls -1 "$RUN/dna/output/world" 2>/dev/null | sort -t_ -k3,3n); do
                sed -n 's/^\[eye cam\([0-9]*\) .*/\1/p' "$RUN/dna/output/world/$f"; done | tr '\n' ' ')"
eq "a short pattern cycles" "${got_cams% }" "0 1 0"

# --- the window summary line ------------------------------------------------
run_eye 4 "0 1 0 0" 0
line="$(grep 'eyewin' "$RUN/senses/senses.log" 2>/dev/null | tail -1)"
case "$line" in
    *"n=4"*)      ok "the window line carries its size" ;;
    *)            bad "the window line carries its size: got '$line'" ;;
esac
case "$line" in
    *"pattern=0,1,0,0"*) ok "the window line carries the pattern" ;;
    *)                   bad "the window line carries the pattern: got '$line'" ;;
esac
# Frame 3 repeats frame 1 word for word: three of four descriptions are new.
case "$line" in
    *"repeat=1"*) ok "the repeated frame is counted" ;;
    *)            bad "the repeated frame is counted: got '$line'" ;;
esac
case "$line" in
    *"novel=0.750"*) ok "novelty is the share that was new" ;;
    *)               bad "novelty is the share that was new: got '$line'" ;;
esac

# --- spacing ----------------------------------------------------------------
t0="$(date -u +%s)"
run_eye 2 "0 0" 4
t1="$(date -u +%s)"
if [ $((t1 - t0)) -ge 4 ]; then
    ok "spacing holds the second capture back"
else
    bad "spacing holds the second capture back: window took $((t1 - t0))s, want >= 4s"
fi

# --- the overlap metric -----------------------------------------------------
# ov <name> <a> <b> <want>
ov() {
    local got
    got="$(bash "$SENSES" overlap "$2" "$3" 2>&1)"
    eq "$1" "$got" "$4"
}

A="A dark room with a chair, a table, and a blanket, with no people or text visible."
ov "identical strings"     "$A" "$A" "1.000"
ov "disjoint strings"      "one two three" "four five six" "0.000"
ov "case and punctuation do not count" "A Dark Room." "a dark room" "1.000"
ov "nothing against something" "" "$A" "0.000"
# The known near-identical case: the same sentence from the same camera one
# window later, one noun different. Over the tree it must land above the 0.8
# the window calls a repeat.
B="A dark room with a chair, a table, and a pillow, with no people or text visible."
got="$(bash "$SENSES" overlap "$A" "$B")"
if awk -v g="$got" 'BEGIN{exit !(g >= 0.8 && g < 1.0)}'; then
    ok "one noun changed is still the same frame ($got)"
else
    bad "one noun changed is still the same frame: got '$got', want 0.8 <= x < 1.0"
fi

# --- hearing: the transcript and the environment ----------------------------
# say <transcript> <label> — one ears pass with a scripted recognizer and a
# scripted describer.
say() {
    rm -rf "$RUN"; mkdir -p "$RUN"
    printf '#!/bin/bash\nprintf "%%s\\n" %q\n' "$1" > "$TMP/asr.sh"
    printf '#!/bin/bash\nprintf "%%s\\n" %q\n' "$2" > "$TMP/snd.sh"
    chmod +x "$TMP/asr.sh" "$TMP/snd.sh"
    MOLEQULA_RUN="$RUN" \
    SENSES_SSH="bash $TMP/ssh.sh" \
    SENSES_ASR="$TMP/asr.sh" SENSES_ASR_KIND=ears SENSES_ASR_MODEL="$TMP/model.bin" \
    SENSES_SOUNDSCAPE="$TMP/snd.sh" \
    SENSES_REC_SECONDS=1 SENSES_TERMUX_HOME="$TMP/termux" \
    bash "$SENSES" ears > "$TMP/ears.out" 2>&1
}

heads() { for f in "$RUN"/dna/output/sound/*.txt; do [ -e "$f" ] || continue; sed -n 's/^\(\[ears [a-z]*\).*/\1]/p' "$f"; done | sort | tr '\n' ' '; }

say "" "quiet room"
eq "silence still leaves the environment" "$(heads)" "[ears env] "
grep -q '^\[ears env .*\] Quiet room\.$' "$RUN"/dna/output/sound/*.txt 2>/dev/null \
    && ok "the environmental line is the describer's" \
    || bad "the environmental line is the describer's: got '$(cat "$RUN"/dna/output/sound/*.txt 2>/dev/null)'"

say "And so my fellow Americans, ask not what your country can do for you." "speech-like modulation, words unclear"
eq "speech leaves the transcript and the environment" "$(heads)" "[ears env] [ears mic] "

# A recognizer that only found a non-speech tag is not speech — and the tag is
# not thrown away either, because that is the hearing this window is for.
say "[Motor]" "repeated mechanical noise"
eq "a bare non-speech tag is not a transcript" "$(heads)" "[ears env] "
grep -q '\[Motor\]' "$RUN"/dna/output/sound/*.txt 2>/dev/null \
    && ok "the recognizer's own non-speech tag survives" \
    || bad "the recognizer's own non-speech tag survives: got '$(cat "$RUN"/dna/output/sound/*.txt 2>/dev/null)'"

printf '\n%d pass, %d fail\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

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

# run_eye <window> <pattern> <spacing> [NAME=value …] — one eye pass on a clean
# field. Anything after the spacing is environment for that pass, which is how
# the cap cases put a small SENSES_EYE_FRAME_TIMEOUT and a hanging engine under
# the same script the ten cases above drive.
run_eye() {
    local w="$1" pat="$2" sp="$3"; shift 3
    rm -rf "$RUN"; mkdir -p "$RUN"
    rm -f "$TMP/cams.seen" "$TMP/eye.n"
    env MOLEQULA_RUN="$RUN" \
    SENSES_SSH="bash $TMP/ssh.sh" \
    SENSES_EYE="$TMP/eye.sh" \
    SENSES_EYE_WINDOW="$w" SENSES_EYE_PATTERN="$pat" SENSES_EYE_SPACING="$sp" \
    SENSES_EYE_MIN_MB=0 SENSES_TERMUX_HOME="$TMP/termux" \
    "$@" bash "$SENSES" eye > "$TMP/eye.out" 2>&1
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

# --- the caps ---------------------------------------------------------------
# Six commands in the pass are capped and none of them is capped by `timeout`
# any more (MOLEQULALOG2.md, 2026-09-17). The cases below give each cap a
# command that sleeps well past it and a cap of two or three seconds: what they
# check is that the cap fires at its own number, that the pass carries on, and
# that nothing is left behind.

# An engine that hangs on the second frame of the window, after writing half a
# sentence to stdout — which is the half that must not become a fragment. The
# subshell is there so the partial line is flushed before the sleep: a killed
# bash does not flush its own stdio.
cat > "$TMP/eye_hang.sh" <<EOF
#!/bin/bash
n=\$(cat "$TMP/eye.n" 2>/dev/null || echo 0)
n=\$((n + 1)); printf '%d' "\$n" > "$TMP/eye.n"
if [ "\$n" = 2 ]; then
    ( printf 'OURS: "A half-written sen' )
    sleep 60
    exit 0
fi
sed -n "\${n}p" "$TMP/eye.lines"
EOF

# The same hang, one process deeper: a grandchild that outlives its parent
# unless the kill reaches the whole group. The real eye is exactly this shape —
# senses/ocelli/eye runs the C engine in a command substitution.
cat > "$TMP/eye_fork.sh" <<EOF
#!/bin/bash
sleep 600 &
printf '%d' "\$!" > "$TMP/fork.pid"
sleep 60
EOF

# An ssh that hangs on whatever SENSES_TEST_SLOW names and otherwise answers
# the way the phone does.
cat > "$TMP/ssh_slow.sh" <<EOF
#!/bin/bash
case "\$*" in
    *\${SENSES_TEST_SLOW:-__nothing__}*) sleep 60; exit 0 ;;
esac
exec bash "$TMP/ssh.sh" "\$@"
EOF

cat > "$TMP/slow.sh" <<'EOF'
#!/bin/bash
sleep 60
EOF

chmod +x "$TMP/eye_hang.sh" "$TMP/eye_fork.sh" "$TMP/ssh_slow.sh" "$TMP/slow.sh"

# cap_lib <cap> <stdout-file> <command> — cap_run on its own, through the
# library door senses_facts_test.sh uses, so that the helper's own contract is
# checked and not only its call sites.
cap_lib() {
    MOLEQULA_RUN="$RUN" SENSES_LIB_ONLY=1 bash -c '
        . "$1"; mkdir -p "$SENSES_DIR"
        cap_run "$2" "$3" /dev/null "$4"
        printf "rc=%s fired=%s elapsed=%s strays=%s\n" \
            "$CAP_RC" "$CAP_FIRED" "$CAP_ELAPSED" "$CAP_STRAYS"
    ' _ "$SENSES" "$1" "$2" "$3" 2>/dev/null | tail -1
}

rm -rf "$RUN"; mkdir -p "$RUN"
got="$(cap_lib 5 "$TMP/cap.out" 'printf hello; exit 3')"
case "$got" in
    "rc=3 fired=0 elapsed="[01]" strays=0") ok "a command inside its cap keeps its own exit status" ;;
    *) bad "a command inside its cap keeps its own exit status: got '$got'" ;;
esac
eq "and its stdout is in the file the caller named" "$(cat "$TMP/cap.out")" "hello"

t0="$(date -u +%s)"
got="$(cap_lib 2 /dev/null 'sleep 60')"
t1="$(date -u +%s)"
case "$got" in
    "rc=124 fired=1 elapsed="[234]" strays="*) ok "a command past its cap is killed at the cap ($got)" ;;
    *)                                         bad "a command past its cap is killed at the cap: got '$got'" ;;
esac
if [ $((t1 - t0)) -lt 10 ]; then
    ok "and the caller gets it back at once, not after the sleep ($((t1 - t0))s)"
else
    bad "and the caller gets it back at once, not after the sleep: $((t1 - t0))s"
fi

# --- a capped frame costs one frame, not the window -------------------------
t0="$(date -u +%s)"
run_eye 3 "0 1 0" 0 SENSES_EYE="$TMP/eye_hang.sh" SENSES_EYE_FRAME_TIMEOUT=3
t1="$(date -u +%s)"
line="$(grep 'pass=eye' "$RUN/senses/senses.log" 2>/dev/null | tail -1)"
win="$(grep 'eyewin' "$RUN/senses/senses.log" 2>/dev/null | tail -1)"
eq "a hung frame leaves the other two fragments" \
   "$(ls -1 "$RUN/dna/output/world" 2>/dev/null | wc -l)" "2"
case "$win" in
    *"frames=3 said=2"*) ok "the window took its three frames and heard two" ;;
    *)                   bad "the window took its three frames and heard two: got '$win'" ;;
esac
case "$line" in
    *"cam1:timeout"*) ok "the capped frame is named in the pass line" ;;
    *)                bad "the capped frame is named in the pass line: got '$line'" ;;
esac
case "$line" in
    *"eye=rc124,"*) ok "and the pass carries the cap's status" ;;
    *)              bad "and the pass carries the cap's status: got '$line'" ;;
esac
if [ $((t1 - t0)) -lt 25 ]; then
    ok "the window did not wait out the hang ($((t1 - t0))s)"
else
    bad "the window did not wait out the hang: $((t1 - t0))s, want < 25s"
fi
# Half a sentence is not an observation: no fragment, and no fact either.
if grep -rq 'half-written' "$RUN/dna/output" 2>/dev/null; then
    bad "a capped frame writes no fragment: '$(grep -rl 'half-written' "$RUN/dna/output")'"
else
    ok "a capped frame writes no fragment"
fi
eq "and no fact line for it" \
   "$(grep -c '"source":"eye"' "$RUN/senses/facts.jsonl" 2>/dev/null || echo 0)" "2"

# --- nothing outlives a capped frame ----------------------------------------
rm -f "$TMP/fork.pid"
run_eye 1 "0" 0 SENSES_EYE="$TMP/eye_fork.sh" SENSES_EYE_FRAME_TIMEOUT=3
kid="$(cat "$TMP/fork.pid" 2>/dev/null)"
if [ -z "$kid" ]; then
    bad "the forking engine never reported its grandchild"
elif kill -0 "$kid" 2>/dev/null; then
    bad "the engine's grandchild is killed with the group: pid $kid is still alive"
    kill -KILL "$kid" 2>/dev/null
else
    ok "the engine's grandchild is killed with the group"
fi

# --- the camera's own cap ----------------------------------------------------
t0="$(date -u +%s)"
run_eye 1 "0" 0 SENSES_SSH="bash $TMP/ssh_slow.sh" \
    SENSES_TEST_SLOW=termux-camera-photo SENSES_CAM_TIMEOUT=2
t1="$(date -u +%s)"
line="$(grep 'pass=eye' "$RUN/senses/senses.log" 2>/dev/null | tail -1)"
case "$line" in
    *"cam0:capture"*) ok "a camera that does not answer is a capture failure, at its own cap" ;;
    *)                bad "a camera that does not answer is a capture failure: got '$line'" ;;
esac
if [ $((t1 - t0)) -lt 20 ]; then
    ok "the grab's cap fires at two seconds, not sixty ($((t1 - t0))s)"
else
    bad "the grab's cap fires at two seconds, not sixty: $((t1 - t0))s"
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
# say <transcript> <label> [NAME=value …] — one ears pass with a scripted
# recognizer and a scripted describer; anything after the label is environment
# for that pass, the way run_eye takes it.
say() {
    local txt="$1" label="$2"; shift 2
    rm -rf "$RUN"; mkdir -p "$RUN"
    printf '#!/bin/bash\nprintf "%%s\\n" %q\n' "$txt" > "$TMP/asr.sh"
    printf '#!/bin/bash\nprintf "%%s\\n" %q\n' "$label" > "$TMP/snd.sh"
    chmod +x "$TMP/asr.sh" "$TMP/snd.sh"
    env MOLEQULA_RUN="$RUN" \
    SENSES_SSH="bash $TMP/ssh.sh" \
    SENSES_ASR="$TMP/asr.sh" SENSES_ASR_KIND=ears SENSES_ASR_MODEL="$TMP/model.bin" \
    SENSES_SOUNDSCAPE="$TMP/snd.sh" \
    SENSES_REC_SECONDS=1 SENSES_TERMUX_HOME="$TMP/termux" \
    "$@" bash "$SENSES" ears > "$TMP/ears.out" 2>&1
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

# --- hearing's two caps ------------------------------------------------------
# A describer that hangs costs the environmental line and nothing else: the
# transcript of the same twelve seconds is still written.
t0="$(date -u +%s)"
say "And so my fellow Americans, ask not what your country can do for you." "" \
    SENSES_SOUNDSCAPE="$TMP/slow.sh" SENSES_SOUNDSCAPE_TIMEOUT=2
t1="$(date -u +%s)"
eq "a hung describer costs the environment, not the transcript" "$(heads)" "[ears mic] "
if [ $((t1 - t0)) -lt 25 ]; then
    ok "the describer's cap fires at two seconds, not sixty ($((t1 - t0))s)"
else
    bad "the describer's cap fires at two seconds, not sixty: $((t1 - t0))s"
fi

# A recorder that never starts is the organ's own `record` failure, at its cap.
t0="$(date -u +%s)"
say "" "quiet room" SENSES_SSH="bash $TMP/ssh_slow.sh" \
    SENSES_TEST_SLOW=termux-microphone-record SENSES_REC_TIMEOUT=2
t1="$(date -u +%s)"
line="$(grep 'pass=ears' "$RUN/senses/senses.log" 2>/dev/null | tail -1)"
case "$line" in
    *",record"*) ok "a recorder that does not start is logged as such, at its own cap" ;;
    *)           bad "a recorder that does not start is logged as such: got '$line'" ;;
esac
if [ $((t1 - t0)) -lt 20 ]; then
    ok "the recorder's cap fires at two seconds, not thirty ($((t1 - t0))s)"
else
    bad "the recorder's cap fires at two seconds, not thirty: $((t1 - t0))s"
fi

# --- place: two caps, one after the other ------------------------------------
# The pair whose 135 s let 1373 s pass on 2026-09-17. Both fixes hang here and
# both are capped at two seconds, so the branch that convicted `timeout` costs
# four seconds and not forty.
rm -rf "$RUN"; mkdir -p "$RUN"
t0="$(date -u +%s)"
env MOLEQULA_RUN="$RUN" SENSES_SSH="bash $TMP/ssh_slow.sh" \
    SENSES_TEST_SLOW=termux-location SENSES_LOC_TIMEOUT=2 SENSES_LOC_GPS_TIMEOUT=2 \
    SENSES_TERMUX_HOME="$TMP/termux" \
    bash "$SENSES" place > "$TMP/place.out" 2>&1
t1="$(date -u +%s)"
line="$(grep 'pass=place' "$RUN/senses/senses.log" 2>/dev/null | tail -1)"
case "$line" in
    *"no-fix"*) ok "two silent providers are no fix, each at its own cap" ;;
    *)          bad "two silent providers are no fix: got '$line'" ;;
esac
if [ $((t1 - t0)) -lt 20 ]; then
    ok "and the branch costs four seconds, not forty ($((t1 - t0))s)"
else
    bad "and the branch costs four seconds, not forty: $((t1 - t0))s"
fi

# --- the ledger's writer has a cap too ---------------------------------------
t0="$(date -u +%s)"
run_eye 1 "0" 0 SENSES_INGEST="$TMP/slow.sh" SENSES_INGEST_TIMEOUT=2
t1="$(date -u +%s)"
line="$(grep 'pass=eye' "$RUN/senses/senses.log" 2>/dev/null | tail -1)"
case "$line" in
    *"world=rc124"*) ok "an ingest that hangs is capped and says so" ;;
    *)               bad "an ingest that hangs is capped and says so: got '$line'" ;;
esac
if [ $((t1 - t0)) -lt 25 ]; then
    ok "the ingest's cap fires at two seconds, not sixty ($((t1 - t0))s)"
else
    bad "the ingest's cap fires at two seconds, not sixty: $((t1 - t0))s"
fi

printf '\n%d pass, %d fail\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

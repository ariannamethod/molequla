#!/bin/bash
# Gate for the scheduler's slot arithmetic. Every case drives the real
# `schedule.sh next --epoch` with a fake now (MOLEQULA_SCHED_NOW) and an
# explicit slot list, so the thing under test is the code that runs at 04:00,
# not a copy of it. Break next_slot on purpose and this goes red.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHED="$HERE/schedule.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0

ok()  { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; }

# eq <name> <got> <want>
eq() {
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1: got '$2', want '$3'"; fi
}

# e <UTC timestamp> -> epoch
e() { date -u -d "$1" +%s; }

# want <name> <slots> <now> <expected UTC>
want() {
    local name="$1" slots="$2" now="$3" expect="$4" got exp
    exp="$(e "$expect")"
    got="$(MOLEQULA_RUN="$TMP" SCHEDULE_CONF=/dev/null SCHEDULE_SLOTS="$slots" \
           MOLEQULA_SCHED_NOW="$(e "$now")" bash "$SCHED" next --epoch 2>&1)"
    if [ "$got" = "$exp" ]; then
        pass=$((pass + 1)); printf 'ok   %s\n' "$name"
    else
        fail=$((fail + 1))
        printf 'FAIL %s: slots [%s] now %s -> got %s (%s), want %s (%s)\n' \
            "$name" "$slots" "$now" "$got" "$(date -u -d "@$got" +%FT%TZ 2>/dev/null || echo '?')" "$exp" "$expect"
    fi
}

# reject <name> <slots>
reject() {
    local name="$1" slots="$2" out rc
    out="$(MOLEQULA_RUN="$TMP" SCHEDULE_CONF=/dev/null SCHEDULE_SLOTS="$slots" \
           MOLEQULA_SCHED_NOW=1789000000 bash "$SCHED" next --epoch 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ]; then
        pass=$((pass + 1)); printf 'ok   %s\n' "$name"
    else
        fail=$((fail + 1)); printf 'FAIL %s: slots [%s] accepted, printed %s\n' "$name" "$slots" "$out"
    fi
}

D="04:00 12:00 20:00"

want "before the first slot"        "$D" "2026-09-13T03:00:00Z" "2026-09-13T04:00:00Z"
want "one second before a slot"     "$D" "2026-09-13T11:59:59Z" "2026-09-13T12:00:00Z"
want "exactly on the boundary"      "$D" "2026-09-13T04:00:00Z" "2026-09-13T04:00:00Z"
want "one second after a boundary"  "$D" "2026-09-13T04:00:01Z" "2026-09-13T12:00:00Z"
want "between two slots"            "$D" "2026-09-13T12:30:00Z" "2026-09-13T20:00:00Z"
want "after the last slot, tomorrow" "$D" "2026-09-13T20:00:01Z" "2026-09-14T04:00:00Z"
want "a second before midnight"     "$D" "2026-09-13T23:59:59Z" "2026-09-14T04:00:00Z"
want "midnight itself"              "$D" "2026-09-14T00:00:00Z" "2026-09-14T04:00:00Z"
want "month boundary"               "$D" "2026-09-30T23:00:00Z" "2026-10-01T04:00:00Z"
want "unsorted list, same answer"   "20:00 04:00 12:00" "2026-09-13T12:30:00Z" "2026-09-13T20:00:00Z"
want "one slot, wraps to tomorrow"  "00:30" "2026-09-13T00:30:01Z" "2026-09-14T00:30:00Z"
want "one slot, later today"        "00:30" "2026-09-13T00:29:59Z" "2026-09-13T00:30:00Z"
want "leading zeros are base ten"   "08:09" "2026-09-13T08:08:59Z" "2026-09-13T08:09:00Z"
want "last minute of the day"       "23:59" "2026-09-13T23:59:00Z" "2026-09-13T23:59:00Z"

# The slots are UTC whatever the host thinks the time zone is.
export TZ="Asia/Jerusalem"
want "host TZ does not move the slots" "$D" "2026-09-13T03:00:00Z" "2026-09-13T04:00:00Z"
unset TZ

reject "hour 24"           "24:00"
reject "minute 60"         "12:60"
reject "single-digit hour" "9:00"
reject "no colon"          "0400"
reject "one bad among good" "04:00 12:0 20:00"

# An empty slot list can only come from the file: an empty environment
# variable means "not overridden".
printf 'SCHEDULE_SLOTS=""\n' > "$TMP/empty.conf"
if out="$(MOLEQULA_RUN="$TMP" SCHEDULE_CONF="$TMP/empty.conf" MOLEQULA_SCHED_NOW=1789000000 \
          bash "$SCHED" next --epoch 2>&1)"; then
    fail=$((fail + 1)); printf 'FAIL empty conf accepted, printed %s\n' "$out"
else
    pass=$((pass + 1)); printf 'ok   empty slot list in the conf is refused\n'
fi

# ── two kinds of slot ───────────────────────────────────────────────────────
# The colony and the senses share one clock. `next` must name the nearest slot
# of either kind and say which kind it is, and a senses slot that falls inside
# a colony window must be recognisable as one before it is run.

# want_kind <name> <colony slots> <senses slots> <now> <expected UTC> <expected kind>
want_kind() {
    local name="$1" cs="$2" ss="$3" now="$4" expect="$5" wantk="$6" got gotk exp
    exp="$(e "$expect")"
    got="$(MOLEQULA_RUN="$TMP" SCHEDULE_CONF=/dev/null SCHEDULE_SLOTS="$cs" SENSES_SLOTS="$ss" \
           MOLEQULA_SCHED_NOW="$(e "$now")" bash "$SCHED" next --epoch 2>&1)"
    gotk="$(MOLEQULA_RUN="$TMP" SCHEDULE_CONF=/dev/null SCHEDULE_SLOTS="$cs" SENSES_SLOTS="$ss" \
           MOLEQULA_SCHED_NOW="$(e "$now")" bash "$SCHED" next --kind 2>&1)"
    if [ "$got" = "$exp" ] && [ "$gotk" = "$wantk" ]; then
        pass=$((pass + 1)); printf 'ok   %s\n' "$name"
    else
        fail=$((fail + 1))
        printf 'FAIL %s: colony [%s] senses [%s] now %s -> got %s/%s (%s), want %s/%s (%s)\n' \
            "$name" "$cs" "$ss" "$now" "$got" "$gotk" \
            "$(date -u -d "@$got" +%FT%TZ 2>/dev/null || echo '?')" "$exp" "$wantk" "$expect"
    fi
}

# window <name> <colony slots> <dur> <time HH:MM> <inside|outside>
window() {
    local name="$1" cs="$2" dur="$3" t="$4" expect="$5" got rc rc_ok=1
    got="$(MOLEQULA_RUN="$TMP" SCHEDULE_CONF=/dev/null SCHEDULE_SLOTS="$cs" SCHEDULE_DUR="$dur" \
           MOLEQULA_SCHED_NOW="$(e "2026-09-13T00:00:00Z")" bash "$SCHED" in-window "$t" 2>&1)"; rc=$?
    # The word and the exit code must agree: a caller may read either.
    case "$expect" in
        inside)  [ "$rc" -eq 0 ] || rc_ok=0 ;;
        outside) [ "$rc" -ne 0 ] || rc_ok=0 ;;
    esac
    if [ "$got" = "$expect" ] && [ "$rc_ok" -eq 1 ]; then
        pass=$((pass + 1)); printf 'ok   %s\n' "$name"
    else
        fail=$((fail + 1)); printf 'FAIL %s: colony [%s] dur %s at %s -> %s (rc %d), want %s\n' \
            "$name" "$cs" "$dur" "$t" "$got" "$rc" "$expect"
    fi
}

S="01:00 03:00 07:00 09:00 11:00 15:00 17:00 19:00 23:00"

want_kind "senses first after midnight"  "$D" "$S" "2026-09-13T00:30:00Z" "2026-09-13T01:00:00Z" senses
want_kind "colony first before 04:00"    "$D" "$S" "2026-09-13T03:30:00Z" "2026-09-13T04:00:00Z" colony
want_kind "the senses take the gap"      "$D" "$S" "2026-09-13T06:30:00Z" "2026-09-13T07:00:00Z" senses
want_kind "and give the slot back"       "$D" "$S" "2026-09-13T11:30:00Z" "2026-09-13T12:00:00Z" colony
want_kind "the last senses slot wraps"   "$D" "$S" "2026-09-13T23:30:00Z" "2026-09-14T01:00:00Z" senses
want_kind "a tie goes to the colony"     "$D" "04:00 09:00" "2026-09-13T03:00:00Z" "2026-09-13T04:00:00Z" colony
want_kind "no senses slots, colony only" "$D" "" "2026-09-13T00:30:00Z" "2026-09-13T04:00:00Z" colony
want_kind "senses only between two colony days" "20:00" "$S" "2026-09-13T21:00:00Z" "2026-09-13T23:00:00Z" senses

window "a senses slot inside a colony window" "$D" 7200 "05:00" inside
window "the colony start itself"              "$D" 7200 "04:00" inside
window "the moment the window closes"         "$D" 7200 "06:00" outside
window "an hour after the window"             "$D" 7200 "07:00" outside
window "a minute before it opens"             "$D" 7200 "03:59" outside
window "the configured senses slots clear the colony" "$D" 7200 "23:00" outside
# A session long enough to run past midnight still covers 01:00 the next day.
window "yesterday's session reaches into today" "20:00" 21600 "01:00" inside

# ── the prekill before a colony session ─────────────────────────────────────
# SCHEDULE_PREKILL runs in the real Android environment immediately before
# launch.sh, to hand the organisms back the memory Android is sitting on in its
# cached bin. Here `android` is a stub on PATH that records its argv, and
# launch.sh / stop.sh are stubs beside a symlink to the real schedule.sh, so the
# slot under test is run_session itself, driven by `schedule.sh __slot`.

STUB="$TMP/phone1"
mkdir -p "$STUB" "$TMP/bin"
ln -s "$SCHED" "$STUB/schedule.sh"
printf '#!/bin/bash\necho launched\nexit 0\n' > "$STUB/launch.sh"
printf '#!/bin/bash\necho stopped\nexit 0\n' > "$STUB/stop.sh"
cat > "$TMP/bin/android" <<EOF
#!/bin/bash
printf '%s\n' "\$*" >> "$TMP/android.seen"
exit \${ANDROID_STUB_RC:-0}
EOF
chmod +x "$TMP/bin/android"

# slot <kind> <env assignments...> -> runs one slot, leaves stdout in $TMP/slot.out
# and the session line in $TMP/run/schedule.log. The android stub's argv, one
# call per line, lands in $TMP/android.seen.
slot() {
    local kind="$1"; shift
    rm -rf "$TMP/run"; mkdir -p "$TMP/run"
    rm -f "$TMP/android.seen"
    env -u ANDROID_STUB_RC PATH="$TMP/bin:$PATH" MOLEQULA_RUN="$TMP/run" \
        SCHEDULE_CONF=/dev/null SCHEDULE_SLOTS="$D" SCHEDULE_DUR=5 \
        SCHEDULE_SAMPLE=1 SCHEDULE_GRACE=1 SENSES_CMD='true' "$@" \
        bash "$STUB/schedule.sh" __slot "$kind" > "$TMP/slot.out" 2>&1
}

# calls -> how many times the android stub was invoked
calls() { [ -f "$TMP/android.seen" ] && wc -l < "$TMP/android.seen" || echo 0; }

slot colony
eq "a colony slot calls the prekill once" "$(calls | tr -d ' ')" "1"
eq "and calls it with am kill-all" "$(cat "$TMP/android.seen" 2>/dev/null)" "am kill-all"
if grep -q 'launched' "$TMP/slot.out" && \
   grep -q 'am kill-all' "$TMP/slot.out" && \
   [ "$(grep -c 'am kill-all' "$TMP/slot.out")" -ge 1 ] && \
   [ "$(awk '/am kill-all/{k=NR} /launched/{l=NR} END{print (k && l && k < l) ? "yes" : "no"}' "$TMP/slot.out")" = yes ]; then
    ok "the prekill runs before launch.sh, not after"
else
    bad "the prekill runs before launch.sh, not after: $(tr '\n' '|' < "$TMP/slot.out")"
fi
line="$(cat "$TMP/run/schedule.log" 2>/dev/null)"
case "$line" in
    *prekill_mb=[0-9]*-\>[0-9]*) ok "the session line carries prekill_mb=A->B" ;;
    *) bad "the session line carries prekill_mb=A->B: got '$line'" ;;
esac

slot senses
eq "a senses slot does not call the prekill" "$(calls | tr -d ' ')" "0"

printf 'SCHEDULE_PREKILL=""\n' > "$TMP/noprekill.conf"
slot colony SCHEDULE_CONF="$TMP/noprekill.conf"
eq "an empty SCHEDULE_PREKILL disables it" "$(calls | tr -d ' ')" "0"
grep -q 'launched' "$TMP/slot.out" && ok "and the slot still runs" \
    || bad "and the slot still runs: $(tr '\n' '|' < "$TMP/slot.out")"

slot colony ANDROID_STUB_RC=3
if grep -q 'launched' "$TMP/slot.out"; then
    ok "a failing prekill does not cost the slot"
else
    bad "a failing prekill does not cost the slot: $(tr '\n' '|' < "$TMP/slot.out")"
fi
grep -qi 'prekill.*fail' "$TMP/slot.out" && ok "and the failure is logged" \
    || bad "and the failure is logged: $(tr '\n' '|' < "$TMP/slot.out")"

# A command that is not there at all is the same promise: log it, run the slot.
slot colony SCHEDULE_PREKILL="no-such-command-for-the-gate"
if grep -q 'launched' "$TMP/slot.out" && grep -qi 'prekill.*fail' "$TMP/slot.out"; then
    ok "a missing prekill command is logged and the slot runs"
else
    bad "a missing prekill command is logged and the slot runs: $(tr '\n' '|' < "$TMP/slot.out")"
fi

printf '\n%d pass, %d fail\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

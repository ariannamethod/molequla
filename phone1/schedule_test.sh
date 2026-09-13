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

printf '\n%d pass, %d fail\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

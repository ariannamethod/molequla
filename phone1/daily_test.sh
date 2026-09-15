#!/bin/bash
# Gate for the cafeteria table `daily.sh` appends to daily/<date>.md. The
# fixture is a stdout tree written by hand — no organism is started here and
# none is needed: what is under test is the arithmetic over the `[cafeteria]`
# lines, not the binary that prints them. Change a column, count the wrong
# session or drop the element filter and this goes red.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0
ok()  { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf 'FAIL %s: %s\n' "$1" "${2:-}"; }

export MOLEQULA_RUN="$TMP/run"
mkdir -p "$MOLEQULA_RUN"/{earth,air,fire,pids,daily}

banner() { printf '[ecology] Element: %s → corpus: nonames_%s.txt\n' "$1" "$1"; }
cafe() { # cafe <element> <admitted> <owner> <resonance> <novelty> <unmeasured> <declined> <band> <warming> <measured>
    printf '[cafeteria] %s admitted=%s (owner=%s resonance=%s novelty=%s unmeasured=%s) declined=%s (band=%s warming=%s) measured=%s\n' \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"
}

# earth: a previous start whose numbers must not be counted, then the start
# that is being reported on — three passes, one of them admitting nothing.
{
    banner earth
    cafe earth 99 99 99 99 99 99 99 99 99
    printf '[dna] earth consumed 100 bytes from 1 files: [air/gen_1_0.txt]\n'
    banner earth
    cafe earth 3 2 1 0 0 1 1 0 2
    cafe earth 0 0 0 0 0 2 0 2 2
    cafe earth 5 1 2 1 1 0 0 0 4
} > "$MOLEQULA_RUN/earth/earth.stdout"

# air: started, never judged anything — a binary without the line, or a
# cafeteria that was never offered a fragment, must read as zero and not as a
# gap in the table.
banner air > "$MOLEQULA_RUN/air/air.stdout"

# fire: one decline-only pass, plus a line naming another organism that must
# not be counted into fire's row.
{
    banner fire
    cafe earth 7 7 0 0 0 0 0 0 0
    cafe fire 0 0 0 0 0 4 4 0 4
} > "$MOLEQULA_RUN/fire/fire.stdout"

# water has no stdout at all.

bash "$HERE/daily.sh" > "$TMP/daily.out" 2> "$TMP/daily.err"
rc=$?
DAY="$(date -u +%F)"
FILE="$MOLEQULA_RUN/daily/$DAY.md"
if [ "$rc" -eq 0 ] && [ -s "$FILE" ]; then
    ok "daily.sh exits 0 and writes daily/$DAY.md"
else
    bad "daily.sh exits 0 and writes daily/$DAY.md" "rc=$rc $(tail -2 "$TMP/daily.err")"
fi

if grep -q "Cafeteria decisions since each organism's last start:" "$FILE"; then
    ok "the table has its heading"
else
    bad "the table has its heading" "absent from $FILE"
fi

# row <element> — the table row, whitespace squeezed so the assertion is on the
# numbers and their order, not on the column widths.
row() { grep -a "^$1 " "$FILE" | tail -1 | tr -s ' ' | sed 's/ *$//'; }

check() {
    local name="$1" want="$2" got
    got="$(row "$3")"
    if [ "$got" = "$want" ]; then ok "$name"; else bad "$name" "got [$got] want [$want]"; fi
}

# 3 + 0 + 5 admitted over three passes, 1 + 2 + 0 declined of which 1 + 0 + 0 is
# band and 0 + 2 + 0 is warming, 2 + 2 + 4 measured; the 99s of the earlier start
# are behind the second banner and are not in it.
check "earth counts the last start only" \
      "earth 3 8 3 3 1 1 3 1 2 8 27.3%" earth
check "air reads as zero, not as a gap" \
      "air 0 0 0 0 0 0 0 0 0 0 -" air
check "fire counts its own lines only" \
      "fire 1 0 0 0 0 0 4 4 0 4 100.0%" fire

if [ "$(row water)" = "water no stdout" ]; then
    ok "a missing stdout is said, not skipped"
else
    bad "a missing stdout is said, not skipped" "got [$(row water)]"
fi

# The block appended is one section per run: a second run must not rewrite the
# first one's numbers.
before="$(wc -l < "$FILE")"
bash "$HERE/daily.sh" > /dev/null 2>&1
after="$(wc -l < "$FILE")"
if [ "$after" -gt "$before" ] && [ "$(grep -ac "Cafeteria decisions" "$FILE")" -eq 2 ]; then
    ok "a second run appends a second section"
else
    bad "a second run appends a second section" "lines $before -> $after"
fi

printf '\n%d pass, %d fail\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

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
# resumed <path> <size MB> <read ms> [peak MB] — the line molequla.go's
# sayCheckpointResume prints, verbatim in shape including the em dash.
#
# Two path shapes occur and both are in the fixtures. The colony's own line
# carries a bare basename, because launch.sh cds into the organism's directory
# (`cd "$dir"`, phone1/launch.sh) and CFG.CkptPath is relative — that is the
# shape a scratch resume of the live air checkpoint printed on 2026-09-17. A
# mitosis child is told an absolute ckpt_path in its birth config
# (molequla.go, `birth["ckpt_path"]`) and prints that instead.
resumed() {
    printf '[ckpt] resumed from %s — %s MB, read in %s ms' "$1" "$2" "$3"
    [ "$#" -ge 4 ] && printf ', peak +%s MB' "$4"
    printf '\n'
}

# earth: a previous start whose numbers must not be counted, then the start
# that is being reported on — three passes, one of them admitting nothing.
# The earlier start resumed from the JSON and this one from the binary, and
# this one also carries a refused sibling and a line naming water's checkpoint:
# the table must report earth's own `.gguf`, not the `.json` of the start
# before it and not water's file.
{
    banner earth
    cafe earth 99 99 99 99 99 99 99 99 99
    resumed molequla_ckpt.json 261.0 8801
    printf '[dna] earth consumed 100 bytes from 1 files: [air/gen_1_0.txt]\n'
    banner earth
    printf '[ckpt] molequla_ckpt.gguf not used (identity aaaaaaaaaaaa does not match the JSON checkpoint bbbbbbbbbbbb) — loading the JSON checkpoint\n'
    resumed "$MOLEQULA_RUN/water/molequla_ckpt.gguf" 53.3 200 238
    resumed "$MOLEQULA_RUN/earth/molequla_ckpt.gguf" 49.4 146 130
    cafe earth 3 2 1 0 0 1 1 0 2
    cafe earth 0 0 0 0 0 2 0 2 2
    cafe earth 5 1 2 1 1 0 0 0 4
} > "$MOLEQULA_RUN/earth/earth.stdout"

# air: started, never judged anything — a binary without the line, or a
# cafeteria that was never offered a fragment, must read as zero and not as a
# gap in the table. The start before it said which checkpoint it read and this
# one said nothing, which is what a binary older than this line looks like: the
# row must say the line is missing rather than report the older start's file.
{
    resumed molequla_ckpt.gguf 28.2 131 126
    banner air
} > "$MOLEQULA_RUN/air/air.stdout"

# fire: one decline-only pass, plus a line naming another organism that must
# not be counted into fire's row. An earlier start of it found no checkpoint at
# all and climbed from an embryo — the case that used to look exactly like a
# clean resume — and this start resumed from the binary, in the bare-basename
# shape the colony actually prints.
{
    printf '[ckpt] no checkpoint at molequla_ckpt.json (open molequla_ckpt.json: no such file or directory) — the organism starts from an embryo\n'
    banner fire
    cafe earth 7 7 0 0 0 0 0 0 0
    cafe fire 0 0 0 0 0 4 4 0 4
    resumed molequla_ckpt.gguf 27.3 124 118
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

# ── the checkpoint each organism resumed from ─────────────────────────────────
# Same discipline as the cafeteria table and for the same reason: the stdout
# files are appended across every session ever run, so a reading that is not
# scoped to the organism's last `[ecology] Element:` banner answers a question
# nobody asked. ckrow slices the last "Checkpoint ..." block out of the file so
# the assertion cannot accidentally match the cafeteria row of the same element.
ckrow() {
    awk -v E="$1" '
        /^Checkpoint each organism resumed from/ { blk = ""; grab = 1; fence = 0; next }
        grab && /^```$/ { fence++; if (fence == 2) { grab = 0 }; next }
        grab && $1 == E { blk = $0 }
        END { print blk }' "$FILE" | tr -s ' ' | sed 's/ *$//'
}

ckcheck() {
    local name="$1" want="$2" got
    got="$(ckrow "$3")"
    if [ "$got" = "$want" ]; then ok "$name"; else bad "$name" "got [$got] want [$want]"; fi
}

if grep -q "Checkpoint each organism resumed from, this session:" "$FILE"; then
    ok "the checkpoint table has its heading"
else
    bad "the checkpoint table has its heading" "absent from $FILE"
fi

# earth's second start read the binary; the first start's .json, the refused
# sibling and water's line are all in the same file and none of them is earth's
# answer. Reading the whole file instead of the last banner reports the .json.
ckcheck "earth reports the last start's file, not the first's" \
        "earth gguf molequla_ckpt.gguf 49.4 146 +130" earth
# air's last start printed nothing, which is a binary older than this line and
# not a resume from the file the start before it read.
ckcheck "a start with no [ckpt] line is said, not filled in from an older start" \
        "air - (no [ckpt] line) - - -" air
# fire's own line is the bare-basename shape the colony prints, and the embryo
# line of the start before it is behind fire's banner and is not its answer.
ckcheck "the bare-basename path the colony prints is read as the organism's own" \
        "fire gguf molequla_ckpt.gguf 27.3 124 +118" fire
if [ "$(ckrow water)" = "water no stdout" ]; then
    ok "a missing stdout is said in the checkpoint table too"
else
    bad "a missing stdout is said in the checkpoint table too" "got [$(ckrow water)]"
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

# A further start of fire, appended to the same file, that found no checkpoint
# and climbed from an embryo. It is the 2026-09-16 error in its sharpest form:
# the start before it resumed from the binary, and a reading that is not scoped
# to fire's last banner reports that resume for a session which has just lost
# its weights. The row for the embryo must also be distinguishable from the row
# for a start that said nothing at all, which is air's above.
{
    banner fire
    printf '[ckpt] no checkpoint at molequla_ckpt.json (open molequla_ckpt.json: no such file or directory) — the organism starts from an embryo\n'
} >> "$MOLEQULA_RUN/fire/fire.stdout"
bash "$HERE/daily.sh" > /dev/null 2>&1
ckcheck "a start that climbed from an embryo is said, not read as the last resume" \
        "fire embryo (none) - - -" fire
ckcheck "the organisms that did not restart keep their own answer" \
        "earth gguf molequla_ckpt.gguf 49.4 146 +130" earth

printf '\n%d pass, %d fail\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

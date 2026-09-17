#!/bin/bash
# Append a dated, timed section to $MOLEQULA_RUN/daily/<UTC date>.md: the
# status screen, the DNA traffic so far, the witness's last words, the disk.
# Running it several times a day appends several sections, nothing is lost.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
export MOLEQULA_RUN="$RUN"
ELEMENTS="earth air water fire"

mkdir -p "$RUN/daily" || exit 1
DAY="$(date -u +%F)"
FILE="$RUN/daily/$DAY.md"

{
    echo
    echo "## $DAY $(date -u +%T) UTC"
    echo
    echo '```'
    bash "$HERE/status.sh"
    echo '```'
    echo
    # Which file each organism was rebuilt from on its last start. Scoped to the
    # organism's last `[ecology] Element:` banner for the same reason the
    # cafeteria table below is: one stdout file spans every session ever run, so
    # a whole-file reading answers about some earlier boot. `from` is gguf or
    # json when the organism resumed, embryo when there was no checkpoint to
    # read, unread when one was there and could not be parsed — which is a loss
    # and not a beginning — and `-` with `(no [ckpt] line)` when the start said
    # nothing at all, which is a binary older than this line. Those four used to
    # be one thing: silence. The resume line's three numbers are the file's size,
    # the wall time the read took and how far it pushed VmHWM, and the peak is
    # absent from the line whenever the high-water was already above it
    # (molequla.go, sayCheckpointResume).
    echo "Checkpoint each organism resumed from, this session:"
    echo
    echo '```'
    printf '%-6s %-6s %-20s %8s %8s %8s\n' org from file size-MB read-ms peak-MB
    for e in $ELEMENTS; do
        out="$RUN/$e/$e.stdout"
        if [ ! -f "$out" ]; then
            printf '%-6s no stdout\n' "$e"
            continue
        fi
        tr -d '\000' < "$out" | awk -v E="$e" '
            $1 == "[ecology]" && $2 == "Element:" { from=""; file=""; sz="-"; rd="-"; pk="-"; next }
            # The path is field 4. A bare basename belongs to this organism by
            # construction: launch.sh cds into its directory and CFG.CkptPath is
            # relative, so that is the shape the colony prints. A mitosis child
            # is told an absolute ckpt_path in its birth config and prints that,
            # and there the element in the path is what says whose line it is —
            # a line naming a sibling can reach this file.
            $1 == "[ckpt]" && $2 == "resumed" && $3 == "from" &&
            (index($4, "/") == 0 || index($4, "/" E "/")) {
                file = $4
                from = (file ~ /\.gguf$/) ? "gguf" : "json"
                n = split(file, seg, "/"); file = seg[n]
                sz = match($0, /[0-9.]+ MB, read in/) ? substr($0, RSTART, RLENGTH - 12) : "-"
                rd = match($0, /read in [0-9]+ ms/)   ? substr($0, RSTART + 8, RLENGTH - 11) : "-"
                pk = match($0, /peak \+[0-9]+ MB/)    ? substr($0, RSTART + 5, RLENGTH - 8) : "-"
                next
            }
            $1 == "[ckpt]" && /starts from an embryo/ {
                sz="-"; rd="-"; pk="-"
                if ($3 == "could" && $4 == "not") {
                    from = "unread"; n = split($2, seg, "/"); file = seg[n]
                } else {
                    from = "embryo"; file = "(none)"
                }
                next
            }
            END {
                if (from == "") { from = "-"; file = "(no [ckpt] line)" }
                printf "%-6s %-6s %-20s %8s %8s %8s\n", E, from, file, sz, rd, pk
            }'
    done
    echo '```'
    echo
    echo "DNA traffic since the stdout files were opened:"
    echo
    echo '```'
    for e in $ELEMENTS; do
        out="$RUN/$e/$e.stdout"
        if [ ! -f "$out" ]; then
            printf '%-6s no stdout\n' "$e"
            continue
        fi
        grep -a "\[dna\] $e " "$out" | tr -d '\000' | awk -v E="$e" '
            /wrote/    {w++; if (match($0, /wrote [0-9]+ bytes/)) wb += substr($0, RSTART+6, RLENGTH-12)}
            /consumed/ {c++; if (match($0, /consumed [0-9]+ bytes/)) cb += substr($0, RSTART+9, RLENGTH-15)}
            END {printf "%-6s wrote=%d lines (%d bytes)  consumed=%d bytes in %d events\n", E, w, wb, cb, c}'
    done
    echo '```'
    echo
    # The cafeteria (experience_routing.go): what each organism was offered and
    # what it took. Counted from the organism's last start — the `[ecology]
    # Element:` banner it prints on boot — and not over the whole file, because
    # a decline rate is a property of a field and a corpus that both move
    # between sessions. admitted is the sum of its four reasons, declined of its
    # two; passes counts the dnaRead calls that judged anything at all. band and
    # warming are separate columns because they mean opposite things: band is a
    # judgement against the organism's own coverage quantiles, warming is a ring
    # too short to have any (experience_routing.go).
    echo "Cafeteria decisions since each organism's last start:"
    echo
    echo '```'
    printf '%-6s %6s %8s %6s %9s %7s %10s %8s %5s %8s %8s %8s\n' \
        org passes admitted owner resonance novelty unmeasured declined band warming measured declined%
    for e in $ELEMENTS; do
        out="$RUN/$e/$e.stdout"
        if [ ! -f "$out" ]; then
            printf '%-6s no stdout\n' "$e"
            continue
        fi
        tr -d '\000' < "$out" | awk -v E="$e" '
            $1 == "[ecology]" && $2 == "Element:" { p=a=o=r=n=u=d=b=w=m=0; next }
            $1 == "[cafeteria]" && $2 == E {
                p++
                for (i = 3; i <= NF; i++) {
                    t = $i; gsub(/[()]/, "", t); split(t, kv, "="); v = kv[2] + 0
                    if      (kv[1] == "admitted")   a += v
                    else if (kv[1] == "owner")      o += v
                    else if (kv[1] == "resonance")  r += v
                    else if (kv[1] == "novelty")    n += v
                    else if (kv[1] == "unmeasured") u += v
                    else if (kv[1] == "declined")   d += v
                    else if (kv[1] == "band")       b += v
                    else if (kv[1] == "warming")    w += v
                    else if (kv[1] == "measured")   m += v
                }
            }
            END {
                share = (a + d > 0) ? sprintf("%.1f%%", 100 * d / (a + d)) : "-"
                printf "%-6s %6d %8d %6d %9d %7d %10d %8d %5d %8d %8d %8s\n", E, p, a, o, r, n, u, d, b, w, m, share
            }'
    done
    echo '```'
    echo
    echo "Newest three \`[dna] ... wrote\` lines (per-file order earth air water fire):"
    echo
    echo '```'
    for e in $ELEMENTS; do
        [ -f "$RUN/$e/$e.stdout" ] && grep -a "\[dna\] $e wrote " "$RUN/$e/$e.stdout" | tr -d '\000'
    done | tail -3
    echo '```'
    echo
    echo "Witness, last five lines:"
    echo
    echo '```'
    if [ -f "$RUN/witness/witness.stdout" ]; then
        grep -a '' "$RUN/witness/witness.stdout" | tr -d '\000' | tail -5
    else
        echo "(no witness.stdout)"
    fi
    echo '```'
    echo
    echo "Disk: $(df -h "$RUN" | tail -1 | tr -s ' ')"
} >> "$FILE"

echo "[daily] appended $DAY $(date -u +%T) UTC to $FILE ($(wc -l < "$FILE") lines)"

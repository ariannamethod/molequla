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
    # one; passes counts the dnaRead calls that judged anything at all.
    echo "Cafeteria decisions since each organism's last start:"
    echo
    echo '```'
    printf '%-6s %6s %8s %6s %9s %7s %10s %8s %5s %8s %8s\n' \
        org passes admitted owner resonance novelty unmeasured declined band measured declined%
    for e in $ELEMENTS; do
        out="$RUN/$e/$e.stdout"
        if [ ! -f "$out" ]; then
            printf '%-6s no stdout\n' "$e"
            continue
        fi
        tr -d '\000' < "$out" | awk -v E="$e" '
            $1 == "[ecology]" && $2 == "Element:" { p=a=o=r=n=u=d=b=m=0; next }
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
                    else if (kv[1] == "measured")   m += v
                }
            }
            END {
                share = (a + d > 0) ? sprintf("%.1f%%", 100 * d / (a + d)) : "-"
                printf "%-6s %6d %8d %6d %9d %7d %10d %8d %5d %8d %8s\n", E, p, a, o, r, n, u, d, b, m, share
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

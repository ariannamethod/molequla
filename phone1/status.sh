#!/bin/bash
# One screen of the colony: process, memory, growth, corpus, DNA, witness.
set -u

RUN="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
BIN="$RUN/molequla_cgo"
PIDDIR="$RUN/pids"
ELEMENTS="earth air water fire"

# A pid file may name a `timeout` wrapper (dry runs); the memory that matters
# belongs to the organism under it.
real_pid() {
    local p="$1" c
    while [ -r "/proc/$p/comm" ] && [ "$(cat "/proc/$p/comm" 2>/dev/null)" = "timeout" ]; do
        c="$(pgrep -P "$p" 2>/dev/null | head -1)"
        [ -n "$c" ] || break
        p="$c"
    done
    printf '%s' "$p"
}

mb() {
    [ -r "$2" ] || { printf -- '-'; return; }
    awk -v k="$1:" '$1==k{printf "%.0f", $2/1024; f=1} END{if(!f)printf "-"}' "$2"
}

echo "=== molequla status $(date -u +%FT%TZ) — run root $RUN"
[ -f "$RUN/BUILD" ] && echo "build: $(cat "$RUN/BUILD")"
printf '%-6s %-8s %-6s %6s %6s %6s %10s %6s %10s %7s %5s\n' \
    org pid state rssMB hwmMB stage ingested nan corpusB lines dna

for e in $ELEMENTS; do
    pid="$(cat "$PIDDIR/$e.pid" 2>/dev/null)"
    state=dead
    rss='-'; hwm='-'
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        state=alive
        rp="$(real_pid "$pid")"
        rss="$(mb VmRSS "/proc/$rp/status")"
        hwm="$(mb VmHWM "/proc/$rp/status")"
    fi
    out="$RUN/$e/$e.stdout"
    stage='-'; ing='-'; nan=0
    if [ -f "$out" ]; then
        line="$(grep -a '\[debug-onto\]' "$out" | tail -1 | tr -d '\000')"
        s="$(printf '%s' "$line" | sed -n 's/.*stage=\([0-9-]*\).*/\1/p')"
        i="$(printf '%s' "$line" | sed -n 's/.*ingested=\([0-9-]*\).*/\1/p')"
        [ -n "$s" ] && stage="$s"
        [ -n "$i" ] && ing="$i"
        nan="$(grep -a -c -i nan "$out")"
    fi
    corp="$RUN/$e/nonames_$e.txt"
    cb=0; cl=0
    if [ -f "$corp" ]; then cb="$(wc -c < "$corp")"; cl="$(wc -l < "$corp")"; fi
    dna="$(ls "$RUN/dna/output/$e" 2>/dev/null | wc -l)"
    printf '%-6s %-8s %-6s %6s %6s %6s %10s %6s %10s %7s %5s\n' \
        "$e" "${pid:--}" "$state" "$rss" "$hwm" "$stage" "$ing" "$nan" "$cb" "$cl" "$dna"
done

wpid="$(cat "$PIDDIR/witness.pid" 2>/dev/null)"
wstate=dead
if [ -n "$wpid" ] && kill -0 "$wpid" 2>/dev/null; then wstate=alive; fi
echo "witness: pid ${wpid:--} $wstate"
if [ -f "$RUN/witness/witness.stdout" ]; then
    echo "witness last line: $(grep -a '' "$RUN/witness/witness.stdout" | tail -1 | tr -d '\000')"
else
    echo "witness last line: (no witness.stdout)"
fi

if [ -x "$BIN" ]; then
    snap="$(cd "$RUN/witness" && "$BIN" --witness --once 2>&1 | tr -d '\000')"
    norg="$(printf '%s\n' "$snap" | grep -c '"id":')"
    fh="$(printf '%s\n' "$snap" | sed -n 's/.*"field_entropy": *\([0-9.eE+-]*\).*/\1/p' | head -1)"
    act="$(printf '%s\n' "$snap" | sed -n 's/.*"action": *"\([^"]*\)".*/\1/p' | head -1)"
    alerts="$(printf '%s\n' "$snap" | awk '
        /"alerts": \[/ {f=1; next}
        f && /\]/      {f=0}
        f              {gsub(/^[ \t"]+/,""); gsub(/[",]+$/,""); a = a (a ? "; " : "") $0}
        END            {print a}')"
    if [ -z "$fh$act" ]; then
        echo "witness --once: no snapshot — $(printf '%s\n' "$snap" | head -2 | tr '\n' ' ')"
    else
        echo "witness --once: organisms=$norg field_entropy=${fh:--} action=${act:--} alerts=${alerts:-none}"
    fi
else
    echo "witness --once: no binary at $BIN"
fi

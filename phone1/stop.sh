#!/bin/bash
# Stop everything named in $MOLEQULA_RUN/pids: SIGTERM, up to 20 s of grace,
# SIGKILL to survivors. Releases the Termux wake lock at the end.
set -u

RUN="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
PIDDIR="$RUN/pids"

shopt -s nullglob
# schedule.pid is the scheduler daemon, not a colony process: it lives in the
# same directory and must survive the stop it calls itself.
files=()
for f in "$PIDDIR"/*.pid; do
    [ "$(basename "$f")" = "schedule.pid" ] && continue
    files+=("$f")
done
if [ ${#files[@]} -eq 0 ]; then
    echo "[stop] no pid files in $PIDDIR"
else
    names=(); pids=()
    for f in "${files[@]}"; do
        n="$(basename "$f" .pid)"
        p="$(cat "$f" 2>/dev/null)"
        if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
            echo "[stop] $n: pid ${p:-none} already gone"
            rm -f "$f"
            continue
        fi
        names+=("$n"); pids+=("$p")
        # Each process is its own session leader (setsid), so the negative pid
        # reaches the whole group — timeout wrapper and mitosis children too.
        kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
        echo "[stop] $n: SIGTERM to pid $p"
    done

    for _ in $(seq 20); do
        left=0
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && left=$((left + 1)); done
        [ "$left" -eq 0 ] && break
        sleep 1
    done

    i=0
    while [ $i -lt ${#pids[@]} ]; do
        n="${names[$i]}"; p="${pids[$i]}"
        if kill -0 "$p" 2>/dev/null; then
            echo "[stop] $n: pid $p still alive after 20 s — SIGKILL"
            kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null
            sleep 1
        else
            echo "[stop] $n: pid $p exited on SIGTERM"
        fi
        kill -0 "$p" 2>/dev/null || rm -f "$PIDDIR/$n.pid"
        i=$((i + 1))
    done
fi

if ssh -o BatchMode=yes -o ConnectTimeout=5 -i /root/.ssh/id_ed25519 \
       -p 8022 u0_a327@localhost termux-wake-unlock >/dev/null 2>&1; then
    echo "[stop] termux wake lock released"
else
    echo "[stop] warning: termux-wake-unlock over ssh failed — the lock may still be held"
fi

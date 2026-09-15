#!/bin/bash
# Start the colony detached: four organisms on the big cores, the witness on
# the little ones five seconds later. Survives the shell that started it.
# Usage: launch.sh [DUR]   — DUR in seconds wraps every start in `timeout`.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RUN="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
BIN="$RUN/molequla_cgo"
DUR="${1:-}"
ELEMENTS="earth air water fire"
# What the senses write while the colony sleeps (phone1/senses.sh): one
# directory per organ under dna/output, beside the four elements. Read-only
# food for every organism — they are told about it with --dna-extra-sources.
SENSES_SOURCES="world sound place"
# The same list in the form the flag takes. One variable, two readers: the four
# organisms eat these directories and the witness counts them. dnaSources("")
# in the witness process is built from CFG.DNAExtraSources like everybody
# else's, so a witness started without the argument reports a DNA field of four
# elements and never names what the eye, the ears and the place left behind.
SENSES_ARG="${SENSES_SOURCES// /,}"

if [ ! -x "$BIN" ]; then
    echo "[launch] no binary at $BIN — run phone1/build.sh first"
    exit 1
fi

mkdir -p "$RUN/pids" "$RUN/witness" "$RUN/daily" "$RUN/dna/output" || exit 1
for e in $ELEMENTS; do
    mkdir -p "$RUN/$e" "$RUN/dna/output/$e" || exit 1
    # Never overwrite a live corpus: the organisms write into their own copy.
    if [ ! -f "$RUN/$e/nonames_$e.txt" ]; then
        cp "$REPO/nonames_$e.txt" "$RUN/$e/nonames_$e.txt" || exit 1
        echo "[launch] $e: corpus copied from $REPO/nonames_$e.txt"
    fi
done
for s in $SENSES_SOURCES; do
    mkdir -p "$RUN/dna/output/$s" || exit 1
done

# The phone sleeps its CPU without a wake lock; the lock lives in Termux.
if ssh -o BatchMode=yes -o ConnectTimeout=5 -i /root/.ssh/id_ed25519 \
       -p 8022 u0_a327@localhost termux-wake-lock >/dev/null 2>&1; then
    echo "[launch] termux wake lock held"
else
    echo "[launch] warning: termux-wake-lock over ssh failed — continuing without a wake lock"
fi

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# start <name> <workdir> -- <command...>
# setsid detaches into its own session; the shim writes its own pid (which
# `exec` keeps) so the pid file names the process group leader.
start() {
    local name="$1" dir="$2"; shift 2
    local pidf="$RUN/pids/$name.pid"
    local old; old="$(cat "$pidf" 2>/dev/null)"
    if alive "$old"; then
        echo "[launch] $name: already running (pid $old) — refusing to start a second one"
        return 1
    fi
    rm -f "$pidf"
    (
        cd "$dir" || exit 1
        setsid nohup bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$pidf" "$@" \
            >> "$name.stdout" 2>> "$name.stderr" < /dev/null &
    )
    local i=0 pid=""
    while [ $i -lt 40 ]; do
        pid="$(cat "$pidf" 2>/dev/null)"
        alive "$pid" && break
        i=$((i + 1)); sleep 0.1
    done
    if alive "$pid"; then
        echo "[launch] $name: pid $pid (cwd $dir)"
        return 0
    fi
    echo "[launch] $name: FAILED to start — see $dir/$name.stderr"
    return 1
}

TMO=""
[ -n "$DUR" ] && TMO="timeout $DUR"

refused=0
for e in $ELEMENTS; do
    # --max-organisms 4: the built-in cap of 16 was written for a pod. Four
    # trainer processes are what 8 GB holds; the byte gate before growth and
    # before division handles the minute-to-minute pressure, this is the
    # declared ceiling on the colony's head count.
    start "$e" "$RUN/$e" $TMO taskset -c 4-7 "$BIN" \
        --organism-id "$e" --element "$e" --max-organisms 4 \
        --dna-extra-sources "$SENSES_ARG" \
        --evolution --cross-graze --corpus-overlay \
        || refused=$((refused + 1))
    sleep 1
done

sleep 5
start witness "$RUN/witness" $TMO taskset -c 0-3 "$BIN" \
    --witness --witness-interval 5 \
    --dna-extra-sources "$SENSES_ARG" || refused=$((refused + 1))

echo "[launch] run root $RUN${DUR:+ (timeout ${DUR}s)}; pids in $RUN/pids"
if [ "$refused" -gt 0 ]; then
    echo "[launch] $refused process(es) not started"
    exit 1
fi
exit 0

#!/bin/sh
# molequla ecology launcher — Railway entrypoint.
#
# Each organism gets its own working directory inside the persistent volume,
# so its memory.sqlite3 / molequla_ckpt.json / spawned children stay isolated.
# DNA exchange flows through the parallel ../dna/output/<element>/ tree,
# which the molequla code expects relative to each organism's WD.

set -e

ROOT="${MOLEQULA_DATA_DIR:-/data}"
mkdir -p \
    "$ROOT/dna/output/earth" \
    "$ROOT/dna/output/air" \
    "$ROOT/dna/output/water" \
    "$ROOT/dna/output/fire" \
    "$ROOT/earth" \
    "$ROOT/air" \
    "$ROOT/water" \
    "$ROOT/fire"

# Each organism reads its corpus from /app — copy once if missing so the
# relative path in the code (`nonames_<element>.txt` resolved from each
# organism's WD) actually finds the file.
for E in earth air water fire; do
    if [ ! -f "$ROOT/$E/nonames_$E.txt" ]; then
        cp "/app/nonames_$E.txt" "$ROOT/$E/nonames_$E.txt"
    fi
done

cd "$ROOT"

# Tag every line with the organism so Railway log stream is readable.
prefix() {
    local tag="$1"
    awk -v T="$tag" '{ print "[" T "] " $0; fflush(); }'
}

# Launch each organism in its own WD. --evolution skips interactive pauses
# (Railway has no TTY anyway). Children spawned by mitosis inherit the WD.
( cd earth && exec /app/molequla --organism-id earth --element earth --evolution ) 2>&1 | prefix earth &
PID_E=$!
( cd air   && exec /app/molequla --organism-id air   --element air   --evolution ) 2>&1 | prefix air   &
PID_A=$!
( cd water && exec /app/molequla --organism-id water --element water --evolution ) 2>&1 | prefix water &
PID_W=$!
( cd fire  && exec /app/molequla --organism-id fire  --element fire  --evolution ) 2>&1 | prefix fire  &
PID_F=$!

echo "[ecology] launched: earth=$PID_E air=$PID_A water=$PID_W fire=$PID_F"
echo "[ecology] data root: $ROOT"

# Wait on any to exit, then forward exit code so Railway restarts the
# whole ecology (container restart > partial collapse).
wait -n $PID_E $PID_A $PID_W $PID_F
echo "[ecology] one organism exited — failing container so Railway restarts"
exit 1

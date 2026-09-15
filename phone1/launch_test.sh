#!/bin/bash
# Gate for what launch.sh actually hands the witness. Two stages, both over the
# real script and the real binary — nothing here restates the argv, it reads it.
#
# Stage 1 runs launch.sh against a stub binary that records its own argv, and
# asserts that the witness was told about the senses with the same list the four
# organisms were told about. Stage 2 takes the recorded witness argv, runs the
# real binary with it over a scratch tree holding one world fragment, and
# asserts that the snapshot's DNA field names `world`. Stage 3 removes the
# argument from a copy of launch.sh and requires stage 1 to go red, so the check
# is known to fail on the thing it is there to catch.
#
# Needs a built binary: $MOLEQULA_BIN, or molequla_cgo beside the repo root, or
# the one phone1/build.sh left in $MOLEQULA_RUN.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RUNDIR="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
BIN="${MOLEQULA_BIN:-}"
[ -n "$BIN" ] || { [ -x "$REPO/molequla_cgo" ] && BIN="$REPO/molequla_cgo"; }
[ -n "$BIN" ] || { [ -x "$RUNDIR/molequla_cgo" ] && BIN="$RUNDIR/molequla_cgo"; }
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "FAIL no binary: set MOLEQULA_BIN, or run phone1/build.sh first"
    exit 1
fi

TMP="$(mktemp -d)"
trap 'for p in "$TMP"/*/pids/*.pid; do [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null; done; rm -rf "$TMP"' EXIT

pass=0; fail=0
ok()   { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; }

# run_launch <script> <workdir> -> prints the argv log path. The stub records
# every invocation as one line and then sleeps, so launch.sh sees live pids and
# returns without its four-second per-process timeout.
run_launch() {
    local script="$1" work="$2" run="$2/run"
    mkdir -p "$run"
    cat > "$run/molequla_cgo" <<STUB
#!/bin/bash
printf '%s\n' "\$*" >> "$work/argv.log"
exec sleep 20
STUB
    chmod +x "$run/molequla_cgo"
    MOLEQULA_RUN="$run" bash "$script" > "$work/launch.out" 2>&1
    for p in "$run"/pids/*.pid; do [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null; done
    printf '%s' "$work/argv.log"
}

# argv_of <log> <marker> -> the recorded line carrying that marker
argv_of() { grep -m1 -- "$2" "$1" 2>/dev/null; }

# --- stage 1: the witness is told what the organisms are told -----------------
L="$(run_launch "$REPO/phone1/launch.sh" "$TMP/real")"
WITNESS_ARGV="$(argv_of "$L" --witness)"
EARTH_ARGV="$(argv_of "$L" '--organism-id earth')"

if [ -n "$WITNESS_ARGV" ]; then ok "launch.sh started a witness"
else bad "launch.sh started no witness (argv log: $(wc -l < "$L" 2>/dev/null || echo 0) lines)"; fi

# the list each of them received, or empty
extras_of() { sed -n 's/.*--dna-extra-sources \([^ ]*\).*/\1/p' <<< "$1"; }
W_EXTRAS="$(extras_of "$WITNESS_ARGV")"
E_EXTRAS="$(extras_of "$EARTH_ARGV")"

if [ -n "$W_EXTRAS" ]; then ok "the witness was told about the senses ($W_EXTRAS)"
else bad "the witness got no --dna-extra-sources: $WITNESS_ARGV"; fi

if [ -n "$E_EXTRAS" ] && [ "$W_EXTRAS" = "$E_EXTRAS" ]; then
    ok "witness and organisms read the same list"
else
    bad "witness reads [$W_EXTRAS], earth reads [$E_EXTRAS] — one list, two readers"
fi

# --- stage 2: the real witness, with that argv, names world -------------------
# The witness needs a mesh to read; two seconds of one organism writes one.
TREE="$TMP/real/run"
mkdir -p "$TREE/dna/output/world" "$TREE/witness" "$TREE/seed"
printf '[eye cam0 2026-09-13T22:12:01Z] A blurry table shows a green bowl and a spoon.\n' \
    > "$TREE/dna/output/world/gen_1789337521_2.txt"
head -c 4000 "$REPO/nonames_earth.txt" > "$TREE/seed/nonames_earth.txt"
( cd "$TREE/seed" && HOME="$TMP/home" timeout 60 "$BIN" --organism-id earth --element earth \
    --evolution --max-growth-stage 0 > "$TMP/seed.out" 2>&1 ) &
seed=$!
for _ in $(seq 120); do [ -f "$TMP/home/.molequla/swarm/mesh.db" ] && break; sleep 0.25; done
pkill -P "$seed" 2>/dev/null; kill "$seed" 2>/dev/null; wait "$seed" 2>/dev/null

if [ -f "$TMP/home/.molequla/swarm/mesh.db" ]; then
    # exactly the flags launch.sh gave the witness: the stub recorded its own
    # argv, so the taskset and timeout wrappers are not in this string
    FLAGS="$WITNESS_ARGV"
    SNAP="$(cd "$TREE/witness" && HOME="$TMP/home" timeout 60 "$BIN" $FLAGS --once 2>"$TMP/w.err")"
    if grep -q '"world"' <<< "$SNAP"; then
        ok "--witness --once names world in its DNA field"
    else
        bad "--witness --once ($FLAGS) never mentions world: $(tr -d '\n' <<< "$SNAP" | head -c 400)"
    fi
else
    bad "no mesh.db after 10 s — cannot run the witness ($(tail -2 "$TMP/seed.out" | tr '\n' ' '))"
fi

# --- stage 3: the check is shown failing on the defect it exists for ----------
mkdir -p "$TMP/mutant"
sed '/--witness --witness-interval/,+1 s/ *--dna-extra-sources "\$SENSES_ARG" *//' \
    "$REPO/phone1/launch.sh" > "$TMP/mutant/launch.sh"
if cmp -s "$REPO/phone1/launch.sh" "$TMP/mutant/launch.sh"; then
    bad "stage 3 could not build the mutant — the edit above no longer matches launch.sh"
else
    ML="$(run_launch "$TMP/mutant/launch.sh" "$TMP/mut")"
    if [ -z "$(extras_of "$(argv_of "$ML" --witness)")" ]; then
        ok "stage 1 goes red when the argument is dropped"
    else
        bad "the mutant witness still carries --dna-extra-sources — stage 1 proves nothing"
    fi
fi

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]

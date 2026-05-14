#!/bin/bash
# molequla coherence-layer measurement sweep — Phase B → Phase C
# 2 cells × 4-organism ecology × DUR seconds each.
# Cell 0 = baseline (no flags). Cell 3 = full coherence (--spa-gate --corpus-overlay).
# Per-org artifacts: train.log, memory.sqlite3, molequla_ckpt.json, DNA writes
# in shared <cell>/dna/. Summary printed per cell.
set -u
cd /workspace/molequla
EXEC=$(pwd)/molequla_cgo
DUR=${DUR:-600}

run_cell() {
    local label=$1; shift
    local flags="$*"
    local cd=runpod/2026-05-14/$label
    rm -rf "$cd" 2>/dev/null; mkdir -p "$cd"
    echo "[$(date -u +%H:%M:%S)] === starting $label flags='$flags' DUR=${DUR}s ==="
    for e in earth air water fire; do
        mkdir -p "$cd/work_$e"
        cp "$EXEC" "$cd/work_$e/"
        cp "nonames_$e.txt" "$cd/work_$e/"
        (
            cd "$cd/work_$e"
            # shellcheck disable=SC2086
            nohup ./molequla_cgo $flags --corpus "nonames_$e.txt" --db memory.sqlite3 --ckpt molequla_ckpt.json --element "$e" --evolution > train.log 2>&1 &
            echo $! > org.pid
        )
    done
    sleep "$DUR"
    echo "[$(date -u +%H:%M:%S)] === killing $label organisms ==="
    pkill -f molequla_cgo 2>/dev/null || true
    sleep 3
    for e in earth air water fire; do
        local l="$cd/work_$e/train.log"
        local lines=$(wc -l < "$l" 2>/dev/null || echo 0)
        local dna=$(grep -c "dna" "$l" 2>/dev/null || echo 0)
        local spa=$(grep -c "\[spa-gate\]" "$l" 2>/dev/null || echo 0)
        local mit=$(grep -cE "mitosis|spawning" "$l" 2>/dev/null || echo 0)
        local stage=$(grep -oE "stage=[0-9]+" "$l" | tail -1 || echo "stage=?")
        echo "$label/$e: lines=$lines dna=$dna spa-gate=$spa mitosis=$mit last=$stage"
    done
    echo
}

run_cell cell_0_baseline
run_cell cell_3_full_coherence --spa-gate --corpus-overlay
echo "=== ALL DONE at $(date -u +%H:%M:%S) ==="

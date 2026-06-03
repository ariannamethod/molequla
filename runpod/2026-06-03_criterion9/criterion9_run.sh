#!/usr/bin/env bash
# molequla Criterion 9 — natural embryo->adult->MITOSIS on GPU.
# Runs ON a fresh RunPod CUDA-12.4-devel pod. Builds notorch (USE_CUDA) +
# molequla from BRANCH molequla-rrpram-inc2 (Stage-A mitosis-enablement edits),
# launches the 4-organism cross-graze ecology, and lets it climb to adult +
# divide. NO corpus seeding. Everything tee'd to /workspace/criterion9.log.
set -uo pipefail
exec > >(tee -a /workspace/criterion9.log) 2>&1
echo "================ $(date -u) molequla Criterion 9 START ================"
export PATH=$PATH:/usr/local/cuda/bin:/usr/local/go/bin
export DEBIAN_FRONTEND=noninteractive

echo "--- env ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nvcc --version 2>/dev/null | tail -2 || echo "WARNING: nvcc missing (need CUDA -devel image)"

echo "--- deps ---"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git build-essential libopenblas-dev wget ca-certificates tmux >/dev/null 2>&1
if ! /usr/local/go/bin/go version 2>/dev/null | grep -q 'go1.2'; then
  wget -qO /tmp/go.tgz https://go.dev/dl/go1.23.4.linux-amd64.tar.gz
  rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz
fi
export PATH=$PATH:/usr/local/go/bin
go version

echo "--- notorch (main: op-33 + g_gpu_dispatch counter) ---"
cd /workspace && rm -rf notorch
git clone -q https://github.com/ariannamethod/notorch.git
cd /workspace/notorch && echo "notorch HEAD: $(git log --oneline -1)"
make USE_CUDA=1 2>&1 | tail -3
make install PREFIX=/usr/local USE_CUDA=1 2>&1 | tail -2
echo "libnotorch_gpu.a: $(ls -la /usr/local/lib/libnotorch_gpu.a 2>&1)"
ldconfig 2>/dev/null || true

echo "--- molequla BRANCH molequla-rrpram-inc2 (Stage-A edits) — CUDA build ---"
cd /workspace && rm -rf molequla
git clone -q https://github.com/ariannamethod/molequla.git
cd /workspace/molequla && git checkout molequla-rrpram-inc2 && echo "molequla HEAD: $(git log --oneline -1)"
if CGO_ENABLED=1 go build -tags cuda -o /workspace/molq_gpu . 2>&1 | tail -8; then :; fi
echo "build exit: ${PIPESTATUS[0]}  molq_gpu: $(ls -la /workspace/molq_gpu 2>&1)"
if [ ! -x /workspace/molq_gpu ]; then echo "FATAL: build failed, aborting"; exit 1; fi

echo "--- 4-organism cross-graze ecology (NO seeding) ---"
ECO=/workspace/eco; rm -rf "$ECO"; mkdir -p "$ECO/dna"
for e in earth air water fire; do
  W="$ECO/work_$e"; mkdir -p "$W"
  cp /workspace/molq_gpu "$W/molq"
  cp /workspace/molequla/nonames_$e.txt "$W/"
done
for e in earth air water fire; do
  ( cd "$ECO/work_$e" && nohup ./molq --evolution --element "$e" --cross-graze \
      --db memory.sqlite3 --ckpt molequla_ckpt.json > train.log 2>&1 & echo $! > org.pid )
  sleep 5
done
echo "=== 4 orgs launched $(date -u) — PIDs: $(cat $ECO/work_*/org.pid | tr '\n' ' ') ==="
echo "monitor: tail -f $ECO/work_earth/train.log ; grep -hE 'ONTOGENESIS|overload|MITOSIS|spawned' $ECO/work_*/train.log"
echo "================ $(date -u) launch complete, ecology running ================"

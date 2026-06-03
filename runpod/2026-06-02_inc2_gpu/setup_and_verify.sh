#!/usr/bin/env bash
# molequla Increment 2 — GPU verify, runs ON a fresh RunPod CUDA-devel pod.
# Builds notorch (USE_CUDA) + molequla (-tags cuda) from main, runs a short
# ecology, and surfaces the op-33 cuBLAS dispatch + loss. Everything tee'd to
# /workspace/gpu_verify.log.
set -uo pipefail
exec > >(tee -a /workspace/gpu_verify.log) 2>&1
echo "================ $(date -u) molequla Inc2 GPU verify START ================"
cd /workspace
# nvcc lives at /usr/local/cuda/bin (CUDA 12.4 devel image), not on default PATH.
export PATH=$PATH:/usr/local/cuda/bin:/usr/local/go/bin

echo "--- environment ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nvcc --version 2>/dev/null | tail -2 || echo "WARNING: nvcc not found (need a CUDA -devel image)"

echo "--- deps (git/build/openblas + Go 1.23.4) ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git build-essential libopenblas-dev wget ca-certificates >/dev/null 2>&1
if ! /usr/local/go/bin/go version 2>/dev/null | grep -q 'go1.2'; then
  wget -qO /tmp/go.tgz https://go.dev/dl/go1.23.4.linux-amd64.tar.gz
  rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz
fi
export PATH=$PATH:/usr/local/go/bin
go version

echo "--- notorch (main: op-33 + g_gpu_dispatch counter) ---"
rm -rf /workspace/notorch
git clone -q https://github.com/ariannamethod/notorch.git
cd /workspace/notorch && echo "notorch HEAD: $(git log --oneline -1)"
make USE_CUDA=1 2>&1 | tail -4
make install PREFIX=/usr/local USE_CUDA=1 2>&1 | tail -2
echo "libnotorch_gpu.a: $(ls -la /usr/local/lib/libnotorch_gpu.a 2>&1)"
ldconfig 2>/dev/null || true

echo "--- molequla (main: Inc2) — CUDA build ---"
cd /workspace
rm -rf /workspace/molequla
git clone -q https://github.com/ariannamethod/molequla.git
cd /workspace/molequla && echo "molequla HEAD: $(git log --oneline -1)"
if CGO_ENABLED=1 go build -tags cuda -o /workspace/molq_gpu . 2>&1 | tail -8; then
  echo "build exit: ${PIPESTATUS[0]}"
fi
echo "molq_gpu: $(ls -la /workspace/molq_gpu 2>&1)"

echo "--- GPU verify run: ecology, RRPRAM active at infant, ~4 min ---"
mkdir -p /workspace/run && cd /workspace/run
cp /workspace/molequla/nonames*.txt . 2>/dev/null || true
timeout 260 /workspace/molq_gpu > /workspace/run/eco_gpu.log 2>&1 < <(sleep 250; printf 'the cave holds\n') || true

echo "================ KEY LINES ================"
echo "[startup GPU enable]"; grep -iE "trainer on GPU|gpu_init|on CPU/BLAS" /workspace/run/eco_gpu.log | head -3
echo "[growth]"; grep -iE "ONTOGENESIS" /workspace/run/eco_gpu.log | head
echo "[notorch bursts — dispatch must climb]"; grep -iE "warmup complete|burst complete" /workspace/run/eco_gpu.log | tail -20
echo "[NaN/crash check]"; grep -cE "NaN|panic|SIGSEGV|fatal error" /workspace/run/eco_gpu.log
echo "[gpu mem]"; nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
echo "================ $(date -u) END ================"

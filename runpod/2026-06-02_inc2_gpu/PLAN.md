# molequla Increment 2 — GPU verify run plan (2026-06-02, polygon → RunPod)

Goal (06_PLAN §11 criterion 4 + §7 speed): confirm the low-rank RRPRAM op-33
**dispatches to cuBLAS on GPU** (the `nt_gpu_dispatch_count` counter climbs),
the organism trains with **0 NaN** on GPU, and measure the **speedup vs CPU** at
the RRPRAM-bearing stages. CPU side already proven (train≡infer parity 1.49e-8,
live ecology smoke green).

## State
- `origin/main` molequla `3b54bf5` (Inc2 complete), notorch `main 41145f8`
  (op-33 + `g_gpu_dispatch` counter). The pod builds both from `main`.
- Prior pod `8yz6rzinzkj0sj` is **gone** (reclaimed after >13d stopped) — fresh pod.
- Balance $13.49, spendLimit $80 (`runpodctl me`). Budget for this verify: ~$2-3.

## Recipe (grounded, not guessed)
- molequla CUDA build (`cgo_notorch_cuda.go:12-13`): `go build -tags cuda`,
  LDFLAGS `-lnotorch_gpu -lcudart -lcublas -lstdc++ -lopenblas`, needs
  `/usr/local/lib/libnotorch_gpu.a`, `/usr/local/cuda/{include,lib64}`, openblas.
- notorch CUDA build (`notorch/CLAUDE.md`): `make USE_CUDA=1` →
  `libnotorch.a + libnotorch_gpu.a + notorch_cuda.o`; `make install PREFIX=/usr/local USE_CUDA=1`.
- Env (HANDOFF_2026_05_19): CUDA 12.4, Go 1.23.4.
- GPU: cheapest available CUDA card (model is ~10M params at adult — any GPU is
  oversized; RTX 3090 / A4000-class). Image: CUDA 12.4 **devel** (needs `nvcc`).

## Steps
1. `runpodctl pod create` — CUDA-devel image, cheap GPU, `--startSSH`, ~30-40GB disk.
2. SSH in, run `setup_and_verify.sh` (clones main of both repos, builds, runs a
   ~4-min ecology, greps dispatch + loss).
3. Pull `gpu_verify.log` + `eco_gpu.log` here. **Stop the pod** (verify volume,
   per `feedback_pod_stop_volume_zero_artifact_loss`).
4. Record: dispatch-count delta, CPU-vs-GPU steps/s, 0 NaN. Memory milestone.

## Pass criteria (this verify)
- `trainer on GPU` printed at startup (ntGPUEnable success).
- `gpu-dispatch=N` with N climbing across bursts (op-33 + content reached cuBLAS).
- finite descending loss, 0 NaN / panic.
- steps/s reported at infant+ (compare to polygon CPU ~88-90 steps/s infant).

Criterion 9 (embryo→adult+mitosis, paper Section 9) is a SEPARATE, longer
attended run — gated on this verify passing.

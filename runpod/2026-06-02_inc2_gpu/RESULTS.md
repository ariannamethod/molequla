# molequla Increment 2 — GPU verify RESULTS (2026-06-02)

Pod `hl0bz261xf3t5z` (RTX 3090 24GB, community CA, $0.22/hr,
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`). ~7 min total,
**≈ $0.03**. Removed after pull (`runpodctl pod delete`, `pod list` empty).
Raw logs: `gpu_verify.log`, `eco_gpu.log` (this dir).

## Build (from main, fresh pod)
- notorch `41145f8` → `make USE_CUDA=1` → `libnotorch_gpu.a` (344968 B) installed.
- molequla `3b54bf5` (Inc2) → `go build -tags cuda` → exit 0, `molq_gpu` 10.48 MB.
- nvcc 12.4.131, Go 1.23.4. First-try clean (no build debugging).

## Verify run (ecology, embryo→infant, RRPRAM active)
Startup: `[notorch] notorch trainer on GPU — tape dispatching to cuBLAS`
(`ntGPUEnable()` → `gpu_init()` → `nt_set_gpu_mode(1)`).

| event | avg loss | steps/s (GPU) | gpu-dispatch |
|---|---|---|---|
| embryo warmup | 5.0979 | 190.3 | 15822 |
| embryo warmup | 4.2078 | 282.9 | 27693 |
| embryo warmup | 3.4015 | 299.3 | 39123 |
| → ONTOGENESIS 0→1 (infant, RRPRAM on) | | | |
| infant warmup | 1.6128 | 242.1 | 78658 |
| infant warmup | 1.3889 | 244.9 | 108178 |
| infant warmup | 1.4444 | 250.3 | 137558 |
| infant ecology burst | 2.8162 | 127.7 | 141660 |

## Criteria (06_PLAN §11)
- **4 — GPU dispatch confirmed** ✅. Counter climbs 15822→141660. The
  embryo→infant jump (39123→78658, +39535 over 320 steps vs +11430 over 120 at
  embryo) is the op-33 RRPRAM head reaching cuBLAS once hybrid heads appear.
- **2 — speed** ✅. Infant GPU ~242-250 steps/s vs polygon CPU ~88-90 (CPU smoke)
  → **≈ 2.8× faster** at infant; the gap grows with size (embryo GPU is slower —
  matmuls too small to amortise kernel launch, as predicted). GPU payoff begins
  at infant and increases toward teen/adult.
- **0 NaN / panic / SIGSEGV** on GPU ✅ (`grep -c` = 0).
- Loss finite + descending; train≡infer architecture runs correctly on device.

## Not done (separate, gated on Oleg)
- Criterion 9 — embryo→adult→mitosis on GPU, paper Section 9. Multi-hour
  attended run; provisioning is scripted (`setup_and_verify.sh` → adapt to a
  longer ecology + watchdog). Recreate a pod when greenlit.
- notorch `nt_sigmoid`/`nt_scale_by_t` GPU-sync fix (branch
  `notorch-gpu-sync-sigmoid-scale 98f3007`) — not needed by frozen-gate Inc2,
  not exercised here; relevant only for a future trainable gate.

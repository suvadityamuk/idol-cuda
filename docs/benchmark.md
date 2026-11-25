# Benchmark (H200, one session, interleaved)

**Setup.** HF Jobs `h200` job 6b0687b0 (2026-11-25 07:02-07:34 UTC): NVIDIA H200 141 GB, driver 595.112.07, max SM
clock 1980 MHz, 700 W limit, 23-vCPU quota; image `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel`, torch
2.10.0+cu130, CUDA 13.0, cuDNN 9.15.1, Pillow 13.0.0 (full list: `logs/bench/pip-freeze.txt`); source commit 66f3e90
of this repository; idol-sdk 8db7003. Driver: `tools/bench.py` (raw JSON lines: `logs/bench/bench.jsonl`; tables:
`tools/bench_summary.py`).

**Method.** All six builds are loaded once as long-lived worker processes (load time reported separately). Per
config: one warm-up request per worker (reported as "first request": includes torch.compile + autotune for the
shipped default, cuDNN plan building, allocator growth), then 5 rounds; in each round every build runs one request,
in an order that rotates and reverses from round to round. "request" = image path in -> Gaussians, CUDA-synced (what
`IdolPipeline.__call__` covers: preprocessing, BiRefNet, encoders, sampling, both decoders); "+ export" = writing the
PLY, the input PNG and the camera JSON (to local disk). Seed 42, 50 steps, CFG 3.0, rescale_t 3.0. Speed-up =
median request time of the shipped default / median of the build. Peak memory = `torch.cuda.max_memory_allocated`
/ `max_memory_reserved` over the request (native: the same CUDACachingAllocator counters).

Rows: the official pipeline as shipped (default `optimize=True`: torch.compile of the denoiser blocks, batched CFG,
cuDNN attention, TF32) and with `--no_optimize` (eager, the parity reference), both with `ATTN_BACKEND=sdpa` (flash-attn
is not an idol-sdk dependency); the two parity builds; the two fast builds. The Python fast build uses
FlashAttention-3 (Hub kernel `kernels-community/flash-attn3`); the native fast build uses cuDNN attention: loading
that kernel's shared library into the native binary failed (`undefined symbol: PyModule_Create2`: the library
carries a Python module init that binds libpython symbols at load time; `logs/bench/native_fa3_smoke.txt`), so the
benchmark ran it on its cuDNN fallback.

Session: NVIDIA H200, 595.112.07, torch 2.10.0+cu130, CUDA 13.0, cuDNN 91501, commit 66f3e90

### robot_262k

| build | request s, median (min-max) | + export s | speed-up vs shipped default | first request s | load s | peak alloc / reserved GB | PLY |
|---|---|---|---|---|---|---|---|
| reference, shipped default (torch.compile + batched CFG + cuDNN attn + TF32) | 6.470 (6.461-6.479), n=5 | 0.130 | 1.00x | 21.46 | 56.3 | 5.02 / 6.72 | 1 distinct sha over 5 runs |
| reference `--no_optimize` (eager) | 9.599 (9.593-9.617), n=5 | 0.148 | 0.67x | 11.13 | 17.2 | 5.02 / 6.72 | byte-identical to eager ref |
| parity, native C++ (`run_native.sh --parity`) | 9.184 (9.153-9.190), n=5 | 0.043 | 0.70x | 10.88 | 7.5 | 4.67 / 5.97 | byte-identical to eager ref |
| parity, Python (`run_kernels.py --parity`) | 9.147 (9.120-9.173), n=5 | 0.151 | 0.71x | 10.53 | 15.8 | 5.02 / 6.72 | byte-identical to eager ref |
| **fast, Python + idol_cuda kernels** (`run_kernels.py`) | 4.221 (4.219-4.233), n=5 | 0.084 | 1.53x | 5.49 | 35.6 | 5.08 / 6.79 | 1 distinct sha over 5 runs |
| fast, native C++ (`run_native.sh`) | 4.747 (4.736-4.755), n=5 | 0.035 | 1.36x | 6.81 | 7.6 | 4.70 / 6.00 | 1 distinct sha over 5 runs |

### avocado_262k

| build | request s, median (min-max) | + export s | speed-up vs shipped default | first request s | load s | peak alloc / reserved GB | PLY |
|---|---|---|---|---|---|---|---|
| reference, shipped default (torch.compile + batched CFG + cuDNN attn + TF32) | 6.512 (6.498-6.517), n=5 | 0.106 | 1.00x | 6.71 | 56.3 | 5.02 / 6.72 | 1 distinct sha over 5 runs |
| reference `--no_optimize` (eager) | 9.639 (9.591-9.672), n=5 | 0.125 | 0.68x | 9.84 | 17.2 | 5.02 / 6.72 | byte-identical to eager ref |
| parity, native C++ (`run_native.sh --parity`) | 9.196 (9.183-9.206), n=5 | 0.039 | 0.71x | 9.41 | 7.5 | 4.67 / 5.97 | byte-identical to eager ref |
| parity, Python (`run_kernels.py --parity`) | 9.162 (9.147-9.185), n=5 | 0.121 | 0.71x | 9.37 | 15.8 | 5.02 / 6.72 | byte-identical to eager ref |
| **fast, Python + idol_cuda kernels** (`run_kernels.py`) | 4.234 (4.229-4.249), n=5 | 0.057 | 1.54x | 4.42 | 35.6 | 5.08 / 6.79 | 1 distinct sha over 5 runs |
| fast, native C++ (`run_native.sh`) | 4.767 (4.760-4.786), n=5 | 0.029 | 1.37x | 4.99 | 7.6 | 4.70 / 6.00 | 1 distinct sha over 5 runs |

### robot_1m

| build | request s, median (min-max) | + export s | speed-up vs shipped default | first request s | load s | peak alloc / reserved GB | PLY |
|---|---|---|---|---|---|---|---|
| reference, shipped default (torch.compile + batched CFG + cuDNN attn + TF32) | 6.750 (6.726-6.786), n=5 | 0.445 | 1.00x | 7.09 | 56.3 | 5.06 / 6.72 | 1 distinct sha over 5 runs |
| reference `--no_optimize` (eager) | 9.894 (9.884-9.900), n=5 | 0.477 | 0.68x | 10.17 | 17.2 | 5.06 / 6.72 | byte-identical to eager ref |
| parity, native C++ (`run_native.sh --parity`) | 9.431 (9.427-9.455), n=5 | 0.124 | 0.72x | 9.71 | 7.5 | 4.67 / 5.97 | byte-identical to eager ref |
| parity, Python (`run_kernels.py --parity`) | 9.425 (9.410-9.446), n=5 | 0.467 | 0.72x | 9.70 | 15.8 | 5.06 / 6.72 | byte-identical to eager ref |
| **fast, Python + idol_cuda kernels** (`run_kernels.py`) | 4.467 (4.463-4.472), n=5 | 0.174 | 1.51x | 4.71 | 35.6 | 5.12 / 6.79 | 1 distinct sha over 5 runs |
| fast, native C++ (`run_native.sh`) | 4.988 (4.966-5.004), n=5 | 0.086 | 1.35x | 5.27 | 7.6 | 4.70 / 6.00 | 1 distinct sha over 5 runs |

### avocado_1m

| build | request s, median (min-max) | + export s | speed-up vs shipped default | first request s | load s | peak alloc / reserved GB | PLY |
|---|---|---|---|---|---|---|---|
| reference, shipped default (torch.compile + batched CFG + cuDNN attn + TF32) | 6.794 (6.757-6.813), n=5 | 0.408 | 1.00x | 7.01 | 56.3 | 5.06 / 6.72 | 1 distinct sha over 5 runs |
| reference `--no_optimize` (eager) | 9.930 (9.880-9.955), n=5 | 0.426 | 0.68x | 10.11 | 17.2 | 5.06 / 6.72 | byte-identical to eager ref |
| parity, native C++ (`run_native.sh --parity`) | 9.504 (9.469-9.513), n=5 | 0.115 | 0.71x | 9.67 | 7.5 | 4.67 / 5.97 | byte-identical to eager ref |
| parity, Python (`run_kernels.py --parity`) | 9.458 (9.412-9.478), n=5 | 0.404 | 0.72x | 9.66 | 15.8 | 5.06 / 6.72 | byte-identical to eager ref |
| **fast, Python + idol_cuda kernels** (`run_kernels.py`) | 4.489 (4.469-4.528), n=5 | 0.147 | 1.51x | 4.70 | 35.6 | 5.12 / 6.79 | 1 distinct sha over 5 runs |
| fast, native C++ (`run_native.sh`) | 5.026 (5.005-5.046), n=5 | 0.082 | 1.35x | 5.18 | 7.6 | 4.70 / 6.00 | 1 distinct sha over 5 runs |


"PLY byte-identical to eager ref" compares the sha256 of every one of the 5 timed outputs with the stage-1 eager
reference dump's PLY (`logs/stage1/job.log`). Every build produced one PLY per config across the 5 rounds (the fast
builds are run-to-run deterministic in this session).

**Headline (robot, 262,144 Gaussians):** fast (Python + idol_cuda kernels) 4.22 s vs 6.47 s for the shipped default,
**1.53x**; with export 4.31 vs 6.60 s (1.53x). Native fast 4.75 s (1.36x). The parity builds are 1.05x faster than the
eager reference they match byte for byte (9.15-9.18 vs 9.60 s) and 0.70x of the shipped default. First-request cost:
shipped default 21.5 s (compile), fast 5.5 s, native fast 6.8 s; load (workers start one after another): shipped default 56 s, eager reference 17 s, Python fast 36 s (includes
fetching the FlashAttention-3 kernel from the local cache and the kernel package), native builds 7.5 s.

**Where the fast build's time goes** (Python fast, second request, dev8 `logs/dev8/run_fast_robot_262k.json`):
sampling 3.95 s (93%), preprocess 0.14 s, encoders 0.08 s, octree 0.04 s, Gaussian decoder 0.04 s (0.27 s at 1M),
export 0.08 s. Sampling at 77-79 ms per step is ~65% attention (FlashAttention-3 at ~420-490 TFLOP/s), ~20% fp16
GEMMs, the rest the fused kernels (`logs/dev3/profile_fast.txt` for the cuDNN variant).

## Closeness to the reference

Tolerance = the reference's own spread: the largest difference between the eager reference and the variants the
official code itself ships (compiled default, run A and B; eager with flash-attn instead of SDPA), and the compiled
default's run-to-run difference. Metrics: rel L2 of the sampled latent [8192 x 16], absolute camera error, symmetric
mean nearest-neighbour distance between Gaussian centres (Chamfer; the scene spans [-0.5, 0.5]^3). Fast rows: Python
fast from dev8 (same code as the benchmark, commit before the benchmark's), native fast from dev7; `tools/closeness.py`.

### robot_262k

| comparison | latent rel L2 | camera abs err: yaw / pitch (rad) / radius / fov (deg) | Chamfer, Gaussian centres |
|---|---|---|---|
| ref: compiled (shipped) run A vs eager | 1.43e-02 | 8.0e-04 / 1.2e-03 / 3.7e-03 / 0.086 | 1.68e-03 |
| ref: compiled run B vs eager | 1.43e-02 | 8.0e-04 / 1.2e-03 / 3.7e-03 / 0.086 | 1.68e-03 |
| ref: eager flash-attn vs eager sdpa | 6.09e-03 | 8.8e-04 / 8.0e-04 / 2.8e-03 / 0.069 | 1.68e-03 |
| ref: compiled A vs compiled B (run-to-run) | 0.00e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 / 0.000 | 1.98e-05 |
| **fast, Python (FA3)** vs eager | 6.95e-03 | 3.2e-04 / 4.2e-04 / 2.5e-03 / 0.045 | 1.69e-03 |
| fast, native (cuDNN) vs eager | 1.57e-02 | 1.1e-03 / 1.9e-03 / 2.5e-04 / 0.011 | 1.68e-03 |

### avocado_262k

| comparison | latent rel L2 | camera abs err: yaw / pitch (rad) / radius / fov (deg) | Chamfer, Gaussian centres |
|---|---|---|---|
| ref: compiled (shipped) run A vs eager | 4.62e-03 | 1.9e-04 / 1.6e-03 / 1.2e-02 / 0.115 | 1.89e-03 |
| ref: compiled run B vs eager | 4.62e-03 | 1.9e-04 / 1.6e-03 / 1.2e-02 / 0.115 | 1.89e-03 |
| ref: eager flash-attn vs eager sdpa | 3.81e-03 | 8.0e-04 / 6.6e-04 / 7.7e-03 / 0.036 | 1.88e-03 |
| ref: compiled A vs compiled B (run-to-run) | 0.00e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 / 0.000 | 1.94e-05 |
| **fast, Python (FA3)** vs eager | 5.63e-03 | 9.9e-04 / 2.1e-04 / 1.8e-02 / 0.179 | 1.89e-03 |
| fast, native (cuDNN) vs eager | 4.28e-03 | 5.2e-04 / 5.3e-04 / 1.5e-02 / 0.157 | 1.88e-03 |

### robot_1m

| comparison | latent rel L2 | camera abs err: yaw / pitch (rad) / radius / fov (deg) | Chamfer, Gaussian centres |
|---|---|---|---|
| ref: compiled (shipped) run A vs eager | 1.43e-02 | 8.0e-04 / 1.2e-03 / 3.7e-03 / 0.086 | 1.01e-03 |
| ref: compiled run B vs eager | 1.43e-02 | 8.0e-04 / 1.2e-03 / 3.7e-03 / 0.086 | 1.01e-03 |
| ref: eager flash-attn vs eager sdpa | 6.09e-03 | 8.8e-04 / 8.0e-04 / 2.8e-03 / 0.069 | 1.00e-03 |
| ref: compiled A vs compiled B (run-to-run) | 0.00e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 / 0.000 | 1.97e-05 |
| **fast, Python (FA3)** vs eager | 6.95e-03 | 3.2e-04 / 4.2e-04 / 2.5e-03 / 0.045 | 1.01e-03 |
| fast, native (cuDNN) vs eager | 1.57e-02 | 1.1e-03 / 1.9e-03 / 2.5e-04 / 0.011 | 1.00e-03 |

### avocado_1m

| comparison | latent rel L2 | camera abs err: yaw / pitch (rad) / radius / fov (deg) | Chamfer, Gaussian centres |
|---|---|---|---|
| ref: compiled (shipped) run A vs eager | 4.62e-03 | 1.9e-04 / 1.6e-03 / 1.2e-02 / 0.115 | 1.19e-03 |
| ref: compiled run B vs eager | 4.62e-03 | 1.9e-04 / 1.6e-03 / 1.2e-02 / 0.115 | 1.19e-03 |
| ref: eager flash-attn vs eager sdpa | 3.81e-03 | 8.0e-04 / 6.6e-04 / 7.7e-03 / 0.036 | 1.18e-03 |
| ref: compiled A vs compiled B (run-to-run) | 0.00e+00 | 0.0e+00 / 0.0e+00 / 0.0e+00 / 0.000 | 1.94e-05 |
| **fast, Python (FA3)** vs eager | 5.63e-03 | 9.9e-04 / 2.1e-04 / 1.8e-02 / 0.179 | 1.18e-03 |
| fast, native (cuDNN) vs eager | 4.28e-03 | 5.2e-04 / 5.3e-04 / 1.5e-02 / 0.157 | 1.18e-03 |

Cross-check: the benchmark session's own first-round PLYs (`logs/scripts_check/close/`) give the same camera errors
and Chamfer distances as these runs for both fast builds and the compiled default (to 3 digits), i.e. the rows above
describe the benchmarked code.

Reading: Chamfer distance of both fast builds equals the reference's own spread on every config. Final latent and
camera: the Python fast build (FlashAttention-3) is inside the band on robot (latent 6.95e-3 vs up to 1.43e-2) and
outside it on avocado (latent 5.63e-3 vs at most 4.62e-3, 1.22x; camera radius 1.8e-2 vs 1.2e-2 and fov 0.179 vs
0.115 deg, about 1.5x). The native fast build (cuDNN) is inside on avocado latent (4.28e-3) and outside on robot
latent (1.57e-2 vs 1.43e-2, 1.10x) and on avocado fov (0.157 vs 0.115). Per call (teacher-forced at steps 0, 1, 10,
25, 49, docs/ab-log.md) both fast builds are closer to the eager reference than the shipped compiled path on both
images at every step checked; the end-to-end differences come from the 50-step integration amplifying per-step
differences, which also makes a single end-to-end sample per variant a noisy estimate of the band.

## Rendered views (end-task metric)

gsplat (`pip install gsplat` at job time; the version was not recorded), `tools/render_psnr.py`: 8 fixed cameras (radius 2,
elevation +20 / -10 degrees, azimuth 0 / 90 / 180 / 270, 40 degree FOV, 512 x 512, black background, SH degree 0),
each PLY rendered and compared with the eager reference PLY's renders (PSNR over the 8 views averaged, SSIM 11x11
Gaussian window). PLYs: first timed round of the benchmark session (`/scratch/bench_out`), flash-attn reference from
the spread runs. Raw: `logs/scripts_check/render_*.json`.

| rendered vs eager reference: PSNR dB / SSIM | robot 262k | avocado 262k | robot 1M | avocado 1M |
|---|---|---|---|---|
| eager reference run B (same code) | identical | identical | identical | identical |
| shipped default (compiled) | 42.76 / 0.9963 | 42.82 / 0.9813 | 46.99 / 0.9982 | 43.86 / 0.9862 |
| eager with flash-attn | 42.98 / 0.9962 | 42.73 / 0.9812 | 47.19 / 0.9983 | 44.02 / 0.9867 |
| **fast, Python (FA3)** | 42.58 / 0.9961 | 42.77 / 0.9812 | 47.23 / 0.9983 | 43.86 / 0.9864 |
| fast, native (cuDNN) | 42.92 / 0.9962 | 42.81 / 0.9813 | 47.05 / 0.9982 | 44.01 / 0.9866 |

The fast builds render as close to the eager reference as the official variants do: within 0.2 dB of the compiled
default and the flash-attn variant on robot 262k (42.58 vs 42.76 / 42.98 dB for Python fast; native 42.92 dB),
and inside their range on the other three configs. SSIM agrees to the fourth decimal.

## Not done or not verified

- FlashAttention-3 in the native binary (the Hub library needs libpython); vendoring FA3's hdim-64 forward kernel
  into the native build was not attempted. The native fast build runs cuDNN attention (1.35-1.37x instead of 1.51-1.54x).
- Fast builds outside the reference's spread on some end-to-end metrics (avocado latent 1.22x and camera fov/radius
  ~1.5x for the Python build; robot latent 1.10x and avocado fov for the native build); per-call and rendered-view
  metrics are inside. The band itself rests on one end-to-end sample per reference variant.
- Kernel package: only the torch210-cu130 variant was run on a GPU (H200, sm_90); the other 16 variants were built
  and passed `check-builds`, but were not loaded. Built for sm_90 only (`cuda-capabilities = ["9.0"]`) at the time of
  this benchmark; the release build (version 1) covers 8.0-12.0, and its torch210-cu128 variant also passed the kernel
  tests on sm_120 (REPORT.md, Release preparation).
- No Hub publishing, `kernels benchmark` (needs a Hub repo) not run; `benchmarks/benchmark.py` was not executed.
- CUDA graphs over a whole step were measured slower (4%) and dropped; the reason was not investigated.
- Any GPU other than the H200, and any torch / CUDA / cuDNN other than 2.10.0+cu130 / 13.0 / 9.15.1. Parity is claimed
  only for this tuple.
- The compiled reference showed run-to-run differences in stage 1 (robot 262k, octree draws only; latent identical)
  but was deterministic within the benchmark session; the cause was not investigated.

## Kernel-builder output vs the development build (job 6b0693dc, `logs/final/bench_kb.jsonl`)

The benchmark's Python fast row loaded the idol_cuda kernels from a `torch.utils.cpp_extension` build of the same
sources (sm_90a). A second, shorter interleaved session (robot 262k, 3 rounds) ran the shipped default, the fast build
on that development build, and the fast build on the kernel-builder output (`torch210-cu130` variant, loaded with
`kernels.get_local_kernel`): the PLYs of the two fast rows are byte-identical in all 3 rounds, and their times are
equal (median 5.44 vs 5.41 s). That session ran on a slower H200 node than the main one (every row ~27% slower:
shipped default 8.23 s median), so only its ratios are comparable: fast 1.52x the shipped default, as in the main
session. The same job re-ran `scripts/parity_test.sh` on the native binary built from commit bd5c29c: PARITY PASS on
all 4 configs (`logs/final/parity_test.txt`).

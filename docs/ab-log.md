# A/B log (fast build, CUDA / H200)

Method: `tools/ab_dit.py` in one process on the H200. Teacher-forced: each variant gets the eager reference's own
inputs at steps 0, 1, 10, 25, 49 (latent, camera, t, features from `/scratch/ref/<cfg>_eager_sdpa_a`, recorded with
`--dump_steps`) and its v_cond, v_uncond and guided velocity v = 4 v_cond - 3 v_uncond are compared with the dump
("rel" = relative L2). Timing: one sampling step (both CFG passes), variants interleaved, order flipped each
repetition, 2 warm-ups, 8 repetitions, CUDA-synced; median (min-max). End to end: 50 steps from the dumped noise, final
latent vs the eager reference's. The reference's own spread is measured in the same harness: `ref_compiled` is the
shipped optimize path (torch.compile + batched CFG + cuDNN attention + TF32) on the same inputs. `ref_eager` re-runs
the reference and is bit-equal to its dump (harness sanity check).

Raw: `logs/dev2/ab_robot_262k.json`, `logs/dev2/ab_avocado_262k.json`.

## Measurements (raw JSON: `logs/dev2/ab_*.json`, `logs/dev8/ab_fa3_*.json`, `logs/dev6/` FP8)

Variants: `fast` = everything below with cuDNN attention; `fast_nocollapse` = uncond keeps 4,101 context tokens;
`fast_flash` / `fast_fa2` = ATen FlashAttention-2 / flash-attn 2.8.3 instead of cuDNN; `fast_fa3` = FlashAttention-3
(Hopper) from the Hub kernel `kernels-community/flash-attn3` v1 (fp16; its log-sum-exp feeds `merge_extra_key`).

### dev2

| variant | step ms robot, median (min-max) | step ms avocado | guided rel robot, steps 0 / 25 / 49 | guided rel avocado, steps 0 / 25 / 49 | e2e final latent rel robot / avocado |
|---|---|---|---|---|---|
| fast | 89.7 (88.0-91.4) | 89.2 (88.2-91.4) | 1.33e-03 / 5.06e-03 / 8.36e-03 | 1.70e-03 / 4.45e-03 / 8.10e-03 | 1.57e-02 / 4.28e-03 |
| fast_nocollapse | 113.5 (110.3-114.3) | 113.0 (112.1-114.0) | 1.33e-03 / 5.08e-03 / 8.32e-03 | 1.69e-03 / 4.34e-03 / 8.10e-03 | 1.18e-02 / 4.79e-03 |
| fast_flash | 106.6 (106.1-107.6) | 106.8 (106.2-107.2) | 1.27e-03 / 5.09e-03 / 8.35e-03 | 1.67e-03 / 4.34e-03 / 8.05e-03 | 1.64e-02 / 7.88e-03 |
| fast_fa2 | 106.9 (104.7-108.7) | 106.6 (105.0-109.0) | 1.22e-03 / 5.15e-03 / 8.41e-03 | 1.68e-03 / 4.50e-03 / 8.13e-03 | 1.56e-02 / 4.82e-03 |
| ref_compiled | 124.1 (122.4-125.4) | 124.9 (123.0-126.1) | 1.66e-03 / 6.66e-03 / 1.00e-02 | 2.15e-03 / 5.34e-03 / 9.61e-03 |  |
| ref_eager | 185.8 (183.9-189.6) | 186.5 (183.8-188.1) | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.00e+00 / 0.00e+00 / 0.00e+00 |  |

### dev8

| variant | step ms robot, median (min-max) | step ms avocado | guided rel robot, steps 0 / 25 / 49 | guided rel avocado, steps 0 / 25 / 49 | e2e final latent rel robot / avocado |
|---|---|---|---|---|---|
| fast | 90.5 (86.2-94.4) | 88.8 (83.7-95.6) | 1.33e-03 / 5.06e-03 / 8.36e-03 | 1.70e-03 / 4.45e-03 / 8.10e-03 | 1.57e-02 / 4.28e-03 |
| fast_fa3 | 77.4 (76.1-78.5) | 79.1 (76.6-80.0) | 1.22e-03 / 4.89e-03 / 8.45e-03 | 1.64e-03 / 4.39e-03 / 8.07e-03 | 6.95e-03 / 5.63e-03 |
| ref_compiled | 125.0 (120.7-127.6) | 125.3 (122.8-130.3) | 1.66e-03 / 6.66e-03 / 1.00e-02 | 2.15e-03 / 5.34e-03 / 9.61e-03 |  |
| ref_eager | 185.6 (183.6-187.9) | 186.3 (184.1-189.2) | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.00e+00 / 0.00e+00 / 0.00e+00 |  |

FP8 attention (dev6, robot, FA3 with q/k/v cast to e4m3): 71.7 ms/step (69.0-74.6) vs 75.7 for fp16 FA3 in that
session, guided rel 5.31e-2 / 1.46e-1 / 2.28e-1 at steps 0 / 25 / 49, final latent rel 0.287. Rejected.

## Changes (cumulative in the fast build)

| # | change | speed (per step, robot, same session) | error vs eager reference |
|---|---|---|---|
| 1 | Context stream once per request; time embedding, adaLN, noise refiner, camera refiner once per step for both CFG passes; cond + uncond rows packed into one [20,488 x 1024] matrix | with #2: `fast_nocollapse` 113.5 vs 124.1 ms compiled (1.09x) | bit-equal reuse in the parity builds (docs/parity.md) |
| 2 | Fused kernels (idol_cuda): residual + gate + LayerNorm32 + adaLN modulation + next RePo LayerNorm in one pass; RePo angles + RoPE + q/k RMSNorm in one pass; fc1 + bias + GELU in the cuBLASLt epilogue; gate/content maps as one TF32 GEMM | included in #1's number | per call closer than the compiled reference at every checked step (table above) |
| 3 | Uncond pass with ONE context token: attention over the other 8,193 keys, then the context key merged exactly with multiplicity 4,101 (`merge_extra_key`) | 89.7 vs 113.5 ms (1.27x) | guided rel unchanged to 2 digits; e2e latent rel robot 1.57e-2 vs 1.18e-2 without, avocado 4.28e-3 vs 4.79e-3 |
| 4 | FlashAttention-3 instead of cuDNN | 77.4 vs 90.5 ms (1.17x) robot, 79.1 vs 88.8 (1.12x) avocado | per call equal or smaller; e2e latent rel robot 6.95e-3 (cuDNN 1.57e-2), avocado 5.63e-3 (cuDNN 4.28e-3) |
| 5 | Decode side: octree descent with loop-free systematic resampling (one rand draw per level, no `.item()` per count), Gaussian activations + PLY columns in one kernel, one-write PLY | octree 0.035 vs ~0.11-0.21 s; to_gaussians + export ~0.1 vs ~0.1-0.2 s (dev3, second request) | different random numbers for the octree draws (same law); Chamfer in docs/benchmark.md |
| | **fast total vs shipped compiled, per step** | **1.62x** (77.4 vs 125.0 ms, robot) | per call closer than compiled at every checked step |

The fast build's tolerance is the reference's own spread (docs/benchmark.md, closeness table).

## Rejected / negative

| idea | result | why dropped |
|---|---|---|
| FlashAttention-2 (ATen) instead of cuDNN | 106.6 vs 89.7 ms/step | cuDNN's sm90 wgmma kernel is faster at these shapes |
| flash-attn 2.8.3 package | 106.9 vs 89.7 ms/step | same |
| CUDA graph over a whole step (velocity + Euler update) | 50 steps 4.655 s vs 4.471 s without (robot), 4.652 vs 4.477 s (avocado) | slower (4%): the step is GPU-bound, so there is no launch time to save; why the replay is slower was not investigated |

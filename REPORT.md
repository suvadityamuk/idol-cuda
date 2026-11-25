# Idol on CUDA (H200): port report

The CUDA port of Idol is finished on an H200. Everything is committed in idol-kernels/idol-cuda (local git, HEAD 2230398). Nothing was published, and no HF Job is still running.

- **Speed:** the fast build is 1.51–1.54x faster than the official pipeline's compiled default end to end, on all four configs. Robot at 262k Gaussians takes 4.22 s per request against 6.47 s.
- **What the speed depends on:**
  - That margin needs a third-party Hub kernel, `kernels-community/flash-attn3`.
  - The native C++ fast build cannot load that kernel, so it runs cuDNN attention and reaches 1.35–1.37x.
- **Closeness:**
  - **Inside the reference's spread:** the rendered views and the Gaussian positions.
  - **Outside it on avocado** (the only image where this is true for the Python build): the final latent (1.22x the spread) and the predicted camera (about 1.5x on fov and radius).
- **Parity:** achieved at the first tier, with every output byte-compared.
  - The .ply and camera JSON of both parity builds (native and Python) are byte-identical to the official eager pipeline's on all four configs.
  - The native binary's regression check passes on the final commit.

## Setup and versions

- **Hardware and driver:** HF Jobs `h200`, NVIDIA H200 141 GB, driver 595.112.07, 23 vCPU.
- **Software:** image `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel`, torch 2.10.0+cu130, CUDA 13.0, cuDNN 9.15.1, transformers 5.24.1, diffusers 0.43.0, Pillow 13.0.0.
- **Reference:** idol-sdk at 8db7003, unmodified, with `ATTN_BACKEND=sdpa`. flash-attn is not an idol-sdk dependency, so plain SDPA is what ships.
- **Weights and data:** weights are mounted into each job from the Hub repos. A private Hub bucket carries code, dumps and results. The token was passed only as a secret, and a scan found it in no log or file.
- **Configs:** robot (synthetic) and avocado (real) images, at 262,144 and 1,048,576 Gaussians; seed 42, 50 steps, cfg 3.0, rescale_t 3.0. Every random draw is stored as data in the reference dumps.

## The previous agent's work

- **Its claim of a bit-identical build had no evidence behind it.** Its only H200 session was an interactive `sleep infinity` job that saved nothing: the bucket held only a probe file and there were no logs.
- **The C++ port itself was correct.** I rebuilt it and verified it from scratch, and it passed. I kept it as the parity build.

## Profile (docs/profile.md)

**The compiled default at steady state takes 6.61 s per request, which matches the published 6.6 s.**
- Sampling is 6.14 s (93%).
- Preprocessing is 0.17 s, CPU-bound in Pillow and BiRefNet.
- The encoders take 0.08 s.
- Octree sampling takes 0.21 s; the GPU is busy only 12% of that time because of a Python loop with `.item()` syncs.
- The Gaussian decoder takes 0.04 s, and `save_ply` 0.08 s.
- The first request in a process takes 20.6 s because of compilation. The eager path takes 9.88 s.

Inside sampling, 123 ms per step, the GPU is busy 99% of the time:
- cuDNN attention: 65% (2.81 ms per call at batch 2 and 12,294 tokens, about 440 TFLOP/s).
- fp16 GEMMs: 18%.
- Triton glue: about 12%.

**What Inductor leaves on the table.** There are no graph breaks. Each block compiles to 5 Triton kernels, 4 cuBLAS GEMMs and one cuDNN attention call, with recompiles for the 8,192, 4,101 and 12,294 token shapes. What it misses:
1. **Structure, the biggest item.** The unconditional pass carries 4,101 identical context tokens through 26 layers and 50 steps. The context stream depends only on the image but is recomputed every step for both guidance halves. The noise refiner, camera refiner and time embedding are computed twice per step on identical inputs.
2. **GELU outside the GEMM.** fc1's bias and GELU run as a separate pass (2.9%).
3. **A separate RePo graph.** RePo is compiled on its own, so its LayerNorm re-reads the block input, and its angles and cos/sin go through memory (about 4%).
4. **Launch overhead is not a lever:** sampling is GPU-bound. Dtype round-trips do not matter either.

**Determinism.**
- The eager reference gives byte-identical results between fresh processes on all four configs. `cudnn.deterministic=True` changes nothing, and the BiRefNet nondeterminism seen on ROCm does not occur here.
- The compiled default differed between runs once, during the first run of the whole pipeline (robot 262k, octree draws only; the latent was identical). Within the benchmark session it was deterministic. I did not investigate the cause.

## Changes (docs/ab-log.md)

Each change was tested on the reference's own recorded inputs at steps 0, 1, 10, 25 and 49, timed interleaved in one process, and run end to end.

**Kept:**
1. **Restructured sampler.** The context stream is computed once per request. The time embedding, adaLN, noise refiner and camera refiner are computed once per step and shared by both guidance passes. Conditional and unconditional rows are packed into one 20,488-row matrix.
2. **Fused kernels:**
   - residual + gate + LayerNorm + adaLN modulation + the next RePo LayerNorm in one pass;
   - RePo angles + RoPE + q/k RMSNorm in one pass;
   - fc1 bias + GELU in the cuBLASLt epilogue.

   Together, items 1 and 2 give 113.5 ms per step against 124.1 ms for the compiled default.
3. **One context token in the unconditional pass.** Attention runs over the other 8,193 keys, and the context key is merged exactly with multiplicity 4,101 using the log-sum-exp. This takes 89.7 ms against 113.5 ms (1.27x).
4. **FlashAttention-3** in place of cuDNN: 77.4 ms against 90.5 ms (1.17x) on robot, 79.1 against 88.8 ms on avocado.
5. **Decode side:** a loop-free octree sampler (0.035 s against about 0.2 s), and the Gaussian activations plus PLY columns fused into one kernel.

A fast step is 77–79 ms against 125 ms for the compiled default (1.62x). On the recorded inputs, every fast variant is closer to the eager reference than the compiled default at every checked step. For example, the guided velocity's relative error at robot step 0 is 1.22e-3 for fast against 1.66e-3 for compiled.

**Rejected:**

| Idea | Result |
|---|---|
| FlashAttention-2 (ATen or the flash-attn 2.8.3 package) | 106.6–106.9 ms per step against 89.7 ms with cuDNN |
| CUDA graph over a whole step | 4% slower (4.655 s against 4.471 s over 50 steps); the step is GPU-bound, and I did not investigate why the replay is slower |
| FP8 attention (FA3, e4m3) | 1.10x more speed, but 30–40x the error (guided rel 5.3e-2, final latent rel 0.29) |
| FlashAttention-3 in the native binary through dlopen | Fails with `undefined symbol: PyModule_Create2`, even with RTLD_LAZY; the binary now falls back to cuDNN with a warning |

## Benchmark (docs/benchmark.md; raw data in logs/bench/bench.jsonl)

Job 6b0687b0 ran on commit 66f3e90:
- All six builds ran as long-lived workers.
- There was one warm-up request per config, then 5 rounds with the order rotated and reversed each round.
- "Request" means image path in to Gaussians out, synced. Export (PLY, PNG, JSON) is reported separately.

Each cell gives the median request time in seconds (min–max), the speed-up over the compiled default, and peak allocated memory in GB.

| Build | Robot 262k | Avocado 262k | Robot 1M | Avocado 1M |
|---|---|---|---|---|
| Shipped default (compiled) | 6.470 (6.461–6.479), 1.00x, 5.02 | 6.512 (6.498–6.517), 1.00x | 6.750 (6.726–6.786), 1.00x, 5.06 | 6.794 (6.757–6.813), 1.00x |
| `--no_optimize` (eager) | 9.599 (9.593–9.617), 0.67x | 9.639, 0.68x | 9.894, 0.68x | 9.930, 0.68x |
| Parity, native C++ | 9.184 (9.153–9.190), 0.70x, 4.67 | 9.196, 0.71x | 9.431, 0.72x | 9.504, 0.71x |
| Parity, Python | 9.147 (9.120–9.173), 0.71x | 9.162, 0.71x | 9.425, 0.72x | 9.458, 0.72x |
| **Fast, Python + kernels + FA3** | **4.221 (4.219–4.233), 1.53x, 5.08** | **4.234 (4.229–4.249), 1.54x** | **4.467 (4.463–4.472), 1.51x, 5.12** | **4.489 (4.469–4.528), 1.51x** |
| Fast, native C++ (cuDNN) | 4.747 (4.736–4.755), 1.36x, 4.70 | 4.767, 1.37x | 4.988, 1.35x | 5.026, 1.35x |

**Export, robot 262k:** 0.130 s for the compiled default, 0.084 s for Python fast, 0.035 s for native fast. At 1M Gaussians: 0.445, 0.174 and 0.086 s.

**First-request cost and load time (robot 262k):**

| Build | First request | Load |
|---|---|---|
| Compiled default | 21.46 s (includes compilation) | 56 s |
| Eager | 11.1 s | 17 s |
| Native parity | 10.9 s | 7.5 s |
| Python fast | 5.49 s | 36 s |
| Native fast | 6.81 s | 7.6 s |

Every row was run-to-run deterministic within the session: one PLY sha per config.

**A second, shorter session (job 6b0693dc)** compared the kernel-builder output with the development build of the same kernels:
- The PLYs were byte-identical in all 3 rounds.
- The medians were equal: 5.41 s against 5.44 s.
- That session landed on a slower H200, with every row about 27% slower (compiled default 8.23 s). The fast-to-default ratio was still 1.52x.

## Closeness (docs/benchmark.md)

**The tolerance band** is the reference's own spread, measured against the eager reference:
- compiled default, runs A and B;
- eager with flash-attn instead of SDPA;
- compiled A against compiled B.

| Comparison | Latent rel L2, robot / avocado | Camera fov error (deg), robot / avocado | Chamfer, robot 262k / avocado 262k |
|---|---|---|---|
| Compiled vs eager | 1.43e-2 / 4.62e-3 | 0.086 / 0.115 | 1.68e-3 / 1.89e-3 |
| Eager flash-attn vs eager SDPA | 6.09e-3 / 3.81e-3 | 0.069 / 0.036 | 1.68e-3 / 1.88e-3 |
| Compiled A vs compiled B | 0 / 0 | 0 / 0 | 2e-5 / 2e-5 |
| **Fast, Python (FA3)** | 6.95e-3 / **5.63e-3** | 0.045 / **0.179** | 1.69e-3 / 1.89e-3 |
| Fast, native (cuDNN) | **1.57e-2** / 4.28e-3 | 0.011 / **0.157** | 1.68e-3 / 1.88e-3 |

- At 1M Gaussians the Chamfer values are 1.00–1.19e-3 for every row.
- Camera radius on avocado: fast 1.8e-2 against 1.2e-2 for the compiled default.

**Rendered views.** I installed gsplat in a job; the version was not recorded. Each PLY was rendered from 8 fixed cameras and compared with the eager reference's renders:

| Build | Robot 262k | Avocado 262k | Robot 1M | Avocado 1M |
|---|---|---|---|---|
| Compiled default | 42.76 dB / 0.9963 | 42.82 / 0.9813 | 46.99 / 0.9982 | 43.86 / 0.9862 |
| Eager flash-attn | 42.98 / 0.9962 | 42.73 / 0.9812 | 47.19 / 0.9983 | 44.02 / 0.9867 |
| Fast, Python | 42.58 / 0.9961 | 42.77 / 0.9812 | 47.23 / 0.9983 | 43.86 / 0.9864 |
| Fast, native | 42.92 / 0.9962 | 42.81 / 0.9813 | 47.05 / 0.9982 | 44.01 / 0.9866 |

**Outside the band, stated plainly:**
- **Python fast on avocado:** final latent 1.22x the band, camera fov and radius about 1.5x.
- **Native fast:** robot latent 1.10x the band, and avocado fov.
- **Inside the band everywhere:**
  - Chamfer and rendered PSNR, within 0.2 dB of the official variants.
  - The per-step error on the recorded inputs.
- **The band itself is thin:** it rests on one end-to-end sample per reference variant.

## Parity (docs/parity.md), first tier

**Definition.** Parity is defined against eager `--no_optimize` with SDPA, for this exact combination: H200, CUDA 13.0, cuDNN 9.15.1, torch 2.10.0+cu130.

**Native build** (`idol_native --build parity`): an op-for-op libtorch port.
- The image networks run as TorchScript traces with the optimiser off. The Pillow resizing and compositing is ported exactly, and the PLY and JSON writers produce the same bytes.
- It adds bit-equal reuse: the context is computed once, and the per-step prefix is shared by both passes.
- `scripts/parity_test.sh` byte-compares 760–1,000 checkpoints per config plus the .ply and camera.json, and exits non-zero on mismatch. It passed:
  - dev1, without reuse;
  - dev2, with reuse, twice in fresh processes;
  - final job, on HEAD commit bd5c29c, all four configs.

**Python mode** (`run_kernels.py --parity`): the reference modules with the same reuse.
- No idol_cuda kernel is used in this mode, because none is bit-equal to the ATen ops it replaces.

**Evidence across both builds:**
- In the benchmark, all 40 parity PLYs (2 builds × 4 configs × 5 rounds) match the reference sha256, and all 40 camera JSONs are byte-identical.
- Speed: 9.15–9.50 s, which is 1.04–1.05x faster than eager.

## Kernel package (kernel/, edition 6, cpp-cuda; sm_90 at first, rebuilt for 8.0-12.0 as version 1, see Release preparation)

- **Ops:**
  - `add_ln_mod`, with an optional second fp32 LayerNorm;
  - `rope_rms_qk`, `rms_heads`, `merge_extra_key`, `gaussians_ply`;
  - `linear_gelu`, the cuBLASLt epilogue through `_addmm_activation`.

  All validate their arguments with TORCH_CHECK.
- **Build:** the `kernel-builder-job` image on cpu-performance ran a real `build-and-copy` and produced 17 variants (torch 2.10–2.15, CUDA 12.6–13.4). The outputs are committed in `kernel/build` of the development tree; they go to the Hub, not to the GitHub repository.
- **Checks:**
  - With kernel-builder 0.19.0-dev1, pinned by flake.lock, `check-config` and `check-builds` both exit 0.
  - The image's own CLI (0.18) rejects edition 6.
- **Tests:** 44/44 pass on the builder's torch210-cu130 variant on the H200. Tolerances come from measured error: at most 2–3 fp16 ulps on a small fraction of elements, caused by a different fp32 reduction order.
- **Not covered:**
  - The other 16 variants were never loaded on a GPU.
  - `benchmarks/benchmark.py` was not executed.
- **Card:** the build's `CARD.md` (in `kernel/build/`, development tree) was refreshed by copying the edited `CARD.md` over it, since it has no template fields.

## Script checks

Each script ran once in a fresh `env -i` shell on robot 262k:

| Script | Result |
|---|---|
| `run_native.sh` (fast) | Clean build, 70 s including the build. Latent rel 1.57e-2 and Chamfer 1.68e-3, the same as the benchmark row. |
| `run_native.sh --parity` | PLY and camera JSON byte-identical. |
| `run_kernels.py --local` | Builder kernels. Chamfer 1.69e-3. |
| `run_kernels.py --local --parity --check` | PLY byte-identical, exit 0; camera JSON identical. |

- The first `run_native.sh` fast attempt crashed on the FA3 dlopen. It now falls back to cuDNN, re-verified in job 6b069248.
- **What is native:** the whole request runs in one C++ process.
- **What is not native:**
  - the one-off TorchScript export, done in Python;
  - CMake finding libtorch through Python at build time;
  - the torch wheel's shared libraries, loaded at run time.

## Spend

26 jobs, about **$16.91** in total; see docs/jobs-log.md.
- The earlier cancelled H200 job is logged as an estimate (40 min, $3.33).
- The last job waited 40 min in scheduling; that wait was not counted as runtime.

## Not done or not verified

1. FlashAttention-3 in the native binary. I did not try vendoring FA3's head-dim-64 forward kernel.
2. The closeness misses listed above.
3. Kernel package variants other than torch210-cu130 (H200, sm_90) and torch210-cu128 (sm_120, tests only), and the architectures sm_80, sm_86, sm_89, sm_100.
4. `benchmarks/benchmark.py` and `kernels benchmark` (the latter needs a Hub repo).
5. The cause of the slower CUDA-graph replay.
6. The cause of the compiled default's one-time run-to-run difference.
7. Any other GPU or software versions; parity is claimed only for the stated combination.
8. Publishing, which was out of scope (the private Space is the only thing created).

## Release preparation (2026-11-25)

Nothing was published except the private Space `suvadityamuk/idol-cuda-demo`. Jobs 27-31 in docs/jobs-log.md
(release spend $2.69, cumulative $19.60).

- **Kernel package, version 1.** `build.toml` now has `version = 1` and `cuda-capabilities` 8.0, 8.6, 8.9, 9.0, 10.0,
  12.0. kernel-builder (pinned 0.19.0-dev1) built 17 variants from source commit 8ab80bb; the CUDA 12.6 variants get
  8.0-9.0, the others all six. `check-builds` exits 0 with both CLIs. Tests: 44/44 on `torch210-cu130` on the H200
  (job 6b0719f2) and 44/44 on `torch210-cu128` on the Space's ZeroGPU GPU (RTX PRO 6000 Blackwell MIG 2g.48gb, sm_120,
  torch 2.10.0+cu128). sm_80, sm_86, sm_89, sm_100 and the other variants are built but untested.
- **ZeroGPU hardware** (probed from the Space): half an NVIDIA RTX PRO 6000 Blackwell Server Edition (MIG 2g.48gb,
  94 SMs, sm_120, 48 GB), driver 595.112.07, torch 2.10.0+cu128, cuDNN 9.10.2, Python 3.12.12. FlashAttention-3 from
  `kernels-community/flash-attn3` refuses to load there (built for 8.0 and 9.0a); cuDNN attention with log-sum-exp
  works. The GPU worker process persisted between calls in the probe.
- **scripts/infer.py**, run from a fresh export of the release tree after `pip install -r requirements.txt` (job
  6b0719f2): fast robot 4.22 s per request after `--warmup`, closeness identical to the benchmark's (camera fov error
  0.045 deg, Chamfer 1.69e-3); fast avocado (fov 0.179 deg, Chamfer 1.89e-3, as in docs/benchmark.md); `--build parity
  --check`: PLY and camera JSON byte-identical to the eager reference; `--build reference --eager --check`:
  byte-identical; `--build reference`: 6.48 s after a 20.8 s warm-up. Error paths checked: no CUDA, no token (the
  SDK's gated-access message), Hub kernel repo missing (points to `--local-kernels`). The default Hub kernel path
  cannot be tested until the kernel repo exists.
- **Space app logic** (space/test_app.py, job 6b071de3, H200): both buttons, the official run inside the app is
  byte-identical to the eager reference PLY, timings 4.18 s fast / 9.74 s official eager, self-check 44/44, and the
  missing-token start-up. On the deployed Space (no token): build, start-up, kernel loading, the self-check on sm_120,
  the missing-token banner and error, and the Model3D viewers rendering the precomputed splats were checked.
  Generation on ZeroGPU itself is untested until a token secret is added.

Main documents: README.md, docs/benchmark.md, docs/parity.md, docs/ab-log.md, docs/profile.md, docs/jobs-log.md; raw data in logs/.

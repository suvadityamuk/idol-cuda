# Profile of the reference on H200 (stage 1)

Setup: HF Jobs `h200` (NVIDIA H200 141 GB, driver 595.112.07, 23 vCPU quota; `nproc` reports 192 host CPUs),
image `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel`, torch 2.10.0+cu130, cuDNN 9.15.1 (91501), CUDA 13.0,
transformers 5.24.1, diffusers 0.43.0, Pillow 13.0.0, numpy 2.5.5. idol-sdk 8db7003 unmodified, `ATTN_BACKEND=sdpa`
(flash-attn is not a dependency of idol-sdk, so the shipped default is PyTorch SDPA). Robot, synthetic, 262,144
Gaussians, seed 42, 50 steps. Raw files (the release tree keeps only those under 300 KB): `logs/stage1/` (job.log, profile_*.txt, trace_*.json.gz,
inductor_cache.tgz); `tools/profile_ref.py` (steady state: second request in the process, torch.profiler) and
`tools/trace_stats.py` (per-stage wall vs GPU busy time from the trace).

## Whole request, steady state (second request in the process)

| stage | compiled (shipped default) wall ms | GPU busy ms | eager `--no_optimize` wall ms | GPU busy ms |
|---|---:|---:|---:|---:|
| preprocess (Lanczos resize on CPU, BiRefNet fp16, alpha, crop) | 169 | 32 | 168 | 32 |
| encoders (DINOv3 ViT-H+ bf16, FLUX.2 VAE encoder bf16) | 79 | 75 | 80 | 75 |
| sampling (50 steps, CFG 3) | **6,143** (93%) | 6,079 | **9,473** (96%) | 8,886 |
| octree decoder + sampling (8 levels) | 214 | 26 | 156 | 26 |
| Gaussian decoder (8,192 anchors) | 41 | 39 | 41 | 39 |
| to_representation (Gaussian activations) | 21 | 0.1 | 13 | 0.1 |
| request total | **6,611** | | **9,883** | |
| save_ply (separate) | 77 | | 85 | |

The compiled request matches the published 6.6 s for H200. First request of a process (what the dumps measured,
`logs/stage1/job.log`): compiled 20.6 s (compile + autotune, inductor cache cold) / 10.5 s (cache warm, still
Dynamo tracing and recompiles for 8192, 4101 and 12294 token shapes); eager 11.3-12.2 s. Load 13-14 s (weights from
the mounted repos).

Outside sampling the GPU is mostly idle: preprocess is CPU-bound (Pillow Lanczos, numpy, BiRefNet's 17 bilinear
upsamplings are small), octree sampling is a Python loop with `.item()` syncs (GPU busy 12% of its wall time),
to_representation and save_ply are CPU numpy/plyfile work. Together about 0.5 s of the compiled 6.6 s.

## Sampling, compiled (shipped default: torch.compile of every UnifiedTransformerBlock and RePo layer, batched CFG
(batch 2), cuDNN attention, TF32)

Per kernel over the 50 steps (6,079 ms GPU, 123 ms per step; busy/wall 0.99, so it is GPU-bound and CUDA graphs
cannot gain much):

| kernel | ms | share |
|---|---:|---:|
| cuDNN SDPA fprop (`sdpa_sm90_flash_fprop_wgmma_f16 64x128x64`), 28 calls/step at B=2, L=12,294 (2.81 ms each, ~440 TFLOP/s) | 3,937 | 64.8% |
| fp16 GEMMs (nvjet: qkv, out, fc1, fc2) | ~1,080 | 17.8% |
| Triton: RoPE + RMSNorm of q/k + transpose to attention layout | 313 | 5.1% |
| Triton: bias + GELU after the fc1 GEMM (reads and writes 24,588 x 4,096 fp16) | 178 | 2.9% |
| TF32 GEMMs of the RePo layers (gate/content/final maps) | 121 | 2.0% |
| Triton: residual + LayerNorm + modulation (2 per block) | 127 | 2.1% |
| Triton: RePo angles (tanh, clamp_mul) / cos-sin stack / LayerNorm / silu*mul | 205 | 3.4% |
| other | ~120 | 2% |

What Inductor generates (`logs/stage1/inductor_cache.tgz`, development tree only, not in the release; one graph per block class and shape; no graph breaks,
3 shape recompiles per class: 8,192 noise-refiner tokens, 4,101 context tokens, 12,294 joint tokens): per block 5
Triton kernels + 4 cuBLAS GEMMs + 1 cuDNN attention.

What it leaves on the table:
1. **Structure, not fusion, is the big one.** Attention is 65% of sampling and Inductor cannot change what is
   attended: the unconditional pass carries 4,101 identical context tokens (zero image features) in every one of 26
   layers and 50 steps; the context stream (cond embedders + 2 context-refiner blocks) depends only on the image but
   is recomputed every step for both CFG halves; the noise refiner, camera refiner and time embedding are computed
   twice per step (batch 2) on identical inputs.
2. fc1 bias + GELU runs as a separate pass over a 24,588 x 4,096 fp16 matrix instead of in the GEMM epilogue (2.9%).
3. The RePo layer is a separate compiled graph: its LayerNorm re-reads the block input the block's own LayerNorm
   reads; angles and cos/sin are materialised in memory and re-read by the RoPE kernel (about 4% together).
4. out-proj and fc2 biases are added in the following Triton kernel (fine), q/k are written once more in the
   attention layout (inherent to the cuDNN call).
5. Not on the table: launch overhead (GPU busy 99% of sampling wall time), dtype round-trips (the fp32 LayerNorm and
   RMSNorm math stays in registers).

## Eager (`--no_optimize`) sampling

9,473 ms wall, GPU busy 94%: cuDNN attention 3,669 ms (PyTorch 2.10's SDPA picks cuDNN for these fp16 shapes on
sm_90 even without the optimize path), fp32 SGEMMs of the RePo layers (no TF32) 452 ms, about 2,000 ms of unfused
elementwise kernels (copies, mul, add, layer_norm, vector_norm, complex RoPE, polar), 219k kernel launches.

## Determinism (fresh processes, `logs/stage1/noise_*.txt`)

- Eager `--no_optimize`: run A and run B are byte-identical at every checkpoint (BiRefNet, encoders, every denoiser
  call, octree, Gaussians) and the PLYs have equal sha256, on all four configs. No cuDNN pinning is needed on CUDA:
  `torch.backends.cudnn.deterministic=True` gives byte-identical results to the default (checked on robot 262k).
  (The ROCm BiRefNet fp16 1x1-conv nondeterminism does not occur here.)
- Compiled (shipped default): avocado 262k A == B byte-identical; robot 262k A != B (156 of 223 checkpoints differ,
  so even the RNG draw shapes of the octree differ): the compiled path is not run-to-run deterministic here.
- Compiled vs eager: every checkpoint after the encoders differs (latent, camera, Gaussians); see docs/benchmark.md
  for the measured spread used as the fast build's tolerance.

## Fast build, one sampling step (cuDNN variant, `tools/profile_fast.py`, `logs/dev3/profile_fast.txt`)

85.2 ms GPU per step (both CFG passes; packed [12,294 cond + 8,194 uncond] rows):

| kernel | ms/step | share |
|---|---:|---:|
| cuDNN SDPA fprop (cond L=12,294 + uncond 8,194 queries x 8,193 keys, 26 calls incl. noise refiner) | 55.6 | 65.3% |
| fp16 GEMMs (qkv / out / fc2 with bias) | 11.8 | 13.8% |
| fc1 GEMM with bias + GELU epilogue (cuBLASLt `gelubias`) | 6.5 | 7.6% |
| `add_ln_mod` (residual + LN + modulation + RePo LN), 54 calls | 4.1 | 4.8% |
| `rope_rms_qk`, 26 calls | 3.0 | 3.6% |
| RePo TF32 GEMMs | 1.1 | 1.3% |
| `merge_extra_key`, 24 calls | 1.0 | 1.2% |
| copies, small ops | ~2 | ~2% |

Attention micro-benchmark at the sampler's shapes (fp16, H=16, D=64, `logs/dev3`/`logs/dev4`): cuDNN 1.48 ms at
L=S=12,294 (419 TFLOP/s), 0.67 ms at 8,194 x 8,193 (410 TFLOP/s); ATen FlashAttention-2 2.01 / 0.90 ms
(~305 TFLOP/s). FlashAttention-3 (Hub kernel) brings a full step from 88.8-90.5 to 77.4-79.1 ms (docs/ab-log.md).
What is left is mostly attention at ~65% of a step and the GEMMs; FP8 attention was tried and rejected for accuracy.

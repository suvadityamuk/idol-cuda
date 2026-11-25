---
title: idol-cuda demo
emoji: 🗿
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Official Idol pipeline vs fused CUDA kernels, timed live
models:
  - speridlabs/idol
  - facebook/dinov3-vith16plus-pretrain-lvd1689m
  - black-forest-labs/FLUX.2-dev
---

# idol-cuda demo

Upload one image of an object and run it through Speri Labs' [Idol](https://huggingface.co/speridlabs/idol) (image to
3D Gaussian splat) twice: once with the official [idol-sdk](https://github.com/speridlabs/idol-sdk) pipeline and
once with the fast build from [idol-cuda](https://github.com/suvadityamuk/idol-cuda). Both are timed on this
Space's GPU and shown in a splat viewer with a download button. "Generate" runs only the fast build; "Compare with
official" runs both with the same seed.

This is an **unofficial** port. Idol and idol-sdk are by Speri Labs (Apache-2.0); the example images come from
idol-sdk's `demo/examples` (Apache-2.0). The image encoders are gated under their own licences:
[DINOv3](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) and the
[FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) VAE.

## How it is faster

The fast build keeps the official pre-processing, encoders and decoders and changes the sampler (93% of a request):

- The image's context tokens are computed once per request instead of at every step for both guidance passes; the
  time embedding and refiners are computed once per step and shared by both passes.
- The unconditional pass attends to one context token instead of 4,101 identical ones; the result is merged exactly
  with a log-sum-exp correction.
- Fused CUDA kernels ([suvadityamuk/idol-cuda](https://huggingface.co/kernels/suvadityamuk/idol-cuda)): residual + gate +
  LayerNorm + adaLN in one pass, rotary embedding + q/k RMSNorm in one pass, GELU in the GEMM epilogue, and the
  Gaussian activations written straight into PLY columns. A loop-free octree sampler replaces a Python loop.
- On Hopper GPUs, FlashAttention-3 from `kernels-community/flash-attn3`. It is Hopper-only, so on this Space's GPU
  the fast build uses cuDNN attention.

## Numbers

Measured offline on an H200 (torch 2.10 + CUDA 13.0, 50 steps, 262,144 Gaussians, robot example, median of 5
interleaved requests after a warm-up):

| build | seconds per request | vs official compiled |
|---|---|---|
| official pipeline, compiled (default) | 6.47 (first request 21.5 s, with compilation) | 1.00x |
| official pipeline, eager (`--no_optimize`) | 9.60 | 0.67x |
| fast build (kernels + FlashAttention-3) | 4.22 | 1.53x |

On this Space the baseline is the **official pipeline in eager mode**. The default path uses `torch.compile`, which
ZeroGPU's documentation lists as unsupported (the official Idol Space runs eager for that reason); its first request
also took about 21 s of compilation on the H200, a cost every new GPU worker would pay again on visitors' quota. The speed-up shown in the app is
measured live on this Space's GPU (half of an RTX PRO 6000 Blackwell, ZeroGPU) against that eager baseline, without
FlashAttention-3; it is not the H200 number. Times exclude model loading and a one-off warm-up per GPU worker, which
the app shows separately.

## Closeness

The fast build is not byte-identical to the official pipeline (the parity build in the GitHub repository is, on the
tested H200 setup). On the H200 its rendered views are within 0.2 dB PSNR of the official pipeline's own variants
(compiled vs eager, flash-attn vs SDPA), and the Gaussian positions are as close as those variants are to each other.
On the avocado example the predicted camera (field of view and radius) is about 1.5x further from the eager
reference than the official compiled pipeline is, i.e. slightly outside the official pipeline's own compiled-vs-eager
spread. Details: [docs/benchmark.md](https://github.com/suvadityamuk/idol-cuda/blob/main/docs/benchmark.md).

## Limits

- The full pipeline is tested on an H200 (sm_90) with torch 2.10 + CUDA 13.0. This Space runs the kernels' torch 2.10
  + CUDA 12.8 variant on sm_120, where the kernel package's 44 tests pass (Diagnostics section); its end-to-end
  timing on this GPU has only been measured by this Space itself.
- Timing on ZeroGPU varies with the shared hardware and includes no queueing; the first run on a new GPU worker
  includes a short warm-up that is reported separately and not counted.
- The Space needs an `HF_TOKEN` secret from an account that accepted the DINOv3 and FLUX.2-dev licences.

Links: [GitHub](https://github.com/suvadityamuk/idol-cuda) ·
[kernels](https://huggingface.co/kernels/suvadityamuk/idol-cuda) · [Idol model](https://huggingface.co/speridlabs/idol) ·
[idol-sdk](https://github.com/speridlabs/idol-sdk)

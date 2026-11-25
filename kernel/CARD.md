---
license: apache-2.0
tags:
  - kernels
  - cuda
---

# idol-cuda

Fused CUDA kernels for [Idol](https://huggingface.co/speridlabs/idol) (Speri Labs' single-image-to-3D-Gaussians
model, [idol-sdk](https://github.com/speridlabs/idol-sdk)) on NVIDIA GPUs. Unofficial; source, benchmarks and the
pipeline that uses these kernels: [github.com/suvadityamuk/idol-cuda](https://github.com/suvadityamuk/idol-cuda).
Live comparison with the official pipeline: [suvadityamuk/idol-cuda-demo](https://huggingface.co/spaces/suvadityamuk/idol-cuda-demo).

They are the hot ops of the repository's fast build: together with a restructured sampler (and FlashAttention-3 from
`kernels-community/flash-attn3`) they make an Idol request 1.51-1.54x faster than the official pipeline's compiled
default on an H200 (4.22 vs 6.47 s at 262,144 Gaussians, 50 steps, torch 2.10 + CUDA 13.0, one interleaved session),
with rendered views within 0.2 dB PSNR of the official variants' own spread (details and the metrics outside that
spread: docs/benchmark.md in the source repository).

**Tested:** the `torch210-cu130-x86_64-linux` variant on an NVIDIA H200 (sm_90): 44/44 tests. Built for compute
capabilities 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0 (CUDA 12.6 variants: up to 9.0, see the per-variant `metadata.json`);
the `torch210-cu128-x86_64-linux` variant also passed the 44 tests on sm_120 (a ZeroGPU MIG slice of an NVIDIA RTX
PRO 6000 Blackwell, torch 2.10.0+cu128). All other variants and architectures (sm_80, sm_86, sm_89, sm_100) are built
but untested.

| op | replaces |
|---|---|
| `add_ln_mod(x, h=, gate=, weight=, bias=, scale=, shift=, w2=, b2=)` | `x + h * gate`; `LayerNorm32(x)` (plain or affine) `* (1 + scale) + shift` (adaLN), each fp16 rounding point kept; optionally the next RePo layer's fp32 affine LayerNorm of the same row |
| `rope_rms_qk(qkv, delta, freqs, gamma_q, gamma_k)` | RePo rotary angles (`clamp_mul`) + RoPE + `F.normalize(.float()) * gamma * 8` for q and k (head dim 64) |
| `rms_heads(x, gamma)` | per-head `F.normalize(.float()) * gamma * 8` (decoders) |
| `merge_extra_key(o, lse, q, k_extra, v_extra, scale, log_mult)` | attention over keys + one extra key repeated exp(log_mult) times, from an attention output and its log-sum-exp |
| `gaussians_ply(feats, points, perturb, ...)` | `GaussianFixedlenDecoder.to_representation` + `Gaussian.__init__` + the PLY column transforms of `save_ply` |
| `linear_gelu(x, weight, bias)` | `gelu(F.linear(x, weight, bias), approximate='tanh')` with GELU in the cuBLASLt epilogue |

```python
from kernels import get_kernel
idol = get_kernel("suvadityamuk/idol-cuda", version=1, trust_remote_code=True)
x_sum, y = idol.add_ln_mod(x, h=h, gate=gate, scale=scale, shift=shift)
```

Inputs are CUDA tensors of the dtypes in each docstring (rows of width 1024 for `add_ln_mod`); every argument is
validated (device, dtype, shape, strides, alignment). Results differ from the PyTorch op chains only through fp32
reduction order (LayerNorm and L2-norm statistics), which moves a small fraction of fp16 outputs by one ulp; the
tests in the source repository state the tolerances.

From the source repository:
- `scripts/infer.py photo.jpg -o photo.ply`: one command from an image to a `.ply` with these kernels (fast build),
  or `--build parity` / `--build reference`.
- `scripts/run_kernels.py`: the official Idol pipeline (front end, decoders) with the fast sampler and decoder head
  on these kernels; `--local` uses a local `build/`, `--parity` runs the bit-exact parity mode instead.
- `scripts/run_native.sh`: the whole pipeline as a libtorch C++ binary (these kernels compiled in), no Python in the
  inference process; `--parity` runs the op-for-op port whose PLY is byte-identical to the official pipeline's.

Licence: Apache-2.0. Idol's weights and its gated image encoders (DINOv3, the FLUX.2-dev VAE) are under their own
licences and are not part of this repository.

# Parity (Amendment 4, tier 1 achieved)

**Definition.** Parity is defined against the official pipeline in eager mode on this machine: idol-sdk 8db7003,
`IdolPipeline(optimize=False)` (`--no_optimize`), `ATTN_BACKEND=sdpa` (flash-attn is not an idol-sdk dependency),
fp16 blocks / fp32 norms / bf16 encoders as shipped, on (H200 sm_90, driver 595.112.07, CUDA 13.0, cuDNN 9.15.1,
torch 2.10.0+cu130, Pillow 13.0.0). The compiled default is not a parity target: it is not run-to-run deterministic
here (robot 262k, `logs/stage1/noise_compiled_robot_262k.txt`), and it is the fast build's tolerance band instead.

**Reference determinism.** Two fresh-process eager runs per config are byte-identical at every checkpoint and in the
PLY (sha256 robot 262k 903bba69..., avocado 262k 4d2ed49e..., robot 1M 84294446..., avocado 1M 11b23ad3...).
`torch.backends.cudnn.deterministic=True` changes nothing (byte-identical to the default). No pinning was needed.

## Native parity build (`idol_native --build parity`, `scripts/run_native.sh --parity`)

An op-for-op libtorch C++ port: the same ATen ops in the same order and dtypes as the eager reference (cuBLAS /
cuDNN chosen by the same ATen dispatch), the three image networks as TorchScript traces run with the graph optimiser
off, Pillow 13 resampling / compositing ported to C++ (`-ffp-contract=off`), plyfile- and json-identical writers.
Bit-equal reuse (default, `--no_reuse` turns it off): the context stream is computed once per request and the
per-step prefix (time embedding, adaLN, noise refiner, camera refiner) once per step for both CFG passes; each reused
tensor is a deterministic function of identical inputs.

Regression check: `scripts/parity_test.sh REF_DIR OUT_DIR [BIN]` runs the binary in a fresh process per config and
byte-compares every recorded checkpoint (BiRefNet input/output, encoder features, all 100 denoiser calls' inputs and
outputs, internal block checkpoints, octree calls, Gaussian decoder, final Gaussians, RNG draws) plus the `.ply` and
`_camera.json` files; exits non-zero on the first mismatch.

| run | job | robot 262k | robot 1M | avocado 262k | avocado 1M |
|---|---|---|---|---|---|
| without reuse (previous agent's build) | dev1 6b065e28 | 760 ckpts identical, PLY + camera identical | 970, identical | 781, identical | 1000, identical |
| with reuse, pass 1 | dev2 6b066ebf | 760 identical, PLY + camera identical | 970 identical | 781 identical | 1000 identical |
| with reuse, pass 2 (fresh processes) | dev2 6b066ebf | identical | identical | identical | identical |

Logs: `logs/dev1/parity_test.txt`, `logs/dev2/parity_test_{1,2}.txt`. "bit-identical" here means byte-compared.

## Python parity mode (`scripts/run_kernels.py --parity`, `fast/parity.py`)

The reference modules with the same bit-equal reuse (context once, shared per-step prefix) and the reference octree
sampling, Gaussian activations and plyfile writer. Robot 262k: PLY and camera JSON byte-identical to the reference
(`logs/dev3/job.log`: "PY PARITY PLY IDENTICAL", "PY PARITY CAMERA IDENTICAL"). No idol_cuda kernel is used in this
mode: none of them has been shown bit-equal to the ATen op chain it replaces (they differ in fp32 reduction order).

## Benchmark session and stage-6 scripts

- Benchmark session (job 6b0687b0, `logs/bench/bench.jsonl`): every one of the 5 timed outputs of `native_parity` and
  `py_parity` on each of the 4 configs has the sha256 of the stage-1 eager reference PLY (40 PLYs byte-identical),
  and all 40 camera JSONs are byte-identical (cmp) to the reference's.
- `scripts/run_native.sh --parity` in a fresh shell (job 6b068fac): PLY and camera JSON byte-identical (cmp) to the
  reference on robot 262k.
- `scripts/run_kernels.py --local --parity --check REF.ply` in a fresh shell (job 6b068fac): "PLY BYTE-IDENTICAL",
  exit code 0; camera JSON byte-identical.
- Speed of the parity builds (benchmark): 9.15-9.50 s per request vs 9.60-9.93 s for the eager reference they
  match (1.04-1.05x, from the bit-equal reuse), 0.70-0.72x of the shipped compiled default.

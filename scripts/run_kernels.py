#!/usr/bin/env python3
"""Idol image -> 3D Gaussians on CUDA in PyTorch, with the hot ops from the idol-cuda kernel package.

The official pipeline (idol-sdk 8db7003: preprocessing, BiRefNet, DINOv3, FLUX.2 VAE, octree and Gaussian decoders)
runs as shipped; the sampler is the fast build's (context once, shared prefix, packed CFG rows, one-token
unconditional context, cuDNN attention) on the package's kernels, and the Gaussian head + PLY columns use the
package's `gaussians_ply`. --parity runs the parity mode instead: the reference arithmetic with only bit-equal reuse,
whose PLY is byte-identical to the official eager pipeline's (no kernels are used in that mode).

  python scripts/run_kernels.py --local                       # kernel from kernel/build (kernel-builder output)
  python scripts/run_kernels.py --local --parity
  python scripts/run_kernels.py                               # kernel from the Hub (suvadityamuk/idol-cuda, v1)

Requires: an NVIDIA GPU (sm_90 build tested on H200), torch 2.10 + CUDA, idol-sdk installed (pip install -e), and
access to the gated encoders (or local copies via --dinov3/--flux2).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image', default=None, help='default: idol-sdk demo/examples/fig-robot-cute.webp')
    p.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg_strength', type=float, default=3.0)
    p.add_argument('--rescale_t', type=float, default=3.0)
    p.add_argument('--num_gaussians', type=int, default=262144)
    p.add_argument('--weights', default=None, help='Idol weights dir (default /w/idol if present, else the Hub)')
    p.add_argument('--dinov3', default=None)
    p.add_argument('--flux2', default=None)
    p.add_argument('--out', default='outputs')
    p.add_argument('--name', default=None)
    p.add_argument('--local', action='store_true', help='load the kernel from kernel/build (get_local_kernel)')
    p.add_argument('--kernel_dir', default=os.path.join(ROOT, 'kernel'))
    p.add_argument('--parity', action='store_true', help='bit-exact parity mode (no kernels)')
    p.add_argument('--check', default=None, help='reference .ply: byte-compare the output, exit 1 on mismatch')
    a = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit('error: no CUDA GPU available (idol-cuda needs an NVIDIA GPU, sm_90 tested)')
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    if a.local:
        if not os.path.isdir(os.path.join(a.kernel_dir, 'build')):
            sys.exit(f'error: {a.kernel_dir}/build not found: run kernel-builder build-and-copy there first')
        os.environ['IDOL_KERNEL_LOCAL_PATH'] = a.kernel_dir
    elif not a.parity:
        os.environ['IDOL_KERNEL_HUB'] = '1'

    import idol
    sdk = os.path.dirname(os.path.dirname(idol.__file__))
    image = a.image or os.path.join(sdk, 'demo', 'examples', 'fig-robot-cute.webp')

    def resolve(path, repo, default):
        if path:
            return path
        if os.path.isdir(default):
            return default
        from huggingface_hub import snapshot_download
        pats = {'speridlabs/idol': None, 'black-forest-labs/FLUX.2-dev': ['vae/*']}.get(repo)
        return snapshot_download(repo, allow_patterns=pats)
    weights = resolve(a.weights, 'speridlabs/idol', '/w/idol')
    dinov3 = resolve(a.dinov3, 'facebook/dinov3-vith16plus-pretrain-lvd1689m', '/w/dinov3')
    flux2 = resolve(a.flux2, 'black-forest-labs/FLUX.2-dev', '/w/flux2')

    from fast.pipeline import FastIdolPipeline, save_outputs
    dev = torch.cuda.get_device_name(0)
    pipe = FastIdolPipeline(weights, dinov3, flux2, parity=a.parity)
    src = 'none (parity mode)' if a.parity else pipe.K.SOURCE
    print(f'device: {dev} | build: {"parity" if a.parity else "fast"} | kernels: {src}', flush=True)
    t = time.perf_counter()
    out = pipe(image, mode=a.mode, seed=a.seed, steps=a.steps, cfg_strength=a.cfg_strength, rescale_t=a.rescale_t,
               num_gaussians=a.num_gaussians)
    torch.cuda.synchronize()
    name = a.name or os.path.splitext(os.path.basename(image))[0]
    save_outputs(out, a.out, name)
    print(f'{name}: {a.out}/{name}.ply ({time.perf_counter() - t:.2f} s incl. first-request warm-up)')
    if a.check:
        import filecmp
        same = filecmp.cmp(os.path.join(a.out, f'{name}.ply'), a.check, shallow=False)
        print(f'check vs {a.check}: {"PLY BYTE-IDENTICAL" if same else "PLY DIFFERS"}')
        if not same:
            sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""idol-cuda: one image in, a 3D Gaussian splat (.ply) out, on an NVIDIA GPU.

  python scripts/infer.py photo.jpg -o photo.ply                       # fast build, kernels from the Hub
  python scripts/infer.py photo.jpg -o photo.ply --local-kernels kernel  # fast build, local kernel-builder output
  python scripts/infer.py robot.webp -o robot.ply --mode synthetic --build parity --check reference.ply
  python scripts/infer.py photo.jpg -o photo.ply --build reference     # the official pipeline (compiled default)

Builds:
  fast       restructured sampler + the idol-cuda kernels (Hub: suvadityamuk/idol-cuda, version 1, or
             --local-kernels) + FlashAttention-3 from kernels-community/flash-attn3 on Hopper (cuDNN attention
             elsewhere, or when FA3 cannot be loaded). Close to the official output, not byte-identical.
  parity     the official arithmetic with only bit-equal reuse; on the tested setup (H200, torch 2.10.0+cu130,
             CUDA 13.0, cuDNN 9.15.1) its .ply is byte-identical to the official eager pipeline's. No custom kernels.
  reference  the official idol-sdk pipeline as shipped (torch.compile; --eager for --no_optimize).

Needs an NVIDIA GPU (compute capability 8.0+; tested on H200), the packages in requirements.txt, and a Hugging Face
token with access to the gated encoders facebook/dinov3-vith16plus-pretrain-lvd1689m and
black-forest-labs/FLUX.2-dev (`hf auth login` or HF_TOKEN).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', help='input image (photo, render or generated image of one object)')
    p.add_argument('-o', '--out', default=None, help='output .ply (default: <image name>.ply)')
    p.add_argument('--mode', choices=['real', 'synthetic'], default='real',
                   help='real: photos (looser crop); synthetic: renders / generated images (default: real)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--gaussians', type=int, default=262144, help='32,768 to 1,048,576 (default 262,144)')
    p.add_argument('--cfg', type=float, default=3.0)
    p.add_argument('--rescale-t', type=float, default=3.0)
    p.add_argument('--build', choices=['fast', 'parity', 'reference'], default='fast')
    p.add_argument('--eager', action='store_true', help='with --build reference: --no_optimize (eager)')
    p.add_argument('--local-kernels', default=None, metavar='PATH',
                   help='kernel-builder project dir holding build/<variant>/ (e.g. kernel/) instead of the Hub')
    p.add_argument('--attn', choices=['auto', 'fa3', 'cudnn'], default='auto', help='fast build attention')
    p.add_argument('--warmup', action='store_true',
                   help='run one untimed request first, so the timing excludes one-off warm-up (and compilation)')
    p.add_argument('--turntable', action='store_true', help='also write <out>.gif (needs gsplat)')
    p.add_argument('--check', default=None, metavar='REF.ply', help='byte-compare the .ply; exit 1 if it differs')
    p.add_argument('--weights', default=None, help='local Idol weights dir (default: Hub speridlabs/idol)')
    p.add_argument('--dinov3', default=None, help='local DINOv3 dir (default: Hub)')
    p.add_argument('--flux2', default=None, help='local FLUX.2-dev dir (default: Hub, vae/ only)')
    return p.parse_args()


def main():
    a = parse()
    if not os.path.isfile(a.image):
        sys.exit(f'error: {a.image} not found')
    out = a.out or os.path.splitext(os.path.basename(a.image))[0] + '.ply'
    if not out.endswith('.ply'):
        sys.exit('error: -o must end in .ply')
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')  # idol-sdk default without flash-attn installed
    t_start = time.perf_counter()

    import torch
    from fast import runner
    try:
        name, cap = runner.check_cuda() if a.build == 'fast' else _cuda_only()
        notice = None
        if a.build == 'fast':
            runner.use_kernels(a.local_kernels)
            K = runner.load_kernels()
            attn, notice = runner.pick_attention(a.attn)
        runner.check_gated_access(*runner.encoder_paths(a.dinov3, a.flux2))
    except runner.SetupError as e:
        sys.exit(f'error: {e}')

    desc = {'fast': 'fast (restructured sampler + idol-cuda kernels)',
            'parity': 'parity (byte-identical to the official eager pipeline on the tested setup)',
            'reference': 'official idol-sdk pipeline, ' + ('eager (--no_optimize)' if a.eager else 'compiled default')}
    print(f'device:  {name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}, CUDA {torch.version.cuda}')
    print(f'build:   {desc[a.build]}')
    if a.build == 'fast':
        print(f'kernels: {K.SOURCE}')
        print(f'attention: {"FlashAttention-3 (kernels-community/flash-attn3)" if attn == "fa3" else "cuDNN"}')
    if notice:
        print(f'notice:  {notice}')

    t = time.perf_counter()
    weights = runner.idol_weights(a.weights)
    dinov3, flux2 = runner.encoder_paths(a.dinov3, a.flux2)
    if a.build == 'reference':
        from idol import IdolPipeline
        enc = {k: v for k, v in (('dinov3', dinov3), ('flux2_vae', flux2)) if v} or None
        pipe = IdolPipeline(weights, device='cuda', optimize=not a.eager, image_encoders=enc)
    else:
        from fast.pipeline import FastIdolPipeline
        pipe = FastIdolPipeline(weights, dinov3, flux2, parity=a.build == 'parity',
                                **({'attn': attn} if a.build == 'fast' else {}))
    torch.cuda.synchronize()
    print(f'load:    {time.perf_counter() - t:.1f} s (weights, encoders, kernels)')

    kw = dict(mode=a.mode, seed=a.seed, steps=a.steps, cfg_strength=a.cfg, rescale_t=a.rescale_t,
              num_gaussians=a.gaussians)
    if a.warmup:
        _, s = runner.timed_request(pipe, a.image, **kw)
        print(f'warm-up: {s:.2f} s (not counted below)')
    marks = [] if a.build != 'reference' else None
    result, req = runner.timed_request(pipe, a.image, **kw, **({'timings': marks} if marks is not None else {}))
    if marks:
        for stage, s in runner.stage_durations(marks):
            print(f'  {stage:<13}{s:7.3f} s')
    first = '' if a.warmup else ' (first request in this process: includes one-off warm-up' + \
        (' and torch.compile' if a.build == 'reference' and not a.eager else '') + '; use --warmup to exclude it)'
    print(f'request: {req:.2f} s{first}')

    t = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if a.build == 'reference':
        result.gaussian.save_ply(out)
    else:
        result.save_ply(out)
    stem = out[:-4]
    result.preprocessed_image.save(stem + '_input.png')
    with open(stem + '_camera.json', 'w') as f:
        json.dump(result.camera, f, indent=2)
    print(f'export:  {time.perf_counter() - t:.2f} s -> {out}, {stem}_input.png, {stem}_camera.json')
    print(f'total:   {time.perf_counter() - t_start:.1f} s wall (including start-up and loading)')

    if a.turntable:
        try:
            sys.path.insert(0, os.path.join(ROOT, 'tools'))
            import turntable
            frames = turntable.compose([turntable.fit_input(a.image, 384), turntable.render_frames(out, size=384)],
                                       size=384)
            turntable.save_gif(frames, stem + '.gif')
            print(f'turntable: {stem}.gif')
        except ImportError as e:
            print(f'turntable skipped: {e} (pip install gsplat)')
    if a.check:
        import filecmp
        same = filecmp.cmp(out, a.check, shallow=False)
        print(f'check vs {a.check}: {"PLY BYTE-IDENTICAL" if same else "PLY DIFFERS"}')
        if not same:
            sys.exit(1)


def _cuda_only():
    import torch
    from fast.runner import SetupError
    if not torch.cuda.is_available():
        raise SetupError('No CUDA GPU is visible to PyTorch; idol-cuda needs an NVIDIA GPU and a CUDA build of torch.')
    return torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)


if __name__ == '__main__':
    main()

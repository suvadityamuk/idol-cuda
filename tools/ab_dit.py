"""A/B harness for the fast sampler against the reference denoiser, in one process on the H200.

Teacher-forced: at selected steps, every variant gets the reference's own inputs (latent, camera, t, features from a
stage-1 `--dump_steps` dump of the eager reference) and its v_cond, v_uncond and guided v are compared with the
dump's. The reference's own spread is measured the same way: the shipped compiled path (torch.compile + batched CFG +
cuDNN attention + TF32) on the same inputs.

Timing: one sampling step (both CFG passes) per variant, interleaved, order flipped every repetition, CUDA-synced.
End to end (--e2e): 50 steps from the dumped noise, final latent / camera vs the reference's.

python tools/ab_dit.py --dump /scratch/ref/robot_262k_eager_sdpa_a.safetensors --variants fast,fast_nocollapse \
    --json out.json
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault('ATTN_BACKEND', 'sdpa')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tensordict import TensorDict  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--dump', required=True)
p.add_argument('--variants', default='fast')
p.add_argument('--steps', default='0,1,10,25,49')
p.add_argument('--reps', type=int, default=8)
p.add_argument('--e2e', action='store_true')
p.add_argument('--no_compiled', action='store_true')
p.add_argument('--json', default=None)
args = p.parse_args()

from idol import IdolPipeline  # noqa: E402
from idol.models import layers  # noqa: E402
from idol.pipeline import compile_for_sampling  # noqa: E402
from fast.model import FastDenoiser  # noqa: E402

D = load_file(args.dump)
dev = torch.device('cuda')
pipe = IdolPipeline('/w/idol', device='cuda', optimize=False,
                    image_encoders={'dinov3': '/w/dinov3', 'flux2_vae': '/w/flux2'})
den = pipe.denoiser
cond = TensorDict({'feature1': D['denoiser#0.in.2.feature1'].to(dev), 'feature2': D['denoiser#0.in.2.feature2'].to(dev)},
                  batch_size=1)
neg = cond.apply(torch.zeros_like)
steps = [int(s) for s in args.steps.split(',')]
n_calls = len([k for k in D if k.startswith('denoiser#') and k.endswith('.out.latent')])
assert n_calls >= 100, 'the dump needs --dump_steps'

# compiled reference (as shipped) in a second denoiser instance
if not args.no_compiled:
    import copy
    den_c = copy.deepcopy(den)
    compile_for_sampling(den_c)


def ref_compiled(z, cam, t):
    backend, tf32 = layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32
    layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32 = 'sdpa_cudnn', True
    try:
        x = TensorDict({'latent': torch.cat([z, z]), 'camera': torch.cat([cam, cam])}, batch_size=2)
        out = den_c(x, torch.cat([t, t]), torch.cat([cond, neg]))
        return (out['latent'][:1], out['camera'][:1]), (out['latent'][1:], out['camera'][1:])
    finally:
        layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32 = backend, tf32


def ref_eager(z, cam, t):
    x = TensorDict({'latent': z, 'camera': cam}, batch_size=1)
    a = den(x, t, cond)
    b = den(x, t, neg)
    return (a['latent'], a['camera']), (b['latent'], b['camera'])


VARIANTS = {
    'fast': dict(collapse_uncond=True, attn='cudnn'),
    'fast_fa3_nocollapse': dict(collapse_uncond=False, attn='fa3'),
    'fast_nocollapse': dict(collapse_uncond=False, attn='cudnn'),
    'fast_flash': dict(collapse_uncond=True, attn='flash'),
    'fast_fa2': dict(collapse_uncond=True, attn='fa2'),
    'fast_notf32': dict(collapse_uncond=True, attn='cudnn', tf32=False),
    'fast_fa3': dict(collapse_uncond=True, attn='fa3'),
    'fast_fa3_fp8': dict(collapse_uncond=True, attn='fa3_fp8'),
}
impls = {}
for v in args.variants.split(','):
    f = FastDenoiser(den, **VARIANTS[v])
    f.prepare(cond)
    impls[v] = f.both
if not args.no_compiled:
    impls['ref_compiled'] = ref_compiled
impls['ref_eager'] = ref_eager


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def stats(a, b):
    d = (a.double() - b.double()).abs()
    return {'rel': rel(a, b), 'max_abs': d.max().item(), 'mean_abs': d.mean().item()}


res = {'dump': args.dump, 'steps': steps, 'errors': {}, 'timing_ms': {}}
cfg = 3.0
with torch.no_grad():
    for s in steps:
        kc, ku = f'denoiser#{2 * s}', f'denoiser#{2 * s + 1}'
        z = D[f'{kc}.in.0.latent'].to(dev)
        cam = D[f'{kc}.in.0.camera'].to(dev)
        t = D[f'{kc}.in.1'].to(dev)
        rc = (D[f'{kc}.out.latent'].to(dev), D[f'{kc}.out.camera'].to(dev))
        ru = (D[f'{ku}.out.latent'].to(dev), D[f'{ku}.out.camera'].to(dev))
        rg = (1 + cfg) * rc[0] - cfg * ru[0]
        for name, fn in impls.items():
            (vc, cc), (vu, cu) = fn(z, cam, t)
            g = (1 + cfg) * vc - cfg * vu
            e = {'v_cond': stats(vc, rc[0]), 'v_uncond': stats(vu, ru[0]), 'guided': stats(g, rg),
                 'cam_cond_max_abs': (cc - rc[1]).abs().max().item(),
                 'v_cond_bitequal': bool(torch.equal(vc, rc[0])), 'v_uncond_bitequal': bool(torch.equal(vu, ru[0]))}
            res['errors'].setdefault(name, {})[s] = e
            print(f'step {s:2d} {name:16s} v_cond rel {e["v_cond"]["rel"]:.3e} (max {e["v_cond"]["max_abs"]:.2e})'
                  f'  v_uncond rel {e["v_uncond"]["rel"]:.3e}  guided rel {e["guided"]["rel"]:.3e}'
                  f'  bit-equal c/u {e["v_cond_bitequal"]}/{e["v_uncond_bitequal"]}', flush=True)

    # timing: one step (both passes), interleaved
    s = steps[min(1, len(steps) - 1)]
    kc = f'denoiser#{2 * s}'
    z, cam, t = (D[f'{kc}.in.0.latent'].to(dev), D[f'{kc}.in.0.camera'].to(dev), D[f'{kc}.in.1'].to(dev))
    names = list(impls)
    for name in names:  # warm-up
        for _ in range(2):
            impls[name](z, cam, t)
    times = {n: [] for n in names}
    for r in range(args.reps):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            impls[name](z, cam, t)
            torch.cuda.synchronize()
            times[name].append(1e3 * (time.perf_counter() - t0))
    for n in names:
        v = sorted(times[n])
        res['timing_ms'][n] = {'median': float(np.median(v)), 'min': v[0], 'max': v[-1], 'all': v}
        print(f'time/step {n:16s} median {np.median(v):8.2f} ms  min {v[0]:8.2f}  max {v[-1]:8.2f}', flush=True)

    if args.e2e:
        z0, c0 = D['rng.001.randn'].to(dev), D['rng.002.randn'].to(dev)
        ref_lat = D['octree#0.in.2'] if 'octree#0.in.2' in D else None
        ref_cam_lat = D[f'denoiser#{n_calls - 2}.in.0.camera']  # last input camera (before final update)
        for v in args.variants.split(','):
            for graph in (False, True):
                f = FastDenoiser(den, **VARIANTS[v], cuda_graph=graph)
                noise = TensorDict({'latent': z0.clone(), 'camera': c0.clone()}, batch_size=1)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                zl, cl = f.sample(noise, cond)
                torch.cuda.synchronize()
                el = time.perf_counter() - t0
                if graph:  # second run reuses the captured graph
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    zl, cl = f.sample(noise, cond)
                    torch.cuda.synchronize()
                    el2 = time.perf_counter() - t0
                else:
                    el2 = None
                e = {'latent': stats(zl.cpu(), ref_lat), 'sampling_s': el, 'sampling_s_second': el2}
                res.setdefault('e2e', {})[f'{v}{"_graph" if graph else ""}'] = e
                print(f'e2e {v} graph={graph}: latent rel {e["latent"]["rel"]:.3e} max {e["latent"]["max_abs"]:.2e}'
                      f'  sampling {el:.3f} s  (second {el2})', flush=True)

if args.json:
    with open(args.json, 'w') as fh:
        json.dump(res, fh, indent=1)

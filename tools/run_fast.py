"""Run the fast (or parity-mode) Python pipeline: one warm-up + N timed requests per config, interleaving allowed by
calling with several configs. Writes outputs + a dump (latent, camera, Gaussians) for tools/closeness.py.

python tools/run_fast.py --image demo/examples/fig-robot-cute.webp --mode synthetic --num_gaussians 262144 \
   --out /scratch/fast --name robot_262k [--parity] [--reps 3]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from fast.pipeline import FastIdolPipeline, save_outputs  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--image', required=True)
p.add_argument('--mode', default='synthetic')
p.add_argument('--num_gaussians', type=int, default=262144)
p.add_argument('--out', required=True)
p.add_argument('--name', required=True)
p.add_argument('--parity', action='store_true')
p.add_argument('--reps', type=int, default=0)
p.add_argument('--nocollapse', action='store_true')
p.add_argument('--slow_octree', action='store_true')
p.add_argument('--json', default=None)
a = p.parse_args()

t0 = time.perf_counter()
pipe = FastIdolPipeline(parity=a.parity, collapse_uncond=not a.nocollapse, fast_octree=not a.slow_octree)
torch.cuda.synchronize()
load_s = time.perf_counter() - t0
print(json.dumps({'load_s': load_s, 'parity': a.parity}), flush=True)


def run(save):
    tm = []
    torch.cuda.reset_peak_memory_stats()
    out = pipe(a.image, mode=a.mode, num_gaussians=a.num_gaussians, timings=tm)
    torch.cuda.synchronize()
    stages = {tm[i][0]: tm[i][1] - tm[i - 1][1] for i in range(1, len(tm))}
    total = tm[-1][1] - tm[0][1]
    t = time.perf_counter()
    if save:
        save_outputs(out, a.out, a.name)
    save_s = time.perf_counter() - t
    return out, {'request_s': total, 'stages': stages, 'save_s': save_s,
                 'peak_alloc_gb': torch.cuda.max_memory_allocated() / 2**30,
                 'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2**30}


out, r0 = run(True)
r0['kind'] = 'first'
print(json.dumps(r0), flush=True)
runs = [r0]
for i in range(a.reps):
    _, r = run(True)
    r['kind'] = 'timed'
    runs.append(r)
    print(json.dumps(r), flush=True)
# dump for closeness
D = {'latent': out.latent.float().cpu().contiguous(),
     'final.camera': torch.tensor([out.camera[k] for k in ('yaw', 'pitch', 'radius', 'fov')], dtype=torch.float64)}
if out.rows is not None:
    rows = out.rows.cpu()
    # PLY columns are y-up (x, -z, y): undo to the z-up object frame of Gaussian.xyz
    D['final.xyz'] = torch.stack([rows[:, 0], rows[:, 2], -rows[:, 1]], 1).contiguous()
else:
    D['final.xyz'] = out.gaussian.xyz.float().cpu().contiguous()
save_file(D, os.path.join(a.out, f'{a.name}.safetensors'))
if a.json:
    with open(a.json, 'w') as f:
        json.dump({'load_s': load_s, 'runs': runs}, f, indent=1)

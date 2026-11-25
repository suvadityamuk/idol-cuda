"""Profile one steady-state request of the official pipeline (after one warm-up request) with torch.profiler.

Prints the CUDA kernels ranked by self device time and writes a Chrome trace in which every pipeline stage is a
`stage:<name>` range (analyse it with tools/trace_stats.py: per-stage wall time, GPU busy time, launches).

python tools/profile_ref.py [--no_optimize] [--attn sdpa|flash_attn] [--steps 50] [--out trace.json.gz]
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument('--no_optimize', dest='optimize', action='store_false')
p.add_argument('--attn', default='sdpa')
p.add_argument('--steps', type=int, default=50)
p.add_argument('--image', default='demo/examples/fig-robot-cute.webp')
p.add_argument('--mode', default='synthetic')
p.add_argument('--num_gaussians', type=int, default=262144)
p.add_argument('--out', default=None)
p.add_argument('--top', type=int, default=60)
args = p.parse_args()
os.environ['ATTN_BACKEND'] = args.attn

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

import idol.pipeline as ip  # noqa: E402
from idol import IdolPipeline  # noqa: E402

pipe = IdolPipeline('/w/idol', device='cuda', optimize=args.optimize,
                    image_encoders={'dinov3': '/w/dinov3', 'flux2_vae': '/w/flux2'})


def ranged(name, fn):
    def wrapped(*a, **k):
        with record_function(f'stage:{name}'):
            return fn(*a, **k)
    return wrapped


pipe.preprocess = ranged('preprocess', pipe.preprocess)
pipe.image_encoder = ranged('encoders', pipe.image_encoder.__call__)
ip.sample_flow = ranged('sampling', ip.sample_flow)
ip.OctreeProbabilityFixedlenDecoder.sample = staticmethod(ranged('octree', ip.OctreeProbabilityFixedlenDecoder.sample))
pipe.decoder_gs.forward = ranged('gs_decoder', pipe.decoder_gs.forward)
_tr = pipe.decoder_gs.to_representation
pipe.decoder_gs.to_representation = ranged('to_gaussians', _tr)

pipe(args.image, mode=args.mode, steps=args.steps, num_gaussians=args.num_gaussians)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function('stage:request'):
        out = pipe(args.image, mode=args.mode, steps=args.steps, num_gaussians=args.num_gaussians)
        torch.cuda.synchronize()
    with record_function('stage:save_ply'):
        out.gaussian.save_ply('/tmp/profile.ply')
ka = prof.key_averages()
tot = sum(e.self_device_time_total for e in ka)
print(f'total self device time: {tot / 1e3:.1f} ms')
for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:args.top]:
    if e.self_device_time_total <= 0:
        continue
    print(f'{e.self_device_time_total / 1e3:10.2f} ms {100 * e.self_device_time_total / tot:5.1f}% {e.count:7d}  '
          f'{e.key[:150]}')
if args.out:
    prof.export_chrome_trace(args.out)

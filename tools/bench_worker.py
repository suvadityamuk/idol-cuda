"""Benchmark worker: loads one implementation once, then serves requests from stdin (one per line:
"image mode num_gaussians name outdir"), printing "RESULT {json}" and "READY". Used by tools/bench.py.

impls: ref_compiled (idol-sdk as shipped: optimize=True), ref_eager (--no_optimize), py_fast, py_parity.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
p = argparse.ArgumentParser()
p.add_argument('--impl', required=True)
a = p.parse_args()

import torch  # noqa: E402

t0 = time.perf_counter()
if a.impl in ('ref_compiled', 'ref_eager'):
    from idol import IdolPipeline
    pipe = IdolPipeline('/w/idol', device='cuda', optimize=a.impl == 'ref_compiled',
                        image_encoders={'dinov3': '/w/dinov3', 'flux2_vae': '/w/flux2'})
else:
    from fast.pipeline import FastIdolPipeline, save_outputs
    pipe = FastIdolPipeline(parity=a.impl == 'py_parity')
    if a.impl == 'py_fast':
        print('kernels: ' + pipe.K.SOURCE, file=sys.stderr, flush=True)
torch.cuda.synchronize()
print('LOAD ' + json.dumps({'load_s': time.perf_counter() - t0}), flush=True)
print('READY', flush=True)
for line in sys.stdin:
    image, mode, ng, name, outdir = line.split()
    os.makedirs(outdir, exist_ok=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    out = pipe(image, mode=mode, seed=42, steps=50, num_gaussians=int(ng))
    torch.cuda.synchronize()
    req = time.perf_counter() - t
    t = time.perf_counter()
    if a.impl.startswith('ref'):
        out.gaussian.save_ply(os.path.join(outdir, f'{name}.ply'))
        out.preprocessed_image.save(os.path.join(outdir, f'{name}_input.png'))
        with open(os.path.join(outdir, f'{name}_camera.json'), 'w') as f:
            json.dump(out.camera, f, indent=2)
    else:
        save_outputs(out, outdir, name)
    save = time.perf_counter() - t
    print('RESULT ' + json.dumps({'request_s': req, 'save_s': save,
                                  'peak_alloc_gb': torch.cuda.max_memory_allocated() / 2**30,
                                  'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2**30}), flush=True)
    print('READY', flush=True)

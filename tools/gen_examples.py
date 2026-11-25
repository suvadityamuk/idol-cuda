#!/usr/bin/env python3
"""Fast-build PLYs for the example images in assets/examples (one process, weights loaded once).

  python tools/gen_examples.py --local-kernels kernel --out outputs/examples [--official]

--official also writes the official eager pipeline's PLY (<name>_official.ply) for the same seed.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EXAMPLES = [('fig-robot-cute', 'synthetic'), ('food-avocado', 'real'), ('obj-backpack', 'synthetic'),
            ('toy-train', 'synthetic'), ('food-donut', 'synthetic'), ('obj-power-drill', 'synthetic'),
            ('fig-fox', 'synthetic'), ('food-mushroom', 'synthetic')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--local-kernels', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--official', action='store_true')
    p.add_argument('--only', nargs='*', default=None)
    a = p.parse_args()
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    from fast import runner
    runner.check_cuda()
    runner.use_kernels(a.local_kernels)
    runner.load_kernels()
    attn, note = runner.pick_attention()
    if note:
        print(note)
    from fast.pipeline import FastIdolPipeline
    pipe = FastIdolPipeline(runner.idol_weights(), *runner.encoder_paths(), attn=attn)
    os.makedirs(a.out, exist_ok=True)
    for name, mode in EXAMPLES:
        if a.only and name not in a.only:
            continue
        img = os.path.join(ROOT, 'assets', 'examples', name + '.webp')
        out, s = runner.timed_request(pipe, img, mode=mode, seed=42)
        out.save_ply(os.path.join(a.out, f'{name}_fast.ply'))
        line = f'{name} ({mode}): fast {s:.2f} s'
        if a.official:
            ref, s2 = runner.timed_request(pipe.ref, img, mode=mode, seed=42)
            ref.gaussian.save_ply(os.path.join(a.out, f'{name}_official.ply'))
            line += f', official eager {s2:.2f} s'
        print(line + ' (first request includes warm-up)', flush=True)


if __name__ == '__main__':
    main()

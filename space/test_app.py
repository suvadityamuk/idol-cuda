#!/usr/bin/env python3
"""End-to-end test of the Space's app logic outside ZeroGPU (the `spaces` decorator is a no-op there).

Run from an assembled Space directory (space/assemble.sh OUT):
  cd OUT && python test_app.py --ref REF.ply --out DIR
Checks: both buttons (fast only; compare), the values the timing cards show, the output files, that the official
run inside the app is byte-identical to the eager reference PLY (same seed), the GPU self-check, and the
missing-token start-up path.
"""
import argparse
import filecmp
import html
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def text(h):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', h or ''))).strip()


def drain(gen):
    # Each .ply is sent once and skipped (gr.skip()) afterwards: keep the last file path seen in each viewer slot.
    n, last, t0, seen, plys = 0, None, time.perf_counter(), [], [None, None]
    for last in gen:
        n += 1
        for i in (0, 1):
            if isinstance(last[i], str) or last[i] is None:
                plys[i] = last[i]
        if n % 8 == 1:
            seen.append(f'{time.perf_counter() - t0:6.1f}s  fast: {text(last[2])[:90]} | official: {text(last[3])[:60]}')
    return last, plys, n, seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='eager reference PLY for fig-robot-cute, synthetic, seed 42, 262,144')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fails = []

    # missing token: the app must start and explain
    env = {k: v for k, v in os.environ.items() if k != 'HF_TOKEN'}
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    r = subprocess.run([sys.executable, '-c', (
        'import app, gradio as gr\n'
        'print("STATE_ERROR:", app.STATE["error"][:160])\n'
        'print("KERNELS:", app.STATE["kernels"], app.STATE.get("kernel_variant"))\n'
        'from PIL import Image\n'
        'try:\n'
        '    next(app.generate(Image.new("RGB", (64, 64)), "Real photo", 42, 50, "262,144", False, False))\n'
        '    print("NO ERROR RAISED")\n'
        'except gr.Error as e:\n'
        '    print("gr.Error raised:", str(e)[:120])\n')], env=env, capture_output=True, text=True, timeout=600)
    print('--- missing-token start-up\n' + '\n'.join(l for l in r.stdout.splitlines() if l.strip()))
    if 'HF_TOKEN secret' not in r.stdout or 'gr.Error raised' not in r.stdout:
        fails.append('missing-token path')
        print(r.stderr[-2000:])

    t = time.perf_counter()
    import app
    print(f'--- app import (loads Idol): {time.perf_counter() - t:.1f} s; STATE: '
          f'{ {k: v for k, v in app.STATE.items() if k != "error"} }, error: {app.STATE["error"]}')
    if app.STATE['error']:
        sys.exit('app did not load')
    from PIL import Image
    img = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'fig-robot-cute.webp'))
    for compare in (False, True):
        t = time.perf_counter()
        last, (fast_ply, off_ply), n, seen = drain(app.generate(img, 'Synthetic (render / generated)', 42, 50, '262,144', False, compare))
        wall = time.perf_counter() - t
        _, _, card_f, card_o, summ, info, seed = last
        print(f'--- button {"Compare with official" if compare else "Generate (fast build)"}: {n} UI updates, '
              f'{wall:.1f} s wall')
        print('\n'.join(seen))
        print('fast card:    ', text(card_f))
        print('official card:', text(card_o))
        print('headline:     ', text(summ)[:400])
        print('details:      ', text(info)[:400])
        for label, f in (('fast', fast_ply), ('official', off_ply)):
            if f:
                print(f'{label} file: {f} ({os.path.getsize(f)} bytes)')
                shutil.copy(f, os.path.join(a.out, f'robot_{label}{"_compare" if compare else ""}.ply'))
        if not fast_ply or not os.path.getsize(fast_ply):
            fails.append(f'fast output (compare={compare})')
        if compare:
            if not off_ply:
                fails.append('official output')
            else:
                same = filecmp.cmp(off_ply, a.ref, shallow=False)
                print('official run in the app vs eager reference PLY:', 'BYTE-IDENTICAL' if same else 'DIFFERS')
                if not same:
                    fails.append('official not byte-identical to the eager reference')
            m = re.search(r'([\d.]+)× faster', text(summ))
            print('speed-up shown:', m.group(1) if m else None)
        elif off_ply:
            fails.append('fast-only run produced an official output')
    print('--- self-check\n' + app.run_diag())
    print('APP TEST', 'FAILED: ' + ', '.join(fails) if fails else 'PASSED')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()

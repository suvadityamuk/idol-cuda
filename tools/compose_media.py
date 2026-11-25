#!/usr/bin/env python3
"""README media from turntable frames (tools/turntable.py --npz): no GPU needed.

  python tools/compose_media.py hero FRAMES.npz INPUT.webp -o assets/hero --labels input official fast \
      --columns input c1 c0 [--size 320] [--max-mb 5]
  python tools/compose_media.py turntable FRAMES.npz INPUT.webp -o assets/turntable-robot --columns input c0

Writes <out>.gif, <out>.mp4 and <out>.png (first frame). The npz holds one uint8 [F, H, W, 3] array per rendered PLY
(c0, c1, ...); "input" is the input image fitted to the same square. For the GIF, the colour count and then the size
are reduced until the file is under --max-mb.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turntable import compose, fit_input, save_gif, save_mp4  # noqa: E402


def build(npz, inp, columns, labels, size, frames_step=1):
    d = np.load(npz)
    cols = []
    for c in columns:
        if c == 'input':
            cols.append(fit_input(inp, size))
        else:
            arr = d[c][::frames_step]
            if arr.shape[1] != size:
                arr = np.stack([np.asarray(Image.fromarray(f).resize((size, size), Image.LANCZOS)) for f in arr])
            cols.append(list(arr))
    return compose(cols, labels, size)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('kind', choices=['hero', 'turntable'])
    p.add_argument('npz')
    p.add_argument('input')
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--columns', nargs='+', default=['input', 'c0'])
    p.add_argument('--labels', nargs='*', default=None)
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--mp4-size', type=int, default=384)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--max-mb', type=float, default=5.0)
    a = p.parse_args()
    save_mp4(build(a.npz, a.input, a.columns, a.labels, a.mp4_size), a.out + '.mp4', a.fps)
    size, colors = a.size, 128
    while True:
        frames = build(a.npz, a.input, a.columns, a.labels, size)
        save_gif(frames, a.out + '.gif', a.fps, colors)
        mb = os.path.getsize(a.out + '.gif') / 1e6
        if mb <= a.max_mb or size <= 192:
            break
        if colors > 64:
            colors //= 2
        else:
            size -= 32
    frames[0].save(a.out + '.png')
    print(f'{a.out}.gif {mb:.2f} MB ({size}px, {colors} colours), '
          f'{a.out}.mp4 {os.path.getsize(a.out + ".mp4") / 1e6:.2f} MB, {a.out}.png')


if __name__ == '__main__':
    main()

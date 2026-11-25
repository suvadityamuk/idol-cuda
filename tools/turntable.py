#!/usr/bin/env python3
"""Turntable renders of Idol Gaussian PLYs with gsplat (CUDA): GIF, MP4 and stills, optionally with the input photo
and several PLYs side by side.

Idol's world is y-down (the object's up is -y) and the front faces +x, so the orbit is around the y axis and starts
at azimuth 90 degrees (camera on +x). SH degree 0, as Idol writes it.

  python tools/turntable.py out.ply -o out.gif [--mp4 out.mp4] [--input photo.png] [--frames 72] [--size 512]
  python tools/turntable.py a.ply b.ply --labels official fast --input photo.png -o hero.gif

Needs: torch with CUDA, gsplat (`pip install gsplat`), Pillow; imageio-ffmpeg for --mp4.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

BG = (12, 12, 12)


def load_ply(path, dev='cuda'):
    with open(path, 'rb') as f:
        n, props = 0, []
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        data = torch.from_numpy(np.frombuffer(f.read(), dtype='<f4').reshape(n, len(props)).copy()).to(dev)
    c = {p: data[:, i] for i, p in enumerate(props)}
    means = torch.stack([c['x'], c['y'], c['z']], 1)
    quats = torch.stack([c[f'rot_{i}'] for i in range(4)], 1)
    scales = torch.exp(torch.stack([c[f'scale_{i}'] for i in range(3)], 1))
    opac = torch.sigmoid(c['opacity'])
    sh0 = torch.stack([c[f'f_dc_{i}'] for i in range(3)], 1)[:, None, :]
    return means, quats, scales, opac, sh0


def view(az_deg, el_deg, radius):
    a, e = math.radians(az_deg), math.radians(el_deg)
    eye = torch.tensor([radius * math.cos(e) * math.sin(a), -radius * math.sin(e), radius * math.cos(e) * math.cos(a)])
    up = torch.tensor([0.0, -1.0, 0.0])
    f = F.normalize(-eye, dim=0)
    r = F.normalize(torch.linalg.cross(f, up), dim=0)
    u = torch.linalg.cross(r, f)
    M = torch.eye(4)
    M[:3, :3] = torch.stack([r, -u, f], 0)  # OpenCV camera: x right, y down, z forward
    M[:3, 3] = -M[:3, :3] @ eye
    return M


@torch.no_grad()
def render_frames(ply, frames=72, size=512, elev=20.0, start_az=90.0, radius=2.0, fov=35.0, bg=BG, batch=12):
    """-> list of HxWx3 uint8 frames, one full turn."""
    from gsplat import rasterization
    means, quats, scales, opac, sh0 = load_ply(ply)
    fpx = 0.5 * size / math.tan(math.radians(fov) / 2)
    K = torch.tensor([[fpx, 0, size / 2], [0, fpx, size / 2], [0, 0, 1]], dtype=torch.float32, device='cuda')
    bgt = torch.tensor(bg, dtype=torch.float32, device='cuda') / 255
    out = []
    for s in range(0, frames, batch):
        V = torch.stack([view(start_az + 360.0 * i / frames, elev, radius) for i in range(s, min(frames, s + batch))])
        V = V.cuda()
        img, alpha, _ = rasterization(means, quats, scales, opac, sh0, V, K[None].expand(len(V), 3, 3), size, size,
                                      sh_degree=0)
        img = img.clamp(0, 1) + (1 - alpha) * bgt
        out += list((img.clamp(0, 1) * 255).round().byte().cpu().numpy())
    return out


def fit_input(path, size, bg=BG):
    im = Image.open(path).convert('RGBA')
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGBA', (size, size), bg + (255,))
    canvas.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas.convert('RGB'))


def _font(px):
    for name in ('DejaVuSans-Bold.ttf', 'Arial Bold.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(columns, labels=None, size=512):
    """columns: list of per-column frame lists (or a single HxWx3 array, repeated); -> list of PIL frames."""
    n = max(len(c) for c in columns if isinstance(c, list))
    bar = int(size * 0.09) if labels else 0
    font = _font(max(12, int(size * 0.045)))
    frames = []
    for i in range(n):
        tiles = [c[i] if isinstance(c, list) else c for c in columns]
        row = np.concatenate(tiles, 1)
        im = Image.new('RGB', (row.shape[1], row.shape[0] + bar), BG)
        im.paste(Image.fromarray(row), (0, bar))
        if labels:
            d = ImageDraw.Draw(im)
            for j, text in enumerate(labels):
                w = d.textlength(text, font=font)
                d.text((j * size + (size - w) / 2, bar * 0.2), text, fill=(235, 235, 235), font=font)
        frames.append(im)
    return frames


def save_gif(frames, path, fps=24, colors=128):
    pal = [f.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True,
                disposal=1)


def save_mp4(frames, path, fps=24):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=16,
                            pixelformat='yuv420p') as w:
        for f in frames:
            w.append_data(np.asarray(f))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ply', nargs='+')
    p.add_argument('-o', '--out', default=None, help='.gif')
    p.add_argument('--npz', default=None, help='save the raw frames of each PLY (uint8 [F, H, W, 3]) instead')
    p.add_argument('--mp4', default=None)
    p.add_argument('--still', default=None, help='PNG of the first frame')
    p.add_argument('--input', default=None, help='input photo shown in a column on the left')
    p.add_argument('--labels', nargs='*', default=None)
    p.add_argument('--frames', type=int, default=72)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--elev', type=float, default=20.0)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--colors', type=int, default=128)
    a = p.parse_args()
    cols = [render_frames(x, a.frames, a.size, a.elev) for x in a.ply]
    if a.npz:
        np.savez_compressed(a.npz, **{f'c{i}': np.stack(c) for i, c in enumerate(cols)})
        return
    if a.input:
        cols.insert(0, fit_input(a.input, a.size))
    frames = compose(cols, a.labels, a.size)
    save_gif(frames, a.out, a.fps, a.colors)
    if a.still:
        frames[0].save(a.still)
    if a.mp4:
        save_mp4(frames, a.mp4, a.fps)


if __name__ == '__main__':
    main()

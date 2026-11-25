"""Rendered-view comparison of Gaussian PLYs with gsplat: 8 fixed cameras around the object (radius 2, elevation 20
and -10 degrees, 4 azimuths, 40 degree FOV, 512 x 512, black background); PSNR and SSIM of each candidate against
the reference PLY, averaged over the views.

python tools/render_psnr.py REF.ply CAND.ply [CAND2.ply ...] [--json out.json]
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


def load_ply(path):
    with open(path, 'rb') as f:
        n = 0
        props = []
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        data = np.frombuffer(f.read(), dtype='<f4').reshape(n, len(props))
    cols = {p: torch.from_numpy(data[:, i].copy()) for i, p in enumerate(props)}
    dev = 'cuda'
    means = torch.stack([cols['x'], cols['y'], cols['z']], 1).to(dev)
    quats = torch.stack([cols[f'rot_{i}'] for i in range(4)], 1).to(dev)
    scales = torch.exp(torch.stack([cols[f'scale_{i}'] for i in range(3)], 1)).to(dev)
    opac = torch.sigmoid(cols['opacity']).to(dev)
    sh0 = torch.stack([cols[f'f_dc_{i}'] for i in range(3)], 1)[:, None, :].to(dev)
    return means, quats, scales, opac, sh0


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    eye, target, up = (torch.tensor(v, dtype=torch.float32) for v in (eye, target, up))
    f = F.normalize(target - eye, dim=0)
    r = F.normalize(torch.linalg.cross(f, up), dim=0)
    u = torch.linalg.cross(r, f)
    R = torch.stack([r, -u, f], 0)  # OpenCV camera: x right, y down, z forward
    t = -R @ eye
    M = torch.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def cameras(size=512, fov=40.0, radius=2.0):
    views = []
    for el in (20.0, -10.0):
        for az in (0.0, 90.0, 180.0, 270.0):
            e, a = math.radians(el), math.radians(az)
            eye = (radius * math.cos(e) * math.sin(a), radius * math.sin(e), radius * math.cos(e) * math.cos(a))
            views.append(look_at(eye))
    fpx = 0.5 * size / math.tan(math.radians(fov) / 2)
    K = torch.tensor([[fpx, 0, size / 2], [0, fpx, size / 2], [0, 0, 1]], dtype=torch.float32)
    return torch.stack(views).cuda(), K[None].expand(len(views), 3, 3).contiguous().cuda(), size


def render(ply, cams):
    from gsplat import rasterization
    means, quats, scales, opac, sh0 = load_ply(ply)
    V, K, size = cams
    img, _, _ = rasterization(means, quats, scales, opac, sh0, V, K, size, size, sh_degree=0)  # black background
    return img.clamp(0, 1)  # [V, H, W, 3]


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean().item()


def ssim(a, b, win=11, sigma=1.5):
    a, b = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    g = torch.exp(-((torch.arange(win, device=a.device) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None].expand(3, 1, win, win)

    def filt(x):
        return F.conv2d(x, w, groups=3)
    mu_a, mu_b = filt(a), filt(b)
    saa, sbb, sab = filt(a * a) - mu_a ** 2, filt(b * b) - mu_b ** 2, filt(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean().item()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ref')
    p.add_argument('cands', nargs='+')
    p.add_argument('--json', default=None)
    a = p.parse_args()
    cams = cameras()
    ref = render(a.ref, cams)
    out = {}
    for c in a.cands:
        img = render(c, cams)
        out[c] = {'psnr': psnr(img, ref), 'ssim': ssim(img, ref)}
        print(json.dumps({c: out[c]}), flush=True)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)

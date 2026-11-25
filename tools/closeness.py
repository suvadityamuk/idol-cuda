"""Closeness of a candidate run to a reference dump: latent rel L2 (sampled latent), camera error, Chamfer distance
between Gaussian centres, and Gaussian attribute errors when the anchors match.

Inputs are safetensors dumps with `octree#0.in.2` or `latent` (sampled latent [1, 8192, 16]), `final.camera`
(yaw, pitch, radius, fov float64) and `final.xyz` [N, 3].

python tools/closeness.py REF.safetensors CAND.safetensors [--json out.json]
"""
import argparse
import json

import torch
from safetensors.torch import load_file


def latent_of(D):
    for k in ('octree#0.in.2', 'latent'):
        if k in D:
            return D[k].double()
    return None


def chamfer(a, b, chunk=8192):
    """Symmetric mean nearest-neighbour distance (not squared) between point sets a [N,3], b [M,3] (on the GPU)."""
    a, b = a.cuda().float(), b.cuda().float()

    def one_way(x, y):
        out = []
        for i in range(0, x.shape[0], chunk):
            d = torch.cdist(x[i:i + chunk], y)
            out.append(d.min(1).values)
        return torch.cat(out).double().mean().item()
    return 0.5 * (one_way(a, b) + one_way(b, a))


def camera_vec(path):
    import json
    with open(path) as f:
        c = json.load(f)
    return torch.tensor([c[k] for k in ('yaw', 'pitch', 'radius', 'fov')], dtype=torch.float64)


def ply_dump(path):
    """A .ply (+ its _camera.json) as a dump: final.xyz in the z-up object frame, final.camera."""
    import os
    import numpy as np
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
        d = np.frombuffer(f.read(), dtype='<f4').reshape(n, len(props))
    x, y, z = (torch.from_numpy(d[:, props.index(k)].copy()) for k in ('x', 'y', 'z'))
    D = {'final.xyz': torch.stack([x, z, -y], 1)}  # PLY is y-up (x, -z, y)
    cj = path[:-4] + '_camera.json'
    if os.path.exists(cj):
        D['final.camera'] = camera_vec(cj)
    return D


def compare(ref, cand, cand_camera=None):
    R = load_file(ref)
    Cd = ply_dump(cand) if cand.endswith('.ply') else load_file(cand)
    if cand_camera:
        Cd['final.camera'] = camera_vec(cand_camera)
    res = {}
    la, lb = latent_of(R), latent_of(Cd)
    if la is not None and lb is not None:
        d = (la - lb)
        res['latent_rel_l2'] = (d.norm() / la.norm()).item()
        res['latent_max_abs'] = d.abs().max().item()
        res['latent_mean_abs'] = d.abs().mean().item()
    if 'final.camera' in R and 'final.camera' in Cd:
        res['camera_abs_err'] = (R['final.camera'] - Cd['final.camera']).abs().tolist()  # yaw, pitch (rad), radius, fov
    if 'final.xyz' in R and 'final.xyz' in Cd:
        res['chamfer_xyz'] = chamfer(R['final.xyz'], Cd['final.xyz'])
        res['n_gaussians'] = [R['final.xyz'].shape[0], Cd['final.xyz'].shape[0]]
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ref')
    p.add_argument('cand')
    p.add_argument('--json', default=None)
    p.add_argument('--cand_camera', default=None, help='camera json of the candidate (native runs)')
    a = p.parse_args()
    r = compare(a.ref, a.cand, a.cand_camera)
    print(json.dumps(r))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(r, f, indent=1)

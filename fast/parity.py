"""Parity mode of the Python build: the reference sampler (eager, sdpa, batch_cfg=False) with only bit-equal reuse.

LatentSeqMMFlowModel.forward is split into its independent parts, calling the reference's own modules in the same
op order: the context stream depends only on the image features (computed once per request), and the per-step prefix
(time embedding, adaLN, input layer + positions, noise refiner, camera refiner) is identical for the cond and uncond
passes (computed once per step). Each reused tensor is a deterministic function of identical inputs, so the result is
bit-identical to recomputing it (checked by byte comparison against the reference dumps, see docs/parity.md).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _context(den, f1, f2):
    h = (den.cond_embedder(f1) + den.cond_embedder2(f2)).type(den.dtype)
    for repo, block in zip(den.context_repo_layers, den.context_refiner):
        h = block(h, mod=None, rotary_emb=repo(h))
    return h


def _prefix(den, z, cam, t):
    h_x = den.input_layer(z).type(den.dtype)
    t_emb = den.t_embedder(t)
    t_mod = den.adaLN_modulation(t_emb).type(den.dtype)
    h_x = h_x + den.pos_embedder(den.pos_pe).type(den.dtype)
    for repo, block in zip(den.noise_repo_layers, den.noise_refiner):
        h_x = block(h_x, mod=t_mod, rotary_emb=repo(h_x))
    h_cam = den.cam_refiner(cam).type(den.dtype)
    return t_emb, t_mod, h_x, h_cam


def _joint(den, pre, h_cond, z, cam):
    t_emb, t_mod, h_x, h_cam = pre
    h = torch.cat([h_x, h_cond, h_cam], dim=1)
    for repo, block in zip(den.repo_layers, den.blocks):
        h = block(h, mod=t_mod, rotary_emb=repo(h))
    hx = h[:, :z.shape[1]].type(z.dtype)
    hx = F.layer_norm(hx, hx.shape[-1:])
    hc = h[:, -cam.shape[1]:].type(cam.dtype)
    hc = F.layer_norm(hc, hc.shape[-1:])
    shift, scale = (den.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
    hx = hx * (1 + scale) + shift
    hc = hc * (1 + scale) + shift
    return den.out_layer(hx), den.cam_out_layer(hc)


@torch.no_grad()
def sample_parity(den, noise, cond, steps=50, rescale_t=3.0, cfg=3.0):
    z, cam = noise['latent'], noise['camera']
    f1, f2 = cond['feature1'], cond['feature2']
    hc = _context(den, f1, f2)
    hu = _context(den, torch.zeros_like(f1), torch.zeros_like(f2)) if cfg > 0 else None
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    for i in range(steps):
        t, t_prev = t_seq[i], t_seq[i + 1]
        tt = torch.tensor([1000 * t] * z.shape[0], device=z.device, dtype=torch.float32)
        pre = _prefix(den, z, cam, tt)
        vz, vc = _joint(den, pre, hc, z, cam)
        if cfg > 0:
            uz, uc = _joint(den, pre, hu, z, cam)
            vz = (1 + cfg) * vz - cfg * uz
            vc = (1 + cfg) * vc - cfg * uc
        z = z - vz * (t - t_prev)
        cam = cam - vc * (t - t_prev)
    return z, cam

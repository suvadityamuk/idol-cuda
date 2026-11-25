"""Closeness table (markdown) from the closeness JSONs: reference spread vs fast builds.

python tools/closeness_table.py
"""
import json
import os

L = 'logs'
CFG = ['robot_262k', 'avocado_262k', 'robot_1m', 'avocado_1m']
SRC = [
    ('ref: compiled (shipped) run A vs eager', 'bench/spread_{c}_compiled_a.json'),
    ('ref: compiled run B vs eager', 'bench/spread_{c}_compiled_b.json'),
    ('ref: eager flash-attn vs eager sdpa', 'bench/spread_{c}_eager_flash_a.json'),
    ('ref: compiled A vs compiled B (run-to-run)', 'bench/spread_{c}_compiled_a_vs_b.json'),
    ('**fast, Python (FA3)** vs eager', 'dev8/close_fast_{c}.json'),
    ('fast, native (cuDNN) vs eager', 'dev7/close_nfast_{c}.json'),
]


def fmt(d):
    cam = d.get('camera_abs_err')
    cam = f'{cam[0]:.1e} / {cam[1]:.1e} / {cam[2]:.1e} / {cam[3]:.3f}' if cam else '-'
    lr = d.get('latent_rel_l2')
    return (f"{lr:.2e}" if lr is not None else '-'), cam, f"{d.get('chamfer_xyz', float('nan')):.2e}"


for c in CFG:
    print(f'\n### {c}\n')
    print('| comparison | latent rel L2 | camera abs err: yaw / pitch (rad) / radius / fov (deg) | Chamfer, Gaussian centres |')
    print('|---|---|---|---|')
    for name, pat in SRC:
        p = os.path.join(L, pat.format(c=c))
        if os.path.exists(p):
            lr, cam, ch = fmt(json.load(open(p)))
            print(f'| {name} | {lr} | {cam} | {ch} |')

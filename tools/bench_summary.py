"""Markdown tables from tools/bench.py's JSON lines.

python tools/bench_summary.py logs/bench/bench.jsonl [--ref_sha logs/stage1/job.log]
"""
import argparse
import json
import re
import statistics
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('jsonl')
p.add_argument('--ref_sha', default=None, help='stage-1 job.log holding the eager reference PLY sha256 per config')
a = p.parse_args()
recs = [json.loads(line) for line in open(a.jsonl)]
env = next((r for r in recs if r['kind'] == 'env'), {})
loads = {r['row']: r['load_s'] for r in recs if r['kind'] == 'load'}
timed = defaultdict(list)
warm = {}
for r in recs:
    if r['kind'] == 'timed':
        timed[(r['config'], r['row'])].append(r)
    elif r['kind'] == 'warmup':
        warm[(r['config'], r['row'])] = r
errors = [r for r in recs if r['kind'] == 'error']

ref_sha = {}
if a.ref_sha:
    txt = open(a.ref_sha).read()
    order = ['robot_262k', 'robot_262k', 'avocado_262k', 'avocado_262k', 'robot_1m', 'robot_1m', 'avocado_1m',
             'avocado_1m']
    shas = re.findall(r'"ply_sha256": "([0-9a-f]+)"', txt)[:8]  # eager runs A, B per config, in stage-1 order
    for cfg, s in zip(order, shas):
        ref_sha.setdefault(cfg, s)

LABEL = {'ref_compiled': 'reference, shipped default (torch.compile + batched CFG + cuDNN attn + TF32)',
         'ref_eager': 'reference `--no_optimize` (eager)', 'native_parity': 'parity, native C++ (`run_native.sh --parity`)',
         'py_parity': 'parity, Python (`run_kernels.py --parity`)', 'py_fast': '**fast, Python + idol_cuda kernels** (`run_kernels.py`)',
         'native_fast': 'fast, native C++ (`run_native.sh`)'}
print(f"Session: {env.get('gpu')}, torch {env.get('torch')}, CUDA {env.get('cuda')}, cuDNN {env.get('cudnn')}, "
      f"commit {env.get('commit')}\n")
cfgs = []
for (c, _r) in timed:
    if c not in cfgs:
        cfgs.append(c)
for c in cfgs:
    rows = [r for (cc, r) in timed if cc == c]
    base = statistics.median(x['request_s'] for x in timed[(c, 'ref_compiled')]) if (c, 'ref_compiled') in timed else None
    print(f'### {c}\n')
    print('| build | request s, median (min-max) | + export s | speed-up vs shipped default | first request s | '
          'load s | peak alloc / reserved GB | PLY |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        xs = timed[(c, r)]
        req = sorted(x['request_s'] for x in xs)
        sv = [x.get('save_s') or 0.0 for x in xs]
        med = statistics.median(req)
        sp = f'{base / med:.2f}x' if base else ''
        w = warm.get((c, r), {})
        pk = max(x['peak_alloc_gb'] for x in xs), max(x['peak_reserved_gb'] for x in xs)
        shas = {x['ply_sha256'] for x in xs}
        if r in ('native_parity', 'py_parity', 'ref_eager') and c in ref_sha:
            ply = 'byte-identical to eager ref' if shas == {ref_sha[c]} else f'DIFFERS ({len(shas)} distinct)'
        else:
            ply = f'{len(shas)} distinct sha over {len(xs)} runs'
        print(f'| {LABEL.get(r, r)} | {med:.3f} ({req[0]:.3f}-{req[-1]:.3f}), n={len(req)} | '
              f'{statistics.median(sv):.3f} | {sp} | {w.get("request_s", float("nan")):.2f} | {loads.get(r, float("nan")):.1f} | '
              f'{pk[0]:.2f} / {pk[1]:.2f} | {ply} |')
    print()
if errors:
    print('Errors:', json.dumps(errors, indent=1))

"""Per-stage statistics from a torch.profiler Chrome trace written by tools/profile_ref.py.

For every `stage:<name>` CPU range: wall time, GPU busy time (union of kernel/memcpy intervals launched inside the
range, by correlation id), number of kernels launched, idle GPU fraction, and the top kernels by time.

python tools/trace_stats.py trace.json.gz [--top 15]
"""
import argparse
import gzip
import json
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('trace')
p.add_argument('--top', type=int, default=15)
args = p.parse_args()
op = gzip.open if args.trace.endswith('.gz') else open
with op(args.trace, 'rt') as f:
    tr = json.load(f)
ev = tr['traceEvents'] if isinstance(tr, dict) else tr
stages = [e for e in ev if e.get('ph') == 'X' and str(e.get('name', '')).startswith('stage:')]
launches = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('cuda_runtime', 'cuda_driver')]
gpu = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
by_corr = defaultdict(list)
for g in gpu:
    c = g.get('args', {}).get('correlation')
    if c is not None:
        by_corr[c].append(g)


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


rows = []
for st in sorted(stages, key=lambda e: e['ts']):
    s0, s1 = st['ts'], st['ts'] + st['dur']
    ks = []
    nl = 0
    for la in launches:
        if s0 <= la['ts'] <= s1:
            c = la.get('args', {}).get('correlation')
            if c in by_corr:
                nl += 1
                ks.extend(by_corr[c])
    busy = union([(k['ts'], k['ts'] + k['dur']) for k in ks])
    span = (max(k['ts'] + k['dur'] for k in ks) - min(k['ts'] for k in ks)) if ks else 0
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for k in ks:
        tot[k['name']] += k['dur']
        cnt[k['name']] += 1
    rows.append((st['name'], st['dur'] / 1e3, busy / 1e3, span / 1e3, nl, tot, cnt))
for name, wall, busy, span, nl, tot, cnt in rows:
    print(f'{name:22s} wall {wall:9.1f} ms  gpu busy {busy:9.1f} ms  gpu span {span:9.1f} ms  launches {nl:7d}  '
          f'busy/wall {busy / max(wall, 1e-9):5.2f}')
for name, wall, busy, span, nl, tot, cnt in rows:
    if name == 'stage:request':
        continue
    print(f'\n== {name}: top kernels')
    T = sum(tot.values()) or 1
    for k, v in sorted(tot.items(), key=lambda x: -x[1])[:args.top]:
        print(f'  {v / 1e3:9.2f} ms {100 * v / T:5.1f}% x{cnt[k]:6d}  {k[:130]}')

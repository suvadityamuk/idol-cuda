"""Compare two checkpoint dumps (safetensors) key by key.

For every key present in both: byte-identical or not; if not, number of differing elements, first differing index,
max abs error, max relative error, and max ULP distance (for floating types).
Exit code 0 only when every compared key is byte-identical (use --allow-diff to always exit 0).

python tools/compare.py A.safetensors B.safetensors [--keys PREFIX ...] [--only-common] [--json out.json]
"""
import argparse
import json
import sys

import torch
from safetensors.torch import load_file


def ulp_dist(a, b):
    """Max distance in units in the last place (sign-magnitude mapped to a monotonic integer line)."""
    if a.dtype in (torch.float16, torch.bfloat16):
        ia, ib, mag = a.view(torch.int16).long(), b.view(torch.int16).long(), 0x7FFF
    elif a.dtype == torch.float32:
        ia, ib, mag = a.view(torch.int32).long(), b.view(torch.int32).long(), 0x7FFFFFFF
    else:
        return None
    ia = torch.where(ia < 0, -(ia & mag), ia)
    ib = torch.where(ib < 0, -(ib & mag), ib)
    return (ia - ib).abs().max().item()


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'status': 'MISMATCH-META', 'a': f'{tuple(a.shape)} {a.dtype}', 'b': f'{tuple(b.shape)} {b.dtype}'}
    ab = a.contiguous().view(-1).view(torch.uint8) if a.numel() else a
    bb = b.contiguous().view(-1).view(torch.uint8) if b.numel() else b
    if torch.equal(ab, bb):
        return {'status': 'IDENTICAL'}
    fa, fb = a.double().reshape(-1), b.double().reshape(-1)
    ne = (a.reshape(-1) != b.reshape(-1))
    if a.is_floating_point():
        ne = ne & ~(torch.isnan(a.reshape(-1)) & torch.isnan(b.reshape(-1)))
    n = int(ne.sum())
    first = int(ne.nonzero()[0]) if n else -1
    diff = (fa - fb).abs()
    finite = torch.isfinite(diff)
    res = {'status': 'DIFF', 'n_diff': n, 'numel': a.numel(), 'first': first,
           'first_a': fa[first].item() if n else None, 'first_b': fb[first].item() if n else None,
           'max_abs': diff[finite].max().item() if finite.any() else float('nan'),
           'mean_abs': diff[finite].mean().item() if finite.any() else float('nan'),
           'ref_rms': fa.pow(2).mean().sqrt().item(),
           'max_rel': (diff / fa.abs().clamp_min(1e-6))[finite].max().item() if finite.any() else float('nan')}
    if a.is_floating_point():
        res['max_ulp'] = ulp_dist(a.reshape(-1), b.reshape(-1))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--keys', nargs='*', default=None)
    p.add_argument('--skip', nargs='*', default=[])
    p.add_argument('--json', default=None)
    p.add_argument('--allow-diff', action='store_true')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    A, B = load_file(args.a), load_file(args.b)
    keys = sorted(set(A) & set(B))
    if args.keys:
        keys = [k for k in keys if any(k.startswith(x) for x in args.keys)]
    keys = [k for k in keys if not any(k.startswith(x) for x in args.skip)]
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    out, ndiff = {}, 0
    for k in keys:
        r = compare(A[k], B[k])
        out[k] = r
        if r['status'] != 'IDENTICAL':
            ndiff += 1
        if not args.quiet or r['status'] != 'IDENTICAL':
            extra = '' if r['status'] == 'IDENTICAL' else ' ' + json.dumps({x: y for x, y in r.items() if x != 'status'})
            print(f'{r["status"]:14s} {k}{extra}')
    print(f'# {len(keys)} compared, {len(keys) - ndiff} identical, {ndiff} differ; '
          f'only in A: {len(only_a)}, only in B: {len(only_b)}')
    if only_a and not args.quiet:
        print('# only in A:', only_a[:20])
    if only_b and not args.quiet:
        print('# only in B:', only_b[:20])
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)
    sys.exit(0 if (ndiff == 0 or args.allow_diff) else 1)


if __name__ == '__main__':
    main()

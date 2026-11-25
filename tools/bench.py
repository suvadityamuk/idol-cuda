"""Interleaved same-session benchmark of every build (tools/bench_worker.py + idol_native --serve).

All workers are loaded first (load time recorded), then per config: one warm-up request per worker (its time is
reported separately: compile / autotune / cuDNN plan building), then --reps rounds; within a round every worker runs
one request, in an order that rotates and reverses from round to round. Request time = image in -> Gaussians
(device-synced, as idol-sdk's pipeline call); save = PLY + input PNG + camera JSON. Raw JSON lines go to --out.

python tools/bench.py --native /work/build/idol_native --reps 5 --out logs/bench.jsonl
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--native', required=True)
p.add_argument('--sdk', default='/work/idol-sdk')
p.add_argument('--reps', type=int, default=5)
p.add_argument('--rows', default='ref_compiled,ref_eager,native_parity,py_parity,py_fast,native_fast')
p.add_argument('--configs', default='robot_262k,avocado_262k,robot_1m,avocado_1m')
p.add_argument('--outdir', default='/work/bench_out')
p.add_argument('--out', required=True)
a = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {'robot_262k': ('fig-robot-cute', 'synthetic', 262144), 'avocado_262k': ('food-avocado', 'real', 262144),
       'robot_1m': ('fig-robot-cute', 'synthetic', 1048576), 'avocado_1m': ('food-avocado', 'real', 1048576)}
log = open(a.out, 'a')


def emit(rec):
    log.write(json.dumps(rec) + '\n')
    log.flush()
    print(json.dumps(rec), flush=True)


class Worker:
    def __init__(self, row):
        self.row = row
        if row.startswith('native_'):
            cmd = [a.native, '--serve', '--build', row.split('_')[1]]
            if row == 'native_fast' and os.environ.get('FA3_SO'):
                cmd += ['--fa3', os.environ['FA3_SO']]
        else:
            cmd = [sys.executable, os.path.join(HERE, 'bench_worker.py'), '--impl', row.replace('_kb', '')]
        env = dict(os.environ)
        if row.endswith('_kb'):  # kernels from the kernel-builder output instead of the dev build
            env['IDOL_KERNEL_LOCAL_PATH'] = os.environ['KB_DIR']
        else:
            env.pop('IDOL_KERNEL_LOCAL_PATH', None)
        t = time.perf_counter()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.wait_ready()
        self.load_s = time.perf_counter() - t

    def wait_ready(self):
        res = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f'{self.row} exited')
            if line.startswith('RESULT '):
                res = json.loads(line[7:])
            if line.strip() == 'READY':
                return res

    def request(self, cfg, tag):
        img, mode, ng = CFG[cfg]
        name = f'{cfg}_{self.row}_{tag}'
        od = os.path.join(a.outdir, self.row)
        self.p.stdin.write(f'{a.sdk}/demo/examples/{img}.webp {mode} {ng} {name} {od}\n')
        self.p.stdin.flush()
        r = self.wait_ready()
        if 'save_s' not in r:
            r['save_s'] = r.get('stages', {}).get('save', None)
        with open(os.path.join(od, f'{name}.ply'), 'rb') as f:
            r['ply_sha256'] = hashlib.sha256(f.read()).hexdigest()
        return r


rows = a.rows.split(',')
env = {'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()}
import torch  # noqa: E402
env.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
           commit=open(os.path.join(HERE, '..', '.source_commit')).read().strip() if os.path.exists(
               os.path.join(HERE, '..', '.source_commit')) else None)
emit({'kind': 'env', **env})
workers = {}
for r in rows:
    try:
        workers[r] = Worker(r)
        emit({'kind': 'load', 'row': r, 'load_s': workers[r].load_s})
    except Exception as e:
        emit({'kind': 'error', 'row': r, 'stage': 'load', 'error': repr(e)})
rows = [r for r in rows if r in workers]


def safe_request(r, cfg, tag):
    try:
        return workers[r].request(cfg, tag)
    except Exception as e:  # a dead worker is recorded and dropped, the other rows continue
        emit({'kind': 'error', 'row': r, 'config': cfg, 'tag': tag, 'error': repr(e)})
        rows.remove(r)
        return None
for cfg in a.configs.split(','):
    for r in list(rows):
        res = safe_request(r, cfg, 'warmup')
        if res is not None:
            emit({'kind': 'warmup', 'row': r, 'config': cfg, **res})
    for rep in range(a.reps):
        order = rows[rep % len(rows):] + rows[:rep % len(rows)]
        if rep % 2:
            order = order[::-1]
        for r in order:
            if r not in rows:
                continue
            res = safe_request(r, cfg, f'r{rep}')
            if res is not None:
                emit({'kind': 'timed', 'row': r, 'config': cfg, 'rep': rep, 'order': order.index(r), **res})
for w in workers.values():
    try:
        w.p.stdin.close()
        w.p.wait(timeout=60)
    except Exception:
        w.p.kill()

"""Instrumented runner for the official Idol pipeline (idol-sdk @ 8db7003). Does not modify the reference: it wraps
torch RNG functions and registers forward hooks from outside.

Configurations (ATTN_BACKEND is read at import, so one configuration per process):
  --attn sdpa|flash_attn   -> ATTN_BACKEND
  --optimize / --no_optimize  (the shipped default on CUDA is optimize=True)

Modes:
  --dump OUT.safetensors   record every RNG draw and stage/first-block checkpoints, write outputs (ply/png/json)
  --replay RNG.safetensors replace every RNG draw with the recorded values (same order)
  --time N                 steady-state timing: 1 warm-up request per (image, num_gaussians), then N timed requests

Usage: python tools/ref_run.py --image demo/examples/fig-robot-cute.webp --mode synthetic --num_gaussians 262144 \
         --attn sdpa --no_optimize --dump /scratch/ref/robot_262k_eager_sdpa.safetensors
"""
import argparse
import hashlib
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--image', action='append', required=True)
p.add_argument('--mode', action='append', required=True)
p.add_argument('--num_gaussians', type=int, action='append', default=None)
p.add_argument('--attn', choices=['sdpa', 'flash_attn'], default='sdpa')
p.add_argument('--no_optimize', dest='optimize', action='store_false')
p.add_argument('--dump', default=None)
p.add_argument('--dump_steps', action='store_true', help='also record every denoiser call (inputs and outputs)')
p.add_argument('--replay', default=None)
p.add_argument('--time', type=int, default=0)
p.add_argument('--warmup', type=int, default=1)
p.add_argument('--weights', default='/w/idol')
p.add_argument('--dinov3', default='/w/dinov3')
p.add_argument('--flux2', default='/w/flux2')
p.add_argument('--out', default=None, help='directory for ply/png/json outputs')
p.add_argument('--json', default=None, help='write timing results here')
p.add_argument('--steps', type=int, default=50)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--deterministic', action='store_true', help='torch.backends.cudnn.deterministic=True (parity pin)')
p.add_argument('--save_timed', default=None, help='also write the outputs of timed requests here (times save_ply)')
args = p.parse_args()
if args.num_gaussians is None:
    args.num_gaussians = [262144]
os.environ['ATTN_BACKEND'] = args.attn

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import idol.pipeline as ip  # noqa: E402
from idol import IdolPipeline  # noqa: E402
from idol.models import layers  # noqa: E402

assert layers.ATTN_BACKEND == args.attn
if args.deterministic:
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------- RNG recording / replay
RNG_LOG = []          # (name, tensor on cpu)
REPLAY = None
REPLAY_POS = [0]
RECORD_RNG = [False]
for _name in ('randn', 'rand', 'rand_like', 'randn_like'):
    _orig = getattr(torch, _name)

    def _make(orig, name):
        def wrapped(*a, **k):
            out = orig(*a, **k)
            if REPLAY is not None:
                key = f'rng.{REPLAY_POS[0]:03d}.{name}'
                ref = REPLAY[key]
                assert ref.shape == out.shape and ref.dtype == out.dtype, (key, ref.shape, out.shape)
                out.copy_(ref.to(out.device))
                REPLAY_POS[0] += 1
            if RECORD_RNG[0]:
                RNG_LOG.append((name, out.detach().clone().cpu()))
            return out
        return wrapped
    setattr(torch, _name, _make(_orig, _name))

# ---------------------------------------------------------------- pipeline
t0 = time.perf_counter()
pipe = IdolPipeline(args.weights, device='cuda', optimize=args.optimize,
                    image_encoders={'dinov3': args.dinov3, 'flux2_vae': args.flux2})
torch.cuda.synchronize()
load_s = time.perf_counter() - t0

ENV = {
    'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
    'gpu': torch.cuda.get_device_name(0), 'cc': list(torch.cuda.get_device_capability(0)),
    'cudnn.benchmark': torch.backends.cudnn.benchmark, 'cudnn.deterministic': torch.backends.cudnn.deterministic,
    'cudnn.allow_tf32': torch.backends.cudnn.allow_tf32, 'matmul.allow_tf32': torch.backends.cuda.matmul.allow_tf32,
    'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    'attn': args.attn, 'optimize': pipe.optimize, 'cudnn_attention': pipe.cudnn_attention,
    'load_s': load_s, 'nproc': os.cpu_count(), 'deterministic': args.deterministic,
    'driver': os.popen('nvidia-smi --query-gpu=driver_version --format=csv,noheader').read().strip(),
}
print(json.dumps(ENV), flush=True)

# ---------------------------------------------------------------- stage timers
STAGES = {}


def timed(name, fn):
    def wrapped(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        STAGES[name] = STAGES.get(name, 0.0) + time.perf_counter() - t
        return out
    return wrapped


pipe.preprocess = timed('preprocess', pipe.preprocess)
_enc = pipe.image_encoder
pipe.image_encoder = timed('encoders', _enc.__call__)
ip.sample_flow = timed('sampling', ip.sample_flow)
_oct = ip.OctreeProbabilityFixedlenDecoder.sample
ip.OctreeProbabilityFixedlenDecoder.sample = staticmethod(timed('octree', _oct))
pipe.decoder_gs.forward = timed('gs_decoder', pipe.decoder_gs.forward)

# ---------------------------------------------------------------- checkpoint hooks
CKPT = {}
RECORD = [False]
CALLS = {}


def _store(key, t):
    if isinstance(t, torch.Tensor):
        CKPT[key] = t.detach().clone().cpu().contiguous()
    elif hasattr(t, 'items'):  # TensorDict / dict
        for k, v in t.items():
            _store(f'{key}.{k}', v)
    elif isinstance(t, (tuple, list)):
        for i, v in enumerate(t):
            _store(f'{key}.{i}', v)


def hook(name, max_calls, with_input=False):
    def fn(module, inputs, kwargs, output):
        if not RECORD[0]:
            return
        n = CALLS.get(name, 0)
        CALLS[name] = n + 1
        if n >= max_calls:
            return
        if with_input:
            ins = list(inputs)
            if name == 'denoiser' and n >= 2:
                ins = ins[:2]  # features are the same on every call: keep them for calls 0 (cond) and 1 (uncond)
            _store(f'{name}#{n}.in', ins)
            for k, v in kwargs.items():
                _store(f'{name}#{n}.in.{k}', v)
        _store(f'{name}#{n}.out', output)
    return fn


def _first_layer(model):
    import re
    for name, m in model.named_modules():
        if re.search(r'(^|\.)layers?\.0$', name):
            return m
    raise RuntimeError('no first layer found')


def add_hooks():
    # BiRefNet.remove_background calls self.forward(t) directly (no module hooks): wrap forward instead
    _bf = pipe.background_remover.forward

    def biref_forward(t):
        out = _bf(t)
        if RECORD[0] and CALLS.get('birefnet', 0) == 0:
            CALLS['birefnet'] = 1
            _store('birefnet#0.in.0', t)
            _store('birefnet#0.out', out)
        return out
    pipe.background_remover.forward = biref_forward
    big = 1000 if args.dump_steps else 0
    targets = [
        (pipe.background_remover.bb.patch_embed, 'birefnet.bb.patch_embed', 1, False),
        (pipe.background_remover.bb.layers[0].blocks[0], 'birefnet.bb.layers0.blocks0', 1, False),
        (pipe.background_remover.bb.layers[2], 'birefnet.bb.layers2', 1, False),
        (pipe.background_remover.squeeze_module, 'birefnet.squeeze', 1, False),
        (pipe.background_remover.decoder.decoder_block4.dec_att.aspp_deforms[0], 'birefnet.dec4.aspp_deform0', 1,
         True),
        (pipe.background_remover.decoder, 'birefnet.decoder', 1, False),
        (_enc.dinov3, 'dinov3', 1, True),
        (_enc.dinov3.embeddings, 'dinov3.embeddings', 1, False),
        (_first_layer(_enc.dinov3), 'dinov3.layer0', 1, False),
        (_enc.flux2_vae.encoder, 'vae.encoder', 1, True),
        (pipe.denoiser, 'denoiser', max(big, 2), True),
        (pipe.denoiser.t_embedder, 'denoiser.t_embedder', 1, True),
        (pipe.denoiser.adaLN_modulation, 'denoiser.adaLN', 1, False),
        (pipe.denoiser.input_layer, 'denoiser.input_layer', 1, False),
        (pipe.denoiser.noise_repo_layers[0], 'denoiser.noise_repo0', 1, True),
        (pipe.denoiser.noise_refiner[0], 'denoiser.noise_refiner0', 1, False),
        (pipe.denoiser.noise_refiner[0].attn.qkv, 'denoiser.noise_refiner0.qkv', 1, False),
        (pipe.denoiser.noise_refiner[0].attn, 'denoiser.noise_refiner0.attn', 1, False),
        (pipe.denoiser.noise_refiner[0].mlp, 'denoiser.noise_refiner0.mlp', 1, True),
        (pipe.denoiser.context_repo_layers[0], 'denoiser.context_repo0', 1, False),
        (pipe.denoiser.context_refiner[0], 'denoiser.context_refiner0', 1, True),
        (pipe.denoiser.context_refiner[1], 'denoiser.context_refiner1', 1, False),
        (pipe.denoiser.cam_refiner, 'denoiser.cam_refiner', 1, True),
        (pipe.denoiser.repo_layers[0], 'denoiser.repo0', 1, True),
        (pipe.denoiser.blocks[0], 'denoiser.blocks0', 1, True),
        (pipe.denoiser.blocks[0].attn, 'denoiser.blocks0.attn', 1, False),
        (pipe.denoiser.blocks[0].mlp, 'denoiser.blocks0.mlp', 1, False),
        (pipe.denoiser.blocks[23], 'denoiser.blocks23', 1, False),
        (pipe.decoder, 'octree', 16, True),
        (pipe.decoder.blocks[0], 'octree.blocks0', 1, True),
        (pipe.decoder.blocks[0].cross_attn, 'octree.blocks0.cross_attn', 1, False),
        (pipe.decoder_gs, 'gsdec', 1, True),
        (pipe.decoder_gs.blocks[0], 'gsdec.blocks0', 1, True),
        (pipe.decoder_gs.blocks[0].self_attn, 'gsdec.blocks0.self_attn', 1, False),
        (pipe.decoder_gs.blocks[0].cross_attn, 'gsdec.blocks0.cross_attn', 1, False),
        (pipe.decoder_gs.blocks[15], 'gsdec.blocks15', 1, False),
    ]
    if args.optimize:  # internals of compiled modules are not observable; keep stage boundaries only
        targets = [t for t in targets if t[1].count('.') == 0 or t[1] in ('dinov3.embeddings', 'vae.encoder')]
    for module, name, max_calls, with_input in targets:
        module.register_forward_hook(hook(name, max_calls, with_input), with_kwargs=True)


if args.dump:
    add_hooks()
if args.replay:
    REPLAY = load_file(args.replay)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_once(image, mode, ng, outdir=None, tag=''):
    STAGES.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    out = pipe(image, mode=mode, seed=args.seed, steps=args.steps, num_gaussians=ng)
    torch.cuda.synchronize()
    request_s = time.perf_counter() - t
    res = {'request_s': request_s, 'stages': dict(STAGES),
           'peak_alloc_gb': torch.cuda.max_memory_allocated() / 2**30,
           'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2**30}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        t = time.perf_counter()
        ply = os.path.join(outdir, f'{tag}.ply')
        out.gaussian.save_ply(ply)
        res['save_ply_s'] = time.perf_counter() - t
        out.preprocessed_image.save(os.path.join(outdir, f'{tag}_input.png'))
        with open(os.path.join(outdir, f'{tag}_camera.json'), 'w') as f:
            json.dump(out.camera, f, indent=2)
        res['ply_sha256'] = sha256(ply)
    return out, res


results = {'env': ENV, 'runs': []}
if args.dump:
    assert len(args.image) == 1 and len(args.num_gaussians) == 1
    image, mode, ng = args.image[0], args.mode[0], args.num_gaussians[0]
    RECORD[0] = True
    RECORD_RNG[0] = True
    out, res = run_once(image, mode, ng, outdir=args.out or os.path.dirname(args.dump),
                        tag=os.path.splitext(os.path.basename(args.dump))[0])
    RECORD[0] = False
    RECORD_RNG[0] = False
    g = out.gaussian
    for k in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
        CKPT[f'final.{k}'] = getattr(g, k).detach().cpu().contiguous()
    CKPT['final.preprocessed_image'] = torch.from_numpy(np.array(out.preprocessed_image)).contiguous()
    CKPT['final.camera'] = torch.tensor([out.camera[k] for k in ('yaw', 'pitch', 'radius', 'fov')],
                                        dtype=torch.float64)
    for i, (name, t) in enumerate(RNG_LOG):
        CKPT[f'rng.{i:03d}.{name}'] = t.contiguous()
    save_file(CKPT, args.dump, metadata={'env': json.dumps(ENV), 'run': json.dumps(res)})
    print(json.dumps(res), flush=True)
    print(f'saved {len(CKPT)} tensors ({len(RNG_LOG)} rng draws) to {args.dump}', flush=True)
    if REPLAY is not None:
        print('replayed', REPLAY_POS[0], 'rng draws', flush=True)

if args.time:
    for image, mode in zip(args.image, args.mode):
        for ng in args.num_gaussians:
            for w in range(args.warmup):
                REPLAY_POS[0] = 0
                torch.cuda.synchronize()
                t = time.perf_counter()
                _, res = run_once(image, mode, ng)
                res.update(image=os.path.basename(image), num_gaussians=ng, kind=f'warmup{w}')
                results['runs'].append(res)
                print(json.dumps(res), flush=True)
            for r in range(args.time):
                REPLAY_POS[0] = 0
                _, res = run_once(image, mode, ng, outdir=args.save_timed, tag=f'timed{r}')
                res.update(image=os.path.basename(image), num_gaussians=ng, kind='timed', rep=r)
                results['runs'].append(res)
                print(json.dumps(res), flush=True)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)

"""Kernel profile of the fast sampler (3 steps after warm-up) + attention micro-benchmark (cuDNN, ATen flash, and
FlashAttention-3 from the Hub's kernels-community/flash-attn3 when it loads) at the sampler's shapes.

python tools/profile_fast.py --dump /scratch/ref/robot_262k_eager_sdpa_a.safetensors
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--dump', required=True)
p.add_argument('--top', type=int, default=30)
a = p.parse_args()

from idol import IdolPipeline  # noqa: E402
from fast.model import FastDenoiser  # noqa: E402

D = load_file(a.dump)
dev = torch.device('cuda')
pipe = IdolPipeline('/w/idol', device='cuda', optimize=False,
                    image_encoders={'dinov3': '/w/dinov3', 'flux2_vae': '/w/flux2'})
cond = TensorDict({'feature1': D['denoiser#0.in.2.feature1'].to(dev), 'feature2': D['denoiser#0.in.2.feature2'].to(dev)},
                  batch_size=1)
f = FastDenoiser(pipe.denoiser).prepare(cond)
z, cam, t = D['denoiser#2.in.0.latent'].to(dev), D['denoiser#2.in.0.camera'].to(dev), D['denoiser#2.in.1'].to(dev)
for _ in range(3):
    f.velocity(z, cam, t, 3.0)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        f.velocity(z, cam, t, 3.0)
    torch.cuda.synchronize()
ka = prof.key_averages()
tot = sum(e.self_device_time_total for e in ka)
print(f'fast step: {tot / 3e3:.2f} ms GPU per step')
for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:a.top]:
    if e.self_device_time_total > 0:
        print(f'{e.self_device_time_total / 3e3:8.3f} ms/step {100 * e.self_device_time_total / tot:5.1f}% '
              f'x{e.count // 3:4d}  {e.key[:120]}')


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n


H, Dh = 16, 64
for L, S in ((12294, 12294), (8194, 8193)):
    q = torch.randn(1, L, H, Dh, device=dev, dtype=torch.float16)
    k = torch.randn(1, S, H, Dh, device=dev, dtype=torch.float16)
    v = torch.randn(1, S, H, Dh, device=dev, dtype=torch.float16)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    fl = 4 * L * S * H * Dh
    res = {}
    res['cudnn'] = bench(lambda: torch.ops.aten._scaled_dot_product_cudnn_attention(qt, kt, vt, None, True, 0.0,
                                                                                     False, False))
    res['flash_aten'] = bench(lambda: torch.ops.aten._scaled_dot_product_flash_attention(qt, kt, vt))
    try:
        from kernels import get_kernel
        os.environ['HF_HUB_OFFLINE'] = '0'
        fa3 = get_kernel('kernels-community/flash-attn3', version=int(os.environ.get('FA3_VERSION', '1')))
        res['fa3_hub'] = bench(lambda: fa3.flash_attn_func(q, k, v))
        o3 = fa3.flash_attn_func(q, k, v)
        o3 = o3[0] if isinstance(o3, (tuple, list)) else o3
        oc = torch.ops.aten._scaled_dot_product_cudnn_attention(qt, kt, vt, None, False, 0.0, False, False)[0]
        print('fa3 vs cudnn max abs diff', (o3.float() - oc.transpose(1, 2).float()).abs().max().item())
    except Exception as e:
        print('fa3 unavailable:', type(e).__name__, str(e)[:200])
    print(f'attention L={L} S={S}: ' + '  '.join(f'{k} {v:.3f} ms ({fl / v / 1e9:.0f} TFLOP/s)' for k, v in res.items()))

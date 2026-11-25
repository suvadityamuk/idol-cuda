"""One-off preparation (Python): export the three third-party image networks of the reference pipeline as TorchScript
traces, so the native C++ binary can run them through libtorch's interpreter (no Python in the inference process).

  birefnet.pt : BiRefNet.forward(t: fp16 [1,3,1024,1024] normalised)       -> alpha fp16 [1,1,1024,1024]
  dinov3.pt   : ImageEncoder._dinov3_tokens(images fp32 [1,3,1024,1024])    -> feature1 fp32 [1,4101,1280]
  vae_enc.pt  : vae.quant_conv(vae.encoder(images.to(bf16) * 2 - 1))        -> moments bf16 [1,64,128,128]
                (DiagonalGaussianDistribution + sample + patchify + batch-norm are ported by hand in C++)

The traces record the exact ATen op sequence the eager reference executes (same kernels, same dtypes). The native
binary runs them with the JIT graph optimiser and profiling executor disabled, so no fusion changes the arithmetic.
`--verify DUMP` reruns the traces here (optimiser off) on a stage-1 dump's inputs and byte-compares the outputs.

python tools/export_encoders.py --out /scratch/ts [--verify /work/ref/robot_262k_eager_sdpa_a.safetensors]
"""
import argparse
import os

os.environ.setdefault('ATTN_BACKEND', 'sdpa')
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from idol import IdolPipeline  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--out', default='/scratch/ts')
p.add_argument('--verify', nargs='*', default=[])
p.add_argument('--weights', default='/w/idol')
p.add_argument('--dinov3', default='/w/dinov3')
p.add_argument('--flux2', default='/w/flux2')
p.add_argument('--skip-export', action='store_true')
p.add_argument('--images', nargs='*', default=[], help='source image of each --verify dump')
args = p.parse_args()

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)


class Dino(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.dinov3 = enc.dinov3

    def forward(self, images):
        return self.enc._dinov3_tokens(images)


class VaeEnc(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, images):
        return self.quant_conv(self.encoder(images.to(torch.bfloat16) * 2 - 1))


os.makedirs(args.out, exist_ok=True)
pipe = IdolPipeline(args.weights, device='cuda', optimize=False,
                    image_encoders={'dinov3': args.dinov3, 'flux2_vae': args.flux2})
mods = {
    'birefnet': (pipe.background_remover, torch.randn(1, 3, 1024, 1024, device='cuda').half()),
    'dinov3': (Dino(pipe.image_encoder), torch.rand(1, 3, 1024, 1024, device='cuda')),
    'vae_enc': (VaeEnc(pipe.image_encoder.flux2_vae), torch.rand(1, 3, 1024, 1024, device='cuda')),
}
if not args.skip_export:
    for name, (m, x) in mods.items():
        with torch.no_grad():
            tr = torch.jit.trace(m.eval(), (x,), check_trace=False)
        tr.save(os.path.join(args.out, f'{name}.pt'))
        print('saved', name, flush=True)

# batch-norm statistics of the VAE latent, used by the hand port
vae = pipe.image_encoder.flux2_vae
from safetensors.torch import save_file  # noqa: E402
save_file({'running_mean': vae.bn.running_mean.contiguous(), 'running_var': vae.bn.running_var.contiguous()},
          os.path.join(args.out, 'vae_bn.safetensors'), metadata={'eps': repr(float(vae.config.batch_norm_eps))})
print('vae bn eps', vae.config.batch_norm_eps, vae.bn.running_mean.dtype, flush=True)

ok = True
for D_idx, dump in enumerate(args.verify):
    D = load_file(dump)
    ts = {n: torch.jit.load(os.path.join(args.out, f'{n}.pt'), map_location='cuda') for n in mods}
    with torch.no_grad():
        if 'birefnet#0.in.0' in D:
            t_in = D['birefnet#0.in.0'].cuda()
        else:  # rebuild BiRefNet.remove_background's input from the image, exactly as the reference does
            import numpy as np
            from PIL import Image
            import torch.nn.functional as F2
            im = Image.open(args.images[D_idx])
            w, h = im.size
            sc = 1024 / min(w, h)
            im = im.resize((max(1, int(round(w * sc))), max(1, int(round(h * sc)))), Image.LANCZOS).convert('RGB')
            arr = np.array(im, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = F2.interpolate(t, size=(1024, 1024), mode='bilinear', align_corners=True)
            mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            t_in = ((t - mean) / std).to(device='cuda', dtype=torch.float16)
        alpha = ts['birefnet'](t_in)
        ref_alpha = D['birefnet#0.out'] if 'birefnet#0.out' in D else torch.sigmoid(D['birefnet.decoder#0.out'].cuda()).cpu()
        same = torch.equal(alpha.cpu(), ref_alpha)
        print(dump, 'birefnet alpha identical:', same, flush=True)
        ok &= same
        img = torch.from_numpy(D['final.preprocessed_image'].numpy()).permute(2, 0, 1).float().div(255.0)[None].cuda()
        f1 = ts['dinov3'](img)
        same = torch.equal(f1.cpu(), D['denoiser#0.in.2.feature1'])
        print(dump, 'dinov3 feature1 identical:', same, flush=True)
        ok &= same
        mom = ts['vae_enc'](img)
        mean, logvar = torch.chunk(mom, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        lat = mean + std * D['rng.000.randn'].cuda()
        b, c, h, w = lat.shape
        lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        mu = vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        sd = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat.device, lat.dtype)
        tok = ((lat - mu) / sd).float().flatten(2).transpose(1, 2).contiguous()
        f2 = torch.cat([torch.zeros(b, 5, tok.shape[2], device=tok.device), tok], dim=1)
        same = torch.equal(f2.cpu(), D['denoiser#0.in.2.feature2'])
        print(dump, 'vae feature2 identical:', same, flush=True)
        ok &= same
print('VERIFY', 'PASS' if ok else 'FAIL')

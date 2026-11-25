import math

import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark

H, D, C = 16, 64, 1024


class IdolCudaBenchmark(Benchmark):
    """Shapes of the Idol denoiser on CUDA: 12,294 rows (8,192 latent + 4,101 image + 1 camera tokens), width 1024,
    16 heads of 64, MLP 4096. Outputs are kept O(1) so the runner's allclose(atol=1e-2) check is meaningful."""

    seed = 0
    rows = 12294

    def setup_add_ln_mod(self):
        R = self.rows
        self.x = torch.randn(R, C, device=self.device).half()
        self.h = torch.randn(R, C, device=self.device).half()
        mod = (0.1 * torch.randn(1, 6 * C, device=self.device)).half()
        self.shift, self.scale, self.gate = mod[:, :C], mod[:, C:2 * C], mod[:, 2 * C:3 * C]

    def benchmark_add_ln_mod(self):
        _, self.out = self.kernel.add_ln_mod(self.x, h=self.h, gate=self.gate, scale=self.scale, shift=self.shift)

    def verify_add_ln_mod(self) -> torch.Tensor:
        s = self.x + self.h * self.gate
        return F.layer_norm(s.float(), (C,), eps=1e-6).half() * (1 + self.scale) + self.shift

    def setup_rope_rms_qk(self):
        R = self.rows
        self.qkv = torch.randn(R, 3 * H * D, device=self.device).half()
        self.delta = 0.3 * torch.randn(R, 3 * H, device=self.device)
        self.fl = [torch.linspace(1.0, 16.0, n, device=self.device) for n in (10, 10, 12)]
        self.freqs = torch.cat(self.fl)
        self.g = torch.ones(H, D, device=self.device)

    def benchmark_rope_rms_qk(self):
        self.out, _ = self.kernel.rope_rms_qk(self.qkv, self.delta, self.freqs, self.g, self.g)
        self.out = self.out / 8

    def verify_rope_rms_qk(self) -> torch.Tensor:
        R = self.rows
        d = self.delta.view(R, H, 3)
        ang = torch.cat([(d[..., a, None] * f.tanh() + d[..., a, None] * (f - f.tanh())) * math.pi
                         for a, f in enumerate(self.fl)], -1)
        x0, x1 = self.qkv.view(R, 3, H, D)[:, 0].float().reshape(R, H, -1, 2).unbind(-1)
        r = torch.stack([x0 * ang.cos() - x1 * ang.sin(), x0 * ang.sin() + x1 * ang.cos()], -1).flatten(-2).half()
        return (F.normalize(r.float(), dim=-1) * self.g * 8.0).half() / 8

    def setup_linear_gelu(self):
        R = self.rows
        self.x = torch.randn(R, C, device=self.device).half()
        self.w = (torch.randn(4 * C, C, device=self.device) / math.sqrt(C)).half()
        self.b = (0.1 * torch.randn(4 * C, device=self.device)).half()

    def benchmark_linear_gelu(self):
        self.out = self.kernel.linear_gelu(self.x, self.w, self.b)

    def verify_linear_gelu(self) -> torch.Tensor:
        return F.gelu(F.linear(self.x, self.w, self.b), approximate="tanh")

    def setup_merge_extra_key(self):
        L = 8194
        self.q = torch.randn(1, L, H, D, device=self.device).half()
        k = torch.randn(1, L - 1, H, D, device=self.device).half()
        v = torch.randn(1, L - 1, H, D, device=self.device).half()
        self.ke = torch.randn(1, H, D, device=self.device).half()
        self.ve = torch.randn(1, H, D, device=self.device).half()
        self.kt, self.vt = k.transpose(1, 2), v.transpose(1, 2)
        o, lse = torch.ops.aten._scaled_dot_product_flash_attention(self.q.transpose(1, 2), self.kt, self.vt)[:2]
        self.o, self.lse = o.transpose(1, 2).contiguous(), lse.contiguous()

    def benchmark_merge_extra_key(self):
        self.out = self.kernel.merge_extra_key(self.o, self.lse, self.q, self.ke, self.ve, 1 / 8, math.log(4101))

    def verify_merge_extra_key(self) -> torch.Tensor:
        N = 4101
        kf = torch.cat([self.kt, self.ke.unsqueeze(2).expand(1, H, N, D)], 2)
        vf = torch.cat([self.vt, self.ve.unsqueeze(2).expand(1, H, N, D)], 2)
        return F.scaled_dot_product_attention(self.q.transpose(1, 2), kf, vf).transpose(1, 2)

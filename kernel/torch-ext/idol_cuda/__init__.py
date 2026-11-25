"""idol_cuda: fused CUDA kernels for the Idol image-to-3D pipeline.

Every function validates its arguments in C++ (TORCH_CHECK) and allocates its outputs here.
"""
from typing import Optional, Tuple

import torch

from ._ops import ops

__all__ = ['add_ln_mod', 'rope_rms_qk', 'rms_heads', 'merge_extra_key', 'gaussians_ply', 'linear_gelu']


def add_ln_mod(x: torch.Tensor, h: Optional[torch.Tensor] = None, gate: Optional[torch.Tensor] = None,
               weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None,
               scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None,
               rows_per_batch: Optional[int] = None, eps: float = 1e-6, return_sum: bool = True,
               w2: Optional[torch.Tensor] = None, b2: Optional[torch.Tensor] = None, eps2: float = 1e-5):
    """Residual + LayerNorm32 + adaLN modulation over rows of width 1024 (fp16 in, fp16 out).

    s = x + h * gate (fp16 rounding after each op, h and gate optional); y = fp16(LayerNorm_fp32(s) [* weight + bias]);
    with modulation y = fp16(fp16(y * fp16(1 + scale)) + shift). scale/shift/gate are [B, 1024] fp16 rows; row r of x
    uses batch r // rows_per_batch (default: all rows of one batch element of x's leading dimension).
    w2/b2 (fp32 [1024]): also return y2 = LayerNorm_fp32(s, eps2) * w2 + b2 in fp32 (appended to the result).
    Returns y, or (s, y) when h is given and return_sum is True, plus y2 when w2/b2 are given."""
    if rows_per_batch is None:  # [B, L, 1024]: L rows per batch element; [M, 1024]: one batch element
        rows_per_batch = x.shape[1] if x.dim() == 3 else x.numel() // 1024
    rows_per_batch = max(1, int(rows_per_batch))
    y = torch.empty_like(x)
    s = torch.empty_like(x) if (h is not None and return_sum) else None
    y2 = torch.empty(x.shape, device=x.device, dtype=torch.float32) if w2 is not None else None
    ops.add_ln_mod(y, s, x, h, gate, weight, bias, scale, shift, int(rows_per_batch), float(eps), y2, w2, b2,
                   float(eps2))
    res = ((s, y) if s is not None else (y,)) + ((y2,) if y2 is not None else ())
    return res if len(res) > 1 else res[0]


def rope_rms_qk(qkv: torch.Tensor, delta: torch.Tensor, freqs: torch.Tensor, gamma_q: torch.Tensor,
                gamma_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """qkv [M, 3*H*64] fp16, delta [M, 3*H] fp32 (RePo offsets), freqs [32] fp32 (freqs_0, freqs_1, freqs_2), gammas
    [H, 64] fp32 -> q, k [M, H, 64] fp16: RePo rotation, fp16 rounding, then RMSNorm * gamma * 8."""
    M, H = qkv.shape[0], qkv.shape[1] // 192
    q = torch.empty(M, H, 64, device=qkv.device, dtype=torch.float16)
    k = torch.empty_like(q)
    ops.rope_rms_qk(q, k, qkv, delta, freqs, gamma_q, gamma_k)
    return q, k


def rms_heads(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """x [M, H*64] fp16 (unit inner stride) -> fp16(x / max(||x||, 1e-12) * gamma * 8), shape [M, H, 64]."""
    out = torch.empty(x.shape[0], x.shape[1] // 64, 64, device=x.device, dtype=torch.float16)
    ops.rms_heads(out, x, gamma)
    return out


def merge_extra_key(o: torch.Tensor, lse: torch.Tensor, q: torch.Tensor, k_extra: torch.Tensor,
                    v_extra: torch.Tensor, scale: float, log_mult: float) -> torch.Tensor:
    """Attention output o [B, L, H, 64] over some keys, with its log-sum-exp lse [B, H, L] (fp32), plus one more
    key/value pair [B, H, 64] repeated exp(log_mult) times -> the attention output over all keys."""
    out = torch.empty_like(o)
    ops.merge_extra_key(out, o, lse, q, k_extra, v_extra, float(scale), float(log_mult))
    return out


def gaussians_ply(feats: torch.Tensor, points: torch.Tensor, perturb: torch.Tensor, base_offset_scale: float,
                  scaling_offset: float, opacity_offset: float, min_kernel_size: float, lr_rotation: float,
                  perturb_size: float) -> torch.Tensor:
    """Gaussian decoder head [A, 480] + anchors [A, 3] -> the 17 float32 columns of the standard 3DGS PLY, [A*32, 17]."""
    out = torch.empty(feats.shape[0] * 32, 17, device=feats.device, dtype=torch.float32)
    ops.gaussians_ply(out, feats, points, perturb, float(base_offset_scale), float(scaling_offset),
                      float(opacity_offset), float(min_kernel_size), float(lr_rotation), float(perturb_size))
    return out


def linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """GELU_tanh(x @ weight.T + bias) with the activation in the cuBLASLt epilogue (fp16)."""
    out = torch.empty(x.shape[0], weight.shape[0], device=x.device, dtype=x.dtype)
    ops.linear_gelu(out, x, weight, bias)
    return out

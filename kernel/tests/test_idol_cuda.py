"""Tests of the idol_cuda ops against the PyTorch op chains of the reference (idol-sdk 8db7003).

Tolerances: the kernels keep every fp16 rounding point of the reference chain; fp32 reductions (LayerNorm statistics,
L2 norms, dot products) are summed in a different order, which moves a few fp16 outputs by one ulp. Tests therefore
require exact equality for most elements and at most 1 fp16 ulp elsewhere (fraction bounded)."""
import math

import pytest
import torch
import torch.nn.functional as F

C, H, D = 1024, 16, 64


def ulp_diff(a, b):
    """fp16 ulp distance per element."""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return (ia - ib).abs()


def assert_close_half(out, ref, max_ulp=1, max_frac=0.02, scale=None):
    """At most max_ulp fp16 ulps of `scale` (default |ref|; for outputs after a cancelling add, the magnitude of the
    operands) apart, and at most max_frac of the elements not identical."""
    assert out.shape == ref.shape and out.dtype == ref.dtype
    o, r = out.contiguous().float(), ref.contiguous().float()
    mag = (r.abs() if scale is None else scale.contiguous().float().abs()).clamp_min(2.0 ** -14)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 10)
    err = (o - r).abs() / ulp
    excess = err.max().item()
    if excess > max_ulp:
        i = int(err.reshape(-1).argmax())
        print(f"worst element {i} (row {i // o.shape[-1]}): out {o.reshape(-1)[i].item()} ref {r.reshape(-1)[i].item()} "
              f"scale {mag.reshape(-1)[i].item()}; elements over tolerance: {int((err > max_ulp).sum())}")
    assert excess <= max_ulp, f"max error {excess:.2f} ulp"
    frac = float((o != r).float().mean())
    print(f"max error {excess:.2f} ulp, fraction differing {frac:.4f}")
    assert frac <= max_frac, f"fraction differing {frac}"


# ------------------------------------------------------------------------------------------------ add_ln_mod
def mod_scale(y, scale, shift):
    """Magnitude that bounds the error of y = fp16(fp16(n * fp16(1 + scale)) + shift) when the LayerNorm output n is
    off by one ulp (fp32 reduction order): |y| + 2 max|shift| bounds |n * (1 + scale)|; one ulp of n becomes up to 2
    ulps of that magnitude after the multiply (measured: 2 ulps on 1-2 of 12.6M elements)."""
    if scale is None:
        return None
    return y.float().abs() + 2 * shift.float().abs().max()


def ref_add_ln_mod(x, h=None, gate=None, w=None, b=None, scale=None, shift=None, rpb=None, eps=1e-6):
    rows = x.reshape(-1, C)
    s = rows
    nb = 1 if rpb is None else (rows.shape[0] + rpb - 1) // rpb
    rpb = rows.shape[0] if rpb is None else rpb
    bidx = torch.arange(rows.shape[0], device=x.device) // rpb
    if h is not None:
        hh = h.reshape(-1, C)
        if gate is not None:
            hh = hh * gate[bidx]
        s = rows + hh
    y = F.layer_norm(s.float(), (C,), w, b, eps).half()
    if scale is not None:
        y = y * (1 + scale[bidx]) + shift[bidx]
    return s.reshape(x.shape), y.reshape(x.shape)


@pytest.mark.parametrize("rows", [1, 37, 4101, 12294])
@pytest.mark.parametrize("variant", ["plain", "affine", "mod", "gate_mod", "add"])
def test_add_ln_mod(kernel, device, rows, variant):
    g = torch.Generator(device="cpu").manual_seed(rows)
    x = (2 * torch.randn(rows, C, generator=g)).half().to(device)
    h = torch.randn(rows, C, generator=g).half().to(device) if variant in ("gate_mod", "add") else None
    gate = (0.5 * torch.randn(1, C, generator=g)).half().to(device) if variant == "gate_mod" else None
    w = (1 + 0.1 * torch.randn(C, generator=g)).to(device) if variant == "affine" else None
    b = (0.1 * torch.randn(C, generator=g)).to(device) if variant == "affine" else None
    sc = (0.2 * torch.randn(1, C, generator=g)).half().to(device) if variant in ("mod", "gate_mod") else None
    sh = (0.2 * torch.randn(1, C, generator=g)).half().to(device) if sc is not None else None
    out = kernel.add_ln_mod(x, h=h, gate=gate, weight=w, bias=b, scale=sc, shift=sh)
    s_ref, y_ref = ref_add_ln_mod(x, h, gate, w, b, sc, sh)
    if h is not None:
        s, y = out
        assert torch.equal(s, s_ref)  # elementwise, exactly specified
    else:
        y = out
    assert_close_half(y, y_ref, max_ulp=2, scale=mod_scale(y_ref, sc, sh))


def test_add_ln_mod_batches_and_offsets(kernel, device):
    """Two batch elements with their own modulation rows, inputs that are views with a storage offset."""
    g = torch.Generator(device="cpu").manual_seed(0)
    big = torch.randn(3 + 2 * 333, C, generator=g).half().to(device)
    x = big[3:]                                  # storage offset 3 rows (16-byte aligned)
    h = torch.randn(2 * 333, C, generator=g).half().to(device)
    mod = (0.3 * torch.randn(2, 6 * C, generator=g)).half().to(device)
    shift, scale, gate = mod[:, :C], mod[:, C:2 * C], mod[:, 2 * C:3 * C]   # strided rows of a [2, 6C] tensor
    s, y = kernel.add_ln_mod(x, h=h, gate=gate, scale=scale, shift=shift, rows_per_batch=333)
    s_ref, y_ref = ref_add_ln_mod(x, h, gate, None, None, scale, shift, rpb=333)
    assert torch.equal(s, s_ref)
    assert_close_half(y, y_ref, max_ulp=2, scale=mod_scale(y_ref, scale, shift[:1]))
    y3 = kernel.add_ln_mod(x.view(2, 333, C), scale=scale, shift=shift)       # [B, L, C] input
    _, y3_ref = ref_add_ln_mod(x, None, None, None, None, scale, shift, rpb=333)
    assert_close_half(y3.view(-1, C), y3_ref, max_ulp=2, scale=mod_scale(y3_ref, scale, shift[:1]))


def test_add_ln_mod_bad_args(kernel, device):
    x = torch.randn(8, C, device=device).half()
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x.float())                                  # dtype
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(torch.randn(8, 512, device=device).half())  # width
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x.cpu())                                    # device
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x[:, :].t().contiguous().t())               # non-contiguous
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x.view(-1)[1:1 + 7 * C].view(7, C))         # misaligned view
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x, scale=torch.zeros(1, C, device=device).half())  # scale without shift
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x, h=torch.randn(4, C, device=device).half())      # h shape
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x, scale=torch.zeros(1, C, device=device).half(), shift=torch.zeros(1, C, device=device).half(),
                          rows_per_batch=4)                                  # 2 batches addressed, 1 row given


# ------------------------------------------------------------------------------------------------ rope_rms_qk
FREQS = [torch.linspace(1.0, 16.0, n) for n in (10, 10, 12)]


def ref_rope_rms(qkv, delta, freqs, gq, gk):
    """RePo3DRotaryEmbedding (real_valued) + apply_rotary_emb + MultiHeadRMSNorm, as in idol/models/layers.py."""
    M = qkv.shape[0]
    d = delta.view(M, H, 3)

    def clamp_mul(x, f):
        ft = f.tanh()
        return x * ft + x * (f - ft)
    ang = torch.cat([clamp_mul(d[..., a].unsqueeze(-1), f) * torch.pi for a, f in enumerate(freqs)], dim=-1)
    cs = torch.stack([ang.cos(), ang.sin()], dim=-1)
    q, k, _ = qkv.view(M, 3, H, D).unbind(1)

    def rot(x):
        x0, x1 = x.float().reshape(M, H, -1, 2).unbind(-1)
        c, s = cs.unbind(-1)
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2).half()

    def rms(x, g):
        return (F.normalize(x.float(), dim=-1) * g * 8.0).half()
    return rms(rot(q), gq), rms(rot(k), gk)


@pytest.mark.parametrize("M", [1, 37, 8194, 12294])
def test_rope_rms_qk(kernel, device, M):
    g = torch.Generator(device="cpu").manual_seed(M)
    qkv = torch.randn(M, 3 * H * D, generator=g).half().to(device)
    delta = (0.3 * torch.randn(M, 3 * H, generator=g)).to(device)
    freqs = [f.to(device) for f in FREQS]
    gq = (1 + 0.1 * torch.randn(H, D, generator=g)).to(device)
    gk = (1 + 0.1 * torch.randn(H, D, generator=g)).to(device)
    q, k = kernel.rope_rms_qk(qkv, delta, torch.cat(freqs), gq, gk)
    q_ref, k_ref = ref_rope_rms(qkv, delta, freqs, gq, gk)
    # sincosf vs torch's cos/sin (1 fp32 ulp) can flip the fp16 rounding of the rotated value, and the norm then moves
    # the output by up to 2-3 fp16 ulp on a few elements (measured: 2 ulp, 0.04% of elements at 12,294 rows); near-zero
    # components after the rotation's cancellation are judged against the row scale (|y| + 0.05, outputs have norm 8)
    assert_close_half(q, q_ref, max_ulp=3, max_frac=0.03, scale=q_ref.float().abs() + 0.05)
    assert_close_half(k, k_ref, max_ulp=3, max_frac=0.03, scale=k_ref.float().abs() + 0.05)


def test_rope_rms_qk_strided_rows(kernel, device):
    """qkv as a column slice of a wider matrix (row stride != width), delta as a column slice too."""
    M = 129
    big = torch.randn(M, 3 * H * D + 64, device=device).half()
    qkv = big[:, 64:]
    dbig = 0.3 * torch.randn(M, 3 * H + 16, device=device)
    delta = dbig[:, 16:]
    freqs = [f.to(device) for f in FREQS]
    g1 = torch.ones(H, D, device=device)
    q, k = kernel.rope_rms_qk(qkv, delta, torch.cat(freqs), g1, g1)
    q_ref, k_ref = ref_rope_rms(qkv.contiguous(), delta.contiguous(), freqs, g1, g1)
    assert_close_half(q, q_ref, max_ulp=3, max_frac=0.03, scale=q_ref.float().abs() + 0.05)


def test_rope_rms_qk_bad_args(kernel, device):
    qkv = torch.randn(4, 3 * H * D, device=device).half()
    delta = torch.randn(4, 3 * H, device=device)
    f = torch.cat(FREQS).to(device)
    g1 = torch.ones(H, D, device=device)
    with pytest.raises(RuntimeError):
        kernel.rope_rms_qk(qkv.float(), delta, f, g1, g1)
    with pytest.raises(RuntimeError):
        kernel.rope_rms_qk(qkv, delta[:3], f, g1, g1)
    with pytest.raises(RuntimeError):
        kernel.rope_rms_qk(qkv, delta, f[:31], g1, g1)
    with pytest.raises(RuntimeError):
        kernel.rope_rms_qk(qkv[:, :1000], delta, f, g1, g1)


# ------------------------------------------------------------------------------------------------ rms_heads
@pytest.mark.parametrize("M", [1, 333, 8192])
def test_rms_heads(kernel, device, M):
    x = torch.randn(M, 2 * H * D, device=device).half()[:, H * D:]   # k half of a to_kv output (strided rows)
    gamma = (1 + 0.1 * torch.randn(H, D, device=device))
    out = kernel.rms_heads(x, gamma)
    ref = (F.normalize(x.reshape(M, H, D).float(), dim=-1) * gamma * 8.0).half()
    assert_close_half(out, ref)
    with pytest.raises(RuntimeError):
        kernel.rms_heads(x.float(), gamma)
    with pytest.raises(RuntimeError):
        kernel.rms_heads(x, gamma[:8])


# ------------------------------------------------------------------------------------------------ merge_extra_key
@pytest.mark.parametrize("L,N", [(1, 1), (100, 4101), (8194, 4101)])
def test_merge_extra_key(kernel, device, L, N):
    """Attention over [keys..., N copies of one extra key] equals the merge of the attention over the keys with the
    extra key of multiplicity N. Reference: fp32 SDPA over the explicitly repeated keys."""
    torch.manual_seed(L)
    B, S = 1, 513
    q = torch.randn(B, L, H, D, device=device).half()
    k = torch.randn(B, S, H, D, device=device).half()
    v = torch.randn(B, S, H, D, device=device).half()
    ke = torch.randn(B, H, D, device=device).half()
    ve = torch.randn(B, H, D, device=device).half()
    scale = 1 / math.sqrt(D)
    qt, kt, vt = (t.permute(0, 2, 1, 3).float() for t in (q, k, v))
    s = (qt @ kt.transpose(-1, -2)) * scale
    lse = torch.logsumexp(s, -1).contiguous()                       # [B, H, L]
    o = (torch.softmax(s, -1) @ vt).permute(0, 2, 1, 3).half().contiguous()  # [B, L, H, D]
    out = kernel.merge_extra_key(o, lse, q, ke, ve, scale, math.log(N))
    kfull = torch.cat([kt, ke.float().unsqueeze(2).expand(B, H, N, D)], 2)
    vfull = torch.cat([vt, ve.float().unsqueeze(2).expand(B, H, N, D)], 2)
    ref = F.scaled_dot_product_attention(qt, kfull, vfull).permute(0, 2, 1, 3)
    assert (out.float() - ref).abs().max() < 2e-3


def test_merge_extra_key_with_flash_lse(kernel, device):
    """The log-sum-exp returned by the flash / cuDNN SDPA kernels is natural-log and plugs in directly."""
    torch.manual_seed(1)
    B, L, S, N = 1, 1000, 1000, 4101
    q = torch.randn(B, L, H, D, device=device).half()
    k = torch.randn(B, S, H, D, device=device).half()
    v = torch.randn(B, S, H, D, device=device).half()
    ke = torch.randn(B, H, D, device=device).half()
    ve = torch.randn(B, H, D, device=device).half()
    qt, kt, vt = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    kfull = torch.cat([kt, ke.unsqueeze(2).expand(B, H, N, D)], 2).float()
    vfull = torch.cat([vt, ve.unsqueeze(2).expand(B, H, N, D)], 2).float()
    ref = F.scaled_dot_product_attention(qt.float(), kfull, vfull).permute(0, 2, 1, 3)
    o, lse = torch.ops.aten._scaled_dot_product_flash_attention(qt, kt, vt)[:2]
    out = kernel.merge_extra_key(o.permute(0, 2, 1, 3), lse, q, ke, ve, 1 / 8, math.log(N))
    assert (out.float() - ref).abs().max() < 4e-3
    try:
        r = torch.ops.aten._scaled_dot_product_cudnn_attention(qt, kt, vt, None, True, 0.0, False, False)
    except RuntimeError as e:  # cuDNN attention unavailable on this GPU / build
        pytest.skip(f"cuDNN attention unavailable: {e}")
    out2 = kernel.merge_extra_key(r[0].permute(0, 2, 1, 3), r[1].reshape(B, H, L).contiguous(), q, ke, ve, 1 / 8,
                                  math.log(N))
    assert (out2.float() - ref).abs().max() < 4e-3


def test_merge_extra_key_bad_args(kernel, device):
    o = torch.randn(1, 4, H, D, device=device).half()
    lse = torch.randn(1, H, 4, device=device)
    e = torch.randn(1, H, D, device=device).half()
    with pytest.raises(RuntimeError):
        kernel.merge_extra_key(o, lse.half(), o, e, e, 0.125, 1.0)
    with pytest.raises(RuntimeError):
        kernel.merge_extra_key(o, lse[:, :, :3], o, e, e, 0.125, 1.0)
    with pytest.raises(RuntimeError):
        kernel.merge_extra_key(o, lse, o, e[:, :8], e, 0.125, 1.0)


# ------------------------------------------------------------------------------------------------ gaussians_ply
def ref_gaussians_ply(feats, points, perturb, bos):
    """GaussianFixedlenDecoder.to_representation + Gaussian.__init__ + the column transforms of Gaussian.save_ply."""
    import numpy as np
    A = feats.shape[0]
    f = feats.view(1, A, 480)
    sl = lambda a, b, *s: f[:, :, a:b].reshape(1, A, 32, *s)  # noqa: E731
    offset_scale = F.softplus(sl(448, 480, 1) + bos)
    offset = sl(0, 96, 3) * 1.0 + perturb
    offset = torch.tanh(offset) * 0.5 * 1.5 * offset_scale
    raw_xyz = (offset[0] + points[:, None, :]).flatten(0, 1)
    fdc = (sl(96, 192, 1, 3)[0].flatten(0, 1) * 1.0)
    rs = sl(192, 288, 3)[0].flatten(0, 1) * 1.0
    rr = sl(288, 416, 4)[0].flatten(0, 1) * 0.1
    ro = sl(416, 448, 1)[0].flatten(0, 1) * 1.0
    dev = feats.device
    aabb = torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], device=dev)
    x = torch.tensor(0.004)
    so = (x + torch.log(-torch.expm1(-x))).to(dev)
    y = torch.tensor(0.1)
    oo = torch.log(y / (1 - y)).to(dev)
    xyz = raw_xyz * aabb[None, 3:] + aabb[None, :3]
    scaling = F.softplus(rs + so)
    scaling = torch.sqrt(torch.square(scaling) + 0.0009 ** 2)
    rot = F.normalize(rr + torch.tensor([1.0, 0, 0, 0], device=dev)[None])
    op = torch.sigmoid(ro + oo)
    xyz = xyz.cpu().numpy()
    fd = fdc.transpose(1, 2).flatten(1).cpu().numpy()
    opc = op.clamp(1e-6, 1 - 1e-6)
    opl = torch.log(opc / (1 - opc)).cpu().numpy()
    sc = torch.log(scaling.clamp_min(1e-6)).cpu().numpy()
    r = rot.cpu().numpy()
    xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
    s = np.sqrt(0.5)
    bw, bx, by, bz = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    q = np.stack([s * bw - s * bx - 0.0 * by - 0.0 * bz, s * bx + s * bw + 0.0 * bz - 0.0 * by,
                  s * by - s * bz + 0.0 * bw + 0.0 * bx, s * bz + s * by - 0.0 * bx + 0.0 * bw], axis=1)
    cols = np.concatenate([xyz, np.zeros_like(xyz), fd, opl, sc, q], axis=1).astype(np.float32)
    return torch.from_numpy(cols), float(so), float(oo)


@pytest.mark.parametrize("A", [1, 1000])
def test_gaussians_ply(kernel, device, A):
    torch.manual_seed(A)
    feats = torch.randn(A, 480, device=device)
    points = torch.rand(A, 3, device=device)
    perturb = 0.1 * torch.randn(32, 3, device=device)
    bos = -2.0
    ref, so, oo = ref_gaussians_ply(feats, points, perturb, bos)
    out = kernel.gaussians_ply(feats, points, perturb, bos, so, oo, 0.0009, 0.1, 1.5).cpu()
    # transcendental functions (tanh, log1p, exp, log, sqrt) of CUDA libdevice vs PyTorch's: a few fp32 ulp
    torch.testing.assert_close(out, ref, rtol=2e-6, atol=2e-6)
    with pytest.raises(RuntimeError):
        kernel.gaussians_ply(feats[:, :479], points, perturb, bos, so, oo, 0.0009, 0.1, 1.5)
    with pytest.raises(RuntimeError):
        kernel.gaussians_ply(feats.double(), points, perturb, bos, so, oo, 0.0009, 0.1, 1.5)


# ------------------------------------------------------------------------------------------------ linear_gelu
@pytest.mark.parametrize("M", [1, 37, 12294])
def test_linear_gelu(kernel, device, M):
    torch.manual_seed(M)
    x = torch.randn(M, C, device=device).half()
    w = (torch.randn(4 * C, C, device=device) / 32).half()
    b = torch.randn(4 * C, device=device).half()
    out = kernel.linear_gelu(x, w, b)
    ref = F.gelu(F.linear(x, w, b), approximate="tanh")
    # the epilogue applies GELU to the fp32 accumulator before the fp16 rounding (the unfused chain rounds twice)
    torch.testing.assert_close(out, ref, rtol=4e-3, atol=4e-3)
    with pytest.raises(RuntimeError):
        kernel.linear_gelu(x, w[:, :512], b)
    with pytest.raises(RuntimeError):
        kernel.linear_gelu(x.float(), w, b)


@pytest.mark.parametrize("rows", [1, 4101, 20488])
def test_add_ln_mod_second_norm(kernel, device, rows):
    """The fp32 affine LayerNorm (eps 1e-5) of the residual sum, as RePo3DRotaryEmbedding.norm computes it."""
    g = torch.Generator(device="cpu").manual_seed(rows + 1)
    x = (2 * torch.randn(rows, C, generator=g)).half().to(device)
    h = torch.randn(rows, C, generator=g).half().to(device)
    mod = (0.3 * torch.randn(1, 6 * C, generator=g)).half().to(device)
    w2 = (1 + 0.1 * torch.randn(C, generator=g)).to(device)
    b2 = (0.1 * torch.randn(C, generator=g)).to(device)
    s, y, y2 = kernel.add_ln_mod(x, h=h, gate=mod[:, :C], scale=mod[:, C:2 * C], shift=mod[:, 2 * C:3 * C], w2=w2,
                                 b2=b2, eps2=1e-5)
    s_ref, y_ref = ref_add_ln_mod(x, h, mod[:, :C], None, None, mod[:, C:2 * C], mod[:, 2 * C:3 * C])
    assert torch.equal(s, s_ref)
    assert_close_half(y, y_ref, max_ulp=2, scale=mod_scale(y_ref, mod, mod[:, 2 * C:3 * C]))
    y2_ref = F.layer_norm(s_ref.float(), (C,), w2, b2, 1e-5)
    torch.testing.assert_close(y2, y2_ref, rtol=1e-5, atol=1e-5)
    with pytest.raises(RuntimeError):
        kernel.add_ln_mod(x, w2=w2, b2=b2[:10])

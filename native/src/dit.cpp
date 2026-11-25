#include "dit.h"

#include <cmath>

namespace idol {

static std::string idx(const std::string& name, int n) { return name + "#" + std::to_string(n); }

// ---------------------------------------------------------------------------------------------- RePo
RePo::RePo(const Weights& ws, const std::string& p)
    : norm(ws, p + ".norm", 1024, 1e-5, true),
      gate_map(ws, p + ".gate_map", false),
      content_map(ws, p + ".content_map", false),
      final_map(ws, p + ".final_map", false) {
  f0 = W(ws, p + ".freqs_0");
  f1 = W(ws, p + ".freqs_1");
  f2 = W(ws, p + ".freqs_2");
}

static at::Tensor clamp_mul(const at::Tensor& x, const at::Tensor& f) {
  at::Tensor f_t = f.tanh();
  return x * f_t + x * (f - f_t);
}

at::Tensor RePo::operator()(const at::Tensor& hidden) const {
  at::Tensor h = norm(hidden.to(at::kFloat));
  at::Tensor out = final_map(at::silu(gate_map(h)) * content_map(h));
  const int64_t B = out.size(0), L = out.size(1);
  at::Tensor d = out.reshape({B, L, heads, 3});
  at::Tensor a0 = clamp_mul(d.select(-1, 0).unsqueeze(-1), f0) * M_PI;
  at::Tensor a1 = clamp_mul(d.select(-1, 1).unsqueeze(-1), f1) * M_PI;
  at::Tensor a2 = clamp_mul(d.select(-1, 2).unsqueeze(-1), f2) * M_PI;
  at::Tensor ang = at::cat({a0, a1, a2}, -1);
  return at::polar(at::ones_like(ang), ang).to(at::kComplexFloat);
}

// ---------------------------------------------------------------------------------------------- rotary
at::Tensor apply_rotary_emb(const at::Tensor& hs, const at::Tensor& freqs) {
  std::vector<int64_t> shape(hs.sizes().begin(), hs.sizes().end() - 1);
  std::vector<int64_t> s2 = shape;
  s2.push_back(-1);
  s2.push_back(2);
  at::Tensor xr = at::view_as_complex(hs.to(at::kFloat).reshape(s2));
  xr = xr * freqs;
  std::vector<int64_t> s3(xr.sizes().begin(), xr.sizes().end() - 1);
  s3.push_back(-1);
  at::Tensor out = at::view_as_real(xr).reshape(s3);
  return out.to(hs.scalar_type());
}

// ---------------------------------------------------------------------------------------------- attention
RopeAttention::RopeAttention(const Weights& ws, const std::string& p)
    : qkv(ws, p + ".qkv"), out(ws, p + ".out"), qn(ws, p + ".q_norm", 64), kn(ws, p + ".k_norm", 64) {}

at::Tensor RopeAttention::operator()(const at::Tensor& x, const at::Tensor& rope, const std::string& rec) const {
  const int64_t B = x.size(0), L = x.size(1), C = x.size(2);
  at::Tensor qkv_out = qkv(x);
  if (!rec.empty()) REC.put(rec + ".qkv#0.out", qkv_out);
  auto parts = qkv_out.reshape({B, L, 3, heads, head_dim}).unbind(2);
  at::Tensor q = qn(apply_rotary_emb(parts[0], rope));
  at::Tensor k = kn(apply_rotary_emb(parts[1], rope));
  at::Tensor o = sdpa_nlhc(q, k, parts[2]).reshape({B, L, C});
  return out(o);
}

// ---------------------------------------------------------------------------------------------- block
UnifiedBlock::UnifiedBlock(const Weights& ws, const std::string& p, bool mod)
    : modulation(mod),
      norm1(ws, p + ".norm1", 1024, 1e-6, !mod),
      norm2(ws, p + ".norm2", 1024, 1e-6, !mod),
      attn(ws, p + ".attn"),
      mlp(ws, p + ".mlp") {
  if (mod) shift_table = W(ws, p + ".shift_table");
}

at::Tensor UnifiedBlock::operator()(const at::Tensor& x_in, const at::Tensor& mod_in, const at::Tensor& rope,
                                    const std::string& rec) const {
  at::Tensor x = x_in;
  if (!modulation) {
    x = x + attn(norm1(x), rope, rec.empty() ? "" : rec + ".attn");
    return x + mlp(norm2(x));
  }
  at::Tensor mod = mod_in + shift_table.to(mod_in.scalar_type());
  auto c = mod.chunk(6, 1);
  at::Tensor h = norm1(x);
  h = h * (c[1].unsqueeze(1) + 1) + c[0].unsqueeze(1);
  h = attn(h, rope, rec.empty() ? "" : rec + ".attn");
  if (!rec.empty()) REC.put(rec + ".attn#0.out", h);
  x = x + h * c[2].unsqueeze(1);
  h = norm2(x);
  h = h * (c[4].unsqueeze(1) + 1) + c[3].unsqueeze(1);
  if (!rec.empty()) REC.put(rec + ".mlp#0.in.0", h);
  h = mlp(h);
  if (!rec.empty()) REC.put(rec + ".mlp#0.out", h);
  return x + h * c[5].unsqueeze(1);
}

// ---------------------------------------------------------------------------------------------- denoiser
Denoiser::Denoiser(const Weights& ws, at::Device dev)
    : t_embedder(ws, "t_embedder", 10000, false, dev),
      adaLN(ws, "adaLN_modulation.1"),
      input_layer(ws, "input_layer"),
      cond_embedder(ws, "cond_embedder"),
      cond_embedder2(ws, "cond_embedder2"),
      cam0(ws, "cam_refiner.mlp.0"),
      cam2(ws, "cam_refiner.mlp.2"),
      out_layer(ws, "out_layer"),
      cam_out_layer(ws, "cam_out_layer") {
  shift_table = W(ws, "shift_table");
  for (int i = 0; i < 2; i++) {
    noise_repo.emplace_back(ws, "noise_repo_layers." + std::to_string(i));
    context_repo.emplace_back(ws, "context_repo_layers." + std::to_string(i));
    noise_refiner.emplace_back(ws, "noise_refiner." + std::to_string(i), true);
    context_refiner.emplace_back(ws, "context_refiner." + std::to_string(i), false);
  }
  for (int i = 0; i < 24; i++) {
    repo.emplace_back(ws, "repo_layers." + std::to_string(i));
    blocks.emplace_back(ws, "blocks." + std::to_string(i), true);
  }
  // PcdAbsolutePositionEmbedder(1024): freqs built on the CPU in __init__, used on the device
  const int64_t freq_dim = C / 3 / 2;  // 170
  const int64_t max_res = 16;
  at::Tensor f2exp = at::arange(max_res, at::TensorOptions().dtype(at::kFloat));
  const int64_t res_dim = std::max<int64_t>(0, freq_dim - max_res);
  at::Tensor fres = at::arange(res_dim, at::TensorOptions().dtype(at::kFloat));
  fres = at::div(fres, at::Scalar(res_dim));
  fres = at::mul(fres, at::Scalar(max_res));
  at::Tensor freqs = at::pow(at::Scalar(2), at::cat({f2exp, fres}, 0).slice(0, 0, freq_dim)).to(dev);
  at::Tensor pos = W(ws, "pos_pe");  // [1, 8192, 3] on the device
  at::Tensor o = at::outer(pos.reshape({-1}), freqs);
  o = o * 2;
  o = o * M_PI;
  at::Tensor e = at::cat({at::sin(o), at::cos(o)}, -1).reshape({1, q_len, -1});
  if (e.size(-1) < C) e = at::cat({e, at::zeros({1, q_len, C - e.size(-1)}, e.options())}, -1);
  pos_embed = e.to(at::kHalf);
}

at::Tensor Denoiser::context(const at::Tensor& f1, const at::Tensor& f2, bool rec) const {
  auto r = [&](const std::string& k, const at::Tensor& v) {
    if (rec) REC.put(k, v);
  };
  at::Tensor h_cond = cond_embedder(f1) + cond_embedder2(f2);
  h_cond = h_cond.to(at::kHalf);
  for (size_t i = 0; i < context_refiner.size(); i++) {
    at::Tensor rope = context_repo[i](h_cond);
    if (i == 0) {
      r("denoiser.context_repo0#0.out", rope);
      r("denoiser.context_refiner0#0.in.0", h_cond);
    }
    h_cond = context_refiner[i](h_cond, at::Tensor(), rope);
    r("denoiser.context_refiner" + std::to_string(i) + "#0.out", h_cond);
  }
  return h_cond;
}

Denoiser::Prefix Denoiser::prefix(const at::Tensor& z, const at::Tensor& cam, const at::Tensor& t, bool rec) const {
  auto r = [&](const std::string& k, const at::Tensor& v) {
    if (rec) REC.put(k, v);
  };
  Prefix P;
  at::Tensor il = input_layer(z);
  r("denoiser.input_layer#0.out", il);
  at::Tensor h_x = il.to(at::kHalf);
  P.t_emb = t_embedder(t);
  r("denoiser.t_embedder#0.out", P.t_emb);
  at::Tensor ada = adaLN(at::silu(P.t_emb));
  r("denoiser.adaLN#0.out", ada);
  P.t_mod = ada.to(at::kHalf);
  h_x = h_x + pos_embed;
  for (size_t i = 0; i < noise_refiner.size(); i++) {
    at::Tensor rope = noise_repo[i](h_x);
    if (i == 0) {
      r("denoiser.noise_repo0#0.in.0", h_x);
      r("denoiser.noise_repo0#0.out", rope);
    }
    h_x = noise_refiner[i](h_x, P.t_mod, rope, (rec && i == 0) ? "denoiser.noise_refiner0" : "");
    if (i == 0) r("denoiser.noise_refiner0#0.out", h_x);
  }
  P.h_x = h_x;
  at::Tensor h_cam = cam2(at::gelu(cam0(cam), "tanh"));
  r("denoiser.cam_refiner#0.out", h_cam);
  P.h_cam = h_cam.to(at::kHalf);
  return P;
}

std::pair<at::Tensor, at::Tensor> Denoiser::joint(const Prefix& P, const at::Tensor& h_cond, const at::Tensor& z,
                                                  const at::Tensor& cam, bool rec) const {
  auto r = [&](const std::string& k, const at::Tensor& v) {
    if (rec) REC.put(k, v);
  };
  at::Tensor h = at::cat({P.h_x, h_cond, P.h_cam}, 1);
  for (size_t i = 0; i < blocks.size(); i++) {
    at::Tensor rope = repo[i](h);
    if (i == 0) {
      r("denoiser.repo0#0.in.0", h);
      r("denoiser.repo0#0.out", rope);
      r("denoiser.blocks0#0.in.0", h);
    }
    h = blocks[i](h, P.t_mod, rope, (rec && i == 0) ? "denoiser.blocks0" : "");
    if (i == 0) r("denoiser.blocks0#0.out", h);
    if (i == 23) r("denoiser.blocks23#0.out", h);
  }
  const int64_t Lz = z.size(1), Lc = cam.size(1);
  at::Tensor hx = h.slice(1, 0, Lz).to(z.scalar_type());
  hx = layer_norm_plain(hx);
  at::Tensor hc = h.slice(1, h.size(1) - Lc, h.size(1)).to(cam.scalar_type());
  hc = layer_norm_plain(hc);
  auto ss = (shift_table + P.t_emb.unsqueeze(1)).chunk(2, 1);
  hx = hx * (ss[1] + 1) + ss[0];
  hc = hc * (ss[1] + 1) + ss[0];
  return {out_layer(hx), cam_out_layer(hc)};
}

std::pair<at::Tensor, at::Tensor> Denoiser::operator()(const at::Tensor& z, const at::Tensor& cam,
                                                       const at::Tensor& t, const at::Tensor& f1,
                                                       const at::Tensor& f2) const {
  // same op order as LatentSeqMMFlowModel.forward (the context stream is computed before the time embedding there,
  // but the two are independent, so the order does not change any value)
  const bool first = REC.on && REC.calls["denoiser.internals"]++ == 0;
  at::Tensor h_cond = context(f1, f2, first);
  Prefix P = prefix(z, cam, t, first);
  return joint(P, h_cond, z, cam, first);
}

}  // namespace idol

// The fast build (--build fast): C++ port of fast/model.py + fast/pipeline.py (see docs/ab-log.md).
//  * context stream once per request; time embedding, adaLN, noise refiner, camera refiner once per step;
//  * cond and uncond rows packed into one matrix; the uncond context is ONE token whose key has multiplicity 4,101
//    (attention over the other keys with cuDNN, log-sum-exp merge in idol_cuda::merge_extra_key);
//  * idol_cuda kernels: residual + LayerNorm + modulation (+ next RePo LayerNorm), RePo RoPE + q/k RMSNorm,
//    fc1 + GELU in the cuBLASLt epilogue; RePo fp32 GEMMs in TF32 (as the shipped optimize path);
//  * octree descent with loop-free systematic resampling; Gaussian activations + PLY columns in one kernel.
#include "fast.h"

#include <ATen/Context.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "idol_kernels.h"

namespace idol {

namespace {
constexpr int64_t C = 1024, H = 16, Dh = 64;

struct RepoW {
  at::Tensor w, b, gc, fin, freqs;
};
struct BlockW {
  bool mod = true;
  at::Tensor qkv_w, qkv_b, out_w, out_b, gq, gk, fc1_w, fc1_b, fc2_w, fc2_b, shift_table, n1w, n1b, n2w, n2b;
};
using Layer = std::pair<RepoW, BlockW>;
using Mods = std::vector<at::Tensor>;  // 6 chunks [1, 1024] fp16 (shift_msa, scale_msa, gate_msa, shift_mlp, ...)

struct Seg {
  int64_t start, len, extra;  // extra < 0: plain attention over the segment
  double log_mult;
};

RepoW repo_w(const RePo& r) {
  return {r.norm.w.to(at::kFloat).contiguous(), r.norm.b.to(at::kFloat).contiguous(),
          at::cat({r.gate_map.w, r.content_map.w}, 0).to(at::kFloat).contiguous(),
          r.final_map.w.to(at::kFloat).contiguous(), at::cat({r.f0, r.f1, r.f2}, 0).to(at::kFloat).contiguous()};
}
BlockW block_w(const UnifiedBlock& b) {
  BlockW w;
  w.mod = b.modulation;
  w.qkv_w = b.attn.qkv.w;
  w.qkv_b = b.attn.qkv.b;
  w.out_w = b.attn.out.w;
  w.out_b = b.attn.out.b;
  w.gq = b.attn.qn.gamma.to(at::kFloat).contiguous();
  w.gk = b.attn.kn.gamma.to(at::kFloat).contiguous();
  w.fc1_w = b.mlp.l0.w;
  w.fc1_b = b.mlp.l0.b;
  w.fc2_w = b.mlp.l2.w;
  w.fc2_b = b.mlp.l2.b;
  if (w.mod) {
    w.shift_table = b.shift_table;
  } else {
    w.n1w = b.norm1.w.to(at::kFloat).contiguous();
    w.n1b = b.norm1.b.to(at::kFloat).contiguous();
    w.n2w = b.norm2.w.to(at::kFloat).contiguous();
    w.n2b = b.norm2.b.to(at::kFloat).contiguous();
  }
  return w;
}

// FlashAttention-3 (Hopper) from the Hub kernel kernels-community/flash-attn3 (stable-ABI build), loaded with dlopen
// and called through the dispatcher with its schema defaults (boxed call); cuDNN when it is not given.
std::optional<c10::OperatorHandle> FA3;

bool load_fa3(const std::string& so) {
  if (so.empty()) return false;
  // RTLD_LAZY: the library also carries an (unused) Python module init referencing libpython symbols; the op
  // registration runs from static initialisers through the torch stable ABI, so no Python is loaded or called.
  if (!dlopen(so.c_str(), RTLD_LAZY | RTLD_GLOBAL)) {
    // The Hub build also binds libpython symbols (PyModule_Create2) at load time, so it does not load into a
    // process without libpython: fall back to cuDNN attention (docs/benchmark.md).
    std::fprintf(stderr, "warning: cannot load FlashAttention-3 (%s); using cuDNN attention\n", dlerror());
    return false;
  }
  for (const auto& name : c10::Dispatcher::singleton().getAllOpNames()) {
    if (name.name.find("flash_attn3") != std::string::npos &&
        name.name.size() >= 20 && name.name.substr(name.name.size() - 20) == "::_flash_attn_forward") {
      FA3 = c10::Dispatcher::singleton().findSchemaOrThrow(name.name.c_str(), "");
      return true;
    }
  }
  throw std::runtime_error("no *flash_attn3*::_flash_attn_forward op registered by " + so);
}

std::tuple<at::Tensor, at::Tensor> fa3_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  const auto& schema = FA3->schema();
  torch::jit::Stack st;
  st.reserve(schema.arguments().size());
  st.emplace_back(q);
  st.emplace_back(k);
  st.emplace_back(v);
  for (size_t i = 3; i < schema.arguments().size(); i++) {
    const auto& d = schema.arguments()[i].default_value();
    if (!d) throw std::runtime_error("FA3 argument without default: " + schema.arguments()[i].name());
    st.push_back(*d);
  }
  FA3->callBoxed(&st);
  return {st[0].toTensor(), st[1].toTensor()};
}

std::tuple<at::Tensor, at::Tensor> attn(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool lse) {
  // q/k/v [1, L, H, 64] views -> out [1, L, H, 64], lse [1, H, L]
  if (FA3) return fa3_fwd(q, k, v);
  auto r = at::_scaled_dot_product_cudnn_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                   std::nullopt, lse, 0.0, false, false, std::nullopt);
  at::Tensor o = std::get<0>(r).transpose(1, 2);
  at::Tensor l;
  if (lse) l = std::get<1>(r).reshape({q.size(0), H, q.size(1)}).contiguous();
  return {o, l};
}

std::vector<double> grid(int64_t steps, double rescale_t) {
  std::vector<double> t(steps + 1);
  for (int64_t i = 0; i <= steps; i++) t[i] = 1.0 + static_cast<double>(i) * (-1.0 / static_cast<double>(steps));
  t[steps] = 0.0;
  for (auto& v : t) v = rescale_t * v / (1 + (rescale_t - 1) * v);
  return t;
}

struct TF32Guard {
  bool prev;
  TF32Guard() : prev(at::globalContext().allowTF32CuBLAS()) { at::globalContext().setAllowTF32CuBLAS(true); }
  ~TF32Guard() { at::globalContext().setAllowTF32CuBLAS(prev); }
};
}  // namespace

struct FastPipeline::Impl {
  std::vector<Layer> noise, ctx, joint;
  at::Tensor h_cond, h_uncond, t_freqs;
  int64_t lc_full = 0;
  double gs_args[6];

  at::Tensor delta(const RepoW& r, const at::Tensor& ln) const {
    at::Tensor gc = at::matmul(ln, r.gc.t());
    return at::matmul(at::silu(gc.slice(1, 0, 128)) * gc.slice(1, 128, 256), r.fin.t());
  }

  at::Tensor attention(const at::Tensor& qkv, const at::Tensor& dl, const RepoW& r, const BlockW& b,
                       const std::vector<Seg>& segs) const {
    const int64_t M = qkv.size(0);
    auto hopt = qkv.options();
    at::Tensor q = at::empty({M, H, Dh}, hopt), k = at::empty({M, H, Dh}, hopt);
    idol_cuda::rope_rms_qk(q, k, qkv, dl, r.freqs, b.gq, b.gk);
    at::Tensor v = qkv.slice(1, 2 * C, 3 * C).view({M, H, Dh});
    at::Tensor out = at::empty({M, H, Dh}, hopt);
    for (const auto& s : segs) {
      at::Tensor qs = q.slice(0, s.start, s.start + s.len).unsqueeze(0);
      if (s.len == 1 && s.extra < 0) {
        out.slice(0, s.start, s.start + 1).copy_(v.slice(0, s.start, s.start + 1));
      } else if (s.extra < 0) {
        auto [o, l] = attn(qs, k.slice(0, s.start, s.start + s.len).unsqueeze(0),
                           v.slice(0, s.start, s.start + s.len).unsqueeze(0), false);
        out.slice(0, s.start, s.start + s.len).copy_(o[0]);
      } else {
        auto [o, l] = attn(qs, k.slice(0, s.start, s.start + s.len - 1).unsqueeze(0),
                           v.slice(0, s.start, s.start + s.len - 1).unsqueeze(0), true);
        if (o.stride(2) != Dh || o.stride(3) != 1) o = o.contiguous();
        at::Tensor merged = at::empty_like(o);
        idol_cuda::merge_extra_key(merged, o, l, qs, k.select(0, s.extra).unsqueeze(0).contiguous(),
                                   v.select(0, s.extra).unsqueeze(0).contiguous(), 1.0 / std::sqrt(double(Dh)),
                                   s.log_mult);
        out.slice(0, s.start, s.start + s.len).copy_(merged[0]);
      }
    }
    return out.view({M, C});
  }

  // y_out (fp16), optional x_out (sum), optional y2 (fp32 RePo LN)
  void addln(at::Tensor& y, std::optional<at::Tensor> xo, const at::Tensor& x, std::optional<at::Tensor> h,
             std::optional<at::Tensor> gate, std::optional<at::Tensor> w, std::optional<at::Tensor> bb,
             std::optional<at::Tensor> sc, std::optional<at::Tensor> sh, std::optional<at::Tensor> y2,
             std::optional<at::Tensor> w2, std::optional<at::Tensor> b2) const {
    idol_cuda::add_ln_mod(y, xo, x, h, gate, w, bb, sc, sh, x.numel() / C, 1e-6, y2, w2, b2, 1e-5);
  }

  at::Tensor stack(at::Tensor x, const std::vector<Layer>& layers, const std::vector<Mods>& mods,
                   const std::vector<Seg>& segs) const {
    const int64_t M = x.size(0);
    auto hopt = x.options();
    auto fopt = hopt.dtype(at::kFloat);
    at::Tensor a1 = at::empty({M, C}, hopt), lnr = at::empty({M, C}, fopt);
    {
      const auto& [r, b] = layers[0];
      if (b.mod)
        addln(a1, std::nullopt, x, std::nullopt, std::nullopt, std::nullopt, std::nullopt, mods[0][1], mods[0][0],
              lnr, r.w, r.b);
      else
        addln(a1, std::nullopt, x, std::nullopt, std::nullopt, b.n1w, b.n1b, std::nullopt, std::nullopt, lnr, r.w,
              r.b);
    }
    for (size_t i = 0; i < layers.size(); i++) {
      const auto& [r, b] = layers[i];
      at::Tensor dl = delta(r, lnr);
      at::Tensor qkv = at::linear(a1, b.qkv_w, b.qkv_b);
      at::Tensor o = attention(qkv, dl, r, b, segs);
      at::Tensor h = at::linear(o, b.out_w, b.out_b);
      at::Tensor x2 = at::empty({M, C}, hopt), a2 = at::empty({M, C}, hopt);
      if (b.mod)
        addln(a2, x2, x, h, mods[i][2], std::nullopt, std::nullopt, mods[i][4], mods[i][3], std::nullopt,
              std::nullopt, std::nullopt);
      else
        addln(a2, x2, x, h, std::nullopt, b.n2w, b.n2b, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
              std::nullopt);
      at::Tensor f = at::empty({M, b.fc1_w.size(0)}, hopt);
      idol_cuda::linear_gelu(f, a2, b.fc1_w, b.fc1_b);
      h = at::linear(f, b.fc2_w, b.fc2_b);
      std::optional<at::Tensor> gate;
      if (b.mod) gate = mods[i][5];
      at::Tensor x3 = at::empty({M, C}, hopt);
      if (i + 1 == layers.size()) {
        if (gate) {
          at::Tensor dummy = at::empty({M, C}, hopt);
          addln(dummy, x3, x2, h, gate, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt);
          return x3;
        }
        return x2 + h;
      }
      const auto& [r2, b2] = layers[i + 1];
      if (b2.mod)
        addln(a1, x3, x2, h, gate, std::nullopt, std::nullopt, mods[i + 1][1], mods[i + 1][0], lnr, r2.w, r2.b);
      else
        addln(a1, x3, x2, h, gate, b2.n1w, b2.n1b, std::nullopt, std::nullopt, lnr, r2.w, r2.b);
      x = x3;
    }
    return x;
  }

  static std::vector<Mods> make_mods(const at::Tensor& t_mod, const std::vector<Layer>& layers) {
    std::vector<Mods> out;
    for (const auto& [r, b] : layers) {
      if (b.mod) {
        at::Tensor m = t_mod + b.shift_table.to(t_mod.scalar_type());
        out.push_back(m.chunk(6, 1));
      } else {
        out.emplace_back();
      }
    }
    return out;
  }
};

bool FastPipeline::use_fa3(const std::string& so) { return load_fa3(so); }

FastPipeline::FastPipeline(const Denoiser& d, const OctreeDecoder& o, const GsDecoder& g, at::Device dv)
    : dn(d), oct(o), gs(g), dev(dv) {
  impl = new Impl();
  for (size_t i = 0; i < d.noise_refiner.size(); i++)
    impl->noise.emplace_back(repo_w(d.noise_repo[i]), block_w(d.noise_refiner[i]));
  for (size_t i = 0; i < d.context_refiner.size(); i++)
    impl->ctx.emplace_back(repo_w(d.context_repo[i]), block_w(d.context_refiner[i]));
  for (size_t i = 0; i < d.blocks.size(); i++) impl->joint.emplace_back(repo_w(d.repo[i]), block_w(d.blocks[i]));
  impl->t_freqs = d.t_embedder.freqs;
  // Gaussian decoder head constants (Gaussian.__init__), computed in fp32 on the CPU as the reference does
  at::Tensor sb = at::scalar_tensor(0.004, at::TensorOptions().dtype(at::kFloat));
  at::Tensor ob = at::scalar_tensor(0.1, at::TensorOptions().dtype(at::kFloat));
  impl->gs_args[0] = g.base_offset_scale.item<double>();
  impl->gs_args[1] = (sb + at::log(-at::expm1(-sb))).item<double>();
  impl->gs_args[2] = at::log(ob / at::rsub(ob, 1)).item<double>();
  impl->gs_args[3] = 0.0009;
  impl->gs_args[4] = 0.1;
  impl->gs_args[5] = 1.5;
}
FastPipeline::~FastPipeline() { delete impl; }

void FastPipeline::prepare(const at::Tensor& f1, const at::Tensor& f2, bool collapse) {
  TF32Guard tf32;
  const auto& d = dn;
  impl->lc_full = f1.size(1);
  at::Tensor hc = (d.cond_embedder(f1) + d.cond_embedder2(f2)).to(at::kHalf)[0].contiguous();
  const int64_t nu = collapse ? 1 : f1.size(1);
  at::Tensor z1 = at::zeros({1, nu, f1.size(2)}, f1.options()), z2 = at::zeros({1, nu, f2.size(2)}, f2.options());
  at::Tensor hu = (d.cond_embedder(z1) + d.cond_embedder2(z2)).to(at::kHalf)[0].contiguous();
  std::vector<Mods> none(impl->ctx.size());
  impl->h_cond = impl->stack(hc, impl->ctx, none, {{0, hc.size(0), -1, 0.0}});
  impl->h_uncond = impl->stack(hu, impl->ctx, none, {{0, hu.size(0), -1, 0.0}});
  collapse_ = collapse;
}

std::pair<at::Tensor, at::Tensor> FastPipeline::velocity(const at::Tensor& z, const at::Tensor& cam,
                                                         const at::Tensor& t, double cfg, bool both,
                                                         std::pair<at::Tensor, at::Tensor>* uncond) {
  TF32Guard tf32;
  const auto& d = dn;
  Impl& I = *impl;
  const int64_t Lz = z.size(1), Lc = I.h_cond.size(0), Lu = I.h_uncond.size(0);
  at::Tensor args = t.unsqueeze(1).to(at::kFloat) * I.t_freqs.unsqueeze(0);
  at::Tensor t_emb = d.t_embedder.l2(at::silu(d.t_embedder.l0(at::cat({at::cos(args), at::sin(args)}, -1))));
  at::Tensor t_mod = d.adaLN(at::silu(t_emb)).to(at::kHalf);
  at::Tensor h_x = d.input_layer(z).to(at::kHalf)[0] + d.pos_embed[0];
  h_x = I.stack(h_x, I.noise, Impl::make_mods(t_mod, I.noise), {{0, Lz, -1, 0.0}});
  at::Tensor h_cam = d.cam2(at::gelu(d.cam0(cam), "tanh")).to(at::kHalf)[0];
  const int64_t n1 = Lz + Lc + 1;
  at::Tensor x;
  std::vector<Seg> segs;
  int64_t cam_u;
  if (collapse_) {
    x = at::cat({h_x, I.h_cond, h_cam, h_x, h_cam, I.h_uncond}, 0);
    segs = {{0, n1, -1, 0.0}, {n1, Lz + 2, n1 + Lz + 1, std::log(static_cast<double>(I.lc_full))}};
    cam_u = n1 + Lz;
  } else {
    x = at::cat({h_x, I.h_cond, h_cam, h_x, I.h_uncond, h_cam}, 0);
    segs = {{0, n1, -1, 0.0}, {n1, Lz + Lu + 1, -1, 0.0}};
    cam_u = n1 + Lz + Lu;
  }
  x = I.stack(x, I.joint, Impl::make_mods(t_mod, I.joint), segs);
  at::Tensor rows = at::cat({x.slice(0, 0, Lz), x.slice(0, n1, n1 + Lz)}, 0).to(at::kFloat);
  at::Tensor cams = at::stack({x[n1 - 1], x[cam_u]}, 0).to(at::kFloat);
  rows = layer_norm_plain(rows);
  cams = layer_norm_plain(cams);
  auto ss = (d.shift_table + t_emb.unsqueeze(1)).chunk(2, 1);
  at::Tensor scale = ss[1][0], shift = ss[0][0];
  rows = rows * (scale + 1) + shift;
  cams = cams * (scale + 1) + shift;
  at::Tensor vz = d.out_layer(rows), vc = d.cam_out_layer(cams);
  at::Tensor vzc = vz.slice(0, 0, Lz).unsqueeze(0), vzu = vz.slice(0, Lz, 2 * Lz).unsqueeze(0);
  at::Tensor vcc = vc.slice(0, 0, 1).unsqueeze(0), vcu = vc.slice(0, 1, 2).unsqueeze(0);
  if (both) {
    if (uncond) *uncond = {vzu, vcu};
    return {vzc, vcc};
  }
  return {vzc * (1 + cfg) - vzu * cfg, vcc * (1 + cfg) - vcu * cfg};
}

std::pair<at::Tensor, at::Tensor> FastPipeline::denoise(const at::Tensor& z, const at::Tensor& cam,
                                                        const at::Tensor& t, const at::Tensor& f1,
                                                        const at::Tensor& f2) {
  // teacher-forced comparison of one call: f1 == 0 means the unconditional pass
  const bool uncond_call = f1.abs().max().item<double>() == 0.0;
  if (!impl->h_cond.defined() || last_f1_.data_ptr() != f1.data_ptr()) {
    if (!uncond_call) {
      prepare(f1, f2, true);
      last_f1_ = f1;
    }
  }
  std::pair<at::Tensor, at::Tensor> u;
  auto c = velocity(z, cam, t, 0.0, true, &u);
  return uncond_call ? u : c;
}

std::pair<at::Tensor, at::Tensor> FastPipeline::sample(at::Tensor z, at::Tensor cam, const at::Tensor& f1,
                                                       const at::Tensor& f2, int64_t steps, double rescale_t,
                                                       double cfg) {
  prepare(f1, f2, true);
  const auto ts = grid(steps, rescale_t);
  for (int64_t i = 0; i < steps; i++) {
    at::Tensor tt = at::full({1}, static_cast<float>(1000 * ts[i]), z.options());
    auto v = velocity(z, cam, tt, cfg);
    const double dt = ts[i] - ts[i + 1];
    z = z - v.first * dt;
    cam = cam - v.second * dt;
  }
  return {z, cam};
}

at::Tensor FastPipeline::octree_sample(const at::Tensor& cond, int64_t num_points, int level, Rng& rng) {
  const auto lopt = at::TensorOptions().dtype(at::kLong).device(dev);
  std::vector<int64_t> co;
  for (int k = 0; k < 2; k++)
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) co.insert(co.end(), {i, j, k});
  at::Tensor child_offset = at::tensor(co, at::TensorOptions().dtype(at::kLong)).view({8, 3}).to(dev);
  at::Tensor coords = at::zeros({1, 3}, lopt);
  at::Tensor counts = at::full({1}, num_points, lopt);
  for (int l = 1; l <= level; l++) {
    const int64_t res_p = int64_t(1) << (l - 1);
    at::Tensor parent = ((coords.to(at::kFloat) + 0.5) / res_p).unsqueeze(0);
    at::Tensor res = at::full({1}, int64_t(1) << l, lopt);
    at::Tensor probs = at::softmax(oct(parent, res, cond), -1)[0].to(at::kFloat).clamp_min(0);
    const int64_t R = probs.size(0), P = probs.size(1);
    at::Tensor s = probs.sum(1, true);
    probs = at::where(s.eq(0), at::full_like(probs, 1.0 / P), probs / s.clamp_min(1));
    at::Tensor cdf = probs.cumsum(1);
    at::Tensor n = counts.to(at::kFloat).unsqueeze(1);
    at::Tensor u0 = rng.rand({R, 1}, at::TensorOptions().dtype(at::kFloat).device(dev)) / n.clamp_min(1);
    at::Tensor F = at::clamp_min(at::floor((cdf.slice(1, 0, P - 1) - u0) * n) + 1, 0);
    F = at::minimum(F, n);
    F = at::cat({at::zeros_like(n), F, n}, 1);
    at::Tensor sampled = (F.slice(1, 1, P + 1) - F.slice(1, 0, P)).to(at::kLong).flatten();
    at::Tensor child = (coords.unsqueeze(1) * 2 + child_offset.unsqueeze(0)).flatten(0, 1);
    at::Tensor keep = sampled > 0;
    coords = child.index({keep});
    counts = sampled.index({keep});
  }
  at::Tensor pts = at::repeat_interleave(coords, counts, std::optional<int64_t>(0), std::optional<int64_t>(num_points));
  at::Tensor r = rng.rand({pts.size(0), 3}, at::TensorOptions().dtype(at::kFloat).device(dev));
  return ((pts.to(at::kFloat) + r) / (int64_t(1) << level)).unsqueeze(0);
}

at::Tensor FastPipeline::gsdec(const at::Tensor& points, const at::Tensor& cond) { return gs(points, cond); }

at::Tensor FastPipeline::ply_rows(const at::Tensor& anchors, const at::Tensor& feats) {
  at::Tensor rows = at::empty({anchors.size(1) * 32, 17}, feats.options().dtype(at::kFloat));
  idol_cuda::gaussians_ply(rows, feats[0].contiguous(), anchors[0].contiguous(), gs.perturb.contiguous(),
                           impl->gs_args[0], impl->gs_args[1], impl->gs_args[2], impl->gs_args[3], impl->gs_args[4],
                           impl->gs_args[5]);
  return rows;
}

void write_ply_rows(const at::Tensor& rows_dev, const std::string& path) {
  at::Tensor rows = rows_dev.to(at::kCPU).contiguous();
  const int64_t N = rows.size(0);
  std::ostringstream h;
  h << "ply\nformat binary_little_endian 1.0\nelement vertex " << N << "\n";
  const char* names[] = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
  for (auto n : names) h << "property float " << n << "\n";
  h << "end_header\n";
  const std::string header = h.str();
  std::ofstream f(path, std::ios::binary);
  f.write(header.data(), header.size());
  f.write(reinterpret_cast<const char*>(rows.data_ptr<float>()), N * 17 * sizeof(float));
  if (!f) throw std::runtime_error("failed writing " + path);
}

}  // namespace idol

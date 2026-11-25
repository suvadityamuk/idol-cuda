#include "decoders.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace idol {

using IdxList = c10::List<std::optional<at::Tensor>>;

// ---------------------------------------------------------------------------------------------- RNG
at::Tensor Rng::take(const char* name, at::Tensor fresh) {
  char key[64];
  std::snprintf(key, sizeof(key), "rng.%03d.%s", pos, name);
  if (replay) {
    auto it = replay->find(key);
    if (it == replay->end()) throw std::runtime_error(std::string("replay: missing ") + key);
    if (it->second.sizes() != fresh.sizes() || it->second.scalar_type() != fresh.scalar_type())
      throw std::runtime_error(std::string("replay: shape/dtype mismatch at ") + key);
    fresh.copy_(it->second);
  }
  REC.put(key, fresh);
  pos++;
  return fresh;
}
at::Tensor Rng::randn(at::IntArrayRef shape, at::TensorOptions opt) { return take("randn", at::randn(shape, opt)); }
at::Tensor Rng::rand(at::IntArrayRef shape, at::TensorOptions opt) { return take("rand", at::rand(shape, opt)); }
at::Tensor Rng::rand_like(const at::Tensor& like, at::ScalarType dtype) {
  return take("rand_like", at::rand_like(like, like.options().dtype(dtype)));
}

// ---------------------------------------------------------------------------------------------- attention
MHA::MHA(const Weights& ws, const std::string& p, bool cross_) : cross(cross_) {
  if (cross) {
    to_q = Linear(ws, p + ".to_q");
    to_kv = Linear(ws, p + ".to_kv");
  } else {
    to_qkv = Linear(ws, p + ".to_qkv");
  }
  qn = MultiHeadRMSNorm(ws, p + ".q_rms_norm", 64);
  kn = MultiHeadRMSNorm(ws, p + ".k_rms_norm", 64);
  to_out = Linear(ws, p + ".to_out");
}

at::Tensor MHA::operator()(const at::Tensor& x, const at::Tensor& ctx) const {
  const int64_t B = x.size(0), L = x.size(1);
  at::Tensor h;
  if (cross) {
    at::Tensor q = qn(to_q(x).reshape({B, L, heads, -1}));
    auto kv = to_kv(ctx).reshape({B, ctx.size(1), 2, heads, -1}).unbind(2);
    h = sdpa_nlhc(q, kn(kv[0]), kv[1]);
  } else {
    auto qkv = to_qkv(x).reshape({B, L, 3, heads, -1}).unbind(2);
    at::Tensor packed = at::stack({qn(qkv[0]), kn(qkv[1]), qkv[2]}, 2);
    auto u = packed.unbind(2);
    h = sdpa_nlhc(u[0], u[1], u[2]);
  }
  return to_out(h.reshape({B, L, -1}));
}

OctreeBlock::OctreeBlock(const Weights& ws, const std::string& p)
    : norm1(ws, p + ".norm1", 1024, 1e-6, false),
      norm2(ws, p + ".norm2", 1024, 1e-6, false),
      cross_attn(ws, p + ".cross_attn", true),
      mlp(ws, p + ".mlp") {}

at::Tensor OctreeBlock::operator()(const at::Tensor& x_in, const at::Tensor& mod, const at::Tensor& ctx) const {
  at::Tensor x = x_in;
  auto c = mod.chunk(6, 1);
  at::Tensor h = norm1(x);
  h = h * (c[1].unsqueeze(1) + 1) + c[0].unsqueeze(1);
  h = cross_attn(h, ctx);
  x = x + h * c[2].unsqueeze(1);
  h = norm2(x);
  h = h * (c[4].unsqueeze(1) + 1) + c[3].unsqueeze(1);
  h = mlp(h);
  return x + h * c[5].unsqueeze(1);
}

GsBlock::GsBlock(const Weights& ws, const std::string& p)
    : norm1(ws, p + ".norm1", 1024, 1e-6, false),
      norm2(ws, p + ".norm2", 1024, 1e-6, true),
      norm3(ws, p + ".norm3", 1024, 1e-6, false),
      self_attn(ws, p + ".self_attn", false),
      cross_attn(ws, p + ".cross_attn", true),
      mlp(ws, p + ".mlp") {}

at::Tensor GsBlock::operator()(const at::Tensor& x_in, const at::Tensor& ctx) const {
  at::Tensor x = x_in + self_attn(norm1(x_in));
  x = x + cross_attn(norm2(x), ctx);
  return x + mlp(norm3(x));
}

PosEmbV2::PosEmbV2(at::Device dev, int64_t ch) : channels(ch) {
  const int64_t freq_dim = channels / 3 / 2;
  at::Tensor logs = at::linspace(0.0, 10.0, freq_dim, at::TensorOptions().dtype(at::kFloat));
  freqs = at::pow(at::Scalar(2.0), logs).to(dev);
}

at::Tensor PosEmbV2::operator()(const at::Tensor& x) const {
  const int64_t N = x.size(0);
  at::Tensor ang = x.unsqueeze(-1) * freqs.to(x.scalar_type());
  ang = ang * M_PI;
  at::Tensor e = at::cat({at::sin(ang), at::cos(ang)}, -1).reshape({N, -1});
  if (e.size(1) < channels) e = at::cat({e, at::zeros({N, channels - e.size(1)}, e.options())}, -1);
  return e;
}

// ---------------------------------------------------------------------------------------------- decoders
OctreeDecoder::OctreeDecoder(const Weights& ws, at::Device dev)
    : in_proj(ws, "in_proj"),
      input_layer(ws, "input_layer"),
      adaLN(ws, "adaLN_modulation.1"),
      out_proj(ws, "out_proj"),
      pos(dev, 1024),
      l_embedder(ws, "l_embedder", 1024, true, dev) {
  for (int i = 0; i < 4; i++) blocks.emplace_back(ws, "blocks." + std::to_string(i));
}

at::Tensor OctreeDecoder::operator()(const at::Tensor& x, const at::Tensor& level, const at::Tensor& cond) const {
  const int n = REC.on ? REC.next("octree") : 0;
  const std::string key = "octree#" + std::to_string(n);
  if (REC.on && n < 16) {
    REC.put(key + ".in.0", x);
    REC.put(key + ".in.1", level);
  }
  const int64_t B = x.size(0), L = x.size(1);
  at::Tensor h = in_proj(x) + pos(x.view({-1, 3})).view({B, L, -1});
  h = input_layer(h);
  at::Tensor level_emb = adaLN(at::silu(l_embedder(level.to(h.scalar_type())))).to(at::kHalf);
  at::Tensor out = h.to(at::kHalf);
  at::Tensor context = cond.to(at::kHalf);
  for (size_t i = 0; i < blocks.size(); i++) {
    out = blocks[i](out, level_emb, context);
    if (i == 0 && n == 0) REC.put("octree.blocks0#0.out", out);
  }
  out = out.to(h.scalar_type()).to(cond.scalar_type());
  at::Tensor res = out_proj(layer_norm_plain(out));
  if (REC.on && n < 16) REC.put(key + ".out", res);
  return res;
}

GsDecoder::GsDecoder(const Weights& ws, at::Device dev)
    : in_proj(ws, "in_proj"), input_layer(ws, "input_layer"), out_proj(ws, "out_proj"), pos(dev, 1024) {
  for (int i = 0; i < 16; i++) blocks.emplace_back(ws, "blocks." + std::to_string(i));
  perturb = W(ws, "points_offset_perturbation");
  base_offset_scale = W(ws, "base_offset_scale");
}

at::Tensor GsDecoder::operator()(const at::Tensor& points, const at::Tensor& cond) const {
  REC.put("gsdec#0.in.0", points);
  const int64_t B = points.size(0), L = points.size(1), C = points.size(2);
  at::Tensor h = in_proj(points) + pos(points.view({-1, C})).view({B, L, -1});
  h = input_layer(h);
  at::Tensor out = h.to(at::kHalf);
  at::Tensor context = cond.to(at::kHalf);
  for (size_t i = 0; i < blocks.size(); i++) {
    out = blocks[i](out, context);
    if (i == 0) REC.put("gsdec.blocks0#0.out", out);
    if (i == 15) REC.put("gsdec.blocks15#0.out", out);
  }
  out = out.to(h.scalar_type()).to(cond.scalar_type());
  at::Tensor res = out_proj(layer_norm_plain(out));
  REC.put("gsdec#0.out", res);
  return res;
}

// ---------------------------------------------------------------------------------------------- octree sampling
static at::Tensor systematic_sample(at::Tensor probs, const at::Tensor& counts_in, Rng& rng) {
  auto batch_shape = counts_in.sizes().vec();
  const int64_t B = counts_in.numel(), P = probs.size(-1);
  const auto dev = probs.device();
  probs = probs.view({B, P});
  at::Tensor counts = counts_in.view({B});
  probs = probs.to(at::kFloat).clamp_min_(0);
  at::Tensor row_sums = probs.sum(1, true);
  at::Tensor zero_mask = row_sums.eq(0);
  probs = probs / row_sums.clamp_min_(1);
  if (zero_mask.any().item<bool>()) {
    probs = probs.clone();
    probs.index_put_(IdxList({zero_mask.expand_as(probs)}), at::scalar_tensor(1.0 / P, probs.options()));
  }
  counts = counts.to(dev, at::kLong);
  at::Tensor out = at::zeros({B, P}, at::TensorOptions().dtype(at::kLong).device(dev));
  at::Tensor cdf = probs.cumsum(1).clamp(c10::nullopt, 1.0 - 1e-12);
  auto uq = at::_unique2(counts, false, true, false);
  at::Tensor unique_n = std::get<0>(uq), inv = std::get<1>(uq);
  at::Tensor un_cpu = unique_n.to(at::kCPU);
  const int64_t* un = un_cpu.data_ptr<int64_t>();
  for (int64_t index = 0; index < un_cpu.numel(); index++) {
    const int64_t n = un[index];
    if (n == 0) continue;
    at::Tensor rows = (inv == index).nonzero().squeeze(1);
    const int64_t r = rows.numel();
    at::Tensor u0 = rng.rand({r, 1}, at::TensorOptions().dtype(at::kFloat).device(dev)) / static_cast<double>(n);
    at::Tensor grid = at::arange(n, at::TensorOptions().dtype(at::kFloat).device(dev)).unsqueeze(0) /
                      static_cast<double>(n);
    at::Tensor us = (u0 + grid).clamp(c10::nullopt, 1.0 - 1e-12);
    at::Tensor idx = at::searchsorted(cdf.index_select(0, rows), us).clamp_max(P - 1);
    at::Tensor buf = at::zeros({r, P}, at::TensorOptions().dtype(at::kFloat).device(dev));
    buf.scatter_add_(1, idx, at::ones_like(idx, idx.options().dtype(buf.scalar_type())));
    out.index_copy_(0, rows, buf.to(at::kLong));
  }
  batch_shape.push_back(P);
  return out.view(batch_shape);
}

at::Tensor octree_sample(const OctreeDecoder& m, const at::Tensor& cond, int64_t num_points, int level, Rng& rng) {
  const int64_t B = cond.size(0);
  const auto dev = cond.device();
  const auto lopt = at::TensorOptions().dtype(at::kLong).device(dev);
  std::vector<int64_t> co;
  for (int k = 0; k < 2; k++)
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) co.insert(co.end(), {i, j, k});
  at::Tensor child_offset = at::tensor(co, at::TensorOptions().dtype(at::kLong)).view({8, 3}).to(dev);
  at::Tensor prev_coords_int = at::zeros({B, 1, 3}, lopt);
  at::Tensor prev_counts = at::full({B, 1}, num_points, lopt);
  at::Tensor batch_indices_range = at::arange(B, lopt).unsqueeze(1);
  for (int l = 1; l <= level; l++) {
    const int64_t res_p = int64_t(1) << (l - 1);
    at::Tensor parent = (prev_coords_int.to(at::kFloat) + 0.5) / res_p;
    at::Tensor res_tensor = at::full({B}, int64_t(1) << l, lopt);
    at::Tensor pred_probs = at::softmax(m(parent, res_tensor, cond), -1);
    at::Tensor sampled = systematic_sample(pred_probs, prev_counts, rng).flatten(1, 2);
    at::Tensor child = (prev_coords_int.unsqueeze(2) * 2 + child_offset.unsqueeze(0).unsqueeze(0)).flatten(1, 2);
    at::Tensor mask = sampled > 0;
    const int64_t max_valid = mask.sum(1).max().item<int64_t>();
    at::Tensor vs = at::index(mask.cumsum(1) - 1, IdxList({mask}));
    at::Tensor vb = at::index(batch_indices_range.expand_as(mask), IdxList({mask}));
    prev_coords_int = at::zeros({B, max_valid, 3}, lopt.dtype(child.scalar_type()));
    prev_coords_int.index_put_(IdxList({vb, vs}), at::index(child, IdxList({mask})));
    prev_counts = at::zeros({B, max_valid}, lopt.dtype(sampled.scalar_type()));
    prev_counts.index_put_(IdxList({vb, vs}), at::index(sampled, IdxList({mask})));
  }
  at::Tensor coords_int =
      at::repeat_interleave(prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), 0).reshape({B, num_points, -1});
  at::Tensor r = rng.rand_like(coords_int, at::kFloat);
  return (coords_int.to(at::kFloat) + r) / (int64_t(1) << level);
}

// ---------------------------------------------------------------------------------------------- gaussians
static at::Tensor slice_feat(const at::Tensor& f, int64_t start, int64_t end, std::vector<int64_t> shape) {
  std::vector<int64_t> s = {f.size(0), -1, 32};
  s.insert(s.end(), shape.begin(), shape.end());
  return f.slice(2, start, end).reshape(s);
}

Gaussians to_gaussians(const GsDecoder& m, const at::Tensor& points, const at::Tensor& features) {
  // layout: _xyz 96, _features_dc 96, _scaling 96, _rotation 128, _opacity 32, _offset_scale 32
  const auto dev = features.device();
  at::Tensor offset_scale = at::softplus(slice_feat(features, 448, 480, {1}) + m.base_offset_scale);
  at::Tensor offset = slice_feat(features, 0, 96, {3}) * 1.0 + m.perturb;
  offset = at::tanh(offset) * 0.5;
  offset = offset * 1.5;
  offset = offset * offset_scale;
  at::Tensor f0 = features.slice(0, 0, 1);
  at::Tensor fdc = slice_feat(f0, 96, 192, {1, 3}).select(0, 0).flatten(0, 1) * 1.0;
  at::Tensor rs = slice_feat(f0, 192, 288, {3}).select(0, 0).flatten(0, 1) * 1.0;
  at::Tensor rr = slice_feat(f0, 288, 416, {4}).select(0, 0).flatten(0, 1) * 0.1;
  at::Tensor ro = slice_feat(f0, 416, 448, {1}).select(0, 0).flatten(0, 1) * 1.0;
  at::Tensor raw_xyz = (offset.select(0, 0) + points.select(0, 0).unsqueeze(1)).flatten(0, 1);

  // Gaussian.__init__
  at::Tensor aabb = at::tensor({-0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f}, at::TensorOptions().dtype(at::kFloat)).to(dev);
  at::Tensor sb = at::scalar_tensor(0.004, at::TensorOptions().dtype(at::kFloat));
  at::Tensor scaling_offset = (sb + at::log(-at::expm1(-sb))).to(dev);
  at::Tensor ob = at::scalar_tensor(0.1, at::TensorOptions().dtype(at::kFloat));
  at::Tensor opacity_offset = at::log(ob / at::rsub(ob, 1)).to(dev);
  at::Tensor rotation_offset = at::tensor({1.0f, 0.0f, 0.0f, 0.0f}, at::TensorOptions().dtype(at::kFloat)).to(dev);
  Gaussians g;
  g.xyz = raw_xyz * aabb.slice(0, 3, 6).unsqueeze(0) + aabb.slice(0, 0, 3).unsqueeze(0);
  g.features_dc = fdc;
  at::Tensor s = at::softplus(rs + scaling_offset);
  g.scaling = at::sqrt(at::square(s) + 0.0009 * 0.0009);
  g.rotation = normalize(rr + rotation_offset.unsqueeze(0), 1);
  g.opacity = at::sigmoid(ro + opacity_offset);
  return g;
}

void save_ply(const Gaussians& g, const std::string& path) {
  const double eps = 1e-6;
  at::Tensor xyz = g.xyz.to(at::kCPU).contiguous();
  at::Tensor fdc = g.features_dc.transpose(1, 2).flatten(1).contiguous().to(at::kCPU);
  at::Tensor op = g.opacity.clamp(eps, 1 - eps);
  op = at::log(op / at::rsub(op, 1)).to(at::kCPU).contiguous();
  at::Tensor sc = at::log(g.scaling.clamp_min(eps)).to(at::kCPU).contiguous();
  at::Tensor rot = g.rotation.to(at::kCPU).contiguous();
  const int64_t N = xyz.size(0);
  const float* X = xyz.data_ptr<float>();
  const float* D = fdc.data_ptr<float>();
  const float* O = op.data_ptr<float>();
  const float* S = sc.data_ptr<float>();
  const float* R = rot.data_ptr<float>();
  std::ostringstream h;
  h << "ply\nformat binary_little_endian 1.0\nelement vertex " << N << "\n";
  const char* names[] = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
  for (auto n : names) h << "property float " << n << "\n";
  h << "end_header\n";
  const std::string header = h.str();
  std::vector<float> buf(static_cast<size_t>(N) * 17);
  const double aw = std::sqrt(0.5), ax = std::sqrt(0.5), ay = 0.0, az = 0.0;
  for (int64_t i = 0; i < N; i++) {
    float* o = &buf[i * 17];
    o[0] = X[i * 3 + 0];
    o[1] = -X[i * 3 + 2];
    o[2] = X[i * 3 + 1];
    o[3] = o[4] = o[5] = 0.0f;
    o[6] = D[i * 3 + 0];
    o[7] = D[i * 3 + 1];
    o[8] = D[i * 3 + 2];
    o[9] = O[i];
    o[10] = S[i * 3 + 0];
    o[11] = S[i * 3 + 1];
    o[12] = S[i * 3 + 2];
    const double bw = R[i * 4 + 0], bx = R[i * 4 + 1], by = R[i * 4 + 2], bz = R[i * 4 + 3];
    // numpy float64 elementwise, left to right, no contraction
    volatile double t;
    double w = aw * bw;
    t = ax * bx; w = w - t;
    t = ay * by; w = w - t;
    t = az * bz; w = w - t;
    double x = aw * bx;
    t = ax * bw; x = x + t;
    t = ay * bz; x = x + t;
    t = az * by; x = x - t;
    double y = aw * by;
    t = ax * bz; y = y - t;
    t = ay * bw; y = y + t;
    t = az * bx; y = y + t;
    double z = aw * bz;
    t = ax * by; z = z + t;
    t = ay * bx; z = z - t;
    t = az * bw; z = z + t;
    o[13] = static_cast<float>(w);
    o[14] = static_cast<float>(x);
    o[15] = static_cast<float>(y);
    o[16] = static_cast<float>(z);
  }
  std::ofstream f(path, std::ios::binary);
  f.write(header.data(), header.size());
  f.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(float));
  if (!f) throw std::runtime_error("failed writing " + path);
}

// Python repr(float)
static std::string py_repr(double v) {
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
  std::string s(buf, res.ptr);
  // split mantissa digits and exponent
  const size_t e = s.find('e');
  std::string mant = s.substr(0, e);
  const int exp10 = std::stoi(s.substr(e + 1));
  bool neg = false;
  if (mant[0] == '-') {
    neg = true;
    mant = mant.substr(1);
  }
  std::string digits;
  for (char c : mant)
    if (c != '.') digits += c;
  std::string out;
  if (exp10 < -4 || exp10 >= 16) {
    out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    out += eb;
  } else if (exp10 >= 0) {
    if (static_cast<int>(digits.size()) <= exp10 + 1) {
      out = digits + std::string(exp10 + 1 - digits.size(), '0') + ".0";
    } else {
      out = digits.substr(0, exp10 + 1) + "." + digits.substr(exp10 + 1);
    }
  } else {
    out = "0." + std::string(-exp10 - 1, '0') + digits;
  }
  return (neg ? "-" : "") + out;
}

std::string camera_json(const at::Tensor& cl) {
  at::Tensor direction = cl.slice(0, 0, 3).to(at::kFloat);
  const double inverse_distance = cl[3].item<double>();
  const double half_width = cl[4].item<double>();
  const double norm = at::linalg_vector_norm(direction, 2, c10::nullopt, false, c10::nullopt).item<double>();
  double dx = 0, dy = 0, dz = 1;
  if (norm > 1e-6) {
    at::Tensor d = (direction / norm).to(at::kCPU);
    dx = d[0].item<double>();
    dy = d[1].item<double>();
    dz = d[2].item<double>();
  }
  const double radToDeg = 180.0 / 3.141592653589793238462643383279502884;
  double radius, fov;
  if (inverse_distance < 1e-3) {
    radius = 1000.0;
    fov = (2 * std::atan(half_width / radius)) * radToDeg;
  } else {
    radius = 1.0 / inverse_distance;
    fov = (2 * std::atan(half_width * inverse_distance)) * radToDeg;
  }
  const double yaw = std::atan2(dx, dy);
  const double pitch = std::asin(std::min(std::max(dz, -1.0), 1.0));
  return "{\n  \"yaw\": " + py_repr(yaw) + ",\n  \"pitch\": " + py_repr(pitch) + ",\n  \"radius\": " +
         py_repr(radius) + ",\n  \"fov\": " + py_repr(fov) + "\n}";
}

}  // namespace idol

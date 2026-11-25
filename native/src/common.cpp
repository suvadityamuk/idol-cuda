#include "common.h"

#include <cmath>
#include <stdexcept>

namespace idol {

Recorder REC;

at::Tensor W(const Weights& w, const std::string& key) {
  auto it = w.find(key);
  if (it == w.end()) throw std::runtime_error("missing weight: " + key);
  return it->second;
}

Linear::Linear(const Weights& ws, const std::string& prefix, bool bias) {
  w = W(ws, prefix + ".weight");
  if (bias) b = W(ws, prefix + ".bias");
}

LayerNorm32::LayerNorm32(const Weights& ws, const std::string& prefix, int64_t dim_, double eps_, bool affine)
    : eps(eps_), dim(dim_) {
  if (affine) {
    w = W(ws, prefix + ".weight");
    b = W(ws, prefix + ".bias");
  }
}

at::Tensor LayerNorm32::operator()(const at::Tensor& x) const {
  // nn.LayerNorm.forward -> F.layer_norm -> torch.layer_norm(input, shape, weight, bias, eps, cudnn_enable)
  at::Tensor y = at::layer_norm(x.to(at::kFloat), {dim}, w.defined() ? w : at::Tensor(),
                                b.defined() ? b : at::Tensor(), eps);
  return y.to(x.scalar_type());
}

at::Tensor normalize(const at::Tensor& x, int64_t dim) {
  // F.normalize: denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input); input / denom
  // Tensor.norm -> torch.norm -> torch.linalg.vector_norm(input, 2.0, [dim], keepdim=True)
  at::Tensor denom = at::linalg_vector_norm(x, 2.0, at::IntArrayRef{dim}, true, c10::nullopt);
  denom = denom.clamp_min(1e-12).expand_as(x);
  return x / denom;
}

MultiHeadRMSNorm::MultiHeadRMSNorm(const Weights& ws, const std::string& prefix, int64_t head_dim) {
  gamma = W(ws, prefix + ".gamma");
  scale = std::pow(static_cast<double>(head_dim), 0.5);
}

at::Tensor MultiHeadRMSNorm::operator()(const at::Tensor& x) const {
  at::Tensor y = normalize(x.to(at::kFloat), -1) * gamma;
  y = y * scale;
  return y.to(x.scalar_type());
}

at::Tensor layer_norm_plain(const at::Tensor& x) {
  return at::layer_norm(x, {x.size(-1)}, at::Tensor(), at::Tensor(), 1e-5);
}

at::Tensor sdpa_nlhc(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  at::Tensor o = at::scaled_dot_product_attention(q.permute({0, 2, 1, 3}), k.permute({0, 2, 1, 3}),
                                                  v.permute({0, 2, 1, 3}));
  return o.permute({0, 2, 1, 3});
}

SinusoidalEmbedder::SinusoidalEmbedder(const Weights& ws, const std::string& prefix, double max_period_, bool two_pi_,
                                       at::Device dev)
    : max_period(max_period_), two_pi(two_pi_) {
  l0 = Linear(ws, prefix + ".mlp.0");
  l2 = Linear(ws, prefix + ".mlp.2");
  const int64_t half = freq_size / 2;
  // torch.exp(-np.log(max_period) * torch.arange(0, half, dtype=float32) / half), on the CPU
  at::Tensor ar = at::arange(0, half, at::TensorOptions().dtype(at::kFloat));
  at::Tensor f = at::mul(ar, at::Scalar(-std::log(max_period)));
  f = at::div(f, at::Scalar(half));
  freqs = at::exp(f).to(dev);
}

at::Tensor SinusoidalEmbedder::operator()(const at::Tensor& t) const {
  at::Tensor args = t.unsqueeze(1).to(at::kFloat) * freqs.unsqueeze(0);
  if (two_pi) {
    args = args * 2;
    args = args * M_PI;
  }
  at::Tensor e = at::cat({at::cos(args), at::sin(args)}, -1);
  return l2(at::silu(l0(e)));
}

}  // namespace idol

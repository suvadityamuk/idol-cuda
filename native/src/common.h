// Shared helpers for the native Idol pipeline (libtorch C++, no Python).
#pragma once
#include <ATen/ATen.h>

#include <map>
#include <string>

namespace idol {

// Checkpoint recorder: same key names as tools/ref_run.py so tools/compare.py can diff the two dumps.
struct Recorder {
  bool on = false;
  std::map<std::string, at::Tensor> m;
  std::map<std::string, int> calls;
  // returns the call index for `name` and increments it
  int next(const std::string& name) { return calls[name]++; }
  void put(const std::string& key, const at::Tensor& t) {
    if (on) m[key] = t.detach().to(at::kCPU).clone().contiguous();
  }
};
extern Recorder REC;

// Model weights: CPU tensors loaded from the reference checkpoints, then cast/moved like the reference does.
using Weights = std::map<std::string, at::Tensor>;

at::Tensor W(const Weights& w, const std::string& key);  // throws with the key name if missing

// nn.Linear (F.linear -> at::linear): bias optional
struct Linear {
  at::Tensor w, b;
  Linear() = default;
  Linear(const Weights& ws, const std::string& prefix, bool bias = true);
  at::Tensor operator()(const at::Tensor& x) const { return at::linear(x, w, b.defined() ? b : at::Tensor()); }
};

// LayerNorm32: layer_norm(x.float()).type(x.dtype); weight/bias optional
struct LayerNorm32 {
  at::Tensor w, b;
  double eps = 1e-5;
  int64_t dim = 0;
  LayerNorm32() = default;
  LayerNorm32(const Weights& ws, const std::string& prefix, int64_t dim, double eps, bool affine);
  at::Tensor operator()(const at::Tensor& x) const;
};

// MultiHeadRMSNorm: (F.normalize(x.float(), dim=-1) * gamma * scale).to(x.dtype)
struct MultiHeadRMSNorm {
  at::Tensor gamma;
  double scale = 8.0;
  MultiHeadRMSNorm() = default;
  MultiHeadRMSNorm(const Weights& ws, const std::string& prefix, int64_t head_dim);
  at::Tensor operator()(const at::Tensor& x) const;
};

// F.normalize(x, p=2, dim, eps=1e-12) exactly as torch 2.10 computes it
at::Tensor normalize(const at::Tensor& x, int64_t dim);

// torch.nn.functional.layer_norm(x, x.shape[-1:]) without affine, eps 1e-5
at::Tensor layer_norm_plain(const at::Tensor& x);

// The reference's attention entry point in ATTN_BACKEND=sdpa mode, q/k/v [N, L, H, C] -> [N, L, H, C]
at::Tensor sdpa_nlhc(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);

// SinusoidalEmbedder (frequencies computed on the CPU as the reference does, then cached)
struct SinusoidalEmbedder {
  Linear l0, l2;
  int64_t freq_size = 256;
  double max_period = 10000;
  bool two_pi = false;
  at::Tensor freqs;  // cached on the device
  SinusoidalEmbedder() = default;
  SinusoidalEmbedder(const Weights& ws, const std::string& prefix, double max_period, bool two_pi, at::Device dev);
  at::Tensor operator()(const at::Tensor& t) const;
};

}  // namespace idol

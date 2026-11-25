// idol/models/decoder.py, idol/gaussian.py and idol/camera.py ported op for op.
#pragma once
#include <functional>
#include <vector>

#include "common.h"
#include "dit.h"

namespace idol {

// Source of the pipeline's random numbers: the libtorch generator (same Philox stream as the reference after
// manual_seed), or a stage-1 dump (replay). Every draw is recorded as rng.NNN.<name> when REC.on.
struct Rng {
  const Weights* replay = nullptr;
  int pos = 0;
  at::Tensor randn(at::IntArrayRef shape, at::TensorOptions opt);
  at::Tensor rand(at::IntArrayRef shape, at::TensorOptions opt);
  at::Tensor rand_like(const at::Tensor& like, at::ScalarType dtype);
 private:
  at::Tensor take(const char* name, at::Tensor fresh);
};

struct MHA {  // MultiHeadAttention (decoders): RMS-normalised q/k
  bool cross = false;
  int64_t heads = 16;
  Linear to_q, to_kv, to_qkv, to_out;
  MultiHeadRMSNorm qn, kn;
  MHA() = default;
  MHA(const Weights& ws, const std::string& p, bool cross);
  at::Tensor operator()(const at::Tensor& x, const at::Tensor& ctx = at::Tensor()) const;
};

struct OctreeBlock {
  LayerNorm32 norm1, norm2;
  MHA cross_attn;
  FFN mlp;
  OctreeBlock(const Weights& ws, const std::string& p);
  at::Tensor operator()(const at::Tensor& x, const at::Tensor& mod, const at::Tensor& ctx) const;
};

struct GsBlock {
  LayerNorm32 norm1, norm2, norm3;
  MHA self_attn, cross_attn;
  FFN mlp;
  GsBlock(const Weights& ws, const std::string& p);
  at::Tensor operator()(const at::Tensor& x, const at::Tensor& ctx) const;
};

// PcdAbsolutePositionEmbedderv2
struct PosEmbV2 {
  at::Tensor freqs;
  int64_t channels = 1024;
  PosEmbV2() = default;
  PosEmbV2(at::Device dev, int64_t channels);
  at::Tensor operator()(const at::Tensor& x) const;
};

struct OctreeDecoder {
  Linear in_proj, input_layer, adaLN, out_proj;
  PosEmbV2 pos;
  SinusoidalEmbedder l_embedder;
  std::vector<OctreeBlock> blocks;
  OctreeDecoder(const Weights& ws, at::Device dev);
  at::Tensor operator()(const at::Tensor& x, const at::Tensor& level, const at::Tensor& cond) const;
};

struct GsDecoder {
  Linear in_proj, input_layer, out_proj;
  PosEmbV2 pos;
  std::vector<GsBlock> blocks;
  at::Tensor perturb, base_offset_scale;
  GsDecoder(const Weights& ws, at::Device dev);
  at::Tensor operator()(const at::Tensor& points, const at::Tensor& cond) const;
};

struct Gaussians {
  at::Tensor xyz, features_dc, scaling, rotation, opacity;
};

// OctreeProbabilityFixedlenDecoder.sample
at::Tensor octree_sample(const OctreeDecoder& m, const at::Tensor& cond, int64_t num_points, int level, Rng& rng);
// GaussianFixedlenDecoder.to_representation(...)[0] + Gaussian.__init__
Gaussians to_gaussians(const GsDecoder& m, const at::Tensor& points, const at::Tensor& features);
// Gaussian.save_ply (plyfile binary little endian, identical bytes)
void save_ply(const Gaussians& g, const std::string& path);
// decode_camera + json.dump(indent=2), identical bytes
std::string camera_json(const at::Tensor& camera_latent);

}  // namespace idol

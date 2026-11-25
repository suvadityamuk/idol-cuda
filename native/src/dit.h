// The denoiser (idol/models/dit.py LatentSeqMMFlowModel) ported op for op.
#pragma once
#include <vector>

#include "common.h"

namespace idol {

struct RePo {
  LayerNorm32 norm;
  Linear gate_map, content_map, final_map;
  at::Tensor f0, f1, f2;
  int64_t heads = 16;
  RePo() = default;
  RePo(const Weights& ws, const std::string& p);
  at::Tensor operator()(const at::Tensor& h) const;  // complex64 [B, L, H, 32]
};

struct RopeAttention {
  Linear qkv, out;
  MultiHeadRMSNorm qn, kn;
  int64_t heads = 16, head_dim = 64;
  RopeAttention() = default;
  RopeAttention(const Weights& ws, const std::string& p);
  at::Tensor operator()(const at::Tensor& x, const at::Tensor& rope, const std::string& rec = "") const;
};

struct FFN {
  Linear l0, l2;
  FFN() = default;
  FFN(const Weights& ws, const std::string& p) : l0(ws, p + ".mlp.0"), l2(ws, p + ".mlp.2") {}
  at::Tensor operator()(const at::Tensor& x) const { return l2(at::gelu(l0(x), "tanh")); }
};

struct UnifiedBlock {
  bool modulation = true;
  LayerNorm32 norm1, norm2;
  RopeAttention attn;
  FFN mlp;
  at::Tensor shift_table;
  UnifiedBlock() = default;
  UnifiedBlock(const Weights& ws, const std::string& p, bool modulation);
  at::Tensor operator()(const at::Tensor& x, const at::Tensor& mod, const at::Tensor& rope,
                        const std::string& rec = "") const;
};

at::Tensor apply_rotary_emb(const at::Tensor& x, const at::Tensor& freqs);

struct Denoiser {
  int64_t q_len = 8192, in_ch = 16, cam_ch = 5, C = 1024;
  SinusoidalEmbedder t_embedder;
  Linear adaLN, input_layer, cond_embedder, cond_embedder2, cam0, cam2, out_layer, cam_out_layer;
  at::Tensor pos_embed;  // pos_embedder(pos_pe).type(fp16), input independent: computed once like the reference
  at::Tensor shift_table;
  std::vector<RePo> noise_repo, context_repo, repo;
  std::vector<UnifiedBlock> noise_refiner, context_refiner, blocks;

  Denoiser(const Weights& ws, at::Device dev);
  // The forward pass split into its independent parts (bit-identical to operator()):
  // context stream (depends only on the image features), the per-step prefix (time embedding, noise refiner,
  // camera refiner: identical for both CFG passes), and the joint blocks + heads.
  struct Prefix {
    at::Tensor t_emb, t_mod, h_x, h_cam;
  };
  at::Tensor context(const at::Tensor& f1, const at::Tensor& f2, bool rec = false) const;
  Prefix prefix(const at::Tensor& z, const at::Tensor& cam, const at::Tensor& t, bool rec = false) const;
  std::pair<at::Tensor, at::Tensor> joint(const Prefix& P, const at::Tensor& h_cond, const at::Tensor& z,
                                          const at::Tensor& cam, bool rec = false) const;
  // x_latent [B,8192,16] fp32, x_cam [B,1,5] fp32, t [B] fp32, f1 [B,Lc,1280], f2 [B,Lc,128]
  std::pair<at::Tensor, at::Tensor> operator()(const at::Tensor& z, const at::Tensor& cam, const at::Tensor& t,
                                               const at::Tensor& f1, const at::Tensor& f2) const;
};

}  // namespace idol

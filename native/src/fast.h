// The fast build (--build fast): C++ port of fast/model.py + fast/pipeline.py. See fast.cpp and docs/ab-log.md.
#pragma once
#include "decoders.h"
#include "dit.h"

namespace idol {

struct FastPipeline {
  const Denoiser& dn;
  const OctreeDecoder& oct;
  const GsDecoder& gs;
  at::Device dev;
  FastPipeline(const Denoiser& d, const OctreeDecoder& o, const GsDecoder& g, at::Device dev);
  ~FastPipeline();
  // context stream for cond and uncond (collapse: one uncond context token of multiplicity L_c)
  void prepare(const at::Tensor& f1, const at::Tensor& f2, bool collapse);
  // guided velocity (both=false) or v_cond with v_uncond in *uncond (both=true)
  std::pair<at::Tensor, at::Tensor> velocity(const at::Tensor& z, const at::Tensor& cam, const at::Tensor& t, double cfg,
                                             bool both = false, std::pair<at::Tensor, at::Tensor>* uncond = nullptr);
  // one reference-style denoiser call (teacher-forced checks): uncond when f1 is all zeros
  std::pair<at::Tensor, at::Tensor> denoise(const at::Tensor& z, const at::Tensor& cam, const at::Tensor& t,
                                            const at::Tensor& f1, const at::Tensor& f2);
  std::pair<at::Tensor, at::Tensor> sample(at::Tensor z, at::Tensor cam, const at::Tensor& f1, const at::Tensor& f2,
                                           int64_t steps, double rescale_t, double cfg);
  at::Tensor octree_sample(const at::Tensor& cond, int64_t num_points, int level, Rng& rng);
  at::Tensor gsdec(const at::Tensor& points, const at::Tensor& cond);
  // Gaussian decoder head -> PLY rows [N, 17] (y-up columns of the reference PLY)
  at::Tensor ply_rows(const at::Tensor& anchors, const at::Tensor& feats);
  // FlashAttention-3 from the Hub kernel's shared library (kernels-community/flash-attn3); false: cuDNN attention
  static bool use_fa3(const std::string& so_path);
  struct Impl;
  Impl* impl = nullptr;

 private:
  bool collapse_ = true;
  at::Tensor last_f1_;
};

void write_ply_rows(const at::Tensor& rows, const std::string& path);

}  // namespace idol

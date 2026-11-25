// Image networks of the reference (BiRefNet, DINOv3 ViT-H+/16, FLUX.2 VAE encoder) run as TorchScript traces
// exported once by tools/export_encoders.py, plus the hand-ported glue around them (preprocess.py,
// BiRefNet.remove_background, ImageEncoder._flux2_tokens sampling/patchify/normalisation).
#pragma once
#include <torch/script.h>

#include "common.h"
#include "decoders.h"
#include "image.h"

namespace idol {

struct Encoders {
  mutable torch::jit::Module birefnet, dinov3, vae_enc;
  at::Tensor bn_mean, bn_var;
  double bn_eps = 1e-4;
  at::Device dev;
  explicit Encoders(const std::string& ts_dir, at::Device dev);

  // BiRefNet.remove_background: RGB image -> RGBA image with the predicted alpha
  Img remove_background(const Img& rgb) const;
  // preprocess_image(image, birefnet, 1024, crop_margin)
  Img preprocess(Img image, double crop_margin, int image_size = 1024) const;
  // ImageEncoder.__call__ -> (feature1, feature2); consumes one randn draw for the VAE posterior sample
  std::pair<at::Tensor, at::Tensor> encode(const at::Tensor& pixels, Rng& rng) const;
};

// torch.from_numpy(np.array(rgb)).permute(2,0,1).float().div(255.0)[None].to(device)
at::Tensor image_to_pixels(const Img& rgb, at::Device dev);

// Must be called once before running any TorchScript module: disables the JIT optimiser/profiling executor so the
// traced graphs run op by op exactly like eager mode.
void configure_jit();

}  // namespace idol

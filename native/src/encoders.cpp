#include "encoders.h"

#include <torch/csrc/jit/runtime/graph_executor.h>

#include <cmath>
#include <cstring>

#include "safetensors.h"
#include <stdexcept>

namespace idol {

void configure_jit() {
  torch::jit::getProfilingMode() = false;
  torch::jit::getExecutorMode() = false;
  torch::jit::setGraphExecutorOptimize(false);
}

Encoders::Encoders(const std::string& ts, at::Device d) : dev(d) {
  birefnet = torch::jit::load(ts + "/birefnet.pt", dev);
  dinov3 = torch::jit::load(ts + "/dinov3.pt", dev);
  vae_enc = torch::jit::load(ts + "/vae_enc.pt", dev);
  auto bn = load_safetensors(ts + "/vae_bn.safetensors");
  bn_mean = bn.at("running_mean").to(dev);  // bf16, as AutoencoderKLFlux2(dtype=bf16) holds them
  bn_var = bn.at("running_var").to(dev);
  bn_eps = 0.0001;                          // vae.config.batch_norm_eps
  birefnet.eval();
  dinov3.eval();
  vae_enc.eval();
}

Img Encoders::remove_background(const Img& image) const {
  const int W = image.w, H = image.h;
  // arr = np.array(image, dtype=np.float32) / 255.0 ; t = from_numpy(arr).permute(2,0,1).unsqueeze(0)
  at::Tensor arr = at::empty({H, W, 3}, at::TensorOptions().dtype(at::kFloat));
  float* a = arr.data_ptr<float>();
  for (int64_t i = 0; i < static_cast<int64_t>(W) * H; i++)
    for (int c = 0; c < 3; c++) a[i * 3 + c] = static_cast<float>(image.px[i * 4 + c]) / 255.0f;
  at::Tensor t = arr.permute({2, 0, 1}).unsqueeze(0);
  t = at::upsample_bilinear2d(t, {1024, 1024}, true);
  at::Tensor mean = at::tensor({0.485f, 0.456f, 0.406f}, at::TensorOptions().dtype(at::kFloat)).view({1, 3, 1, 1});
  at::Tensor std = at::tensor({0.229f, 0.224f, 0.225f}, at::TensorOptions().dtype(at::kFloat)).view({1, 3, 1, 1});
  t = ((t - mean) / std).to(dev, at::kHalf);
  REC.put("birefnet#0.in.0", t);
  at::Tensor alpha = birefnet.forward({t}).toTensor();
  REC.put("birefnet#0.out", alpha);
  alpha = at::upsample_bilinear2d(alpha.to(at::kFloat), {H, W}, true).select(0, 0).select(0, 0);
  at::Tensor u8 = (alpha.clamp(0, 1) * 255).to(at::kByte).to(at::kCPU).contiguous();
  Img rgba = image;
  Img al = new_image("L", W, H);
  std::memcpy(al.px.data(), u8.data_ptr<uint8_t>(), al.px.size());
  putalpha(rgba, al);
  return rgba;
}

Img Encoders::preprocess(Img image, double crop_margin, int image_size) const {
  const int width = image.w, height = image.h;
  const double scale = static_cast<double>(image_size) / std::min(width, height);
  // max(1, int(round(width * scale))): Python round() is round-half-even
  const int nw = std::max(1, static_cast<int>(std::nearbyint(width * scale)));
  const int nh = std::max(1, static_cast<int>(std::nearbyint(height * scale)));
  image = resize_lanczos(image, nw, nh);
  if (!has_real_alpha(image)) image = remove_background(image.mode == "RGB" ? image : convert(image, "RGB"));
  putalpha(image, min_filter3(getchannel(image, 3)));
  Img alpha = getchannel(image, 3);
  int left = INT32_MAX, top = INT32_MAX, right = -1, bottom = -1;
  for (int y = 0; y < alpha.h; y++)
    for (int x = 0; x < alpha.w; x++)
      if (alpha.row(y)[x]) {
        left = std::min(left, x);
        right = std::max(right, x);
        top = std::min(top, y);
        bottom = std::max(bottom, y);
      }
  if (right < 0) throw std::runtime_error("no object found in the image (the alpha channel is empty)");
  const double cx = (left + right) / 2.0, cy = (top + bottom) / 2.0;
  const double half = std::max(right - left, bottom - top) / 2.0 * crop_margin;
  // int() truncates toward zero
  image = crop(image, static_cast<int>(cx - half), static_cast<int>(cy - half), static_cast<int>(cx + half),
               static_cast<int>(cy + half));
  image = resize_lanczos(image, image_size, image_size);
  Img composite = new_image("RGB", image_size, image_size, 0);
  // Image.new('RGB', ..., (0,0,0)) stores the padding byte as 255 (unused by np.array(prepared))
  for (size_t i = 0; i < composite.px.size(); i += 4) composite.px[i + 3] = 255;
  paste_with_mask(composite, image, getchannel(image, 3));
  return composite;
}

at::Tensor image_to_pixels(const Img& rgb, at::Device dev) {
  at::Tensor u8 = at::empty({rgb.h, rgb.w, 3}, at::TensorOptions().dtype(at::kByte));
  uint8_t* p = u8.data_ptr<uint8_t>();
  for (int64_t i = 0; i < static_cast<int64_t>(rgb.w) * rgb.h; i++)
    for (int c = 0; c < 3; c++) p[i * 3 + c] = rgb.px[i * 4 + c];
  return u8.permute({2, 0, 1}).to(at::kFloat).div(255.0).unsqueeze(0).to(dev);
}

std::pair<at::Tensor, at::Tensor> Encoders::encode(const at::Tensor& pixels, Rng& rng) const {
  at::Tensor feature1 = dinov3.forward({pixels}).toTensor();
  // _flux2_tokens: vae.encode(images.to(bf16) * 2 - 1).latent_dist.sample()
  at::Tensor moments = vae_enc.forward({pixels}).toTensor();
  auto mv = at::chunk(moments, 2, 1);
  at::Tensor mean = mv[0];
  at::Tensor logvar = at::clamp(mv[1], -30.0, 20.0);
  at::Tensor std = at::exp(logvar * 0.5);
  at::Tensor noise = rng.randn(mean.sizes(), moments.options());
  at::Tensor lat = mean + std * noise;
  const int64_t b = lat.size(0), c = lat.size(1), h = lat.size(2), w = lat.size(3);
  lat = lat.view({b, c, h / 2, 2, w / 2, 2}).permute({0, 1, 3, 5, 2, 4});
  lat = lat.reshape({b, c * 4, h / 2, w / 2});
  at::Tensor mu = bn_mean.view({1, -1, 1, 1}).to(lat.device(), lat.scalar_type());
  at::Tensor sd = at::sqrt(bn_var.view({1, -1, 1, 1}) + bn_eps).to(lat.device(), lat.scalar_type());
  at::Tensor tokens = ((lat - mu) / sd).to(at::kFloat).flatten(2).transpose(1, 2).contiguous();
  at::Tensor padding = at::zeros({b, 5, tokens.size(2)}, tokens.options());
  at::Tensor feature2 = at::cat({padding, tokens}, 1);
  return {feature1, feature2};
}

}  // namespace idol

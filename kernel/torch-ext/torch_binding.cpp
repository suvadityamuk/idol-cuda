#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

namespace {
void add_ln_mod_op(at::Tensor& y_out, const std::optional<at::Tensor>& x_out, const at::Tensor& x,
                   const std::optional<at::Tensor>& h, const std::optional<at::Tensor>& gate,
                   const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
                   const std::optional<at::Tensor>& scale, const std::optional<at::Tensor>& shift,
                   int64_t rows_per_batch, double eps, const std::optional<at::Tensor>& y2_out,
                   const std::optional<at::Tensor>& w2, const std::optional<at::Tensor>& b2, double eps2) {
  idol_cuda::add_ln_mod(y_out, x_out, x, h, gate, weight, bias, scale, shift, rows_per_batch, eps, y2_out, w2, b2,
                        eps2);
}
void rope_rms_qk_op(at::Tensor& q_out, at::Tensor& k_out, const at::Tensor& qkv, const at::Tensor& delta,
                    const at::Tensor& freqs, const at::Tensor& gamma_q, const at::Tensor& gamma_k) {
  idol_cuda::rope_rms_qk(q_out, k_out, qkv, delta, freqs, gamma_q, gamma_k);
}
void rms_heads_op(at::Tensor& out, const at::Tensor& x, const at::Tensor& gamma) { idol_cuda::rms_heads(out, x, gamma); }
void merge_extra_key_op(at::Tensor& out, const at::Tensor& o, const at::Tensor& lse, const at::Tensor& q,
                        const at::Tensor& k_extra, const at::Tensor& v_extra, double scale, double log_mult) {
  idol_cuda::merge_extra_key(out, o, lse, q, k_extra, v_extra, scale, log_mult);
}
void gaussians_ply_op(at::Tensor& out, const at::Tensor& feats, const at::Tensor& points, const at::Tensor& perturb,
                      double base_offset_scale, double scaling_offset, double opacity_offset, double min_kernel_size,
                      double lr_rotation, double perturb_size) {
  idol_cuda::gaussians_ply(out, feats, points, perturb, base_offset_scale, scaling_offset, opacity_offset,
                           min_kernel_size, lr_rotation, perturb_size);
}
void linear_gelu_op(at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
  idol_cuda::linear_gelu(out, x, weight, bias);
}
}  // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "add_ln_mod(Tensor! y_out, Tensor(a!)? x_out, Tensor x, Tensor? h, Tensor? gate, Tensor? weight, Tensor? bias, "
      "Tensor? scale, Tensor? shift, int rows_per_batch, float eps, Tensor(b!)? y2_out, Tensor? w2, Tensor? b2, "
      "float eps2) -> ()");
  ops.def("rope_rms_qk(Tensor! q_out, Tensor! k_out, Tensor qkv, Tensor delta, Tensor freqs, Tensor gamma_q, "
          "Tensor gamma_k) -> ()");
  ops.def("rms_heads(Tensor! out, Tensor x, Tensor gamma) -> ()");
  ops.def("merge_extra_key(Tensor! out, Tensor o, Tensor lse, Tensor q, Tensor k_extra, Tensor v_extra, "
          "float scale, float log_mult) -> ()");
  ops.def("gaussians_ply(Tensor! out, Tensor feats, Tensor points, Tensor perturb, float base_offset_scale, "
          "float scaling_offset, float opacity_offset, float min_kernel_size, float lr_rotation, "
          "float perturb_size) -> ()");
  ops.def("linear_gelu(Tensor! out, Tensor x, Tensor weight, Tensor bias) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("add_ln_mod", c10::DeviceType::CUDA, &add_ln_mod_op);
  ops.impl("rope_rms_qk", c10::DeviceType::CUDA, &rope_rms_qk_op);
  ops.impl("rms_heads", c10::DeviceType::CUDA, &rms_heads_op);
  ops.impl("merge_extra_key", c10::DeviceType::CUDA, &merge_extra_key_op);
  ops.impl("gaussians_ply", c10::DeviceType::CUDA, &gaussians_ply_op);
  ops.impl("linear_gelu", c10::DeviceType::CUDA, &linear_gelu_op);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

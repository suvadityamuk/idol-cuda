// idol_cuda ops (see idol_kernels.cu). All write into caller-provided outputs and validate every argument.
#pragma once
#include <ATen/ATen.h>

#include <optional>

namespace idol_cuda {

// y = fp16 modulated LayerNorm32 of (x [+ h [* gate]]); x_out receives the residual sum when given; y2_out (fp32)
// receives a second, affine LayerNorm of the same sum (eps2, w2, b2) when given.
void add_ln_mod(at::Tensor& y_out, const std::optional<at::Tensor>& x_out, const at::Tensor& x,
                const std::optional<at::Tensor>& h, const std::optional<at::Tensor>& gate,
                const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
                const std::optional<at::Tensor>& scale, const std::optional<at::Tensor>& shift, int64_t rows_per_batch,
                double eps, const std::optional<at::Tensor>& y2_out, const std::optional<at::Tensor>& w2,
                const std::optional<at::Tensor>& b2, double eps2);
// RePo rotary (angles from delta and freqs) + head RMSNorm for q and k of a packed qkv matrix.
void rope_rms_qk(at::Tensor& q_out, at::Tensor& k_out, const at::Tensor& qkv, const at::Tensor& delta,
                 const at::Tensor& freqs, const at::Tensor& gamma_q, const at::Tensor& gamma_k);
// Head RMSNorm (MultiHeadRMSNorm) of x [M, H*64].
void rms_heads(at::Tensor& out, const at::Tensor& x, const at::Tensor& gamma);
// Add one key/value of multiplicity exp(log_mult) to an attention output given its log-sum-exp.
void merge_extra_key(at::Tensor& out, const at::Tensor& o, const at::Tensor& lse, const at::Tensor& q,
                     const at::Tensor& k_extra, const at::Tensor& v_extra, double scale, double log_mult);
// Gaussian decoder head -> PLY rows [A*32, 17].
void gaussians_ply(at::Tensor& out, const at::Tensor& feats, const at::Tensor& points, const at::Tensor& perturb,
                   double base_offset_scale, double scaling_offset, double opacity_offset, double min_kernel_size,
                   double lr_rotation, double perturb_size);
// x @ weight.T + bias, GELU(tanh) in the cuBLASLt epilogue.
void linear_gelu(at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias);

}  // namespace idol_cuda

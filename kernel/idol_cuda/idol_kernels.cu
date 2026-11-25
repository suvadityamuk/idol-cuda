// idol_cuda: fused CUDA kernels for the Idol image-to-3D pipeline (fast build).
//
// Every op validates its arguments with TORCH_CHECK and writes into caller-provided outputs. The kernels keep the
// reference's fp16 rounding points (LayerNorm32 -> fp16, then the fp16 modulation ops; RoPE output rounded to fp16
// before the fp32 RMS norm), so the only differences from the PyTorch op chains are fp32 reduction orders.
//
// Shared by the kernel-builder package (torch-ext/torch_binding.cpp) and the native binary (native/src/fast*.cpp).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cmath>
#include <initializer_list>
#include <utility>
#include <optional>

#include "idol_kernels.h"

namespace idol_cuda {
namespace {

inline std::pair<const at::Tensor*, const char*> TP(const at::Tensor* t, const char* n) { return {t, n}; }


constexpr int kC = 1024;           // model width handled by the row kernels
constexpr int kRowThreads = 128;   // 8 halves per thread
constexpr int kVec = kC / kRowThreads;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sh) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; i++) t += sh[i];
  return t;
}

__device__ __forceinline__ float h2f(__half h) { return __half2float(h); }
__device__ __forceinline__ __half f2h(float f) { return __float2half_rn(f); }

// --------------------------------------------------------------------------------------------------------------------
// add + LayerNorm(fp32) + modulation, one row (C = 1024) per block.
//   xn  = x + h * gate      (each op rounded to fp16, as the eager reference)   [optional: h, gate]
//   y   = fp16(LN32(xn) [* w + b])                                               [optional affine w, b (fp32)]
//   y   = fp16(fp16(y * fp16(1 + scale)) + shift)                                [optional modulation]
// scale/shift/gate are fp16 rows of a [B, ld] tensor; the row's batch is row / rows_per_batch.
// Optionally a second LayerNorm of the same sum with its own eps and fp32 affine, written in fp32 (the RePo layer's
// LayerNorm32 of the next block reads the same input).
struct AddLnArgs {
  const __half* x;
  const __half* h;
  const __half* gate;
  const float* w;
  const float* b;
  const __half* scale;
  const __half* shift;
  __half* x_out;
  __half* y_out;
  const float* w2;
  const float* b2;
  float* y2_out;
  int64_t rows, rows_per_batch, mod_ld;
  float eps, eps2;
};

__global__ void __launch_bounds__(kRowThreads) add_ln_mod_kernel(AddLnArgs a) {
  __shared__ float sh[kRowThreads / 32];
  const int64_t row = blockIdx.x;
  if (row >= a.rows) return;
  const int64_t bi = row / a.rows_per_batch;
  const int c0 = threadIdx.x * kVec;
  const __half* xr = a.x + row * kC + c0;
  float v[kVec];
  {
    uint4 raw = *reinterpret_cast<const uint4*>(xr);
    const __half* hx = reinterpret_cast<const __half*>(&raw);
    if (a.h) {
      uint4 rh = *reinterpret_cast<const uint4*>(a.h + row * kC + c0);
      const __half* hh = reinterpret_cast<const __half*>(&rh);
      __half outv[kVec];
#pragma unroll
      for (int i = 0; i < kVec; i++) {
        __half t = hh[i];
        if (a.gate) t = f2h(h2f(t) * h2f(a.gate[bi * a.mod_ld + c0 + i]));
        outv[i] = f2h(h2f(hx[i]) + h2f(t));
        v[i] = h2f(outv[i]);
      }
      if (a.x_out) *reinterpret_cast<uint4*>(a.x_out + row * kC + c0) = *reinterpret_cast<uint4*>(outv);
    } else {
#pragma unroll
      for (int i = 0; i < kVec; i++) v[i] = h2f(hx[i]);
    }
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < kVec; i++) s += v[i];
  const float mean = block_sum<kRowThreads>(s, sh) / kC;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const float d = v[i] - mean;
    q += d * d;
  }
  const float var = block_sum<kRowThreads>(q, sh) / kC;
  const float rstd = rsqrtf(var + a.eps);
  __half outv[kVec];
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    float y = (v[i] - mean) * rstd;
    if (a.w) y = y * a.w[c0 + i] + a.b[c0 + i];
    __half yh = f2h(y);
    if (a.scale) {
      const __half sc = f2h(1.0f + h2f(a.scale[bi * a.mod_ld + c0 + i]));
      yh = f2h(h2f(yh) * h2f(sc));
      yh = f2h(h2f(yh) + h2f(a.shift[bi * a.mod_ld + c0 + i]));
    }
    outv[i] = yh;
  }
  *reinterpret_cast<uint4*>(a.y_out + row * kC + c0) = *reinterpret_cast<uint4*>(outv);
  if (a.y2_out) {
    const float rstd2 = rsqrtf(var + a.eps2);
    float o2[kVec];
#pragma unroll
    for (int i = 0; i < kVec; i++) o2[i] = (v[i] - mean) * rstd2 * a.w2[c0 + i] + a.b2[c0 + i];
    float4* dst = reinterpret_cast<float4*>(a.y2_out + row * kC + c0);
    dst[0] = make_float4(o2[0], o2[1], o2[2], o2[3]);
    dst[1] = make_float4(o2[4], o2[5], o2[6], o2[7]);
  }
}

// --------------------------------------------------------------------------------------------------------------------
// RePo rotary + RMSNorm for q and k. One warp per (token, head); lane i owns the pair (2i, 2i+1) of the 64 channels.
//   angle_i = clamp_mul(delta[axis(i)], f_i) * pi, clamp_mul(x, f) = x * tanh(f) + x * (f - tanh(f))
//   r = fp16(rotate(x, angle))   then   y = fp16(r / max(||r||, 1e-12) * gamma * 8)
// qkv: [M, 3, H, 64] fp16 (row stride qkv_ld halves); delta: [M, H, 3] fp32 (row stride delta_ld floats).
// freqs: fp32 [32] = cat(freqs_0[10], freqs_1[10], freqs_2[12]). Outputs q, k: [M, H, 64] fp16 contiguous.
__global__ void rope_rms_qk_kernel(const __half* __restrict__ qkv, int64_t qkv_ld, const float* __restrict__ delta,
                                   int64_t delta_ld, const float* __restrict__ freqs, const float* __restrict__ gq,
                                   const float* __restrict__ gk, __half* __restrict__ qo, __half* __restrict__ ko,
                                   int64_t M, int H) {
  const int lane = threadIdx.x & 31;
  const int64_t wid = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  if (wid >= M * H) return;
  const int64_t m = wid / H;
  const int hd = static_cast<int>(wid % H);
  const int axis = lane < 10 ? 0 : (lane < 20 ? 1 : 2);
  const float d = delta[m * delta_ld + hd * 3 + axis];
  const float f = freqs[lane];
  const float ft = tanhf(f);
  const float ang = (d * ft + d * (f - ft)) * static_cast<float>(M_PI);
  float sn, cs;
  sincosf(ang, &sn, &cs);
#pragma unroll
  for (int which = 0; which < 2; which++) {
    const __half2 x2 = reinterpret_cast<const __half2*>(qkv + m * qkv_ld + which * H * 64 + hd * 64)[lane];
    const float x0 = __low2float(x2), x1 = __high2float(x2);
    const float r0 = h2f(f2h(x0 * cs - x1 * sn));
    const float r1 = h2f(f2h(x0 * sn + x1 * cs));
    const float ss = warp_sum(r0 * r0 + r1 * r1);
    const float den = fmaxf(sqrtf(ss), 1e-12f);
    const float* g = which ? gk : gq;
    const float y0 = (r0 / den) * g[hd * 64 + 2 * lane] * 8.0f;
    const float y1 = (r1 / den) * g[hd * 64 + 2 * lane + 1] * 8.0f;
    __half* o = (which ? ko : qo) + (m * H + hd) * 64;
    reinterpret_cast<__half2*>(o)[lane] = __halves2half2(f2h(y0), f2h(y1));
  }
}

// Head RMSNorm (decoders): x [M, H, 64] fp16 (row stride x_ld halves) -> fp16(x/max(||x||,1e-12) * gamma * 8)
__global__ void rms_heads_kernel(const __half* __restrict__ x, int64_t x_ld, const float* __restrict__ g,
                                 __half* __restrict__ out, int64_t M, int H) {
  const int lane = threadIdx.x & 31;
  const int64_t wid = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  if (wid >= M * H) return;
  const int64_t m = wid / H;
  const int hd = static_cast<int>(wid % H);
  const __half2 x2 = reinterpret_cast<const __half2*>(x + m * x_ld + hd * 64)[lane];
  const float r0 = __low2float(x2), r1 = __high2float(x2);
  const float den = fmaxf(sqrtf(warp_sum(r0 * r0 + r1 * r1)), 1e-12f);
  const float y0 = (r0 / den) * g[hd * 64 + 2 * lane] * 8.0f;
  const float y1 = (r1 / den) * g[hd * 64 + 2 * lane + 1] * 8.0f;
  reinterpret_cast<__half2*>(out + (m * H + hd) * 64)[lane] = __halves2half2(f2h(y0), f2h(y1));
}

// --------------------------------------------------------------------------------------------------------------------
// Merge one extra key/value of multiplicity N into an attention output computed over the other keys:
//   s = scale * <q, k_e>,  L = logaddexp(lse, s + log N),  out = exp(lse - L) * o + exp(s + log N - L) * v_e
// o, q: [B, L, H, 64] fp16 (row stride ld halves); lse: [B, H, L] fp32; k_e, v_e: [B, H, 64] fp16; out like o.
__global__ void merge_key_kernel(const __half* __restrict__ o, int64_t o_ld, const float* __restrict__ lse,
                                 const __half* __restrict__ q, int64_t q_ld, const __half* __restrict__ ke,
                                 const __half* __restrict__ ve, __half* __restrict__ out, int64_t out_ld, int64_t Bn,
                                 int64_t L, int H, float scale, float log_mult) {
  const int lane = threadIdx.x & 31;
  const int64_t wid = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  if (wid >= Bn * L * H) return;
  const int hd = static_cast<int>(wid % H);
  const int64_t l = (wid / H) % L;
  const int64_t b = wid / (static_cast<int64_t>(H) * L);
  const int64_t row = b * L + l;
  const __half2 q2 = reinterpret_cast<const __half2*>(q + row * q_ld + hd * 64)[lane];
  const __half2 k2 = reinterpret_cast<const __half2*>(ke + (b * H + hd) * 64)[lane];
  const float s = warp_sum(__low2float(q2) * __low2float(k2) + __high2float(q2) * __high2float(k2)) * scale + log_mult;
  const float l0 = lse[(b * H + hd) * L + l];
  const float mx = fmaxf(l0, s);
  const float e0 = expf(l0 - mx), e1 = expf(s - mx);
  const float inv = 1.0f / (e0 + e1);
  const float w0 = e0 * inv, w1 = e1 * inv;
  const __half2 o2 = reinterpret_cast<const __half2*>(o + row * o_ld + hd * 64)[lane];
  const __half2 v2 = reinterpret_cast<const __half2*>(ve + (b * H + hd) * 64)[lane];
  const float y0 = w0 * __low2float(o2) + w1 * __low2float(v2);
  const float y1 = w0 * __high2float(o2) + w1 * __high2float(v2);
  reinterpret_cast<__half2*>(out + row * out_ld + hd * 64)[lane] = __halves2half2(f2h(y0), f2h(y1));
}

// --------------------------------------------------------------------------------------------------------------------
// GaussianFixedlenDecoder.to_representation + Gaussian.__init__ + the PLY transforms of Gaussian.save_ply, fused.
// feats [A, 480] fp32 (32 Gaussians per anchor), points [A, 3] fp32, perturb [32, 3] fp32 -> rows [A*32, 17] fp32:
// x, y, z, nx, ny, nz, f_dc_0..2, opacity, scale_0..2, rot_0..3 (y-up), exactly the columns of the reference PLY.
struct GsArgs {
  const float* feats;
  const float* points;
  const float* perturb;
  float* out;
  int64_t anchors;
  float base_offset_scale, scaling_offset, opacity_offset, min_kernel2, lr_rot, half_perturb;
};

__device__ __forceinline__ float softplus_t(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__global__ void gaussians_ply_kernel(GsArgs a) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= a.anchors * 32) return;
  const int64_t an = i / 32;
  const int g = static_cast<int>(i % 32);
  const float* f = a.feats + an * 480;
  const float os = softplus_t(f[448 + g] + a.base_offset_scale);
  float xyz[3];
#pragma unroll
  for (int c = 0; c < 3; c++) {
    const float off = tanhf(f[g * 3 + c] * 1.0f + a.perturb[g * 3 + c]) * a.half_perturb * os;
    xyz[c] = (off + a.points[an * 3 + c]) * 1.0f + (-0.5f);
  }
  float* o = a.out + i * 17;
  o[0] = xyz[0];
  o[1] = -xyz[2];
  o[2] = xyz[1];
  o[3] = o[4] = o[5] = 0.f;
#pragma unroll
  for (int c = 0; c < 3; c++) o[6 + c] = f[96 + g * 3 + c] * 1.0f;
  const float op = 1.f / (1.f + expf(-(f[416 + g] * 1.0f + a.opacity_offset)));
  const float opc = fminf(fmaxf(op, 1e-6f), 1.f - 1e-6f);
  o[9] = logf(opc / (1.f - opc));
#pragma unroll
  for (int c = 0; c < 3; c++) {
    const float sp = softplus_t(f[192 + g * 3 + c] * 1.0f + a.scaling_offset);
    const float sc = sqrtf(sp * sp + a.min_kernel2);
    o[10 + c] = logf(fmaxf(sc, 1e-6f));
  }
  float r[4];
#pragma unroll
  for (int c = 0; c < 4; c++) r[c] = f[288 + g * 4 + c] * a.lr_rot + (c == 0 ? 1.f : 0.f);
  const float n = fmaxf(sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]), 1e-12f);
#pragma unroll
  for (int c = 0; c < 4; c++) r[c] /= n;
  // quaternion_multiply([sqrt(.5), sqrt(.5), 0, 0], r) in float64 as numpy does, then float32
  const double s = 0.70710678118654757;
  const double bw = r[0], bx = r[1], by = r[2], bz = r[3];
  o[13] = static_cast<float>(s * bw - s * bx);
  o[14] = static_cast<float>(s * bx + s * bw);
  o[15] = static_cast<float>(s * by - s * bz);
  o[16] = static_cast<float>(s * bz + s * by);
}

void check_cuda_half(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be float16, got ", t.scalar_type());
}
void check_cuda_float(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32, got ", t.scalar_type());
}
void check_same_device(const at::Tensor& a, const at::Tensor& b, const char* name) {
  TORCH_CHECK(a.device() == b.device(), name, " is on ", b.device(), " but the input is on ", a.device());
}
// [rows, C] view with unit inner stride and 16-byte aligned rows
void check_rows(const at::Tensor& t, const char* name, int64_t C) {
  TORCH_CHECK(t.dim() >= 1 && t.size(-1) == C, name, " must have last dimension ", C, ", got ", t.sizes());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}
unsigned blocks_for(int64_t threads, int per_block) { return static_cast<unsigned>((threads + per_block - 1) / per_block); }

}  // namespace

// --------------------------------------------------------------------------------------------------------------------
void add_ln_mod(at::Tensor& y_out, const std::optional<at::Tensor>& x_out, const at::Tensor& x,
                const std::optional<at::Tensor>& h, const std::optional<at::Tensor>& gate,
                const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
                const std::optional<at::Tensor>& scale, const std::optional<at::Tensor>& shift, int64_t rows_per_batch,
                double eps, const std::optional<at::Tensor>& y2_out, const std::optional<at::Tensor>& w2,
                const std::optional<at::Tensor>& b2, double eps2) {
  check_cuda_half(x, "x");
  check_rows(x, "x", kC);
  check_cuda_half(y_out, "y_out");
  check_rows(y_out, "y_out", kC);
  const int64_t rows = x.numel() / kC;
  TORCH_CHECK(y_out.numel() == x.numel(), "y_out must have as many elements as x");
  TORCH_CHECK(eps > 0, "eps must be positive");
  TORCH_CHECK(rows_per_batch > 0, "rows_per_batch must be positive");
  TORCH_CHECK(!gate.has_value() || h.has_value(), "gate needs h");
  TORCH_CHECK(!x_out.has_value() || h.has_value(), "x_out needs h");
  TORCH_CHECK(weight.has_value() == bias.has_value(), "weight and bias must be given together");
  TORCH_CHECK(scale.has_value() == shift.has_value(), "scale and shift must be given together");
  const int64_t nb = (rows + rows_per_batch - 1) / rows_per_batch;
  int64_t mod_ld = kC;
  auto check_mod = [&](const std::optional<at::Tensor>& t, const char* name) {
    if (!t) return;
    check_cuda_half(*t, name);
    check_same_device(x, *t, name);
    TORCH_CHECK(t->dim() == 2 && t->size(1) == kC && t->stride(1) == 1, name, " must be [B, 1024] with unit stride");
    TORCH_CHECK(t->size(0) >= nb, name, " has ", t->size(0), " rows but ", nb, " batches are addressed");
  };
  check_mod(gate, "gate");
  check_mod(scale, "scale");
  check_mod(shift, "shift");
  if (gate && scale) TORCH_CHECK(gate->stride(0) == scale->stride(0), "gate/scale/shift must share a row stride");
  if (scale) {
    TORCH_CHECK(scale->stride(0) == shift->stride(0), "scale and shift must share a row stride");
    mod_ld = scale->stride(0);
  } else if (gate) {
    mod_ld = gate->stride(0);
  }
  if (h) {
    check_cuda_half(*h, "h");
    check_rows(*h, "h", kC);
    check_same_device(x, *h, "h");
    TORCH_CHECK(h->numel() == x.numel(), "h must match x");
  }
  if (x_out) {
    check_cuda_half(*x_out, "x_out");
    check_rows(*x_out, "x_out", kC);
    TORCH_CHECK(x_out->numel() == x.numel(), "x_out must match x");
  }
  if (weight) {
    check_cuda_float(*weight, "weight");
    check_cuda_float(*bias, "bias");
    TORCH_CHECK(weight->numel() == kC && bias->numel() == kC && weight->is_contiguous() && bias->is_contiguous(),
                "weight/bias must be contiguous [1024]");
  }
  TORCH_CHECK(y2_out.has_value() == w2.has_value() && w2.has_value() == b2.has_value(),
              "y2_out, w2 and b2 must be given together");
  if (y2_out) {
    check_cuda_float(*y2_out, "y2_out");
    check_rows(*y2_out, "y2_out", kC);
    TORCH_CHECK(y2_out->numel() == x.numel(), "y2_out must match x");
    check_cuda_float(*w2, "w2");
    check_cuda_float(*b2, "b2");
    TORCH_CHECK(w2->numel() == kC && b2->numel() == kC && w2->is_contiguous() && b2->is_contiguous(),
                "w2/b2 must be contiguous [1024]");
    TORCH_CHECK(eps2 > 0, "eps2 must be positive");
  }
  for (const auto* t : {&h, &x_out, &weight, &bias, &y2_out, &w2, &b2})
    if (t->has_value()) check_same_device(x, **t, "argument");
  if (rows == 0) return;
  c10::cuda::CUDAGuard guard(x.device());
  AddLnArgs a{reinterpret_cast<const __half*>(x.data_ptr()),
              h ? reinterpret_cast<const __half*>(h->data_ptr()) : nullptr,
              gate ? reinterpret_cast<const __half*>(gate->data_ptr()) : nullptr,
              weight ? weight->data_ptr<float>() : nullptr,
              bias ? bias->data_ptr<float>() : nullptr,
              scale ? reinterpret_cast<const __half*>(scale->data_ptr()) : nullptr,
              shift ? reinterpret_cast<const __half*>(shift->data_ptr()) : nullptr,
              x_out ? reinterpret_cast<__half*>(x_out->data_ptr()) : nullptr,
              reinterpret_cast<__half*>(y_out.data_ptr()),
              w2 ? w2->data_ptr<float>() : nullptr,
              b2 ? b2->data_ptr<float>() : nullptr,
              y2_out ? y2_out->data_ptr<float>() : nullptr,
              rows,
              rows_per_batch,
              mod_ld,
              static_cast<float>(eps),
              static_cast<float>(eps2)};
  add_ln_mod_kernel<<<static_cast<unsigned>(rows), kRowThreads, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_rms_qk(at::Tensor& q_out, at::Tensor& k_out, const at::Tensor& qkv, const at::Tensor& delta,
                 const at::Tensor& freqs, const at::Tensor& gamma_q, const at::Tensor& gamma_k) {
  check_cuda_half(qkv, "qkv");
  check_cuda_float(delta, "delta");
  check_cuda_float(freqs, "freqs");
  check_cuda_float(gamma_q, "gamma_q");
  check_cuda_float(gamma_k, "gamma_k");
  check_cuda_half(q_out, "q_out");
  check_cuda_half(k_out, "k_out");
  TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1, "qkv must be a [M, 3*H*64] matrix with unit inner stride");
  const int64_t M = qkv.size(0);
  TORCH_CHECK(qkv.size(1) % (3 * 64) == 0, "qkv width must be 3*H*64");
  const int H = static_cast<int>(qkv.size(1) / (3 * 64));
  TORCH_CHECK(qkv.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 4 == 0, "qkv misaligned");
  TORCH_CHECK(delta.dim() == 2 && delta.size(0) == M && delta.size(1) == 3 * H && delta.stride(1) == 1,
              "delta must be [M, 3*H] with unit inner stride, got ", delta.sizes());
  TORCH_CHECK(freqs.numel() == 32 && freqs.is_contiguous(), "freqs must be contiguous [32] (10 + 10 + 12)");
  TORCH_CHECK(gamma_q.numel() == H * 64 && gamma_q.is_contiguous() && gamma_k.numel() == H * 64 &&
                  gamma_k.is_contiguous(),
              "gamma_q/gamma_k must be contiguous [H, 64]");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&q_out, &k_out})
    TORCH_CHECK(t->numel() == M * H * 64 && t->is_contiguous(), "q_out/k_out must be contiguous [M, H, 64]");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&delta, &freqs, &gamma_q, &gamma_k, &q_out, &k_out})
    check_same_device(qkv, *t, "argument");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(qkv.device());
  const int64_t threads = M * H * 32;
  rope_rms_qk_kernel<<<blocks_for(threads, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(qkv.data_ptr()), qkv.stride(0), delta.data_ptr<float>(), delta.stride(0),
      freqs.data_ptr<float>(), gamma_q.data_ptr<float>(), gamma_k.data_ptr<float>(),
      reinterpret_cast<__half*>(q_out.data_ptr()), reinterpret_cast<__half*>(k_out.data_ptr()), M, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rms_heads(at::Tensor& out, const at::Tensor& x, const at::Tensor& gamma) {
  check_cuda_half(x, "x");
  check_cuda_half(out, "out");
  check_cuda_float(gamma, "gamma");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.size(1) % 64 == 0, "x must be [M, H*64] with unit inner stride");
  TORCH_CHECK(x.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0, "x misaligned");
  const int64_t M = x.size(0);
  const int H = static_cast<int>(x.size(1) / 64);
  TORCH_CHECK(gamma.numel() == H * 64 && gamma.is_contiguous(), "gamma must be contiguous [H, 64]");
  TORCH_CHECK(out.numel() == M * H * 64 && out.is_contiguous(), "out must be contiguous [M, H, 64]");
  check_same_device(x, gamma, "gamma");
  check_same_device(x, out, "out");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(x.device());
  rms_heads_kernel<<<blocks_for(M * H * 32, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(x.data_ptr()), x.stride(0), gamma.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr()), M, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void merge_extra_key(at::Tensor& out, const at::Tensor& o, const at::Tensor& lse, const at::Tensor& q,
                     const at::Tensor& k_extra, const at::Tensor& v_extra, double scale, double log_mult) {
  for (auto p : std::initializer_list<std::pair<const at::Tensor*, const char*>>{TP(&out, "out"), TP(&o, "o"), TP(&q, "q"),
                 TP(&k_extra, "k_extra"), TP(&v_extra, "v_extra")}) {
    check_cuda_half(*p.first, p.second);
    check_same_device(o, *p.first, p.second);
  }
  check_cuda_float(lse, "lse");
  check_same_device(o, lse, "lse");
  TORCH_CHECK(o.dim() == 4 && o.size(3) == 64, "o must be [B, L, H, 64]");
  const int64_t B = o.size(0), L = o.size(1);
  const int H = static_cast<int>(o.size(2));
  for (auto p : std::initializer_list<std::pair<const at::Tensor*, const char*>>{TP(&out, "out"), TP(&o, "o"), TP(&q, "q")}) {
    const at::Tensor& t = *p.first;
    TORCH_CHECK(t.dim() == 4 && t.size(0) == B && t.size(1) == L && t.size(2) == H && t.size(3) == 64, p.second,
                " must be [B, L, H, 64]");
    const int64_t ld = L == 1 ? t.stride(0) : t.stride(1);  // a size-1 dim may carry any stride
    TORCH_CHECK(t.stride(3) == 1 && t.stride(2) == 64 && (B == 1 || t.stride(0) == L * ld) && ld % 2 == 0,
                p.second, " must have [B, L, H, 64] layout with a row stride (token) and packed heads");
  }
  TORCH_CHECK(lse.dim() == 3 && lse.size(0) == B && lse.size(1) == H && lse.size(2) == L && lse.is_contiguous(),
              "lse must be contiguous [B, H, L]");
  TORCH_CHECK(k_extra.numel() == B * H * 64 && k_extra.is_contiguous() && v_extra.numel() == B * H * 64 &&
                  v_extra.is_contiguous(),
              "k_extra/v_extra must be contiguous [B, H, 64]");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(log_mult), "scale/log_mult must be finite");
  if (B * L == 0) return;
  c10::cuda::CUDAGuard guard(o.device());
  merge_key_kernel<<<blocks_for(B * L * H * 32, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(o.data_ptr()), L == 1 ? o.stride(0) : o.stride(1), lse.data_ptr<float>(),
      reinterpret_cast<const __half*>(q.data_ptr()), L == 1 ? q.stride(0) : q.stride(1),
      reinterpret_cast<const __half*>(k_extra.data_ptr()), reinterpret_cast<const __half*>(v_extra.data_ptr()),
      reinterpret_cast<__half*>(out.data_ptr()), L == 1 ? out.stride(0) : out.stride(1),
      B, L, H, static_cast<float>(scale), static_cast<float>(log_mult));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gaussians_ply(at::Tensor& out, const at::Tensor& feats, const at::Tensor& points, const at::Tensor& perturb,
                   double base_offset_scale, double scaling_offset, double opacity_offset, double min_kernel_size,
                   double lr_rotation, double perturb_size) {
  check_cuda_float(out, "out");
  check_cuda_float(feats, "feats");
  check_cuda_float(points, "points");
  check_cuda_float(perturb, "perturb");
  for (auto p : std::initializer_list<std::pair<const at::Tensor*, const char*>>{TP(&out, "out"), TP(&points, "points"), TP(&perturb, "perturb")})
    check_same_device(feats, *p.first, p.second);
  TORCH_CHECK(feats.dim() == 2 && feats.size(1) == 480 && feats.is_contiguous(), "feats must be contiguous [A, 480]");
  const int64_t A = feats.size(0);
  TORCH_CHECK(points.dim() == 2 && points.size(0) == A && points.size(1) == 3 && points.is_contiguous(),
              "points must be contiguous [A, 3]");
  TORCH_CHECK(perturb.numel() == 96 && perturb.is_contiguous(), "perturb must be contiguous [32, 3]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == A * 32 && out.size(1) == 17 && out.is_contiguous(),
              "out must be contiguous [A*32, 17]");
  if (A == 0) return;
  c10::cuda::CUDAGuard guard(feats.device());
  GsArgs a{feats.data_ptr<float>(),
           points.data_ptr<float>(),
           perturb.data_ptr<float>(),
           out.data_ptr<float>(),
           A,
           static_cast<float>(base_offset_scale),
           static_cast<float>(scaling_offset),
           static_cast<float>(opacity_offset),
           static_cast<float>(min_kernel_size * min_kernel_size),
           static_cast<float>(lr_rotation),
           static_cast<float>(0.5 * perturb_size)};
  gaussians_ply_kernel<<<blocks_for(A * 32, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// fc1 + bias + GELU(tanh) through cuBLASLt's GELU epilogue (at::_addmm_activation).
void linear_gelu(at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
  check_cuda_half(x, "x");
  check_cuda_half(weight, "weight");
  check_cuda_half(bias, "bias");
  check_cuda_half(out, "out");
  check_same_device(x, weight, "weight");
  check_same_device(x, bias, "bias");
  check_same_device(x, out, "out");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && bias.dim() == 1, "x [M, K], weight [N, K], bias [N] expected");
  TORCH_CHECK(x.size(1) == weight.size(1) && bias.size(0) == weight.size(0), "shape mismatch: x ", x.sizes(),
              " weight ", weight.sizes(), " bias ", bias.sizes());
  TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == weight.size(0) && out.is_contiguous(),
              "out must be contiguous [M, N]");
  c10::cuda::CUDAGuard guard(x.device());
  at::_addmm_activation_out(out, bias, x, weight.t(), 1, 1, /*use_gelu=*/true);
}

}  // namespace idol_cuda

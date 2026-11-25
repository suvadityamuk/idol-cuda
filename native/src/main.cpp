// idol_native: the Idol image-to-3D pipeline in C++ (libtorch + CUDA), no Python in the process.
//
//   idol_native --weights W --ts TS --image IMG [--mode synthetic|real] [--seed 42] [--steps 50] [--cfg 3.0]
//               [--rescale_t 3.0] [--num_gaussians 262144] [--out DIR] [--build parity|fast]
//               [--dump FILE] [--replay DUMP] [--teacher DUMP] [--repeat N] [--timing FILE]
//
// --build parity : op-for-op port of the reference (ATTN_BACKEND=sdpa, optimize=False); bit-identical on the stated
//                  (GPU, CUDA, cuDNN, cuBLAS, torch) tuple.
// --build fast   : the optimised build (see docs/ab-log.md).
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/version.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/detail/CUDAHooksInterface.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "decoders.h"
#include "dit.h"
#include "encoders.h"
#include "fast.h"
#include "image.h"
#include "safetensors.h"

using namespace idol;

namespace {

struct Args {
  std::string weights = "/w/idol", ts = "/scratch/ts", image, mode = "synthetic", out = "out", name;
  std::string build = "parity", dump, replay, teacher, timing, fa3;
  double crop_margin = -1, cfg = 3.0, rescale_t = 3.0;
  int64_t seed = 42, steps = 50, num_gaussians = 262144, repeat = 1;
  bool write_outputs = true, reuse = true, serve = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto v = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--weights") a.weights = v();
    else if (k == "--ts") a.ts = v();
    else if (k == "--image") a.image = v();
    else if (k == "--mode") a.mode = v();
    else if (k == "--crop_margin") a.crop_margin = std::stod(v());
    else if (k == "--seed") a.seed = std::stoll(v());
    else if (k == "--steps") a.steps = std::stoll(v());
    else if (k == "--cfg" || k == "--cfg_strength") a.cfg = std::stod(v());
    else if (k == "--rescale_t") a.rescale_t = std::stod(v());
    else if (k == "--num_gaussians") a.num_gaussians = std::stoll(v());
    else if (k == "--out") a.out = v();
    else if (k == "--name") a.name = v();
    else if (k == "--build") a.build = v();
    else if (k == "--dump") a.dump = v();
    else if (k == "--replay") a.replay = v();
    else if (k == "--teacher") a.teacher = v();
    else if (k == "--repeat") a.repeat = std::stoll(v());
    else if (k == "--timing") a.timing = v();
    else if (k == "--no_write") a.write_outputs = false;
    else if (k == "--no_reuse") a.reuse = false;
    else if (k == "--serve") a.serve = true;
    else if (k == "--fa3") a.fa3 = v();
    else throw std::runtime_error("unknown argument " + k);
  }
  if (a.image.empty() && a.teacher.empty() && !a.serve) throw std::runtime_error("--image is required");
  if (a.mode != "synthetic" && a.mode != "real") throw std::runtime_error("--mode must be synthetic or real");
  if (a.build != "parity" && a.build != "fast") throw std::runtime_error("--build must be parity or fast");
  if (a.crop_margin < 0) a.crop_margin = a.mode == "synthetic" ? 1.2 : 1.7;
  if (a.name.empty() && !a.image.empty()) {
    std::string b = a.image.substr(a.image.find_last_of('/') + 1);
    a.name = b.substr(0, b.find_last_of('.'));
  }
  return a;
}

// Load a reference checkpoint the way IdolPipeline does: Linear layers inside the transformer blocks are fp16
// (convert_module_to_f16 before load_state_dict, which casts on copy), every other parameter/buffer fp32.
Weights load_model(const std::string& path, const std::vector<std::string>& half_blocks,
                   const std::vector<std::string>& half_linear, at::Device dev) {
  Weights ws = load_safetensors(path);
  for (auto& kv : ws) {
    bool in_block = false, is_linear = false;
    for (auto& p : half_blocks)
      if (kv.first.rfind(p, 0) == 0) in_block = true;
    for (auto& l : half_linear)
      if (kv.first.find(l) != std::string::npos) is_linear = true;
    at::Tensor t = kv.second;
    if (t.is_floating_point()) t = t.to(in_block && is_linear ? at::kHalf : at::kFloat);
    kv.second = t.to(dev);
  }
  return ws;
}

struct Timer {
  std::map<std::string, double> acc;
  std::chrono::steady_clock::time_point t0;
  bool on = false;
  void start() {
    if (!on) return;
    cudaDeviceSynchronize();
    t0 = std::chrono::steady_clock::now();
  }
  void stop(const std::string& k) {
    if (!on) return;
    cudaDeviceSynchronize();
    acc[k] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
};

// numpy: t_seq = np.linspace(1, 0, steps + 1); t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
std::vector<double> time_grid(int64_t steps, double rescale_t) {
  std::vector<double> t(steps + 1);
  const double step = (0.0 - 1.0) / static_cast<double>(steps);
  for (int64_t i = 0; i <= steps; i++) {
    volatile double y = static_cast<double>(i) * step;
    t[i] = y + 1.0;
  }
  t[steps] = 0.0;
  for (auto& v : t) {
    volatile double a = rescale_t * v;
    volatile double b = (rescale_t - 1) * v;
    volatile double c = 1 + b;
    v = a / c;
  }
  return t;
}

struct Models {
  std::unique_ptr<Denoiser> denoiser;
  std::unique_ptr<OctreeDecoder> octree;
  std::unique_ptr<GsDecoder> gsdec;
  std::unique_ptr<Encoders> enc;
};

void record_denoiser_call(const at::Tensor& z, const at::Tensor& cam, const at::Tensor& t, const at::Tensor& f1,
                          const at::Tensor& f2, const at::Tensor& oz, const at::Tensor& oc) {
  if (!REC.on) return;
  const std::string k = "denoiser#" + std::to_string(REC.next("denoiser"));
  REC.put(k + ".in.0.latent", z);
  REC.put(k + ".in.0.camera", cam);
  REC.put(k + ".in.1", t);
  REC.put(k + ".in.2.feature1", f1);
  REC.put(k + ".in.2.feature2", f2);
  REC.put(k + ".out.latent", oz);
  REC.put(k + ".out.camera", oc);
}

// sample_flow(batch_cfg=False), as in --no_optimize. reuse=true: the context stream is computed once per request and
// the per-step prefix (time embedding, noise refiner, camera refiner) once per step for both CFG passes. Every reused
// tensor is a deterministic function of the same inputs, so the result is bit-identical to recomputing it.
std::pair<at::Tensor, at::Tensor> sample_parity(const Denoiser& m, at::Tensor z, at::Tensor cam, const at::Tensor& f1,
                                                const at::Tensor& f2, int64_t steps, double rescale_t, double cfg,
                                                bool reuse) {
  const at::Tensor n1 = at::zeros_like(f1), n2 = at::zeros_like(f2);
  const auto ts = time_grid(steps, rescale_t);
  at::Tensor hc, hu;
  if (reuse) {
    const bool rec = REC.on && REC.calls["denoiser.internals"]++ == 0;
    hc = m.context(f1, f2, rec);
    if (cfg > 0) hu = m.context(n1, n2);
  }
  for (int64_t i = 0; i < steps; i++) {
    const double t = ts[i], t_prev = ts[i + 1];
    at::Tensor tt = at::full({z.size(0)}, static_cast<float>(1000 * t), z.options());
    std::pair<at::Tensor, at::Tensor> vc, vu;
    Denoiser::Prefix P;
    if (reuse) {
      P = m.prefix(z, cam, tt, REC.on && i == 0);
      vc = m.joint(P, hc, z, cam, REC.on && i == 0);
    } else {
      vc = m(z, cam, tt, f1, f2);
    }
    record_denoiser_call(z, cam, tt, f1, f2, vc.first, vc.second);
    at::Tensor vz = vc.first, vcam = vc.second;
    if (cfg > 0) {
      vz = vz * (1 + cfg);
      vcam = vcam * (1 + cfg);
      vu = reuse ? m.joint(P, hu, z, cam) : m(z, cam, tt, n1, n2);
      record_denoiser_call(z, cam, tt, n1, n2, vu.first, vu.second);
      vz = vz - vu.first * cfg;
      vcam = vcam - vu.second * cfg;
    }
    const double dt = t - t_prev;
    z = z - vz * dt;
    cam = cam - vcam * dt;
  }
  return {z, cam};
}

void write_text(const std::string& path, const std::string& s) {
  std::ofstream f(path, std::ios::binary);
  f << s;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    if (!torch::cuda::is_available()) {
      std::cerr << "error: no CUDA device available (idol_native needs an NVIDIA GPU)\n";
      return 2;
    }
    configure_jit();
    torch::NoGradGuard no_grad;
    const at::Device dev(at::kCUDA, 0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("idol_native: build=%s device=%s (sm_%d%d) torch=%s cudnn=%ld\n", a.build.c_str(), prop.name,
                prop.major, prop.minor, TORCH_VERSION, at::detail::getCUDAHooks().versionCuDNN());
    std::fflush(stdout);

    auto t_load = std::chrono::steady_clock::now();
    Models M;
    M.denoiser = std::make_unique<Denoiser>(
        load_model(a.weights + "/denoiser.safetensors", {"blocks.", "noise_refiner.", "context_refiner."},
                   {".attn.qkv.", ".attn.out.", ".mlp.mlp."}, dev),
        dev);
    const std::vector<std::string> dec_lin = {".to_q.", ".to_kv.", ".to_qkv.", ".to_out.", ".mlp.mlp."};
    M.octree = std::make_unique<OctreeDecoder>(load_model(a.weights + "/decoder.safetensors", {"blocks."}, dec_lin, dev),
                                               dev);
    M.gsdec = std::make_unique<GsDecoder>(load_model(a.weights + "/decoder_gs.safetensors", {"blocks."}, dec_lin, dev),
                                          dev);
    M.enc = std::make_unique<Encoders>(a.ts, dev);
    const bool fast = a.build == "fast";
    std::unique_ptr<FastPipeline> FP;
    if (fast) {
      FP = std::make_unique<FastPipeline>(*M.denoiser, *M.octree, *M.gsdec, dev);
      const bool fa3 = FastPipeline::use_fa3(a.fa3);
      std::printf("fast build attention: %s\n", fa3 ? "FlashAttention-3 (Hub kernel)" : "cuDNN");
    }
    cudaDeviceSynchronize();
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
    std::printf("loaded in %.2f s\n", load_s);

    Weights replay, teacher;
    if (!a.replay.empty()) replay = load_safetensors(a.replay);
    if (!a.teacher.empty()) teacher = load_safetensors(a.teacher);

    // ------------------------------------------------------------------------------------ teacher-forced stage tests
    if (!a.teacher.empty()) {
      REC.on = true;
      Weights& T = teacher;
      auto cu = [&](const std::string& k) { return W(T, k).to(dev); };
      // encoders on the reference's preprocessed image, reference VAE noise
      {
        Rng rng;
        rng.replay = &T;
        at::Tensor img = W(T, "final.preprocessed_image");
        Img rgb = new_image("RGB", img.size(1), img.size(0));
        for (int64_t i = 0; i < img.size(0) * img.size(1); i++)
          for (int c = 0; c < 3; c++) rgb.px[i * 4 + c] = img.data_ptr<uint8_t>()[i * 3 + c];
        auto f = M.enc->encode(image_to_pixels(rgb, dev), rng);
        REC.put("denoiser#0.in.2.feature1", f.first);
        REC.put("denoiser#0.in.2.feature2", f.second);
      }
      // every denoiser call on the reference's inputs
      for (int k = 0;; k++) {
        const std::string p = "denoiser#" + std::to_string(k);
        if (!T.count(p + ".in.0.latent")) break;
        std::pair<at::Tensor, at::Tensor> o;
        if (fast)
          o = FP->denoise(cu(p + ".in.0.latent"), cu(p + ".in.0.camera"), cu(p + ".in.1"), cu(p + ".in.2.feature1"),
                          cu(p + ".in.2.feature2"));
        else
          o = (*M.denoiser)(cu(p + ".in.0.latent"), cu(p + ".in.0.camera"), cu(p + ".in.1"), cu(p + ".in.2.feature1"),
                            cu(p + ".in.2.feature2"));
        REC.put(p + ".out.latent", o.first);
        REC.put(p + ".out.camera", o.second);
      }
      // decoders on the reference latent / anchors, reference RNG
      {
        Rng rng;
        rng.replay = &T;
        rng.pos = 3;
        at::Tensor lat = cu("octree#0.in.2");
        const int64_t npts = std::max<int64_t>(1, W(T, "gsdec#0.in.0").size(1));
        at::Tensor anchors = fast ? FP->octree_sample(lat, npts, 8, rng) : octree_sample(*M.octree, lat, npts, 8, rng);
        REC.put("tf.anchors", anchors);  // compare with gsdec#0.in.0 (tools/check_teacher.py)
        at::Tensor feats = fast ? FP->gsdec(cu("gsdec#0.in.0"), lat) : (*M.gsdec)(cu("gsdec#0.in.0"), lat);
        REC.put("gsdec#0.out", feats);
        Gaussians g = to_gaussians(*M.gsdec, cu("gsdec#0.in.0"), cu("gsdec#0.out"));
        REC.put("final.xyz", g.xyz);
        REC.put("final.features_dc", g.features_dc);
        REC.put("final.scaling", g.scaling);
        REC.put("final.rotation", g.rotation);
        REC.put("final.opacity", g.opacity);
      }
      save_safetensors(a.dump.empty() ? a.out + "/teacher.safetensors" : a.dump, REC.m);
      std::printf("teacher-forced dump written (%zu tensors)\n", REC.m.size());
      return 0;
    }

    // ------------------------------------------------------------------------------------ end-to-end requests
    std::ostringstream timing_json;
    timing_json << "{\"load_s\":" << load_s << ",\"build\":\"" << a.build << "\",\"runs\":[";
    // --serve: one request per stdin line "image mode num_gaussians name outdir" (benchmark worker); prints
    // "RESULT {json}" per request and "READY" when waiting for the next line.
    std::string line;
    if (a.serve) {
      a.timing = "-";
      a.repeat = 1;
      std::printf("READY\n");
      std::fflush(stdout);
    }
    while (!a.serve || std::getline(std::cin, line)) {
    if (a.serve) {
      std::istringstream ls(line);
      std::string ng;
      ls >> a.image >> a.mode >> ng >> a.name >> a.out;
      a.num_gaussians = std::stoll(ng);
      a.crop_margin = a.mode == "synthetic" ? 1.2 : 1.7;
    }
    for (int64_t rep = 0; rep < a.repeat; rep++) {
      Timer T;
      T.on = !a.timing.empty();
      REC.on = !a.dump.empty() && rep == 0;
      REC.m.clear();
      REC.calls.clear();
      cudaDeviceSynchronize();
      if (T.on) at::cuda::CUDACachingAllocator::resetPeakStats(0);
      auto t_req = std::chrono::steady_clock::now();

      T.start();
      Img prepared = M.enc->preprocess(decode_webp(a.image), a.crop_margin);
      at::Tensor pixels = image_to_pixels(prepared, dev);
      T.stop("preprocess");

      torch::manual_seed(a.seed);
      Rng rng;
      if (!a.replay.empty()) rng.replay = &replay;
      T.start();
      auto feats = M.enc->encode(pixels, rng);
      T.stop("encoders");
      at::Tensor latent = rng.randn({1, M.denoiser->q_len, M.denoiser->in_ch},
                                    at::TensorOptions().dtype(at::kFloat).device(dev));
      at::Tensor camera = rng.randn({1, 1, M.denoiser->cam_ch}, at::TensorOptions().dtype(at::kFloat).device(dev));

      T.start();
      auto sample = fast ? FP->sample(latent, camera, feats.first, feats.second, a.steps, a.rescale_t, a.cfg)
                         : sample_parity(*M.denoiser, latent, camera, feats.first, feats.second, a.steps, a.rescale_t,
                                         a.cfg, a.reuse);
      T.stop("sampling");
      REC.put("octree#0.in.2", sample.first);

      T.start();
      const int64_t npts = std::max<int64_t>(1, a.num_gaussians / 32);
      at::Tensor anchors = fast ? FP->octree_sample(sample.first, npts, 8, rng)
                                : octree_sample(*M.octree, sample.first, npts, 8, rng);
      T.stop("octree");
      T.start();
      at::Tensor gfeat = fast ? FP->gsdec(anchors, sample.first) : (*M.gsdec)(anchors, sample.first);
      Gaussians g;
      at::Tensor rows;
      if (fast)
        rows = FP->ply_rows(anchors, gfeat);
      else
        g = to_gaussians(*M.gsdec, anchors, gfeat);
      T.stop("gs_decoder");
      cudaDeviceSynchronize();
      const double request_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_req).count();

      T.start();
      if (a.write_outputs) {
        std::string cmd = "mkdir -p '" + a.out + "'";
        if (std::system(cmd.c_str()) != 0) throw std::runtime_error("cannot create " + a.out);
        if (fast)
          write_ply_rows(rows, a.out + "/" + a.name + ".ply");
        else
          save_ply(g, a.out + "/" + a.name + ".ply");
        write_png_rgb(a.out + "/" + a.name + "_input.png", prepared);
        write_text(a.out + "/" + a.name + "_camera.json", camera_json(sample.second.select(0, 0).select(0, 0)));
      }
      T.stop("save");

      if (REC.on) {
        if (fast) {  // PLY columns are y-up (x, -z, y): back to the z-up object frame
          REC.put("final.xyz", at::stack({rows.select(1, 0), rows.select(1, 2), -rows.select(1, 1)}, 1));
          REC.put("latent", sample.first);
          auto cj = sample.second.select(0, 0).select(0, 0);
          REC.put("final.camera_latent", cj);
        } else {
          REC.put("final.xyz", g.xyz);
          REC.put("final.features_dc", g.features_dc);
          REC.put("final.scaling", g.scaling);
          REC.put("final.rotation", g.rotation);
          REC.put("final.opacity", g.opacity);
        }
        at::Tensor img = at::empty({prepared.h, prepared.w, 3}, at::TensorOptions().dtype(at::kByte));
        for (int64_t i = 0; i < int64_t(prepared.w) * prepared.h; i++)
          for (int c = 0; c < 3; c++) img.data_ptr<uint8_t>()[i * 3 + c] = prepared.px[i * 4 + c];
        REC.put("final.preprocessed_image", img);
        REC.put("final.camera_latent", sample.second);
        save_safetensors(a.dump, REC.m);
        REC.on = false;
        std::printf("dump written: %s (%zu tensors)\n", a.dump.c_str(), REC.m.size());
      }
      const auto stats = at::cuda::CUDACachingAllocator::getDeviceStats(0);
      const double peak_gb =
          stats.allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].peak / 1073741824.0;
      const double res_gb =
          stats.reserved_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].peak / 1073741824.0;
      std::printf("request %ld: %.3f s  peak_alloc %.2f GB  peak_reserved %.2f GB", rep, request_s, peak_gb, res_gb);
      for (auto& kv : T.acc) std::printf("  %s %.3f", kv.first.c_str(), kv.second);
      std::printf("\n");
      std::fflush(stdout);
      timing_json << (rep ? "," : "") << "{\"request_s\":" << request_s << ",\"peak_alloc_gb\":" << peak_gb
                  << ",\"peak_reserved_gb\":" << res_gb << ",\"stages\":{";
      bool first = true;
      for (auto& kv : T.acc) {
        timing_json << (first ? "" : ",") << "\"" << kv.first << "\":" << kv.second;
        first = false;
      }
      timing_json << "}}";
      if (a.serve) {
        std::printf("RESULT {\"request_s\":%.6f,\"peak_alloc_gb\":%.4f,\"peak_reserved_gb\":%.4f,\"stages\":{", request_s,
                    peak_gb, res_gb);
        bool f1 = true;
        for (auto& kv : T.acc) {
          std::printf("%s\"%s\":%.6f", f1 ? "" : ",", kv.first.c_str(), kv.second);
          f1 = false;
        }
        std::printf("}}\nREADY\n");
        std::fflush(stdout);
      }
    }
    if (!a.serve) break;
    }
    timing_json << "]}";
    if (!a.timing.empty() && a.timing != "-") write_text(a.timing, timing_json.str());
    if (a.write_outputs) std::printf("%s: %s/%s.ply\n", a.name.c_str(), a.out.c_str(), a.name.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

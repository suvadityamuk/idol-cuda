#include "image.h"

#include <webp/decode.h>
#include <webp/demux.h>
#include <zlib.h>

#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace idol {

Img new_image(const std::string& mode, int w, int h, uint8_t fill) {
  Img im;
  im.w = w;
  im.h = h;
  im.mode = mode;
  im.px.assign(static_cast<size_t>(w) * h * im.pixelsize(), fill);
  return im;
}

// ------------------------------------------------------------------------------------------------ WebP
Img decode_webp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open image " + path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  WebPData data{bytes.data(), bytes.size()};
  // Pillow: WebPAnimDecoderNew(&data, NULL) -> color_mode MODE_RGBA, use_threads 0
  WebPAnimDecoder* dec = WebPAnimDecoderNew(&data, nullptr);
  if (!dec) throw std::runtime_error("not a WebP file: " + path);
  WebPAnimInfo info;
  WebPAnimDecoderGetInfo(dec, &info);
  WebPDemuxer* demux = const_cast<WebPDemuxer*>(WebPAnimDecoderGetDemuxer(dec));
  const uint32_t flags = WebPDemuxGetI(demux, WEBP_FF_FORMAT_FLAGS);
  uint8_t* buf = nullptr;
  int ts = 0;
  if (!WebPAnimDecoderGetNext(dec, &buf, &ts)) {
    WebPAnimDecoderDelete(dec);
    throw std::runtime_error("WebP decode failed: " + path);
  }
  // Pillow: rawmode "RGBA" when the file has an alpha flag, else "RGBX" -> mode "RGB"
  Img im = new_image((flags & ALPHA_FLAG) ? "RGBA" : "RGB", info.canvas_width, info.canvas_height);
  std::memcpy(im.px.data(), buf, im.px.size());
  WebPAnimDecoderDelete(dec);
  return im;
}

// ------------------------------------------------------------------------------------------------ Resample.c
namespace {
constexpr int PRECISION_BITS = 32 - 8 - 2;

double sinc_filter(double x) {
  if (x == 0.0) return 1.0;
  x = x * M_PI;
  return sin(x) / x;
}
double lanczos_filter(double x) {
  if (-3.0 <= x && x < 3.0) return sinc_filter(x) * sinc_filter(x / 3);
  return 0.0;
}
constexpr double LANCZOS_SUPPORT = 3.0;

inline uint8_t clip8(int in) {
  // Pillow's lookup table: index (in >> PRECISION_BITS) + 640, 0 below 0, 255 above 255
  int v = in >> PRECISION_BITS;
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

int precompute_coeffs(int inSize, float in0, float in1, int outSize, std::vector<int>& bounds,
                      std::vector<double>& kk) {
  double support, scale, filterscale;
  filterscale = scale = static_cast<double>(in1 - in0) / outSize;
  if (filterscale < 1.0) filterscale = 1.0;
  support = LANCZOS_SUPPORT * filterscale;
  const int ksize = static_cast<int>(ceil(support)) * 2 + 1;
  kk.assign(static_cast<size_t>(outSize) * ksize, 0.0);
  bounds.assign(static_cast<size_t>(outSize) * 2, 0);
  const double inv_filterscale = 1.0 / filterscale;
  for (int xx = 0; xx < outSize; xx++) {
    double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    double* k = &kk[static_cast<size_t>(xx) * ksize];
    int x;
    for (x = 0; x < xmax; x++) {
      double w = lanczos_filter((x + xmin - center + 0.5) * inv_filterscale);
      k[x] = w;
      ww += w;
    }
    if (ww != 0.0)
      for (x = 0; x < xmax; x++) k[x] /= ww;
    for (; x < ksize; x++) k[x] = 0;
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return ksize;
}

std::vector<int32_t> normalize_coeffs_8bpc(const std::vector<double>& prekk) {
  std::vector<int32_t> kk(prekk.size());
  for (size_t x = 0; x < prekk.size(); x++) {
    if (prekk[x] < 0)
      kk[x] = static_cast<int>(-0.5 + prekk[x] * (1 << PRECISION_BITS));
    else
      kk[x] = static_cast<int>(0.5 + prekk[x] * (1 << PRECISION_BITS));
  }
  return kk;
}

int bands_of(const Img& im) {
  if (im.mode == "L") return 1;
  if (im.mode == "RGB") return 3;
  return 4;
}

Img resample_horizontal(const Img& in, int xsize, int ysize, int offset, int ksize, const std::vector<int>& bounds,
                        const std::vector<double>& prekk) {
  Img out = new_image(in.mode, xsize, ysize);
  auto kk = normalize_coeffs_8bpc(prekk);
  const int bands = bands_of(in);
  for (int yy = 0; yy < ysize; yy++) {
    const uint8_t* lineIn = in.row(yy + offset);
    uint8_t* lineOut = out.row(yy);
    for (int xx = 0; xx < xsize; xx++) {
      const int xmin = bounds[xx * 2 + 0], xmax = bounds[xx * 2 + 1];
      const int32_t* k = &kk[static_cast<size_t>(xx) * ksize];
      if (bands == 1) {
        int ss0 = 1 << (PRECISION_BITS - 1);
        for (int x = 0; x < xmax; x++) ss0 += lineIn[x + xmin] * k[x];
        lineOut[xx] = clip8(ss0);
      } else {
        int ss[4] = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1),
                     1 << (PRECISION_BITS - 1)};
        for (int x = 0; x < xmax; x++)
          for (int b = 0; b < 4; b++) ss[b] += lineIn[(x + xmin) * 4 + b] * k[x];
        lineOut[xx * 4 + 0] = clip8(ss[0]);
        lineOut[xx * 4 + 1] = clip8(ss[1]);
        lineOut[xx * 4 + 2] = clip8(ss[2]);
        lineOut[xx * 4 + 3] = bands == 3 ? 0 : clip8(ss[3]);
      }
    }
  }
  return out;
}

Img resample_vertical(const Img& in, int xsize, int ysize, int ksize, const std::vector<int>& bounds,
                      const std::vector<double>& prekk) {
  Img out = new_image(in.mode, xsize, ysize);
  auto kk = normalize_coeffs_8bpc(prekk);
  const int bands = bands_of(in);
  for (int yy = 0; yy < ysize; yy++) {
    uint8_t* lineOut = out.row(yy);
    const int32_t* k = &kk[static_cast<size_t>(yy) * ksize];
    const int ymin = bounds[yy * 2 + 0], ymax = bounds[yy * 2 + 1];
    for (int xx = 0; xx < xsize; xx++) {
      if (bands == 1) {
        int ss0 = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < ymax; y++) ss0 += in.row(y + ymin)[xx] * k[y];
        lineOut[xx] = clip8(ss0);
      } else {
        int ss[4] = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1),
                     1 << (PRECISION_BITS - 1)};
        for (int y = 0; y < ymax; y++) {
          const uint8_t* r = in.row(y + ymin);
          for (int b = 0; b < 4; b++) ss[b] += r[xx * 4 + b] * k[y];
        }
        lineOut[xx * 4 + 0] = clip8(ss[0]);
        lineOut[xx * 4 + 1] = clip8(ss[1]);
        lineOut[xx * 4 + 2] = clip8(ss[2]);
        lineOut[xx * 4 + 3] = bands == 3 ? 0 : clip8(ss[3]);
      }
    }
  }
  return out;
}

// ImagingResampleInner with filter LANCZOS and box = (0, 0, in.w, in.h)
Img resample(const Img& imIn, int xsize, int ysize) {
  const float box[4] = {0.f, 0.f, static_cast<float>(imIn.w), static_cast<float>(imIn.h)};
  const bool need_horizontal = xsize != imIn.w || box[0] || box[2] != xsize;
  const bool need_vertical = ysize != imIn.h || box[1] || box[3] != ysize;
  const bool horizontal_first = !((imIn.h - ysize) > 0 && (imIn.h - ysize) > (imIn.w - xsize) * 2);
  std::vector<int> bounds_vert, bounds_horiz;
  std::vector<double> kk_vert, kk_horiz;
  int ksize_vert = 0, ksize_horiz = 0, ybox_first = 0, ybox_last = 0;
  if ((need_horizontal && horizontal_first) || need_vertical)
    ksize_vert = precompute_coeffs(imIn.h, box[1], box[3], ysize, bounds_vert, kk_vert);
  if (need_horizontal) {
    if (horizontal_first) {
      ybox_first = bounds_vert[0];
      ybox_last = bounds_vert[ysize * 2 - 2] + bounds_vert[ysize * 2 - 1];
      if (ybox_first != 0)
        for (int i = 0; i < ysize; i++) bounds_vert[i * 2] -= ybox_first;
    }
    ksize_horiz = precompute_coeffs(imIn.w, box[0], box[2], xsize, bounds_horiz, kk_horiz);
  }
  Img cur = imIn;
  if (horizontal_first) {
    if (need_horizontal)
      cur = resample_horizontal(cur, xsize, ybox_last - ybox_first, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
    if (need_vertical) cur = resample_vertical(cur, xsize, ysize, ksize_vert, bounds_vert, kk_vert);
  } else {
    if (need_vertical) cur = resample_vertical(cur, cur.w, ysize, ksize_vert, bounds_vert, kk_vert);
    if (need_horizontal) cur = resample_horizontal(cur, xsize, ysize, 0, ksize_horiz, bounds_horiz, kk_horiz);
  }
  if (!need_horizontal && !need_vertical) cur = imIn;  // ImagingCopy
  return cur;
}

inline unsigned muldiv255(unsigned a, unsigned b) {
  unsigned tmp = a * b + 128;
  return ((tmp >> 8) + tmp) >> 8;
}
inline int clip8i(int v) { return v <= 0 ? 0 : v < 256 ? v : 255; }

}  // namespace

Img convert(const Img& im, const std::string& mode) {
  if (im.mode == mode) return im;
  Img out = new_image(mode, im.w, im.h);
  const size_t n = static_cast<size_t>(im.w) * im.h;
  if (im.mode == "RGBA" && mode == "RGBa") {  // rgbA2rgba
    for (size_t i = 0; i < n; i++) {
      const uint8_t* in = &im.px[i * 4];
      uint8_t* o = &out.px[i * 4];
      const unsigned alpha = in[3];
      o[0] = muldiv255(in[0], alpha);
      o[1] = muldiv255(in[1], alpha);
      o[2] = muldiv255(in[2], alpha);
      o[3] = in[3];
    }
  } else if (im.mode == "RGBa" && mode == "RGBA") {  // rgba2rgbA
    for (size_t i = 0; i < n; i++) {
      const uint8_t* in = &im.px[i * 4];
      uint8_t* o = &out.px[i * 4];
      const unsigned alpha = in[3];
      if (alpha == 255 || alpha == 0) {
        o[0] = in[0];
        o[1] = in[1];
        o[2] = in[2];
      } else {
        o[0] = clip8i((255 * in[0]) / alpha);
        o[1] = clip8i((255 * in[1]) / alpha);
        o[2] = clip8i((255 * in[2]) / alpha);
      }
      o[3] = in[3];
    }
  } else if (im.mode == "RGBA" && mode == "RGB") {  // rgba2rgb: copy channels, padding 255
    for (size_t i = 0; i < n; i++) {
      std::memcpy(&out.px[i * 4], &im.px[i * 4], 3);
      out.px[i * 4 + 3] = 255;
    }
  } else {
    throw std::runtime_error("convert: unsupported " + im.mode + " -> " + mode);
  }
  return out;
}

Img resize_lanczos(const Img& im, int w, int h) {
  if (im.w == w && im.h == h) return im;  // Image.resize returns self.copy()
  if (im.mode == "RGBA") return convert(resample(convert(im, "RGBa"), w, h), "RGBA");
  return resample(im, w, h);
}

Img crop(const Img& im, int x0, int y0, int x1, int y1) {
  // ImagingCrop: zero-filled outside the source
  Img out = new_image(im.mode, x1 - x0, y1 - y0, 0);
  const int ps = im.pixelsize();
  for (int y = 0; y < out.h; y++) {
    const int sy = y + y0;
    if (sy < 0 || sy >= im.h) continue;
    for (int x = 0; x < out.w; x++) {
      const int sx = x + x0;
      if (sx < 0 || sx >= im.w) continue;
      std::memcpy(out.row(y) + x * ps, im.row(sy) + sx * ps, ps);
    }
  }
  return out;
}

Img getchannel(const Img& im, int band) {
  Img out = new_image("L", im.w, im.h);
  const size_t n = static_cast<size_t>(im.w) * im.h;
  for (size_t i = 0; i < n; i++) out.px[i] = im.px[i * 4 + band];
  return out;
}

void putalpha(Img& im, const Img& alpha) {
  if (im.mode != "RGB" && im.mode != "RGBA") throw std::runtime_error("putalpha: mode " + im.mode);
  im.mode = "RGBA";
  const size_t n = static_cast<size_t>(im.w) * im.h;
  for (size_t i = 0; i < n; i++) im.px[i * 4 + 3] = alpha.px[i];
}

Img min_filter3(const Img& l) {
  // image.expand(1) replicates the border, then the rank-0 filter over each 3x3 window
  Img out = new_image("L", l.w, l.h);
  for (int y = 0; y < l.h; y++)
    for (int x = 0; x < l.w; x++) {
      uint8_t m = 255;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          const int yy = std::min(std::max(y + dy, 0), l.h - 1), xx = std::min(std::max(x + dx, 0), l.w - 1);
          m = std::min(m, l.row(yy)[xx]);
        }
      out.row(y)[x] = m;
    }
  return out;
}

void paste_with_mask(Img& dst, const Img& src, const Img& mask) {
  // paste_mask_L: BLEND(mask, out, in) = DIV255(out * (255 - mask) + in * mask) on all 4 bytes
  const size_t n = static_cast<size_t>(dst.w) * dst.h;
  for (size_t i = 0; i < n; i++) {
    const unsigned a = mask.px[i];
    for (int b = 0; b < 4; b++) {
      unsigned tmp = dst.px[i * 4 + b] * (255 - a) + src.px[i * 4 + b] * a + 128;
      dst.px[i * 4 + b] = static_cast<uint8_t>(((tmp >> 8) + tmp) >> 8);
    }
  }
}

bool has_real_alpha(const Img& im) {
  if (im.mode != "RGBA") return false;
  const size_t n = static_cast<size_t>(im.w) * im.h;
  for (size_t i = 0; i < n; i++)
    if (im.px[i * 4 + 3] < 255) return true;
  return false;
}

void write_png_rgb(const std::string& path, const Img& im) {
  auto crc_chunk = [](std::ofstream& f, const char* type, const std::vector<uint8_t>& data) {
    uint32_t len = static_cast<uint32_t>(data.size());
    uint8_t be[4] = {uint8_t(len >> 24), uint8_t(len >> 16), uint8_t(len >> 8), uint8_t(len)};
    f.write(reinterpret_cast<char*>(be), 4);
    f.write(type, 4);
    if (!data.empty()) f.write(reinterpret_cast<const char*>(data.data()), data.size());
    uLong crc = crc32(0L, reinterpret_cast<const Bytef*>(type), 4);
    if (!data.empty()) crc = crc32(crc, data.data(), data.size());
    uint8_t cb[4] = {uint8_t(crc >> 24), uint8_t(crc >> 16), uint8_t(crc >> 8), uint8_t(crc)};
    f.write(reinterpret_cast<char*>(cb), 4);
  };
  std::vector<uint8_t> raw;
  raw.reserve(static_cast<size_t>(im.h) * (im.w * 3 + 1));
  for (int y = 0; y < im.h; y++) {
    raw.push_back(0);
    const uint8_t* r = im.row(y);
    for (int x = 0; x < im.w; x++) raw.insert(raw.end(), r + x * 4, r + x * 4 + 3);
  }
  uLongf zlen = compressBound(raw.size());
  std::vector<uint8_t> z(zlen);
  if (compress2(z.data(), &zlen, raw.data(), raw.size(), 6) != Z_OK) throw std::runtime_error("png: zlib failed");
  z.resize(zlen);
  std::ofstream f(path, std::ios::binary);
  const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  f.write(reinterpret_cast<const char*>(sig), 8);
  std::vector<uint8_t> ihdr = {uint8_t(im.w >> 24), uint8_t(im.w >> 16), uint8_t(im.w >> 8), uint8_t(im.w),
                               uint8_t(im.h >> 24), uint8_t(im.h >> 16), uint8_t(im.h >> 8), uint8_t(im.h),
                               8, 2, 0, 0, 0};
  crc_chunk(f, "IHDR", ihdr);
  crc_chunk(f, "IDAT", z);
  crc_chunk(f, "IEND", {});
}

}  // namespace idol

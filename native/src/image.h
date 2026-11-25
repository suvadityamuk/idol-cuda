// Ports of the Pillow 13.0.0 operations idol/preprocess.py uses, bit-exact with Pillow's C code
// (libImaging/Resample.c, Convert.c, Paste.c, RankFilter.c, Crop). Compile with -ffp-contract=off.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace idol {

// Pillow layout: "L" = 1 byte per pixel; "RGB"/"RGBA"/"RGBa" = 4 bytes per pixel (RGB keeps a padding byte).
struct Img {
  int w = 0, h = 0;
  std::string mode;  // "L", "RGB", "RGBA", "RGBa"
  std::vector<uint8_t> px;
  int pixelsize() const { return mode == "L" ? 1 : 4; }
  uint8_t* row(int y) { return px.data() + static_cast<size_t>(y) * w * pixelsize(); }
  const uint8_t* row(int y) const { return px.data() + static_cast<size_t>(y) * w * pixelsize(); }
};

Img new_image(const std::string& mode, int w, int h, uint8_t fill = 0);
// WebP decoding as Pillow 13 does it (WebPAnimDecoder, default options, RGBA canvas; "RGB" when no alpha).
Img decode_webp(const std::string& path);
// Image.resize(size, Image.LANCZOS) with the default box (handles RGBA via RGBa premultiplication like Pillow).
Img resize_lanczos(const Img& im, int w, int h);
Img convert(const Img& im, const std::string& mode);
Img crop(const Img& im, int x0, int y0, int x1, int y1);
Img getchannel(const Img& im, int band);
void putalpha(Img& im, const Img& alpha);              // im becomes RGBA
Img min_filter3(const Img& l);                          // ImageFilter.MinFilter(3) on an "L" image
void paste_with_mask(Img& dst, const Img& src, const Img& mask);  // dst.paste(src, mask=mask), same size, at (0,0)
bool has_real_alpha(const Img& im);
void write_png_rgb(const std::string& path, const Img& im);       // zlib PNG (pixels match; bytes differ from Pillow)

}  // namespace idol

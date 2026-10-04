// PNG loading, RGB -> YUV 4:2:0 conversion, and PSNR. No Jetson dependencies.
#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>
#include <string>
#include <vector>

struct RgbImage {
  int width = 0, height = 0;
  std::vector<uint8_t> px;       // width*height*3, packed RGB
  unsigned long file_size = 0;   // on-disk size of the source file
};

struct Yuv420 {
  int width = 0, height = 0;     // luma size; chroma planes are half each way
  std::vector<uint8_t> y, u, v;
};

struct Psnr {
  double y = 0, u = 0, v = 0;
  double yuv = 0;                // (6*Y + U + V) / 8
};

bool load_png(const std::string &path, RgbImage &out, std::string &err);

// Full-range (JFIF) BT.601 YCbCr, chroma averaged over 2x2. The image is
// cropped (top-left anchored) so both dimensions are multiples of `align`.
Yuv420 rgb_to_yuv420(const RgbImage &rgb, int align);

Psnr psnr(const Yuv420 &ref, const Yuv420 &test);

// Image files in `path` if it is a directory (sorted), else `path` itself.
std::vector<std::string> collect_images(const std::string &path);

#endif

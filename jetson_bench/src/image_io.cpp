#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <dirent.h>
#include <png.h>
#include <sys/stat.h>

bool load_png(const std::string &path, RgbImage &out, std::string &err) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    err = "cannot stat file";
    return false;
  }

  png_image image;
  memset(&image, 0, sizeof(image));
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, path.c_str())) {
    err = image.message;
    return false;
  }
  image.format = PNG_FORMAT_RGB;
  out.px.resize(PNG_IMAGE_SIZE(image));
  if (!png_image_finish_read(&image, NULL, out.px.data(), 0, NULL)) {
    err = image.message;
    png_image_free(&image);
    return false;
  }
  out.width = (int)image.width;
  out.height = (int)image.height;
  out.file_size = (unsigned long)st.st_size;
  return true;
}

static uint8_t clamp8(double v) {
  return (uint8_t)(v < 0.0 ? 0 : v > 255.0 ? 255 : v + 0.5);
}

Yuv420 rgb_to_yuv420(const RgbImage &rgb, int align) {
  Yuv420 out;
  out.width = rgb.width / align * align;
  out.height = rgb.height / align * align;
  int w = out.width, h = out.height, cw = w / 2, ch = h / 2;
  out.y.resize((size_t)w * h);
  out.u.resize((size_t)cw * ch);
  out.v.resize((size_t)cw * ch);

  for (int y = 0; y < h; y++) {
    const uint8_t *row = &rgb.px[(size_t)y * rgb.width * 3];
    uint8_t *dst = &out.y[(size_t)y * w];
    for (int x = 0; x < w; x++)
      dst[x] = clamp8(0.299 * row[x * 3] + 0.587 * row[x * 3 + 1] +
                      0.114 * row[x * 3 + 2]);
  }

  for (int y = 0; y < ch; y++) {
    const uint8_t *row0 = &rgb.px[(size_t)(2 * y) * rgb.width * 3];
    const uint8_t *row1 = row0 + (size_t)rgb.width * 3;
    for (int x = 0; x < cw; x++) {
      const uint8_t *a = row0 + x * 6, *b = row1 + x * 6;
      double r = (a[0] + a[3] + b[0] + b[3]) / 4.0;
      double g = (a[1] + a[4] + b[1] + b[4]) / 4.0;
      double bl = (a[2] + a[5] + b[2] + b[5]) / 4.0;
      out.u[(size_t)y * cw + x] =
          clamp8(-0.168736 * r - 0.331264 * g + 0.5 * bl + 128.0);
      out.v[(size_t)y * cw + x] =
          clamp8(0.5 * r - 0.418688 * g - 0.081312 * bl + 128.0);
    }
  }
  return out;
}

static double plane_psnr(const std::vector<uint8_t> &a,
                         const std::vector<uint8_t> &b) {
  double sse = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    int d = (int)a[i] - (int)b[i];
    sse += d * d;
  }
  if (sse == 0.0) return 100.0;
  return 10.0 * std::log10(255.0 * 255.0 * (double)a.size() / sse);
}

Psnr psnr(const Yuv420 &ref, const Yuv420 &test) {
  Psnr p;
  p.y = plane_psnr(ref.y, test.y);
  p.u = plane_psnr(ref.u, test.u);
  p.v = plane_psnr(ref.v, test.v);
  p.yuv = (6.0 * p.y + p.u + p.v) / 8.0;
  return p;
}

std::vector<std::string> collect_images(const std::string &path) {
  std::vector<std::string> files;
  DIR *dir = opendir(path.c_str());
  if (!dir) {
    files.push_back(path);
    return files;
  }
  while (struct dirent *entry = readdir(dir)) {
    std::string name = entry->d_name;
    size_t dot = name.rfind('.');
    if (dot == std::string::npos) continue;
    std::string ext = name.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".png") files.push_back(path + "/" + name);
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

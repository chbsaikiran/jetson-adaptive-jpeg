#include "features.h"

#include <cmath>
#include <cstddef>
#include <vector>

extern "C" void score(double *input, double *output);  // complexity_model.c

namespace {

double shannon_entropy(const unsigned long hist[256], size_t n) {
  double entropy = 0.0;
  for (int b = 0; b < 256; b++) {
    if (!hist[b]) continue;
    double p = (double)hist[b] / (double)n;
    entropy -= p * std::log2(p);
  }
  return entropy;
}

// scipy.ndimage.sobel's default "reflect" boundary repeats the edge sample,
// which for a 3-tap kernel is the same as clamping the coordinate.
void sobel_stats(const std::vector<uint8_t> &gray, int w, int h,
                 double *density, double *variance) {
  double sum = 0.0, sum_sq = 0.0;
  for (int y = 0; y < h; y++) {
    const uint8_t *r0 = &gray[(size_t)(y > 0 ? y - 1 : 0) * w];
    const uint8_t *r1 = &gray[(size_t)y * w];
    const uint8_t *r2 = &gray[(size_t)(y < h - 1 ? y + 1 : y) * w];
    for (int x = 0; x < w; x++) {
      int xl = x > 0 ? x - 1 : 0, xr = x < w - 1 ? x + 1 : x;
      int gx = (r0[xr] + 2 * r1[xr] + r2[xr]) - (r0[xl] + 2 * r1[xl] + r2[xl]);
      int gy = (r2[xl] + 2 * r2[x] + r2[xr]) - (r0[xl] + 2 * r0[x] + r0[xr]);
      double mag_sq = (double)gx * gx + (double)gy * gy;
      sum += std::sqrt(mag_sq);
      sum_sq += mag_sq;
    }
  }
  double n = (double)w * h, mean = sum / n;
  *density = mean;
  *variance = sum_sq / n - mean * mean;
}

// Orthonormal 8-point DCT-II, as scipy.fftpack.dct(..., norm="ortho").
struct DctTable {
  double c[8][8];
  DctTable() {
    const double pi = 3.14159265358979323846;
    for (int k = 0; k < 8; k++)
      for (int n = 0; n < 8; n++)
        c[k][n] = std::cos(pi / 8.0 * (n + 0.5) * k) *
                  std::sqrt((k == 0 ? 1.0 : 2.0) / 8.0);
  }
};

double mean_dct_ac_energy(const std::vector<uint8_t> &gray, int w, int h) {
  static const DctTable table;
  int h8 = (h / 8) * 8, w8 = (w / 8) * 8;
  double total = 0.0;
  long blocks = 0;

  for (int by = 0; by < h8; by += 8) {
    for (int bx = 0; bx < w8; bx += 8) {
      double tmp[8][8];
      for (int r = 0; r < 8; r++) {
        const uint8_t *row = &gray[(size_t)(by + r) * w + bx];
        for (int k = 0; k < 8; k++) {
          double s = 0.0;
          for (int n = 0; n < 8; n++) s += row[n] * table.c[k][n];
          tmp[r][k] = s;
        }
      }
      double energy = 0.0;
      for (int c = 0; c < 8; c++) {
        for (int k = 0; k < 8; k++) {
          double s = 0.0;
          for (int n = 0; n < 8; n++) s += tmp[n][c] * table.c[k][n];
          if (k || c) energy += std::fabs(s);
        }
      }
      total += energy;
      blocks++;
    }
  }
  return blocks ? total / (double)blocks : 0.0;
}

}  // namespace

void extract_features(const uint8_t *rgb, int width, int height,
                      unsigned long file_size_bytes, int downsample,
                      double features[NUM_FEATURES]) {
  int step = downsample > 1 ? downsample : 1;
  int dw = (width + step - 1) / step, dh = (height + step - 1) / step;
  size_t n = (size_t)dw * dh;

  std::vector<uint8_t> gray(n);
  unsigned long hist[256] = {0};
  double sum[3] = {0, 0, 0}, sum_sq[3] = {0, 0, 0}, sat_sum = 0.0;
  size_t out = 0;

  for (int y = 0; y < height; y += step) {
    const uint8_t *row = rgb + (size_t)y * width * 3;
    for (int x = 0; x < width; x += step) {
      int r = row[x * 3], g = row[x * 3 + 1], b = row[x * 3 + 2];

      // Pillow's RGB -> "L" fixed-point weights.
      uint8_t l = (uint8_t)((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16);
      gray[out++] = l;
      hist[l]++;

      sum[0] += r; sum[1] += g; sum[2] += b;
      sum_sq[0] += r * r; sum_sq[1] += g * g; sum_sq[2] += b * b;

      // Pillow's RGB -> HSV saturation: float divide, truncated to 0-255.
      int maxc = r > g ? (r > b ? r : b) : (g > b ? g : b);
      int minc = r < g ? (r < b ? r : b) : (g < b ? g : b);
      if (maxc != minc)
        sat_sum += (int)(((float)(maxc - minc) / (float)maxc) * 255.0);
    }
  }

  double megapixels = (double)width * (double)height / 1e6;
  features[0] = (double)file_size_bytes / megapixels;
  features[1] = shannon_entropy(hist, n);
  sobel_stats(gray, dw, dh, &features[2], &features[3]);
  for (int c = 0; c < 3; c++) {
    double mean = sum[c] / (double)n;
    features[4 + c] = sum_sq[c] / (double)n - mean * mean;
  }
  features[7] = sat_sum / (double)n;
  features[8] = mean_dct_ac_energy(gray, dw, dh);
}

int classify(const double features[NUM_FEATURES], double proba[2]) {
  // scikit-learn's trees compare float32 inputs against their thresholds.
  double input[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; i++) input[i] = (double)(float)features[i];
  score(input, proba);
  return proba[1] > proba[0] ? 1 : 0;
}

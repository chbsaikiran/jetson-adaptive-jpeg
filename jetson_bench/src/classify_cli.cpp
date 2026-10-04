// Classify images with the C++ feature port + exported forest. Builds
// anywhere (no Jetson libraries) -- use it to cross-check against
// Random-Forest/predict.py, which prints the same lines.
//
//   classify_cli [--features] <image-or-dir>...
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "features.h"
#include "image_io.h"

int main(int argc, char **argv) {
  static const char *const names[NUM_FEATURES] = {
      "bytes_per_mpx", "entropy",    "edge_density",    "edge_variance",
      "r_variance",    "g_variance", "b_variance",      "mean_saturation",
      "dct_ac_energy"};
  bool show_features = false;
  std::vector<std::string> images;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--features") == 0) {
      show_features = true;
    } else {
      std::vector<std::string> found = collect_images(argv[i]);
      images.insert(images.end(), found.begin(), found.end());
    }
  }
  if (images.empty()) {
    fprintf(stderr, "usage: %s [--features] <image-or-dir>...\n", argv[0]);
    return 2;
  }

  int simple = 0, total = 0;
  for (size_t i = 0; i < images.size(); i++) {
    RgbImage rgb;
    std::string err;
    if (!load_png(images[i], rgb, err)) {
      fprintf(stderr, "%s: skipped (%s)\n", images[i].c_str(), err.c_str());
      continue;
    }
    double features[NUM_FEATURES], proba[2];
    extract_features(rgb.px.data(), rgb.width, rgb.height, rgb.file_size,
                     MODEL_DOWNSAMPLE, features);
    int label = classify(features, proba);
    printf("%s: predicted %s (P(simple)=%.3f P(complex)=%.3f)\n",
           images[i].c_str(), label == 0 ? "simple" : "complex", proba[0],
           proba[1]);
    if (show_features)
      for (int f = 0; f < NUM_FEATURES; f++)
        printf("    %-16s %.4f\n", names[f], features[f]);
    total++;
    simple += label == 0;
  }
  if (total > 1)
    printf("\n%d images: %d simple, %d complex\n", total, simple,
           total - simple);
  return total ? 0 : 1;
}

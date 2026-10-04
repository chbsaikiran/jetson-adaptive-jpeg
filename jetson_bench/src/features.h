// Pre-encode image features + Random Forest inference, in plain C++.
//
// Port of Random-Forest/extract_features.py. The 9 features, in the order
// complexity_model.c's score() expects them:
//   [0] bytes_per_mpx    source file size per megapixel of the source image
//   [1] entropy          grayscale histogram Shannon entropy
//   [2] edge_density     mean Sobel gradient magnitude
//   [3] edge_variance    variance of Sobel gradient magnitude
//   [4] r_variance  [5] g_variance  [6] b_variance
//   [7] mean_saturation  mean HSV saturation (0-255, Pillow convention)
//   [8] dct_ac_energy    mean |AC coefficient| sum per 8x8 block
#ifndef FEATURES_H
#define FEATURES_H

#include <cstdint>

#define NUM_FEATURES 9

// The factor the shipped model was trained at (DEFAULT_DOWNSAMPLE in
// extract_features.py). Texture features shift with the factor, so the
// model is only valid for features extracted at this one.
#define MODEL_DOWNSAMPLE 4

// `rgb` is width*height packed 8-bit RGB. Every feature except
// bytes_per_mpx is computed from every `downsample`-th pixel of every
// `downsample`-th row.
void extract_features(const uint8_t *rgb, int width, int height,
                      unsigned long file_size_bytes, int downsample,
                      double features[NUM_FEATURES]);

// Runs the forest. proba[0] = P(simple), proba[1] = P(complex).
// Returns the label: 0 = simple, 1 = complex.
int classify(const double features[NUM_FEATURES], double proba[2]);

#endif

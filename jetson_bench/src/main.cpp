// jpeg_energy_bench -- energy and PSNR of classifier-routed JPEG round trips
// on a Jetson, using the hardware JPEG codec (NVJPG) and scaler (VIC).
//
// Two round trips are measured for every image, both starting from the same
// full-size YUV 4:2:0 hardware buffer:
//
//   direct       NVJPG encode -> NVJPG decode
//   downsampled  VIC scale 1/2 -> NVJPG encode -> NVJPG decode -> VIC scale x2
//
// plus the classifier on its own (feature extraction + Random Forest, CPU).
// From those:
//
//   baseline  = direct, for every image
//   adaptive  = classifier + (downsampled if predicted simple, else direct)
//
// A single round trip is far shorter than the power sensor's refresh time,
// so each stage is repeated for --min-time seconds while a background thread
// samples the board's input power rail; energy per image is mean power x
// time per repeat. PSNR is taken in the YUV domain, between the source
// buffer and what each round trip hands back.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "NvJpegDecoder.h"
#include "NvJpegEncoder.h"
#include "nvbuf_utils.h"

#include "features.h"
#include "image_io.h"
#include "power.h"

// Source images are cropped so that both the full-size and the half-size
// frame are whole 16x16 JPEG MCUs -- neither round trip pays for edge
// padding the other doesn't.
static const int CROP_ALIGN = 32;

typedef std::chrono::steady_clock Clock;

static double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static void check(int ret, const char *what) {
  if (ret != 0) throw std::runtime_error(std::string(what) + " failed");
}

// --- hardware buffers ------------------------------------------------------

struct HwBuffer {
  int fd = -1;
  HwBuffer(int width, int height) {
    NvBufferCreateParams params;
    memset(&params, 0, sizeof(params));
    params.width = width;
    params.height = height;
    params.payloadType = NvBufferPayload_SurfArray;
    params.layout = NvBufferLayout_Pitch;
    params.colorFormat = NvBufferColorFormat_YUV420;
    params.nvbuf_tag = NvBufferTag_NONE;
    check(NvBufferCreateEx(&fd, &params), "NvBufferCreateEx");
  }
  ~HwBuffer() {
    if (fd >= 0) NvBufferDestroy(fd);
  }
  HwBuffer(const HwBuffer &) = delete;
  HwBuffer &operator=(const HwBuffer &) = delete;
};

static void upload(int fd, const Yuv420 &img) {
  NvBufferParams par;
  check(NvBufferGetParams(fd, &par), "NvBufferGetParams");
  const uint8_t *planes[3] = {img.y.data(), img.u.data(), img.v.data()};
  for (int p = 0; p < 3; p++) {
    int w = p ? img.width / 2 : img.width, h = p ? img.height / 2 : img.height;
    void *va = NULL;
    check(NvBufferMemMap(fd, p, NvBufferMem_Write, &va), "NvBufferMemMap");
    for (int r = 0; r < h; r++)
      memcpy((uint8_t *)va + (size_t)r * par.pitch[p], planes[p] + (size_t)r * w, w);
    NvBufferMemSyncForDevice(fd, p, &va);
    NvBufferMemUnMap(fd, p, &va);
  }
}

static void vic_scale(int src_fd, int dst_fd, NvBufferTransform_Filter filter) {
  NvBufferTransformParams params;
  memset(&params, 0, sizeof(params));
  params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
  params.transform_flip = NvBufferTransform_None;
  params.transform_filter = filter;
  check(NvBufferTransform(src_fd, dst_fd, &params), "NvBufferTransform");
}

// Copies a width x height YUV420 frame out of `fd`. If the buffer isn't
// planar YUV420 (the decoder picks its own output format), it is first
// converted into `scratch_fd`, a YUV420 buffer of the same size.
static Yuv420 download(int fd, int width, int height, int scratch_fd) {
  NvBufferParams par;
  check(NvBufferGetParams(fd, &par), "NvBufferGetParams");
  if (par.pixel_format != NvBufferColorFormat_YUV420 || par.num_planes != 3) {
    vic_scale(fd, scratch_fd, NvBufferTransform_Filter_Nearest);
    fd = scratch_fd;
    check(NvBufferGetParams(fd, &par), "NvBufferGetParams");
  }

  Yuv420 img;
  img.width = width;
  img.height = height;
  img.y.resize((size_t)width * height);
  img.u.resize((size_t)(width / 2) * (height / 2));
  img.v.resize(img.u.size());
  uint8_t *planes[3] = {img.y.data(), img.u.data(), img.v.data()};
  for (int p = 0; p < 3; p++) {
    int w = p ? width / 2 : width, h = p ? height / 2 : height;
    if ((int)par.width[p] < w || (int)par.height[p] < h)
      throw std::runtime_error("decoded frame is smaller than the source");
    void *va = NULL;
    check(NvBufferMemMap(fd, p, NvBufferMem_Read, &va), "NvBufferMemMap");
    NvBufferMemSyncForCpu(fd, p, &va);
    for (int r = 0; r < h; r++)
      memcpy(planes[p] + (size_t)r * w, (uint8_t *)va + (size_t)r * par.pitch[p], w);
    NvBufferMemUnMap(fd, p, &va);
  }
  return img;
}

// --- hardware JPEG round trip ----------------------------------------------

struct JpegCodec {
  NvJPEGEncoder *enc;
  NvJPEGDecoder *dec;
  std::vector<unsigned char> jpeg;
  unsigned long jpeg_size = 0;
  int dec_fd = -1;  // owned by the decoder; valid until its next decode

  JpegCodec(const char *name, int width, int height)
      : enc(NvJPEGEncoder::createJPEGEncoder(name)),
        dec(NvJPEGDecoder::createJPEGDecoder(name)),
        jpeg((size_t)width * height * 3 / 2) {
    if (!enc || !dec) throw std::runtime_error("cannot create NVJPG codec");
  }
  ~JpegCodec() {
    delete enc;
    delete dec;
  }
  JpegCodec(const JpegCodec &) = delete;
  JpegCodec &operator=(const JpegCodec &) = delete;

  void roundtrip(int src_fd, int quality) {
    unsigned char *out = jpeg.data();
    unsigned long size = jpeg.size();
    check(enc->encodeFromFd(src_fd, JCS_YCbCr, &out, size, quality),
          "NVJPG encode");
    if (out != jpeg.data())
      throw std::runtime_error("JPEG larger than the output buffer");
    jpeg_size = size;

    uint32_t pixfmt = 0, width = 0, height = 0;
    check(dec->decodeToFd(dec_fd, out, size, pixfmt, width, height),
          "NVJPG decode");
  }
};

// --- measurement -----------------------------------------------------------

struct Meas {
  int iters = 0;
  double ms = 0.0;   // wall time per repeat
  double mw = NAN;   // mean board power while repeating
  double gross_mj() const { return mw * ms / 1000.0; }
  double net_mj(double idle_mw) const { return (mw - idle_mw) * ms / 1000.0; }
};

// Repeats `fn` for `min_s` seconds. The first quarter of that again is run
// beforehand and thrown away, so the sensor's averaging window has moved
// onto this workload before sampling starts.
template <class F>
static Meas measure(PowerSampler &power, double min_s, F fn) {
  Clock::time_point t0 = Clock::now();
  do fn(); while (seconds_since(t0) < min_s * 0.25);

  Meas m;
  power.begin();
  t0 = Clock::now();
  do {
    fn();
    m.iters++;
  } while (seconds_since(t0) < min_s);
  double elapsed = seconds_since(t0);
  m.mw = power.end();
  m.ms = elapsed * 1000.0 / m.iters;
  return m;
}

static double measure_idle(PowerSampler &power, double seconds) {
  power.begin();
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  return power.end();
}

// --- per-image result ------------------------------------------------------

struct Row {
  std::string name;
  int width = 0, height = 0;
  double p_simple = 0.0;
  int label = 1;  // 0 = simple, 1 = complex
  Meas classify, direct, down;
  unsigned long bytes_direct = 0, bytes_down = 0;
  Psnr psnr_direct, psnr_down;

  bool simple() const { return label == 0; }
  const Meas &routed() const { return simple() ? down : direct; }
  const Psnr &routed_psnr() const { return simple() ? psnr_down : psnr_direct; }
};

static Row run_image(const std::string &path, size_t index, PowerSampler &power,
                     double min_s, int quality,
                     NvBufferTransform_Filter filter) {
  RgbImage rgb;
  std::string err;
  if (!load_png(path, rgb, err)) throw std::runtime_error(err);
  Yuv420 src = rgb_to_yuv420(rgb, CROP_ALIGN);
  if (src.width == 0 || src.height == 0)
    throw std::runtime_error("image too small");

  Row row;
  size_t slash = path.rfind('/');
  row.name = slash == std::string::npos ? path : path.substr(slash + 1);
  row.width = src.width;
  row.height = src.height;

  int w = src.width, h = src.height;
  HwBuffer full(w, h), half(w / 2, h / 2), restored(w, h);
  upload(full.fd, src);
  JpegCodec direct_codec("jpeg_direct", w, h);
  JpegCodec down_codec("jpeg_down", w / 2, h / 2);

  // Classifier: features from the image as loaded (uncropped, like
  // Random-Forest/predict.py) + forest inference.
  double features[NUM_FEATURES], proba[2] = {0.0, 0.0};
  row.classify = measure(power, min_s, [&] {
    extract_features(rgb.px.data(), rgb.width, rgb.height, rgb.file_size,
                     MODEL_DOWNSAMPLE, features);
    row.label = classify(features, proba);
  });
  row.p_simple = proba[0];

  auto run_direct = [&] {
    row.direct = measure(power, min_s,
                         [&] { direct_codec.roundtrip(full.fd, quality); });
    row.bytes_direct = direct_codec.jpeg_size;
    row.psnr_direct = psnr(src, download(direct_codec.dec_fd, w, h, restored.fd));
  };
  auto run_down = [&] {
    row.down = measure(power, min_s, [&] {
      vic_scale(full.fd, half.fd, filter);
      down_codec.roundtrip(half.fd, quality);
      vic_scale(down_codec.dec_fd, restored.fd, filter);
    });
    row.bytes_down = down_codec.jpeg_size;
    row.psnr_down = psnr(src, download(restored.fd, w, h, restored.fd));
  };

  // Alternate the order so neither round trip always inherits the other's
  // heat and clock state.
  if (index % 2 == 0) {
    run_direct();
    run_down();
  } else {
    run_down();
    run_direct();
  }
  return row;
}

// --- reporting -------------------------------------------------------------

static void write_csv(const std::string &path, const std::vector<Row> &rows,
                      double idle_mw) {
  FILE *f = fopen(path.c_str(), "w");
  if (!f) {
    fprintf(stderr, "cannot write %s\n", path.c_str());
    return;
  }
  fprintf(f,
          "image,width,height,p_simple,predicted,"
          "classify_ms,direct_ms,down_ms,"
          "classify_mw,direct_mw,down_mw,idle_mw,"
          "classify_gross_mj,direct_gross_mj,down_gross_mj,"
          "classify_net_mj,direct_net_mj,down_net_mj,"
          "baseline_gross_mj,adaptive_gross_mj,baseline_net_mj,adaptive_net_mj,"
          "jpeg_bytes_direct,jpeg_bytes_down,"
          "psnr_y_direct,psnr_y_down,psnr_yuv_direct,psnr_yuv_down,"
          "psnr_y_adaptive,psnr_yuv_adaptive\n");
  for (size_t i = 0; i < rows.size(); i++) {
    const Row &r = rows[i];
    fprintf(f,
            "%s,%d,%d,%.4f,%s,"
            "%.4f,%.4f,%.4f,"
            "%.1f,%.1f,%.1f,%.1f,"
            "%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,"
            "%lu,%lu,"
            "%.3f,%.3f,%.3f,%.3f,"
            "%.3f,%.3f\n",
            r.name.c_str(), r.width, r.height, r.p_simple,
            r.simple() ? "simple" : "complex",
            r.classify.ms, r.direct.ms, r.down.ms,
            r.classify.mw, r.direct.mw, r.down.mw, idle_mw,
            r.classify.gross_mj(), r.direct.gross_mj(), r.down.gross_mj(),
            r.classify.net_mj(idle_mw), r.direct.net_mj(idle_mw),
            r.down.net_mj(idle_mw),
            r.direct.gross_mj(),
            r.classify.gross_mj() + r.routed().gross_mj(),
            r.direct.net_mj(idle_mw),
            r.classify.net_mj(idle_mw) + r.routed().net_mj(idle_mw),
            r.bytes_direct, r.bytes_down,
            r.psnr_direct.y, r.psnr_down.y, r.psnr_direct.yuv, r.psnr_down.yuv,
            r.routed_psnr().y, r.routed_psnr().yuv);
  }
  fclose(f);
  printf("\nper-image results -> %s\n", path.c_str());
}

static void print_compare(const char *label, double baseline, double adaptive,
                          const char *unit) {
  printf("  %-22s baseline %10.2f %s   adaptive %10.2f %s   saving %+.1f%%\n",
         label, baseline, unit, adaptive, unit,
         100.0 * (baseline - adaptive) / baseline);
}

static void print_summary(const std::vector<Row> &rows, double idle_mw,
                          bool have_power) {
  size_t n = rows.size(), n_simple = 0;
  // Per-image sums: baseline vs adaptive, over all images.
  double t_base = 0, t_adapt = 0, g_base = 0, g_adapt = 0, n_base = 0, n_adapt = 0;
  double g_classify = 0, g_all_down = 0;
  double psnr_base = 0, psnr_adapt = 0, bytes_base = 0, bytes_adapt = 0;
  // Sums over the images routed to the downsampled round trip.
  double s_g_direct = 0, s_g_down = 0, s_g_classify = 0;
  double s_n_direct = 0, s_n_down = 0, s_n_classify = 0;
  double s_y_direct = 0, s_y_down = 0, s_yuv_direct = 0, s_yuv_down = 0;
  double s_loss_min = 1e9, s_loss_max = -1e9;

  for (size_t i = 0; i < n; i++) {
    const Row &r = rows[i];
    t_base += r.direct.ms;
    t_adapt += r.classify.ms + r.routed().ms;
    g_base += r.direct.gross_mj();
    g_adapt += r.classify.gross_mj() + r.routed().gross_mj();
    n_base += r.direct.net_mj(idle_mw);
    n_adapt += r.classify.net_mj(idle_mw) + r.routed().net_mj(idle_mw);
    g_classify += r.classify.gross_mj();
    g_all_down += r.down.gross_mj();
    psnr_base += r.psnr_direct.y;
    psnr_adapt += r.routed_psnr().y;
    bytes_base += r.bytes_direct;
    bytes_adapt += r.simple() ? r.bytes_down : r.bytes_direct;
    if (!r.simple()) continue;

    n_simple++;
    s_g_direct += r.direct.gross_mj();
    s_g_down += r.down.gross_mj();
    s_g_classify += r.classify.gross_mj();
    s_n_direct += r.direct.net_mj(idle_mw);
    s_n_down += r.down.net_mj(idle_mw);
    s_n_classify += r.classify.net_mj(idle_mw);
    s_y_direct += r.psnr_direct.y;
    s_y_down += r.psnr_down.y;
    s_yuv_direct += r.psnr_direct.yuv;
    s_yuv_down += r.psnr_down.yuv;
    double loss = r.psnr_direct.y - r.psnr_down.y;
    if (loss < s_loss_min) s_loss_min = loss;
    if (loss > s_loss_max) s_loss_max = loss;
  }

  printf("\n================ summary ================\n");
  printf("%zu images: %zu predicted simple (downsampled), %zu complex (direct)\n",
         n, n_simple, n - n_simple);

  printf("\nAll images, totals (adaptive includes the classifier):\n");
  print_compare("time", t_base, t_adapt, "ms");
  if (have_power) {
    print_compare("energy, gross", g_base, g_adapt, "mJ");
    print_compare("energy, above idle", n_base, n_adapt, "mJ");
    printf("  classifier share of adaptive gross energy: %.1f%%\n",
           100.0 * g_classify / g_adapt);
    printf("  for reference, downsampling every image: %.2f mJ gross\n",
           g_all_down);
  }
  print_compare("JPEG size", bytes_base / 1024.0, bytes_adapt / 1024.0, "KB");
  printf("  mean PSNR-Y: baseline %.2f dB, adaptive %.2f dB\n",
         psnr_base / n, psnr_adapt / n);

  if (n_simple == 0) {
    printf("\nNo image was predicted simple -- nothing was downsampled.\n");
    return;
  }
  double k = (double)n_simple;
  printf("\nSimple images only (%zu), mean per image:\n", n_simple);
  if (have_power) {
    printf("  energy, gross:      direct %.2f mJ   classifier %.2f + downsampled %.2f mJ"
           "   saving %+.1f%%\n",
           s_g_direct / k, s_g_classify / k, s_g_down / k,
           100.0 * (s_g_direct - s_g_classify - s_g_down) / s_g_direct);
    printf("  energy, above idle: direct %.2f mJ   classifier %.2f + downsampled %.2f mJ"
           "   saving %+.1f%%\n",
           s_n_direct / k, s_n_classify / k, s_n_down / k,
           100.0 * (s_n_direct - s_n_classify - s_n_down) / s_n_direct);
  }
  printf("  PSNR-Y:   direct %.2f dB   downsampled %.2f dB   loss %.2f dB"
         " (min %.2f, max %.2f)\n",
         s_y_direct / k, s_y_down / k, (s_y_direct - s_y_down) / k, s_loss_min,
         s_loss_max);
  printf("  PSNR-YUV: direct %.2f dB   downsampled %.2f dB   loss %.2f dB\n",
         s_yuv_direct / k, s_yuv_down / k, (s_yuv_direct - s_yuv_down) / k);
}

// --- main ------------------------------------------------------------------

static bool parse_filter(const std::string &name, NvBufferTransform_Filter *out) {
  if (name == "nearest") *out = NvBufferTransform_Filter_Nearest;
  else if (name == "bilinear") *out = NvBufferTransform_Filter_Bilinear;
  else if (name == "5tap") *out = NvBufferTransform_Filter_5_Tap;
  else if (name == "10tap") *out = NvBufferTransform_Filter_10_Tap;
  else if (name == "smart") *out = NvBufferTransform_Filter_Smart;
  else if (name == "nicest") *out = NvBufferTransform_Filter_Nicest;
  else return false;
  return true;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "usage: %s [options] <image.png | directory>...\n"
          "  --quality Q       JPEG quality, 1-100 (default 85)\n"
          "  --min-time S      seconds each stage is repeated for (default 1.0)\n"
          "  --limit N         only the first N images\n"
          "  --filter F        VIC scaling filter: nearest, bilinear, 5tap,\n"
          "                    10tap, smart, nicest (default smart)\n"
          "  --csv FILE        per-image results (default results.csv)\n"
          "  --sample-ms MS    power sampling period (default 10)\n"
          "  --power-file FILE read power (mW) from this file instead of\n"
          "                    auto-detecting the INA3221 input rail\n",
          prog);
}

int main(int argc, char **argv) {
  int quality = 85, sample_ms = 10;
  long limit = -1;
  double min_s = 1.0;
  std::string csv_path = "results.csv", power_file, filter_name = "smart";
  std::vector<std::string> images;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    bool has_value = i + 1 < argc;
    if (arg == "--quality" && has_value) quality = atoi(argv[++i]);
    else if (arg == "--min-time" && has_value) min_s = atof(argv[++i]);
    else if (arg == "--limit" && has_value) limit = atol(argv[++i]);
    else if (arg == "--filter" && has_value) filter_name = argv[++i];
    else if (arg == "--csv" && has_value) csv_path = argv[++i];
    else if (arg == "--sample-ms" && has_value) sample_ms = atoi(argv[++i]);
    else if (arg == "--power-file" && has_value) power_file = argv[++i];
    else if (arg.compare(0, 2, "--") == 0) { usage(argv[0]); return 2; }
    else {
      std::vector<std::string> found = collect_images(arg);
      images.insert(images.end(), found.begin(), found.end());
    }
  }
  NvBufferTransform_Filter filter;
  if (images.empty() || !parse_filter(filter_name, &filter) || quality < 1 ||
      quality > 100 || min_s <= 0.0 || sample_ms < 1) {
    usage(argv[0]);
    return 2;
  }
  if (limit >= 0 && (size_t)limit < images.size()) images.resize(limit);

  PowerSampler power;
  bool have_power = power.open(power_file);
  if (have_power) {
    power.start(sample_ms);
    printf("power rail: %s, sampled every %d ms\n", power.rail().c_str(), sample_ms);
  } else {
    printf("WARNING: no power sensor found -- time and PSNR only, energy "
           "columns will be nan\n");
  }
  printf("quality %d, VIC filter %s, %.2f s per stage, %zu images\n", quality,
         filter_name.c_str(), min_s, images.size());

  double idle_before = measure_idle(power, 3.0);
  if (have_power) printf("idle power before: %.0f mW\n", idle_before);

  std::vector<Row> rows;
  for (size_t i = 0; i < images.size(); i++) {
    try {
      Row r = run_image(images[i], i, power, min_s, quality, filter);
      printf("[%zu/%zu] %s %dx%d %-7s P(simple)=%.3f | direct %.2f ms %.2f mJ "
             "%.2f dB | down %.2f ms %.2f mJ %.2f dB | classify %.2f ms %.2f mJ\n",
             i + 1, images.size(), r.name.c_str(), r.width, r.height,
             r.simple() ? "simple" : "complex", r.p_simple, r.direct.ms,
             r.direct.gross_mj(), r.psnr_direct.y, r.down.ms, r.down.gross_mj(),
             r.psnr_down.y, r.classify.ms, r.classify.gross_mj());
      fflush(stdout);
      rows.push_back(r);
    } catch (const std::exception &e) {
      fprintf(stderr, "%s: skipped (%s)\n", images[i].c_str(), e.what());
    }
  }

  double idle_after = measure_idle(power, 3.0);
  double idle_mw = (idle_before + idle_after) / 2.0;
  if (have_power)
    printf("idle power after: %.0f mW (using mean %.0f mW)\n", idle_after, idle_mw);
  power.stop();

  if (rows.empty()) {
    fprintf(stderr, "no image could be processed\n");
    return 1;
  }
  write_csv(csv_path, rows, idle_mw);
  print_summary(rows, idle_mw, have_power);
  return 0;
}

# Jetson Nano energy / PSNR benchmark

Measures whether routing images through the Random Forest classifier saves
energy on the Jetson Nano's hardware JPEG path, and what it costs in PSNR.

| | pipeline |
|---|---|
| **baseline** | every image: NVJPG encode → NVJPG decode |
| **adaptive** | classifier (CPU) → *simple*: VIC ½ → NVJPG encode → NVJPG decode → VIC ×2; *complex*: NVJPG encode → NVJPG decode |

`jpeg_energy_bench` runs three stages on every image — the classifier, the
direct round trip, the downsampled round trip — and reports energy, time,
JPEG size and PSNR for each, then the baseline-vs-adaptive totals.

## Status

- The classifier (`src/features.cpp` + `src/complexity_model.c`) is verified:
  on the 100 DIV2K validation images it prints the same features and
  probabilities as `Random-Forest/predict.py` (19 simple, 81 complex).
- The benchmark's control flow, PSNR, CSV and summary were run on a Mac
  against in-memory stand-ins for the Jetson APIs.
- **The hardware calls themselves (`NvJPEGEncoder`, `NvJPEGDecoder`,
  `NvBufferTransform`, INA3221 sysfs) have not been compiled or run on a
  Jetson yet.** They follow the Multimedia API samples `05_jpeg_encode` and
  `06_jpeg_decode`; expect to fix small things on first build.

Written for the original Jetson Nano on JetPack 4.x (L4T r32), which is where
`nvbuf_utils` lives. JetPack 6 removed `nvbuf_utils` (replaced by
`NvBufSurface`), so the buffer code in `src/main.cpp` would need porting there.

## Build (on the Jetson)

```bash
sudo apt install -y build-essential pkg-config libpng-dev nvidia-l4t-jetson-multimedia-api
cd jetson_bench
make
```

No Python is needed on the board: the forest is already exported to
`src/complexity_model.c`.

## Run

```bash
# Fixed power state, so the numbers are repeatable.
sudo nvpmodel -m 0
sudo jetson_clocks

./jpeg_energy_bench --limit 5 /path/to/DIV2K_valid_HR      # quick trial
./jpeg_energy_bench /path/to/DIV2K_valid_HR | tee summary.txt
```

Use `DIV2K_valid_HR` — the 100 images the model was not trained on. A full
run is 100 images × 3 stages × 1.25 s, about 7 minutes. Run it headless
(SSH, no desktop session) with nothing else busy on the board.

| option | default | |
|---|---|---|
| `--quality Q` | 85 | JPEG quality |
| `--min-time S` | 1.0 | seconds each stage is repeated for |
| `--limit N` | all | only the first N images |
| `--filter F` | smart | VIC filter: nearest, bilinear, 5tap, 10tap, smart, nicest |
| `--csv FILE` | results.csv | per-image results |
| `--sample-ms MS` | 10 | power sampling period |
| `--power-file FILE` | auto | a file holding power in mW, instead of the detected rail |

Inputs must be PNG: `bytes_per_mpx`, one of the model's features, is the PNG
file size per megapixel.

## How it measures

- **Energy.** One round trip takes milliseconds, less than the power
  sensor's refresh time, so each stage is repeated for `--min-time` seconds
  (after a discarded warm-up of a quarter of that) while a thread samples the
  board's input rail (`POM_5V_IN`, the whole board). Energy per image = mean
  power × time per repeat.
  - *gross* — everything the board drew.
  - *above idle* — gross minus the idle draw measured before and after the
    run; the part attributable to the work.
- **Adaptive energy** is composed per image: classifier + whichever round
  trip the classifier picked. Because both round trips are measured on every
  image, the CSV also shows what the *other* choice would have cost.
- **PSNR** is computed in YUV 4:2:0, between the frame uploaded to the
  hardware and the frame each round trip returns, so it contains only the
  loss from JPEG (and scaling), not from RGB conversion. "Loss" in the
  summary is `PSNR(direct) − PSNR(downsampled)` on the same image.
- Images are cropped to a multiple of 32 pixels each way (DIV2K 2040 → 2016
  wide) so full- and half-size frames are both whole JPEG MCUs. The
  classifier still sees the uncropped image, as `predict.py` does.
- Not measured, because both pipelines share it: PNG decoding and the
  RGB → YUV conversion.

## Reading the output

The summary's first block is the headline: total energy for baseline vs
adaptive over all images. The second block isolates the images predicted
simple and gives the PSNR loss.

Two things can eat the saving, and the per-stage numbers show which:

- the classifier runs on the CPU for *every* image, including the complex
  ones that gain nothing from it;
- the downsampled path adds two VIC passes, and NVJPG is already cheap.

## Classifier check on any machine

```bash
make classify_cli
./classify_cli --features /path/to/DIV2K_valid_HR
```

Output lines match `uv run predict.py --features --model-path
complexity_model.joblib <dir>` in `../Random-Forest` (minus its `-> N cores`
suffix).

## After retraining

```bash
cd ../Random-Forest
uv run export_c_model.py --model-path complexity_model.joblib \
    --out ../jetson_bench/src/complexity_model.c
```

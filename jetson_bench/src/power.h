// Background sampler for the Jetson's on-board INA3221 power monitor.
//
//   ina3221x  original Jetson Nano / JetPack 4 (L4T r32):
//             /sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power<N>_input
//             (mW), rail named by rail_name_<N>. Main rail: POM_5V_IN.
//   hwmon     JetPack 5+ boards (e.g. Orin Nano):
//             /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in<N>_input (mV)
//             and curr<N>_input (mA), named by in<N>_label. Main rail: VDD_IN.
#ifndef POWER_H
#define POWER_H

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

class PowerSampler {
 public:
  ~PowerSampler();

  // Finds the board's main input rail. `override_path`, if non-empty, is a
  // file holding the power in mW instead. Returns false if nothing readable.
  bool open(const std::string &override_path);
  void start(int period_ms);
  void stop();

  const std::string &rail() const { return rail_; }

  // Mean power (mW) over the window between begin() and end(). NaN when no
  // sensor is open.
  void begin();
  double end();

 private:
  double read_mw();
  void run(int period_ms);

  int fd_a_ = -1, fd_b_ = -1;  // power (mW), or voltage (mV) + current (mA)
  std::string rail_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::mutex mutex_;
  double sum_ = 0.0, last_ = 0.0;
  long count_ = 0;
};

#endif

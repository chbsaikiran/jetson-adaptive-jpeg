#include "power.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <glob.h>
#include <unistd.h>

namespace {

std::vector<std::string> glob_paths(const std::string &pattern) {
  std::vector<std::string> out;
  glob_t g;
  if (glob(pattern.c_str(), 0, NULL, &g) == 0)
    for (size_t i = 0; i < g.gl_pathc; i++) out.push_back(g.gl_pathv[i]);
  globfree(&g);
  return out;
}

std::string read_line(const std::string &path) {
  std::ifstream f(path.c_str());
  std::string line;
  std::getline(f, line);
  return line;
}

double read_number(int fd) {
  char buf[32];
  ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
  if (n <= 0) return NAN;
  buf[n] = '\0';
  return atof(buf);
}

bool is_main_rail(const std::string &name) {
  return name == "POM_5V_IN" || name == "VDD_IN";
}

}  // namespace

PowerSampler::~PowerSampler() {
  stop();
  if (fd_a_ >= 0) close(fd_a_);
  if (fd_b_ >= 0) close(fd_b_);
}

bool PowerSampler::open(const std::string &override_path) {
  if (!override_path.empty()) {
    rail_ = override_path;
    fd_a_ = ::open(override_path.c_str(), O_RDONLY);
    return fd_a_ >= 0;
  }

  std::vector<std::string> names =
      glob_paths("/sys/bus/i2c/drivers/ina3221x/*/iio:device*/rail_name_*");
  for (size_t i = 0; i < names.size(); i++) {
    std::string name = read_line(names[i]);
    if (!is_main_rail(name)) continue;
    size_t slash = names[i].rfind('/');
    std::string index = names[i].substr(names[i].rfind('_') + 1);
    std::string power =
        names[i].substr(0, slash) + "/in_power" + index + "_input";
    fd_a_ = ::open(power.c_str(), O_RDONLY);
    if (fd_a_ >= 0) {
      rail_ = name;
      return true;
    }
  }

  std::vector<std::string> labels =
      glob_paths("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label");
  for (size_t i = 0; i < labels.size(); i++) {
    std::string name = read_line(labels[i]);
    if (!is_main_rail(name)) continue;
    size_t slash = labels[i].rfind('/');
    std::string dir = labels[i].substr(0, slash);
    std::string file = labels[i].substr(slash + 1);          // in<N>_label
    std::string index = file.substr(2, file.find('_') - 2);
    fd_a_ = ::open((dir + "/in" + index + "_input").c_str(), O_RDONLY);
    fd_b_ = ::open((dir + "/curr" + index + "_input").c_str(), O_RDONLY);
    if (fd_a_ >= 0 && fd_b_ >= 0) {
      rail_ = name;
      return true;
    }
    if (fd_a_ >= 0) close(fd_a_);
    if (fd_b_ >= 0) close(fd_b_);
    fd_a_ = fd_b_ = -1;
  }
  return false;
}

double PowerSampler::read_mw() {
  double a = read_number(fd_a_);
  if (fd_b_ < 0) return a;
  return a * read_number(fd_b_) / 1000.0;  // mV * mA -> mW
}

void PowerSampler::run(int period_ms) {
  auto next = std::chrono::steady_clock::now();
  while (running_) {
    double mw = read_mw();
    if (!std::isnan(mw)) {
      std::lock_guard<std::mutex> lock(mutex_);
      sum_ += mw;
      count_++;
      last_ = mw;
    }
    next += std::chrono::milliseconds(period_ms);
    std::this_thread::sleep_until(next);
  }
}

void PowerSampler::start(int period_ms) {
  if (fd_a_ < 0 || running_) return;
  running_ = true;
  thread_ = std::thread(&PowerSampler::run, this, period_ms);
}

void PowerSampler::stop() {
  if (!running_) return;
  running_ = false;
  thread_.join();
}

void PowerSampler::begin() {
  std::lock_guard<std::mutex> lock(mutex_);
  sum_ = 0.0;
  count_ = 0;
}

double PowerSampler::end() {
  if (fd_a_ < 0) return NAN;
  std::lock_guard<std::mutex> lock(mutex_);
  return count_ ? sum_ / (double)count_ : last_;
}

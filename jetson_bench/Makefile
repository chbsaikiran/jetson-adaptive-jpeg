# Builds on the Jetson itself:
#   make                 -> jpeg_energy_bench (needs the Jetson Multimedia API)
#   make classify_cli    -> classifier only, builds on any machine with libpng

# JetPack 4.5+ installs the Multimedia API here; older releases used
# /usr/src/tegra_multimedia_api.
MMAPI ?= $(firstword $(wildcard /usr/src/jetson_multimedia_api /usr/src/tegra_multimedia_api))
CLASSES := $(MMAPI)/samples/common/classes

CC ?= gcc
CXX ?= g++
CFLAGS ?= -O2
CXXFLAGS ?= -O2 -std=c++11 -Wall
PNG_CFLAGS ?= $(shell pkg-config --cflags libpng 2>/dev/null)
PNG_LIBS ?= $(shell pkg-config --libs libpng 2>/dev/null || echo -lpng)

BUILD := build
COMMON_OBJS := $(BUILD)/features.o $(BUILD)/image_io.o $(BUILD)/complexity_model.o
# The Multimedia API ships its helper classes as source, to be compiled in.
NV_OBJS := $(addprefix $(BUILD)/nv_, NvJpegEncoder.o NvJpegDecoder.o NvBuffer.o \
             NvElement.o NvElementProfiler.o NvLogging.o)
NV_CPPFLAGS := -I$(MMAPI)/include -I$(MMAPI)/include/libjpeg-8b
NV_LIBS := -L/usr/lib/aarch64-linux-gnu/tegra -lnvjpeg -lnvbuf_utils -lv4l2

all: jpeg_energy_bench

jpeg_energy_bench: $(BUILD)/main.o $(BUILD)/power.o $(COMMON_OBJS) $(NV_OBJS)
	$(CXX) -o $@ $^ $(NV_LIBS) $(PNG_LIBS) -lpthread -lm

classify_cli: $(BUILD)/classify_cli.o $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(PNG_LIBS) -lm

$(BUILD)/main.o: src/main.cpp src/features.h src/image_io.h src/power.h | $(BUILD)
	@test -n "$(MMAPI)" || { echo "Jetson Multimedia API not found: sudo apt install nvidia-l4t-jetson-multimedia-api"; exit 1; }
	$(CXX) $(CXXFLAGS) $(NV_CPPFLAGS) $(PNG_CFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cpp $(wildcard src/*.h) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(PNG_CFLAGS) -c $< -o $@

$(BUILD)/complexity_model.o: src/complexity_model.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/nv_%.o: $(CLASSES)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -w $(NV_CPPFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) jpeg_energy_bench classify_cli

.PHONY: all clean

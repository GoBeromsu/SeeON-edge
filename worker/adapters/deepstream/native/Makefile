# Builds the three native libraries the Rust adapter links by directory:
#   make -C worker/adapters/deepstream/native OUT=<absolute dir> all
# No fallback objects: a missing SDK header or library fails the build.

OUT ?= $(CURDIR)/build
DEEPSTREAM ?= /opt/nvidia/deepstream/deepstream
CUDA ?= /usr/local/cuda
CXX ?= g++
PKG_CONFIG ?= pkg-config

CXXFLAGS := -std=c++17 -shared -fPIC -O2 -Wall -Wextra -Werror
# SDK headers are system headers: their own warnings must not fail our -Werror build.
DS_INCLUDES := -isystem $(DEEPSTREAM)/sources/includes -isystem $(CUDA)/include
GST_FLAGS = $(shell $(PKG_CONFIG) --cflags --libs gstreamer-1.0 gstreamer-app-1.0)
# Headers only: clip_decode.cpp dlopens libav RTLD_LOCAL so its dependency
# closure never enters the process global scope.
AV_CFLAGS = $(shell $(PKG_CONFIG) --cflags libavformat libavcodec libswscale libavutil)

LIBRARIES := $(OUT)/libseeon_gpu.so $(OUT)/libseeon_media.so $(OUT)/libseeon_clipdec.so

.PHONY: all clean
.DELETE_ON_ERROR:

all: $(LIBRARIES)

$(OUT):
	mkdir -p $@

$(OUT)/libseeon_gpu.so: gpu_runtime.cpp gpu_build.cpp gpu_runtime.h | $(OUT)
	$(CXX) $(CXXFLAGS) -o $@ gpu_runtime.cpp gpu_build.cpp $(DS_INCLUDES) \
		-L$(CUDA)/lib64 -lnvinfer -lnvonnxparser -lcudart -ldl -Wl,--no-undefined

$(OUT)/libseeon_media.so: media_runtime.cpp media_metadata.cpp media_recording.cpp \
		media_runtime.h media_internal.h | $(OUT)
	@$(PKG_CONFIG) --exists gstreamer-1.0 gstreamer-app-1.0
	$(CXX) $(CXXFLAGS) -o $@ media_runtime.cpp media_metadata.cpp media_recording.cpp \
		$(DS_INCLUDES) $(GST_FLAGS) -L$(DEEPSTREAM)/lib -lnvdsgst_meta -lnvds_meta -lpthread \
		-Wl,--no-undefined

$(OUT)/libseeon_clipdec.so: clip_decode.cpp clip_decode.h | $(OUT)
	@$(PKG_CONFIG) --exists libavformat libavcodec libswscale libavutil
	$(CXX) $(CXXFLAGS) -o $@ clip_decode.cpp $(AV_CFLAGS) -ldl -Wl,--no-undefined

clean:
	rm -f $(LIBRARIES)
# Opt-in CPU state tests. Not a shipping library and not a dependency of all.
# media_recording.cpp is included by its test TU; do not compile it again there.
.PHONY: test-media-recording
test-media-recording: media_recording_tests.cpp media_shutdown_tests.cpp \
		media_runtime.cpp media_metadata.cpp media_recording.cpp \
		media_runtime.h media_internal.h | $(OUT)
	@$(PKG_CONFIG) --exists gstreamer-1.0 gstreamer-app-1.0
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -o $(OUT)/media_recording_tests \
		media_recording_tests.cpp media_runtime.cpp media_metadata.cpp \
		$(DS_INCLUDES) $(GST_FLAGS) -L$(DEEPSTREAM)/lib -lnvdsgst_meta -lnvds_meta -lpthread \
		-Wl,--no-undefined
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -o $(OUT)/media_shutdown_tests \
		media_shutdown_tests.cpp media_runtime.cpp media_metadata.cpp media_recording.cpp \
		$(DS_INCLUDES) $(GST_FLAGS) -L$(DEEPSTREAM)/lib -lnvdsgst_meta -lnvds_meta -lpthread \
		-Wl,--no-undefined
	$(OUT)/media_recording_tests
	$(OUT)/media_shutdown_tests

# ONNX Runtime C API 29 headers, pinned to the shipped ORT 1.29.0 revision:
# https://raw.githubusercontent.com/microsoft/onnxruntime/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/include/onnxruntime/core/session/
# Supply that include directory explicitly; builds never download or select a runtime.
OUT ?= $(CURDIR)/build
CXX ?= g++
PYTHON ?= python3
ORT_INCLUDE_DIR ?=
CXXFLAGS := -std=c++17 -shared -fPIC -fvisibility=hidden -O2 -Wall -Wextra -Werror -pthread

.PHONY: all check-headers test-ort
.DELETE_ON_ERROR:

all: $(OUT)/libseeon_ort.so

check-headers:
	@test -n "$(ORT_INCLUDE_DIR)" || { echo 'ORT_INCLUDE_DIR must name the pinned C API include directory' >&2; exit 1; }
	@printf '%s  %s\n' \
		acc0cf4b3f28d39339c76770d76164bb7a0637dc89f5fde764b4017b632f6743 "$(ORT_INCLUDE_DIR)/onnxruntime_c_api.h" \
		5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721 "$(ORT_INCLUDE_DIR)/onnxruntime_error_code.h" \
		e6c986c9e98583f8113b2c6bc3864814883b806d501cf24da4d239c45753e235 "$(ORT_INCLUDE_DIR)/onnxruntime_ep_c_api.h" | sha256sum --check --strict

$(OUT):
	mkdir -p "$@"

$(OUT)/libseeon_ort.so: ort_runtime.cpp ort_tensor.cpp ort_runtime.h ort_internal.h Makefile | check-headers $(OUT)
	$(CXX) $(CXXFLAGS) -isystem "$(ORT_INCLUDE_DIR)" -o "$@" \
		ort_runtime.cpp ort_tensor.cpp -ldl -Wl,--no-undefined

# CPU-only real ORT execution; the selected Python needs onnx, numpy and ORT 1.29.0.
# Missing native/model dependencies fail, never skip or substitute another provider.
test-ort: all
	ORT_DISABLE_TELEMETRY=1 SEEON_TEST_ORT_ADAPTER="$(OUT)/libseeon_ort.so" $(PYTHON) -m unittest discover \
		-s "$(CURDIR)" -p test_ort_runtime.py -v

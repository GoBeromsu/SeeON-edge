// TensorRT-free stub used ONLY by host/CTest targets that exercise command and
// lifecycle logic without a GPU. The production child and the preflight binary
// always link the real trt_perception.cpp.

#include "trt_perception.hpp"

namespace seeon::trt {

class TrtPerception::Impl {};

TrtPerception::TrtPerception() : impl_(nullptr) {}
TrtPerception::~TrtPerception() = default;

std::unique_ptr<TrtPerception> TrtPerception::load(const std::string& cache_dir,
                                                   std::string* error) {
  static_cast<void>(cache_dir);
  *error = "tensorrt_unavailable_in_stub_build";
  return nullptr;
}

bool TrtPerception::infer(const std::uint8_t* rgba, int width, int height, int stride,
                          bool run_person_engine, PerceptionResult* result,
                          std::string* error) {
  static_cast<void>(rgba);
  static_cast<void>(width);
  static_cast<void>(height);
  static_cast<void>(stride);
  static_cast<void>(run_person_engine);
  static_cast<void>(result);
  *error = "tensorrt_unavailable_in_stub_build";
  return false;
}

std::vector<std::string> TrtPerception::engine_names() const { return {}; }

void preprocess_rgba_to_bgr_tensor(const std::uint8_t* rgba, int width, int height,
                                   int stride, const perception::AffineMetadata& affine,
                                   float* output_chw) {
  static_cast<void>(rgba);
  static_cast<void>(width);
  static_cast<void>(height);
  static_cast<void>(stride);
  static_cast<void>(affine);
  static_cast<void>(output_chw);
}

}  // namespace seeon::trt

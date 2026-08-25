#pragma once

// TensorRT-backed perception stage for the DeepStream child.
//
// Owns the three pinned engines (pose/person/bed) from the content-addressed
// engine cache, the C3-parity preprocessing (letterbox 640-side, stride 32,
// pad 114, BGR planes, FP32/255 -- see parity/preprocess.py), and the parse +
// association stage (native_perception.hpp). One instance per child process;
// infer() is serialized internally (one GPU, one execution context per task).

#include "native_perception.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace seeon::trt {

struct PerceptionResult {
  perception::ParsedPose pose;
  std::vector<perception::ParsedBox> person;
  std::vector<perception::ParsedBedRegion> bed;
  int source_width = 0;
  int source_height = 0;
};

struct EngineIdentity {
  std::string name;
  std::string path;
};

// Confidence thresholds mirror the shipped Python adapters: pose keeps its own
// strict >0.05 rule inside the parser; person/bed use the shipped 0.25 default
// (worker/adapters/model/yolo_pose.py conf handling is proven in C3 evidence).
inline constexpr double kPersonConfidence = 0.25;
inline constexpr double kBedConfidence = 0.25;

class TrtPerception {
 public:
  ~TrtPerception();
  TrtPerception(const TrtPerception&) = delete;
  TrtPerception& operator=(const TrtPerception&) = delete;

  // Loads pose.engine / person.engine / bed.engine from cache_dir. Returns
  // nullptr and fills *error on any failure (missing file, deserialize
  // failure, tensor-name mismatch).
  static std::unique_ptr<TrtPerception> load(const std::string& cache_dir,
                                             std::string* error);

  // rgba: HxWx4 host frame (RGBA byte order as the caps negotiate); stride in
  // bytes per row. Runs preprocess + pose + bed (+ person when
  // run_person_engine mirrors box_source=="person") + parsers. Thread-safe
  // (internal mutex).
  [[nodiscard]] bool infer(const std::uint8_t* rgba, int width, int height, int stride,
                           bool run_person_engine, PerceptionResult* result,
                           std::string* error);

  [[nodiscard]] std::vector<std::string> engine_names() const;

 private:
  TrtPerception();
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::mutex mutex_;
};

// C3-parity preprocessing on the host: float bilinear resize with the OpenCV
// coordinate convention, value-114 letterbox padding, RGBA -> BGR plane order
// ("the shipped second flip"), FP32/255, NCHW. Exposed for the warmup receipt
// and for unit probes; production callers go through TrtPerception::infer.
void preprocess_rgba_to_bgr_tensor(const std::uint8_t* rgba, int width, int height,
                                   int stride, const perception::AffineMetadata& affine,
                                   float* output_chw);

}  // namespace seeon::trt

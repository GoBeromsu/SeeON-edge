#include "trt_perception.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

namespace seeon::trt {
namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    // TensorRT INFO/VERBOSE is build noise; warnings and errors surface on
    // stderr where the parent's child-monitor captures them.
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "tensorrt: %s\n", message);
    }
  }
};

Logger& logger() {
  static Logger instance;
  return instance;
}

struct EngineSlot {
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::string input_name;
  std::vector<std::string> output_names;
};

bool read_file(const std::string& path, std::vector<char>* bytes) {
  std::ifstream input{path, std::ios::binary};
  if (!input) return false;
  bytes->assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return !bytes->empty();
}

constexpr int kMaxTensorSide = seeon::perception::kLetterboxSize;
constexpr std::size_t kInputCapacity = 3ULL * kMaxTensorSide * kMaxTensorSide;
constexpr std::size_t kPoseOutput = 300ULL * 57ULL;
constexpr std::size_t kPersonOutput = 300ULL * 6ULL;
constexpr std::size_t kBedOutput = 300ULL * 38ULL;
constexpr std::size_t kBedPrototypes = 32ULL * 160ULL * 160ULL;

}  // namespace

class TrtPerception::Impl {
 public:
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  EngineSlot pose;
  EngineSlot person;
  EngineSlot bed;
  cudaStream_t stream = nullptr;
  float* device_input = nullptr;
  float* device_pose = nullptr;
  float* device_person = nullptr;
  float* device_bed = nullptr;
  float* device_bed_prototypes = nullptr;
  std::vector<float> host_input;
  std::vector<float> host_pose;
  std::vector<float> host_person;
  std::vector<float> host_bed;
  std::vector<float> host_bed_prototypes;

  ~Impl() {
    for (float* pointer : {device_input, device_pose, device_person, device_bed,
                           device_bed_prototypes}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  bool load_engine(const std::string& path, EngineSlot* slot, std::string* error) {
    std::vector<char> bytes;
    if (!read_file(path, &bytes)) {
      *error = "engine_missing: " + path;
      return false;
    }
    slot->engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (slot->engine == nullptr) {
      *error = "engine_deserialize_failed: " + path;
      return false;
    }
    slot->context.reset(slot->engine->createExecutionContext());
    if (slot->context == nullptr) {
      *error = "engine_context_failed: " + path;
      return false;
    }
    const int tensors = slot->engine->getNbIOTensors();
    for (int index = 0; index < tensors; ++index) {
      const char* name = slot->engine->getIOTensorName(index);
      if (slot->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        slot->input_name = name;
      } else {
        slot->output_names.emplace_back(name);
      }
    }
    // Deterministic output order: ultralytics exports name outputs output0
    // (rows) and output1 (prototypes); sort so index 0 is always the row plane.
    std::sort(slot->output_names.begin(), slot->output_names.end());
    if (slot->input_name.empty() || slot->output_names.empty()) {
      *error = "engine_tensor_names_invalid: " + path;
      return false;
    }
    return true;
  }

  bool run_engine(EngineSlot* slot, int tensor_height, int tensor_width,
                  float* device_rows, std::size_t rows_capacity, float* host_rows,
                  float* device_extra, std::size_t extra_capacity, float* host_extra,
                  std::string* error) {
    nvinfer1::Dims4 shape{1, 3, tensor_height, tensor_width};
    if (!slot->context->setInputShape(slot->input_name.c_str(), shape)) {
      *error = "engine_input_shape_rejected";
      return false;
    }
    if (!slot->context->setTensorAddress(slot->input_name.c_str(), device_input)) {
      *error = "engine_input_bind_failed";
      return false;
    }
    if (!slot->context->setTensorAddress(slot->output_names[0].c_str(), device_rows)) {
      *error = "engine_output_bind_failed";
      return false;
    }
    if (slot->output_names.size() > 1) {
      if (device_extra == nullptr ||
          !slot->context->setTensorAddress(slot->output_names[1].c_str(), device_extra)) {
        *error = "engine_prototype_bind_failed";
        return false;
      }
    }
    if (!slot->context->enqueueV3(stream)) {
      *error = "engine_enqueue_failed";
      return false;
    }
    if (cudaMemcpyAsync(host_rows, device_rows, rows_capacity * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      *error = "engine_output_copy_failed";
      return false;
    }
    if (slot->output_names.size() > 1 && host_extra != nullptr) {
      if (cudaMemcpyAsync(host_extra, device_extra, extra_capacity * sizeof(float),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        *error = "engine_prototype_copy_failed";
        return false;
      }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      *error = "engine_stream_sync_failed";
      return false;
    }
    return true;
  }
};

TrtPerception::TrtPerception() : impl_(std::make_unique<Impl>()) {}
TrtPerception::~TrtPerception() = default;

std::unique_ptr<TrtPerception> TrtPerception::load(const std::string& cache_dir,
                                                   std::string* error) {
  std::unique_ptr<TrtPerception> perception{new TrtPerception()};
  Impl& impl = *perception->impl_;
  impl.runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (impl.runtime == nullptr) {
    *error = "tensorrt_runtime_unavailable";
    return nullptr;
  }
  if (!impl.load_engine(cache_dir + "/pose.engine", &impl.pose, error) ||
      !impl.load_engine(cache_dir + "/person.engine", &impl.person, error) ||
      !impl.load_engine(cache_dir + "/bed.engine", &impl.bed, error)) {
    return nullptr;
  }
  if (cudaStreamCreate(&impl.stream) != cudaSuccess) {
    *error = "cuda_stream_failed";
    return nullptr;
  }
  const std::array<std::pair<float**, std::size_t>, 5> allocations{{
      {&impl.device_input, kInputCapacity},
      {&impl.device_pose, kPoseOutput},
      {&impl.device_person, kPersonOutput},
      {&impl.device_bed, kBedOutput},
      {&impl.device_bed_prototypes, kBedPrototypes},
  }};
  for (const auto& [pointer, capacity] : allocations) {
    if (cudaMalloc(reinterpret_cast<void**>(pointer), capacity * sizeof(float)) !=
        cudaSuccess) {
      *error = "cuda_alloc_failed";
      return nullptr;
    }
  }
  impl.host_input.resize(kInputCapacity);
  impl.host_pose.resize(kPoseOutput);
  impl.host_person.resize(kPersonOutput);
  impl.host_bed.resize(kBedOutput);
  impl.host_bed_prototypes.resize(kBedPrototypes);
  return perception;
}

std::vector<std::string> TrtPerception::engine_names() const {
  return {"bed", "person", "pose"};
}

void preprocess_rgba_to_bgr_tensor(const std::uint8_t* rgba, int width, int height,
                                   int stride, const perception::AffineMetadata& affine,
                                   float* output_chw) {
  const int tensor_width = affine.tensor_width;
  const int tensor_height = affine.tensor_height;
  const int content_width = affine.content_width;
  const int content_height = affine.content_height;
  const int pad_left = affine.pad_left();
  const int pad_top = affine.pad_top();
  const std::size_t plane = static_cast<std::size_t>(tensor_width) * tensor_height;
  constexpr float kPad = static_cast<float>(seeon::perception::kLetterboxPadValue) / 255.0F;
  std::fill(output_chw, output_chw + 3 * plane, kPad);
  const double scale_x = static_cast<double>(width) / content_width;
  const double scale_y = static_cast<double>(height) / content_height;
  for (int y = 0; y < content_height; ++y) {
    const double source_y = (y + 0.5) * scale_y - 0.5;
    const double clamped_y = std::clamp(source_y, 0.0, static_cast<double>(height - 1));
    const int y0 = static_cast<int>(clamped_y);
    const int y1 = std::min(y0 + 1, height - 1);
    const double wy = clamped_y - y0;
    const std::uint8_t* row0 = rgba + static_cast<std::size_t>(y0) * stride;
    const std::uint8_t* row1 = rgba + static_cast<std::size_t>(y1) * stride;
    const std::size_t out_row =
        static_cast<std::size_t>(y + pad_top) * tensor_width + pad_left;
    for (int x = 0; x < content_width; ++x) {
      const double source_x = (x + 0.5) * scale_x - 0.5;
      const double clamped_x = std::clamp(source_x, 0.0, static_cast<double>(width - 1));
      const int x0 = static_cast<int>(clamped_x);
      const int x1 = std::min(x0 + 1, width - 1);
      const double wx = clamped_x - x0;
      const std::size_t offset = out_row + static_cast<std::size_t>(x);
      for (int channel = 0; channel < 3; ++channel) {
        // Tensor plane order is BGR ("the shipped second flip"): plane 0 reads
        // source byte 2 (B), plane 1 byte 1 (G), plane 2 byte 0 (R).
        const int source_byte = 2 - channel;
        const double top =
            (1.0 - wx) * row0[static_cast<std::size_t>(x0) * 4 + source_byte] +
            wx * row0[static_cast<std::size_t>(x1) * 4 + source_byte];
        const double bottom =
            (1.0 - wx) * row1[static_cast<std::size_t>(x0) * 4 + source_byte] +
            wx * row1[static_cast<std::size_t>(x1) * 4 + source_byte];
        const double value = (1.0 - wy) * top + wy * bottom;
        output_chw[static_cast<std::size_t>(channel) * plane + offset] =
            static_cast<float>(value / 255.0);
      }
    }
  }
}

bool TrtPerception::infer(const std::uint8_t* rgba, int width, int height, int stride,
                          bool run_person_engine, PerceptionResult* result,
                          std::string* error) {
  if (width <= 0 || height <= 0) {
    *error = "invalid_source_geometry";
    return false;
  }
  std::lock_guard lock{mutex_};
  Impl& impl = *impl_;
  const auto affine = perception::letterbox_affine(height, width);
  const std::size_t tensor_size =
      3ULL * static_cast<std::size_t>(affine.tensor_height) * affine.tensor_width;
  preprocess_rgba_to_bgr_tensor(rgba, width, height, stride, affine,
                                impl.host_input.data());
  if (cudaMemcpyAsync(impl.device_input, impl.host_input.data(),
                      tensor_size * sizeof(float), cudaMemcpyHostToDevice,
                      impl.stream) != cudaSuccess) {
    *error = "input_copy_failed";
    return false;
  }
  if (!impl.run_engine(&impl.pose, affine.tensor_height, affine.tensor_width,
                       impl.device_pose, kPoseOutput, impl.host_pose.data(), nullptr, 0,
                       nullptr, error) ||
      (run_person_engine &&
       !impl.run_engine(&impl.person, affine.tensor_height, affine.tensor_width,
                        impl.device_person, kPersonOutput, impl.host_person.data(),
                        nullptr, 0, nullptr, error)) ||
      !impl.run_engine(&impl.bed, affine.tensor_height, affine.tensor_width,
                       impl.device_bed, kBedOutput, impl.host_bed.data(),
                       impl.device_bed_prototypes, kBedPrototypes,
                       impl.host_bed_prototypes.data(), error)) {
    return false;
  }
  const std::vector<double> pose_rows{impl.host_pose.begin(), impl.host_pose.end()};
  const std::vector<double> bed_rows{impl.host_bed.begin(), impl.host_bed.end()};
  const std::vector<double> prototypes{impl.host_bed_prototypes.begin(),
                                       impl.host_bed_prototypes.end()};
  result->pose = perception::parse_pose_rows(pose_rows, affine);
  if (run_person_engine) {
    const std::vector<double> person_rows{impl.host_person.begin(), impl.host_person.end()};
    result->person = perception::parse_person_rows(person_rows, affine, kPersonConfidence);
  } else {
    result->person.clear();
  }
  result->bed = perception::parse_bed_rows(bed_rows, prototypes, affine, kBedConfidence);
  result->source_width = width;
  result->source_height = height;
  return true;
}

}  // namespace seeon::trt

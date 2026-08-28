#include "trt_perception.hpp"

#include "workspace_pool.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <vector>

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

// Shared, read-only after load. The engine holds the weights; every concurrent
// inference needs its own execution context, but they all bind against this.
struct EngineSlot {
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::string input_name;
  std::vector<std::string> output_names;
};

// Everything one concurrent inference needs that cannot be shared: an execution
// context per engine, its own CUDA stream, and its own device/host buffers.
// The engines stay shared, so an extra workspace costs activation memory and
// buffers - not another copy of the weights.
struct Workspace {
  std::unique_ptr<nvinfer1::IExecutionContext> pose_context;
  std::unique_ptr<nvinfer1::IExecutionContext> person_context;
  std::unique_ptr<nvinfer1::IExecutionContext> bed_context;
  cudaStream_t stream = nullptr;
  float* device_input = nullptr;
  float* device_pose = nullptr;
  float* device_person = nullptr;
  float* device_bed = nullptr;
  float* device_bed_prototypes = nullptr;
  std::vector<float> host_pose;
  std::vector<float> host_person;
  std::vector<float> host_bed;
  std::vector<float> host_bed_prototypes;

  ~Workspace() {
    for (float* pointer : {device_input, device_pose, device_person, device_bed,
                           device_bed_prototypes}) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
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

// How many inferences may run concurrently.
//
// A single execution context serializes every camera: GPU work measured at
// ~5.0ms per call caps a 13-camera deployment at 15.4fps against a 15fps
// target, and the stack measured 11.3fps. TensorRT execution contexts can run
// concurrently on separate streams, so the pool overlaps compute with the
// host/device copies and the per-call tensor binding.
//
// Four is a deliberate, named capacity rather than a fall-through default:
// raising it trades GPU memory (roughly 8.3MB of buffers plus one set of
// TensorRT activation memory per workspace) for concurrency, and the engines
// themselves are shared so the weights are not duplicated.
constexpr std::size_t kInferenceWorkspaces = 4;

}  // namespace

class TrtPerception::Impl {
 public:
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  EngineSlot pose;
  EngineSlot person;
  EngineSlot bed;
  std::vector<std::unique_ptr<Workspace>> workspaces;
  BoundedPool<Workspace> pool;

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

  static bool run_engine(const EngineSlot& slot, nvinfer1::IExecutionContext* context,
                         Workspace& workspace, int tensor_height, int tensor_width,
                         float* device_rows, std::size_t rows_capacity, float* host_rows,
                         float* device_extra, std::size_t extra_capacity,
                         float* host_extra, std::string* error) {
    nvinfer1::Dims4 shape{1, 3, tensor_height, tensor_width};
    if (!context->setInputShape(slot.input_name.c_str(), shape)) {
      *error = "engine_input_shape_rejected";
      return false;
    }
    if (!context->setTensorAddress(slot.input_name.c_str(), workspace.device_input)) {
      *error = "engine_input_bind_failed";
      return false;
    }
    if (!context->setTensorAddress(slot.output_names[0].c_str(), device_rows)) {
      *error = "engine_output_bind_failed";
      return false;
    }
    if (slot.output_names.size() > 1) {
      if (device_extra == nullptr ||
          !context->setTensorAddress(slot.output_names[1].c_str(), device_extra)) {
        *error = "engine_prototype_bind_failed";
        return false;
      }
    }
    if (!context->enqueueV3(workspace.stream)) {
      *error = "engine_enqueue_failed";
      return false;
    }
    // After enqueueV3 the workspace's buffers belong to the GPU until the
    // stream drains. The lease returns them to the pool on any false, so a
    // failure past this point must synchronize first or the next lessee
    // overwrites device_input under a kernel that is still reading it.
    if (cudaMemcpyAsync(host_rows, device_rows, rows_capacity * sizeof(float),
                        cudaMemcpyDeviceToHost, workspace.stream) != cudaSuccess) {
      static_cast<void>(cudaStreamSynchronize(workspace.stream));
      *error = "engine_output_copy_failed";
      return false;
    }
    if (slot.output_names.size() > 1 && host_extra != nullptr) {
      if (cudaMemcpyAsync(host_extra, device_extra, extra_capacity * sizeof(float),
                          cudaMemcpyDeviceToHost, workspace.stream) != cudaSuccess) {
        static_cast<void>(cudaStreamSynchronize(workspace.stream));
        *error = "engine_prototype_copy_failed";
        return false;
      }
    }
    if (cudaStreamSynchronize(workspace.stream) != cudaSuccess) {
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
  // Build the whole pool up front. A workspace that cannot be created is a
  // hard load failure rather than a silently smaller pool: a deployment that
  // quietly runs at a fraction of its configured concurrency is exactly the
  // kind of implicit degradation this change exists to remove.
  impl.workspaces.reserve(kInferenceWorkspaces);
  impl.pool.reserve(kInferenceWorkspaces);
  for (std::size_t index = 0; index < kInferenceWorkspaces; ++index) {
    auto workspace = std::make_unique<Workspace>();
    workspace->pose_context.reset(impl.pose.engine->createExecutionContext());
    workspace->person_context.reset(impl.person.engine->createExecutionContext());
    workspace->bed_context.reset(impl.bed.engine->createExecutionContext());
    if (workspace->pose_context == nullptr || workspace->person_context == nullptr ||
        workspace->bed_context == nullptr) {
      *error = "engine_context_failed: workspace " + std::to_string(index);
      return nullptr;
    }
    if (cudaStreamCreate(&workspace->stream) != cudaSuccess) {
      *error = "cuda_stream_failed: workspace " + std::to_string(index);
      return nullptr;
    }
    const std::array<std::pair<float**, std::size_t>, 5> allocations{{
        {&workspace->device_input, kInputCapacity},
        {&workspace->device_pose, kPoseOutput},
        {&workspace->device_person, kPersonOutput},
        {&workspace->device_bed, kBedOutput},
        {&workspace->device_bed_prototypes, kBedPrototypes},
    }};
    for (const auto& [pointer, capacity] : allocations) {
      if (cudaMalloc(reinterpret_cast<void**>(pointer), capacity * sizeof(float)) !=
          cudaSuccess) {
        *error = "cuda_alloc_failed: workspace " + std::to_string(index);
        return nullptr;
      }
    }
    workspace->host_pose.resize(kPoseOutput);
    workspace->host_person.resize(kPersonOutput);
    workspace->host_bed.resize(kBedOutput);
    workspace->host_bed_prototypes.resize(kBedPrototypes);
    impl.pool.add(workspace.get());
    impl.workspaces.push_back(std::move(workspace));
  }
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
  // Every camera serializes through mutex_, so whatever runs inside it sets the
  // per-camera frame rate for the whole deployment: with N sources the rate is
  // 1/(N * critical_section). The letterbox is pure CPU over caller-owned
  // pixels and shares nothing with the CUDA state, so it is hoisted out of the
  // lock and staged in a per-thread buffer. Measured on the 13-camera stack
  // before this change: lock_wait 91.1ms, preprocess 1.6ms, gpu 4.9ms.
  const auto affine = perception::letterbox_affine(height, width);
  const std::size_t tensor_size =
      3ULL * static_cast<std::size_t>(affine.tensor_height) * affine.tensor_width;
  thread_local std::vector<float> staging_input;
  staging_input.resize(tensor_size);
  const auto t_wait0 = std::chrono::steady_clock::now();
  preprocess_rgba_to_bgr_tensor(rgba, width, height, stride, affine,
                                staging_input.data());
  const auto t_pre = std::chrono::steady_clock::now();
  Impl& impl = *impl_;
  // RAII lease: every early return below hands the workspace back, so a failing
  // engine cannot leak pool capacity and slowly starve the deployment.
  Workspace* leased = impl.pool.acquire();
  const PoolLease<Workspace> lease{&impl.pool, leased};
  Workspace& workspace = *leased;
  const auto t_lock = std::chrono::steady_clock::now();
  if (cudaMemcpyAsync(workspace.device_input, staging_input.data(),
                      tensor_size * sizeof(float), cudaMemcpyHostToDevice,
                      workspace.stream) != cudaSuccess) {
    *error = "input_copy_failed";
    return false;
  }
  if (!Impl::run_engine(impl.pose, workspace.pose_context.get(), workspace,
                        affine.tensor_height, affine.tensor_width,
                        workspace.device_pose, kPoseOutput, workspace.host_pose.data(),
                        nullptr, 0, nullptr, error) ||
      (run_person_engine &&
       !Impl::run_engine(impl.person, workspace.person_context.get(), workspace,
                         affine.tensor_height, affine.tensor_width,
                         workspace.device_person, kPersonOutput,
                         workspace.host_person.data(), nullptr, 0, nullptr, error)) ||
      !Impl::run_engine(impl.bed, workspace.bed_context.get(), workspace,
                        affine.tensor_height, affine.tensor_width, workspace.device_bed,
                        kBedOutput, workspace.host_bed.data(),
                        workspace.device_bed_prototypes, kBedPrototypes,
                        workspace.host_bed_prototypes.data(), error)) {
    return false;
  }
  const std::vector<double> pose_rows{workspace.host_pose.begin(),
                                      workspace.host_pose.end()};
  const std::vector<double> bed_rows{workspace.host_bed.begin(),
                                     workspace.host_bed.end()};
  const std::vector<double> prototypes{workspace.host_bed_prototypes.begin(),
                                       workspace.host_bed_prototypes.end()};
  result->pose = perception::parse_pose_rows(pose_rows, affine);
  if (run_person_engine) {
    const std::vector<double> person_rows{workspace.host_person.begin(),
                                          workspace.host_person.end()};
    result->person = perception::parse_person_rows(person_rows, affine, kPersonConfidence);
  } else {
    result->person.clear();
  }
  result->bed = perception::parse_bed_rows(bed_rows, prototypes, affine, kBedConfidence);
  result->source_width = width;
  result->source_height = height;
  {
    // Diagnostic: attribute the inference cost. pool_wait is time spent waiting
    // for a free workspace and is the signal that kInferenceWorkspaces is too
    // small; gpu covers the engines plus the host-side row parsing that
    // follows them, so total_us is the whole per-call critical path. Keeping these separable is the only reason
    // the previous serialization was measured rather than guessed at.
    const auto t_gpu = std::chrono::steady_clock::now();
    static std::atomic<std::uint64_t> infer_calls{0};
    if ((++infer_calls % 64) == 0) {
      const auto us = [](auto a, auto b) {
        return static_cast<long long>(
            std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
      };
      std::fprintf(stderr,
                   "seeon-infer: pool_wait_us=%lld preprocess_us=%lld gpu_us=%lld "
                   "total_us=%lld person_engine=%d calls=%llu\n",
                   us(t_pre, t_lock), us(t_wait0, t_pre), us(t_lock, t_gpu),
                   us(t_wait0, t_gpu), run_person_engine ? 1 : 0,
                   static_cast<unsigned long long>(infer_calls.load()));
    }
  }
  return true;
}

}  // namespace seeon::trt

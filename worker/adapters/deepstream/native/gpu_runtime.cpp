#include "gpu_runtime.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <nvml.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kMaxTensorBytes = 256U * 1024U * 1024U;
constexpr size_t kMaxEngineBytes = 512U * 1024U * 1024U;
constexpr size_t kMaxOutputs = 8;
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char *) noexcept override {
        if (severity <= Severity::kWARNING) {
            // SDK messages may contain asset paths. Keep severity observable,
            // while the C ABI returns a stable, path-free failure reason.
            std::fprintf(stderr, "TensorRT diagnostic severity=%d\n", static_cast<int>(severity));
        }
    }
};
Logger logger;
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t result) {
    require(result == cudaSuccess, "CUDA operation failed; inference unavailable");
}
int error_text(char *buffer, size_t size, const char *message) noexcept {
    if (buffer && size) {
        const size_t length = std::min(size - 1, std::strlen(message));
        std::memcpy(buffer, message, length);
        buffer[length] = '\0';
    }
    return -1;
}
std::vector<char> read_engine(const char *path) {
    require(path && path[0], "engine path is required");
    const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    require(fd >= 0, "engine is not readable");
    struct CloseFd { int fd; ~CloseFd() { close(fd); } } guard{fd};
    struct stat info{};
    require(fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 &&
            static_cast<uint64_t>(info.st_size) <= kMaxEngineBytes, "engine size or type is invalid");
    std::vector<char> bytes(static_cast<size_t>(info.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = read(fd, bytes.data() + offset, bytes.size() - offset);
        require(count > 0, "engine read failed");
        offset += static_cast<size_t>(count);
    }
    return bytes;
}
size_t tensor_size(const nvinfer1::Dims &shape) {
    require(shape.nbDims > 0 && shape.nbDims <= 8, "tensor rank is unsupported");
    size_t count = 1;
    for (int i = 0; i < shape.nbDims; ++i) {
        require(shape.d[i] > 0 && static_cast<uint64_t>(shape.d[i]) <= kMaxTensorBytes / sizeof(float) / count,
                "tensor shape is unresolved or exceeds the memory bound");
        count *= static_cast<size_t>(shape.d[i]);
    }
    return count;
}
struct DeviceBuffer {
    void *data = nullptr;
    size_t size = 0;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void reserve(size_t required) {
        if (required <= size) return;
        if (data) { cuda_check(cudaFree(data)); data = nullptr; size = 0; }
        cuda_check(cudaMalloc(&data, required));
        size = required;
    }
};
} // namespace

struct SeeonGpuModel {
    int32_t device = -1;
    std::mutex mutex;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string input_name;
    std::vector<std::string> output_names;
    cudaStream_t stream = nullptr;
    DeviceBuffer input;
    std::array<DeviceBuffer, kMaxOutputs> outputs;
    SeeonGpuMetrics metrics{};
    bool poisoned = false;
    ~SeeonGpuModel() {
        if (device >= 0) cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
};

extern "C" int seeon_gpu_open(const char *path, int32_t device, SeeonGpuModel **out,
                               char *error, size_t error_size) {
    if (!out) return error_text(error, error_size, "model output is required");
    *out = nullptr;
    try {
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices));
        require(device >= 0 && device < devices, "requested GPU does not exist");
        cuda_check(cudaSetDevice(device));
        auto model = std::make_unique<SeeonGpuModel>();
        model->device = device;
        model->metrics.device = device;
        const auto bytes = read_engine(path);
        model->runtime.reset(nvinfer1::createInferRuntime(logger));
        require(bool(model->runtime), "TensorRT runtime creation failed");
        model->engine.reset(model->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(model->engine), "TensorRT engine admission failed");
        for (int i = 0; i < model->engine->getNbIOTensors(); ++i) {
            const char *name = model->engine->getIOTensorName(i);
            require(name && model->engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
                    model->engine->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE &&
                    model->engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR &&
                    !model->engine->isShapeInferenceIO(name), "engine requires unsupported tensor IO");
            if (model->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                require(model->input_name.empty(), "engine must have exactly one input");
                model->input_name = name;
            } else {
                model->output_names.emplace_back(name);
            }
        }
        require(!model->input_name.empty() && !model->output_names.empty() &&
                model->output_names.size() <= kMaxOutputs, "engine IO count is unsupported");
        model->context.reset(model->engine->createExecutionContext());
        require(bool(model->context), "TensorRT context creation failed");
        cuda_check(cudaStreamCreateWithFlags(&model->stream, cudaStreamNonBlocking));
        *out = model.release();
        if (error && error_size) error[0] = '\0';
        return 0;
    } catch (const std::runtime_error &failure) {
        return error_text(error, error_size, failure.what());
    } catch (...) {
        return error_text(error, error_size, "GPU model admission failed");
    }
}

extern "C" int seeon_gpu_run(SeeonGpuModel *model, const SeeonGpuTensor *input,
                              SeeonGpuTensor *outputs, size_t output_count,
                              char *error, size_t error_size) {
    if (!model) return error_text(error, error_size, "GPU model is required");
    std::lock_guard<std::mutex> lock(model->mutex);
    const auto start = std::chrono::steady_clock::now();
    ++model->metrics.attempted;
    try {
        require(!model->poisoned, "GPU model is unavailable after a failed execution");
        require(input && input->name && input->values && input->rank > 0 && input->rank <= 8 &&
                model->input_name == input->name, "input tensor identity is invalid");
        require(outputs && output_count == model->output_names.size(), "output tensor count is invalid");
        cuda_check(cudaSetDevice(model->device));
        nvinfer1::Dims shape{};
        shape.nbDims = input->rank;
        for (int i = 0; i < shape.nbDims; ++i) shape.d[i] = input->dimensions[i];
        const size_t input_count = tensor_size(shape);
        require(input_count == input->capacity, "input tensor size does not match its shape");
        require(model->context->setInputShape(input->name, shape), "engine rejected the input shape");
        require(model->context->inferShapes(0, nullptr) == 0, "engine output shape is unresolved");
        std::array<nvinfer1::Dims, kMaxOutputs> shapes{};
        std::array<size_t, kMaxOutputs> sizes{};
        // Validate all caller buffers before queueing any GPU work.
        for (size_t i = 0; i < output_count; ++i) {
            require(outputs[i].name && outputs[i].values && model->output_names[i] == outputs[i].name,
                    "output tensor identity is invalid");
            shapes[i] = model->context->getTensorShape(outputs[i].name);
            sizes[i] = tensor_size(shapes[i]);
            require(outputs[i].capacity >= sizes[i], "output tensor buffer is too small");
        }
        model->input.reserve(input_count * sizeof(float));
        require(model->context->setTensorAddress(input->name, model->input.data), "input tensor binding failed");
        for (size_t i = 0; i < output_count; ++i) {
            model->outputs[i].reserve(sizes[i] * sizeof(float));
            require(model->context->setTensorAddress(outputs[i].name, model->outputs[i].data), "output tensor binding failed");
        }
        cuda_check(cudaMemcpyAsync(model->input.data, input->values, input_count * sizeof(float),
                                   cudaMemcpyHostToDevice, model->stream));
        model->metrics.host_to_device_bytes += input_count * sizeof(float);
        require(model->context->enqueueV3(model->stream), "TensorRT GPU enqueue failed");
        for (size_t i = 0; i < output_count; ++i) {
            cuda_check(cudaMemcpyAsync(outputs[i].values, model->outputs[i].data, sizes[i] * sizeof(float),
                                       cudaMemcpyDeviceToHost, model->stream));
        }
        cuda_check(cudaStreamSynchronize(model->stream));
        for (size_t i = 0; i < output_count; ++i) {
            outputs[i].rank = shapes[i].nbDims;
            for (int j = 0; j < shapes[i].nbDims; ++j) outputs[i].dimensions[j] = static_cast<int32_t>(shapes[i].d[j]);
            model->metrics.device_to_host_bytes += sizes[i] * sizeof(float);
        }
        ++model->metrics.succeeded;
        model->metrics.elapsed_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
        if (error && error_size) error[0] = '\0';
        return 0;
    } catch (const std::runtime_error &failure) {
        model->poisoned = true;
        ++model->metrics.failed;
        cudaStreamSynchronize(model->stream);
        return error_text(error, error_size, failure.what());
    } catch (...) {
        model->poisoned = true;
        ++model->metrics.failed;
        cudaStreamSynchronize(model->stream);
        return error_text(error, error_size, "GPU inference failed");
    }
}

extern "C" int seeon_gpu_metrics(SeeonGpuModel *model, SeeonGpuMetrics *metrics) {
    if (!model || !metrics) return -1;
    std::lock_guard<std::mutex> lock(model->mutex);
    *metrics = model->metrics;
    return 0;
}
extern "C" void seeon_gpu_close(SeeonGpuModel *model) { delete model; }

namespace {
// NVML is resolved at call time so hosts without the driver library still boot.
class Nvml {
  public:
    Nvml() : library_(dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL)) {}
    ~Nvml() {
        if (initialized_) shutdown_();
        if (library_) dlclose(library_);
    }
    Nvml(const Nvml &) = delete;
    Nvml &operator=(const Nvml &) = delete;
    int32_t start() {
        if (!library_) return SEEON_NVML_LIBRARY_MISSING;
        if (!bind(init_, "nvmlInit_v2") || !bind(shutdown_, "nvmlShutdown") ||
            !bind(driver_, "nvmlSystemGetDriverVersion") || !bind(count_, "nvmlDeviceGetCount_v2") ||
            !bind(handle_, "nvmlDeviceGetHandleByIndex_v2") || !bind(name_, "nvmlDeviceGetName")) {
            return SEEON_NVML_SYMBOL_MISSING;
        }
        initialized_ = init_() == NVML_SUCCESS;
        return initialized_ ? SEEON_NVML_OK : SEEON_NVML_INIT_FAILED;
    }
    bool driver_version(char *buffer, unsigned int size) { return driver_(buffer, size) == NVML_SUCCESS; }
    bool device_count(unsigned int *count) { return count_(count) == NVML_SUCCESS; }
    bool first_device_name(char *buffer, unsigned int size) {
        nvmlDevice_t device{};
        return handle_(0, &device) == NVML_SUCCESS && name_(device, buffer, size) == NVML_SUCCESS;
    }

  private:
    template <typename Function> bool bind(Function &target, const char *symbol) {
        target = reinterpret_cast<Function>(dlsym(library_, symbol));
        return target != nullptr;
    }
    void *library_;
    bool initialized_ = false;
    nvmlReturn_t (*init_)() = nullptr;
    nvmlReturn_t (*shutdown_)() = nullptr;
    nvmlReturn_t (*driver_)(char *, unsigned int) = nullptr;
    nvmlReturn_t (*count_)(unsigned int *) = nullptr;
    nvmlReturn_t (*handle_)(unsigned int, nvmlDevice_t *) = nullptr;
    nvmlReturn_t (*name_)(nvmlDevice_t, char *, unsigned int) = nullptr;
};
} // namespace

extern "C" int seeon_gpu_device_report(int32_t device, SeeonGpuDeviceReport *report) {
    if (!report) return -1;
    std::memset(report, 0, sizeof(*report));
    try {
        Nvml nvml;
        report->nvml_status = nvml.start();
        if (report->nvml_status == SEEON_NVML_OK) {
            report->has_driver_version =
                nvml.driver_version(report->driver_version, sizeof(report->driver_version)) ? 1 : 0;
            unsigned int count = 0;
            if (!nvml.device_count(&count)) {
                report->nvml_status = SEEON_NVML_DEVICE_COUNT_FAILED;
            } else if (count == 0) {
                report->nvml_status = SEEON_NVML_NO_DEVICE;
            } else {
                report->has_device_name =
                    nvml.first_device_name(report->device_name, sizeof(report->device_name)) ? 1 : 0;
            }
        }
    } catch (...) {
        report->nvml_status = SEEON_NVML_INIT_FAILED;
    }
    report->driver_version[sizeof(report->driver_version) - 1] = '\0';
    report->device_name[sizeof(report->device_name) - 1] = '\0';
    if (!report->has_driver_version) report->driver_version[0] = '\0';
    if (!report->has_device_name) report->device_name[0] = '\0';
    int devices = 0;
    report->cuda_context_ok = cudaGetDeviceCount(&devices) == cudaSuccess && device >= 0 && device < devices &&
                              cudaSetDevice(device) == cudaSuccess && cudaFree(nullptr) == cudaSuccess ? 1 : 0;
    return 0;
}

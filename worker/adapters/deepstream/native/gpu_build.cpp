#include "gpu_runtime.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t kMaxOnnxBytes = 512U * 1024U * 1024U;
constexpr size_t kMaxEngineBytes = 512U * 1024U * 1024U;
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char *) noexcept override {
        if (severity <= Severity::kWARNING) {
            // Parser and builder messages may contain asset paths.
            std::fprintf(stderr, "TensorRT build diagnostic severity=%d\n", static_cast<int>(severity));
        }
    }
};
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
int error_text(char *buffer, size_t size, const char *message) noexcept {
    if (buffer && size) {
        const size_t length = std::min(size - 1, std::strlen(message));
        std::memcpy(buffer, message, length);
        buffer[length] = '\0';
    }
    return -1;
}
struct CloseFd {
    int fd;
    ~CloseFd() { if (fd >= 0) close(fd); }
};
std::vector<char> read_onnx(const char *path) {
    require(path && path[0], "ONNX path is required");
    CloseFd guard{open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
    require(guard.fd >= 0, "ONNX model is not readable");
    struct stat info{};
    require(fstat(guard.fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 &&
            static_cast<uint64_t>(info.st_size) <= kMaxOnnxBytes, "ONNX model size or type is invalid");
    std::vector<char> bytes(static_cast<size_t>(info.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = read(guard.fd, bytes.data() + offset, bytes.size() - offset);
        require(count > 0, "ONNX model read failed");
        offset += static_cast<size_t>(count);
    }
    return bytes;
}
// The engine file is created exclusively and removed again unless fully synced.
void write_engine(const char *path, const void *data, size_t size) {
    require(path && path[0], "engine path is required");
    CloseFd guard{open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644)};
    require(guard.fd >= 0, "engine file could not be created; it must not already exist");
    const auto *bytes = static_cast<const char *>(data);
    size_t offset = 0;
    bool written = true;
    while (written && offset < size) {
        const ssize_t count = write(guard.fd, bytes + offset, size - offset);
        written = count > 0;
        if (written) offset += static_cast<size_t>(count);
    }
    written = written && fsync(guard.fd) == 0;
    const int fd = guard.fd;
    guard.fd = -1;
    written = close(fd) == 0 && written;
    if (!written) {
        unlink(path);
        throw std::runtime_error("engine write failed");
    }
}
} // namespace

extern "C" int seeon_gpu_build(const char *onnx_path, const char *engine_path, int32_t device,
                                const char *input_name, const int32_t *dimensions, int32_t rank,
                                SeeonGpuBuildIdentity *identity, char *error, size_t error_size) {
    if (!identity) return error_text(error, error_size, "build identity output is required");
    std::memset(identity, 0, sizeof(*identity));
    try {
        require(input_name && input_name[0], "input tensor name is required");
        require(dimensions && rank > 0 && rank <= 8, "input rank is unsupported");
        nvinfer1::Dims requested{};
        requested.nbDims = rank;
        for (int i = 0; i < rank; ++i) {
            require(dimensions[i] > 0, "input dimensions must be positive");
            requested.d[i] = dimensions[i];
        }
        int devices = 0;
        require(cudaGetDeviceCount(&devices) == cudaSuccess && device >= 0 && device < devices,
                "requested GPU does not exist");
        require(cudaSetDevice(device) == cudaSuccess, "CUDA operation failed; build unavailable");
        cudaDeviceProp properties{};
        require(cudaGetDeviceProperties(&properties, device) == cudaSuccess, "GPU properties are unavailable");
        const auto onnx = read_onnx(onnx_path);

        Logger logger;
        std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        require(bool(builder), "TensorRT builder creation failed");
        // Strong typing keeps every layer at the ONNX type; the builder cannot pick FP16 or INT8.
        const auto typed = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(typed)};
        require(bool(network) && network->getFlag(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED),
                "TensorRT network creation failed");
        std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
        require(bool(parser), "ONNX parser creation failed");
        require(parser->parse(onnx.data(), onnx.size()), "ONNX model parse failed");

        require(network->getNbInputs() == 1, "network must have exactly one input");
        nvinfer1::ITensor *input = network->getInput(0);
        require(input && std::strcmp(input->getName(), input_name) == 0, "network input name is unexpected");
        require(input->getType() == nvinfer1::DataType::kFLOAT, "network input must be FP32");
        require(network->getNbOutputs() > 0, "network has no outputs");
        for (int i = 0; i < network->getNbOutputs(); ++i) {
            require(network->getOutput(i)->getType() == nvinfer1::DataType::kFLOAT, "network outputs must be FP32");
        }
        const nvinfer1::Dims declared = input->getDimensions();
        require(declared.nbDims == requested.nbDims, "input rank does not match the model");
        bool dynamic = false;
        for (int i = 0; i < rank; ++i) {
            require(declared.d[i] == -1 || declared.d[i] == requested.d[i], "input shape does not match the model");
            dynamic = dynamic || declared.d[i] == -1;
        }

        std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        require(bool(config), "TensorRT builder config creation failed");
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        require(!config->getFlag(nvinfer1::BuilderFlag::kTF32), "builder precision must stay FP32");
        if (dynamic) {
            // A dynamic ONNX input is pinned to the one admitted shape.
            nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
            require(profile != nullptr, "optimization profile creation failed");
            for (const auto selector : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT,
                                        nvinfer1::OptProfileSelector::kMAX}) {
                require(profile->setDimensions(input_name, selector, requested), "optimization profile was refused");
            }
            require(config->addOptimizationProfile(profile) == 0, "optimization profile was refused");
        }
        std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        require(plan && plan->size() > 0 && plan->size() <= kMaxEngineBytes, "TensorRT engine build failed");
        write_engine(engine_path, plan->data(), plan->size());

        identity->trt_version = getInferLibVersion();
        identity->compute_major = properties.major;
        identity->compute_minor = properties.minor;
        identity->tf32_enabled = config->getFlag(nvinfer1::BuilderFlag::kTF32) ? 1 : 0;
        const size_t name_length = strnlen(properties.name, sizeof(properties.name));
        std::memcpy(identity->device_name, properties.name,
                    std::min(name_length, sizeof(identity->device_name) - 1));
        if (error && error_size) error[0] = '\0';
        return 0;
    } catch (const std::runtime_error &failure) {
        std::memset(identity, 0, sizeof(*identity));
        return error_text(error, error_size, failure.what());
    } catch (...) {
        std::memset(identity, 0, sizeof(*identity));
        return error_text(error, error_size, "TensorRT engine build failed");
    }
}

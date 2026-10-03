#include "ort_internal.h"

#include <cstdlib>
#include <dlfcn.h>

using namespace seeon_ort;

namespace {
struct AllocatedName {
    OrtAllocator *allocator;
    char *value = nullptr;
    ~AllocatedName() noexcept { if (value) allocator->Free(allocator, value); }
};

void discover(SeeonOrtModel &model, bool input, size_t index, TensorSpec &spec) {
    const OrtApi &api = *model.api;
    Object<OrtTypeInfo> type(api.ReleaseTypeInfo);
    check(api, input ? api.SessionGetInputTypeInfo(model.session, index, &type.value) :
                       api.SessionGetOutputTypeInfo(model.session, index, &type.value),
          SEEON_ORT_MODEL, "model tensor metadata is unavailable");
    require(type.value != nullptr, SEEON_ORT_MODEL, "model tensor metadata is missing");
    ONNXType kind = ONNX_TYPE_UNKNOWN;
    check(api, api.GetOnnxTypeFromTypeInfo(type.value, &kind), SEEON_ORT_MODEL,
          "model tensor kind is unavailable");
    require(kind == ONNX_TYPE_TENSOR, SEEON_ORT_MODEL, "model IO must be tensors");
    const OrtTensorTypeAndShapeInfo *shape = nullptr; // Borrowed from type.
    check(api, api.CastTypeInfoToTensorInfo(type.value, &shape), SEEON_ORT_MODEL,
          "model tensor shape is unavailable");
    require(shape != nullptr, SEEON_ORT_MODEL, "model IO must be tensors");
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    check(api, api.GetTensorElementType(shape, &dtype), SEEON_ORT_MODEL,
          "model tensor element type is unavailable");
    require(dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, SEEON_ORT_MODEL,
            "model IO must be float32 tensors");
    check(api, api.GetDimensionsCount(shape, &spec.rank), SEEON_ORT_MODEL,
          "model tensor rank is unavailable");
    require(spec.rank > 0 && spec.rank <= spec.shape.size(), SEEON_ORT_MODEL,
            "model tensor rank must be between 1 and 8");
    check(api, api.GetDimensions(shape, spec.shape.data(), spec.rank), SEEON_ORT_MODEL,
          "model tensor dimensions are unavailable");
    elements(spec.shape.data(), spec.rank, SEEON_ORT_MODEL, true);
    AllocatedName name{model.allocator};
    check(api, input ? api.SessionGetInputName(model.session, index, model.allocator, &name.value) :
                       api.SessionGetOutputName(model.session, index, model.allocator, &name.value),
          SEEON_ORT_MODEL, "model tensor name is unavailable");
    require(name.value && name.value[0], SEEON_ORT_MODEL, "model tensor name is empty");
    spec.name = name.value;
}

void load_runtime(SeeonOrtModel &model, const char *library) {
    model.library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    require(model.library != nullptr, SEEON_ORT_UNAVAILABLE,
            "ONNX Runtime shared library could not be loaded");
    const auto get_base = reinterpret_cast<decltype(&OrtGetApiBase)>(
        dlsym(model.library, "OrtGetApiBase"));
    require(get_base != nullptr, SEEON_ORT_UNAVAILABLE,
            "ONNX Runtime entry point is unavailable");
    const OrtApiBase *base = get_base();
    require(base && base->GetApi && base->GetVersionString, SEEON_ORT_UNAVAILABLE,
            "ONNX Runtime API base is unavailable");
    model.api = base->GetApi(ORT_API_VERSION);
    require(model.api != nullptr, SEEON_ORT_UNAVAILABLE,
            "ONNX Runtime does not support API 29");
    const char *version = base->GetVersionString();
    require(version && version[0], SEEON_ORT_UNAVAILABLE,
            "ONNX Runtime version is unavailable");
    size_t length = 0;
    while (length < sizeof(model.info.runtime_version) && version[length]) ++length;
    require(length < sizeof(model.info.runtime_version), SEEON_ORT_UNAVAILABLE,
            "ONNX Runtime version exceeds ABI capacity");
    std::memcpy(model.info.runtime_version, version, length + 1);
}
} // namespace

SeeonOrtModel::~SeeonOrtModel() noexcept {
    if (memory) api->ReleaseMemoryInfo(memory);
    if (session) api->ReleaseSession(session);
    if (env) api->ReleaseEnv(env);
    if (library) dlclose(library);
}

extern "C" SEEON_ORT_EXPORT int seeon_ort_open(const char *runtime_library,
    const void *onnx, size_t onnx_size, uint32_t threads, SeeonOrtModel **out,
    char *error, size_t error_size) {
    try {
        if (out) *out = nullptr;
        require(out && (error || !error_size), SEEON_ORT_INVALID_ARGUMENT,
                "model output and valid diagnostic storage are required");
        require(runtime_library && runtime_library[0], SEEON_ORT_INVALID_ARGUMENT,
                "runtime library path is required");
        require(onnx && onnx_size > 0 && onnx_size <= kMaxOnnxBytes,
                SEEON_ORT_INVALID_ARGUMENT, "ONNX bytes must be nonempty and at most 512 MiB");
        require(threads <= 1, SEEON_ORT_INVALID_ARGUMENT, "threads must be 0 or 1");
        const char *telemetry = std::getenv("ORT_DISABLE_TELEMETRY");
        require(telemetry && std::strcmp(telemetry, "1") == 0, SEEON_ORT_UNAVAILABLE,
                "ORT_DISABLE_TELEMETRY=1 must be set before starting the process");
        auto model = std::make_unique<SeeonOrtModel>();
        load_runtime(*model, runtime_library);
        const OrtApi &api = *model->api;
        check(api, api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "seeon-ort-cpu", &model->env),
              SEEON_ORT_UNAVAILABLE, "ONNX Runtime environment creation failed");
        require(model->env != nullptr, SEEON_ORT_UNAVAILABLE,
                "ONNX Runtime environment is missing");
        // API suppression alone leaves POSIX ProcessInfo active; the process
        // opt-out above prevents its logger and identity store initialization.
        check(api, api.DisableTelemetryEvents(model->env), SEEON_ORT_UNAVAILABLE,
              "ONNX Runtime telemetry could not be disabled");
        Object<OrtSessionOptions> options(api.ReleaseSessionOptions);
        check(api, api.CreateSessionOptions(&options.value), SEEON_ORT_UNAVAILABLE,
              "ONNX Runtime session options creation failed");
        require(options.value != nullptr, SEEON_ORT_UNAVAILABLE,
                "ONNX Runtime session options are missing");
        if (threads == 1) {
            check(api, api.SetIntraOpNumThreads(options.value, 1), SEEON_ORT_UNAVAILABLE,
                  "ONNX Runtime intra-op thread configuration failed");
            check(api, api.SetInterOpNumThreads(options.value, 1), SEEON_ORT_UNAVAILABLE,
                  "ONNX Runtime inter-op thread configuration failed");
        }
        // No EP is appended: fresh session options intentionally select ORT CPU.
        check(api, api.CreateSessionFromArray(model->env, onnx, onnx_size,
              options.value, &model->session), SEEON_ORT_MODEL,
              "ONNX Runtime model session creation failed");
        require(model->session != nullptr, SEEON_ORT_MODEL, "model session is missing");
        size_t input_count = 0, output_count = 0;
        check(api, api.SessionGetInputCount(model->session, &input_count), SEEON_ORT_MODEL,
              "model input count is unavailable");
        check(api, api.SessionGetOutputCount(model->session, &output_count), SEEON_ORT_MODEL,
              "model output count is unavailable");
        require(input_count == 1 && output_count > 0 && output_count <= kMaxOutputs,
                SEEON_ORT_MODEL, "model requires one input and one or two outputs");
        check(api, api.GetAllocatorWithDefaultOptions(&model->allocator), SEEON_ORT_UNAVAILABLE,
              "ONNX Runtime default allocator is unavailable");
        require(model->allocator && model->allocator->Free, SEEON_ORT_UNAVAILABLE,
                "ONNX Runtime default allocator is missing");
        discover(*model, true, 0, model->input);
        for (size_t i = 0; i < output_count; ++i) {
            discover(*model, false, i, model->outputs[i]);
            for (size_t j = 0; j < i; ++j)
                require(model->outputs[i].name != model->outputs[j].name, SEEON_ORT_MODEL,
                        "model output names must be unique");
        }
        check(api, api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &model->memory),
              SEEON_ORT_UNAVAILABLE, "ONNX Runtime CPU memory description creation failed");
        require(model->memory != nullptr, SEEON_ORT_UNAVAILABLE,
                "ONNX Runtime CPU memory description is missing");
        model->info.abi_version = SEEON_ORT_ABI_VERSION;
        model->info.input_count = static_cast<uint32_t>(input_count);
        model->info.output_count = static_cast<uint32_t>(output_count);
        model->info.threads = threads;
        *out = model.release();
        return diagnostic(SEEON_ORT_OK, error, error_size, "");
    } catch (const Failure &failure) {
        return diagnostic(failure.code, error, error_size, failure.text);
    } catch (const std::bad_alloc &) {
        return diagnostic(SEEON_ORT_UNAVAILABLE, error, error_size,
                          "native CPU adapter allocation failed");
    } catch (...) {
        return diagnostic(SEEON_ORT_UNAVAILABLE, error, error_size,
                          "unexpected native CPU adapter exception");
    }
}

extern "C" SEEON_ORT_EXPORT int seeon_ort_info(const SeeonOrtModel *model,
    SeeonOrtInfo *out, char *error, size_t error_size) {
    try {
        require(model && out && (error || !error_size), SEEON_ORT_INVALID_ARGUMENT,
                "model, info output and valid diagnostic storage are required");
        *out = model->info;
        return diagnostic(SEEON_ORT_OK, error, error_size, "");
    } catch (const Failure &failure) {
        return diagnostic(failure.code, error, error_size, failure.text);
    } catch (...) {
        return diagnostic(SEEON_ORT_UNAVAILABLE, error, error_size,
                          "unexpected native CPU adapter exception");
    }
}

extern "C" SEEON_ORT_EXPORT void seeon_ort_close(SeeonOrtModel *model) {
    try { delete model; } catch (...) { /* A C boundary cannot propagate exceptions. */ }
}

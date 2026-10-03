#include "ort_internal.h"

#include <cmath>
#include <limits>

using namespace seeon_ort;

namespace {
struct Range { uintptr_t begin; uintptr_t end; };
Range range(const void *pointer, size_t count, size_t width) {
    require(pointer && count <= std::numeric_limits<size_t>::max() / width,
            SEEON_ORT_INVALID_ARGUMENT, "tensor storage byte range is invalid");
    const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
    const size_t bytes = count * width;
    require(bytes <= std::numeric_limits<uintptr_t>::max() - begin,
            SEEON_ORT_INVALID_ARGUMENT, "tensor storage address range overflows");
    return {begin, begin + bytes};
}
bool overlaps(Range a, Range b) noexcept {
    return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
}
bool aligned(const void *pointer, size_t alignment) noexcept {
    return pointer && reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

std::array<size_t, kMaxOutputs> validate(const SeeonOrtModel &model,
    const SeeonOrtTensor &input, SeeonOrtTensor *outputs, size_t output_count) {
    const Range input_descriptor = range(&input, 1, sizeof(input));
    const Range output_descriptors = range(outputs, output_count, sizeof(*outputs));
    require(input.name && input.data && aligned(input.data, alignof(float)),
            SEEON_ORT_INVALID_ARGUMENT, "input name and aligned float storage are required");
    require(model.input.name == input.name, SEEON_ORT_INVALID_ARGUMENT,
            "input name does not match the model");
    const size_t count = elements(input.shape, input.rank, SEEON_ORT_INVALID_ARGUMENT);
    require(input.elements == count && input.rank == model.input.rank,
            SEEON_ORT_INVALID_ARGUMENT, "input size or rank does not match the model");
    const Range input_storage = range(input.data, count, sizeof(float));
    for (size_t i = 0; i < input.rank; ++i)
        require(model.input.shape[i] == -1 || model.input.shape[i] == input.shape[i],
                SEEON_ORT_INVALID_ARGUMENT, "input dimensions do not match the model");
    for (size_t i = 0; i < count; ++i)
        require(std::isfinite(input.data[i]), SEEON_ORT_INVALID_ARGUMENT,
                "input contains a non-finite value");
    std::array<Range, kMaxOutputs> destinations{};
    std::array<size_t, kMaxOutputs> order{};
    for (size_t i = 0; i < output_count; ++i) {
        require(outputs[i].name && aligned(outputs[i].data, alignof(float)),
                SEEON_ORT_INVALID_ARGUMENT, "output name and aligned float storage are required");
        size_t match = 0;
        while (match < output_count && model.outputs[match].name != outputs[i].name) ++match;
        require(match < output_count, SEEON_ORT_INVALID_ARGUMENT,
                "output name does not match the model");
        order[i] = match;
        destinations[i] = range(outputs[i].data, outputs[i].elements, sizeof(float));
        require(!overlaps(destinations[i], input_descriptor) &&
                !overlaps(destinations[i], output_descriptors), SEEON_ORT_INVALID_ARGUMENT,
                "output storage overlaps a tensor descriptor");
        require(!overlaps(destinations[i], input_storage), SEEON_ORT_INVALID_ARGUMENT,
                "input and output storage regions overlap");
        for (size_t j = 0; j < i; ++j) {
            require(order[j] != match, SEEON_ORT_INVALID_ARGUMENT,
                    "each model output must be named exactly once");
            require(!overlaps(destinations[i], destinations[j]), SEEON_ORT_INVALID_ARGUMENT,
                    "output storage regions overlap");
        }
    }
    return order;
}

struct OutputView {
    const float *data = nullptr;
    size_t count = 0;
    size_t rank = 0;
    std::array<int64_t, 8> shape{};
};
OutputView inspect(const OrtApi &api, OrtValue *value, const TensorSpec &spec,
                   size_t capacity) {
    require(value != nullptr, SEEON_ORT_OUTPUT, "ONNX Runtime output value is missing");
    int tensor = 0;
    check(api, api.IsTensor(value, &tensor), SEEON_ORT_OUTPUT,
          "ONNX Runtime output tensor kind is unavailable");
    require(tensor == 1, SEEON_ORT_OUTPUT, "ONNX Runtime output is not a tensor");
    Object<OrtTensorTypeAndShapeInfo> metadata(api.ReleaseTensorTypeAndShapeInfo);
    check(api, api.GetTensorTypeAndShape(value, &metadata.value), SEEON_ORT_OUTPUT,
          "ONNX Runtime output shape is unavailable");
    require(metadata.value != nullptr, SEEON_ORT_OUTPUT, "ONNX Runtime output shape is missing");
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    check(api, api.GetTensorElementType(metadata.value, &dtype), SEEON_ORT_OUTPUT,
          "ONNX Runtime output element type is unavailable");
    require(dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, SEEON_ORT_OUTPUT,
            "ONNX Runtime output is not float32");
    OutputView view;
    check(api, api.GetDimensionsCount(metadata.value, &view.rank), SEEON_ORT_OUTPUT,
          "ONNX Runtime output rank is unavailable");
    require(view.rank > 0 && view.rank <= view.shape.size() && view.rank == spec.rank,
            SEEON_ORT_OUTPUT, "ONNX Runtime output rank does not match the model contract");
    check(api, api.GetDimensions(metadata.value, view.shape.data(), view.rank), SEEON_ORT_OUTPUT,
          "ONNX Runtime output dimensions are unavailable");
    view.count = elements(view.shape.data(), view.rank, SEEON_ORT_OUTPUT);
    for (size_t i = 0; i < view.rank; ++i)
        require(spec.shape[i] == -1 || spec.shape[i] == view.shape[i], SEEON_ORT_OUTPUT,
                "ONNX Runtime output dimensions do not match the model contract");
    size_t reported = 0;
    check(api, api.GetTensorShapeElementCount(metadata.value, &reported), SEEON_ORT_OUTPUT,
          "ONNX Runtime output size is unavailable");
    require(reported == view.count && view.count <= capacity, SEEON_ORT_OUTPUT,
            "ONNX Runtime output size is inconsistent or exceeds caller capacity");
    void *data = nullptr;
    check(api, api.GetTensorMutableData(value, &data), SEEON_ORT_OUTPUT,
          "ONNX Runtime output storage is unavailable");
    require(aligned(data, alignof(float)), SEEON_ORT_OUTPUT,
            "ONNX Runtime output float storage is missing or unaligned");
    view.data = static_cast<const float *>(data);
    for (size_t i = 0; i < view.count; ++i)
        require(std::isfinite(view.data[i]), SEEON_ORT_OUTPUT,
                "ONNX Runtime output contains a non-finite value");
    return view;
}

struct Values {
    const OrtApi &api;
    std::array<OrtValue *, kMaxOutputs> data{};
    ~Values() noexcept { for (auto *value : data) if (value) api.ReleaseValue(value); }
};
struct Execution {
    SeeonOrtModel &model;
    bool succeeded = false;
    // Declared after the lock: poisoning is published before another run enters.
    ~Execution() noexcept { if (!succeeded) model.poisoned = true; }
};
} // namespace

extern "C" SEEON_ORT_EXPORT int seeon_ort_run(SeeonOrtModel *model,
    const SeeonOrtTensor *input, SeeonOrtTensor *outputs, size_t output_count,
    char *error, size_t error_size) {
    SeeonOrtResult phase = SEEON_ORT_UNAVAILABLE;
    try {
        require(model && (error || !error_size), SEEON_ORT_INVALID_ARGUMENT,
                "model and valid diagnostic storage are required");
        std::lock_guard<std::mutex> lock(model->mutex);
        require(!model->poisoned, SEEON_ORT_POISONED, "model is permanently poisoned");
        require(aligned(input, alignof(SeeonOrtTensor)) &&
                aligned(outputs, alignof(SeeonOrtTensor)) && output_count == model->info.output_count,
                SEEON_ORT_INVALID_ARGUMENT, "tensor pointers or output count are invalid");
        const auto order = validate(*model, *input, outputs, output_count);
        phase = SEEON_ORT_EXECUTION;
        Execution execution{*model};
        const OrtApi &api = *model->api;
        Object<OrtValue> input_value(api.ReleaseValue);
        Values results{api};
        check(api, api.CreateTensorWithDataAsOrtValue(model->memory, input->data,
              input->elements * sizeof(float), input->shape, input->rank,
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_value.value), SEEON_ORT_EXECUTION,
              "ONNX Runtime input tensor creation failed");
        require(input_value.value != nullptr, SEEON_ORT_EXECUTION,
                "ONNX Runtime input tensor is missing");
        const char *input_names[]{model->input.name.c_str()};
        const OrtValue *inputs[]{input_value.value};
        std::array<const char *, kMaxOutputs> output_names{};
        for (size_t i = 0; i < output_count; ++i)
            output_names[i] = model->outputs[order[i]].name.c_str();
        check(api, api.Run(model->session, nullptr, input_names, inputs, 1,
              output_names.data(), output_count, results.data.data()), SEEON_ORT_EXECUTION,
              "ONNX Runtime CPU execution failed");
        phase = SEEON_ORT_OUTPUT;
        std::array<OutputView, kMaxOutputs> views{};
        for (size_t i = 0; i < output_count; ++i)
            views[i] = inspect(api, results.data[i], model->outputs[order[i]], outputs[i].elements);
        // Nothing that can reject a result remains after this commit point.
        for (size_t i = 0; i < output_count; ++i) {
            std::memcpy(outputs[i].data, views[i].data, views[i].count * sizeof(float));
            outputs[i].elements = views[i].count;
            std::memcpy(outputs[i].shape, views[i].shape.data(), sizeof(outputs[i].shape));
            outputs[i].rank = views[i].rank;
        }
        execution.succeeded = true;
        return diagnostic(SEEON_ORT_OK, error, error_size, "");
    } catch (const Failure &failure) {
        return diagnostic(failure.code, error, error_size, failure.text);
    } catch (const std::bad_alloc &) {
        return diagnostic(phase, error, error_size, "native CPU adapter allocation failed");
    } catch (...) {
        return diagnostic(phase, error, error_size, "unexpected native CPU adapter exception");
    }
}

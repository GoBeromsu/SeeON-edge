#ifndef SEEON_ORT_INTERNAL_H
#define SEEON_ORT_INTERNAL_H

#include "ort_runtime.h"
#include <onnxruntime_c_api.h>

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>

static_assert(ORT_API_VERSION == 29, "ONNX Runtime API 29 headers are required");
static_assert(sizeof(float) == 4, "float32 storage is required");

namespace seeon_ort {
constexpr size_t kMaxOnnxBytes = 512U * 1024U * 1024U;
constexpr size_t kMaxElements = 16U * 1024U * 1024U;
constexpr size_t kMaxOutputs = 2;

// Static diagnostics make failure handling independent of heap availability.
struct Failure { SeeonOrtResult code; const char *text; };
inline void require(bool condition, SeeonOrtResult code, const char *text) {
    if (!condition) throw Failure{code, text};
}
inline void check(const OrtApi &api, OrtStatus *status, SeeonOrtResult code,
                  const char *text) {
    if (!status) return;
    api.ReleaseStatus(status);
    throw Failure{code, text};
}
inline int diagnostic(SeeonOrtResult code, char *buffer, size_t size,
                      const char *text) noexcept {
    if (buffer && size) {
        size_t length = 0;
        while (length < size - 1 && text[length]) ++length;
        std::memcpy(buffer, text, length);
        buffer[length] = '\0';
    }
    return static_cast<int>(code);
}
template<class T> struct Object {
    using Release = void (ORT_API_CALL *)(T *);
    T *value = nullptr;
    Release release;
    explicit Object(Release destroy) noexcept : release(destroy) {}
    ~Object() noexcept { if (value) release(value); }
    Object(const Object &) = delete;
    Object &operator=(const Object &) = delete;
};
struct TensorSpec {
    std::string name;
    std::array<int64_t, 8> shape{};
    size_t rank = 0;
};
inline size_t elements(const int64_t *shape, size_t rank, SeeonOrtResult code,
                       bool symbolic = false) {
    require(rank > 0 && rank <= 8, code, "tensor rank must be between 1 and 8");
    size_t count = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (symbolic && shape[i] == -1) continue;
        require(shape[i] > 0 && static_cast<uint64_t>(shape[i]) <= kMaxElements / count,
                code, "tensor dimensions are invalid or exceed 16 Mi elements");
        count *= static_cast<size_t>(shape[i]);
    }
    return count;
}
} // namespace seeon_ort

struct SeeonOrtModel {
    void *library = nullptr;
    const OrtApi *api = nullptr;
    OrtEnv *env = nullptr;
    OrtSession *session = nullptr;
    OrtMemoryInfo *memory = nullptr;
    OrtAllocator *allocator = nullptr; // Borrowed default allocator; never released.
    seeon_ort::TensorSpec input;
    std::array<seeon_ort::TensorSpec, seeon_ort::kMaxOutputs> outputs;
    SeeonOrtInfo info{}; // Immutable after publication.
    std::mutex mutex;
    bool poisoned = false; // Protected by mutex.
    SeeonOrtModel() = default;
    ~SeeonOrtModel() noexcept;
    SeeonOrtModel(const SeeonOrtModel &) = delete;
    SeeonOrtModel &operator=(const SeeonOrtModel &) = delete;
};
#endif

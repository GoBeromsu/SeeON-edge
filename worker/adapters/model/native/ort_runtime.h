#ifndef SEEON_ORT_RUNTIME_H
#define SEEON_ORT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#define SEEON_ORT_ABI_VERSION 1U
#define SEEON_ORT_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

struct SeeonOrtModel;
enum SeeonOrtResult {
    SEEON_ORT_OK = 0,
    SEEON_ORT_INVALID_ARGUMENT = 1,
    SEEON_ORT_UNAVAILABLE = 2,
    SEEON_ORT_MODEL = 3,
    SEEON_ORT_EXECUTION = 4,
    SEEON_ORT_OUTPUT = 5,
    SEEON_ORT_POISONED = 6,
};
struct SeeonOrtTensor {
    const char *name;
    float *data;
    size_t elements;
    int64_t shape[8];
    size_t rank;
};
struct SeeonOrtInfo {
    uint32_t abi_version;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t threads;
    char runtime_version[64];
};

/* ABI v1: one float32 input and one or two float32 outputs, rank 1..8,
 * positive resolved dimensions, at most 16 Mi elements per resolved tensor.
 * ONNX bytes are borrowed during open only, nonempty and at most 512 MiB.
 * threads must be 0 (ORT defaults) or 1 (both intra/inter-op counts are 1).
 * No accelerator execution provider is registered: ORT's default CPU provider
 * is the intentional backend, not an accelerator fallback.
 * The process must start with ORT_DISABLE_TELEMETRY=1 (image-owned policy).
 * Open refuses before loading ORT otherwise; no process environment is mutated.
 *
 * Caller owns valid, aligned storage and NUL-terminated names throughout each
 * synchronous call. Handle, diagnostic, name, descriptor and input data storage
 * must not overlap output destinations; output destinations must not overlap each other.
 * Calls to run are serialized internally; close must not overlap any call.
 * A null diagnostic buffer requires error_size == 0. Otherwise diagnostics
 * are bounded and NUL-terminated; successful calls clear the diagnostic.
 *
 * Input elements is the exact shape product. Output elements is capacity on
 * entry and the actual count on success; output shape/rank are ignored on entry.
 * All model outputs must be named exactly once, in any order. Output buffers,
 * elements, shape and rank are untouched on failure. Invalid caller arguments
 * do not poison the handle; native execution or output validation failures do
 * (including insufficient output capacity discovered after execution).
 * Info remains available after poisoning. No SDK pointers or evidence identity
 * are exposed through this ABI.
 */
SEEON_ORT_EXPORT int seeon_ort_open(const char *runtime_library, const void *onnx,
    size_t onnx_size, uint32_t threads, struct SeeonOrtModel **out,
    char *error, size_t error_size);
SEEON_ORT_EXPORT int seeon_ort_info(const struct SeeonOrtModel *model,
    struct SeeonOrtInfo *out, char *error, size_t error_size);
SEEON_ORT_EXPORT int seeon_ort_run(struct SeeonOrtModel *model,
    const struct SeeonOrtTensor *input, struct SeeonOrtTensor *outputs,
    size_t output_count, char *error, size_t error_size);
SEEON_ORT_EXPORT void seeon_ort_close(struct SeeonOrtModel *model);

#ifdef __cplusplus
}
#endif
#endif

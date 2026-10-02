#ifndef SEEON_GPU_RUNTIME_H
#define SEEON_GPU_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Owns one TensorRT context and CUDA stream. No SDK pointers cross this ABI. */
struct SeeonGpuModel;
struct SeeonGpuTensor {
    const char *name;
    float *values;
    size_t capacity;
    int32_t dimensions[8];
    int32_t rank;
};
struct SeeonGpuMetrics {
    uint64_t attempted;
    uint64_t succeeded;
    uint64_t failed;
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint64_t elapsed_ns;
    int32_t device;
};

/* Return 0 on success; all errors are static, credential-free text. No fallback.
 * Caller owns tensors/strings for the duration of the synchronous call. Output
 * buffers must cover the resolved shape, written back only on successful GPU
 * synchronization. A model handle must not be closed concurrently with a call.
 */
int seeon_gpu_open(const char *engine_path, int32_t device,
                   struct SeeonGpuModel **model, char *error, size_t error_size);
int seeon_gpu_run(struct SeeonGpuModel *model, const struct SeeonGpuTensor *input,
                  struct SeeonGpuTensor *outputs, size_t output_count,
                  char *error, size_t error_size);
int seeon_gpu_metrics(struct SeeonGpuModel *model, struct SeeonGpuMetrics *metrics);
void seeon_gpu_close(struct SeeonGpuModel *model);

/* Offline FP32 engine build: strongly typed from the ONNX, TF32 off, one static input profile. */
struct SeeonGpuBuildIdentity {
    int32_t trt_version;
    int32_t compute_major;
    int32_t compute_minor;
    int32_t tf32_enabled;
    char device_name[256];
};
/* Parses borrowed ONNX bytes, refuses non-FP32 IO or an input other than
 * `input_name`, and writes a new engine file (never overwrites). `onnx_data`
 * is borrowed only for this synchronous call and is not retained. The worker
 * never calls this at runtime. Empty, oversized (>512 MiB), or null input is
 * refused before CUDA.
 */
int seeon_gpu_build(const uint8_t *onnx_data, size_t onnx_size, const char *engine_path,
                    int32_t device, const char *input_name, const int32_t *dimensions,
                    int32_t rank, struct SeeonGpuBuildIdentity *identity, char *error,
                    size_t error_size);

/* NVML is loaded on demand; a missing library is a status, never a crash. */
enum {
    SEEON_NVML_OK = 0,
    SEEON_NVML_LIBRARY_MISSING = 1,
    SEEON_NVML_SYMBOL_MISSING = 2,
    SEEON_NVML_INIT_FAILED = 3,
    SEEON_NVML_DEVICE_COUNT_FAILED = 4,
    SEEON_NVML_NO_DEVICE = 5,
};
struct SeeonGpuDeviceReport {
    int32_t nvml_status;
    int32_t cuda_context_ok;
    int32_t has_driver_version;
    int32_t has_device_name;
    char driver_version[80];
    char device_name[96];
};
/* Mirrors the Python probe: NVML answers availability, driver and first device
 * name; a separate CUDA context on `device` answers usability. Returns 0 when
 * the report was filled, -1 only for a null report.
 */
int seeon_gpu_device_report(int32_t device, struct SeeonGpuDeviceReport *report);

/* Loaded TensorRT and CUDA runtime identities. Both fields are the exact
 * integer encodings returned by getInferLibVersion() and cudaRuntimeGetVersion();
 * they are not compile-time version macros and are not decomposed here.
 */
struct SeeonGpuRuntimeVersions {
    int32_t trt_version;
    int32_t cuda_runtime_version;
};
/* Fills both encodings from the linked runtimes. Returns 0 when the report was
 * filled, -1 for a null report or a refused query. Creates no inference owner
 * and no CUDA context. Errors stay static and path-free.
 */
int seeon_gpu_runtime_versions(struct SeeonGpuRuntimeVersions *versions);
/* CUDA device properties plus the loaded TensorRT library version for one
 * ordinal. Fields are measured by cudaGetDeviceProperties and
 * getInferLibVersion(); they are not compile-time macros, NVML names, or a
 * physical GPU UUID. Creates no builder, engine, or inference owner.
 */
struct SeeonGpuHardwareIdentity {
    int32_t trt_version;
    int32_t compute_major;
    int32_t compute_minor;
    char device_name[256];
};
/* Fills the requested ordinal's measured identity. Returns 0 only when every
 * field came from a successful query. Returns nonzero for a null output or a
 * negative ordinal before any CUDA call, and for a refused property or
 * version query. Errors stay static and path-free.
 */
int seeon_gpu_hardware_identity(int32_t device, struct SeeonGpuHardwareIdentity *identity);

#ifdef __cplusplus
}
#endif
#endif

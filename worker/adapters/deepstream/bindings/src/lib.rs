//! Safe, thread-confined access to the TensorRT/CUDA, DeepStream media and clip
//! decoder C ABIs.
//!
//! Build with `SEEON_GPU_LIB_DIR`, `SEEON_MEDIA_LIB_DIR` and `SEEON_CLIPDEC_LIB_DIR`
//! set to the exact canonical directories containing real `libseeon_gpu.so`,
//! `libseeon_media.so` and `libseeon_clipdec.so` (built by `../native/Makefile`);
//! deployment supplies their loader search paths. There is no CPU provider,
//! model substitution, or hardware-free link fallback. MediaOwner documents the
//! mandatory process-lifetime DSO retention policy when bounded shutdown fails.
//! Native diagnostics never escape as owned text or include caller paths here.
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(not(unix))]
compile_error!("seeon-deepstream-native requires the Unix native GPU/media runtime");

use std::ffi::{CStr, CString};
use std::mem::size_of;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

mod clip_decode;
mod gpu;
mod gpu_build;
mod nvml;
pub use clip_decode::{ClipDecoder, ClipError, ClipFrame};
pub use gpu::GpuModel;
pub use gpu_build::{EngineBuildIdentity, build_engine};
pub use nvml::{
    GpuDeviceReport, GpuHardwareIdentity, NvmlStatus, RuntimeVersions, device_report,
    hardware_identity, runtime_versions,
};

mod media;
mod media_ffi;
mod media_types;
pub use media::MediaOwner;
pub use media_types::{
    FrameIdentity, MEDIA_ABI_VERSION, MEDIA_MAX_OBJECTS, MEDIA_MAX_PREVIEW_BYTES,
    MEDIA_MAX_RECORDS, MEDIA_MAX_SOURCES, MEDIA_PATH_BYTES, MEDIA_POSE_COLUMNS, MEDIA_POSE_ROWS,
    MediaArgument, MediaBinding, MediaCallStatus, MediaConfig, MediaDiagnostic, MediaError,
    MediaPoll, MediaResult, MediaState, MediaStatus, PosePacket, PreviewPacket, RecordReceipt,
    RecordTicket, SourceConfig, SourceStatus, TrackedObject,
};

const MAX_RANK: usize = 8;
const MAX_OUTPUTS: usize = 8;
const MAX_ONNX_BYTES: usize = 512 * 1024 * 1024;
const MAX_ELEMENTS: usize = 256 * 1024 * 1024 / size_of::<f32>();

/// Path-free failures. Admission failure includes missing GPUs, unreadable or
/// unsupported engines, and unavailable TensorRT/CUDA; it never selects a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateError {
    InvalidDevice,
    InvalidPath,
    InvalidOnnx,
    InvalidName,
    InvalidShape,
    InvalidCapacity,
    InvalidOutputCount,
    Unavailable,
    ExecutionFailed,
    MetricsFailed,
    NativeContract,
    Poisoned,
}

impl std::fmt::Display for StateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDevice => "GPU device must be nonnegative",
            Self::InvalidPath => "engine path must be nonempty and contain no NUL",
            Self::InvalidOnnx => "ONNX bytes must be nonempty and at most 512 MiB",
            Self::InvalidName => "tensor name must be nonempty",
            Self::InvalidShape => "tensor rank, dimension, or dimension product is invalid",
            Self::InvalidCapacity => "tensor capacity is invalid or does not match its shape",
            Self::InvalidOutputCount => "output count must be between one and eight",
            Self::Unavailable => "GPU/model unavailable or unsupported; no fallback",
            Self::ExecutionFailed => "GPU execution failed; model poisoned",
            Self::MetricsFailed => "GPU metrics read failed; model poisoned",
            Self::NativeContract => "native GPU result violated the ABI contract",
            Self::Poisoned => "GPU model is unavailable after a failed native call",
        })
    }
}
impl std::error::Error for StateError {}

/// One contiguous float32 input, with rank 1..=8 and strictly positive dimensions.
/// Its length must equal the checked dimension product and fit within 256 MiB.
/// `CStr` guarantees a terminating NUL and excludes interior NULs.
pub struct TensorInput<'a> {
    pub name: &'a CStr,
    pub values: &'a [f32],
    pub dimensions: &'a [i32],
}

/// A nonempty, exclusively borrowed float32 buffer, at most 256 MiB. Outputs
/// follow the engine's declared order. Native admission checks resolved capacity.
pub struct TensorOutput<'a> {
    pub name: &'a CStr,
    pub values: &'a mut [f32],
}

/// Resolved only after successful GPU synchronization. Only the first `rank`
/// dimensions are significant; unused entries are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub rank: i32,
    pub dimensions: [i32; MAX_RANK],
}

/// Exact native cumulative counters, not evidence of image qualification.
/// `attempted` includes rejected native calls; elapsed time covers successes only.
/// Locally rejected arguments and locally refused poisoned runs do not count.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GpuMetrics {
    pub attempted: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub elapsed_ns: u64,
    pub device: i32,
}

// Private ABI mirror of ../native/gpu_runtime.h: C int/char/float use their
// platform aliases; int32_t/uint64_t and size_t map to i32/u64 and usize.
mod ffi {
    use super::GpuMetrics;
    use std::ffi::{c_char, c_float, c_int};

    #[repr(C)]
    pub struct Model {
        _opaque: [u8; 0],
    }
    #[repr(C)]
    #[derive(Default)]
    pub struct Tensor {
        pub name: *const c_char,
        pub values: *mut c_float,
        pub capacity: usize,
        pub dimensions: [i32; 8],
        pub rank: i32,
    }
    unsafe extern "C" {
        pub fn seeon_gpu_open(
            engine_path: *const c_char,
            device: i32,
            model: *mut *mut Model,
            error: *mut c_char,
            error_size: usize,
        ) -> c_int;
        pub fn seeon_gpu_run(
            model: *mut Model,
            input: *const Tensor,
            outputs: *mut Tensor,
            output_count: usize,
            error: *mut c_char,
            error_size: usize,
        ) -> c_int;
        pub fn seeon_gpu_metrics(model: *mut Model, metrics: *mut GpuMetrics) -> c_int;
        pub fn seeon_gpu_close(model: *mut Model);
        pub fn seeon_gpu_build(
            onnx_data: *const u8,
            onnx_size: usize,
            engine_path: *const c_char,
            device: i32,
            input_name: *const c_char,
            dimensions: *const i32,
            rank: i32,
            identity: *mut BuildIdentity,
            error: *mut c_char,
            error_size: usize,
        ) -> c_int;
        pub fn seeon_gpu_device_report(device: i32, report: *mut DeviceReport) -> c_int;
        pub fn seeon_gpu_runtime_versions(versions: *mut RuntimeVersions) -> c_int;
        pub fn seeon_gpu_hardware_identity(device: i32, identity: *mut HardwareIdentity) -> c_int;
    }
    #[repr(C)]
    pub struct BuildIdentity {
        pub trt_version: i32,
        pub compute_major: i32,
        pub compute_minor: i32,
        pub tf32_enabled: i32,
        pub device_name: [c_char; 256],
    }
    #[repr(C)]
    pub struct DeviceReport {
        pub nvml_status: i32,
        pub cuda_context_ok: i32,
        pub has_driver_version: i32,
        pub has_device_name: i32,
        pub driver_version: [c_char; 80],
        pub device_name: [c_char; 96],
    }
    #[repr(C)]
    pub struct RuntimeVersions {
        pub trt_version: i32,
        pub cuda_runtime_version: i32,
    }
    #[repr(C)]
    pub struct HardwareIdentity {
        pub trt_version: i32,
        pub compute_major: i32,
        pub compute_minor: i32,
        pub device_name: [c_char; 256],
    }
}

fn engine_path(path: &Path, device: i32) -> Result<CString, StateError> {
    if device < 0 {
        return Err(StateError::InvalidDevice);
    }
    let bytes = path.as_os_str().as_bytes();
    if bytes.is_empty() {
        return Err(StateError::InvalidPath);
    }
    CString::new(bytes).map_err(|_| StateError::InvalidPath)
}

/// Reads a NUL-terminated native text field; unterminated or non-UTF-8 text
/// violates the ABI contract.
fn fixed_text(field: &[std::ffi::c_char]) -> Result<String, StateError> {
    let bytes: Vec<u8> = field.iter().map(|&byte| byte as u8).collect();
    let text = CStr::from_bytes_until_nul(&bytes).map_err(|_| StateError::NativeContract)?;
    text.to_str()
        .map(str::to_owned)
        .map_err(|_| StateError::NativeContract)
}

fn validate_name(name: &CStr) -> Result<(), StateError> {
    if name.to_bytes().is_empty() {
        return Err(StateError::InvalidName);
    }
    Ok(())
}

fn validate_capacity(count: usize) -> Result<(), StateError> {
    if !(1..=MAX_ELEMENTS).contains(&count) {
        return Err(StateError::InvalidCapacity);
    }
    Ok(())
}

fn validate_output_count(count: usize) -> Result<(), StateError> {
    if !(1..=MAX_OUTPUTS).contains(&count) {
        return Err(StateError::InvalidOutputCount);
    }
    Ok(())
}

fn element_count(dimensions: &[i32]) -> Result<usize, StateError> {
    if !(1..=MAX_RANK).contains(&dimensions.len()) {
        return Err(StateError::InvalidShape);
    }
    let count = dimensions.iter().try_fold(1_usize, |count, &dimension| {
        if dimension <= 0 {
            return Err(StateError::InvalidShape);
        }
        count
            .checked_mul(dimension as usize)
            .ok_or(StateError::InvalidShape)
    })?;
    validate_capacity(count)?;
    Ok(count)
}

fn input_tensor(input: &TensorInput<'_>) -> Result<ffi::Tensor, StateError> {
    validate_name(input.name)?;
    if element_count(input.dimensions)? != input.values.len() {
        return Err(StateError::InvalidCapacity);
    }
    let mut dimensions = [0; MAX_RANK];
    dimensions[..input.dimensions.len()].copy_from_slice(input.dimensions);
    Ok(ffi::Tensor {
        name: input.name.as_ptr(),
        // The header shares a float* field with outputs, but run accepts a const
        // input struct and gpu_runtime.cpp only reads this buffer for H2D copies.
        values: input.values.as_ptr().cast_mut(),
        capacity: input.values.len(),
        dimensions,
        rank: input.dimensions.len() as i32,
    })
}

fn output_tensor(output: &mut TensorOutput<'_>) -> Result<ffi::Tensor, StateError> {
    validate_name(output.name)?;
    validate_capacity(output.values.len())?;
    Ok(ffi::Tensor {
        name: output.name.as_ptr(),
        values: output.values.as_mut_ptr(),
        capacity: output.values.len(),
        dimensions: [0; MAX_RANK],
        rank: 0,
    })
}

fn output_shape(output: &ffi::Tensor) -> Result<TensorShape, StateError> {
    let rank = usize::try_from(output.rank).map_err(|_| StateError::NativeContract)?;
    let dimensions = output
        .dimensions
        .get(..rank)
        .ok_or(StateError::NativeContract)?;
    let count = element_count(dimensions).map_err(|_| StateError::NativeContract)?;
    if count > output.capacity {
        return Err(StateError::NativeContract);
    }
    Ok(TensorShape {
        rank: output.rank,
        dimensions: output.dimensions,
    })
}

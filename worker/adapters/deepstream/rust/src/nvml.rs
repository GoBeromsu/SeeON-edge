use crate::{StateError, ffi, fixed_text};

/// Outcome of the NVML half of the device probe, in the order the Python probe
/// (`worker/runtime/telemetry/probe.py`) takes its steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvmlStatus {
    Ok,
    LibraryMissing,
    SymbolMissing,
    InitFailed,
    DeviceCountFailed,
    NoDevice,
}

impl NvmlStatus {
    fn from_native(code: i32) -> Result<Self, StateError> {
        Ok(match code {
            0 => Self::Ok,
            1 => Self::LibraryMissing,
            2 => Self::SymbolMissing,
            3 => Self::InitFailed,
            4 => Self::DeviceCountFailed,
            5 => Self::NoDevice,
            _ => return Err(StateError::NativeContract),
        })
    }

    /// Static, path-free reason for the relay `nvml_error` field; `None` when available.
    pub fn reason(self) -> Option<&'static str> {
        match self {
            Self::Ok => None,
            Self::LibraryMissing => Some("NVML library is not loadable"),
            Self::SymbolMissing => Some("NVML library lacks a required symbol"),
            Self::InitFailed => Some("nvmlInit failed"),
            Self::DeviceCountFailed => Some("nvmlDeviceGetCount failed"),
            Self::NoDevice => Some("NVML reports no GPU devices"),
        }
    }
}

/// One snapshot of `RelayGpuPayload` minus its capture time. NVML answers
/// availability, driver and the first device name; `cuda_context_ok` comes
/// from a separate CUDA context on the requested ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceReport {
    pub nvml: NvmlStatus,
    pub cuda_context_ok: bool,
    pub driver_version: Option<String>,
    pub device_name: Option<String>,
}

impl GpuDeviceReport {
    pub fn nvml_available(&self) -> bool {
        self.nvml == NvmlStatus::Ok
    }
}

/// Never fails for a missing driver or GPU: those are statuses. Only a broken
/// native result (unknown status, non-boolean flag, unterminated text) is an error.
pub fn device_report(device: i32) -> Result<GpuDeviceReport, StateError> {
    if device < 0 {
        return Err(StateError::InvalidDevice);
    }
    let mut report = ffi::DeviceReport {
        nvml_status: -1,
        cuda_context_ok: -1,
        has_driver_version: -1,
        has_device_name: -1,
        driver_version: [0; 80],
        device_name: [0; 96],
    };
    // SAFETY: the repr(C) report outlives this synchronous call; native writes
    // only within it, NUL-terminates both strings and retains no pointer.
    if unsafe { ffi::seeon_gpu_device_report(device, &mut report) } != 0 {
        return Err(StateError::NativeContract);
    }
    report_from_native(&report)
}

fn flag(value: i32) -> Result<bool, StateError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(StateError::NativeContract),
    }
}

fn report_from_native(report: &ffi::DeviceReport) -> Result<GpuDeviceReport, StateError> {
    let nvml = NvmlStatus::from_native(report.nvml_status)?;
    let driver_version = flag(report.has_driver_version)?
        .then(|| fixed_text(&report.driver_version))
        .transpose()?;
    let device_name = flag(report.has_device_name)?
        .then(|| fixed_text(&report.device_name))
        .transpose()?;
    // Device names exist only after a successful count, as in the Python probe.
    if device_name.is_some() && nvml != NvmlStatus::Ok {
        return Err(StateError::NativeContract);
    }
    Ok(GpuDeviceReport {
        nvml,
        cuda_context_ok: flag(report.cuda_context_ok)?,
        driver_version,
        device_name,
    })
}

/// Exact integer encodings from the loaded TensorRT and CUDA runtimes.
/// `trt_version` is `getInferLibVersion()`; `cuda_runtime_version` is
/// `cudaRuntimeGetVersion()`. Neither value is a compile-time macro or a
/// dotted-version decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeVersions {
    pub trt_version: i32,
    pub cuda_runtime_version: i32,
}

/// Queries the linked runtimes. A nonzero native status or a non-positive
/// encoding is a contract failure, not a substitute version.
pub fn runtime_versions() -> Result<RuntimeVersions, StateError> {
    let mut versions = ffi::RuntimeVersions {
        trt_version: 0,
        cuda_runtime_version: 0,
    };
    // SAFETY: the repr(C) report outlives this synchronous call; native writes
    // only the two int32 encodings and retains no pointer. The call creates no
    // inference owner and no CUDA context.
    if unsafe { ffi::seeon_gpu_runtime_versions(&mut versions) } != 0 {
        return Err(StateError::NativeContract);
    }
    versions_from_native(&versions)
}

fn versions_from_native(versions: &ffi::RuntimeVersions) -> Result<RuntimeVersions, StateError> {
    if versions.trt_version <= 0 || versions.cuda_runtime_version <= 0 {
        return Err(StateError::NativeContract);
    }
    Ok(RuntimeVersions {
        trt_version: versions.trt_version,
        cuda_runtime_version: versions.cuda_runtime_version,
    })
}
/// Measured CUDA device properties and the loaded TensorRT library version.
/// `trt_version` is `getInferLibVersion()`; compute capability and
/// `device_name` come from `cudaGetDeviceProperties` for the requested
/// ordinal. This is not an NVML report and not a physical GPU UUID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuHardwareIdentity {
    pub trt_version: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub device_name: String,
}
/// Queries one ordinal. A negative ordinal is refused before FFI. A nonzero
/// native status, missing NUL, blank name, non-positive TensorRT version or
/// compute major, or a negative compute minor is a contract failure.
pub fn hardware_identity(device: i32) -> Result<GpuHardwareIdentity, StateError> {
    if device < 0 {
        return Err(StateError::InvalidDevice);
    }
    let mut identity = ffi::HardwareIdentity {
        trt_version: 0,
        compute_major: 0,
        compute_minor: 0,
        device_name: [0; 256],
    };
    // SAFETY: the repr(C) identity outlives this synchronous call; native
    // writes only within it, NUL-terminates the name and retains no pointer.
    // The call creates no inference owner and does not require NVML.
    if unsafe { ffi::seeon_gpu_hardware_identity(device, &mut identity) } != 0 {
        return Err(StateError::NativeContract);
    }
    hardware_identity_from_native(&identity)
}
fn hardware_identity_from_native(
    identity: &ffi::HardwareIdentity,
) -> Result<GpuHardwareIdentity, StateError> {
    if identity.trt_version <= 0 || identity.compute_major <= 0 || identity.compute_minor < 0 {
        return Err(StateError::NativeContract);
    }
    let device_name = fixed_text(&identity.device_name)?;
    if device_name.trim().is_empty() {
        return Err(StateError::NativeContract);
    }
    Ok(GpuHardwareIdentity {
        trt_version: identity.trt_version,
        compute_major: identity.compute_major,
        compute_minor: identity.compute_minor,
        device_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_char;
    use std::mem::{align_of, offset_of, size_of};

    fn native(status: i32) -> ffi::DeviceReport {
        let mut report = ffi::DeviceReport {
            nvml_status: status,
            cuda_context_ok: 1,
            has_driver_version: 1,
            has_device_name: i32::from(status == 0),
            driver_version: [0; 80],
            device_name: [0; 96],
        };
        for (field, &byte) in report.driver_version.iter_mut().zip(b"595.84\0") {
            *field = byte as c_char;
        }
        for (field, &byte) in report.device_name.iter_mut().zip(b"GPU\0") {
            *field = byte as c_char;
        }
        report
    }

    #[test]
    fn device_report_layout_matches_header() {
        assert_eq!(
            [
                offset_of!(ffi::DeviceReport, nvml_status),
                offset_of!(ffi::DeviceReport, cuda_context_ok),
                offset_of!(ffi::DeviceReport, has_driver_version),
                offset_of!(ffi::DeviceReport, has_device_name),
                offset_of!(ffi::DeviceReport, driver_version),
                offset_of!(ffi::DeviceReport, device_name)
            ],
            [0, 4, 8, 12, 16, 96]
        );
        assert_eq!(align_of::<ffi::DeviceReport>(), align_of::<i32>());
        assert_eq!(size_of::<ffi::DeviceReport>(), 192);
    }

    #[test]
    fn statuses_map_one_to_one_and_unknown_codes_are_refused() {
        let ok = report_from_native(&native(0)).unwrap();
        assert!(ok.nvml_available() && ok.cuda_context_ok);
        assert_eq!(ok.nvml.reason(), None);
        assert_eq!(ok.driver_version.as_deref(), Some("595.84"));
        assert_eq!(ok.device_name.as_deref(), Some("GPU"));
        for (code, status) in [
            (1, NvmlStatus::LibraryMissing),
            (2, NvmlStatus::SymbolMissing),
            (3, NvmlStatus::InitFailed),
            (4, NvmlStatus::DeviceCountFailed),
            (5, NvmlStatus::NoDevice),
        ] {
            let report = report_from_native(&native(code)).unwrap();
            assert_eq!(report.nvml, status);
            assert!(!report.nvml_available());
            assert!(status.reason().is_some());
            assert_eq!(report.device_name, None);
        }
        for code in [-1, 6] {
            assert_eq!(
                report_from_native(&native(code)),
                Err(StateError::NativeContract)
            );
        }
    }

    #[test]
    fn flags_and_text_are_checked_before_they_are_trusted() {
        let mut raw = native(4);
        raw.has_device_name = 1;
        assert_eq!(
            report_from_native(&raw),
            Err(StateError::NativeContract),
            "a name without a device count is a contract break"
        );
        let mut raw = native(0);
        raw.has_driver_version = 0;
        assert_eq!(report_from_native(&raw).unwrap().driver_version, None);
        raw.cuda_context_ok = 2;
        assert_eq!(report_from_native(&raw), Err(StateError::NativeContract));
        let mut raw = native(0);
        raw.device_name = [b'x' as c_char; 96];
        assert_eq!(report_from_native(&raw), Err(StateError::NativeContract));
        assert_eq!(device_report(-1), Err(StateError::InvalidDevice));
    }

    #[test]
    fn runtime_versions_layout_matches_header() {
        assert_eq!(
            [
                offset_of!(ffi::RuntimeVersions, trt_version),
                offset_of!(ffi::RuntimeVersions, cuda_runtime_version)
            ],
            [0, 4]
        );
        assert_eq!(align_of::<ffi::RuntimeVersions>(), align_of::<i32>());
        assert_eq!(size_of::<ffi::RuntimeVersions>(), 8);
    }

    #[test]
    fn runtime_versions_require_two_positive_encodings() {
        assert_eq!(
            versions_from_native(&ffi::RuntimeVersions {
                trt_version: 101600,
                cuda_runtime_version: 12080,
            }),
            Ok(RuntimeVersions {
                trt_version: 101600,
                cuda_runtime_version: 12080,
            })
        );
        for versions in [
            ffi::RuntimeVersions {
                trt_version: 0,
                cuda_runtime_version: 12080,
            },
            ffi::RuntimeVersions {
                trt_version: -1,
                cuda_runtime_version: 12080,
            },
            ffi::RuntimeVersions {
                trt_version: 101600,
                cuda_runtime_version: 0,
            },
            ffi::RuntimeVersions {
                trt_version: 101600,
                cuda_runtime_version: -1,
            },
        ] {
            assert_eq!(
                versions_from_native(&versions),
                Err(StateError::NativeContract)
            );
        }
    }

    #[test]
    #[ignore = "requires SEEON_TEST_PYTHON and the actual linked TensorRT/CUDA libraries"]
    fn runtime_versions_match_direct_loaded_library_queries() {
        let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON");
        let output = std::process::Command::new(python)
            .args([
                "-c",
                r#"
import ctypes
trt = ctypes.CDLL("libnvinfer.so")
trt.getInferLibVersion.argtypes = []
trt.getInferLibVersion.restype = ctypes.c_int32
cuda = ctypes.CDLL("libcudart.so")
cuda.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuda.cudaRuntimeGetVersion.restype = ctypes.c_int
version = ctypes.c_int()
status = cuda.cudaRuntimeGetVersion(ctypes.byref(version))
assert status == 0, status
print(trt.getInferLibVersion(), version.value)
"#,
            ])
            .output()
            .expect("direct runtime version oracle");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<i32> = std::str::from_utf8(&output.stdout)
            .unwrap()
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(expected.len(), 2);
        let actual = runtime_versions().expect("real native query");
        assert_eq!(
            [actual.trt_version, actual.cuda_runtime_version],
            expected.as_slice()
        );
        println!("measured_runtime_versions={actual:?}");
    }
    fn hardware(name: &[u8]) -> ffi::HardwareIdentity {
        let mut identity = ffi::HardwareIdentity {
            trt_version: 101600,
            compute_major: 12,
            compute_minor: 0,
            device_name: [0; 256],
        };
        for (field, &byte) in identity.device_name.iter_mut().zip(name) {
            *field = byte as c_char;
        }
        identity
    }
    #[test]
    fn hardware_identity_layout_matches_header() {
        assert_eq!(
            [
                offset_of!(ffi::HardwareIdentity, trt_version),
                offset_of!(ffi::HardwareIdentity, compute_major),
                offset_of!(ffi::HardwareIdentity, compute_minor),
                offset_of!(ffi::HardwareIdentity, device_name)
            ],
            [0, 4, 8, 12]
        );
        assert_eq!(align_of::<ffi::HardwareIdentity>(), align_of::<i32>());
        assert_eq!(size_of::<ffi::HardwareIdentity>(), 268);
    }
    #[test]
    fn hardware_identity_decode_refuses_incomplete_facts() {
        let identity = hardware_identity_from_native(&hardware(b"GPU\0")).unwrap();
        assert_eq!(
            identity,
            GpuHardwareIdentity {
                trt_version: 101600,
                compute_major: 12,
                compute_minor: 0,
                device_name: "GPU".to_owned(),
            }
        );
        for name in [b"\0" as &[u8], b" \0", b"\t\0", &[b'x'; 256], b"\xff\0"] {
            assert_eq!(
                hardware_identity_from_native(&hardware(name)),
                Err(StateError::NativeContract)
            );
        }
        for (trt_version, compute_major, compute_minor) in [
            (0, 12, 0),
            (-1, 12, 0),
            (101600, 0, 0),
            (101600, -1, 0),
            (101600, 12, -1),
        ] {
            let mut raw = hardware(b"GPU\0");
            raw.trt_version = trt_version;
            raw.compute_major = compute_major;
            raw.compute_minor = compute_minor;
            assert_eq!(
                hardware_identity_from_native(&raw),
                Err(StateError::NativeContract)
            );
        }
        assert_eq!(hardware_identity(-1), Err(StateError::InvalidDevice));
    }
    #[test]
    #[ignore = "requiresGPU"]
    fn hardware_identity_matches_requested_ordinal() {
        let identity = hardware_identity(0).expect("requested GPU ordinal");
        assert!(identity.trt_version > 0);
        assert!(identity.compute_major > 0);
        assert!(identity.compute_minor >= 0);
        assert!(!identity.device_name.trim().is_empty());
    }
}

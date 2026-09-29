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
}

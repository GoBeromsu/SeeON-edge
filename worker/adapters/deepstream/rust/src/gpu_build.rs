use crate::{StateError, element_count, engine_path, ffi, fixed_text, validate_name};
use std::ffi::{CStr, c_char};
use std::path::Path;

/// What TensorRT reported for one offline build; the caller records it next to
/// the ONNX and engine sha256 in `engine-identity.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineBuildIdentity {
    /// `getInferLibVersion()`, e.g. 101600 for TensorRT 10.16.0.
    pub trt_version: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub tf32_enabled: bool,
    pub device_name: String,
}

/// Builds a strongly typed FP32 engine with TF32 off and one static input
/// profile, writing a new file at `engine` (never overwriting). Offline only:
/// the worker never builds at runtime. The native call selects `device` on the
/// calling thread. Native diagnostics are not copied; failure is `Unavailable`.
pub fn build_engine(
    onnx: &Path,
    engine: &Path,
    device: i32,
    input_name: &CStr,
    dimensions: &[i32],
) -> Result<EngineBuildIdentity, StateError> {
    let onnx = engine_path(onnx, device)?;
    let engine = engine_path(engine, device)?;
    validate_name(input_name)?;
    element_count(dimensions)?;
    let mut identity = ffi::BuildIdentity {
        trt_version: 0,
        compute_major: 0,
        compute_minor: 0,
        tf32_enabled: 0,
        device_name: [0; 256],
    };
    let mut error: [c_char; 256] = [0; 256];
    // SAFETY: both terminated paths, the terminated input name, the dimension
    // slice (rank checked to 1..=8 above) and the repr(C) identity outlive this
    // synchronous call. Native retains no pointer and writes at most the sizes given.
    let status = unsafe {
        ffi::seeon_gpu_build(
            onnx.as_ptr(),
            engine.as_ptr(),
            device,
            input_name.as_ptr(),
            dimensions.as_ptr(),
            dimensions.len() as i32,
            &mut identity,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 {
        return Err(StateError::Unavailable);
    }
    identity_from_native(&identity)
}

fn identity_from_native(identity: &ffi::BuildIdentity) -> Result<EngineBuildIdentity, StateError> {
    let tf32_enabled = match identity.tf32_enabled {
        0 => false,
        1 => true,
        _ => return Err(StateError::NativeContract),
    };
    if identity.trt_version <= 0 || identity.compute_major <= 0 || identity.compute_minor < 0 {
        return Err(StateError::NativeContract);
    }
    Ok(EngineBuildIdentity {
        trt_version: identity.trt_version,
        compute_major: identity.compute_major,
        compute_minor: identity.compute_minor,
        tf32_enabled,
        device_name: fixed_text(&identity.device_name)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    fn native(name: &[u8]) -> ffi::BuildIdentity {
        let mut identity = ffi::BuildIdentity {
            trt_version: 101600,
            compute_major: 12,
            compute_minor: 0,
            tf32_enabled: 0,
            device_name: [0; 256],
        };
        for (field, &byte) in identity.device_name.iter_mut().zip(name) {
            *field = byte as c_char;
        }
        identity
    }

    #[test]
    fn build_identity_layout_matches_header() {
        assert_eq!(
            [
                offset_of!(ffi::BuildIdentity, trt_version),
                offset_of!(ffi::BuildIdentity, compute_major),
                offset_of!(ffi::BuildIdentity, compute_minor),
                offset_of!(ffi::BuildIdentity, tf32_enabled),
                offset_of!(ffi::BuildIdentity, device_name)
            ],
            [0, 4, 8, 12, 16]
        );
        assert_eq!(align_of::<ffi::BuildIdentity>(), align_of::<i32>());
        assert_eq!(size_of::<ffi::BuildIdentity>(), 272);
    }

    #[test]
    fn native_identity_is_checked_before_it_is_trusted() {
        let identity = identity_from_native(&native(b"GPU\0")).unwrap();
        assert_eq!(identity.device_name, "GPU");
        assert!(!identity.tf32_enabled);
        let mut raw = native(b"GPU\0");
        raw.tf32_enabled = 1;
        assert!(identity_from_native(&raw).unwrap().tf32_enabled);
        for tf32 in [-1, 2] {
            raw.tf32_enabled = tf32;
            assert_eq!(identity_from_native(&raw), Err(StateError::NativeContract));
        }
        assert_eq!(
            identity_from_native(&native(&[b'x'; 256])),
            Err(StateError::NativeContract),
            "an unterminated name is refused"
        );
        assert_eq!(
            identity_from_native(&native(b"\xff\0")),
            Err(StateError::NativeContract)
        );
        let mut raw = native(b"GPU\0");
        raw.trt_version = 0;
        assert_eq!(identity_from_native(&raw), Err(StateError::NativeContract));
    }

    #[test]
    fn build_arguments_are_rejected_locally() {
        let path = Path::new("model.onnx");
        let dims = [1, 3, 640, 640];
        assert_eq!(
            build_engine(path, Path::new(""), 0, c"images", &dims),
            Err(StateError::InvalidPath)
        );
        assert_eq!(
            build_engine(path, path, -1, c"images", &dims),
            Err(StateError::InvalidDevice)
        );
        assert_eq!(
            build_engine(path, path, 0, c"", &dims),
            Err(StateError::InvalidName)
        );
        for dims in [&[][..], &[1, 0], &[1; 9]] {
            assert_eq!(
                build_engine(path, path, 0, c"images", dims),
                Err(StateError::InvalidShape)
            );
        }
    }
}

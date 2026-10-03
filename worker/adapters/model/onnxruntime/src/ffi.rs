//! Private exact mirror of ../native/ort_runtime.h plus admission checks.
use crate::{DIAGNOSTIC_BYTES, Error, ErrorKind, Input, MAX_RANK, Output, Shape, Threads};
use std::{
    ffi::{CStr, CString, c_char, c_float, c_int, c_void},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
};
const MAX_ELEMENTS: usize = 16 * 1024 * 1024;

#[repr(C)]
pub(super) struct Model {
    _opaque: [u8; 0],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct Tensor {
    pub name: *const c_char,
    pub data: *mut c_float,
    pub elements: usize,
    pub shape: [i64; 8],
    pub rank: usize,
}
#[repr(C)]
pub(super) struct Info {
    pub abi_version: u32,
    pub input_count: u32,
    pub output_count: u32,
    pub threads: u32,
    pub runtime_version: [c_char; 64],
}
impl Info {
    pub(super) fn empty() -> Self {
        Self {
            abi_version: 0,
            input_count: 0,
            output_count: 0,
            threads: 0,
            runtime_version: [0; 64],
        }
    }
}
unsafe extern "C" {
    pub(super) fn seeon_ort_open(
        runtime_library: *const c_char,
        onnx: *const c_void,
        onnx_size: usize,
        threads: u32,
        out: *mut *mut Model,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub(super) fn seeon_ort_info(
        model: *const Model,
        out: *mut Info,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub(super) fn seeon_ort_run(
        model: *mut Model,
        input: *const Tensor,
        outputs: *mut Tensor,
        output_count: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub(super) fn seeon_ort_close(model: *mut Model);
}

pub(super) struct OpenGuard(pub NonNull<Model>);
impl OpenGuard {
    pub(super) fn into_handle(self) -> NonNull<Model> {
        let handle = self.0;
        std::mem::forget(self);
        handle
    }
}
impl Drop for OpenGuard {
    fn drop(&mut self) {
        // SAFETY: a nonnull handle returned by open belongs to this guard alone
        // until transfer. Rejected metadata/partial admission cannot leak it.
        unsafe { seeon_ort_close(self.0.as_ptr()) };
    }
}

impl Error {
    pub(super) fn new(kind: ErrorKind, text: &str) -> Self {
        let mut diagnostic = [0; DIAGNOSTIC_BYTES];
        let length = text.len().min(DIAGNOSTIC_BYTES - 1);
        diagnostic[..length].copy_from_slice(&text.as_bytes()[..length]);
        Self {
            kind,
            diagnostic,
            length,
        }
    }
    pub(super) fn native(status: c_int, text: &[c_char; DIAGNOSTIC_BYTES]) -> Self {
        let kind = match status {
            1 => ErrorKind::InvalidArgument,
            2 => ErrorKind::Unavailable,
            3 => ErrorKind::Model,
            4 => ErrorKind::Execution,
            5 => ErrorKind::Output,
            6 => ErrorKind::Poisoned,
            other => ErrorKind::UnknownNativeStatus(other),
        };
        let diagnostic = text.map(|byte| byte as u8);
        let length = diagnostic
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(DIAGNOSTIC_BYTES);
        Self {
            kind,
            diagnostic,
            length,
        }
    }
}

pub(super) fn runtime_path(path: &Path) -> Result<CString, Error> {
    if !path.is_absolute() {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "runtime library path must be absolute",
        ));
    }
    CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        Error::new(
            ErrorKind::InvalidArgument,
            "runtime library path contains NUL",
        )
    })
}

pub(super) fn info(raw: &Info, threads: Threads) -> Result<crate::Info, Error> {
    if raw.abi_version != 1
        || raw.input_count != 1
        || !(1..=2).contains(&raw.output_count)
        || raw.threads != threads as u32
    {
        return Err(Error::new(
            ErrorKind::Model,
            "native info violates ABI v1 metadata",
        ));
    }
    let version = raw.runtime_version.map(|byte| byte as u8);
    let version = CStr::from_bytes_until_nul(&version)
        .map_err(|_| Error::new(ErrorKind::Model, "native runtime version is unterminated"))?
        .to_str()
        .map_err(|_| Error::new(ErrorKind::Model, "native runtime version is not UTF-8"))?;
    if version.is_empty() {
        return Err(Error::new(
            ErrorKind::Model,
            "native runtime version is empty",
        ));
    }
    Ok(crate::Info {
        abi_version: raw.abi_version,
        input_count: raw.input_count,
        output_count: raw.output_count,
        threads,
        runtime_version: version.to_owned(),
    })
}

fn elements(shape: &[i64]) -> Option<usize> {
    if !(1..=MAX_RANK).contains(&shape.len()) {
        return None;
    }
    shape.iter().try_fold(1usize, |count, &dimension| {
        let dimension = usize::try_from(dimension).ok()?;
        let count = count.checked_mul(dimension)?;
        (count > 0 && count <= MAX_ELEMENTS).then_some(count)
    })
}
pub(super) fn input_tensor(input: &Input<'_>) -> Result<Tensor, Error> {
    if input.name.to_bytes().is_empty()
        || elements(input.shape) != Some(input.data.len())
        || !input.data.iter().all(|value| value.is_finite())
    {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "input name, shape, size or finite values are invalid",
        ));
    }
    let mut shape = [0; MAX_RANK];
    shape[..input.shape.len()].copy_from_slice(input.shape);
    Ok(Tensor {
        name: input.name.as_ptr(),
        data: input.data.as_ptr().cast_mut(),
        elements: input.data.len(),
        shape,
        rank: input.shape.len(),
    })
}
pub(super) fn output_tensor(output: &mut Output<'_>) -> Result<Tensor, Error> {
    if output.name.to_bytes().is_empty() || output.data.len() > MAX_ELEMENTS {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "output name or capacity exceeds the tensor contract",
        ));
    }
    // Zero capacity is legal to describe; resolved positive outputs will fail
    // native capacity validation and poison, exactly as in the C ABI.
    Ok(Tensor {
        name: output.name.as_ptr(),
        data: output.data.as_mut_ptr(),
        elements: output.data.len(),
        shape: [0; MAX_RANK],
        rank: 0,
    })
}
pub(super) fn output_shape(raw: &Tensor, output: &mut Output<'_>) -> Result<Shape, Error> {
    let invalid = || {
        Error::new(
            ErrorKind::Output,
            "successful native result violates the output ABI",
        )
    };
    let dimensions = raw.shape.get(..raw.rank).ok_or_else(invalid)?;
    let count = elements(dimensions).ok_or_else(invalid)?;
    if raw.name != output.name.as_ptr()
        || raw.data != output.data.as_mut_ptr()
        || raw.elements != count
        || count > output.data.len()
        || !output
            .data
            .get(..count)
            .is_some_and(|values| values.iter().all(|value| value.is_finite()))
    {
        return Err(invalid());
    }
    let mut shape = [0; MAX_RANK];
    shape[..dimensions.len()].copy_from_slice(dimensions);
    Ok(Shape {
        dimensions: shape,
        rank: dimensions.len(),
        elements: count,
    })
}

//! Thread-confined ownership of the ONNX Runtime CPU-only native ABI v1.
//! Build-time SEEON_ORT_LIB_DIR selects libseeon_ort.so; deployment owns its
//! loader search path. The actual ORT library is an explicit absolute open argument.
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(not(unix))]
compile_error!("seeon-onnxruntime-native requires the Unix native ORT runtime");

use std::{ffi::CStr, fmt, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};
mod ffi;
const MAX_OUTPUTS: usize = 2;
const MAX_RANK: usize = 8;
// Native diagnostics are static and at most 67 bytes; keep errors compact
// without allocating on failure or truncating any current diagnostic.
const DIAGNOSTIC_BYTES: usize = 96;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threads {
    Default = 0,
    Single = 1,
}

/// Immutable admission metadata copied from the loaded native runtime, not GPU evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    pub abi_version: u32,
    pub input_count: u32,
    pub output_count: u32,
    pub threads: Threads,
    pub runtime_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    Unavailable,
    Model,
    Execution,
    Output,
    Poisoned,
    UnknownNativeStatus(i32),
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    diagnostic: [u8; DIAGNOSTIC_BYTES],
    length: usize,
}
impl Error {
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    /// Exact bounded bytes before the first NUL; no UTF-8 assumption or allocation.
    pub fn diagnostic(&self) -> &[u8] {
        &self.diagnostic[..self.length]
    }
}
impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?}: {}",
            self.kind,
            self.diagnostic().escape_ascii()
        )
    }
}
impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}
impl std::error::Error for Error {}

pub struct Input<'a> {
    pub name: &'a CStr,
    pub data: &'a [f32],
    pub shape: &'a [i64],
}
pub struct Output<'a> {
    pub name: &'a CStr,
    pub data: &'a mut [f32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dimensions: [i64; MAX_RANK],
    rank: usize,
    elements: usize,
}
impl Shape {
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions[..self.rank]
    }
    pub fn elements(&self) -> usize {
        self.elements
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RunResult {
    shapes: [Shape; MAX_OUTPUTS],
    length: usize,
}
impl RunResult {
    /// Returned entries only, in the caller's output order.
    pub fn shapes(&self) -> &[Shape] {
        &self.shapes[..self.length]
    }
}
impl fmt::Debug for RunResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("RunResult")
            .field(&self.shapes())
            .finish()
    }
}

/// Construct, run and drop on the owning actor thread. Neither Send nor Sync;
/// exclusive run prevents concurrent close. No native pointer is public.
pub struct Model {
    handle: NonNull<ffi::Model>,
    info: Info,
    poisoned: bool,
    _thread_confined: PhantomData<Rc<()>>,
}
impl Model {
    /// Bytes are borrowed only during admission. No provider or path fallback.
    pub fn open(runtime_library: &Path, onnx: &[u8], threads: Threads) -> Result<Self, Error> {
        let library = ffi::runtime_path(runtime_library)?;
        if onnx.is_empty() || onnx.len() > 512 * 1024 * 1024 {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "ONNX bytes must be nonempty and at most 512 MiB",
            ));
        }
        let mut raw = std::ptr::null_mut();
        let mut diagnostic = [0; DIAGNOSTIC_BYTES];
        // SAFETY: the CString, captured bytes and writable out-parameters live
        // through this synchronous call. Native retains no caller storage.
        let status = unsafe {
            ffi::seeon_ort_open(
                library.as_ptr(),
                onnx.as_ptr().cast(),
                onnx.len(),
                threads as u32,
                &mut raw,
                diagnostic.as_mut_ptr(),
                diagnostic.len(),
            )
        };
        // Close even an unexpected partial handle on every subsequent rejection.
        let guard = NonNull::new(raw).map(ffi::OpenGuard);
        if status != 0 {
            return Err(Error::native(status, &diagnostic));
        }
        let guard = guard
            .ok_or_else(|| Error::new(ErrorKind::Model, "native open returned a null handle"))?;
        let mut raw_info = ffi::Info::empty();
        // SAFETY: guard owns the live handle; raw_info and diagnostic have the
        // exact C layouts and capacities. Native copies metadata synchronously.
        let status = unsafe {
            ffi::seeon_ort_info(
                guard.0.as_ptr(),
                &mut raw_info,
                diagnostic.as_mut_ptr(),
                diagnostic.len(),
            )
        };
        if status != 0 {
            return Err(Error::native(status, &diagnostic));
        }
        let info = ffi::info(&raw_info, threads)?;
        Ok(Self {
            handle: guard.into_handle(),
            info,
            poisoned: false,
            _thread_confined: PhantomData,
        })
    }

    /// Cached, validated actual metadata; allocation-free even after poisoning.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// No Rust-side allocation during run; ONNX Runtime may allocate.
    /// Input refusals do not poison.
    /// Every native run failure except InvalidArgument permanently fails closed,
    /// as does malformed successful metadata. Discard output data on error.
    pub fn run(
        &mut self,
        input: Input<'_>,
        outputs: &mut [Output<'_>],
    ) -> Result<RunResult, Error> {
        if self.poisoned {
            return Err(Error::new(
                ErrorKind::Poisoned,
                "model is permanently poisoned",
            ));
        }
        if outputs.len() != self.info.output_count as usize || outputs.len() > MAX_OUTPUTS {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "output count does not match the model",
            ));
        }
        let native_input = ffi::input_tensor(&input)?;
        let mut tensors = [ffi::Tensor::default(); MAX_OUTPUTS];
        let native_outputs = &mut tensors[..outputs.len()];
        for (native, output) in native_outputs.iter_mut().zip(outputs.iter_mut()) {
            *native = ffi::output_tensor(output)?;
        }
        let mut diagnostic = [0; DIAGNOSTIC_BYTES];
        // SAFETY: exclusive self and Rust borrows exclude overlapping calls,
        // close, and buffer aliases. Checked descriptors, names and storage
        // outlive the call. Native only reads the float* input and writes output
        // within supplied capacities after validating all results; it retains
        // no pointer and returns synchronously, including failure paths.
        let status = unsafe {
            ffi::seeon_ort_run(
                self.handle.as_ptr(),
                &native_input,
                native_outputs.as_mut_ptr(),
                native_outputs.len(),
                diagnostic.as_mut_ptr(),
                diagnostic.len(),
            )
        };
        if status != 0 {
            let error = Error::native(status, &diagnostic);
            self.poisoned = error.kind() != ErrorKind::InvalidArgument;
            return Err(error);
        }
        let mut result = RunResult {
            shapes: [Shape {
                dimensions: [0; MAX_RANK],
                rank: 0,
                elements: 0,
            }; MAX_OUTPUTS],
            length: outputs.len(),
        };
        for ((shape, native), output) in result
            .shapes
            .iter_mut()
            .zip(native_outputs)
            .zip(outputs.iter_mut())
        {
            match ffi::output_shape(native, output) {
                Ok(value) => *shape = value,
                Err(error) => {
                    self.poisoned = true;
                    return Err(error);
                }
            }
        }
        Ok(result)
    }
}
impl fmt::Debug for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Model")
            .field("info", &self.info)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: this unique thread-confined owner closes exactly once; no
        // borrow/call overlaps Drop. Native releases ORT objects before dlclose.
        unsafe { ffi::seeon_ort_close(self.handle.as_ptr()) };
    }
}

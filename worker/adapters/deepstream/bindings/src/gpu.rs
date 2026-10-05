use crate::{
    GpuMetrics, MAX_OUTPUTS, StateError, TensorInput, TensorOutput, TensorShape, engine_path, ffi,
    input_tensor, output_shape, output_tensor, validate_output_count,
};
use std::ffi::c_char;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;

/// Sole owner of a native TensorRT context and CUDA stream. Not `Send` or `Sync`:
/// all calls and destruction stay on the opening thread, without concurrent close.
/// Native run synchronizes even on failure; close synchronizes before destruction.
/// These are required lifetime guarantees of `gpu_runtime.cpp`, not Rust polling.
pub struct GpuModel {
    handle: NonNull<ffi::Model>,
    poisoned: bool,
    _thread_confined: PhantomData<Rc<()>>,
}

impl GpuModel {
    /// Admit the specified engine on exactly this GPU; no fallback or path logging.
    pub fn open(path: &Path, device: i32) -> Result<Self, StateError> {
        let path = engine_path(path, device)?;
        let mut handle = std::ptr::null_mut();
        let mut error: [c_char; 256] = [0; 256];
        // SAFETY: the terminated Unix path and writable out-parameters live for
        // this synchronous call. Native owns partial allocations on failure and
        // returns a uniquely owned nonnull handle only on success.
        let status = unsafe {
            ffi::seeon_gpu_open(
                path.as_ptr(),
                device,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            // Do not copy native diagnostics or caller-controlled paths into errors.
            return Err(StateError::Unavailable);
        }
        Ok(Self {
            handle: NonNull::new(handle).ok_or(StateError::NativeContract)?,
            poisoned: false,
            _thread_confined: PhantomData,
        })
    }

    /// Run synchronously, returning resolved shapes only after checked success.
    /// Local validation errors make no native call and leave the model usable.
    /// Any native run failure permanently poisons this owner; discard all output
    /// values on error. Engine-dependent names/shapes/capacities are checked natively.
    pub fn run(
        &mut self,
        input: TensorInput<'_>,
        outputs: &mut [TensorOutput<'_>],
    ) -> Result<Vec<TensorShape>, StateError> {
        if self.poisoned {
            return Err(StateError::Poisoned);
        }
        validate_output_count(outputs.len())?;
        let native_input = input_tensor(&input)?;
        let mut tensors: [ffi::Tensor; MAX_OUTPUTS] =
            std::array::from_fn(|_| ffi::Tensor::default());
        let native_outputs = &mut tensors[..outputs.len()];
        for (native, output) in native_outputs.iter_mut().zip(outputs.iter_mut()) {
            *native = output_tensor(output)?;
        }
        let mut error: [c_char; 256] = [0; 256];
        // SAFETY: exclusive self prevents overlapping calls or close. The caller's
        // borrowed names and contiguous buffers outlive this call; Rust's mutable
        // output borrows exclude aliases with each other and the immutable input.
        // Native only reads input.values despite the ABI's float* field, validates
        // every output capacity before writing, and synchronizes on every return.
        let status = unsafe {
            ffi::seeon_gpu_run(
                self.handle.as_ptr(),
                &native_input,
                native_outputs.as_mut_ptr(),
                native_outputs.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            self.poisoned = true;
            return Err(StateError::ExecutionFailed);
        }
        // Native only publishes ranks/dimensions after successful synchronization.
        let shapes: Result<Vec<_>, _> = native_outputs.iter().map(output_shape).collect();
        if shapes.is_err() {
            self.poisoned = true;
        }
        shapes
    }

    /// Read native counters, including after poisoning. A failed read also
    /// poisons this owner; successful diagnostics cannot make it runnable again.
    pub fn metrics(&mut self) -> Result<GpuMetrics, StateError> {
        let mut metrics = GpuMetrics::default();
        // SAFETY: the live, exclusively borrowed handle and repr(C) output match
        // the header. Native copies metrics synchronously and retains no pointer.
        if unsafe { ffi::seeon_gpu_metrics(self.handle.as_ptr(), &mut metrics) } != 0 {
            self.poisoned = true;
            return Err(StateError::MetricsFailed);
        }
        Ok(metrics)
    }
}

impl Drop for GpuModel {
    fn drop(&mut self) {
        // SAFETY: this non-cloneable owner closes exactly once on its opening
        // thread. No call can overlap Drop; native close drains the CUDA stream
        // before releasing its context, buffers, engine and runtime.
        unsafe { ffi::seeon_gpu_close(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MAX_ELEMENTS, element_count, validate_capacity};
    use std::ffi::{CStr, CString, OsStr};
    use std::mem::{align_of, offset_of, size_of};
    use std::os::unix::ffi::OsStrExt;

    #[test]
    fn c_abi_layout_matches_header() {
        let word = size_of::<usize>();
        assert_eq!(size_of::<*const c_char>(), word);
        assert_eq!(align_of::<ffi::Tensor>(), align_of::<usize>());
        assert_eq!(
            [
                offset_of!(ffi::Tensor, name),
                offset_of!(ffi::Tensor, values),
                offset_of!(ffi::Tensor, capacity),
                offset_of!(ffi::Tensor, dimensions),
                offset_of!(ffi::Tensor, rank)
            ],
            [0, word, 2 * word, 3 * word, 3 * word + 32]
        );
        assert_eq!(
            size_of::<ffi::Tensor>(),
            (3 * word + 36).next_multiple_of(align_of::<ffi::Tensor>())
        );
        assert_eq!(
            [
                offset_of!(GpuMetrics, attempted),
                offset_of!(GpuMetrics, succeeded),
                offset_of!(GpuMetrics, failed),
                offset_of!(GpuMetrics, host_to_device_bytes),
                offset_of!(GpuMetrics, device_to_host_bytes),
                offset_of!(GpuMetrics, elapsed_ns),
                offset_of!(GpuMetrics, device)
            ],
            [0, 8, 16, 24, 32, 40, 48]
        );
        assert_eq!(align_of::<GpuMetrics>(), align_of::<u64>());
        assert_eq!(
            size_of::<GpuMetrics>(),
            52_usize.next_multiple_of(align_of::<u64>())
        );
    }

    #[test]
    fn paths_use_unix_bytes_and_reject_nul_empty_or_negative_device_locally() {
        for bytes in [b"".as_slice(), b"/private/secret\0.engine".as_slice()] {
            assert_eq!(
                engine_path(Path::new(OsStr::from_bytes(bytes)), 0).err(),
                Some(StateError::InvalidPath)
            );
        }
        let path = Path::new(OsStr::from_bytes(b"\xff.engine"));
        assert_eq!(engine_path(path, 0).unwrap().as_bytes(), b"\xff.engine");
        assert_eq!(engine_path(path, -1).err(), Some(StateError::InvalidDevice));
    }

    #[test]
    fn input_validation_checks_rank_dimensions_product_and_exact_length() {
        for dimensions in [&[][..], &[1; 9], &[0], &[-1], &[i32::MAX; 8]] {
            assert_eq!(element_count(dimensions), Err(StateError::InvalidShape));
        }
        assert_eq!(
            element_count(&[MAX_ELEMENTS as i32 + 1]),
            Err(StateError::InvalidCapacity)
        );
        assert_eq!(element_count(&[1, 30, 56]), Ok(1680));
        let mut input = TensorInput {
            name: c"window",
            values: &[0.0; 2],
            dimensions: &[1, 2],
        };
        assert_eq!(
            input_tensor(&input).unwrap().dimensions,
            [1, 2, 0, 0, 0, 0, 0, 0]
        );
        for dimensions in [&[1][..], &[3]] {
            input.dimensions = dimensions;
            assert_eq!(
                input_tensor(&input).err(),
                Some(StateError::InvalidCapacity)
            );
        }
        input.dimensions = &[1, 2];
        input.values = &[];
        assert_eq!(
            input_tensor(&input).err(),
            Some(StateError::InvalidCapacity)
        );
        input.name = c"";
        assert_eq!(input_tensor(&input).err(), Some(StateError::InvalidName));
    }

    #[test]
    fn outputs_enforce_bounded_capacity_count_names_and_resolved_shapes() {
        for count in [0, MAX_ELEMENTS + 1, usize::MAX] {
            assert_eq!(validate_capacity(count), Err(StateError::InvalidCapacity));
        }
        for count in [1, MAX_ELEMENTS] {
            assert_eq!(validate_capacity(count), Ok(()));
        }
        for count in [0, MAX_OUTPUTS + 1] {
            assert_eq!(
                validate_output_count(count),
                Err(StateError::InvalidOutputCount)
            );
        }
        for count in [1, MAX_OUTPUTS] {
            assert_eq!(validate_output_count(count), Ok(()));
        }
        let mut empty = TensorOutput {
            name: c"output",
            values: &mut [],
        };
        assert_eq!(
            output_tensor(&mut empty).err(),
            Some(StateError::InvalidCapacity)
        );
        let mut value = [0.0];
        let mut output = TensorOutput {
            name: c"",
            values: &mut value,
        };
        assert_eq!(
            output_tensor(&mut output).err(),
            Some(StateError::InvalidName)
        );
        output.name = c"output";
        let mut raw = output_tensor(&mut output).unwrap();
        raw.rank = 2;
        raw.dimensions[..2].copy_from_slice(&[1, 1]);
        assert_eq!(output_shape(&raw).unwrap().dimensions[..2], [1, 1]);
        for rank in [-1, 0, 9] {
            raw.rank = rank;
            assert_eq!(output_shape(&raw), Err(StateError::NativeContract));
        }
        raw.rank = 2;
        for dimension in [-1, 0, 2] {
            raw.dimensions[1] = dimension;
            assert_eq!(output_shape(&raw), Err(StateError::NativeContract));
        }
    }

    // Test-only protobuf traversal reads ModelProto.graph (7), GraphProto.output
    // (12), ValueInfoProto.name (1). It never executes ONNX. Artifact hashes and
    // numerical CPU-reference/image qualification remain the parent's onsite gates.
    fn varint(bytes: &mut &[u8]) -> u64 {
        let mut value = 0;
        for shift in (0..70).step_by(7) {
            let (&byte, rest) = bytes.split_first().expect("truncated ONNX varint");
            *bytes = rest;
            assert!(shift < 63 || byte <= 1, "overflowing ONNX varint");
            value |= u64::from(byte & 127) << shift;
            if byte & 128 == 0 {
                return value;
            }
        }
        panic!("unterminated ONNX varint");
    }

    fn bytes_field(mut bytes: &[u8], number: u64) -> &[u8] {
        let mut found = None;
        while !bytes.is_empty() {
            let tag = varint(&mut bytes);
            assert_ne!(tag >> 3, 0, "invalid ONNX field number");
            if tag >> 3 == number {
                assert_eq!(tag & 7, 2, "unexpected ONNX field type");
            }
            match tag & 7 {
                0 => {
                    varint(&mut bytes);
                }
                1 => bytes = bytes.get(8..).expect("truncated ONNX fixed64"),
                2 => {
                    let len = usize::try_from(varint(&mut bytes)).expect("ONNX field too large");
                    let (field, rest) = bytes.split_at_checked(len).expect("truncated ONNX field");
                    bytes = rest;
                    if tag >> 3 == number {
                        assert!(found.replace(field).is_none(), "expected one ONNX field");
                    }
                }
                5 => bytes = bytes.get(4..).expect("truncated ONNX fixed32"),
                _ => panic!("unsupported ONNX wire type"),
            }
        }
        found.expect("required ONNX field missing")
    }

    #[test]
    #[ignore = "requires actual GPU, SEEON_TEST_FALL_ENGINE and SEEON_TEST_FALL_ONNX"]
    fn real_fall_gpu_execution_and_fail_closed_errors() {
        let engine = std::path::PathBuf::from(
            std::env::var_os("SEEON_TEST_FALL_ENGINE").expect("SEEON_TEST_FALL_ENGINE is required"),
        );
        let onnx =
            std::env::var_os("SEEON_TEST_FALL_ONNX").expect("SEEON_TEST_FALL_ONNX is required");
        let onnx = std::fs::read(onnx).expect("test ONNX must be readable");
        let name = CString::new(bytes_field(bytes_field(bytes_field(&onnx, 7), 12), 1))
            .expect("ONNX output name must contain no NUL");
        assert!(!name.as_bytes().is_empty(), "ONNX output name is required");
        let run = |model: &mut GpuModel, input_name: &CStr, window: &[f32], values: &mut [f32]| {
            model.run(
                TensorInput {
                    name: input_name,
                    values: window,
                    dimensions: &[1, 30, 56],
                },
                &mut [TensorOutput {
                    name: &name,
                    values,
                }],
            )
        };
        assert!(matches!(
            GpuModel::open(&engine, 999),
            Err(StateError::Unavailable)
        ));
        let mut model =
            GpuModel::open(&engine, 0).expect("actual TensorRT GPU admission must succeed");
        let mut window = [0.0; 30 * 56];
        let mut result = [f32::NAN];
        let mut seed = 4819_u32;
        for index in 0..14 {
            for value in &mut window {
                *value = match index {
                    0 => 0.0,
                    1 => 1.0,
                    _ => {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed >> 8) as f32 / 16_777_216.0
                    }
                };
            }
            result.fill(f32::NAN);
            let shapes = run(&mut model, c"window", &window, &mut result)
                .expect("actual synchronous GPU inference must succeed");
            assert_eq!(
                shapes,
                [TensorShape {
                    rank: 2,
                    dimensions: [1, 1, 0, 0, 0, 0, 0, 0]
                }]
            );
            assert!(result.iter().all(|value| value.is_finite()));
        }
        let metrics = model
            .metrics()
            .expect("native GPU metrics must be readable");
        assert_eq!(
            (metrics.attempted, metrics.succeeded, metrics.failed),
            (14, 14, 0)
        );
        assert_eq!(metrics.host_to_device_bytes, 14 * 30 * 56 * 4);
        assert_eq!(metrics.device_to_host_bytes, 14 * 4);
        assert_eq!(metrics.device, 0);
        assert!(metrics.elapsed_ns > 0);

        let mut poisoned = GpuModel::open(&engine, 0).expect("second native handle must open");
        result.fill(-987.0);
        assert_eq!(
            run(&mut poisoned, c"undeclared", &window, &mut result),
            Err(StateError::ExecutionFailed)
        );
        assert_eq!(result, [-987.0]);
        let failed = poisoned
            .metrics()
            .expect("poisoned native metrics must remain readable");
        assert_eq!(
            failed,
            GpuMetrics {
                attempted: 1,
                succeeded: 0,
                failed: 1,
                host_to_device_bytes: 0,
                device_to_host_bytes: 0,
                elapsed_ns: 0,
                device: 0,
            }
        );
        assert_eq!(
            run(&mut poisoned, c"window", &window, &mut result),
            Err(StateError::Poisoned)
        );
        assert_eq!(
            poisoned.metrics().unwrap(),
            failed,
            "refused retries must not call native run"
        );
    }
}

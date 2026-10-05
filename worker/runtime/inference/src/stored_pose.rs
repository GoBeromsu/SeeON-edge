//! Exact pure RGB conversion and ordered decoding around synchronous native inference.
//! Engine digest, model-role and deployed-batch admission remain the caller's duty
//! before source activation. Execution counters do not establish numeric equivalence.

use seeon_deepstream_native::{GpuMetrics, GpuModel, StateError, TensorInput, TensorOutput};
use seeon_worker::stored_pose::{
    OUTPUT_SHAPE, OUTPUT_VALUES, PersonBox, StoredPoseError, StoredPoseTensor, person_boxes,
};

/// Static, path-free failures; no pixels, tensor values or native diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoredPoseGpuError {
    Threshold,
    /// Pure preprocessing refused input without calling or poisoning the model.
    Input(StoredPoseError),
    /// Native execution, ABI or metrics failure; permanently poisons this owner.
    Native(StateError),
    /// Wrong resolved shape or rejected raw output; permanently poisons this owner.
    Output(StoredPoseError),
    Poisoned,
}

impl std::fmt::Display for StoredPoseGpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Threshold => {
                formatter.write_str("stored-pose threshold must be finite and in [0, 1]")
            }
            Self::Input(error) => write!(formatter, "stored-pose input refused: {error}"),
            Self::Native(error) => {
                write!(
                    formatter,
                    "stored-pose native failure; owner poisoned: {error}"
                )
            }
            Self::Output(error) => {
                write!(
                    formatter,
                    "stored-pose output refused; owner poisoned: {error}"
                )
            }
            Self::Poisoned => {
                formatter.write_str("stored-pose owner is unusable after a fatal model result")
            }
        }
    }
}
impl std::error::Error for StoredPoseGpuError {}

/// Exclusive, synchronous owner; inherits `!Send` and `!Sync` from `GpuModel`.
/// Holds one reusable core tensor and one fixed 17,100-f32 output allocation.
/// No engine opening, admission authentication, provider selection or fallback.
/// Dropping the owner closes the native model on its opening thread.
pub struct StoredPoseGpu {
    model: GpuModel,
    threshold: f64,
    tensor: StoredPoseTensor,
    output: Box<[f32; OUTPUT_VALUES]>,
    poisoned: bool,
}

impl StoredPoseGpu {
    /// Takes ownership of an already-open model and an explicit finite [0, 1] threshold.
    /// Invalid thresholds consume/drop the model without allocating frame buffers.
    /// The caller must have admitted the image-owned engine and selected its device.
    pub fn new(model: GpuModel, threshold: f64) -> Result<Self, StoredPoseGpuError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(StoredPoseGpuError::Threshold);
        }
        Ok(Self {
            model,
            threshold,
            tensor: StoredPoseTensor::default(),
            output: Box::new([0.0; OUTPUT_VALUES]),
            poisoned: false,
        })
    }

    /// Packed RGB8, using the core's exact preprocessing and original dimensions.
    /// Input refusal leaves the owner usable. Native/output failures latch it unusable;
    /// later inference is refused before preprocessing, even for otherwise bad input.
    /// Returns only completely decoded boxes, in unmodified core output order.
    pub fn infer(
        &mut self,
        rgb: &[u8],
        width: i64,
        height: i64,
    ) -> Result<Vec<PersonBox>, StoredPoseGpuError> {
        if self.poisoned {
            return Err(StoredPoseGpuError::Poisoned);
        }
        let input = self
            .tensor
            .preprocess(rgb, width, height)
            .map_err(StoredPoseGpuError::Input)?;
        let shapes = self
            .model
            .run(
                TensorInput {
                    name: c"images",
                    values: input,
                    dimensions: &[1, 3, 640, 640],
                },
                &mut [TensorOutput {
                    name: c"output0",
                    values: &mut self.output[..],
                }],
            )
            .map_err(|error| {
                self.poisoned = true;
                StoredPoseGpuError::Native(error)
            })?;
        // GpuModel publishes these dimensions only after checked native synchronization.
        if shapes.len() != 1 || shapes[0].rank != 3 || shapes[0].dimensions[..3] != [1, 300, 57] {
            self.poisoned = true;
            return Err(StoredPoseGpuError::Output(StoredPoseError::OutputShape));
        }
        person_boxes(
            &self.output[..],
            &OUTPUT_SHAPE,
            width,
            height,
            self.threshold,
        )
        .map_err(|error| {
            self.poisoned = true;
            StoredPoseGpuError::Output(error)
        })
    }

    /// Native cumulative counters remain readable after poisoning. A failed read
    /// also poisons this owner; a successful read never restores inference access.
    pub fn metrics(&mut self) -> Result<GpuMetrics, StoredPoseGpuError> {
        self.model.metrics().map_err(|error| {
            self.poisoned = true;
            StoredPoseGpuError::Native(error)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Read;
    use std::os::unix::{ffi::OsStrExt, fs::MetadataExt};
    use std::path::PathBuf;

    const MAX_RGB_FIXTURE_BYTES: usize = 12 * 1024 * 1024;

    fn fixture_path(variable: &str) -> PathBuf {
        let value = std::env::var_os(variable).unwrap_or_else(|| panic!("{variable} is required"));
        assert!(
            !value.as_bytes().iter().all(u8::is_ascii_whitespace),
            "{variable} must not be empty or blank"
        );
        assert!(!value.as_bytes().contains(&0), "{variable} contains a NUL");
        PathBuf::from(value)
    }

    // Test-only envelope, not a production file format. Parent hash-binds the
    // approved synthetic endpoint and engines and keeps fixtures immutable.
    fn rgb_fixture_dimensions(bytes: &[u8]) -> Result<(i64, i64), &'static str> {
        if !(16..=MAX_RGB_FIXTURE_BYTES).contains(&bytes.len()) {
            return Err("RGB fixture length is outside its bound");
        }
        if &bytes[..8] != b"SPRGB001" {
            return Err("RGB fixture magic is invalid");
        }
        let width = u32::from_le_bytes(bytes[8..12].try_into().expect("four-byte width"));
        let height = u32::from_le_bytes(bytes[12..16].try_into().expect("four-byte height"));
        if !(1..=4096).contains(&width) || !(1..=4096).contains(&height) {
            return Err("RGB fixture dimensions are outside their bound");
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|payload| payload.checked_add(16))
            .ok_or("RGB fixture dimensions overflow")?;
        if bytes.len() != expected {
            return Err("RGB fixture payload length is not exact");
        }
        Ok((i64::from(width), i64::from(height)))
    }

    fn rgb_fixture() -> (Vec<u8>, i64, i64) {
        let path = fixture_path("SEEON_TEST_STORED_POSE_RGB");
        let before = fs::symlink_metadata(&path)
            .unwrap_or_else(|_| panic!("RGB fixture metadata must be readable"));
        assert!(
            before.file_type().is_file(),
            "RGB fixture must be regular, not a symlink"
        );
        assert!(
            (16..=MAX_RGB_FIXTURE_BYTES as u64).contains(&before.len()),
            "RGB fixture file length is outside its bound"
        );
        let file = File::open(&path).unwrap_or_else(|_| panic!("RGB fixture must be readable"));
        let opened = file
            .metadata()
            .expect("opened RGB fixture metadata must be readable");
        assert!(
            opened.file_type().is_file(),
            "opened RGB fixture must be regular"
        );
        assert_eq!(
            (opened.dev(), opened.ino(), opened.len()),
            (before.dev(), before.ino(), before.len()),
            "RGB fixture must not change while opening"
        );
        let mut bytes = Vec::with_capacity(opened.len() as usize);
        (&file)
            .take((MAX_RGB_FIXTURE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .unwrap_or_else(|_| panic!("RGB fixture payload must be readable"));
        assert_eq!(
            bytes.len() as u64,
            opened.len(),
            "RGB fixture length changed"
        );
        let after = fs::symlink_metadata(&path)
            .unwrap_or_else(|_| panic!("RGB fixture metadata must remain readable"));
        assert!(
            after.file_type().is_file(),
            "RGB fixture must remain regular, not a symlink"
        );
        assert_eq!(
            (after.dev(), after.ino(), after.len()),
            (opened.dev(), opened.ino(), opened.len()),
            "RGB fixture must not change while reading"
        );
        let (width, height) = rgb_fixture_dimensions(&bytes).expect("invalid RGB fixture");
        assert_eq!(
            (width, height),
            (640, 360),
            "requires the admitted single-actor endpoint"
        );
        (bytes, width, height)
    }

    #[test]
    fn rgb_fixture_requires_exact_bounded_rgb_contract() {
        let mut valid = b"SPRGB001".to_vec();
        valid.extend_from_slice(&1_u32.to_le_bytes());
        valid.extend_from_slice(&1_u32.to_le_bytes());
        valid.extend_from_slice(&[12, 34, 56]);
        assert_eq!(rgb_fixture_dimensions(&valid), Ok((1, 1)));
        for length in [0, 7, 8, 12, 15, 16, 18] {
            assert!(rgb_fixture_dimensions(&valid[..length]).is_err());
        }
        let mut wrong_magic = valid.clone();
        wrong_magic[0] = 0;
        assert!(rgb_fixture_dimensions(&wrong_magic).is_err());
        for (width, height) in [
            (0_u32, 1_u32),
            (1, 0),
            (4097, 1),
            (1, 4097),
            (u32::MAX, u32::MAX),
        ] {
            let mut invalid = valid.clone();
            invalid[8..12].copy_from_slice(&width.to_le_bytes());
            invalid[12..16].copy_from_slice(&height.to_le_bytes());
            assert!(rgb_fixture_dimensions(&invalid).is_err());
        }
        for (width, height) in [(4096_u32, 1_u32), (1, 4096)] {
            let mut boundary = vec![0; 16 + 4096 * 3];
            boundary[..16].copy_from_slice(&valid[..16]);
            boundary[8..12].copy_from_slice(&width.to_le_bytes());
            boundary[12..16].copy_from_slice(&height.to_le_bytes());
            assert_eq!(
                rgb_fixture_dimensions(&boundary),
                Ok((i64::from(width), i64::from(height)))
            );
        }
        valid.push(0);
        assert!(rgb_fixture_dimensions(&valid).is_err());
        assert!(rgb_fixture_dimensions(&vec![0; MAX_RGB_FIXTURE_BYTES + 1]).is_err());
    }

    #[test]
    fn error_messages_are_static_and_path_free() {
        for (error, display, debug) in [
            (
                StoredPoseGpuError::Threshold,
                "stored-pose threshold must be finite and in [0, 1]",
                "Threshold",
            ),
            (
                StoredPoseGpuError::Input(StoredPoseError::ImageLength),
                "stored-pose input refused: RGB byte length does not match dimensions",
                "Input(ImageLength)",
            ),
            (
                StoredPoseGpuError::Native(StateError::ExecutionFailed),
                "stored-pose native failure; owner poisoned: GPU execution failed; model poisoned",
                "Native(ExecutionFailed)",
            ),
            (
                StoredPoseGpuError::Native(StateError::MetricsFailed),
                "stored-pose native failure; owner poisoned: GPU metrics read failed; model poisoned",
                "Native(MetricsFailed)",
            ),
            (
                StoredPoseGpuError::Output(StoredPoseError::OutputShape),
                "stored-pose output refused; owner poisoned: output0 must have shape [1, 300, 57] and matching length",
                "Output(OutputShape)",
            ),
            (
                StoredPoseGpuError::Output(StoredPoseError::NonFiniteOutput),
                "stored-pose output refused; owner poisoned: output0 must contain only finite float32 values",
                "Output(NonFiniteOutput)",
            ),
            (
                StoredPoseGpuError::Poisoned,
                "stored-pose owner is unusable after a fatal model result",
                "Poisoned",
            ),
        ] {
            assert_eq!(error.to_string(), display);
            assert_eq!(format!("{error:?}"), debug);
        }
    }

    #[test]
    #[ignore = "GPU0; requires SEEON_TEST_STORED_POSE_ENGINE and SEEON_TEST_STORED_POSE_RGB"]
    fn real_stored_pose_gpu_rejects_input_then_returns_one_person_with_exact_metrics() {
        let (bytes, width, height) = rgb_fixture();
        let model = GpuModel::open(&fixture_path("SEEON_TEST_STORED_POSE_ENGINE"), 0)
            .expect("actual stored-pose GPU model must open");
        let mut owner = StoredPoseGpu::new(model, 0.25).expect("explicit threshold must be valid");
        let before = owner.metrics().expect("native metrics must be readable");
        assert_eq!(before.device, 0);
        let tensor_allocation = owner.tensor.as_slice().as_ptr();
        let output_allocation = owner.output.as_ptr();
        for (rgb, width, height, error) in [
            (&[][..], 0, 1, StoredPoseError::Dimensions),
            (&[][..], 1, -1, StoredPoseError::Dimensions),
            (&[][..], i64::MAX, 2, StoredPoseError::DimensionsOverflow),
            (&[1, 2][..], 1, 1, StoredPoseError::ImageLength),
            (&[1, 2, 3, 4][..], 1, 1, StoredPoseError::ImageLength),
            (
                b"private-resident-pixels".as_slice(),
                1,
                1,
                StoredPoseError::ImageLength,
            ),
            (
                &bytes[16..bytes.len() - 1],
                width,
                height,
                StoredPoseError::ImageLength,
            ),
        ] {
            assert_eq!(
                owner.infer(rgb, width, height).err(),
                Some(StoredPoseGpuError::Input(error))
            );
            assert_eq!(
                owner.metrics().unwrap(),
                before,
                "input refusal must not call native inference"
            );
        }
        let boxes = owner
            .infer(&bytes[16..], width, height)
            .expect("actual admitted stored-pose inference must succeed after input refusal");
        assert_eq!(
            boxes.len(),
            1,
            "approved endpoint must yield exactly one person"
        );
        let [x1, y1, x2, y2, score] = boxes[0];
        assert!(
            boxes[0].iter().all(|value| value.is_finite()),
            "box must be finite"
        );
        assert!(x1 >= 0.0 && y1 >= 0.0 && x2 <= width as f64 && y2 <= height as f64);
        assert!(x1 < x2 && y1 < y2, "box must have positive ordered extent");
        assert!(score >= 0.25, "person must meet the explicit threshold");
        assert_eq!(owner.tensor.as_slice().as_ptr(), tensor_allocation);
        assert_eq!(owner.output.as_ptr(), output_allocation);
        let after = owner.metrics().expect("success metrics must be readable");
        assert_eq!(after.device, 0);
        assert_eq!(after.attempted.checked_sub(before.attempted), Some(1));
        assert_eq!(after.succeeded.checked_sub(before.succeeded), Some(1));
        assert_eq!(after.failed.checked_sub(before.failed), Some(0));
        assert_eq!(
            after
                .host_to_device_bytes
                .checked_sub(before.host_to_device_bytes),
            Some(3 * 640 * 640 * 4)
        );
        assert_eq!(
            after
                .device_to_host_bytes
                .checked_sub(before.device_to_host_bytes),
            Some(300 * 57 * 4)
        );
        assert!(
            after.elapsed_ns > before.elapsed_ns,
            "successful execution must take positive time"
        );
    }

    #[test]
    #[ignore = "GPU0; requires SEEON_TEST_STORED_POSE_ENGINE"]
    fn real_stored_pose_gpu_refuses_invalid_thresholds_without_private_errors() {
        let engine = fixture_path("SEEON_TEST_STORED_POSE_ENGINE");
        for threshold in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -f64::EPSILON,
            1.0 + f64::EPSILON,
        ] {
            let model = GpuModel::open(&engine, 0).expect("actual stored-pose GPU model must open");
            let error = match StoredPoseGpu::new(model, threshold) {
                Err(error) => error,
                Ok(_) => panic!("invalid threshold must be refused"),
            };
            assert_eq!(error, StoredPoseGpuError::Threshold);
            assert_eq!(
                error.to_string(),
                "stored-pose threshold must be finite and in [0, 1]"
            );
            assert_eq!(format!("{error:?}"), "Threshold");
        }
        for threshold in [0.0, 1.0] {
            let model = GpuModel::open(&engine, 0).expect("actual stored-pose GPU model must open");
            let mut owner =
                StoredPoseGpu::new(model, threshold).expect("inclusive boundary is valid");
            let metrics = owner
                .metrics()
                .expect("boundary owner metrics must be readable");
            assert_eq!(
                (metrics.attempted, metrics.succeeded, metrics.failed),
                (0, 0, 0)
            );
            assert_eq!(metrics.device, 0);
        }
    }

    #[test]
    #[ignore = "GPU0; requires SEEON_TEST_FALL_ENGINE and SEEON_TEST_STORED_POSE_RGB"]
    fn real_wrong_role_gpu_failure_latches_and_keeps_metrics_readable() {
        let (bytes, width, height) = rgb_fixture();
        let model = GpuModel::open(&fixture_path("SEEON_TEST_FALL_ENGINE"), 0)
            .expect("actual wrong-role fall engine must open; missing hardware is not a pass");
        let mut owner = StoredPoseGpu::new(model, 0.25).expect("explicit threshold must be valid");
        let before = owner
            .metrics()
            .expect("initial native metrics must be readable");
        assert_eq!(before.device, 0);
        let error = match owner.infer(&bytes[16..], width, height) {
            Err(error) => error,
            Ok(_) => panic!("wrong role must fail"),
        };
        assert_eq!(
            error,
            StoredPoseGpuError::Native(StateError::ExecutionFailed)
        );
        assert_eq!(format!("{error:?}"), "Native(ExecutionFailed)");
        assert_eq!(
            error.to_string(),
            "stored-pose native failure; owner poisoned: GPU execution failed; model poisoned"
        );
        let failed = owner
            .metrics()
            .expect("failure metrics must remain readable");
        assert_eq!(failed.device, 0);
        assert_eq!(failed.attempted.checked_sub(before.attempted), Some(1));
        assert_eq!(failed.succeeded.checked_sub(before.succeeded), Some(0));
        assert_eq!(failed.failed.checked_sub(before.failed), Some(1));
        assert_eq!(failed.host_to_device_bytes, before.host_to_device_bytes);
        assert_eq!(failed.device_to_host_bytes, before.device_to_host_bytes);
        assert_eq!(failed.elapsed_ns, before.elapsed_ns);
        assert_eq!(
            owner.infer(&bytes[16..], width, height).err(),
            Some(StoredPoseGpuError::Poisoned)
        );
        assert_eq!(
            owner.infer(&[], 0, 0).err(),
            Some(StoredPoseGpuError::Poisoned)
        );
        assert_eq!(
            owner.metrics().unwrap(),
            failed,
            "refused retries must never run a fallback"
        );
    }
}

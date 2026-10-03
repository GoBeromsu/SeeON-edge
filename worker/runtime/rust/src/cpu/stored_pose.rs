//! Exact pure RGB conversion and ordered person decoding around CPU inference.
//! Model admission and opening remain the caller's duty.

use seeon_onnxruntime_native::{ErrorKind, Input, Model, Output};
use seeon_worker::stored_pose::{
    NET_SIZE, OUTPUT_SHAPE, OUTPUT_VALUES, PersonBox, StoredPoseError, StoredPoseTensor,
    person_boxes,
};

/// Static, path-free failures; no pixels, tensor values or native diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoredPoseCpuError {
    Threshold,
    /// Pure preprocessing refused input without calling or poisoning the model.
    Input(StoredPoseError),
    /// Any native failure permanently poisons this owner.
    Native(ErrorKind),
    /// Wrong resolved shape or rejected raw output permanently poisons this owner.
    Output(StoredPoseError),
    Poisoned,
}

impl std::fmt::Display for StoredPoseCpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Threshold => {
                formatter.write_str("stored-pose threshold must be finite and in [0, 1]")
            }
            Self::Input(error) => write!(formatter, "stored-pose input refused: {error}"),
            Self::Native(error) => {
                write!(
                    formatter,
                    "stored-pose native failure; owner poisoned: {error:?}"
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
impl std::error::Error for StoredPoseCpuError {}

/// Exclusive, synchronous owner; inherits `!Send` and `!Sync` from `Model`.
/// Holds one reusable core tensor and one fixed output allocation.
/// No asset opening, provider selection, accelerator evidence or fallback.
pub struct StoredPoseCpu {
    model: Model,
    threshold: f64,
    tensor: StoredPoseTensor,
    output: Box<[f32; OUTPUT_VALUES]>,
    poisoned: bool,
}

impl StoredPoseCpu {
    /// Takes ownership of an already-open model and an explicit finite [0, 1] threshold.
    /// Invalid thresholds consume/drop the model without allocating frame buffers.
    pub fn new(model: Model, threshold: f64) -> Result<Self, StoredPoseCpuError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(StoredPoseCpuError::Threshold);
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
    ) -> Result<Vec<PersonBox>, StoredPoseCpuError> {
        if self.poisoned {
            return Err(StoredPoseCpuError::Poisoned);
        }
        let input = self
            .tensor
            .preprocess(rgb, width, height)
            .map_err(StoredPoseCpuError::Input)?;
        let result = self
            .model
            .run(
                Input {
                    name: c"images",
                    data: input,
                    shape: &[1, 3, NET_SIZE as i64, NET_SIZE as i64],
                },
                &mut [Output {
                    name: c"output0",
                    data: &mut self.output[..],
                }],
            )
            .map_err(|error| {
                self.poisoned = true;
                StoredPoseCpuError::Native(error.kind())
            })?;
        let shapes = result.shapes();
        if shapes.len() != 1
            || shapes[0].dimensions() != OUTPUT_SHAPE.map(|value| value as i64)
            || shapes[0].elements() != OUTPUT_VALUES
        {
            self.poisoned = true;
            return Err(StoredPoseCpuError::Output(StoredPoseError::OutputShape));
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
            StoredPoseCpuError::Output(error)
        })
    }
}

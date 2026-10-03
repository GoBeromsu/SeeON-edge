//! Synchronous bed segmentation on an already-open CPU model.
//! Returns borrowed raw outputs and the core letterbox; mask decoding and
//! polygon policy remain the caller's duty.

use seeon_onnxruntime_native::{ErrorKind, Input, Model, Output};
use seeon_worker::bed_input::{BedInputError, BedInputTensor, Letterbox, NET_SIZE};

pub const DETECTIONS_SHAPE: [usize; 3] = [1, 300, 38];
pub const DETECTIONS_VALUES: usize =
    DETECTIONS_SHAPE[0] * DETECTIONS_SHAPE[1] * DETECTIONS_SHAPE[2];
pub const PROTOS_SHAPE: [usize; 4] = [1, 32, 320, 320];
pub const PROTOS_VALUES: usize =
    PROTOS_SHAPE[0] * PROTOS_SHAPE[1] * PROTOS_SHAPE[2] * PROTOS_SHAPE[3];

/// Static, path-free failures; no pixels, tensor values or native diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedCpuError {
    /// Pure preprocessing refused input without calling or poisoning the model.
    Input(BedInputError),
    /// Any native failure permanently poisons this owner.
    Native(ErrorKind),
    /// Wrong resolved shapes or non-finite values permanently poison this owner.
    Output,
    Poisoned,
}

impl std::fmt::Display for BedCpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(error) => write!(formatter, "bed input refused: {error}"),
            Self::Native(error) => {
                write!(formatter, "bed native failure; owner poisoned: {error:?}")
            }
            Self::Output => formatter.write_str(
                "bed output refused; owner poisoned: outputs must have shapes [1, 300, 38] and [1, 32, 320, 320] with finite values",
            ),
            Self::Poisoned => formatter.write_str("bed owner is unusable after a fatal model result"),
        }
    }
}
impl std::error::Error for BedCpuError {}

/// Borrowed until the next call on the owner.
#[derive(Debug)]
pub struct BedCpuRaw<'a> {
    pub detections: &'a [f32],
    pub protos: &'a [f32],
    pub letterbox: Letterbox,
}

/// Exclusive, synchronous owner; inherits `!Send` and `!Sync` from `Model`.
/// Holds one reusable input tensor and fixed output allocations.
/// No asset opening, provider selection, accelerator evidence or fallback.
pub struct BedCpu {
    model: Model,
    tensor: BedInputTensor,
    detections: Box<[f32; DETECTIONS_VALUES]>,
    protos: Vec<f32>,
    poisoned: bool,
}

impl BedCpu {
    /// Takes ownership of a model opened on the calling thread.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            tensor: BedInputTensor::default(),
            detections: Box::new([0.0; DETECTIONS_VALUES]),
            protos: vec![0.0; PROTOS_VALUES],
            poisoned: false,
        }
    }

    /// Packed RGB8 through the core letterbox. Input refusal leaves the owner
    /// usable; native/output failures latch it unusable before later preprocessing.
    pub fn infer(
        &mut self,
        rgb: &[u8],
        width: i64,
        height: i64,
    ) -> Result<BedCpuRaw<'_>, BedCpuError> {
        if self.poisoned {
            return Err(BedCpuError::Poisoned);
        }
        let (input, letterbox) = self
            .tensor
            .preprocess(rgb, width, height)
            .map_err(BedCpuError::Input)?;
        let result = self
            .model
            .run(
                Input {
                    name: c"images",
                    data: input,
                    shape: &[1, 3, NET_SIZE as i64, NET_SIZE as i64],
                },
                &mut [
                    Output {
                        name: c"output0",
                        data: &mut self.detections[..],
                    },
                    Output {
                        name: c"output1",
                        data: &mut self.protos[..],
                    },
                ],
            )
            .map_err(|error| {
                self.poisoned = true;
                BedCpuError::Native(error.kind())
            })?;
        let shapes = result.shapes();
        if shapes.len() != 2
            || shapes[0].dimensions() != DETECTIONS_SHAPE.map(|value| value as i64)
            || shapes[0].elements() != DETECTIONS_VALUES
            || shapes[1].dimensions() != PROTOS_SHAPE.map(|value| value as i64)
            || shapes[1].elements() != PROTOS_VALUES
            || !self.detections.iter().all(|v| v.is_finite())
            || !self.protos.iter().all(|v| v.is_finite())
        {
            self.poisoned = true;
            return Err(BedCpuError::Output);
        }
        Ok(BedCpuRaw {
            detections: &self.detections[..],
            protos: &self.protos,
            letterbox,
        })
    }

    /// The preprocessed tensor of the latest accepted input.
    pub fn input(&self) -> &[f32] {
        self.tensor.as_slice()
    }
}

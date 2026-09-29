//! Synchronous bed segmentation on an explicitly supplied FP32 engine.
//! Returns raw detections and prototypes with the letterbox and a per-call
//! accelerator receipt; mask decoding and polygon policy remain the caller's duty.

use crate::evidence::{AcceleratorEvidence, EngineDigest, EvidenceError, Precision};
use seeon_deepstream_native::{GpuMetrics, GpuModel, StateError, TensorInput, TensorOutput};
use seeon_worker::bed_input::{BedInputError, BedInputTensor, Letterbox, NET_SIZE};

pub const DETECTIONS_SHAPE: [usize; 3] = [1, 300, 38];
pub const DETECTIONS_VALUES: usize = 300 * 38;
pub const PROTOS_SHAPE: [usize; 4] = [1, 32, 320, 320];
pub const PROTOS_VALUES: usize = 32 * 320 * 320;

/// Static, path-free failures; no pixels, tensor values or native diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedGpuError {
    /// Pure preprocessing refused input without calling or poisoning the model.
    Input(BedInputError),
    /// Native execution, ABI or metrics failure; permanently poisons this owner.
    Native(StateError),
    /// Wrong resolved shapes or non-finite values; permanently poisons this owner.
    Output,
    /// Incomplete accelerator receipt; permanently poisons this owner.
    Evidence(EvidenceError),
    Poisoned,
}

impl std::fmt::Display for BedGpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(error) => write!(formatter, "bed input refused: {error}"),
            Self::Native(error) => {
                write!(formatter, "bed native failure; owner poisoned: {error}")
            }
            Self::Output => formatter.write_str(
                "bed output refused; owner poisoned: outputs must have shapes [1, 300, 38] and [1, 32, 320, 320] with finite values",
            ),
            Self::Evidence(error) => {
                write!(formatter, "bed evidence refused; owner poisoned: {error}")
            }
            Self::Poisoned => formatter.write_str("bed owner is unusable after a fatal model result"),
        }
    }
}
impl std::error::Error for BedGpuError {}

/// Borrowed until the next call on the owner.
#[derive(Debug)]
pub struct BedRaw<'a> {
    pub detections: &'a [f32],
    pub protos: &'a [f32],
    pub letterbox: Letterbox,
    pub evidence: AcceleratorEvidence,
}

/// Exclusive, synchronous owner; inherits `!Send` and `!Sync` from `GpuModel`.
/// Holds one reusable input tensor and fixed output allocations.
/// No engine opening, admission authentication, provider selection or fallback.
pub struct BedGpu {
    model: GpuModel,
    device_ordinal: i32,
    engine_sha256: EngineDigest,
    tensor: BedInputTensor,
    detections: Box<[f32; DETECTIONS_VALUES]>,
    protos: Vec<f32>,
    poisoned: bool,
}

impl BedGpu {
    /// The caller opened `model` on `device_ordinal` from the engine whose bytes
    /// hash to `engine_sha256`; both are bound into every receipt.
    pub fn new(model: GpuModel, device_ordinal: i32, engine_sha256: EngineDigest) -> Self {
        Self {
            model,
            device_ordinal,
            engine_sha256,
            tensor: BedInputTensor::default(),
            detections: Box::new([0.0; DETECTIONS_VALUES]),
            protos: vec![0.0; PROTOS_VALUES],
            poisoned: false,
        }
    }

    /// Packed RGB8 through the core letterbox. Input refusal leaves the owner
    /// usable; every later failure latches it unusable, including a bad receipt.
    pub fn infer(
        &mut self,
        rgb: &[u8],
        width: i64,
        height: i64,
    ) -> Result<BedRaw<'_>, BedGpuError> {
        if self.poisoned {
            return Err(BedGpuError::Poisoned);
        }
        let (input, letterbox) = self
            .tensor
            .preprocess(rgb, width, height)
            .map_err(BedGpuError::Input)?;
        let before = self.model.metrics().map_err(|error| {
            self.poisoned = true;
            BedGpuError::Native(error)
        })?;
        let shapes = self
            .model
            .run(
                TensorInput {
                    name: c"images",
                    values: input,
                    dimensions: &[1, 3, NET_SIZE as i32, NET_SIZE as i32],
                },
                &mut [
                    TensorOutput {
                        name: c"output0",
                        values: &mut self.detections[..],
                    },
                    TensorOutput {
                        name: c"output1",
                        values: &mut self.protos[..],
                    },
                ],
            )
            .map_err(|error| {
                self.poisoned = true;
                BedGpuError::Native(error)
            })?;
        if shapes.len() != 2
            || shapes[0].rank != 3
            || shapes[0].dimensions[..3] != [1, 300, 38]
            || shapes[1].rank != 4
            || shapes[1].dimensions[..4] != [1, 32, 320, 320]
            || !self.detections.iter().all(|v| v.is_finite())
            || !self.protos.iter().all(|v| v.is_finite())
        {
            self.poisoned = true;
            return Err(BedGpuError::Output);
        }
        let after = self.metrics()?;
        let evidence = AcceleratorEvidence::from_delta(
            &before,
            &after,
            self.device_ordinal,
            self.engine_sha256,
            Precision::Fp32,
        )
        .map_err(|error| {
            self.poisoned = true;
            BedGpuError::Evidence(error)
        })?;
        Ok(BedRaw {
            detections: &self.detections[..],
            protos: &self.protos,
            letterbox,
            evidence,
        })
    }

    /// The preprocessed tensor of the latest accepted input.
    pub fn input(&self) -> &[f32] {
        self.tensor.as_slice()
    }

    /// A failed read poisons this owner; a successful read never restores access.
    pub fn metrics(&mut self) -> Result<GpuMetrics, BedGpuError> {
        self.model.metrics().map_err(|error| {
            self.poisoned = true;
            BedGpuError::Native(error)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_static_and_path_free() {
        for (error, display, debug) in [
            (
                BedGpuError::Input(BedInputError::ImageLength),
                "bed input refused: RGB byte length does not match dimensions",
                "Input(ImageLength)",
            ),
            (
                BedGpuError::Native(StateError::ExecutionFailed),
                "bed native failure; owner poisoned: GPU execution failed; model poisoned",
                "Native(ExecutionFailed)",
            ),
            (
                BedGpuError::Output,
                "bed output refused; owner poisoned: outputs must have shapes [1, 300, 38] and [1, 32, 320, 320] with finite values",
                "Output",
            ),
            (
                BedGpuError::Evidence(EvidenceError::DeviceMismatch),
                "bed evidence refused; owner poisoned: accelerator evidence names another device",
                "Evidence(DeviceMismatch)",
            ),
            (
                BedGpuError::Poisoned,
                "bed owner is unusable after a fatal model result",
                "Poisoned",
            ),
        ] {
            assert_eq!(error.to_string(), display);
            assert_eq!(format!("{error:?}"), debug);
        }
    }
}

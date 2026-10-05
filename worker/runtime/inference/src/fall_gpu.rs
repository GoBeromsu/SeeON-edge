//! Synchronous pose-bbox56 GRU scoring on an explicitly supplied FP32 engine.
//! Returns the raw logit with a per-call accelerator receipt; calibration,
//! policy and event emission remain the caller's duty.

use crate::evidence::{AcceleratorEvidence, EngineDigest, EvidenceError, Precision};
use seeon_deepstream_native::{GpuMetrics, GpuModel, StateError, TensorInput, TensorOutput};
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, POSE_BBOX56_DIM, PoseBbox56Row};

pub const WINDOW_VALUES: usize = FALL_WINDOW_FRAMES * POSE_BBOX56_DIM;

/// Static, path-free failures; no window values or native diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallGpuError {
    /// Window refused without calling or poisoning the model.
    Window,
    /// Native execution, ABI or metrics failure; permanently poisons this owner.
    Native(StateError),
    /// Wrong resolved shape or non-finite logit; permanently poisons this owner.
    Output,
    /// Incomplete accelerator receipt; permanently poisons this owner.
    Evidence(EvidenceError),
    Poisoned,
}

impl std::fmt::Display for FallGpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Window => {
                formatter.write_str("fall window must hold exactly 30 finite 56-value rows")
            }
            Self::Native(error) => {
                write!(formatter, "fall native failure; owner poisoned: {error}")
            }
            Self::Output => formatter
                .write_str("fall output refused; owner poisoned: logit must be one finite value"),
            Self::Evidence(error) => {
                write!(formatter, "fall evidence refused; owner poisoned: {error}")
            }
            Self::Poisoned => {
                formatter.write_str("fall owner is unusable after a fatal model result")
            }
        }
    }
}
impl std::error::Error for FallGpuError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FallScore {
    pub logit: f32,
    pub evidence: AcceleratorEvidence,
}

/// Exclusive, synchronous owner; inherits `!Send` and `!Sync` from `GpuModel`.
/// No engine opening, admission authentication, provider selection or fallback.
pub struct FallGpu {
    model: GpuModel,
    device_ordinal: i32,
    engine_sha256: EngineDigest,
    window: Box<[f32; WINDOW_VALUES]>,
    output: [f32; 1],
    poisoned: bool,
}

impl FallGpu {
    /// The caller opened `model` on `device_ordinal` from the engine whose bytes
    /// hash to `engine_sha256`; both are bound into every receipt.
    pub fn new(model: GpuModel, device_ordinal: i32, engine_sha256: EngineDigest) -> Self {
        Self {
            model,
            device_ordinal,
            engine_sha256,
            window: Box::new([0.0; WINDOW_VALUES]),
            output: [0.0],
            poisoned: false,
        }
    }

    /// Oldest-first rows. Window refusal leaves the owner usable; every later
    /// failure latches it unusable, including an incomplete receipt.
    pub fn score(&mut self, rows: &[PoseBbox56Row]) -> Result<FallScore, FallGpuError> {
        if self.poisoned {
            return Err(FallGpuError::Poisoned);
        }
        if rows.len() != FALL_WINDOW_FRAMES || rows.iter().flatten().any(|v| !v.is_finite()) {
            return Err(FallGpuError::Window);
        }
        for (target, row) in self.window.chunks_exact_mut(POSE_BBOX56_DIM).zip(rows) {
            target.copy_from_slice(row);
        }
        let before = self.metrics()?;
        let shapes = self
            .model
            .run(
                TensorInput {
                    name: c"window",
                    values: &self.window[..],
                    dimensions: &[1, FALL_WINDOW_FRAMES as i32, POSE_BBOX56_DIM as i32],
                },
                &mut [TensorOutput {
                    name: c"84",
                    values: &mut self.output,
                }],
            )
            .map_err(|error| {
                self.poisoned = true;
                FallGpuError::Native(error)
            })?;
        if shapes.len() != 1
            || shapes[0].rank != 2
            || shapes[0].dimensions[..2] != [1, 1]
            || !self.output[0].is_finite()
        {
            self.poisoned = true;
            return Err(FallGpuError::Output);
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
            FallGpuError::Evidence(error)
        })?;
        Ok(FallScore {
            logit: self.output[0],
            evidence,
        })
    }

    /// A failed read poisons this owner; a successful read never restores access.
    pub fn metrics(&mut self) -> Result<GpuMetrics, FallGpuError> {
        self.model.metrics().map_err(|error| {
            self.poisoned = true;
            FallGpuError::Native(error)
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
                FallGpuError::Window,
                "fall window must hold exactly 30 finite 56-value rows",
                "Window",
            ),
            (
                FallGpuError::Native(StateError::ExecutionFailed),
                "fall native failure; owner poisoned: GPU execution failed; model poisoned",
                "Native(ExecutionFailed)",
            ),
            (
                FallGpuError::Output,
                "fall output refused; owner poisoned: logit must be one finite value",
                "Output",
            ),
            (
                FallGpuError::Evidence(EvidenceError::NoDeviceToHost),
                "fall evidence refused; owner poisoned: accelerator evidence shows no device-to-host copy",
                "Evidence(NoDeviceToHost)",
            ),
            (
                FallGpuError::Poisoned,
                "fall owner is unusable after a fatal model result",
                "Poisoned",
            ),
        ] {
            assert_eq!(error.to_string(), display);
            assert_eq!(format!("{error:?}"), debug);
        }
    }
}

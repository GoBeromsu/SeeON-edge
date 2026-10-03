//! Synchronous pose-bbox56 GRU scoring on an already-open CPU model.
//! Returns the raw logit; calibration, policy and event emission remain the caller's duty.

use seeon_onnxruntime_native::{ErrorKind, Input, Model, Output};
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, POSE_BBOX56_DIM, PoseBbox56Row};

pub const WINDOW_VALUES: usize = FALL_WINDOW_FRAMES * POSE_BBOX56_DIM;

/// Static, path-free failures; no window values or native diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallCpuError {
    /// Window refused without calling or poisoning the model.
    Window,
    /// Any native failure permanently poisons this owner.
    Native(ErrorKind),
    /// Wrong resolved shape or non-finite logit permanently poisons this owner.
    Output,
    Poisoned,
}

impl std::fmt::Display for FallCpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Window => {
                formatter.write_str("fall window must hold exactly 30 finite 56-value rows")
            }
            Self::Native(error) => {
                write!(formatter, "fall native failure; owner poisoned: {error:?}")
            }
            Self::Output => formatter
                .write_str("fall output refused; owner poisoned: logit must be one finite value"),
            Self::Poisoned => {
                formatter.write_str("fall owner is unusable after a fatal model result")
            }
        }
    }
}
impl std::error::Error for FallCpuError {}

/// Exclusive, synchronous owner; inherits `!Send` and `!Sync` from `Model`.
/// No asset opening, provider selection, accelerator evidence or fallback.
pub struct FallCpu {
    model: Model,
    window: Box<[f32; WINDOW_VALUES]>,
    output: [f32; 1],
    poisoned: bool,
}

impl FallCpu {
    /// Takes ownership of a model opened on the calling thread.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            window: Box::new([0.0; WINDOW_VALUES]),
            output: [0.0],
            poisoned: false,
        }
    }

    /// Oldest-first rows. Window refusal leaves the owner usable; native and
    /// output failures latch it unusable before any subsequent window validation.
    pub fn score(&mut self, rows: &[PoseBbox56Row]) -> Result<f32, FallCpuError> {
        if self.poisoned {
            return Err(FallCpuError::Poisoned);
        }
        if rows.len() != FALL_WINDOW_FRAMES || rows.iter().flatten().any(|v| !v.is_finite()) {
            return Err(FallCpuError::Window);
        }
        for (target, row) in self.window.chunks_exact_mut(POSE_BBOX56_DIM).zip(rows) {
            target.copy_from_slice(row);
        }
        let result = self
            .model
            .run(
                Input {
                    name: c"window",
                    data: &self.window[..],
                    shape: &[1, FALL_WINDOW_FRAMES as i64, POSE_BBOX56_DIM as i64],
                },
                &mut [Output {
                    name: c"84",
                    data: &mut self.output,
                }],
            )
            .map_err(|error| {
                self.poisoned = true;
                FallCpuError::Native(error.kind())
            })?;
        let shapes = result.shapes();
        if shapes.len() != 1
            || shapes[0].dimensions() != [1, 1]
            || shapes[0].elements() != self.output.len()
            || !self.output[0].is_finite()
        {
            self.poisoned = true;
            return Err(FallCpuError::Output);
        }
        Ok(self.output[0])
    }
}

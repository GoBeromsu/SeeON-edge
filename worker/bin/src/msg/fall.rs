//! Explicit provider identity at the fall-owner boundary, not inferred from receipts.

use seeon_worker_runtime::cpu::fall::FallCpuError;
use seeon_worker_runtime::evidence::AcceleratorEvidence;
use seeon_worker_runtime::fall_gpu::{FallGpuError, FallScore as GpuScore};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FallScore {
    Cpu(f32),
    TensorRt(GpuScore),
}

impl FallScore {
    pub fn logit(&self) -> f32 {
        match self {
            Self::Cpu(logit) => *logit,
            Self::TensorRt(score) => score.logit,
        }
    }

    pub fn accelerator(&self) -> Option<&AcceleratorEvidence> {
        match self {
            Self::Cpu(_) => None,
            Self::TensorRt(score) => Some(&score.evidence),
        }
    }
}

impl From<GpuScore> for FallScore {
    fn from(score: GpuScore) -> Self {
        Self::TensorRt(score)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallInferenceError {
    Cpu(FallCpuError),
    TensorRt(FallGpuError),
}

impl From<FallCpuError> for FallInferenceError {
    fn from(error: FallCpuError) -> Self {
        Self::Cpu(error)
    }
}

impl From<FallGpuError> for FallInferenceError {
    fn from(error: FallGpuError) -> Self {
        Self::TensorRt(error)
    }
}

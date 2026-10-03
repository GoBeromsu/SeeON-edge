//! Provider-tagged image inference failures; input refusals remain distinguishable.

use seeon_worker_runtime::bed_gpu::BedGpuError;
use seeon_worker_runtime::cpu::bed::BedCpuError;
use seeon_worker_runtime::cpu::stored_pose::StoredPoseCpuError;
use seeon_worker_runtime::stored_pose::StoredPoseGpuError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedInferenceError {
    Cpu(BedCpuError),
    TensorRt(BedGpuError),
}

impl From<BedCpuError> for BedInferenceError {
    fn from(error: BedCpuError) -> Self {
        Self::Cpu(error)
    }
}

impl From<BedGpuError> for BedInferenceError {
    fn from(error: BedGpuError) -> Self {
        Self::TensorRt(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoredPoseInferenceError {
    Cpu(StoredPoseCpuError),
    TensorRt(StoredPoseGpuError),
}

impl From<StoredPoseCpuError> for StoredPoseInferenceError {
    fn from(error: StoredPoseCpuError) -> Self {
        Self::Cpu(error)
    }
}

impl From<StoredPoseGpuError> for StoredPoseInferenceError {
    fn from(error: StoredPoseGpuError) -> Self {
        Self::TensorRt(error)
    }
}

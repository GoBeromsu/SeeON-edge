//! Explicit composition inputs; admission and provider selection belong to callers.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;

use super::super::settings::BootPolicy;
use super::super::{AdmittedModels, ModelEngines};
use crate::gpu::owners;
use crate::inference::Owner;
use crate::inference::cpu::{self, CapturedModel};
use crate::msg::{BedRequest, FallRequest, FallResponse, StoredPoseRequest};

/// Captured immutable ONNX bytes and one image-owned runtime library path.
/// Spawning clones the Arcs, never reopens a model path or selects a fallback.
pub struct CpuModels {
    pub runtime_library: PathBuf,
    pub fall: Arc<[u8]>,
    pub bed: Arc<[u8]>,
    pub stored_pose: Arc<[u8]>,
}

pub enum ModelInputs<'a> {
    TensorRt(&'a ModelEngines),
    OnnxRuntimeCpu(&'a CpuModels),
}

impl<'a> From<&'a AdmittedModels> for ModelInputs<'a> {
    fn from(models: &'a AdmittedModels) -> Self {
        match models {
            AdmittedModels::TensorRt(engines) => Self::TensorRt(engines),
            AdmittedModels::OnnxRuntimeCpu(models) => Self::OnnxRuntimeCpu(models),
        }
    }
}

impl ModelInputs<'_> {
    pub(super) fn spawn_fall(
        &self,
        policy: BootPolicy,
        stop: Arc<AtomicBool>,
    ) -> io::Result<(Owner<FallRequest>, Receiver<FallResponse>)> {
        match self {
            Self::TensorRt(engines) => owners::spawn_fall(
                engines.fall.path.clone(),
                policy.device_ordinal,
                engines.fall.digest,
                stop,
            ),
            Self::OnnxRuntimeCpu(models) => cpu::spawn_fall(models.capture(&models.fall), stop),
        }
    }

    pub(super) fn spawn_bed(
        &self,
        policy: BootPolicy,
        stop: Arc<AtomicBool>,
    ) -> io::Result<Owner<BedRequest>> {
        match self {
            Self::TensorRt(engines) => owners::spawn_bed(
                engines.bed.path.clone(),
                policy.device_ordinal,
                engines.bed.digest,
                stop,
            ),
            Self::OnnxRuntimeCpu(models) => cpu::spawn_bed(models.capture(&models.bed), stop),
        }
    }

    pub(super) fn spawn_stored_pose(
        &self,
        policy: BootPolicy,
        stop: Arc<AtomicBool>,
    ) -> io::Result<Owner<StoredPoseRequest>> {
        match self {
            Self::TensorRt(engines) => owners::spawn_stored_pose(
                engines.stored_pose.path.clone(),
                policy.device_ordinal,
                engines.stored_pose.digest,
                policy.stored_pose_threshold,
                stop,
            ),
            Self::OnnxRuntimeCpu(models) => cpu::spawn_stored_pose(
                models.capture(&models.stored_pose),
                policy.stored_pose_threshold,
                stop,
            ),
        }
    }
}

impl CpuModels {
    fn capture(&self, onnx: &Arc<[u8]>) -> CapturedModel {
        CapturedModel {
            runtime_library: self.runtime_library.clone(),
            onnx: Arc::clone(onnx),
        }
    }
}

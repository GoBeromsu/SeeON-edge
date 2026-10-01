//! Classify accelerator faults before the fall policy can turn them into missing data.

use std::sync::atomic::{AtomicBool, Ordering};

use seeon_worker::episode::BusinessEvent;
use seeon_worker_runtime::fall_gpu::FallGpuError;

use crate::exit::Exit;
use crate::msg::FallResponse;
use crate::policy::fall::{DecisionUpdate, FallStage, FallStageError};

#[derive(Clone, Debug, PartialEq)]
pub enum FallResponseError {
    Accelerator(FallGpuError),
    Policy(FallStageError),
}

impl FallResponseError {
    pub const fn exit(&self) -> Exit {
        match self {
            Self::Accelerator(_) => Exit::FatalAccelerator,
            Self::Policy(_) => Exit::Runtime,
        }
    }
}

/// Classify before routing or policy mutation, including unknown/stale frames.
/// A refused input window is recoverable; fatal accelerator faults stop the run.
pub fn validate_fall_response(
    response: &FallResponse,
    stop: &AtomicBool,
) -> Result<(), FallResponseError> {
    match &response.score {
        Ok(_) | Err(FallGpuError::Window) => {}
        Err(
            error @ (FallGpuError::Native(_)
            | FallGpuError::Output
            | FallGpuError::Evidence(_)
            | FallGpuError::Poisoned),
        ) => {
            stop.store(true, Ordering::SeqCst);
            return Err(FallResponseError::Accelerator(*error));
        }
    }
    Ok(())
}

/// A refused input window is missing data; a poisoned accelerator ends the run.
/// Fatal responses never reach `FallStage::consume`, even when their frame is stale.
/// The required observer follows the receipt-only [`DecisionUpdate`] contract.
pub fn consume_fall_response(
    stage: &mut FallStage,
    response: FallResponse,
    stop: &AtomicBool,
    observer: &mut dyn FnMut(DecisionUpdate<'_>),
) -> Result<Vec<BusinessEvent>, FallResponseError> {
    validate_fall_response(&response, stop)?;
    stage
        .consume(response, observer)
        .map_err(FallResponseError::Policy)
}

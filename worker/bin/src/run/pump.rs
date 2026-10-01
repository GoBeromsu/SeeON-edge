//! Synchronous CPU policy composition; the root owns channels and drain deadlines.

use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::SyncSender;

use seeon_deepstream_native::FrameIdentity;
use seeon_worker_runtime::fall_gpu::FallScore;

use super::policy::{FallResponseError, validate_fall_response};
use crate::exit::Exit;
use crate::msg::{FallRequest, FallResponse, PosePacket};
use crate::policy::fall::{DecisionUpdate, FallStage, FallStageError};
use crate::policy::ingest::{IngestRefusal, ingest};

pub struct CameraPolicy {
    pub source_id: u32,
    pub stage: FallStage,
}

/// Synchronous, non-reentrant, receipt-only callbacks. The root supplies a real
/// sink. Events are delivered only through `decision`, never again via returns.
pub trait PolicySink {
    fn decision(&mut self, update: DecisionUpdate<'_>);
    fn score(&mut self, frame: FrameIdentity, track_id: u64, score: &FallScore);
}

#[derive(Clone, Debug, PartialEq)]
pub enum PumpError {
    DuplicateSource(u32),
    UnknownSource(u32),
    Ingest {
        source_id: u32,
        cause: IngestRefusal,
    },
    Response(FallResponseError),
}

impl PumpError {
    /// Duplicate roster entries are constructor configuration refusals (2).
    /// Routing/ingest/policy composition failures are runtime errors (1);
    /// fatal accelerator faults retain exit 4.
    pub const fn exit(&self) -> Exit {
        match self {
            Self::DuplicateSource(_) => Exit::Config,
            Self::UnknownSource(_) | Self::Ingest { .. } => Exit::Runtime,
            Self::Response(error) => error.exit(),
        }
    }
}

impl From<FallStageError> for PumpError {
    fn from(error: FallStageError) -> Self {
        Self::Response(FallResponseError::Policy(error))
    }
}

pub struct PolicyPump {
    cameras: Vec<CameraPolicy>,
}

impl PolicyPump {
    /// Keeps declared roster order for flush, independently of source ID order.
    /// An empty roster is a valid idle pump.
    pub fn new(cameras: Vec<CameraPolicy>) -> Result<Self, PumpError> {
        let mut sources = BTreeSet::new();
        for camera in &cameras {
            if !sources.insert(camera.source_id) {
                return Err(PumpError::DuplicateSource(camera.source_id));
            }
        }
        Ok(Self { cameras })
    }

    pub fn observe(
        &mut self,
        packet: &PosePacket,
        requests: &SyncSender<FallRequest>,
        sink: &mut dyn PolicySink,
    ) -> Result<(), PumpError> {
        let source_id = packet.frame.source_id;
        let stage = self.stage(source_id)?;
        let frame = ingest(packet).map_err(|cause| PumpError::Ingest { source_id, cause })?;
        stage.observe(&frame, requests, &mut |update| sink.decision(update))?;
        Ok(())
    }

    pub fn consume(
        &mut self,
        response: FallResponse,
        stop: &AtomicBool,
        sink: &mut dyn PolicySink,
    ) -> Result<(), PumpError> {
        validate_fall_response(&response, stop).map_err(PumpError::Response)?;
        let stage = self.stage(response.frame.source_id)?;
        if let Ok(score) = &response.score {
            sink.score(response.frame, response.track_id, score);
        }
        stage.consume(response, &mut |update| sink.decision(update))?;
        Ok(())
    }

    /// Completes real pending updates in roster order, stopping at the first
    /// failure without retracting any earlier successful receipt.
    pub fn flush(&mut self, sink: &mut dyn PolicySink) -> Result<(), PumpError> {
        for camera in &mut self.cameras {
            camera.stage.flush(&mut |update| sink.decision(update))?;
        }
        Ok(())
    }

    pub fn pending_scores(&self) -> usize {
        self.cameras
            .iter()
            .map(|camera| camera.stage.pending_scores())
            .sum()
    }

    fn stage(&mut self, source_id: u32) -> Result<&mut FallStage, PumpError> {
        self.cameras
            .iter_mut()
            .find(|camera| camera.source_id == source_id)
            .map(|camera| &mut camera.stage)
            .ok_or(PumpError::UnknownSource(source_id))
    }
}

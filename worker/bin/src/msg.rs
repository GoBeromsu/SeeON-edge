//! Channel messages and capacities of design §2.3. Every channel is a
//! `std::sync::mpsc::sync_channel` and every producer uses `try_send`.
//! Native media payloads cross the media channels unchanged. Frozen at the
//! end of stage 0; later stages ask the owner for any change.

use std::sync::mpsc::SyncSender;

use seeon_deepstream_native::FrameIdentity;
use seeon_worker::bed_input::Letterbox;
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, PoseBbox56Row};
use seeon_worker::stored_pose::PersonBox;
use seeon_worker_runtime::bed_gpu::BedGpuError;
use seeon_worker_runtime::evidence::AcceleratorEvidence;
use seeon_worker_runtime::fall_gpu::{FallGpuError, FallScore};
use seeon_worker_runtime::stored_pose::StoredPoseGpuError;

use crate::exit::Exit;

pub use seeon_deepstream_native::{PosePacket, PreviewPacket, RecordReceipt};

/// media → policy, per roster camera (`pose_tx` holds 4 × cameras).
pub const POSE_PER_CAMERA: usize = 4;
/// media → preview.
pub const PREVIEW_CAPACITY: usize = 8;
/// media → clip-publisher.
pub const RECORD_CAPACITY: usize = 32;
/// policy → gpu-fall.
pub const FALL_REQUEST_CAPACITY: usize = 64;
/// gpu-fall → policy.
pub const FALL_RESPONSE_CAPACITY: usize = 64;
/// policy → delivery.
pub const EMIT_CAPACITY: usize = 256;
/// Request queue of gpu-bed and gpu-stored-pose; full means 503.
pub const GPU_REQUEST_CAPACITY: usize = 2;
/// Queued analysis jobs; busy means 409 or 503.
pub const ANALYSIS_CAPACITY: usize = 1;
/// One reply per request, and one readiness report per GPU owner thread.
pub const ONESHOT_CAPACITY: usize = 1;

/// One 30-frame pose-bbox56 window for one track, scored by `FallGpu`.
pub struct FallRequest {
    pub frame: FrameIdentity,
    pub track_id: u64,
    pub window: Box<[PoseBbox56Row; FALL_WINDOW_FRAMES]>,
}

/// The score for the request with the same `frame` and `track_id`.
pub struct FallResponse {
    pub frame: FrameIdentity,
    pub track_id: u64,
    pub score: Result<FallScore, FallGpuError>,
}

/// One RGB frame for a request/oneshot GPU owner; the owner answers once on
/// `reply`.
pub struct FrameRequest<T> {
    pub rgb: Vec<u8>,
    pub width: i64,
    pub height: i64,
    pub reply: SyncSender<T>,
}

/// Owned copy of `BedRaw`, which borrows the owner's output buffers and so
/// cannot leave the gpu-bed thread.
pub struct BedOutput {
    pub detections: Vec<f32>,
    pub protos: Vec<f32>,
    pub letterbox: Letterbox,
    pub evidence: AcceleratorEvidence,
}

pub type BedRequest = FrameRequest<Result<BedOutput, BedGpuError>>;
pub type StoredPoseRequest = FrameRequest<Result<Vec<PersonBox>, StoredPoseGpuError>>;

/// A GPU owner is built inside its own thread and reports once whether it
/// opened; `Err` carries the exit the process takes.
pub type Readiness = Result<(), Exit>;

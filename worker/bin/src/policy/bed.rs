//! Bed request and response glue against the gpu-bed owner. The request
//! queue holds `GPU_REQUEST_CAPACITY` frames; a full queue is a typed
//! refusal, never a wait. The reply comes on a per-request one-shot channel.

use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::time::Duration;

use seeon_worker_runtime::bed_gpu::BedGpuError;

use crate::msg::{BedOutput, BedRequest, FrameRequest, ONESHOT_CAPACITY};
use crate::poll::poll_until;
use crate::seam::Clock;

pub type BedReply = Result<BedOutput, BedGpuError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BedRefusal {
    /// The gpu-bed queue already holds `GPU_REQUEST_CAPACITY` frames.
    Full,
    /// The gpu-bed owner is gone.
    Disconnected,
    /// No reply before the deadline.
    Timeout,
}

/// Queues one RGB frame for the gpu-bed owner without blocking.
pub fn submit(
    requests: &SyncSender<BedRequest>,
    rgb: Vec<u8>,
    width: i64,
    height: i64,
) -> Result<Receiver<BedReply>, BedRefusal> {
    let (reply, replies) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let request = FrameRequest {
        rgb,
        width,
        height,
        reply,
    };
    match requests.try_send(request) {
        Ok(()) => Ok(replies),
        Err(TrySendError::Full(_)) => Err(BedRefusal::Full),
        Err(TrySendError::Disconnected(_)) => Err(BedRefusal::Disconnected),
    }
}

/// Waits for the reply of one `submit` up to the absolute monotonic
/// `deadline`. A reply dropped unsent means the owner is gone.
pub fn receive(
    clock: &dyn Clock,
    deadline: Duration,
    replies: &Receiver<BedReply>,
) -> Result<BedReply, BedRefusal> {
    let mut received = None;
    poll_until(clock, deadline, "gpu-bed reply", || {
        match replies.try_recv() {
            Ok(reply) => received = Some(Ok(reply)),
            Err(TryRecvError::Disconnected) => received = Some(Err(BedRefusal::Disconnected)),
            Err(TryRecvError::Empty) => {}
        }
        received.is_some()
    })
    .map_err(|_| BedRefusal::Timeout)?;
    received.unwrap_or(Err(BedRefusal::Timeout))
}

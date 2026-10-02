//! Media-thread counters readable from other threads, ported from
//! `service_maker.py` `published_frames` and `perception_counters`
//! (~L243-262). They are in-process values; nothing here reaches the wire.
//! Cameras are roster positions, which equal `SourceConfig::source_id`.

use std::sync::Mutex;

use crate::exit::Exit;

use seeon_deepstream_native::{MediaResult, MediaState, MediaStatus};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CameraCounters {
    /// Last native video-link flag, retained without inferring frame admission.
    pub video_linked: u32,
    /// Python `_publish_sequence`: native `SourceStatus::frames`, monotonic
    /// and never consumed, so it survives the policy side draining `pose_rx`.
    pub published_frames: u64,
    /// Python `_objects_observed`: native `SourceStatus::objects`.
    pub objects_observed: u64,
    /// Python `_frames_without_pose_tensor`: native `SourceStatus::tensor_absent`.
    pub frames_without_pose_tensor: u64,
    /// Pose packets dropped because `pose_tx` was full or its receiver gone.
    pub handoff_dropped_frames: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Open returned an error without transferring a native owner. This is
    /// distinct from stop/finalization/close and must never fabricate them.
    pub open_refused: bool,
    /// Actual Rust owner termination cause. `None` is an ordinary explicit
    /// stop. This is not a native diagnostic and not a native-close fact.
    pub failure: Option<Exit>,
    /// `None` until the first successful `read_status`.
    pub state: Option<MediaState>,
    /// The last status read reported a native fatal.
    pub fatal: bool,
    pub callbacks_active: u32,
    pub records_reserved: u32,
    pub cameras: Vec<CameraCounters>,
    pub previews_dropped: u64,
    pub receipts_dropped: u64,
    /// Command replies the requester was no longer waiting for.
    pub replies_dropped: u64,
    /// Release entered the native stop/reap attempt, including on failure.
    pub finalization_started: bool,
    /// That attempt finished (possibly by unwinding), before waiting for root
    /// close permission. This does not prove downstream clip publication.
    pub finalization_complete: bool,
    /// Native stop returned OK during release.
    pub stopped: bool,
    /// Native close returned OK after a successful stop with no record slot
    /// reserved. `false` after release means the native handle was leaked on
    /// purpose; the process must exit nonzero rather than reopen media.
    pub closed: bool,
    /// Close is withheld unless stop succeeded, reap observed zero reserved
    /// slots, and root permission is valid before the shared deadline.
    /// `records_reserved` holds the last count read, not proof of finalization.
    pub close_withheld: bool,
}

pub struct Diagnostics {
    inner: Mutex<Snapshot>,
}

impl Diagnostics {
    pub fn new(cameras: usize) -> Self {
        let snapshot = Snapshot {
            cameras: vec![CameraCounters::default(); cameras],
            ..Snapshot::default()
        };
        Self {
            inner: Mutex::new(snapshot),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.lock().clone()
    }

    /// `None` for a camera outside the roster.
    pub fn published_frames(&self, camera: usize) -> Option<u64> {
        self.camera(camera)
            .map(|counters| counters.published_frames)
    }

    /// `(objects_observed, frames_without_pose_tensor)`, as Python returns it.
    pub fn perception_counters(&self, camera: usize) -> Option<(u64, u64)> {
        self.camera(camera).map(|counters| {
            (
                counters.objects_observed,
                counters.frames_without_pose_tensor,
            )
        })
    }

    pub fn handoff_dropped_frames(&self, camera: usize) -> Option<u64> {
        self.camera(camera)
            .map(|counters| counters.handoff_dropped_frames)
    }

    /// Copies the native counters of one status read. A roster shorter than
    /// the status grows; handoff drops are this side's own and are kept.
    ///
    /// Order contract: `status.sources[i]` is camera `i`. Native status lists
    /// sources in `MediaConfig::sources` order, which is the order `owner.rs`
    /// `run` enumerates, and `MediaOwner::open` refuses a configuration whose
    /// `source_id` differs from its position. `SourceStatus` carries no source
    /// id, so the position is the only link and is not re-checked here.
    pub(crate) fn record_status(&self, status: &MediaStatus) {
        let mut snapshot = self.lock();
        snapshot.state = Some(status.state);
        snapshot.fatal = status.result == MediaResult::Fatal;
        snapshot.callbacks_active = status.callbacks_active;
        snapshot.records_reserved = status.records_reserved;
        if snapshot.cameras.len() < status.sources.len() {
            snapshot
                .cameras
                .resize(status.sources.len(), CameraCounters::default());
        }
        for (source_id, (counters, source)) in
            snapshot.cameras.iter_mut().zip(&status.sources).enumerate()
        {
            let linked = counters.video_linked == 0 && source.video_linked == 1;
            let first_frame = counters.published_frames == 0 && source.frames > 0;
            let first_objects = counters.objects_observed == 0 && source.objects > 0;
            if linked || first_frame || first_objects {
                eprintln!(
                    "ml-worker: media observation source_id={source_id} video_linked={} frames={} objects={} tensor_absent={} overwritten={} dropped={} malformed={}",
                    source.video_linked,
                    source.frames,
                    source.objects,
                    source.tensor_absent,
                    source.overwritten,
                    source.dropped,
                    source.malformed
                );
            }
            counters.video_linked = source.video_linked;
            counters.published_frames = source.frames;
            counters.objects_observed = source.objects;
            counters.frames_without_pose_tensor = source.tensor_absent;
        }
    }

    /// One pose packet `pose_tx` did not take; the roster grows to `camera`.
    pub(crate) fn count_handoff_drop(&self, camera: usize) {
        let mut snapshot = self.lock();
        if snapshot.cameras.len() <= camera {
            snapshot
                .cameras
                .resize(camera + 1, CameraCounters::default());
        }
        snapshot.cameras[camera].handoff_dropped_frames += 1;
    }

    pub(crate) fn update(&self, change: impl FnOnce(&mut Snapshot)) {
        change(&mut self.lock());
    }

    fn camera(&self, camera: usize) -> Option<CameraCounters> {
        self.lock().cameras.get(camera).copied()
    }

    /// Counters stay readable after a panicking writer; each write is whole.
    fn lock(&self) -> std::sync::MutexGuard<'_, Snapshot> {
        match self.inner.lock() {
            Ok(snapshot) => snapshot,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

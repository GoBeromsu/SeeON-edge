//! One camera's retained writer and the pump fields capture mutates.
//!
//! Track history is not cleared on reconnect. A new observation is validated
//! before that history changes. Sequence is consumed before the night-window
//! callback, and it stops after `u64::MAX` rather than wrapping.

use std::path::Path;
use std::time::Duration;

use seeon_deepstream_native::MEDIA_MAX_OBJECTS;

use crate::policy::ingest::{Frame, ObservedPose};
use crate::relay::cameras::RuntimeCamera;
use crate::trace_out::{
    BedPolygon, Lifecycle, ReplayRow, ReplayTraceWriter, ReplayTrack, SourceEvent, TraceError,
};

use super::SOURCE_LOSS;
use super::geometry::{RowMeta, base_row, validated_track};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

pub(super) struct CameraTrace {
    camera_id: String,
    writer: ReplayTraceWriter,
    epoch: Option<(u64, u64)>,
    /// Previous-state insertion order, matching Python dict order.
    tracks: Vec<ReplayTrack>,
    source_lost: bool,
    dims: Option<(u64, u64)>,
    last_pts_ns: u64,
    /// Next sequence, or `None` after `u64::MAX` was consumed.
    seq: Option<u64>,
    failure_logged: bool,
    failures: u64,
    /// Last `processed` time. `None` before one.
    waiting_since: Option<Duration>,
}

impl CameraTrace {
    pub(super) fn open(directory: &Path, camera: &RuntimeCamera) -> Result<Self, TraceError> {
        Ok(Self {
            writer: ReplayTraceWriter::new(
                directory,
                &camera.camera_id,
                crate::trace_out::DEFAULT_MAX_BYTES,
                crate::trace_out::DEFAULT_ROTATION_COUNT,
            )?,
            camera_id: camera.camera_id.clone(),
            epoch: None,
            tracks: Vec::new(),
            source_lost: false,
            dims: None,
            last_pts_ns: 0,
            seq: Some(0),
            failure_logged: false,
            failures: 0,
            waiting_since: None,
        })
    }
    pub(super) fn emit_frame(
        &mut self,
        seq: u64,
        meta: &RowMeta,
        tracks: Vec<ReplayTrack>,
        bed: Option<BedPolygon>,
        night: bool,
    ) -> Result<(), TraceError> {
        let mut row = base_row(&self.camera_id, seq, meta);
        row.source_event = SourceEvent::Frame;
        row.tracks = tracks;
        row.bed = bed;
        row.night_window_active = night;
        self.append(&row)
    }

    /// Open, reconnect, or frame. Prior tracks stay across the change.
    pub(super) fn advance_epoch(&mut self, epoch: (u64, u64)) -> SourceEvent {
        let event = match self.epoch {
            None => SourceEvent::Open,
            Some(previous) if previous != epoch => SourceEvent::Reconnect,
            Some(_) => SourceEvent::Frame,
        };
        self.epoch = Some(epoch);
        self.source_lost = false;
        event
    }

    pub(super) fn control(&mut self, meta: &RowMeta) -> Result<(), TraceError> {
        let Some(seq) = self.take_seq() else {
            self.note_failure();
            return Err(TraceError::Bounds);
        };
        let row = base_row(&self.camera_id, seq, meta);
        self.append(&row)
    }

    /// Validate every new track before `retain` or `update`.
    pub(super) fn compose(
        &mut self,
        frame: &Frame,
        observed: &[ObservedPose],
        width: u64,
        height: u64,
    ) -> Result<Vec<ReplayTrack>, ()> {
        if observed.len() > MEDIA_MAX_OBJECTS
            || frame.live_track_ids.len() > MEDIA_MAX_OBJECTS
            || observed
                .iter()
                .any(|pose| !frame.live_track_ids.contains(&pose.track_id))
        {
            return Err(());
        }
        let mut current: Vec<ReplayTrack> = Vec::new();
        for pose in observed {
            let track_id = i128::from(pose.track_id);
            let known = self.tracks.iter().any(|track| track.track_id == track_id);
            let track = validated_track(pose, known, width, height)?;
            match current.iter_mut().find(|item| item.track_id == track_id) {
                Some(existing) => *existing = track,
                None => current.push(track),
            }
        }
        let mut non_observed = Vec::new();
        self.tracks.retain(|previous| {
            if current
                .iter()
                .any(|track| track.track_id == previous.track_id)
            {
                return true;
            }
            let live = frame
                .live_track_ids
                .iter()
                .any(|&id| i128::from(id) == previous.track_id);
            non_observed.push(ReplayTrack {
                track_id: previous.track_id,
                lifecycle: if live {
                    Lifecycle::Shadow
                } else {
                    Lifecycle::Lost
                },
                bbox: previous.bbox,
                keypoints: previous.keypoints,
            });
            live
        });
        for track in &current {
            match self
                .tracks
                .iter_mut()
                .find(|item| item.track_id == track.track_id)
            {
                Some(stored) => {
                    stored.bbox = track.bbox;
                    stored.keypoints = track.keypoints;
                }
                None => self.tracks.push(track.clone()),
            }
        }
        current.extend(non_observed);
        Ok(current)
    }

    pub(super) fn remember(&mut self, width: u64, height: u64, pts: u64) {
        self.dims = Some((width, height));
        self.last_pts_ns = pts;
    }

    pub(super) fn processed(&mut self, now: Duration) {
        self.waiting_since = Some(now);
    }

    /// No lost row before an accepted epoch or without saved dimensions.
    pub(super) fn lose_if_due(&mut self, now: Duration) {
        let due = self.waiting_since.is_some_and(|started| {
            self.epoch.is_some() && !self.source_lost && now.saturating_sub(started) >= SOURCE_LOSS
        });
        let Some((width, height)) = self.dims.filter(|_| due) else {
            return;
        };
        let Some(epoch) = self.epoch.map(|pair| pair.0) else {
            return;
        };
        self.source_lost = true;
        let meta = RowMeta {
            pts: self.last_pts_ns,
            epoch,
            event: SourceEvent::Lost,
            width,
            height,
        };
        let _ = self.control(&meta);
    }

    pub(super) fn take_seq(&mut self) -> Option<u64> {
        let seq = self.seq?;
        self.seq = seq.checked_add(1);
        Some(seq)
    }

    /// `Ok(false)` is the writer's own drop and is not a capture failure.
    fn append(&mut self, row: &ReplayRow) -> Result<(), TraceError> {
        match self.writer.append(row) {
            Ok(_) => Ok(()),
            Err(error) => {
                self.note_failure();
                Err(error)
            }
        }
    }

    pub(super) fn note_failure(&mut self) {
        self.failures = self.failures.saturating_add(1);
        if self.failure_logged {
            return;
        }
        self.failure_logged = true;
        eprintln!(
            "ml-worker: replay trace write failed: camera_id={}",
            self.camera_id
        );
    }
}

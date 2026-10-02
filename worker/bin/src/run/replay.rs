//! One retained writer per admitted camera.
//!
//! Sequence follows `NativePolicyPump._capture_replay_row_unchecked` and
//! `_capture_source_lost`. Persistence is outside the policy result: a
//! writer drop continues bookkeeping rather than counting as a failure.
//! Unlike Python's initial 1x1 placeholder, source loss requires saved actual
//! dimensions; no geometry is invented when the first capture fails.
//! Frame geometry is in the configured mux/perception plane, matching Python
//! convert_frame. Native identity separately retains the decoded source size.

mod camera;
mod geometry;

use std::path::Path;
use std::time::Duration;

use seeon_deepstream_native::MEDIA_MAX_SOURCES;
use seeon_worker::detection_window::DetectionWindow;

use crate::config::pull::PulledConfig;
use crate::policy::ingest::{Frame, ObservedPose};
use crate::relay::cameras::{BedZoneRegion, RuntimeCamera};
use crate::seam::Clock;
use crate::trace_out::{BedPolygon, TraceError};

use camera::CameraTrace;
use geometry::{RowMeta, persisted_polygon};

/// Python `wait_accepted(..., timeout_sec=0.5)`.
const SOURCE_LOSS: Duration = Duration::from_millis(500);

struct StoredPolygon {
    points: Vec<(i128, i128)>,
    image_width: Option<u64>,
    image_height: Option<u64>,
}

/// Retained writers for an enabled replay-trace directory.
pub(crate) struct ReplayCapture {
    cameras: Vec<CameraTrace>,
    polygons: Vec<Option<StoredPolygon>>,
    bed_exit_enabled: bool,
    window: Option<DetectionWindow>,
}

impl ReplayCapture {
    /// Domain selection is resolved before any writer directory is created.
    /// Source ids are the roster indexes media assembly assigns.
    pub(crate) fn new(directory: &Path, config: &PulledConfig) -> Result<Self, TraceError> {
        if config.cameras.len() > MEDIA_MAX_SOURCES {
            return Err(TraceError::Bounds);
        }
        let domains = config
            .config
            .domain_selection()
            .resolve()
            .map_err(|_| TraceError::Bounds)?;
        let mut cameras = Vec::with_capacity(config.cameras.len());
        let mut polygons = Vec::with_capacity(config.cameras.len());
        for camera in &config.cameras {
            cameras.push(CameraTrace::open(directory, camera)?);
            polygons.push(stored_polygon(camera));
        }
        let window = domains
            .bed_exit
            .then(|| {
                config
                    .windows
                    .get("bed_exit")
                    .map(|item| item.window.clone())
            })
            .flatten();
        Ok(Self {
            cameras,
            polygons,
            bed_exit_enabled: domains.bed_exit,
            window,
        })
    }
    /// After successful ingest, before disabled-stage and window branches.
    pub(crate) fn capture(&mut self, frame: &Frame, observed: &[ObservedPose], clock: &dyn Clock) {
        let Some(index) = source_index(frame.identity.source_id, self.cameras.len()) else {
            return;
        };
        let Some((width, height)) = perception_size(frame) else {
            self.cameras[index].note_failure();
            return;
        };
        let Some(pts) = presentation(frame) else {
            self.cameras[index].note_failure();
            return;
        };
        let epoch = (
            frame.identity.binding.epoch,
            frame.identity.binding.generation,
        );
        let event = self.cameras[index].advance_epoch(epoch);
        let meta = RowMeta {
            pts,
            epoch: epoch.0,
            event,
            width,
            height,
        };
        if event != crate::trace_out::SourceEvent::Frame
            && self.cameras[index].control(&meta).is_err()
        {
            return;
        }
        let tracks = match self.cameras[index].compose(frame, observed, width, height) {
            Ok(tracks) => tracks,
            Err(()) => {
                self.cameras[index].note_failure();
                return;
            }
        };
        let bed = self.bed(index, width, height);
        let Some(seq) = self.cameras[index].take_seq() else {
            self.cameras[index].note_failure();
            return;
        };
        let night = match self.night(clock) {
            Ok(active) => active,
            Err(_) => {
                self.cameras[index].note_failure();
                return;
            }
        };
        if self.cameras[index]
            .emit_frame(seq, &meta, tracks, bed, night)
            .is_ok()
        {
            self.cameras[index].remember(width, height, pts);
        }
    }

    /// After `observe` returns, including a frame-local refusal.
    pub(crate) fn processed(&mut self, source_id: u32, now: Duration) {
        let Some(index) = source_index(source_id, self.cameras.len()) else {
            return;
        };
        self.cameras[index].processed(now);
    }

    /// Only a real empty receive evaluates source loss.
    pub(crate) fn wait_empty(&mut self, now: Duration) {
        for camera in &mut self.cameras {
            camera.lose_if_due(now);
        }
    }

    fn night(&self, clock: &dyn Clock) -> Result<bool, geometry::GeometryError> {
        geometry::night_window_active(self.bed_exit_enabled, self.window.as_ref(), clock.wall())
    }
    fn bed(&self, index: usize, width: u64, height: u64) -> Option<BedPolygon> {
        let stored = self.polygons.get(index)?.as_ref()?;
        Some(persisted_polygon(
            &stored.points,
            stored.image_width,
            stored.image_height,
            width,
            height,
        ))
    }
}

fn stored_polygon(camera: &RuntimeCamera) -> Option<StoredPolygon> {
    let BedZoneRegion { polygon, .. } = camera.bed_zone_regions.first()?;
    Some(StoredPolygon {
        points: polygon.clone(),
        image_width: camera.bed_zone_image_width,
        image_height: camera.bed_zone_image_height,
    })
}

fn source_index(source_id: u32, len: usize) -> Option<usize> {
    usize::try_from(source_id).ok().filter(|index| *index < len)
}

fn perception_size(frame: &Frame) -> Option<(u64, u64)> {
    let size = (
        u64::try_from(frame.width).ok()?,
        u64::try_from(frame.height).ok()?,
    );
    (size.0 > 0 && size.1 > 0).then_some(size)
}

/// `0` is absent, `1` is the actual PTS. Any other value is not fabricated.
fn presentation(frame: &Frame) -> Option<u64> {
    match frame.identity.pts_valid {
        0 => Some(0),
        1 => Some(frame.identity.pts_ns),
        _ => None,
    }
}

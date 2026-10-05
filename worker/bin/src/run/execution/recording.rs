//! Persist sealed attribution before moving media; retire it only after a terminal publication.
use std::collections::BTreeMap;
use std::path::Path;

use seeon_deepstream_native::MediaResult;

use crate::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use crate::clips::publish::{PublishError, Publisher};
use crate::clips::recorder::ClipSealed;
use crate::clips::rendition::{Tools, video_codec};
use crate::clips::reserve::{ReservePool, SaveOutcome};
use crate::clips::sealed::{
    SealedClip, SealedContributor, SealedEvent, SealedObservation, SealedSidecars,
    SealedUnavailable,
};
use crate::clips::store::ClipStore;
use crate::clips::time::Utc;
use crate::delivery::DeliveryQueue;

pub(super) struct RecordingPublisher<'a> {
    pub reserve: &'a mut ReservePool,
    pub store: &'a ClipStore,
    pub queue: &'a DeliveryQueue,
    pub sidecars: &'a SealedSidecars,
    pub events: &'a BTreeMap<String, SealedEvent>,
}

impl RecordingPublisher<'_> {
    pub fn save(
        &mut self,
        clip_id: &str,
        sealed: &ClipSealed,
        now: Utc,
    ) -> Result<SaveOutcome, PublishError> {
        let extension = sealed.extension();
        let boundary = extension.boundary.clone();
        let mut events = BTreeMap::new();
        for contributor in &sealed.contributors {
            let event = self
                .events
                .get(&contributor.event_ref)
                .ok_or(PublishError::Reservation("contributor"))?;
            events.insert(
                contributor.event_ref.clone(),
                ContributorEvent {
                    camera_id: event.camera_id.clone(),
                    facility_id: event.facility_id.clone(),
                    domain: event.domain.clone(),
                    event_type: event.event_type.clone(),
                },
            );
        }
        let ready = sealed.result == MediaResult::Ok && sealed.contains_video;
        let path = sealed
            .path
            .as_deref()
            .map(|path| {
                path.to_str()
                    .map(str::to_owned)
                    .ok_or(PublishError::Reservation("media path"))
            })
            .transpose()?;
        let contributors = sealed
            .contributors
            .iter()
            .map(|contributor| SealedContributor {
                event_ref: contributor.event_ref.clone(),
                detected_at: contributor.detected_at.iso_micros(),
            })
            .collect();
        let durable = if ready {
            SealedObservation::Ready(SealedClip {
                clip_id: clip_id.to_owned(),
                path: path.ok_or(PublishError::MissingMedia)?,
                duration_ms: i64::try_from(sealed.duration_ms)
                    .map_err(|_| PublishError::Reservation("duration"))?,
                boundary,
                contributors,
            })
        } else {
            SealedObservation::Unavailable(SealedUnavailable {
                clip_id: clip_id.to_owned(),
                path,
                duration_ms: sealed.duration_ms,
                boundary,
                contributors,
                native_result: native_result(sealed.result),
                contains_video: sealed.contains_video,
            })
        };
        let sidecar = self.sidecars.persist(&durable, self.events).map_err(|_| {
            PublishError::Io(std::io::Error::other(
                "sealed attribution persistence failed",
            ))
        })?;
        let meta = flow_metadata(clip_id, &events, extension, FLOW_ENCODER, now)
            .map_err(|_| PublishError::Reservation("sealed metadata"))?;
        // Unavailable publications carry no media facts and therefore no codec claim.
        let codec = if ready {
            measured_codec(sealed.path.as_deref().ok_or(PublishError::MissingMedia)?)
        } else {
            Ok(String::new())
        };
        let publisher = Publisher::new(self.queue);
        let saved = match codec {
            Ok(codec) => self
                .reserve
                .save(self.store, &publisher, &meta, sealed, &codec)?,
            Err(_) => self
                .reserve
                .publish_preterminal_finalize_failed(self.store, &publisher, &meta)?,
        };
        if !matches!(saved, SaveOutcome::FinalizeFailed(None)) {
            self.sidecars.remove(&sidecar).map_err(|_| {
                PublishError::Io(std::io::Error::other(
                    "sealed attribution retirement failed",
                ))
            })?;
        }
        Ok(saved)
    }
}

fn native_result(result: MediaResult) -> i32 {
    match result {
        MediaResult::Ok => 0,
        MediaResult::Empty => 1,
        MediaResult::Busy => 2,
        MediaResult::Stale => 3,
        MediaResult::TooSmall => 4,
        MediaResult::Unsupported => 5,
        MediaResult::Fatal => 6,
        MediaResult::Unknown(value) => value,
    }
}

pub(super) fn measured_codec(path: &Path) -> Result<String, PublishError> {
    video_codec(&Tools::default(), path)
        .map_err(|_| PublishError::Io(std::io::Error::other("sealed media codec probe failed")))
}

pub(super) fn admits_path(directory: &Path, receipt: &crate::msg::RecordReceipt) -> bool {
    let mut parts = Path::new(&receipt.filename).components();
    if !matches!(parts.next(), Some(std::path::Component::Normal(_))) || parts.next().is_some() {
        return false;
    }
    let (Ok(expected), Ok(actual)) = (directory.canonicalize(), receipt.directory.canonicalize())
    else {
        return false;
    };
    expected == actual
        && std::fs::symlink_metadata(actual.join(&receipt.filename))
            .is_ok_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
}

#[cfg(test)]
#[path = "recording_tests.rs"]
mod tests;

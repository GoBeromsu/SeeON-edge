//! Persist sealed attribution before moving media; retire it only after a terminal publication.
use std::collections::BTreeMap;
use std::path::Path;

use seeon_deepstream_native::MediaResult;

use crate::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use crate::clips::publish::{PublishError, Publisher};
use crate::clips::recorder::ClipSealed;
use crate::clips::rendition::{Tools, video_codec};
use crate::clips::reserve::{ReservePool, SaveOutcome};
use crate::clips::sealed::{SealedClip, SealedContributor, SealedEvent, SealedSidecars};
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
        let meta = flow_metadata(clip_id, &events, extension, FLOW_ENCODER, now)
            .map_err(|_| PublishError::Reservation("sealed metadata"))?;
        let ready = sealed.result == MediaResult::Ok && sealed.contains_video;
        let sidecar = if ready {
            let durable = SealedClip {
                clip_id: clip_id.to_owned(),
                path: sealed
                    .path
                    .to_str()
                    .ok_or(PublishError::Reservation("media path"))?
                    .to_owned(),
                duration_ms: i64::try_from(sealed.duration_ms)
                    .map_err(|_| PublishError::Reservation("duration"))?,
                boundary,
                contributors: sealed
                    .contributors
                    .iter()
                    .map(|contributor| SealedContributor {
                        event_ref: contributor.event_ref.clone(),
                        detected_at: contributor.detected_at.iso_micros(),
                    })
                    .collect(),
            };
            Some(self.sidecars.persist(&durable, self.events).map_err(|_| {
                PublishError::Io(std::io::Error::other(
                    "sealed attribution persistence failed",
                ))
            })?)
        } else {
            None
        };
        // Unavailable publications carry no media facts and therefore no codec claim.
        let codec = if ready {
            measured_codec(&sealed.path)?
        } else {
            String::new()
        };
        let saved = self.reserve.save(
            self.store,
            &Publisher::new(self.queue),
            &meta,
            sealed,
            &codec,
        )?;
        if !matches!(saved, SaveOutcome::FinalizeFailed(None))
            && let Some(path) = sidecar
        {
            self.sidecars.remove(&path).map_err(|_| {
                PublishError::Io(std::io::Error::other(
                    "sealed attribution retirement failed",
                ))
            })?;
        }
        Ok(saved)
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

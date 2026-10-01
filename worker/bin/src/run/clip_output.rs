//! Ready sealed-replay composition: a persisted `Recovery` becomes one
//! READY clip through the existing store, publisher and durable file
//! primitives. Sidecar retirement stays with the caller. Space exhaustion
//! stays with `ReservePool`; this adapter never fabricates a ticket for it.

use std::collections::BTreeMap;
use std::path::Path;

use crate::clips::durable::{self, Existing, PUBLIC_FILE};
use crate::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use crate::clips::manifest::{Contributor, Extension, MAX_MANIFEST_BYTES, MediaFacts};
use crate::clips::publish::{MANIFEST_FILE, MEDIA_FILE, PublishError, Published, Publisher};
use crate::clips::sealed::{Recovery, SealedContributor};
use crate::clips::store::{ClipStore, Reservation};
use crate::clips::time::{TimeError, Utc};
use crate::delivery::DeliveryQueue;
use crate::json::{Json, JsonError};

/// A sealed recovery that cannot become the clip it claims.
#[derive(Debug)]
pub enum ClipOutputError {
    /// A contributor timestamp is not a UTC instant.
    Time(TimeError),
    /// The recovery's camera does not match every contributor event.
    CameraMismatch,
    /// Flow metadata refused the contributors, span, or duration.
    Metadata(crate::clips::manifest::ManifestError),
    /// The caller did not name the admitted codec.
    BlankCodec,
    /// An earlier manifest is unreadable, so its timestamp cannot be recovered.
    ManifestUnreadable,
    /// Publication, staging, or hashing failed. The sidecar is untouched.
    Publish(PublishError),
}

impl From<PublishError> for ClipOutputError {
    fn from(error: PublishError) -> Self {
        Self::Publish(error)
    }
}
impl From<std::io::Error> for ClipOutputError {
    fn from(error: std::io::Error) -> Self {
        Self::Publish(PublishError::Io(error))
    }
}

impl std::fmt::Display for ClipOutputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "sealed recovery publication refused: {self:?}")
    }
}

impl std::error::Error for ClipOutputError {}

/// Publishes one ready sealed recovery. `codec` is the admitted Smart Record
/// codec (H.264 or H.265 as measured); it is not inferred from the bytes.
/// `now` is the publication clock, except an earlier finalized manifest keeps
/// its own `finalized_at`.
pub fn publish_recovery(
    recovery: &Recovery,
    store: &ClipStore,
    queue: &DeliveryQueue,
    codec: &str,
    now: Utc,
) -> Result<Published, ClipOutputError> {
    if codec.trim().is_empty() {
        return Err(ClipOutputError::BlankCodec);
    }
    let events = contributor_events(recovery)?;
    let extension = extension(recovery)?;
    let reservation = store.reserve(&recovery.camera_id, &recovery.sealed.clip_id)?;
    let finalized_at = recovered_finalized_at(&reservation)?;
    let meta = flow_metadata(
        &recovery.sealed.clip_id,
        &events,
        extension,
        FLOW_ENCODER,
        finalized_at.unwrap_or(now),
    )
    .map_err(ClipOutputError::Metadata)?;
    stage_source(&recovery.sealed.path, &reservation)?;
    let facts = media_facts(&reservation, codec, meta.duration_ms)?;
    Ok(Publisher::new(queue).publish_ready(&reservation, &meta, &facts)?)
}

fn contributor_events(
    recovery: &Recovery,
) -> Result<BTreeMap<String, ContributorEvent>, ClipOutputError> {
    if recovery.sealed.contributors.is_empty() {
        return Err(ClipOutputError::Metadata(
            crate::clips::manifest::ManifestError::Blank("contributors"),
        ));
    }
    let mut events = BTreeMap::new();
    for contributor in &recovery.sealed.contributors {
        let event =
            recovery
                .events
                .get(&contributor.event_ref)
                .ok_or(ClipOutputError::Metadata(
                    crate::clips::manifest::ManifestError::MissingEvent,
                ))?;
        if event.camera_id != recovery.camera_id {
            return Err(ClipOutputError::CameraMismatch);
        }
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
    Ok(events)
}

fn extension(recovery: &Recovery) -> Result<Extension, ClipOutputError> {
    let contributors = recovery
        .sealed
        .contributors
        .iter()
        .map(contributor_time)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Extension {
        boundary: recovery.sealed.boundary.clone(),
        contributors,
        duration_ms: recovery.sealed.duration_ms,
    })
}

fn contributor_time(contributor: &SealedContributor) -> Result<Contributor, ClipOutputError> {
    Ok(Contributor {
        event_ref: contributor.event_ref.clone(),
        detected_at: Utc::parse(&contributor.detected_at).map_err(ClipOutputError::Time)?,
    })
}

/// Only a canonical object with a parseable `finalized_at` is a prior
/// publication. Identity is not trusted here; byte equality still belongs
/// to `Publisher`.
fn recovered_finalized_at(reservation: &Reservation) -> Result<Option<Utc>, ClipOutputError> {
    let path = reservation.final_dir.join(MANIFEST_FILE);
    let bytes = match durable::read_bounded(&path, MAX_MANIFEST_BYTES as u64)? {
        Existing::Missing => return Ok(None),
        Existing::Unreadable => return Err(ClipOutputError::ManifestUnreadable),
        Existing::Bytes(bytes) => bytes,
    };
    let text = std::str::from_utf8(&bytes).map_err(|_| ClipOutputError::ManifestUnreadable)?;
    let (body, newline) = text
        .split_once('\n')
        .ok_or(ClipOutputError::ManifestUnreadable)?;
    if !newline.is_empty() {
        return Err(ClipOutputError::ManifestUnreadable);
    }
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|_| ClipOutputError::ManifestUnreadable)?;
    let model = Json::from(&value);
    let canonical = crate::json::Serialiser::ModelSelection
        .canonical(&model)
        .map_err(manifest_json_error)?;
    if canonical != body {
        return Err(ClipOutputError::ManifestUnreadable);
    }
    let Json::Object(members) = model else {
        return Err(ClipOutputError::ManifestUnreadable);
    };
    let Some(Json::Str(stamp)) = members
        .iter()
        .find(|(key, _)| key == "finalized_at")
        .map(|(_, v)| v)
    else {
        return Err(ClipOutputError::ManifestUnreadable);
    };
    Utc::parse(stamp)
        .map(Some)
        .map_err(|_| ClipOutputError::ManifestUnreadable)
}

fn manifest_json_error(error: JsonError) -> ClipOutputError {
    ClipOutputError::Metadata(crate::clips::manifest::ManifestError::Json(error))
}

/// Moves the source into the reservation artifact unless a crash already
/// staged it or `publish_ready` already placed the final media.
fn stage_source(source: &str, reservation: &Reservation) -> Result<(), ClipOutputError> {
    let artifact = reservation.artifact_path();
    let video = reservation.final_dir.join(MEDIA_FILE);
    let source = Path::new(source);
    if regular_file(&artifact) || regular_file(&video) {
        return Ok(());
    }
    if !regular_file(source) {
        return Err(PublishError::MissingMedia.into());
    }
    durable::move_durable(source, &artifact, PUBLIC_FILE)?;
    Ok(())
}

fn media_facts(
    reservation: &Reservation,
    codec: &str,
    duration_ms: i64,
) -> Result<MediaFacts, ClipOutputError> {
    let artifact = reservation.artifact_path();
    let video = reservation.final_dir.join(MEDIA_FILE);
    let path = if regular_file(&artifact) {
        &artifact
    } else {
        &video
    };
    let (sha256, size) = durable::sha256_file(path)?;
    Ok(MediaFacts {
        sha256,
        size_bytes: i64::try_from(size).map_err(|_| PublishError::MissingMedia)?,
        codec: codec.to_owned(),
        duration_ms,
    })
}
fn regular_file(path: &Path) -> bool {
    path.symlink_metadata().is_ok_and(|meta| meta.is_file())
}

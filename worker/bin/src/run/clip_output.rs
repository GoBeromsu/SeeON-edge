//! Sealed-replay composition: reconcile a prior terminal before a persisted
//! `Recovery` becomes a fresh READY clip through the store and publisher
//! primitives. Sidecar retirement stays with the caller. Space exhaustion
//! stays with `ReservePool`; this adapter never fabricates a ticket for it.

mod terminal;

pub use terminal::resume_terminal;

use std::collections::BTreeMap;
use std::path::Path;

use crate::clips::durable::{self, PUBLIC_FILE};
use crate::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use crate::clips::manifest::{ClipMetadata, Contributor, Extension, MediaFacts};
use crate::clips::publish::{MEDIA_FILE, PublishError, Published, Publisher};
use crate::clips::sealed::{Recovery, SealedContributor};
use crate::clips::store::{ClipStore, Reservation};
use crate::clips::time::{TimeError, Utc};
use crate::delivery::DeliveryQueue;

/// A sealed recovery that cannot become the clip it claims.
#[derive(Debug)]
pub enum ClipOutputError {
    /// A contributor timestamp is not a UTC instant.
    Time(TimeError),
    /// The recovery's camera does not match every contributor event.
    CameraMismatch,
    /// Flow metadata refused the contributors, span, or duration.
    Metadata(crate::clips::manifest::ManifestError),
    /// A fresh READY publication did not name the admitted codec.
    BlankCodec,
    /// An existing manifest is not a bounded canonical terminal snapshot.
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

/// Resumes a canonical terminal, or publishes one fresh ready sealed recovery.
/// Only the fresh path uses `codec` (the measured Smart Record codec) and `now`.
pub fn publish_recovery(
    recovery: &Recovery,
    store: &ClipStore,
    queue: &DeliveryQueue,
    codec: &str,
    now: Utc,
) -> Result<Published, ClipOutputError> {
    if let Some(published) = resume_terminal(recovery, store, queue)? {
        return Ok(published);
    }
    if codec.trim().is_empty() {
        return Err(ClipOutputError::BlankCodec);
    }
    let meta = metadata(recovery, now)?;
    let reservation = store.reserve(&recovery.camera_id, &recovery.sealed.clip_id)?;
    stage_source(&recovery.sealed.path, &reservation)?;
    let artifact = reservation.artifact_path();
    let video = reservation.final_dir.join(MEDIA_FILE);
    let path = if regular_file(&artifact) {
        &artifact
    } else {
        &video
    };
    let facts = media_facts(path, codec, meta.duration_ms)?;
    Ok(Publisher::new(queue).publish_ready(&reservation, &meta, &facts)?)
}

fn metadata(recovery: &Recovery, finalized_at: Utc) -> Result<ClipMetadata, ClipOutputError> {
    flow_metadata(
        &recovery.sealed.clip_id,
        &contributor_events(recovery)?,
        extension(recovery)?,
        FLOW_ENCODER,
        finalized_at,
    )
    .map_err(ClipOutputError::Metadata)
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

fn media_facts(path: &Path, codec: &str, duration_ms: i64) -> Result<MediaFacts, ClipOutputError> {
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

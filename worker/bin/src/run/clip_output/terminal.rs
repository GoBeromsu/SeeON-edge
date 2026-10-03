//! Resume only a bounded canonical terminal with the sealed attribution.
//! The manifest supplies its old clock and, for READY, its admitted codec;
//! it never supplies unverified identity or permission to retire missing media.

use std::path::Path;

use serde_json::Value;

use super::{ClipOutputError, media_facts, metadata, regular_file};
use crate::clips::durable::{self, Existing};
use crate::clips::manifest::{MAX_MANIFEST_BYTES, MediaFacts, Terminal, manifest_bytes};
use crate::clips::publish::{
    MANIFEST_FILE, MEDIA_FILE, PublishError, Published, Publisher, TERMINAL_MARKER,
};
use crate::clips::reserve::FINALIZE_FAILED;
use crate::clips::sealed::Recovery;
use crate::clips::store::ClipStore;
use crate::clips::time::Utc;
use crate::delivery::DeliveryQueue;
use crate::json::{Json, Serialiser};

/// Reconciles an existing terminal before any media search or codec probe.
/// Only absent manifest and marker return `None`. A READY snapshot requires its
/// final media and exact hash, size and sealed duration. A READY sidecar may
/// resume UNAVAILABLE only for an existing FINALIZE_FAILED without a source
/// error. `Publisher` owns all terminal barriers and at-least-once admission;
/// the caller retires the sidecar only after this handoff succeeds.
pub fn resume_terminal(
    recovery: &Recovery,
    store: &ClipStore,
    queue: &DeliveryQueue,
) -> Result<Option<Published>, ClipOutputError> {
    let reservation = store.reserve(&recovery.camera_id, &recovery.sealed.clip_id)?;
    let Some((bytes, value)) = snapshot(&reservation.final_dir.join(MANIFEST_FILE))? else {
        return match reservation
            .final_dir
            .join(TERMINAL_MARKER)
            .symlink_metadata()
        {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
            Ok(_) => Err(PublishError::Conflict.into()),
        };
    };
    let stamp = text(&value, "finalized_at")?;
    let finalized_at = Utc::parse(stamp).map_err(|_| ClipOutputError::ManifestUnreadable)?;
    let meta = metadata(recovery, finalized_at)?;
    let terminal = terminal(&value)?;
    let expected = manifest_bytes(&meta, &terminal).map_err(ClipOutputError::Metadata)?;
    if expected != bytes {
        return Err(PublishError::Conflict.into());
    }
    let publisher = Publisher::new(queue);
    let published = match terminal {
        Terminal::Ready(facts) => {
            if facts.duration_ms != meta.duration_ms {
                return Err(PublishError::Conflict.into());
            }
            let video = reservation.final_dir.join(MEDIA_FILE);
            if !regular_file(&video) {
                return Err(PublishError::MissingMedia.into());
            }
            if media_facts(&video, &facts.codec, meta.duration_ms)? != facts {
                return Err(PublishError::Conflict.into());
            }
            publisher.publish_ready(&reservation, &meta, &facts)?
        }
        Terminal::Unavailable {
            reason_code,
            source_error_reason,
        } => publisher.publish_unavailable(
            &reservation,
            &meta,
            &reason_code,
            source_error_reason.as_deref(),
        )?,
    };
    Ok(Some(published))
}

fn snapshot(path: &Path) -> Result<Option<(Vec<u8>, Value)>, ClipOutputError> {
    let bytes = match durable::read_bounded(path, MAX_MANIFEST_BYTES as u64)? {
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
    let value: Value =
        serde_json::from_str(body).map_err(|_| ClipOutputError::ManifestUnreadable)?;
    let canonical = Serialiser::ModelSelection
        .canonical(&Json::from(&value))
        .map_err(|error| {
            ClipOutputError::Metadata(crate::clips::manifest::ManifestError::Json(error))
        })?;
    if canonical != body || !value.is_object() {
        return Err(ClipOutputError::ManifestUnreadable);
    }
    Ok(Some((bytes, value)))
}

fn terminal(value: &Value) -> Result<Terminal, ClipOutputError> {
    match text(value, "state")? {
        "READY" => Ok(Terminal::Ready(MediaFacts {
            sha256: text(value, "sha256")?.to_owned(),
            size_bytes: integer(value, "size_bytes")?,
            codec: text(value, "codec")?.to_owned(),
            duration_ms: integer(value, "duration_ms")?,
        })),
        "UNAVAILABLE"
            if text(value, "reason_code")? == FINALIZE_FAILED
                && value.get("source_error_reason").is_none() =>
        {
            Ok(Terminal::Unavailable {
                reason_code: FINALIZE_FAILED.to_owned(),
                source_error_reason: None,
            })
        }
        _ => Err(PublishError::Conflict.into()),
    }
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, ClipOutputError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| PublishError::Conflict.into())
}

fn integer(value: &Value, key: &str) -> Result<i64, ClipOutputError> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| PublishError::Conflict.into())
}

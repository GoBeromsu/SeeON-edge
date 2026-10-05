//! Reading sidecars back (`pending_for_camera`, `_read`).

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::observation::valid_clip_id;
use super::{
    MAX_SIDECAR_BYTES, Pending, Recovery, SealedClip, SealedContributor, SealedError, SealedEvent,
    SealedObservation, SealedUnavailable,
};
use crate::clips::durable::{self, Existing};

pub(super) fn pending(directory: &Path, camera_id: &str) -> Result<Pending, SealedError> {
    let mut names = match fs::read_dir(directory) {
        Ok(entries) => entries
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<io::Result<Vec<OsString>>>()?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Pending::default()),
        Err(error) => return Err(error.into()),
    };
    names.retain(|name| {
        name.to_str()
            .is_some_and(|name| name.ends_with(".json") && !name.starts_with('.'))
    });
    names.sort();
    let mut pending = Pending::default();
    for name in names {
        let path = directory.join(&name);
        match durable::read_bounded(&path, MAX_SIDECAR_BYTES)? {
            Existing::Missing => {}
            Existing::Unreadable => pending.malformed.push(path),
            Existing::Bytes(bytes) => match parse(&bytes, path.clone()) {
                Some(recovery) if recovery.camera_id == camera_id => {
                    pending.recoveries.push(recovery);
                }
                Some(_) => {}
                None => pending.malformed.push(path),
            },
        }
    }
    Ok(pending)
}

fn parse(bytes: &[u8], sidecar_path: PathBuf) -> Option<Recovery> {
    let value: Value = serde_json::from_slice(bytes).ok()?;
    let document = value.as_object()?;
    let (sealed, camera_id, event_values, is_unavailable) =
        if let Some(unavailable_value) = document.get("unavailable") {
            if document.len() != 1 {
                return None;
            }
            let fields = unavailable_value.as_object()?;
            if fields.keys().any(|key| {
                !matches!(
                    key.as_str(),
                    "clip_id"
                        | "camera_id"
                        | "duration_ms"
                        | "boundary"
                        | "contributors"
                        | "events"
                        | "native_result"
                        | "contains_video"
                        | "path"
                )
            }) {
                return None;
            }
            let path = match fields.get("path") {
                None => None,
                Some(value) => {
                    let path = value.as_str()?;
                    if path.is_empty() {
                        return None;
                    }
                    Some(path.to_owned())
                }
            };
            let native_result = i32::try_from(fields.get("native_result")?.as_i64()?).ok()?;
            let contains_video = fields.get("contains_video")?.as_bool()?;
            if native_result == 0 && contains_video {
                return None;
            }
            let unavailable = SealedUnavailable {
                clip_id: text(unavailable_value, "clip_id")?,
                path,
                duration_ms: fields.get("duration_ms")?.as_u64()?,
                boundary: text(unavailable_value, "boundary")?,
                contributors: parse_contributors(unavailable_value.get("contributors")?)?,
                native_result,
                contains_video,
            };
            if unavailable.contributors.is_empty() {
                return None;
            }
            let camera_id = text(unavailable_value, "camera_id")?;
            if camera_id.trim().is_empty() {
                return None;
            }
            (
                SealedObservation::Unavailable(unavailable),
                camera_id,
                unavailable_value.get("events")?,
                true,
            )
        } else {
            if document.contains_key("native_result") || document.contains_key("contains_video") {
                return None;
            }
            let sealed = SealedClip {
                clip_id: text(&value, "clip_id")?,
                path: text(&value, "path")?,
                duration_ms: value.get("duration_ms")?.as_i64()?,
                boundary: text(&value, "boundary")?,
                contributors: parse_contributors(value.get("contributors")?)?,
            };
            (
                SealedObservation::Ready(sealed),
                text(&value, "camera_id")?,
                value.get("events")?,
                false,
            )
        };
    if !valid_clip_id(sealed.clip_id())
        || sidecar_path.file_name()?.to_str()? != format!("{}.json", sealed.clip_id())
    {
        return None;
    }
    let events = parse_events(
        event_values,
        sealed.contributors(),
        &camera_id,
        is_unavailable,
    )?;
    Some(Recovery {
        camera_id,
        sealed,
        events,
        sidecar_path,
    })
}

fn parse_contributors(value: &Value) -> Option<Vec<SealedContributor>> {
    value
        .as_array()?
        .iter()
        .map(|item| {
            Some(SealedContributor {
                event_ref: text(item, "event_ref")?,
                detected_at: text(item, "detected_at")?,
            })
        })
        .collect()
}

fn parse_events(
    values: &Value,
    contributors: &[SealedContributor],
    camera_id: &str,
    is_unavailable: bool,
) -> Option<BTreeMap<String, SealedEvent>> {
    let mut references = BTreeSet::new();
    for contributor in contributors {
        if contributor.event_ref.trim().is_empty()
            || !references.insert(contributor.event_ref.as_str())
        {
            return None;
        }
    }
    let rows = values.as_array()?;
    if is_unavailable && rows.is_empty() {
        return None;
    }
    let mut events = BTreeMap::new();
    for item in rows {
        let event = SealedEvent {
            domain: text(item, "domain")?,
            event_type: text(item, "event_type")?,
            identity: text(item, "identity")?,
            camera_id: text(item, "camera_id")?,
            facility_id: text(item, "facility_id")?,
            time_sec: item.get("time_sec")?.as_f64()?,
            probability: finite_probability(item.get("probability")?)?,
        };
        if event.identity.trim().is_empty()
            || event.camera_id.trim().is_empty()
            || event.camera_id != camera_id
            || !references.contains(event.identity.as_str())
            || events.insert(event.identity.clone(), event).is_some()
        {
            return None;
        }
    }
    (events.len() == references.len()).then_some(events)
}

fn text(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_owned)
}

/// Null is absent. A finite number, including zero, is present. Missing,
/// bool, string, object, array, and non-finite values are not a probability.
fn finite_probability(value: &Value) -> Option<Option<f64>> {
    match value {
        Value::Null => Some(None),
        Value::Number(number) => {
            let probability = number.as_f64()?;
            probability.is_finite().then_some(Some(probability))
        }
        Value::Bool(_) | Value::String(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

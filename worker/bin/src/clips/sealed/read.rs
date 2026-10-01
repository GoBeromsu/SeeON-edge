//! Reading sidecars back (`pending_for_camera`, `_read`).

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{Pending, Recovery, SealedClip, SealedContributor, SealedError, SealedEvent};
use crate::clips::durable::{self, Existing};

const MAX_SIDECAR_BYTES: u64 = 1 << 20;

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
    let contributors = value
        .get("contributors")?
        .as_array()?
        .iter()
        .map(|item| {
            Some(SealedContributor {
                event_ref: text(item, "event_ref")?,
                detected_at: text(item, "detected_at")?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let sealed = SealedClip {
        clip_id: text(&value, "clip_id")?,
        path: text(&value, "path")?,
        duration_ms: value.get("duration_ms")?.as_i64()?,
        boundary: text(&value, "boundary")?,
        contributors,
    };
    let mut events = BTreeMap::new();
    for item in value.get("events")?.as_array()? {
        let event = SealedEvent {
            domain: text(item, "domain")?,
            event_type: text(item, "event_type")?,
            identity: text(item, "identity")?,
            camera_id: text(item, "camera_id")?,
            facility_id: text(item, "facility_id")?,
            time_sec: item.get("time_sec")?.as_f64()?,
            probability: item.get("probability")?.as_f64()?,
        };
        events.insert(event.identity.clone(), event);
    }
    let complete = sealed
        .contributors
        .iter()
        .all(|contributor| events.contains_key(&contributor.event_ref));
    complete.then_some(())?;
    Some(Recovery {
        camera_id: text(&value, "camera_id")?,
        sealed,
        events,
        sidecar_path,
    })
}

fn text(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_owned)
}

//! Policy emit (G3, T23): the Python `DurableEvidenceStager.stage` envelope
//! that turns one policy event into a delivery [`EventEntry`], and the plain
//! [`ModelScore`] value behind `emit_policy.model_score_record`. Nothing here
//! admits to the queue or calls an exporter; the caller owns both.

use std::fmt;

use crate::delivery::{EntryError, EventEntry, EventFields};
use crate::json::{Json, JsonError, Serialiser};

mod score;
mod text;

pub use score::{ModelEvidence, ModelScore};
use text::{dict, lookup, remove, required_text, upsert};

/// `shared/contracts/envelope_limits.py:VALUES_BYTES_MAX`.
pub const VALUES_BYTES_MAX: usize = 32 * 1024;
/// `shared/contracts/envelope_limits.py:DECISION_TRACE_BYTES_MAX`.
pub const DECISION_TRACE_BYTES_MAX: usize = 16 * 1024;
/// `worker/domains/fall/classifier.py:FALL_WINDOW_FRAMES`.
pub const FALL_WINDOW_FRAMES: i128 = 30;

const REQUIRED_ALERT_FIELDS: [&str; 5] = [
    "camera_id",
    "detected_at",
    "event_type",
    "facility_id",
    "probability",
];
const RELAY_AUDIT_FIELDS: [&str; 7] = [
    "clock_source",
    "config_version",
    "decision_trace_id",
    "detector_version",
    "model_version",
    "operating_threshold",
    "runtime_manifest_sha256",
];
const CONFIG_VERSION_KEY: &str = "config_version";
const MANIFEST_SHA_KEY: &str = "runtime_manifest_sha256";
const PROTECTED_TRACE_KEYS: [&str; 2] = [CONFIG_VERSION_KEY, MANIFEST_SHA_KEY];

/// An insertion-ordered JSON object, as a Python `dict` holds it.
pub type Payload = Vec<(String, Json)>;

/// Encoded `values`, encoded `decision_trace`, and the shed detail keys.
type Envelope = (Vec<u8>, Vec<u8>, Vec<String>);

/// Every refusal of [`Stager`]; Python raises `ValueError` for the first four.
#[derive(Debug)]
pub enum EmitError {
    /// `_required_text`: the named event field is absent, null or blank.
    BlankField(&'static str),
    /// The named field is an array or object, which has no text form here.
    FieldType(&'static str),
    /// `validate_runtime_manifest_sha256`: not 64 lowercase hex characters.
    ManifestSha,
    /// `_shed_to_limit`: the protected core alone exceeds `limit`.
    TooLarge {
        limit: usize,
        size: usize,
    },
    Json(JsonError),
    Entry(EntryError),
}

impl fmt::Display for EmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlankField(key) => write!(formatter, "event {key} must be set"),
            Self::FieldType(key) => write!(formatter, "event {key} has no text form"),
            Self::ManifestSha => formatter.write_str("runtime manifest sha256 is malformed"),
            Self::TooLarge { limit, size } => write!(
                formatter,
                "protected envelope core is {size} bytes, over {limit}"
            ),
            Self::Json(error) => write!(formatter, "envelope json: {error}"),
            Self::Entry(error) => write!(formatter, "event entry: {error}"),
        }
    }
}

impl std::error::Error for EmitError {}

impl From<JsonError> for EmitError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

impl From<EntryError> for EmitError {
    fn from(error: EntryError) -> Self {
        Self::Entry(error)
    }
}

/// The per-camera identity `DurableEvidenceStager` stamps on every event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stager {
    camera_id: String,
    facility_id: String,
    resident_id: Option<String>,
    config_version: i64,
    runtime_manifest_sha256: Option<String>,
}

impl Stager {
    pub fn new(
        camera_id: &str,
        facility_id: &str,
        config_version: i64,
        runtime_manifest_sha256: Option<&str>,
    ) -> Result<Self, EmitError> {
        let sha = runtime_manifest_sha256.map(|sha| Json::Str(sha.to_owned()));
        validate_manifest_sha(sha.as_ref())?;
        Ok(Self {
            camera_id: camera_id.to_owned(),
            facility_id: facility_id.to_owned(),
            resident_id: None,
            config_version,
            runtime_manifest_sha256: runtime_manifest_sha256.map(str::to_owned),
        })
    }

    pub fn with_resident_id(mut self, resident_id: &str) -> Self {
        self.resident_id = Some(resident_id.to_owned());
        self
    }

    /// Python `stage` up to, not including, `try_admit`.
    pub fn event_entry(&self, event: &[(String, Json)]) -> Result<EventEntry, EmitError> {
        let event = dict(event);
        let edge_event_id = required_text(&event, "edge_event_id")?;
        let detected_at = required_text(&event, "detected_at")?;
        let event_type = required_text(&event, "event_type")?;
        let (values, decision_trace, shed_detail_keys) = self.envelope(event)?;
        Ok(EventEntry::new(EventFields {
            edge_event_id,
            event_type,
            detected_at,
            camera_id: self.camera_id.clone(),
            facility_id: self.facility_id.clone(),
            decision_trace,
            values,
            shed_detail_keys,
            entry_id: String::new(),
        })?)
    }

    fn envelope(&self, mut values: Payload) -> Result<Envelope, EmitError> {
        remove(&mut values, "snapshot_jpeg");
        remove(&mut values, "snapshot");
        let audit = remove(&mut values, "audit");
        upsert(&mut values, "camera_id", Json::Str(self.camera_id.clone()));
        upsert(
            &mut values,
            "facility_id",
            Json::Str(self.facility_id.clone()),
        );
        if let Some(resident_id) = &self.resident_id {
            upsert(&mut values, "resident_id", Json::Str(resident_id.clone()));
        }
        let audit = match audit {
            Some(Json::Object(members)) => Some(dict(&members)),
            _ => None,
        };
        let mut shed_audit_keys = Vec::new();
        let mut trace = Payload::new();
        if audit.is_some() || self.runtime_manifest_sha256.is_some() {
            for (key, value) in audit.unwrap_or_default() {
                if RELAY_AUDIT_FIELDS.contains(&key.as_str()) {
                    trace.push((key, value));
                } else {
                    shed_audit_keys.push(format!("audit.{key}"));
                }
            }
            validate_manifest_sha(lookup(&trace, MANIFEST_SHA_KEY))?;
            let version = Json::Int(i128::from(self.config_version));
            upsert(&mut trace, CONFIG_VERSION_KEY, version);
            if let Some(sha) = &self.runtime_manifest_sha256 {
                upsert(&mut trace, MANIFEST_SHA_KEY, Json::Str(sha.clone()));
            }
        }
        let (values, mut shed) = shed_to_limit(values, VALUES_BYTES_MAX, &REQUIRED_ALERT_FIELDS)?;
        let (trace, _) = shed_to_limit(trace, DECISION_TRACE_BYTES_MAX, &PROTECTED_TRACE_KEYS)?;
        shed.extend(shed_audit_keys);
        shed.sort();
        Ok((values, trace, shed))
    }
}

/// `_shed_to_limit`: drop unprotected keys largest-first (ties keep
/// insertion order, as Python's stable reverse sort does) until it fits.
fn shed_to_limit(
    payload: Payload,
    limit: usize,
    protected: &[&str],
) -> Result<(Vec<u8>, Vec<String>), EmitError> {
    let mut encoded = canonical(&Json::Object(payload.clone()))?;
    if encoded.len() <= limit {
        return Ok((encoded, Vec::new()));
    }
    let mut sheddable = Vec::new();
    for (key, value) in &payload {
        if !protected.contains(&key.as_str()) {
            sheddable.push((key.clone(), canonical(value)?.len()));
        }
    }
    sheddable.sort_by(|left, right| right.1.cmp(&left.1));
    let mut remaining = payload;
    let mut shed = Vec::new();
    for (key, _) in sheddable {
        remaining.retain(|(name, _)| *name != key);
        shed.push(key);
        encoded = canonical(&Json::Object(remaining.clone()))?;
        if encoded.len() <= limit {
            shed.sort();
            return Ok((encoded, shed));
        }
    }
    Err(EmitError::TooLarge {
        limit,
        size: encoded.len(),
    })
}

/// `_canonical_bytes`: sorted keys, compact separators, ASCII escapes.
fn canonical(value: &Json) -> Result<Vec<u8>, EmitError> {
    Ok(Serialiser::ModelSelection.canonical(value)?.into_bytes())
}

/// `validate_runtime_manifest_sha256`: absent or null passes.
fn validate_manifest_sha(value: Option<&Json>) -> Result<(), EmitError> {
    match value {
        None | Some(Json::Null) => Ok(()),
        Some(Json::Str(sha))
            if sha.len() == 64 && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) =>
        {
            Ok(())
        }
        Some(_) => Err(EmitError::ManifestSha),
    }
}

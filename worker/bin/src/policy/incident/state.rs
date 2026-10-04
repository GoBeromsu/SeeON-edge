use std::collections::HashMap;
use std::time::Duration;

use seeon_worker::episode::BusinessEvent;

use super::{COOLDOWN, IncidentAuditSnapshot, IncidentError};
use crate::json::{Json, Serialiser};

/// Exact cooldown tuple: (camera, domain, event_type, ORIGINAL identity).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CooldownKey {
    pub camera: String,
    pub domain: String,
    pub event_type: String,
    pub source_identity: String,
}

impl From<&BusinessEvent> for CooldownKey {
    fn from(source: &BusinessEvent) -> Self {
        Self {
            camera: source.camera_id.clone(),
            domain: source.domain.clone(),
            event_type: source.event_type.clone(),
            source_identity: source.identity.clone(),
        }
    }
}

/// Charge is the byte length of ModelSelection's compact Python ASCII JSON
/// `[camera,domain,event_type,original_identity,seconds,nanoseconds]`, with
/// integers `now.as_secs()`/`now.subsec_nanos()`, and NO newline. This bounds
/// encoded retained cooldown records, not Rust allocator overhead. Each entry
/// stores its original charge so expiration/release debit it exactly once.
pub(super) fn encoded_charge(key: &CooldownKey, now: Duration) -> Result<usize, IncidentError> {
    let record = Json::Array(vec![
        Json::Str(key.camera.clone()),
        Json::Str(key.domain.clone()),
        Json::Str(key.event_type.clone()),
        Json::Str(key.source_identity.clone()),
        Json::Int(i128::from(now.as_secs())),
        Json::Int(i128::from(now.subsec_nanos())),
    ]);
    Ok(Serialiser::ModelSelection
        .canonical(&record)
        .map_err(IncidentError::Serialization)?
        .len())
}

pub(super) fn source_key(source: &BusinessEvent) -> Result<String, IncidentError> {
    Serialiser::ModelSelection
        .canonical(&Json::Array(vec![
            Json::Str(source.facility_id.clone()),
            Json::Str(source.camera_id.clone()),
            Json::Str(source.domain.clone()),
            Json::Str(source.event_type.clone()),
            Json::Str(source.identity.clone()),
            Json::Float(source.time_sec),
        ]))
        .map_err(IncidentError::Serialization)
}

impl IncidentAuditSnapshot {
    pub(super) fn capture(source: &BusinessEvent, key: &CooldownKey, edge_event_id: &str) -> Self {
        Self {
            edge_event_id: edge_event_id.to_owned(),
            source_identity: source.identity.clone(),
            cooldown_key: key.clone(),
            domain: source.domain.clone(),
            event_type: source.event_type.clone(),
            camera: source.camera_id.clone(),
            facility: source.facility_id.clone(),
            time_sec: source.time_sec,
            probability: source.probability,
            person_id: source.person_id,
            bed_id: source.bed_id,
        }
    }
}

struct Entry {
    admitted_at: Duration,
    charge: usize,
}

pub(super) struct Prepared {
    charge: usize,
    total_bytes: usize,
}

pub(super) struct CooldownState {
    entries: HashMap<CooldownKey, Entry>,
    max_bytes: usize,
    last_now: Option<Duration>,
    pub(super) suppressed_total: u64,
    pub(super) accounted_bytes: usize,
}

impl CooldownState {
    pub(super) fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
            last_now: None,
            suppressed_total: 0,
            accounted_bytes: 0,
        }
    }

    pub(super) fn observe(&mut self, now: Duration) -> Result<(), IncidentError> {
        if let Some(previous) = self.last_now
            && now < previous
        {
            return Err(IncidentError::ClockRegression { previous, now });
        }
        self.last_now = Some(now);
        Ok(())
    }

    pub(super) fn prune_expired(&mut self, now: Duration) {
        self.entries.retain(|_, entry| {
            let age = now
                .checked_sub(entry.admitted_at)
                .expect("nondecreasing admission samples precede no active entry");
            if age < COOLDOWN {
                return true;
            }
            self.accounted_bytes = self
                .accounted_bytes
                .checked_sub(entry.charge)
                .expect("expired cooldown charge is accounted exactly once");
            false
        });
    }

    pub(super) fn suppress(&mut self, key: &CooldownKey) -> Result<bool, IncidentError> {
        if !self.entries.contains_key(key) {
            return Ok(false);
        }
        self.suppressed_total = self
            .suppressed_total
            .checked_add(1)
            .ok_or(IncidentError::CounterOverflow)?;
        Ok(true)
    }

    pub(super) fn prepare(
        &mut self,
        key: &CooldownKey,
        now: Duration,
    ) -> Result<Prepared, IncidentError> {
        let charge = encoded_charge(key, now)?;
        let total_bytes = self
            .accounted_bytes
            .checked_add(charge)
            .filter(|total| *total <= self.max_bytes)
            .ok_or(IncidentError::Capacity)?;
        // Reserve before UUID allocation; a table allocation failure is fatal,
        // not a successful admission followed by an unrecordable cooldown.
        self.entries
            .try_reserve(1)
            .map_err(|_| IncidentError::Capacity)?;
        Ok(Prepared {
            charge,
            total_bytes,
        })
    }

    pub(super) fn commit(&mut self, key: CooldownKey, now: Duration, prepared: Prepared) {
        let replaced = self.entries.insert(
            key,
            Entry {
                admitted_at: now,
                charge: prepared.charge,
            },
        );
        debug_assert!(replaced.is_none());
        self.accounted_bytes = prepared.total_bytes;
    }

    pub(super) fn release(&mut self, key: &CooldownKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.accounted_bytes = self
                .accounted_bytes
                .checked_sub(entry.charge)
                .expect("released cooldown charge is accounted exactly once");
        }
    }

    pub(super) fn reset(&mut self) {
        self.entries.clear();
        self.accounted_bytes = 0;
    }
}

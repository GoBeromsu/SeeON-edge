//! Per-camera admission between opaque episode events and delivery preparation.
//! The caller supplies one nondecreasing monotonic sample per decision update.
//! No journal is opened here; pending admissions must retain their returned UUID.
//! Identity reuse depends on the existing in-process cache retaining the key;
//! this primitive provides neither prequeue crash recovery nor permanent keys.

mod state;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use seeon_worker::episode::BusinessEvent;

use super::identity::{EventIdentityStore, IdentityError, Limits};
use crate::json::JsonError;
use crate::seam::{Clock, IdSource};
use state::CooldownState;

pub use state::CooldownKey;

pub const COOLDOWN: Duration = Duration::from_secs(30);

/// Internal Python-compatible audit vocabulary, not an EventDelivery payload.
#[derive(Clone, Debug, PartialEq)]
pub struct IncidentAuditSnapshot {
    pub edge_event_id: String,
    pub source_identity: String,
    pub cooldown_key: CooldownKey,
    pub domain: String,
    pub event_type: String,
    pub camera: String,
    pub facility: String,
    pub time_sec: f64,
    pub probability: Option<f64>,
    pub person_id: Option<u64>,
    pub bed_id: Option<u64>,
}

/// Owns the original key, without a UUID reverse map or generation filtering.
/// Release only definitively unqueued abandonment, never retained retries or
/// uncertain/post-acceptance errors. A stale receipt must not release a new use
/// of its key; callers own that lifecycle constraint.
#[derive(Clone, Debug, PartialEq)]
pub struct ReleaseReceipt {
    key: CooldownKey,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedIncident {
    pub event: BusinessEvent,
    pub source_key: String,
    pub audit: IncidentAuditSnapshot,
    pub release: ReleaseReceipt,
}

/// Every error is fatal to the caller; only `Ok(None)` means suppression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IncidentError {
    ClockRegression { previous: Duration, now: Duration },
    Identity(IdentityError),
    Serialization(JsonError),
    Capacity,
    CounterOverflow,
}

impl fmt::Display for IncidentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClockRegression { previous, now } => {
                write!(
                    formatter,
                    "incident monotonic clock regressed: {previous:?} to {now:?}"
                )
            }
            Self::Identity(error) => write!(formatter, "incident identity: {error}"),
            Self::Serialization(error) => write!(formatter, "incident serialization: {error}"),
            Self::Capacity => formatter.write_str("incident cooldown capacity exceeded"),
            Self::CounterOverflow => formatter.write_str("incident suppression counter overflow"),
        }
    }
}

impl std::error::Error for IncidentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identity(error) => Some(error),
            Self::Serialization(error) => Some(error),
            _ => None,
        }
    }
}

pub struct IncidentManager {
    identities: EventIdentityStore,
    cooldown: CooldownState,
    last_audit_snapshot: Option<IncidentAuditSnapshot>,
}

impl IncidentManager {
    /// `limits.max_bytes` independently bounds the identity cache and the
    /// encoded cooldown table; it is not a combined budget or a heap ceiling.
    pub fn new(
        clock: Arc<dyn Clock>,
        ids: Arc<dyn IdSource>,
        limits: Limits,
    ) -> Result<Self, IncidentError> {
        Ok(Self {
            identities: EventIdentityStore::open(None, limits, clock, ids)
                .map_err(IncidentError::Identity)?,
            cooldown: CooldownState::new(limits.max_bytes),
            last_audit_snapshot: None,
        })
    }

    /// Uses only `now` for cooldown, without recapturing `Clock::monotonic`.
    /// Prunes expired keys first and refuses capacity before resolving an id.
    /// The delivery copy changes only identity; `source` is never mutated.
    pub fn admit(
        &mut self,
        source: &BusinessEvent,
        now: Duration,
    ) -> Result<Option<AdmittedIncident>, IncidentError> {
        self.cooldown.observe(now)?;
        self.cooldown.prune_expired(now);
        let key = CooldownKey::from(source);
        if self.cooldown.suppress(&key)? {
            return Ok(None);
        }
        let prepared = self.cooldown.prepare(&key, now)?;
        let source_key = state::source_key(source)?;
        let edge_event_id = self
            .identities
            .resolve(&source_key)
            .map_err(IncidentError::Identity)?;
        let audit = IncidentAuditSnapshot::capture(source, &key, &edge_event_id);
        let release = ReleaseReceipt { key: key.clone() };
        let mut event = source.clone();
        event.identity = edge_event_id;
        self.cooldown.commit(key, now, prepared);
        self.last_audit_snapshot = Some(audit.clone());
        Ok(Some(AdmittedIncident {
            event,
            source_key,
            audit,
            release,
        }))
    }

    /// Removes only the receipt's original key. Already absent is a no-op.
    /// Identity mappings and the last successful audit are left intact.
    pub fn release(&mut self, receipt: ReleaseReceipt) {
        self.cooldown.release(&receipt.key);
    }

    /// Clears cooldown/audit, preserving identity mappings, the suppression
    /// counter and the monotonic watermark. Pending admissions remain owned.
    pub fn reset(&mut self) {
        self.cooldown.reset();
        self.last_audit_snapshot = None;
    }

    pub fn last_audit_snapshot(&self) -> Option<&IncidentAuditSnapshot> {
        self.last_audit_snapshot.as_ref()
    }

    pub fn cooldown_suppressed_total(&self) -> u64 {
        self.cooldown.suppressed_total
    }

    pub fn accounted_bytes(&self) -> usize {
        self.cooldown.accounted_bytes
    }
}

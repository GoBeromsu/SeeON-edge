//! Python `RuntimeStatusSender._post` over `RelayRuntimeStatusTransport`:
//! `seq` rises before every attempt, `generation` echoes the last accepted
//! value, and an acceptance needs `accepted: true` with a non-negative
//! integer `generation`.

use std::collections::BTreeMap;

use crate::json::Json;
use crate::relay::RelayClient;
use crate::telemetry::{Publish, SendError};

use super::{FacilityStatus, RUNTIME_STATUS_PATH};

/// The accepted `generation`, or [`SendError::Malformed`] when the body is
/// not an object with `accepted: true` and a non-negative integer
/// `generation` (Python `MALFORMED_RESPONSE`).
pub fn parse_acceptance(body: &[u8]) -> Result<u64, SendError> {
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|_| SendError::Malformed)?;
    let Json::Object(members) = Json::from(&value) else {
        return Err(SendError::Malformed);
    };
    let field = |name: &str| {
        members
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    };
    if field("accepted") != Some(&Json::Bool(true)) {
        return Err(SendError::Malformed);
    }
    match field("generation") {
        Some(Json::Int(generation)) => u64::try_from(*generation).map_err(|_| SendError::Malformed),
        _ => Err(SendError::Malformed),
    }
}

type Source = Box<dyn FnMut() -> Vec<FacilityStatus> + Send>;

/// Posts every facility snapshot `source` yields, one request each.
pub struct StatusSender {
    client: RelayClient,
    source: Source,
    seq_by_facility: BTreeMap<String, u64>,
    generation_by_facility: BTreeMap<String, u64>,
}

impl StatusSender {
    pub fn new(
        client: RelayClient,
        source: impl FnMut() -> Vec<FacilityStatus> + Send + 'static,
    ) -> Self {
        Self {
            client,
            source: Box::new(source),
            seq_by_facility: BTreeMap::new(),
            generation_by_facility: BTreeMap::new(),
        }
    }

    /// The last `seq` stamped for `facility_id`, failed attempts included.
    pub fn seq(&self, facility_id: &str) -> Option<u64> {
        self.seq_by_facility.get(facility_id).copied()
    }

    /// The last `generation` the relay accepted for `facility_id`.
    pub fn generation(&self, facility_id: &str) -> Option<u64> {
        self.generation_by_facility.get(facility_id).copied()
    }

    /// One attempt for one facility; `seq` rises whatever the outcome.
    pub fn send(&mut self, status: &FacilityStatus) -> Result<u64, SendError> {
        let seq = self
            .seq_by_facility
            .entry(status.facility_id.clone())
            .or_insert(0);
        *seq = seq.saturating_add(1);
        let seq = *seq;
        let generation = self.generation(&status.facility_id);
        let body = status.body(seq, generation)?;
        let response = self.client.post_json(RUNTIME_STATUS_PATH, &body)?;
        if !(200..300).contains(&response.status) {
            return Err(SendError::Status(response.status));
        }
        let accepted = parse_acceptance(&response.body)?;
        self.generation_by_facility
            .insert(status.facility_id.clone(), accepted);
        Ok(accepted)
    }
}

impl Publish for StatusSender {
    /// Python `_post`: stops at the first failed facility.
    fn publish(&mut self) -> Result<(), SendError> {
        for status in (self.source)() {
            self.send(&status)?;
        }
        Ok(())
    }
}

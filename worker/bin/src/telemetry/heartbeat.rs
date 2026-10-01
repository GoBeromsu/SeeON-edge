//! The relay heartbeat (Python `worker.py:_RelayHeartbeat.mark_ready`): one
//! `{camera_id, facility_id, config_version}` POST per ready camera. Any
//! outcome other than 2xx is a failure that the loop counts; none raises.

use crate::json::{Json, Serialiser};
use crate::relay::RelayClient;

use super::{PayloadError, Publish, SendError};

/// Python `RELAY_HEARTBEAT_PATH`, relative to the relay base URL.
pub const HEARTBEAT_PATH: &str = "api/v1/relay/heartbeat";

/// One camera's heartbeat body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Heartbeat {
    pub camera_id: String,
    pub facility_id: String,
    pub config_version: i64,
}

impl Heartbeat {
    /// The wire object; blank ids are refused.
    pub fn payload(&self) -> Result<Json, PayloadError> {
        if self.camera_id.is_empty() {
            return Err(PayloadError::BlankId("camera_id"));
        }
        if self.facility_id.is_empty() {
            return Err(PayloadError::BlankId("facility_id"));
        }
        Ok(Json::Object(vec![
            ("camera_id".into(), Json::Str(self.camera_id.clone())),
            ("facility_id".into(), Json::Str(self.facility_id.clone())),
            (
                "config_version".into(),
                Json::Int(i128::from(self.config_version)),
            ),
        ]))
    }

    /// The request body: Python `encode_json(payload)`.
    pub fn body(&self) -> Result<Vec<u8>, PayloadError> {
        Ok(Serialiser::ModelSelection
            .canonical(&self.payload()?)?
            .into_bytes())
    }
}

type Source = Box<dyn FnMut() -> Vec<Heartbeat> + Send>;

/// Sends one heartbeat per camera `source` reports ready.
pub struct HeartbeatSender {
    client: RelayClient,
    source: Source,
}

impl HeartbeatSender {
    pub fn new(
        client: RelayClient,
        source: impl FnMut() -> Vec<Heartbeat> + Send + 'static,
    ) -> Self {
        Self {
            client,
            source: Box::new(source),
        }
    }

    fn send(&self, heartbeat: &Heartbeat) -> Result<(), SendError> {
        let response = self.client.post_json(HEARTBEAT_PATH, &heartbeat.body()?)?;
        if (200..300).contains(&response.status) {
            Ok(())
        } else {
            Err(SendError::Status(response.status))
        }
    }
}

impl Publish for HeartbeatSender {
    /// Attempts every ready camera, so one camera's failure does not starve
    /// the others, and reports the first failure.
    fn publish(&mut self) -> Result<(), SendError> {
        let mut first = None;
        for heartbeat in (self.source)() {
            if let Err(error) = self.send(&heartbeat) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

//! Python `worker/runtime/config/config_pull.py` (`_pull_payload`,
//! `load_worker_config_from_relay`, `_snapshot_from_stored`,
//! `pull_worker_config_poll`) and `release_pair.py`
//! (`require_api_release_identity`). Fresh versus last-known-good follows
//! `load_worker_config_from_relay`; the LKG is written only through
//! `config::lkg`. Python's `--config` YAML fallback is not carried, so no
//! relay and no LKG is a typed refusal.

use std::io::Read;
use std::path::Path;
use std::time::Duration;

use ureq::Agent;
use ureq::http::HeaderValue;

use crate::config::lkg::{LkgStore, StoredConfig};
use crate::config::restart::{RestartDirective, Roster};
use crate::config::{lookup, parse_json};
use crate::exit::Exit;
use crate::json::Json;
use crate::relay::cameras::{RuntimeCamera, WorkerConfigPayload};
use crate::relay::client::TOKEN_HEADER;
use crate::relay::wire::MAX_RESPONSE_BYTES;

/// Python `WORKER_CONFIG_PATH`, appended to the relay URL.
pub const WORKER_CONFIG_PATH: &str = "/api/v1/cameras/worker-config";
/// Python `require_api_release_identity` endpoint.
pub const RELEASE_IDENTITY_PATH: &str = "/health/release-identity";
/// Python `timeout_sec=5.0` of both pulls.
pub const CONFIG_PULL_TIMEOUT: Duration = Duration::from_secs(5);
/// Python `EDGE_DATABASE_SCHEMA_VERSION`.
pub const EDGE_DATABASE_SCHEMA_VERSION: i128 = 19;
/// Python `EDGE_DATABASE_FORMAT_IDENTITY`.
pub const EDGE_DATABASE_FORMAT_IDENTITY: &str = "seeon-edge-v1";
/// The schema Python assumes for a relay without the identity endpoint
/// (HTTP 404 gives `require_peer_schema_identity(17)`).
const PRE_IDENTITY_SCHEMA_VERSION: i128 = 17;

/// Python `ConfigSource` without `YAML`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigSource {
    Pulled,
    Lkg,
}

/// Python `ConfigSnapshot` for a relay config.
#[derive(Clone, Debug, PartialEq)]
pub struct PulledConfig {
    /// The payload as pulled or stored.
    pub payload: Json,
    pub config: WorkerConfigPayload,
    /// `to_worker_config` cameras; never empty when cameras were declared.
    pub cameras: Vec<RuntimeCamera>,
    pub directive: RestartDirective,
    pub source: ConfigSource,
    /// True for an LKG config.
    pub stale: bool,
}

impl PulledConfig {
    /// The roster the media sources open with.
    pub fn roster(&self) -> Roster {
        Roster::from_cameras(&self.cameras)
    }
}

/// Python `WorkerConfigPoll`: one periodic pull.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerConfigPoll {
    pub config: WorkerConfigPayload,
    pub cameras: Vec<RuntimeCamera>,
    pub directive: RestartDirective,
    pub clip_export_enabled: bool,
    pub clip_export_version: i128,
}

impl WorkerConfigPoll {
    /// What `RestartCheck::check` observes.
    pub fn restart_candidate(&self) -> (RestartDirective, Roster) {
        (self.directive, Roster::from_cameras(&self.cameras))
    }
}

/// Why startup refuses; `exit` gives the process exit code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PullError {
    /// `ReleaseIdentityMismatchError` for another schema version.
    SchemaMismatch { schema_version: i128 },
    /// The identity names another database format.
    FormatMismatch,
    /// A 2xx identity body Python cannot read (it raises and exits 1).
    MalformedReleaseIdentity,
    /// No valid fresh pull and no usable LKG.
    NoConfig,
}

impl PullError {
    pub fn exit(&self) -> Exit {
        match self {
            Self::SchemaMismatch { .. } | Self::FormatMismatch => Exit::RefuseToStart,
            Self::MalformedReleaseIdentity => Exit::Runtime,
            Self::NoConfig => Exit::Config,
        }
    }
}

/// Python `require_api_release_identity`: a transport failure or a non-2xx
/// other than 404 passes (Python catches `OSError`); 404 and a different
/// schema version or format refuse.
pub fn check_release_identity(relay_url: &str) -> Result<(), PullError> {
    let Some((status, body)) = fetch(relay_url, RELEASE_IDENTITY_PATH, None) else {
        return Ok(());
    };
    if status == 404 {
        return Err(PullError::SchemaMismatch {
            schema_version: PRE_IDENTITY_SCHEMA_VERSION,
        });
    }
    if !(200..300).contains(&status) {
        return Ok(());
    }
    let Some(Json::Object(members)) = parse_json(&body) else {
        return Err(PullError::MalformedReleaseIdentity);
    };
    let schema_version = match lookup(&members, "edge_database_schema_version") {
        Some(Json::Int(value)) => *value,
        Some(Json::Str(text)) => text
            .trim()
            .parse()
            .map_err(|_| PullError::MalformedReleaseIdentity)?,
        _ => return Err(PullError::MalformedReleaseIdentity),
    };
    if schema_version != EDGE_DATABASE_SCHEMA_VERSION {
        return Err(PullError::SchemaMismatch { schema_version });
    }
    match lookup(&members, "format") {
        None => Ok(()),
        Some(Json::Str(format)) if format == EDGE_DATABASE_FORMAT_IDENTITY => Ok(()),
        Some(_) => Err(PullError::FormatMismatch),
    }
}

/// Python `load_worker_config_from_relay`: a valid fresh pull wins unless a
/// strictly newer LKG exists and re-validates; with no valid pull the LKG
/// is used; with neither, `NoConfig`.
pub fn pull_startup_config(
    relay_url: &str,
    relay_token: &str,
    state_dir: &Path,
) -> Result<PulledConfig, PullError> {
    let store = LkgStore::new(state_dir);
    let fresh = fetch_worker_config(relay_url, relay_token)
        .and_then(|payload| snapshot(payload, ConfigSource::Pulled));
    if let Some(fresh) = fresh {
        if store.save(&fresh.payload, fresh.directive).unwrap_or(false) {
            return Ok(fresh);
        }
        return Ok(match store.load() {
            Ok(Some(stored)) if stored.directive > fresh.directive => {
                from_stored(stored).unwrap_or(fresh)
            }
            _ => fresh,
        });
    }
    match store.load() {
        Ok(Some(stored)) => from_stored(stored).ok_or(PullError::NoConfig),
        _ => Err(PullError::NoConfig),
    }
}

/// Python `pull_worker_config_poll`: `None` on any pull or payload failure.
pub fn poll_worker_config(relay_url: &str, relay_token: &str) -> Option<WorkerConfigPoll> {
    let pulled = snapshot(
        fetch_worker_config(relay_url, relay_token)?,
        ConfigSource::Pulled,
    )?;
    Some(WorkerConfigPoll {
        clip_export_enabled: pulled.config.clip_export_enabled(),
        clip_export_version: pulled.config.clip_export_version(),
        directive: pulled.directive,
        cameras: pulled.cameras,
        config: pulled.config,
    })
}

/// Python `_snapshot_from_stored`: the stored payload re-validated, and
/// its directive equal to the stored one.
fn from_stored(stored: StoredConfig) -> Option<PulledConfig> {
    let pulled = snapshot(stored.payload, ConfigSource::Lkg)?;
    (pulled.directive == stored.directive).then_some(pulled)
}

/// Python `_snapshot_from_payload`: `BackendWorkerConfigPayload` then
/// `to_worker_config`.
fn snapshot(payload: Json, source: ConfigSource) -> Option<PulledConfig> {
    let config = WorkerConfigPayload::parse(&payload).ok()?;
    let cameras = config.runtime_cameras().ok()?;
    Some(PulledConfig {
        directive: config.directive(),
        payload,
        config,
        cameras,
        source,
        stale: source == ConfigSource::Lkg,
    })
}

/// Python `_pull_payload`: only a 200 JSON object counts.
fn fetch_worker_config(relay_url: &str, relay_token: &str) -> Option<Json> {
    let (status, body) = fetch(relay_url, WORKER_CONFIG_PATH, Some(relay_token))?;
    match parse_json(&body)? {
        payload @ Json::Object(_) if status == 200 => Some(payload),
        _ => None,
    }
}

/// GET `relay_url` (trailing slashes dropped) plus `path` with
/// `Accept: application/json` and the token when given: no redirect, no
/// proxy, one 5 s deadline. `None` for any transport failure or a body
/// over `MAX_RESPONSE_BYTES`.
fn fetch(relay_url: &str, path: &str, token: Option<&str>) -> Option<(u16, Vec<u8>)> {
    let url = format!("{}{path}", relay_url.trim_end_matches('/'));
    let agent: Agent = Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(0)
        .timeout_global(Some(CONFIG_PULL_TIMEOUT))
        .proxy(None)
        .build()
        .into();
    let mut request = agent.get(url).header("Accept", "application/json");
    if let Some(token) = token.filter(|token| !token.is_empty()) {
        let mut value = HeaderValue::from_str(token).ok()?;
        value.set_sensitive(true);
        request = request.header(TOKEN_HEADER, value);
    }
    let response = request.call().ok()?;
    let status = response.status().as_u16();
    let limit = u64::try_from(MAX_RESPONSE_BYTES).ok()?;
    let mut body = Vec::new();
    response
        .into_body()
        .into_reader()
        .take(limit.saturating_add(1))
        .read_to_end(&mut body)
        .ok()?;
    (body.len() <= MAX_RESPONSE_BYTES).then_some((status, body))
}

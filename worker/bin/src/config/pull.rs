//! Python `worker/runtime/config/config_pull.py` (`_pull_payload`,
//! `load_worker_config_from_relay`, `_snapshot_from_stored`,
//! `pull_worker_config_poll`) and `release_pair.py`
//! (`require_api_release_identity`). Fresh versus last-known-good follows
//! `load_worker_config_from_relay`; the LKG is written only through
//! `config::lkg`. Python's `--config` YAML fallback is not carried, so no
//! relay and no LKG is a typed refusal.

use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Duration;

use rustix::fs::{FlockOperation, flock};
use ureq::Agent;
use ureq::http::HeaderValue;

use crate::config::lkg::{LkgError, LkgStore, StoredConfig};
use crate::config::restart::RestartDirective;
use crate::config::{lookup, parse_json};
use crate::exit::Exit;
use crate::json::Json;
use crate::relay::cameras::policies::{PolicyBundle, resolve_detection_policies};
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
    /// `resolved_detection_policies`: the parsed bundle, or the image
    /// default for every parsed camera when the payload carries none.
    pub policies: PolicyBundle,
    pub directive: RestartDirective,
    pub source: ConfigSource,
    /// True for an LKG config.
    pub stale: bool,
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
    /// What `RestartCheck::check` observes (Python
    /// `RestartDirective.from_pulled`).
    pub fn restart_candidate(&self) -> RestartDirective {
        self.directive
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
/// strictly newer LKG exists and re-validates; a strictly newer LKG that
/// fails re-validation is cleared and the fresh pull wins
/// (`config_pull.py:150-158`); with no valid pull the LKG is used; with
/// neither, `NoConfig`. Store failures are reported and treated as Python
/// treats `save` returning `False` and `load` returning `None`.
pub fn pull_startup_config(
    relay_url: &str,
    relay_token: &str,
    state_dir: &Path,
) -> Result<PulledConfig, PullError> {
    let store = LkgStore::new(state_dir);
    let directory = state_dir.join("config-lkg");
    let fresh = fetch_worker_config(relay_url, relay_token)
        .and_then(|payload| snapshot(payload, ConfigSource::Pulled));
    if let Some(fresh) = fresh {
        match store.save(&fresh.payload, fresh.directive) {
            Ok(true) => return Ok(fresh),
            Ok(false) => {}
            Err(error) => report_store_error(&directory, &error),
        }
        let stored = match store.load() {
            Ok(Some(stored)) if stored.directive > fresh.directive => stored,
            Ok(_) => return Ok(fresh),
            Err(error) => {
                report_store_error(&directory, &error);
                return Ok(fresh);
            }
        };
        if let Some(pulled) = from_stored(stored) {
            return Ok(pulled);
        }
        eprintln!(
            "ml-worker: WARNING: worker config LKG at {} failed re-validation; \
             deleting corrupt LKG and using fresh (race-losing) snapshot instead",
            directory.display()
        );
        clear_lkg(&directory);
        return Ok(fresh);
    }
    match store.load() {
        Ok(Some(stored)) => from_stored(stored).ok_or(PullError::NoConfig),
        Ok(None) => Err(PullError::NoConfig),
        Err(error) => {
            report_store_error(&directory, &error);
            Err(PullError::NoConfig)
        }
    }
}

/// Python `_report_unavailable` and the `save` payload message.
fn report_store_error(directory: &Path, error: &LkgError) {
    match error {
        LkgError::Io(error) => eprintln!(
            "ml-worker: worker config LKG store unavailable at {}: {error}",
            directory.display()
        ),
        LkgError::Payload => {
            eprintln!("ml-worker: worker config LKG payload unavailable: non-finite float");
        }
        LkgError::Record => eprintln!(
            "ml-worker: worker config LKG store unavailable at {}: malformed record",
            directory.display()
        ),
    }
}

/// Python `WorkerConfigLkgStore.clear` (`lkg_store.py:89-106`): under the
/// store's own `.lock` protocol (`_locked`), unlink `current.json` and
/// fsync the directory; revisions stay. `config::lkg` is outside this
/// module's ownership, so the protocol is repeated here.
fn clear_lkg(directory: &Path) {
    if !fs::metadata(directory).is_ok_and(|info| info.is_dir()) {
        return;
    }
    if let Err(error) = unlink_current(directory) {
        report_store_error(directory, &LkgError::Io(error));
    }
}

fn unlink_current(directory: &Path) -> io::Result<()> {
    let lock = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .mode(0o600)
        .open(directory.join(".lock"))?;
    flock(&lock, FlockOperation::LockExclusive)?;
    match fs::remove_file(directory.join("current.json")) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        removed => removed?,
    }
    File::open(directory)?.sync_all()
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
/// `to_worker_config`, whose `resolved_detection_policies` refuses a
/// malformed bundle (`pull_models.py:287-295`). A refused bundle is never
/// saved: a fresh pull falls back to the LKG, an LKG is skipped and a poll
/// yields no restart candidate. The refusal lines are the ones
/// `load_worker_config_from_relay` prints (`config_pull.py:135-136`, `167-168`).
fn snapshot(payload: Json, source: ConfigSource) -> Option<PulledConfig> {
    let config = WorkerConfigPayload::parse(&payload).ok()?;
    let cameras = config.runtime_cameras().ok()?;
    let camera_ids: Vec<String> = config
        .cameras()
        .into_iter()
        .map(|camera| camera.camera_id)
        .collect();
    let policies = match resolve_detection_policies(config.detection_policies(), &camera_ids) {
        Ok(policies) => policies,
        Err(error) => {
            match source {
                ConfigSource::Pulled => {
                    eprintln!("ml-worker: detection policy refused: {error}");
                }
                ConfigSource::Lkg => {
                    eprintln!("ml-worker: worker config LKG detection policy refused: {error}");
                }
            }
            return None;
        }
    };
    Some(PulledConfig {
        directive: config.directive(),
        payload,
        config,
        cameras,
        policies,
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

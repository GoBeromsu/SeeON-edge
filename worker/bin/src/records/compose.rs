//! Execution-record composition (`worker/runtime/execution_records.py`
//! `compose_execution_records`): settings, relay, provenance, then the lanes
//! and the exporter over one shared `Lanes`. Refusals keep the Python order:
//! settings first, then a blank relay url or token, then missing provenance.
//! Disabled records compose to nothing.

use std::fmt;
use std::sync::Arc;

use crate::config::env::{Env, EnvError, ExecutionRecordsSettings, execution_records};
use crate::records::exporter::{Exporter, ExporterError, RELAY_TIMEOUT};
use crate::records::lanes::{Lanes, LanesError};
use crate::records::provenance::{Identities, ProvenanceError, build_provenance};
use crate::relay::{ConfigError, RelayClient};

/// The composed pipeline: producers emit into `lanes`, the stage 4 loop
/// drives `exporter`.
pub struct Composed {
    pub settings: ExecutionRecordsSettings,
    pub lanes: Arc<Lanes>,
    pub exporter: Exporter,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComposeError {
    /// Python `WorkerConfigError` from the execution-record settings.
    Settings(EnvError),
    /// Python `ExecutionRecordProvenanceError`: records are enabled but the
    /// relay url or token is blank.
    RelayMissing,
    /// Python `ExecutionRecordProvenanceError` from `build_wire_provenance`.
    Provenance(ProvenanceError),
    /// Python `ValueError` from the relay client (`normalize_http_base`).
    Relay(ConfigError),
    /// Unreachable with settings the environment accepted; kept typed.
    Lanes(LanesError),
    /// Unreachable with settings the environment accepted; kept typed.
    Exporter(ExporterError),
}

impl fmt::Display for ComposeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Settings(_) => formatter.write_str("execution-record settings are invalid"),
            Self::RelayMissing => {
                formatter.write_str("execution records enabled but relay URL/token missing")
            }
            Self::Provenance(error) => error.fmt(formatter),
            Self::Relay(error) => error.fmt(formatter),
            Self::Lanes(error) => error.fmt(formatter),
            Self::Exporter(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ComposeError {}

/// Python `str.strip()`: Unicode whitespace plus the separators U+001C..=U+001F.
fn strip(raw: &str) -> &str {
    raw.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

/// `compose_execution_records`. `config_digest` identifies the effective
/// worker configuration; the caller chooses its source.
pub fn compose(
    env: &Env,
    relay_url: &str,
    relay_token: &str,
    identities: &Identities,
    config_digest: &str,
) -> Result<Option<Composed>, ComposeError> {
    let Some(settings) = execution_records(env).map_err(ComposeError::Settings)? else {
        return Ok(None);
    };
    let token = strip(relay_token);
    if strip(relay_url).is_empty() || token.is_empty() {
        return Err(ComposeError::RelayMissing);
    }
    let provenance =
        build_provenance(identities, config_digest).map_err(ComposeError::Provenance)?;
    let lanes = Arc::new(Lanes::new(settings.lane_capacity).map_err(ComposeError::Lanes)?);
    let client = RelayClient::new(relay_url, token, RELAY_TIMEOUT).map_err(ComposeError::Relay)?;
    let exporter = Exporter::new(
        Arc::clone(&lanes),
        client,
        provenance,
        settings.batch_max,
        settings.flush_ms,
    )
    .map_err(ComposeError::Exporter)?;
    Ok(Some(Composed {
        settings,
        lanes,
        exporter,
    }))
}

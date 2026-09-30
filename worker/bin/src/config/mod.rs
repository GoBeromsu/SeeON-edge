//! What `ml-worker check-config` proves before any GPU, camera or relay is
//! touched: the env (`env`), the model selection (`selection`), the admitted
//! bundle and engine identity (`models`) and the last-known-good worker
//! config (`lkg`). Everything here reads files only; the process env arrives
//! as a map.

pub mod env;
pub mod lkg;
pub mod models;
pub mod selection;

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use rustix::io::Errno;

use crate::exit::Exit;
use crate::json::{Json, Serialiser};
use env::{EnvError, ExecutionRecordsSettings};
use lkg::{LkgError, LkgStore, StoredConfig};
use models::AdmissionError;
use models::bundle::{BundleProof, admit_model_bundle};
use models::identity::{FLOW_IDENTITY_FILES, IdentityError, verify_engine_identity};
use selection::{ModelSelection, SelectionError, parse_model_selection};

/// Moves the fall selection document (Python fixes `FALL_SELECTION_PATH`).
pub const MODEL_SELECTION_PATH_ENV: &str = "ML_WORKER_MODEL_SELECTION_PATH";
/// Moves the models root (Python fixes `FALL_MODELS_ROOT`).
pub const MODELS_ROOT_ENV: &str = "ML_WORKER_MODELS_ROOT";
const DEFAULT_SELECTION_PATH: &str = "/app/model-selection.json";
const DEFAULT_MODELS_ROOT: &str = "/models";
const FLOW_ENGINE_ENV: &str = "ML_WORKER_FLOW_ENGINE_PATH";
const FLOW_ENGINE_IDENTITY_ENV: &str = "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH";

/// The first entry named `key` of a parsed object.
pub(crate) fn lookup<'a>(members: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    members
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

/// Exactly `len` lowercase hex digits.
pub(crate) fn is_hex(text: &str, len: usize) -> bool {
    text.len() == len
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// One path segment: `[A-Za-z0-9][A-Za-z0-9._-]*`.
pub(crate) fn is_segment(text: &str) -> bool {
    let mut bytes = text.bytes();
    bytes
        .next()
        .is_some_and(|first| first.is_ascii_alphanumeric())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// `json.loads` of a file body; `None` for anything Python would refuse
/// (and for `NaN`/`Infinity`, which no checked document may carry).
pub(crate) fn parse_json(raw: &[u8]) -> Option<Json> {
    let value: serde_json::Value = serde_json::from_slice(raw).ok()?;
    Some(Json::from(&value))
}

/// Why the fall selection document was refused (`invalid fall selection`).
#[derive(Clone, Debug, PartialEq)]
pub enum SelectionLoadError {
    /// The document exists but cannot be read.
    Unreadable,
    /// Not JSON, or not byte-equal to its own canonical encoding.
    NotCanonical,
    /// `desired_model_bundle_from_selection_document` refused it.
    Refused(SelectionError),
}

/// A check-config refusal; `exit` gives its process exit code.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckConfigError {
    Env(EnvError),
    Selection(SelectionLoadError),
    Admission(AdmissionError),
    Identity(IdentityError),
}

impl CheckConfigError {
    /// Env and selection errors are configuration errors (2); a bundle or
    /// engine that does not match its proof refuses to start (3).
    pub const fn exit(&self) -> Exit {
        match self {
            Self::Env(_) | Self::Selection(_) => Exit::Config,
            Self::Admission(_) | Self::Identity(_) => Exit::RefuseToStart,
        }
    }
}

/// What a passing check-config established.
#[derive(Debug)]
pub struct Checked {
    pub state_dir: PathBuf,
    pub execution_records: Option<ExecutionRecordsSettings>,
    /// `None` when no selection document exists (packaged model stays active).
    pub selection: Option<(ModelSelection, BundleProof)>,
    /// `None` when the flow engine is not wired.
    pub engine_identity: Option<BTreeMap<String, String>>,
    /// An unreadable store is reported, never fatal (Python falls back).
    pub last_known_good: Result<Option<StoredConfig>, LkgError>,
}

fn env_path(env: &BTreeMap<String, String>, key: &str, default: &str) -> PathBuf {
    PathBuf::from(env.get(key).map_or(default, String::as_str))
}

/// `selected_fall_bundle_config_from_environment`: an absent document is
/// `None`; anything else must be canonical JSON naming a valid selection.
fn load_selection(path: &Path) -> Result<Option<ModelSelection>, SelectionLoadError> {
    let raw = match fs::read(path) {
        Ok(raw) => raw,
        // `Path.exists()` is false on ENOENT, ENOTDIR and ELOOP.
        Err(error)
            if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory)
                || Errno::from_io_error(&error) == Some(Errno::LOOP) =>
        {
            return Ok(None);
        }
        Err(_) => return Err(SelectionLoadError::Unreadable),
    };
    let document = parse_json(&raw).ok_or(SelectionLoadError::NotCanonical)?;
    let canonical = Serialiser::FetchModelsManifest.canonical(&document);
    if canonical.ok().as_deref().map(str::as_bytes) != Some(raw.as_slice()) {
        return Err(SelectionLoadError::NotCanonical);
    }
    parse_model_selection(&document)
        .map(Some)
        .map_err(SelectionLoadError::Refused)
}

/// `verify_engine_identity` over the flow wiring, when any of it is set;
/// a partly wired flow refuses on the first absent file.
fn engine_identity(
    env: &BTreeMap<String, String>,
) -> Result<Option<BTreeMap<String, String>>, IdentityError> {
    let artifact_keys = FLOW_IDENTITY_FILES.iter().map(|(_, name)| *name);
    let mut wiring = [FLOW_ENGINE_ENV, FLOW_ENGINE_IDENTITY_ENV]
        .into_iter()
        .chain(artifact_keys);
    if !wiring.any(|name| env.get(name).is_some_and(|value| !value.is_empty())) {
        return Ok(None);
    }
    let files: Vec<(&str, PathBuf)> = FLOW_IDENTITY_FILES
        .iter()
        .map(|(key, name)| (*key, env_path(env, name, "")))
        .collect();
    let engine = env_path(env, FLOW_ENGINE_ENV, "");
    let identity = env_path(env, FLOW_ENGINE_IDENTITY_ENV, "");
    verify_engine_identity(&engine, &identity, &files, None).map(Some)
}

/// `ml-worker check-config`: design §2.4 steps 1 and 3, file-only.
pub fn check_config(
    env: &BTreeMap<String, String>,
    explicit_state_dir: Option<&Path>,
) -> Result<Checked, CheckConfigError> {
    let state_dir = env::state_dir(env, explicit_state_dir).map_err(CheckConfigError::Env)?;
    env::reject_retired(env).map_err(CheckConfigError::Env)?;
    env::relay_token(env).map_err(CheckConfigError::Env)?;
    let execution_records = env::execution_records(env).map_err(CheckConfigError::Env)?;
    let selection_path = env_path(env, MODEL_SELECTION_PATH_ENV, DEFAULT_SELECTION_PATH);
    let selection = match load_selection(&selection_path).map_err(CheckConfigError::Selection)? {
        None => None,
        Some(desired) => {
            let models_root = env_path(env, MODELS_ROOT_ENV, DEFAULT_MODELS_ROOT);
            let proof =
                admit_model_bundle(&models_root, &desired).map_err(CheckConfigError::Admission)?;
            Some((desired, proof))
        }
    };
    let engine_identity = engine_identity(env).map_err(CheckConfigError::Identity)?;
    let last_known_good = LkgStore::new(&state_dir).load();
    Ok(Checked {
        state_dir,
        execution_records,
        selection,
        engine_identity,
        last_known_good,
    })
}

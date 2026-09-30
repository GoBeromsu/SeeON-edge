//! The worker env as a map (never the process env): `config/local_env.py`,
//! `config/execution_records.py`, the `RELAY_TOKEN` rule and the state dir.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `_RETIRED_WORKER_ENV`, sorted. Presence alone refuses, even when empty.
pub const RETIRED_KEYS: [&str; 18] = [
    "CLIP_STORE_DIR",
    "EDGE_CAMERA_CONFIG",
    "EDGE_CAMERA_CONFIG_FILE",
    "ML_WORKER_CLIP_RECORDING_ENABLED",
    "ML_WORKER_DEV_MJPEG",
    "ML_WORKER_DEV_MJPEG_HOST",
    "ML_WORKER_DEV_MJPEG_PORT",
    "ML_WORKER_EVENT_CLIP_EXPORT_ENABLED",
    "ML_WORKER_FALL_MODEL_ARCHITECTURE",
    "ML_WORKER_FALL_MODEL_ARTIFACT_DIR",
    "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD",
    "ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY",
    "ML_WORKER_FALL_MODEL_SCHEMA_VERSION",
    "ML_WORKER_FALL_MODEL_STRIDE",
    "ML_WORKER_FALL_MODEL_TYPE",
    "ML_WORKER_FALL_MODEL_WEIGHTS",
    "ML_WORKER_FALL_MODEL_WINDOW",
    "RELAY_URL",
];

const RECORDS_ENABLED: &str = "ML_WORKER_EXECUTION_RECORDS_ENABLED";
/// Checked in this order once the lane is enabled.
const RECORDS_SIZING: [&str; 3] = [
    "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY",
    "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX",
    "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS",
];
const RELAY_TOKEN: &str = "RELAY_TOKEN";
const REPLAY_TRACE_DIR: &str = "WORKER_REPLAY_TRACE_DIR";
const HOME: &str = "HOME";
const STATE_BELOW_HOME: &str = ".local/state/ml-worker";
/// `sys.int_info.default_max_str_digits`: more digits fail `int()`.
const MAX_INT_DIGITS: usize = 4300;

/// Which env rule refused, one per Python `WorkerConfigError` message.
/// `Home` has no Python peer: without `--state-dir` the default needs `HOME`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnvKind {
    Retired,
    NotBoolean,
    Required,
    NotInteger,
    NotPositive,
    RelayToken,
    Home,
}

/// An env refusal and the keys it names, in the Python message order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvError {
    pub kind: EnvKind,
    pub keys: Vec<String>,
}

pub type Env = BTreeMap<String, String>;

fn refuse<T>(kind: EnvKind, key: &str) -> Result<T, EnvError> {
    let keys = vec![key.to_owned()];
    Err(EnvError { kind, keys })
}

/// `ExecutionRecordsSettings` of the enabled lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionRecordsSettings {
    pub lane_capacity: u64,
    pub batch_max: u64,
    pub flush_ms: u64,
}

/// `reject_retired_worker_environment`: names every retired key present.
pub fn reject_retired(env: &Env) -> Result<(), EnvError> {
    let keys: Vec<String> = RETIRED_KEYS
        .iter()
        .filter(|key| env.contains_key(**key))
        .map(|key| (*key).to_owned())
        .collect();
    let kind = EnvKind::Retired;
    if keys.is_empty() {
        return Ok(());
    }
    Err(EnvError { kind, keys })
}

/// Python `str.strip()`: Unicode whitespace plus `\x1c`-`\x1f`.
fn strip(raw: Option<&String>) -> &str {
    let raw = raw.map_or("", String::as_str);
    raw.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

/// Python `int(text)` of a stripped string: optional sign, ASCII digits with
/// single `_` between them. `Some(None)` is an integer below 1; values beyond
/// `u64` saturate.
fn python_int(text: &str) -> Option<Option<u64>> {
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let runs_ok = |run: &str| !run.is_empty() && run.bytes().all(|byte| byte.is_ascii_digit());
    let plain: String = digits.chars().filter(|c| *c != '_').collect();
    if !digits.split('_').all(runs_ok) || plain.len() > MAX_INT_DIGITS {
        return None;
    }
    let value = plain.parse().unwrap_or(u64::MAX);
    Some((!negative && value >= 1).then_some(value))
}

/// `_bool_env`: unset or blank is `None`.
fn bool_env(env: &Env, key: &str) -> Result<Option<bool>, EnvError> {
    match strip(env.get(key)).to_lowercase().as_str() {
        "" => Ok(None),
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => refuse(EnvKind::NotBoolean, key),
    }
}

/// `_required_positive_int`.
fn required_positive_int(env: &Env, key: &str) -> Result<u64, EnvError> {
    let raw = strip(env.get(key));
    match python_int(raw) {
        _ if raw.is_empty() => refuse(EnvKind::Required, key),
        None => refuse(EnvKind::NotInteger, key),
        Some(None) => refuse(EnvKind::NotPositive, key),
        Some(Some(value)) => Ok(value),
    }
}

/// `execution_records_settings_from_environment`: `None` unless enabled.
pub fn execution_records(env: &Env) -> Result<Option<ExecutionRecordsSettings>, EnvError> {
    if bool_env(env, RECORDS_ENABLED)? != Some(true) {
        return Ok(None);
    }
    let [lane, batch, flush] = RECORDS_SIZING;
    Ok(Some(ExecutionRecordsSettings {
        lane_capacity: required_positive_int(env, lane)?,
        batch_max: required_positive_int(env, batch)?,
        flush_ms: required_positive_int(env, flush)?,
    }))
}

/// `RELAY_TOKEN` must be non-blank; check-config never contacts the relay.
pub fn relay_token(env: &Env) -> Result<&str, EnvError> {
    match strip(env.get(RELAY_TOKEN)) {
        "" => refuse(EnvKind::RelayToken, RELAY_TOKEN),
        token => Ok(token),
    }
}

/// `resolve_state_dir`: `--state-dir`, else `$HOME/.local/state/ml-worker`.
pub fn state_dir(env: &Env, explicit: Option<&Path>) -> Result<PathBuf, EnvError> {
    match (explicit, env.get(HOME)) {
        (Some(path), _) => Ok(path.to_path_buf()),
        (None, Some(home)) if !home.is_empty() => Ok(Path::new(home).join(STATE_BELOW_HOME)),
        (None, _) => refuse(EnvKind::Home, HOME),
    }
}

/// `replay_trace_directory_from_environment`: the opt-in replay trace
/// directory, stripped; unset or blank is `None`.
pub fn replay_trace_dir(env: &Env) -> Option<PathBuf> {
    match strip(env.get(REPLAY_TRACE_DIR)) {
        "" => None,
        raw => Some(PathBuf::from(raw)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const GOLDEN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/worker-wire/d/env-refusals.json"
    );

    /// Reviewed map from the Python `WorkerConfigError` message to the kind
    /// and the keys it names.
    fn expected_refusal(message: &str) -> (EnvKind, Vec<String>) {
        if let Some(rest) = message.strip_prefix("retired edge environment key(s): ") {
            let listed = rest.split("; use").next().unwrap_or_default();
            let keys = listed.split(", ").map(str::to_owned).collect();
            return (EnvKind::Retired, keys);
        }
        let table = [
            ("must be a boolean", EnvKind::NotBoolean),
            ("is required when", EnvKind::Required),
            ("must be a positive integer", EnvKind::NotPositive),
            ("must be an integer", EnvKind::NotInteger),
        ];
        let found = table.into_iter().find(|(text, _)| message.contains(text));
        let (_, kind) = found.unwrap_or_else(|| panic!("unmapped message {message:?}"));
        let key = message.split(' ').next().unwrap_or_default();
        (kind, vec![key.to_owned()])
    }

    fn settings(value: &Value) -> Option<ExecutionRecordsSettings> {
        let field = |key: &str| value[key].as_u64().expect("golden settings field");
        (!value.is_null()).then(|| ExecutionRecordsSettings {
            lane_capacity: field("lane_capacity"),
            batch_max: field("batch_max"),
            flush_ms: field("flush_ms"),
        })
    }

    /// Runs every case; returns how many there were.
    fn check_cases<T>(
        cases: &Value,
        run: impl Fn(&BTreeMap<String, String>) -> Result<T, EnvError>,
        accepted: impl Fn(&Value, T),
    ) -> usize {
        let cases = cases.as_array().expect("golden case list");
        for case in cases {
            let env: BTreeMap<String, String> =
                serde_json::from_value(case["env"].clone()).expect("golden env map");
            let name = &case["name"];
            match (case["verdict"].as_str(), run(&env)) {
                (Some("accepted"), Ok(value)) => accepted(case, value),
                (Some("refused"), Err(error)) => {
                    let expected = expected_refusal(case["message"].as_str().expect("message"));
                    assert_eq!((error.kind, error.keys), expected, "case {name}");
                }
                (verdict, outcome) => panic!("case {name}: {verdict:?} vs {:?}", outcome.err()),
            }
        }
        cases.len()
    }

    #[test]
    fn env_refusals_match_python() {
        let raw = std::fs::read(GOLDEN).expect("env-refusals golden");
        let golden: Value = serde_json::from_slice(&raw).expect("golden JSON");
        let retired: Vec<&str> = golden["retired_keys"]
            .as_array()
            .expect("retired_keys")
            .iter()
            .map(|key| key.as_str().expect("retired key"))
            .collect();
        assert_eq!(retired, RETIRED_KEYS);
        let retired_cases = check_cases(&golden["retired_key_cases"], reject_retired, |_, ()| {});
        let execution_cases = check_cases(
            &golden["execution_records_cases"],
            execution_records,
            |case, value| assert_eq!(value, settings(&case["settings"]), "{}", case["name"]),
        );
        assert!(retired_cases > 0 && execution_cases > 0);
    }

    /// Oracle: `local_env.py` L487-493 at 030aaf1,
    /// `raw = env.get("WORKER_REPLAY_TRACE_DIR", "").strip()` then
    /// `None if not raw else Path(raw)`.
    #[test]
    fn replay_trace_dir_strips_as_python() {
        let env = |raw: &str| Env::from([(REPLAY_TRACE_DIR.to_owned(), raw.to_owned())]);
        assert_eq!(replay_trace_dir(&Env::new()), None);
        for blank in ["", " \t\n\x1c\x1f\u{3000} "] {
            assert_eq!(replay_trace_dir(&env(blank)), None, "{blank:?}");
        }
        let padded = replay_trace_dir(&env("  /tmp/trace \n"));
        assert_eq!(padded, Some(PathBuf::from("/tmp/trace")));
    }
}

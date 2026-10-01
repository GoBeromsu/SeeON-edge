//! Environment policy and file-only admission; admission runs only under the lease.

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use crate::cli::{self, Flags};
use crate::config::env::{self, Env, EnvError, ExecutionRecordsSettings};
use crate::config::model_bundle::flow_boot;
use crate::config::{self, CheckConfigError};
use crate::exit::Exit;

use super::{Admitted, EngineFiles, FlowSettings, IdentityError, ModelEngines};

/// Explicit deployment/model policy; no device, threshold or timing fallback.
#[derive(Clone, Copy, Debug)]
pub struct BootPolicy {
    pub device_ordinal: i32,
    pub stored_pose_threshold: f64,
    pub deployed_batch: Option<i128>,
    /// One total readiness budget shared by all three model owners.
    pub readiness_budget: Duration,
}

/// Safe diagnostics: never include argument values, addresses or credentials.
#[derive(Clone, Debug, PartialEq)]
pub enum SettingsError {
    Flags,
    Environment(EnvError),
    Required(&'static str),
    Policy(&'static str),
}

impl SettingsError {
    pub const fn exit(&self) -> Exit {
        Exit::Config
    }
}

/// Validated step-1 inputs. The retained environment is deliberately not Debug.
pub struct Settings {
    pub(crate) env: Env,
    pub(crate) flags: Flags,
    pub(crate) state_dir: PathBuf,
    pub(crate) execution_records: Option<ExecutionRecordsSettings>,
    pub(crate) engines: EngineFiles,
    pub(crate) policy: BootPolicy,
}

impl Settings {
    /// Arguments are flags after the parent-owned `run`/default dispatch.
    pub fn parse(
        env: Env,
        arguments: &[OsString],
        policy: BootPolicy,
    ) -> Result<Self, SettingsError> {
        let flags = cli::parse_flags(arguments).map_err(|_| SettingsError::Flags)?;
        Self::from_flags(env, flags, policy)
    }

    pub fn from_flags(env: Env, flags: Flags, policy: BootPolicy) -> Result<Self, SettingsError> {
        let state_dir =
            env::state_dir(&env, flags.state_dir.as_deref()).map_err(SettingsError::Environment)?;
        env::reject_retired(&env).map_err(SettingsError::Environment)?;
        env::relay_token(&env).map_err(SettingsError::Environment)?;
        let execution_records = env::execution_records(&env).map_err(SettingsError::Environment)?;
        if env.get("ML_WORKER_PROFILE").is_some_and(|raw| {
            let name =
                raw.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c));
            !name.is_empty() && name != "flow"
        }) {
            return Err(SettingsError::Policy("ML_WORKER_PROFILE"));
        }
        if policy.device_ordinal < 0 {
            return Err(SettingsError::Policy("device_ordinal"));
        }
        if !policy.stored_pose_threshold.is_finite()
            || !(0.0..=1.0).contains(&policy.stored_pose_threshold)
        {
            return Err(SettingsError::Policy("stored_pose_threshold"));
        }
        if policy.deployed_batch.is_some_and(|batch| batch < 0) {
            return Err(SettingsError::Policy("deployed_batch"));
        }
        if policy.readiness_budget.is_zero() {
            return Err(SettingsError::Policy("readiness_budget"));
        }
        let required = |key| match env.get(key) {
            Some(raw) if !raw.trim().is_empty() => Ok(PathBuf::from(raw)),
            _ => Err(SettingsError::Required(key)),
        };
        // Names approved in design §6.3; frozen config/env has no parser for these yet.
        let engines = EngineFiles {
            fall: required("ML_WORKER_FALL_ENGINE_PATH")?,
            bed: required("ML_WORKER_BED_ENGINE_PATH")?,
            stored_pose: required("ML_WORKER_STORED_POSE_ENGINE_PATH")?,
        };
        Ok(Self {
            env,
            flags,
            state_dir,
            execution_records,
            engines,
            policy,
        })
    }

    pub fn flags(&self) -> &Flags {
        &self.flags
    }
    pub fn state_dir(&self) -> &std::path::Path {
        &self.state_dir
    }
    pub fn policy(&self) -> BootPolicy {
        self.policy
    }
    pub fn execution_records(&self) -> Option<ExecutionRecordsSettings> {
        self.execution_records
    }
    pub fn relay_token(&self) -> Result<&str, EnvError> {
        env::relay_token(&self.env)
    }
    pub fn replay_trace_dir(&self) -> Option<PathBuf> {
        env::replay_trace_dir(&self.env)
    }
}

/// Step 3, called after lease acquisition. No network, CUDA or owner creation.
pub(crate) fn admit(settings: &Settings) -> Result<Admitted, IdentityError> {
    let checked = config::check_config(&settings.env, Some(&settings.state_dir)).map_err(
        |error| match error {
            CheckConfigError::Env(_) => IdentityError::Environment,
            CheckConfigError::Selection(_) => IdentityError::Selection,
            CheckConfigError::Admission(error) => IdentityError::Bundle(error.kind),
            CheckConfigError::Identity(error) => IdentityError::Engine(error.kind),
        },
    )?;
    let env = &settings.env;
    let identity = flow_boot::verify_flow_boot_inputs(env, settings.policy.deployed_batch)
        .map_err(|error| IdentityError::Flow(error.kind))?;
    let rtsp_reconnect_interval_sec = flow_boot::rtsp_reconnect_interval_sec(env)
        .map_err(|error| IdentityError::Flow(error.kind))?;
    let path = |key| {
        env.get(key)
            .map(PathBuf::from)
            .ok_or(IdentityError::FlowValue(key))
    };
    let positive = |key| {
        let raw = env.get(key).ok_or(IdentityError::FlowValue(key))?;
        positive_u32(raw).ok_or(IdentityError::FlowValue(key))
    };
    let batch_size = identity
        .get("batch_size")
        .and_then(|raw| raw.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .ok_or(IdentityError::FlowValue("ML_WORKER_FLOW_BATCH_SIZE"))?;
    let flow = FlowSettings {
        engine: path("ML_WORKER_FLOW_ENGINE_PATH")?,
        identity_path: path("ML_WORKER_FLOW_ENGINE_IDENTITY_PATH")?,
        infer_config: path("ML_WORKER_FLOW_INFER_CONFIG")?,
        tracker_config: path("ML_WORKER_FLOW_TRACKER_CONFIG")?,
        tracker_library: path("ML_WORKER_FLOW_TRACKER_LIBRARY")?,
        onnx: path("ML_WORKER_FLOW_ONNX_PATH")?,
        parser_library: path("ML_WORKER_FLOW_PARSER_LIBRARY")?,
        record_dir: path("ML_WORKER_FLOW_RECORD_DIR")?,
        record_cache_seconds: positive("ML_WORKER_FLOW_RECORD_CACHE_SECONDS")?,
        frame_width: positive("ML_WORKER_FLOW_FRAME_WIDTH")?,
        frame_height: positive("ML_WORKER_FLOW_FRAME_HEIGHT")?,
        batch_size,
        rtsp_reconnect_interval_sec,
        identity,
    };
    let engines = ModelEngines::admit(&settings.engines).map_err(IdentityError::Model)?;
    Ok(Admitted {
        checked,
        flow,
        engines,
    })
}

// Python int-compatible ASCII spelling, narrowed to the native u32 ABI.
fn positive_u32(raw: &str) -> Option<u32> {
    let trimmed = raw.trim_matches(char::is_whitespace);
    let digits = match trimmed.strip_prefix('+') {
        Some(rest) => rest,
        None => trimmed,
    };
    if !digits
        .split('_')
        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    let mut value = 0u32;
    let mut count = 0usize;
    for byte in digits.bytes().filter(|byte| *byte != b'_') {
        count += 1;
        if count > 4300 {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(u32::from(byte - b'0'))?;
    }
    (value > 0).then_some(value)
}

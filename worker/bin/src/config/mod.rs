//! What `ml-worker check-config` proves before any GPU, camera or relay is
//! touched: the env (`env`), the model selection (`selection`), the admitted
//! bundle and engine identity (`model_bundle`) and the last-known-good worker
//! config (`lkg`). Everything here reads files only; the process env arrives
//! as a map.

pub(crate) mod build_revision;
pub mod env;
pub mod lkg;
pub mod model_bundle;
pub mod pull;
pub mod restart;
pub mod selection;
pub mod windows;

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use rustix::io::Errno;

use crate::exit::Exit;
use crate::json::{Json, Serialiser};
use env::{EnvError, ExecutionRecordsSettings};
use lkg::{LkgError, LkgStore, StoredConfig};
use model_bundle::AdmissionError;
use model_bundle::bundle::{BundleProof, admit_model_bundle};
use model_bundle::identity::{
    AuxiliaryRuntime, CpuModelHashes, FLOW_IDENTITY_FILES, IdentityError, VerifiedEnvironment,
    verify_environment,
};
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
    /// Exact verified CPU source hashes; `None` for TensorRT or unwired Flow.
    pub cpu_model_hashes: Option<CpuModelHashes>,
    /// An unreadable store is reported, never fatal (Python falls back).
    pub last_known_good: Result<Option<StoredConfig>, LkgError>,
}

fn env_path(env: &BTreeMap<String, String>, key: &str, default: &str) -> PathBuf {
    PathBuf::from(env.get(key).map_or(default, String::as_str))
}

pub(crate) fn models_root(env: &env::Env) -> PathBuf {
    env_path(env, MODELS_ROOT_ENV, DEFAULT_MODELS_ROOT)
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

/// A wired deployment must admit the explicitly selected provider's identity.
fn engine_identity(
    env: &BTreeMap<String, String>,
    selection: Option<&(ModelSelection, BundleProof)>,
    auxiliary_runtime: AuxiliaryRuntime,
) -> Result<Option<VerifiedEnvironment>, IdentityError> {
    let artifact_keys = FLOW_IDENTITY_FILES.iter().map(|(_, name)| *name);
    let auxiliary_keys: &[&str] = match auxiliary_runtime {
        AuxiliaryRuntime::TensorRt => &[
            "ML_WORKER_STORED_POSE_ENGINE_PATH",
            "ML_WORKER_BED_ENGINE_PATH",
            "ML_WORKER_FALL_ENGINE_PATH",
        ],
        AuxiliaryRuntime::OnnxRuntimeCpu => &[],
    };
    let mut wiring = [
        FLOW_ENGINE_ENV,
        FLOW_ENGINE_IDENTITY_ENV,
        "ML_WORKER_FLOW_ONNX_PATH",
    ]
    .into_iter()
    .chain(auxiliary_keys.iter().copied())
    .chain(artifact_keys);
    if !wiring.any(|name| env.get(name).is_some_and(|value| !value.is_empty())) {
        return Ok(None);
    }
    verify_environment(env, selection, None, auxiliary_runtime).map(Some)
}

/// Selected-model document plus bundle proof, without relay, state, or LKG.
/// An absent document is `None`. Present documents keep the check-config
/// selection-then-admission refusal order.
pub(crate) fn admit_selected_bundle(
    env: &env::Env,
) -> Result<Option<(ModelSelection, BundleProof)>, CheckConfigError> {
    let selection_path = env_path(env, MODEL_SELECTION_PATH_ENV, DEFAULT_SELECTION_PATH);
    match load_selection(&selection_path).map_err(CheckConfigError::Selection)? {
        None => Ok(None),
        Some(desired) => {
            let models_root = models_root(env);
            let proof =
                admit_model_bundle(&models_root, &desired).map_err(CheckConfigError::Admission)?;
            Ok(Some((desired, proof)))
        }
    }
}
/// `ml-worker check-config`: design §2.4 steps 1 and 3, file-only.
pub fn check_config(
    env: &BTreeMap<String, String>,
    explicit_state_dir: Option<&Path>,
    auxiliary_runtime: AuxiliaryRuntime,
) -> Result<Checked, CheckConfigError> {
    let state_dir = env::state_dir(env, explicit_state_dir).map_err(CheckConfigError::Env)?;
    env::reject_retired(env).map_err(CheckConfigError::Env)?;
    env::relay_token(env).map_err(CheckConfigError::Env)?;
    let execution_records = env::execution_records(env).map_err(CheckConfigError::Env)?;
    let selection = admit_selected_bundle(env)?;
    let (engine_identity, cpu_model_hashes) =
        match engine_identity(env, selection.as_ref(), auxiliary_runtime)
            .map_err(CheckConfigError::Identity)?
        {
            Some(verified) => (Some(verified.flow), verified.cpu_model_hashes),
            None => (None, None),
        };
    let last_known_good = LkgStore::new(&state_dir).load();
    Ok(Checked {
        state_dir,
        execution_records,
        selection,
        engine_identity,
        cpu_model_hashes,
        last_known_good,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::{IdSource, RandomIds};
    use selection::Publication;

    struct Fixture {
        root: PathBuf,
        state: PathBuf,
        selection: PathBuf,
        env: env::Env,
    }

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                ".config-admission-test-{}",
                RandomIds.uuid4().expect("id")
            ));
            fs::create_dir(&root).expect("owned scratch directory");
            let state = root.join("state");
            let selection = root.join("selection.json");
            let env = env::Env::from([
                ("RELAY_TOKEN".to_owned(), "test-relay-token".to_owned()),
                (
                    MODEL_SELECTION_PATH_ENV.to_owned(),
                    selection.to_str().expect("path").to_owned(),
                ),
                (
                    MODELS_ROOT_ENV.to_owned(),
                    root.join("models").to_str().expect("path").to_owned(),
                ),
            ]);
            Self {
                root,
                state,
                selection,
                env,
            }
        }

        fn check(&self, runtime: AuxiliaryRuntime) -> Result<Checked, CheckConfigError> {
            check_config(&self.env, Some(&self.state), runtime)
        }

        fn write_selection(&self) {
            let digest = "a".repeat(64);
            let publication = Publication {
                source_locator: "test/model".to_owned(),
                revision: "b".repeat(40),
                content: digest.clone(),
            };
            let selection = ModelSelection {
                model_publication: publication.clone(),
                dataset_publication: publication,
                bundle_members_digest: digest.clone(),
                evaluation_receipt_digest: digest.clone(),
                field_evaluation_receipt_digest: digest.clone(),
                calibration_digest: digest.clone(),
                conformance_digest: digest.clone(),
                input_observation_schema: "synthetic-schema".to_owned(),
                output_class_count: 2,
                output_class_semantics_digest: digest.clone(),
                policy_digest: digest,
                runtime_format: "onnx".to_owned(),
                bundle_format: "synthetic-bundle".to_owned(),
                preprocessing_identity: "synthetic-preprocessing".to_owned(),
                transition_threshold: 0.5,
                threshold_source: "default".to_owned(),
            };
            let canonical = Serialiser::FetchModelsManifest
                .canonical(&selection.as_json())
                .expect("canonical selection");
            fs::write(&self.selection, canonical).expect("owned selection");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).expect("remove owned scratch directory");
        }
    }

    #[test]
    fn unwired_check_config_remains_optional_for_both_explicit_providers() {
        let fixture = Fixture::new();
        for runtime in [AuxiliaryRuntime::TensorRt, AuxiliaryRuntime::OnnxRuntimeCpu] {
            let checked = fixture.check(runtime).expect("unwired static check");
            assert!(checked.selection.is_none());
            assert!(checked.engine_identity.is_none());
            assert!(checked.cpu_model_hashes.is_none());
            assert!(matches!(checked.last_known_good, Ok(None)));
            assert!(
                !fixture.state.exists(),
                "static check must not create a state directory"
            );
        }
    }

    #[test]
    fn unused_gpu_only_wiring_neither_selects_nor_admits_the_cpu_provider() {
        let mut fixture = Fixture::new();
        for key in [
            "ML_WORKER_STORED_POSE_ENGINE_PATH",
            "ML_WORKER_BED_ENGINE_PATH",
            "ML_WORKER_FALL_ENGINE_PATH",
        ] {
            fixture.env.insert(key.to_owned(), "\0".to_owned());
            let checked = fixture
                .check(AuxiliaryRuntime::OnnxRuntimeCpu)
                .expect("unused GPU wiring");
            assert!(checked.engine_identity.is_none());
            assert!(checked.cpu_model_hashes.is_none());
            let error = fixture
                .check(AuxiliaryRuntime::TensorRt)
                .expect_err("GPU wiring requires live engine");
            assert_eq!(
                error,
                CheckConfigError::Identity(IdentityError {
                    kind: model_bundle::identity::IdentityKind::ArtifactAbsent,
                    subject: FLOW_ENGINE_ENV.to_owned(),
                })
            );
            assert_eq!(error.exit(), Exit::RefuseToStart);
        }
    }

    #[test]
    fn either_provider_still_requires_live_wiring_when_any_common_artifact_is_set() {
        for runtime in [AuxiliaryRuntime::TensorRt, AuxiliaryRuntime::OnnxRuntimeCpu] {
            for key in [FLOW_ENGINE_IDENTITY_ENV, "ML_WORKER_FLOW_ONNX_PATH"]
                .into_iter()
                .chain(FLOW_IDENTITY_FILES.iter().map(|(_, name)| *name))
            {
                let env = env::Env::from([(key.to_owned(), "/unused".to_owned())]);
                let error = engine_identity(&env, None, runtime)
                    .err()
                    .expect("incomplete live wiring");
                assert_eq!(
                    error.kind,
                    model_bundle::identity::IdentityKind::ArtifactAbsent
                );
                assert_eq!(error.subject, FLOW_ENGINE_ENV);
            }
        }
    }

    #[test]
    fn provider_selection_preserves_env_selection_admission_identity_then_lkg_order() {
        for runtime in [AuxiliaryRuntime::TensorRt, AuxiliaryRuntime::OnnxRuntimeCpu] {
            let mut fixture = Fixture::new();
            let lkg = fixture.state.join("config-lkg");
            fs::create_dir_all(&lkg).expect("owned LKG directory");
            fs::write(lkg.join("current.json"), b"not JSON").expect("owned invalid LKG");
            fs::write(&fixture.selection, b"not JSON").expect("owned invalid selection");
            fixture
                .env
                .insert(FLOW_ENGINE_IDENTITY_ENV.to_owned(), "/unused".to_owned());
            let token = fixture.env.remove("RELAY_TOKEN").expect("fixture token");
            let error = fixture.check(runtime).expect_err("environment first");
            assert!(matches!(&error, CheckConfigError::Env(_)));
            assert_eq!(error.exit(), Exit::Config);
            assert!(!lkg.join(".lock").exists());
            fixture.env.insert("RELAY_TOKEN".to_owned(), token);
            let error = fixture
                .check(runtime)
                .expect_err("selection before identity");
            assert_eq!(
                error,
                CheckConfigError::Selection(SelectionLoadError::NotCanonical)
            );
            assert_eq!(error.exit(), Exit::Config);
            assert!(!lkg.join(".lock").exists());
            fixture.write_selection();
            let error = fixture
                .check(runtime)
                .expect_err("bundle admission before identity");
            assert!(matches!(&error, CheckConfigError::Admission(_)));
            assert_eq!(error.exit(), Exit::RefuseToStart);
            assert!(!lkg.join(".lock").exists());
            fs::remove_file(&fixture.selection).expect("remove owned selection");
            let error = fixture.check(runtime).expect_err("identity before LKG");
            assert!(matches!(&error, CheckConfigError::Identity(_)));
            assert_eq!(error.exit(), Exit::RefuseToStart);
            assert!(!lkg.join(".lock").exists());
            fixture.env.remove(FLOW_ENGINE_IDENTITY_ENV);
            let checked = fixture
                .check(runtime)
                .expect("unreadable LKG stays nonfatal");
            assert!(matches!(checked.last_known_good, Err(LkgError::Record)));
            assert!(lkg.join(".lock").is_file());
        }
    }
}

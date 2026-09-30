//! `identity_for` of `worker/tools/edge_engine_build.py` and
//! `verify_engine_identity` of `worker/runtime/flow/cold_start.py`, in the
//! Python check order. The engine is never built or loaded; only its bytes
//! are hashed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::tree::sha256_hex;
use crate::config::{lookup, parse_json};
use crate::json::{Json, Serialiser};

/// `FLOW_IDENTITY_FILES`: each identity key and the env key naming its file,
/// in the Python check order (after `engine_sha256`).
pub const FLOW_IDENTITY_FILES: [(&str, &str); 5] = [
    ("infer_config_sha256", "ML_WORKER_FLOW_INFER_CONFIG"),
    ("tracker_config_sha256", "ML_WORKER_FLOW_TRACKER_CONFIG"),
    ("tracker_library_sha256", "ML_WORKER_FLOW_TRACKER_LIBRARY"),
    ("onnx_sha256", "ML_WORKER_FLOW_ONNX_PATH"),
    ("parser_lib_sha256", "ML_WORKER_FLOW_PARSER_LIBRARY"),
];

/// `sys.int_info.default_max_str_digits`: a longer string fails `int()`.
const MAX_INT_DIGITS: usize = 4300;

/// Which `EngineIdentityError` message (or uncaught Python error) refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityKind {
    /// `Flow engine is absent: {path}`.
    EngineAbsent,
    /// `Flow engine identity is absent: {path}`.
    IdentityAbsent,
    /// `Flow engine identity is unreadable: {path}` (an OS or JSON error).
    IdentityUnreadable,
    /// The identity bytes are not UTF-8: Python's uncaught `UnicodeDecodeError`.
    IdentityNotUtf8,
    /// `Flow engine identity must be a JSON object`.
    NotObject,
    /// `Flow engine identity lacks valid batch_size`.
    BatchSize,
    /// `deployed Flow roster batch must not be negative`.
    NegativeDeployedBatch,
    /// `Flow engine batch {n} does not cover deployed roster batch {m}`.
    BatchNotCovering,
    /// `Flow engine identity lacks valid {key}`.
    DigestInvalid,
    /// `Flow artifact is absent for {key}: {path}`.
    ArtifactAbsent,
    /// The artifact cannot be read: Python's uncaught `OSError` in `_sha256`.
    ArtifactUnreadable,
    /// `Flow artifact digest mismatch for {key}: {path}`.
    DigestMismatch,
    /// `Flow engine identity lacks image_digest`.
    ImageDigest,
}

/// A refusal: the check and the identity key it names (empty when the
/// Python message names no key).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityError {
    pub kind: IdentityKind,
    pub subject: String,
}

type Verified<T> = Result<T, IdentityError>;

fn refuse<T>(kind: IdentityKind, subject: &str) -> Verified<T> {
    Err(IdentityError {
        kind,
        subject: subject.to_owned(),
    })
}

/// `Path.is_file()`: follows symlinks; any stat error is `False`.
fn is_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|info| info.is_file())
}

/// `int(x)` with `n > 0 and str(n) == x`: only a string of ASCII digits
/// without a leading zero passes. Values beyond `u128` saturate, which keeps
/// every comparison with a roster batch exact.
fn batch_size(value: Option<&Json>) -> Option<u128> {
    let Some(Json::Str(text)) = value else {
        return None;
    };
    let canonical = text.len() <= MAX_INT_DIGITS
        && text.bytes().next().is_some_and(|first| first != b'0')
        && text.bytes().all(|byte| byte.is_ascii_digit());
    canonical.then(|| text.parse().unwrap_or(u128::MAX))
}

/// Python `str(value)`. Numbers print as their `repr`, which is their JSON
/// form; containers print as compact JSON rather than a Python `repr`.
fn python_str(value: &Json) -> String {
    match value {
        Json::Str(text) => text.clone(),
        Json::Bool(true) => "True".to_owned(),
        Json::Bool(false) => "False".to_owned(),
        Json::Null => "None".to_owned(),
        // A parsed document holds only finite floats, so this never fails.
        other => Serialiser::ModelSelection
            .canonical(other)
            .unwrap_or_default(),
    }
}

/// `verify_engine_identity(engine, identity, files, deployed_batch=...)`:
/// returns every identity entry stringified.
pub fn verify_engine_identity(
    engine: &Path,
    identity: &Path,
    files: &[(&str, PathBuf)],
    deployed_batch: Option<i128>,
) -> Verified<BTreeMap<String, String>> {
    if !is_file(engine) {
        return refuse(IdentityKind::EngineAbsent, "");
    }
    if !is_file(identity) {
        return refuse(IdentityKind::IdentityAbsent, "");
    }
    let Ok(raw) = fs::read(identity) else {
        return refuse(IdentityKind::IdentityUnreadable, "");
    };
    if std::str::from_utf8(&raw).is_err() {
        return refuse(IdentityKind::IdentityNotUtf8, "");
    }
    let Some(document) = parse_json(&raw) else {
        return refuse(IdentityKind::IdentityUnreadable, "");
    };
    let Json::Object(members) = &document else {
        return refuse(IdentityKind::NotObject, "");
    };
    let Some(engine_batch) = batch_size(lookup(members, "batch_size")) else {
        return refuse(IdentityKind::BatchSize, "");
    };
    if let Some(deployed) = deployed_batch {
        let Ok(deployed) = u128::try_from(deployed) else {
            return refuse(IdentityKind::NegativeDeployedBatch, "");
        };
        if engine_batch < deployed {
            return refuse(IdentityKind::BatchNotCovering, "");
        }
    }
    let artifacts = files.iter().map(|(key, path)| (*key, path.as_path()));
    for (key, path) in std::iter::once(("engine_sha256", engine)).chain(artifacts) {
        let recorded = match lookup(members, key) {
            Some(Json::Str(digest)) if digest.chars().count() == 64 => digest,
            _ => return refuse(IdentityKind::DigestInvalid, key),
        };
        if !is_file(path) {
            return refuse(IdentityKind::ArtifactAbsent, key);
        }
        let Ok(content) = fs::read(path) else {
            return refuse(IdentityKind::ArtifactUnreadable, key);
        };
        if sha256_hex(&content) != *recorded {
            return refuse(IdentityKind::DigestMismatch, key);
        }
    }
    if !matches!(lookup(members, "image_digest"), Some(Json::Str(image)) if !image.is_empty()) {
        return refuse(IdentityKind::ImageDigest, "");
    }
    let mut verified = BTreeMap::new();
    for (key, value) in members {
        verified.insert(key.clone(), python_str(value));
    }
    Ok(verified)
}

/// A JSON string as Python `json.dumps` writes it (`ensure_ascii=True`).
fn quoted(text: &str) -> String {
    let text = Json::Str(text.to_owned());
    // A string always has a JSON form.
    Serialiser::ModelSelection
        .canonical(&text)
        .unwrap_or_default()
}

/// `identity_for`: the identity file an engine build writes next to the
/// engine, byte for byte (`json.dumps(identity, sort_keys=True)` plus a
/// newline). `files` pairs each identity key with its artifact, as for
/// `verify_engine_identity`. Python's `EngineBuildError` for an empty image
/// digest or a batch below 1 is `ImageDigest` or `BatchSize`; an artifact that
/// cannot be read is `ArtifactUnreadable`.
pub fn identity_for(
    engine: &Path,
    files: &[(&str, PathBuf)],
    image_digest: &str,
    batch_size: i128,
) -> Verified<String> {
    if image_digest.is_empty() {
        return refuse(IdentityKind::ImageDigest, "");
    }
    if batch_size <= 0 {
        return refuse(IdentityKind::BatchSize, "");
    }
    let mut identity = BTreeMap::new();
    let artifacts = files.iter().map(|(key, path)| (*key, path.as_path()));
    for (key, path) in std::iter::once(("engine_sha256", engine)).chain(artifacts) {
        let Ok(content) = fs::read(path) else {
            return refuse(IdentityKind::ArtifactUnreadable, key);
        };
        identity.insert(key.to_owned(), sha256_hex(&content));
    }
    identity.insert("image_digest".to_owned(), image_digest.to_owned());
    identity.insert("batch_size".to_owned(), batch_size.to_string());
    let entries: Vec<String> = identity
        .iter()
        .map(|(key, value)| format!("{}: {}", quoted(key), quoted(value)))
        .collect();
    Ok(format!("{{{}}}\n", entries.join(", ")))
}

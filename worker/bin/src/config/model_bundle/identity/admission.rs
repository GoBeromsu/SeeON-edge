//! File-only admission of the single aggregate identity. No CUDA or build calls.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use rustix::fs::{Mode, OFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::infer;
use super::{IdentityError, IdentityKind, deployment_image_digest, refuse, schema};

const MAX_ARTIFACT: u64 = 512 * 1024 * 1024;
const MAX_IDENTITY: u64 = 64 * 1024;
const FLOW_KEYS: [&str; 4] = [
    "infer_config_sha256",
    "tracker_config_sha256",
    "tracker_library_sha256",
    "parser_lib_sha256",
];

pub struct EnginePaths<'a> {
    pub live_pose: &'a Path,
    pub stored_pose: &'a Path,
    pub bed: &'a Path,
    pub fall: &'a Path,
}

pub struct IdentityInputs<'a> {
    pub engines: EnginePaths<'a>,
    pub pose_onnx_sha256: &'a str,
    pub bed_onnx_sha256: &'a str,
    pub fall_onnx_sha256: &'a str,
    pub flow: &'a [(&'a str, PathBuf)],
    pub observer_library: &'a Path,
    pub image_digest: &'a str,
    pub configured_batch: Option<u32>,
    pub deployed_batch: Option<i128>,
}

fn error(kind: IdentityKind, subject: &str) -> IdentityError {
    IdentityError {
        kind,
        subject: subject.to_owned(),
    }
}

fn open_regular(path: &Path, limit: u64, subject: &str) -> Result<(File, u64), IdentityError> {
    // Ordinary artifact symlinks remain valid. Selected bundle capture applies
    // its stricter policy before its admitted source digest reaches this gate.
    let fd = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(|_| error(IdentityKind::ArtifactUnreadable, subject))?;
    let file = File::from(fd);
    let metadata = file
        .metadata()
        .map_err(|_| error(IdentityKind::ArtifactUnreadable, subject))?;
    let size = metadata.len();
    if !metadata.is_file() || size == 0 || size > limit {
        return refuse(IdentityKind::ArtifactUnreadable, subject);
    }
    Ok((file, size))
}

pub fn fingerprint(path: &Path, subject: &str) -> Result<String, IdentityError> {
    capture(path, subject, false).map(|(digest, _)| digest)
}

fn capture(path: &Path, subject: &str, retain: bool) -> Result<(String, Vec<u8>), IdentityError> {
    let (file, declared) = open_regular(path, MAX_ARTIFACT, subject)?;
    let mut input = file.take(declared + 1);
    let mut raw = Vec::new();
    let mut count = 0_u64;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|_| error(IdentityKind::ArtifactUnreadable, subject))?;
        if read == 0 {
            break;
        }
        count += read as u64;
        if count > declared {
            return refuse(IdentityKind::ArtifactUnreadable, subject);
        }
        digest.update(&buffer[..read]);
        if retain {
            raw.extend_from_slice(&buffer[..read]);
        }
    }
    if count != declared {
        return refuse(IdentityKind::ArtifactUnreadable, subject);
    }
    let mut result = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(result, "{byte:02x}")
            .map_err(|_| error(IdentityKind::ArtifactUnreadable, subject))?;
    }
    Ok((result, raw))
}

pub fn verify_aggregate(
    path: &Path,
    inputs: IdentityInputs<'_>,
) -> Result<BTreeMap<String, String>, IdentityError> {
    let (file, size) = open_regular(path, MAX_IDENTITY, "identity")
        .map_err(|_| error(IdentityKind::IdentityUnreadable, "identity"))?;
    let mut raw = Vec::with_capacity(size as usize);
    file.take(size + 1)
        .read_to_end(&mut raw)
        .map_err(|_| error(IdentityKind::IdentityUnreadable, "identity"))?;
    if raw.len() as u64 != size {
        return refuse(IdentityKind::IdentityUnreadable, "identity");
    }
    let document: Value = serde_json::from_slice(&raw)
        .map_err(|_| error(IdentityKind::IdentityUnreadable, "identity"))?;
    let members = document
        .as_object()
        .ok_or_else(|| error(IdentityKind::NotObject, "identity"))?;
    if members.len() != 4
        || document["schema_version"].as_u64() != Some(1)
        || !["schema_version", "engines", "flow", "batch_size"]
            .iter()
            .all(|key| members.contains_key(*key))
    {
        return refuse(IdentityKind::Schema, "identity");
    }
    let batch = document["batch_size"]
        .as_u64()
        .and_then(|batch| u32::try_from(batch).ok())
        .filter(|batch| (1..=16).contains(batch))
        .ok_or_else(|| error(IdentityKind::BatchSize, "batch_size"))?;
    if inputs
        .configured_batch
        .is_some_and(|configured| configured != batch)
    {
        return refuse(IdentityKind::BatchSize, "batch_size");
    }
    if let Some(deployed) = inputs.deployed_batch {
        if deployed < 0 {
            return refuse(IdentityKind::NegativeDeployedBatch, "batch_size");
        }
        if deployed > i128::from(batch) {
            return refuse(IdentityKind::BatchNotCovering, "batch_size");
        }
    }
    let image = deployment_image_digest(inputs.image_digest)
        .ok_or_else(|| error(IdentityKind::ImageDigest, "image_digest"))?;
    schema::validate_entries(&document["engines"], image, batch).map_err(
        |failure| match failure {
            schema::SchemaError::Image => error(IdentityKind::ImageDigest, "image_digest"),
            schema::SchemaError::Receipt => error(IdentityKind::Schema, "engines"),
        },
    )?;
    for (role, engine, source) in [
        (
            "live_pose",
            inputs.engines.live_pose,
            inputs.pose_onnx_sha256,
        ),
        (
            "stored_pose",
            inputs.engines.stored_pose,
            inputs.pose_onnx_sha256,
        ),
        ("bed", inputs.engines.bed, inputs.bed_onnx_sha256),
        ("fall", inputs.engines.fall, inputs.fall_onnx_sha256),
    ] {
        let entry = &document["engines"][role];
        if entry["onnx_sha256"].as_str() != Some(source) {
            return refuse(IdentityKind::DigestMismatch, role);
        }
        if entry["engine"].as_str() != engine.file_name().and_then(|name| name.to_str()) {
            return refuse(IdentityKind::DigestMismatch, role);
        }
        if entry["engine_sha256"].as_str() != Some(fingerprint(engine, role)?.as_str()) {
            return refuse(IdentityKind::DigestMismatch, role);
        }
    }
    let flow = document["flow"]
        .as_object()
        .ok_or_else(|| error(IdentityKind::Schema, "flow"))?;
    let provided: BTreeSet<_> = inputs.flow.iter().map(|(key, _)| *key).collect();
    if flow.len() != 4
        || inputs.flow.len() != 4
        || provided.len() != 4
        || !FLOW_KEYS
            .iter()
            .all(|key| provided.contains(key) && flow.contains_key(*key))
    {
        return refuse(IdentityKind::Schema, "flow");
    }
    for (key, artifact) in inputs.flow {
        let expected = flow[*key]
            .as_str()
            .filter(|sha| crate::config::is_hex(sha, 64))
            .ok_or_else(|| error(IdentityKind::DigestInvalid, key))?;
        let is_infer = *key == "infer_config_sha256";
        let (digest, bytes) = capture(artifact, key, is_infer)?;
        if digest != expected {
            return refuse(IdentityKind::DigestMismatch, key);
        }
        if is_infer {
            let text = std::str::from_utf8(&bytes)
                .map_err(|_| error(IdentityKind::Schema, "infer_config_sha256"))?;
            if !infer::engine_only(text) {
                return refuse(IdentityKind::Schema, "infer_config_sha256");
            }
        }
    }
    let live = document["engines"]["live_pose"].as_object().unwrap();
    if live["observer_library_sha256"].as_str()
        != Some(fingerprint(inputs.observer_library, "observer_library_sha256")?.as_str())
    {
        return refuse(IdentityKind::DigestMismatch, "observer_library_sha256");
    }
    // Only the media plane's facts are projected into its existing runtime
    // metadata map. No legacy flat identity file is accepted by this reader.
    let mut media: BTreeMap<String, String> = live
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string()),
            )
        })
        .collect();
    media.insert("batch_size".to_owned(), batch.to_string());
    Ok(media)
}

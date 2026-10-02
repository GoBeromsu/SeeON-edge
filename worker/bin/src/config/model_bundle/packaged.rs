//! Packaged-model admission from `pose_bbox56_bundle_support.verify_bundle`.
//! Unlike selected bundles, packaged manifests do not declare an exact tree
//! and follow symlinks. No environment lookup, model loading, or writes occur.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use super::tree::{read_following, resolve_lenient, sha256_hex};
use super::{Admission, AdmissionKind, refuse};
use crate::config::{lookup, parse_json};
use crate::json::Json;

/// Python local_env / FallModelConfig packaged root. `MODELS_ROOT_ENV` is selected-only.
pub(crate) const PACKAGED_FALL_ROOT: &str = "models/fall/pose-bbox56-gru";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackagedProof {
    pub member_digests: BTreeMap<String, String>,
    pub calibration: Vec<u8>,
    pub conformance: (String, Vec<u8>),
}

/// Verify every listed file, then retain the exact verified runtime documents.
/// The three evidence members must be listed under their canonical names.
/// Duplicate entries are all verified; member identity is first-wins, matching
/// `member_digest`. Duplicate conformance entries are still ambiguous/refused.
pub fn admit_packaged_bundle(root: &Path) -> Admission<PackagedProof> {
    let root = resolve_lenient(root)?;
    match fs::metadata(&root) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return refuse(AdmissionKind::NotRegularDirectory, "packaged root"),
        Err(_) => return refuse(AdmissionKind::Unavailable, "packaged root"),
    }
    let raw = read_following(&root.join("bundle-manifest.json"), "bundle-manifest.json")?;
    let Some(Json::Object(manifest)) = parse_json(&raw) else {
        return refuse(AdmissionKind::ManifestInvalidJson, "bundle-manifest.json");
    };
    let Some(Json::Array(files)) = lookup(&manifest, "files") else {
        return refuse(AdmissionKind::ShapeMismatch, "files");
    };
    let mut member_digests = BTreeMap::new();
    let mut calibration = None;
    let mut conformances = Vec::new();
    for entry in files {
        let Json::Object(entry) = entry else {
            return refuse(AdmissionKind::MemberInvalid, "");
        };
        let (Some(Json::Str(relative)), Some(Json::Str(digest)), Some(size)) = (
            lookup(entry, "relative_path"),
            lookup(entry, "sha256"),
            lookup(entry, "size"),
        ) else {
            return refuse(AdmissionKind::MemberInvalid, "");
        };
        let path = lexical_path(relative)?;
        let size = match size {
            Json::Int(size) => *size,
            Json::Bool(value) => i128::from(*value),
            _ => return refuse(AdmissionKind::MemberInvalid, relative),
        };
        if digest.chars().count() != 64 {
            return refuse(AdmissionKind::MemberInvalid, relative);
        }
        let bytes = read_following(&root.join(&path), relative)?;
        if i128::try_from(bytes.len()).ok() != Some(size) || sha256_hex(&bytes) != *digest {
            return refuse(AdmissionKind::MemberMismatch, relative);
        }
        member_digests
            .entry(relative.clone())
            .or_insert_with(|| digest.clone());
        if relative == "calibration.json" && calibration.is_none() {
            calibration = Some(bytes);
        } else if path.parent() == Some(Path::new("conformance")) {
            conformances.push((relative.clone(), bytes));
        }
    }
    for member in ["model.onnx", "model.pt", "calibration.json"] {
        if !member_digests.contains_key(member) {
            return refuse(AdmissionKind::MemberUnlisted, member);
        }
    }
    if conformances.len() != 1 {
        return refuse(AdmissionKind::ConformanceDigest, "packaged conformance");
    }
    Ok(PackagedProof {
        member_digests,
        calibration: calibration.expect("listed calibration was captured"),
        conformance: conformances.pop().expect("exactly one conformance"),
    })
}

/// Match pathlib's lexical normalization, without adding selected-bundle
/// containment rules: leading/interior `.` is allowed, any `..` is refused.
fn lexical_path(relative: &str) -> Admission<PathBuf> {
    let mut normalized = PathBuf::new();
    for part in Path::new(relative).components() {
        match part {
            Component::Normal(name) => normalized.push(name),
            Component::CurDir => {}
            _ => return refuse(AdmissionKind::PathEscapes, relative),
        }
    }
    Ok(normalized)
}

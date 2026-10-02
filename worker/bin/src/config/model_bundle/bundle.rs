//! `admit_model_bundle` of `worker/runtime/provenance/model_bundle.py`, in
//! the Python check order: the content digest is verified before the member
//! shapes, and the exact tree last. Nothing is written or selected.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::receipt::verify_required_members;
use super::tree::{read_regular, require_below, require_directory, sha256_hex, verify_exact_tree};
use super::{Admission, AdmissionKind, refuse};
use crate::config::selection::ModelSelection;
use crate::config::{is_hex, is_segment, lookup, parse_json};
use crate::json::{Json, Serialiser, model_selection_digest};

/// What an admitted bundle proves (`ModelBundleProof.observed`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleProof {
    pub bundle_sha256: String,
    pub members: Vec<String>,
    pub receipts: Vec<String>,
    pub member_digests: BTreeMap<String, String>,
    /// Exact digest-verified bytes for runtime admission, not a mutable reread.
    pub calibration: Vec<u8>,
    pub conformance: (String, Vec<u8>),
    /// The seven bundle identities plus the `evaluation` and `field` receipt
    /// digests.
    pub identities: BTreeMap<String, String>,
}

/// `_BUNDLE_IDENTITY_FIELDS` with the desired value of each
/// (`desired_model_bundle_from_selection_document`).
fn bundle_identities(desired: &ModelSelection) -> [(&'static str, &String); 7] {
    [
        ("dataset", &desired.dataset_publication.content),
        ("calibration", &desired.calibration_digest),
        ("conformance", &desired.conformance_digest),
        ("class", &desired.output_class_semantics_digest),
        ("input", &desired.input_observation_schema),
        ("policy", &desired.policy_digest),
        ("members", &desired.bundle_members_digest),
    ]
}

fn fetch_canonical(value: &Json) -> Admission<String> {
    Serialiser::FetchModelsManifest
        .canonical(value)
        .or_else(|_| refuse(AdmissionKind::ManifestNonJson, ""))
}

/// `_validate_bundle_identity`: sha256 of the canonical members triples and
/// payload must equal the bundle sha, before any member shape is checked.
fn validate_bundle_identity(members: &[Json], payload: &Json, sha: &str) -> Admission<()> {
    let mut triples = Vec::new();
    for member in members {
        let Json::Object(entry) = member else {
            return refuse(AdmissionKind::MemberInvalid, "");
        };
        let mut triple = Vec::new();
        for key in ["path", "sha256", "size"] {
            let Some(value) = lookup(entry, key) else {
                return refuse(AdmissionKind::MemberInvalid, "");
            };
            triple.push((key.to_owned(), value.clone()));
        }
        triples.push(Json::Object(triple));
    }
    let document = Json::Object(vec![
        ("members".to_owned(), Json::Array(triples)),
        ("payload".to_owned(), payload.clone()),
    ]);
    if sha256_hex(fetch_canonical(&document)?.as_bytes()) != sha {
        return refuse(AdmissionKind::ContentIdentity, "");
    }
    Ok(())
}

/// `_verify_members`: returns `(path, sha256)` in manifest order.
fn verify_members(root: &Path, members: &[Json]) -> Admission<Vec<(String, String)>> {
    let mut seen: Vec<(String, String)> = Vec::new();
    for member in members {
        let Json::Object(entry) = member else {
            return refuse(AdmissionKind::MemberInvalid, "");
        };
        let (path, size, digest) = match (
            lookup(entry, "path"),
            lookup(entry, "size"),
            lookup(entry, "sha256"),
        ) {
            (Some(Json::Str(path)), Some(Json::Int(size)), Some(Json::Str(digest)))
                if path.split('/').all(is_segment)
                    && path != "manifest.json"
                    && *size >= 0
                    && is_hex(digest, 64)
                    && !seen.iter().any(|(known, _)| known == path) =>
            {
                (path, *size, digest)
            }
            _ => return refuse(AdmissionKind::MemberInvalid, ""),
        };
        let member_path = root.join(path);
        require_below(root, &member_path)?;
        let content = read_regular(&member_path, &format!("member {path}"))?;
        if i128::try_from(content.len()) != Ok(size) || sha256_hex(&content) != *digest {
            return refuse(AdmissionKind::MemberMismatch, path);
        }
        seen.push((path.clone(), digest.clone()));
    }
    if seen.is_empty() {
        return refuse(AdmissionKind::MembersMissing, "");
    }
    Ok(seen)
}

/// `_verify_selection_bundle_format` and `_verify_selection_policy_digest`.
fn verify_selection_documents(
    root: &Path,
    digests: &BTreeMap<String, String>,
    desired: &ModelSelection,
) -> Admission<Vec<u8>> {
    const FORMAT: &str = "bundle-manifest.json";
    const CALIBRATION: &str = "calibration.json";
    if !digests.contains_key(FORMAT) {
        return refuse(AdmissionKind::NoBundleManifestMember, "");
    }
    let raw = read_verified_member(root, digests, FORMAT)?;
    let Some(manifest) = parse_json(&raw) else {
        return refuse(AdmissionKind::MemberInvalidJson, FORMAT);
    };
    let observed = match &manifest {
        Json::Object(m) => lookup(m, "schema_version"),
        _ => None,
    };
    if observed != Some(&Json::Str(desired.bundle_format.clone())) {
        return refuse(AdmissionKind::BundleFormat, "");
    }
    let raw = read_verified_member(root, digests, CALIBRATION)?;
    let Some(calibration) = parse_json(&raw) else {
        return refuse(AdmissionKind::MemberInvalidJson, CALIBRATION);
    };
    let temporal_rule = match &calibration {
        Json::Object(m) => lookup(m, "temporal_rule").cloned().unwrap_or(Json::Null),
        _ => Json::Null,
    };
    if model_selection_digest(&temporal_rule).ok().as_ref() != Some(&desired.policy_digest) {
        return refuse(AdmissionKind::PolicyDigest, "");
    }
    Ok(raw)
}

fn read_verified_member(
    root: &Path,
    digests: &BTreeMap<String, String>,
    relative: &str,
) -> Admission<Vec<u8>> {
    let raw = read_regular(&root.join(relative), &format!("member {relative}"))?;
    if digests.get(relative) != Some(&sha256_hex(&raw)) {
        return refuse(AdmissionKind::MemberMismatch, relative);
    }
    Ok(raw)
}

/// `admit_model_bundle(models_root, desired)` for a parsed selection.
pub fn admit_model_bundle(models_root: &Path, desired: &ModelSelection) -> Admission<BundleProof> {
    let sha = &desired.model_publication.content;
    require_directory(models_root, "models root")?;
    let bundles = models_root.join("bundles");
    require_directory(&bundles, "bundles root")?;
    let root = bundles.join(sha);
    require_directory(&root, "bundle root")?;
    require_below(&bundles, &root)?;
    let raw = read_regular(&root.join("manifest.json"), "manifest")?;
    let Some(document) = parse_json(&raw) else {
        return refuse(AdmissionKind::ManifestInvalidJson, "");
    };
    let Json::Object(manifest) = &document else {
        return refuse(AdmissionKind::ManifestNotCanonical, "");
    };
    if format!("{}\n", fetch_canonical(&document)?).as_bytes() != raw.as_slice() {
        return refuse(AdmissionKind::ManifestNotCanonical, "");
    }
    let version = lookup(manifest, "schema_version");
    if !matches!(version, Some(Json::Int(1) | Json::Bool(true)))
        && version != Some(&Json::Float(1.0))
    {
        return refuse(AdmissionKind::SchemaMismatch, "");
    }
    if lookup(manifest, "bundle_sha256") != Some(&Json::Str(sha.clone())) {
        return refuse(AdmissionKind::IdentityMismatch, "");
    }
    if lookup(manifest, "runtime_format") != Some(&Json::Str(desired.runtime_format.clone())) {
        return refuse(AdmissionKind::RuntimeFormat, "");
    }
    let (members, receipts, payload) = match (
        lookup(manifest, "members"),
        lookup(manifest, "receipts"),
        lookup(manifest, "payload"),
    ) {
        (Some(Json::Array(m)), Some(Json::Array(r)), Some(p @ Json::Object(_))) => (m, r, p),
        _ => return refuse(AdmissionKind::ShapeMismatch, ""),
    };
    let Json::Object(payload_members) = payload else {
        return refuse(AdmissionKind::ShapeMismatch, "");
    };
    let Some(Json::Object(identities)) = lookup(payload_members, "identities") else {
        return refuse(AdmissionKind::IdentitiesMissing, "");
    };
    let expected = bundle_identities(desired);
    for (field, want) in expected {
        if lookup(identities, field) != Some(&Json::Str(want.clone())) {
            return refuse(AdmissionKind::FieldIdentity, field);
        }
    }
    let present: BTreeSet<&str> = identities.iter().map(|(key, _)| key.as_str()).collect();
    if present.len() != expected.len() {
        return refuse(AdmissionKind::UnknownIdentityFields, "");
    }
    validate_bundle_identity(members, payload, sha)?;
    let observed_members = verify_members(&root, members)?;
    let observed_receipts = verify_members(&root, receipts)?;
    let member_digests: BTreeMap<String, String> = observed_members.iter().cloned().collect();
    if member_digests.get("calibration.json") != Some(&desired.calibration_digest) {
        return refuse(AdmissionKind::CalibrationDigest, "");
    }
    let conformance = member_digests
        .values()
        .filter(|digest| **digest == desired.conformance_digest)
        .count();
    if conformance != 1 {
        return refuse(AdmissionKind::ConformanceDigest, "");
    }
    let calibration = verify_selection_documents(&root, &member_digests, desired)?;
    let receipt_identities = verify_required_members(&root, members, receipts, desired)?;
    let path = |(path, _): &(String, String)| path.clone();
    let members: Vec<String> = observed_members.iter().map(path).collect();
    let receipts: Vec<String> = observed_receipts.iter().map(path).collect();
    let mut tree: BTreeSet<String> = members.iter().chain(&receipts).cloned().collect();
    tree.insert("manifest.json".to_owned());
    verify_exact_tree(&root, &tree)?;
    let mut identities: BTreeMap<String, String> = expected
        .iter()
        .map(|(field, value)| ((*field).to_owned(), (*value).clone()))
        .collect();
    identities.extend(receipt_identities);
    let conformance_path = member_digests
        .iter()
        .find(|(_, digest)| **digest == desired.conformance_digest)
        .map(|(path, _)| path)
        .expect("one conformance member was verified");
    let conformance = (
        conformance_path.clone(),
        read_verified_member(&root, &member_digests, conformance_path)?,
    );
    Ok(BundleProof {
        bundle_sha256: sha.clone(),
        members,
        receipts,
        member_digests,
        calibration,
        conformance,
        identities,
    })
}

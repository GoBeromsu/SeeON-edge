//! `_verify_required_members` of `model_bundle.py` with the receipt
//! validators of `contracts/model_selection.py`
//! (`validate_evaluation_receipt_identity`, `validate_field_receipt_identity`).

use std::collections::BTreeMap;
use std::path::Path;

use super::tree::{read_regular, sha256_hex};
use super::{Admission, AdmissionKind, refuse};
use crate::config::selection::{
    ModelSelection, digest, exact_keys, object, positive_integer, string,
};
use crate::config::{lookup, parse_json};
use crate::json::{Json, Serialiser, model_selection_digest};

const COMMON: [&str; 9] = [
    "bundle_sha256",
    "bundle_members_digest",
    "dataset_payload_digest",
    "calibration_digest",
    "conformance_digest",
    "input_observation_schema",
    "output_class_count",
    "output_class_semantics_digest",
    "policy_digest",
];

/// `_receipt_identity` then the external digest and
/// `_require_receipt_matches`; every failure is one `ContractError`.
fn validate(desired: &ModelSelection, document: &Json, field: bool) -> Option<()> {
    let place = if field {
        "field-evaluation-receipt"
    } else {
        "evaluation-receipt"
    };
    let receipt = object(document, place).ok()?;
    let extra: &[&str] = if field {
        &["evaluation_receipt_digest", "status"]
    } else {
        &[]
    };
    exact_keys(receipt, COMMON.iter().chain(extra).copied(), place).ok()?;
    let mut expected = vec![
        ("bundle_sha256", &desired.model_publication.content),
        ("bundle_members_digest", &desired.bundle_members_digest),
        (
            "dataset_payload_digest",
            &desired.dataset_publication.content,
        ),
        ("calibration_digest", &desired.calibration_digest),
        ("conformance_digest", &desired.conformance_digest),
        (
            "output_class_semantics_digest",
            &desired.output_class_semantics_digest,
        ),
        ("policy_digest", &desired.policy_digest),
    ];
    if field {
        expected.push((
            "evaluation_receipt_digest",
            &desired.evaluation_receipt_digest,
        ));
    }
    for (key, want) in expected {
        if digest(receipt, key, place).ok()? != *want {
            return None;
        }
    }
    let schema = string(receipt, "input_observation_schema", place).ok()?;
    let count = positive_integer(receipt, "output_class_count", place).ok()?;
    if field && lookup(receipt, "status") != Some(&Json::Str("green".to_owned())) {
        return None;
    }
    let external = if field {
        &desired.field_evaluation_receipt_digest
    } else {
        &desired.evaluation_receipt_digest
    };
    let matches = schema == desired.input_observation_schema
        && count == desired.output_class_count
        && model_selection_digest(document).ok()? == *external;
    matches.then_some(())
}

/// Returns the `evaluation` and `field` receipt digests. Python's final
/// "receipt identities are incomplete" check is unreachable: two receipts,
/// each recorded under a distinct identity or refused as duplicated.
pub(super) fn verify_required_members(
    root: &Path,
    members: &[Json],
    receipts: &[Json],
    desired: &ModelSelection,
) -> Admission<BTreeMap<String, String>> {
    if members.is_empty() || receipts.len() != 2 {
        return refuse(AdmissionKind::ReceiptsMissing, "");
    }
    let mut observed = BTreeMap::new();
    for receipt in receipts {
        let path = match receipt {
            Json::Object(entry) => match lookup(entry, "path") {
                Some(Json::Str(path)) => path,
                _ => return refuse(AdmissionKind::ReceiptInvalid, ""),
            },
            _ => return refuse(AdmissionKind::ReceiptInvalid, ""),
        };
        let raw = read_regular(&root.join(path), path)?;
        let Some(document) = parse_json(&raw) else {
            return refuse(AdmissionKind::ReceiptNotValid, path);
        };
        let Ok(canonical) = Serialiser::ModelSelection.canonical(&document) else {
            return refuse(AdmissionKind::ReceiptNotValid, path);
        };
        if format!("{canonical}\n").as_bytes() != raw.as_slice() {
            return refuse(AdmissionKind::ReceiptNotCanonical, path);
        }
        let field = matches!(&document, Json::Object(m) if lookup(m, "status").is_some());
        let identity = if field { "field" } else { "evaluation" };
        if observed.contains_key(identity) {
            return refuse(AdmissionKind::ReceiptDuplicated, "");
        }
        if validate(desired, &document, field).is_none() {
            return refuse(AdmissionKind::ReceiptNotValid, path);
        }
        observed.insert(identity.to_owned(), sha256_hex(canonical.as_bytes()));
    }
    Ok(observed)
}

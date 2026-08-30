from __future__ import annotations

import pytest

from contracts.model_selection import (
    ContractError,
    canonical_digest,
    canonical_json_bytes,
    compare_desired_to_applied,
    parse_applied_model_selection,
    parse_model_selection,
    validate_applied_against_desired,
    validate_evaluation_receipt_identity,
    validate_field_receipt_identity,
)
from worker.runtime.provenance.model_bundle import (
    desired_model_bundle_from_selection_document,
    runtime_revision_digest,
)

WORKER_IMAGE_DIGEST = "a" * 64
BUNDLE_PAYLOAD_DIGEST = "b" * 64
DATASET_PAYLOAD_DIGEST = "c" * 64
SEED_RULE_DIGEST = "f" * 64
CALIBRATION_DIGEST = "1" * 64
CONFORMANCE_DIGEST = "2" * 64
CLASS_ORDER_DIGEST = "3" * 64
INPUT_CONTRACT_DIGEST = "4" * 64
FALL_POLICY_DIGEST = "5" * 64
CONFIG_REVISION_DIGEST = "6" * 64
RESTART_REVISION_DIGEST = "7" * 64
MISMATCH_DIGEST = "8" * 64


def _evaluation_document() -> dict[str, object]:
    return {
        "bundle_payload_digest": BUNDLE_PAYLOAD_DIGEST,
        "dataset_payload_digest": DATASET_PAYLOAD_DIGEST,
        "calibration_digest": CALIBRATION_DIGEST,
        "conformance_digest": CONFORMANCE_DIGEST,
        "class_order_digest": CLASS_ORDER_DIGEST,
        "input_contract_digest": INPUT_CONTRACT_DIGEST,
        "fall_policy_v2_digest": FALL_POLICY_DIGEST,
    }


EVALUATION_RECEIPT_DIGEST = canonical_digest(_evaluation_document())
FIELD_RECEIPT_DIGEST = canonical_digest(
    {
        **_evaluation_document(),
        "evaluation_receipt_digest": EVALUATION_RECEIPT_DIGEST,
        "status": "green",
        "selected_deployment_seed": "seed-0004",
        "selected_deployment_seed_rule_digest": SEED_RULE_DIGEST,
    }
)


def desired_raw() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_family": "gru-pose-bbox",
        "worker_image_digest": WORKER_IMAGE_DIGEST,
        "bundle_path": f"bundles/{BUNDLE_PAYLOAD_DIGEST}",
        "bundle_payload_digest": BUNDLE_PAYLOAD_DIGEST,
        "dataset_publication": {
            "hf_repo": "seeon/golden73",
            "hf_revision": "0" * 40,
            "payload_digest": DATASET_PAYLOAD_DIGEST,
        },
        "evaluation_receipt_digest": EVALUATION_RECEIPT_DIGEST,
        "field_evaluation_receipt_digest": FIELD_RECEIPT_DIGEST,
        "selected_deployment_seed": "seed-0004",
        "selected_deployment_seed_rule_digest": SEED_RULE_DIGEST,
        "calibration_digest": CALIBRATION_DIGEST,
        "conformance_digest": CONFORMANCE_DIGEST,
        "class_order_digest": CLASS_ORDER_DIGEST,
        "input_contract_digest": INPUT_CONTRACT_DIGEST,
        "fall_policy_v2_digest": FALL_POLICY_DIGEST,
        "config_revision": CONFIG_REVISION_DIGEST,
        "restart_revision": RESTART_REVISION_DIGEST,
    }


def evaluation_receipt() -> dict[str, object]:
    return _evaluation_document()


def field_receipt() -> dict[str, object]:
    return {
        "status": "green",
        "bundle_payload_digest": BUNDLE_PAYLOAD_DIGEST,
        "dataset_payload_digest": DATASET_PAYLOAD_DIGEST,
        "evaluation_receipt_digest": EVALUATION_RECEIPT_DIGEST,
        "selected_deployment_seed": "seed-0004",
        "selected_deployment_seed_rule_digest": SEED_RULE_DIGEST,
        "calibration_digest": CALIBRATION_DIGEST,
        "conformance_digest": CONFORMANCE_DIGEST,
        "class_order_digest": CLASS_ORDER_DIGEST,
        "input_contract_digest": INPUT_CONTRACT_DIGEST,
        "fall_policy_v2_digest": FALL_POLICY_DIGEST,
    }


def applied_raw(desired: object) -> dict[str, object]:
    raw = desired_raw()
    raw.pop("model_family")
    raw["desired_selection_digest"] = desired.digest
    raw.update(
        {
            "boot_id": "boot-1",
            "restart_id": "restart-1",
            "verified_at": "2026-08-31T00:00:00Z",
            "status": "match",
            "reasons": [],
        }
    )
    return raw


def test_canonical_identity_is_key_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    assert parse_model_selection(desired_raw()).digest == canonical_digest(desired_raw())


def test_runtime_admission_selection_parser_preserves_g001_identity_projection() -> None:
    desired = desired_model_bundle_from_selection_document(desired_raw())
    assert desired.bundle_sha256 == BUNDLE_PAYLOAD_DIGEST
    assert desired.identities["dataset"] == DATASET_PAYLOAD_DIGEST
    assert desired.identities["worker"] == WORKER_IMAGE_DIGEST
    assert desired.identities["config"] == CONFIG_REVISION_DIGEST
    assert desired.identities["restart"] == RESTART_REVISION_DIGEST
    assert runtime_revision_digest("restart", 4) == runtime_revision_digest("restart", 4)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("worker_image_digest", "sha256:" + WORKER_IMAGE_DIGEST),
        ("worker_image_digest", "latest"),
    ],
)
def test_desired_rejects_mutable_or_noncanonical_digest(key: str, value: str) -> None:
    raw = desired_raw()
    raw[key] = value
    with pytest.raises(ContractError):
        parse_model_selection(raw)


def test_desired_rejects_mutable_hf_ref_and_bundle_path_mismatch() -> None:
    raw = desired_raw()
    raw["dataset_publication"]["hf_revision"] = "main"
    with pytest.raises(ContractError):
        parse_model_selection(raw)

    raw = desired_raw()
    raw["bundle_path"] = f"bundles/{WORKER_IMAGE_DIGEST}"
    with pytest.raises(ContractError):
        parse_model_selection(raw)


def test_field_receipt_must_be_present_green_and_identity_bound() -> None:
    desired = parse_model_selection(desired_raw())
    with pytest.raises(ContractError):
        validate_field_receipt_identity(desired, None)

    red = field_receipt()
    red["status"] = "red"
    with pytest.raises(ContractError):
        validate_field_receipt_identity(desired, red)

    wrong_seed = field_receipt()
    wrong_seed["selected_deployment_seed"] = "seed-0005"
    with pytest.raises(ContractError):
        validate_field_receipt_identity(desired, wrong_seed)

    missing_evaluation_binding = field_receipt()
    missing_evaluation_binding.pop("evaluation_receipt_digest")
    with pytest.raises(ContractError):
        validate_field_receipt_identity(desired, missing_evaluation_binding)


def test_sidecar_and_evaluation_receipts_cannot_drift_from_desired() -> None:
    desired = parse_model_selection(desired_raw())
    evaluation = evaluation_receipt()
    assert "receipt_digest" not in evaluation
    assert "evaluation_receipt_digest" not in evaluation
    assert canonical_digest(evaluation) == desired.evaluation_receipt_digest
    validate_evaluation_receipt_identity(desired, evaluation)

    evaluation["calibration_digest"] = MISMATCH_DIGEST
    with pytest.raises(ContractError):
        validate_evaluation_receipt_identity(desired, evaluation)

    field = field_receipt()
    assert "receipt_digest" not in field
    assert canonical_digest(field) == desired.field_evaluation_receipt_digest
    field["evaluation_receipt_digest"] = MISMATCH_DIGEST
    with pytest.raises(ContractError):
        validate_field_receipt_identity(desired, field)


def test_desired_and_applied_matching_proof_succeeds() -> None:
    desired = parse_model_selection(desired_raw())
    applied = parse_applied_model_selection(applied_raw(desired))
    assert compare_desired_to_applied(desired, applied) == ()
    assert validate_applied_against_desired(desired, applied) == ()


def test_applied_mismatch_is_terminal_proof_not_authority() -> None:
    desired = parse_model_selection(desired_raw())
    raw = applied_raw(desired)
    raw["bundle_path"] = f"bundles/{MISMATCH_DIGEST}"
    raw["bundle_payload_digest"] = MISMATCH_DIGEST
    raw["status"] = "mismatch"
    raw["reasons"] = ["bundle_path", "bundle_payload_digest"]
    applied = parse_applied_model_selection(raw)
    assert validate_applied_against_desired(desired, applied) == (
        "bundle_path",
        "bundle_payload_digest",
    )

    assert desired.bundle_payload_digest == BUNDLE_PAYLOAD_DIGEST
    assert desired.digest == canonical_digest(desired_raw())


def test_applied_cannot_claim_match_when_identity_or_reasons_disagree() -> None:
    desired = parse_model_selection(desired_raw())
    raw = applied_raw(desired)
    raw["fall_policy_v2_digest"] = MISMATCH_DIGEST
    applied = parse_applied_model_selection(raw)
    with pytest.raises(ContractError):
        validate_applied_against_desired(desired, applied)

    raw = applied_raw(desired)
    raw["fall_policy_v2_digest"] = MISMATCH_DIGEST
    raw["status"] = "mismatch"
    raw["reasons"] = ["bundle_payload_digest"]
    applied = parse_applied_model_selection(raw)
    with pytest.raises(ContractError):
        validate_applied_against_desired(desired, applied)

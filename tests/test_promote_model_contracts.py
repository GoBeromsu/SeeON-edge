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
from worker.runtime.provenance.model_bundle import desired_model_bundle_from_selection_document

BUNDLE_SHA256 = "b" * 64
BUNDLE_MEMBERS_DIGEST = "a" * 64
DATASET_PAYLOAD_DIGEST = "c" * 64
CALIBRATION_DIGEST = "1" * 64
CONFORMANCE_DIGEST = "2" * 64
OUTPUT_SEMANTICS_DIGEST = "3" * 64
POLICY_DIGEST = "5" * 64
MISMATCH_DIGEST = "8" * 64


def _evaluation_document() -> dict[str, object]:
    return {
        "bundle_sha256": BUNDLE_SHA256,
        "bundle_members_digest": BUNDLE_MEMBERS_DIGEST,
        "dataset_payload_digest": DATASET_PAYLOAD_DIGEST,
        "calibration_digest": CALIBRATION_DIGEST,
        "conformance_digest": CONFORMANCE_DIGEST,
        "input_observation_schema": "pose-bbox56.v1",
        "output_class_count": 2,
        "output_class_semantics_digest": OUTPUT_SEMANTICS_DIGEST,
        "policy_digest": POLICY_DIGEST,
    }


EVALUATION_RECEIPT_DIGEST = canonical_digest(_evaluation_document())
FIELD_RECEIPT_DIGEST = canonical_digest(
    {
        **_evaluation_document(),
        "evaluation_receipt_digest": EVALUATION_RECEIPT_DIGEST,
        "status": "green",
    }
)


def desired_raw() -> dict[str, object]:
    return {
        "schema_version": 2,
        "model_publication": {
            "source_locator": "seeon/fall-model",
            "revision": "0" * 40,
            "bundle_sha256": BUNDLE_SHA256,
        },
        "bundle_members_digest": BUNDLE_MEMBERS_DIGEST,
        "dataset_publication": {
            "source_locator": "seeon/golden73",
            "revision": "1" * 40,
            "payload_digest": DATASET_PAYLOAD_DIGEST,
        },
        "evaluation_receipt_digest": EVALUATION_RECEIPT_DIGEST,
        "field_evaluation_receipt_digest": FIELD_RECEIPT_DIGEST,
        "calibration_digest": CALIBRATION_DIGEST,
        "conformance_digest": CONFORMANCE_DIGEST,
        "input_observation_schema": "pose-bbox56.v1",
        "output_class_count": 2,
        "output_class_semantics_digest": OUTPUT_SEMANTICS_DIGEST,
        "policy_digest": POLICY_DIGEST,
        "runtime_format": "opaque-bundle-format",
        "bundle_format": "bundle-manifest/proxy-v0",
        "preprocessing_identity": "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        "transition_threshold": 0.5,
        "threshold_source": "default",
    }


def field_receipt() -> dict[str, object]:
    return {
        **_evaluation_document(),
        "evaluation_receipt_digest": EVALUATION_RECEIPT_DIGEST,
        "status": "green",
    }


def applied_raw(desired: object) -> dict[str, object]:
    return {
        **desired_raw(),
        "desired_selection_digest": desired.digest,
        "boot_id": "boot-1",
        "restart_id": "restart-1",
        "verified_at": "2026-08-31T00:00:00Z",
        "status": "match",
        "reasons": [],
    }


def test_canonical_identity_is_key_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    assert parse_model_selection(desired_raw()).digest == canonical_digest(desired_raw())


def test_runtime_admission_selection_parser_preserves_content_and_io_identity() -> None:
    desired = desired_model_bundle_from_selection_document(desired_raw())
    assert desired.bundle_sha256 == BUNDLE_SHA256
    assert desired.identities["dataset"] == DATASET_PAYLOAD_DIGEST
    assert desired.identities["input"] == "pose-bbox56.v1"
    assert desired.selection is not None
    assert desired.selection.runtime_format == "opaque-bundle-format"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model_publication", "revision"), "main"),
        (("model_publication", "bundle_sha256"), "sha256:" + BUNDLE_SHA256),
        (("output_class_count",), 0),
        (("runtime_format",), ""),
    ],
)
def test_desired_rejects_nonimmutable_or_invalid_contract_values(
    path: tuple[str, ...], value: object
) -> None:
    raw = desired_raw()
    target: dict[str, object] = raw
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    with pytest.raises(ContractError):
        parse_model_selection(raw)


def test_contract_rejects_removed_architecture_seed_and_runtime_pin_fields() -> None:
    for key, value in (
        ("model_family", "architecture-name"),
        ("selected_deployment_seed", "seed-0004"),
        ("worker_image_digest", "a" * 64),
        ("config_revision", "a" * 64),
        ("restart_revision", "a" * 64),
    ):
        raw = desired_raw()
        raw[key] = value
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

    wrong = field_receipt()
    wrong["output_class_count"] = 3
    with pytest.raises(ContractError):
        validate_field_receipt_identity(desired, wrong)


def test_receipts_cannot_drift_from_desired() -> None:
    desired = parse_model_selection(desired_raw())
    evaluation = _evaluation_document()
    assert canonical_digest(evaluation) == desired.evaluation_receipt_digest
    validate_evaluation_receipt_identity(desired, evaluation)
    evaluation["policy_digest"] = MISMATCH_DIGEST
    with pytest.raises(ContractError):
        validate_evaluation_receipt_identity(desired, evaluation)


def test_desired_and_applied_matching_proof_succeeds() -> None:
    desired = parse_model_selection(desired_raw())
    applied = parse_applied_model_selection(applied_raw(desired))
    assert compare_desired_to_applied(desired, applied) == ()
    assert validate_applied_against_desired(desired, applied) == ()


def test_applied_mismatch_is_terminal_proof_not_authority() -> None:
    desired = parse_model_selection(desired_raw())
    raw = applied_raw(desired)
    raw["model_publication"] = {
        **raw["model_publication"],
        "bundle_sha256": MISMATCH_DIGEST,
    }
    raw["status"] = "mismatch"
    raw["reasons"] = ["model_publication"]
    applied = parse_applied_model_selection(raw)
    assert validate_applied_against_desired(desired, applied) == ("model_publication",)

"""Pure validation for immutable desired and derived applied model identities.

This module deliberately does not read files, contact a registry, or import runtime
code.  The committed model-selection document is desired authority.  An applied
selection is only a terminal observation that can be compared with that authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

SCHEMA_VERSION: Final = 1
MODEL_FAMILY: Final = "gru-pose-bbox"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_HF_REF_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_HF_REPO_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_BUNDLE_PATH_RE: Final = re.compile(r"^bundles/([0-9a-f]{64})$")
_TIMESTAMP_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class ContractError(ValueError):
    """A promotion identity is absent, mutable, or internally inconsistent."""


@dataclass(frozen=True)
class DatasetPublication:
    hf_repo: str
    hf_revision: str
    payload_digest: str


@dataclass(frozen=True)
class ModelSelection:
    worker_image_digest: str
    bundle_path: str
    bundle_payload_digest: str
    dataset_publication: DatasetPublication
    evaluation_receipt_digest: str
    field_evaluation_receipt_digest: str
    selected_deployment_seed: str
    selected_deployment_seed_rule_digest: str
    calibration_digest: str
    conformance_digest: str
    class_order_digest: str
    input_contract_digest: str
    fall_policy_v2_digest: str
    config_revision: str
    restart_revision: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_family": MODEL_FAMILY,
            "worker_image_digest": self.worker_image_digest,
            "bundle_path": self.bundle_path,
            "bundle_payload_digest": self.bundle_payload_digest,
            "dataset_publication": {
                "hf_repo": self.dataset_publication.hf_repo,
                "hf_revision": self.dataset_publication.hf_revision,
                "payload_digest": self.dataset_publication.payload_digest,
            },
            "evaluation_receipt_digest": self.evaluation_receipt_digest,
            "field_evaluation_receipt_digest": self.field_evaluation_receipt_digest,
            "selected_deployment_seed": self.selected_deployment_seed,
            "selected_deployment_seed_rule_digest": (self.selected_deployment_seed_rule_digest),
            "calibration_digest": self.calibration_digest,
            "conformance_digest": self.conformance_digest,
            "class_order_digest": self.class_order_digest,
            "input_contract_digest": self.input_contract_digest,
            "fall_policy_v2_digest": self.fall_policy_v2_digest,
            "config_revision": self.config_revision,
            "restart_revision": self.restart_revision,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.as_dict())


@dataclass(frozen=True)
class AppliedModelSelection:
    desired_selection_digest: str
    worker_image_digest: str
    bundle_path: str
    bundle_payload_digest: str
    dataset_publication: DatasetPublication
    evaluation_receipt_digest: str
    field_evaluation_receipt_digest: str
    selected_deployment_seed: str
    selected_deployment_seed_rule_digest: str
    calibration_digest: str
    conformance_digest: str
    class_order_digest: str
    input_contract_digest: str
    fall_policy_v2_digest: str
    config_revision: str
    restart_revision: str
    boot_id: str
    restart_id: str
    verified_at: str
    status: str
    reasons: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    """Return the one JSON encoding used when hashing a contract document."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc
    return encoded.encode("ascii")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(raw: object, where: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ContractError(f"{where} must be an object")
    return raw


def _exact_keys(raw: Mapping[str, object], expected: frozenset[str], where: str) -> None:
    actual = frozenset(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ContractError(f"{where} keys differ: missing={missing}, unexpected={unexpected}")


def _string(raw: Mapping[str, object], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}.{key} must be a non-empty string")
    return value


def _digest(raw: Mapping[str, object], key: str, where: str) -> str:
    value = _string(raw, key, where)
    if _SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{where}.{key} must be a lowercase 64-hex SHA-256 digest")
    return value


def _dataset_publication(raw: object, where: str) -> DatasetPublication:
    value = _object(raw, where)
    _exact_keys(value, frozenset({"hf_repo", "hf_revision", "payload_digest"}), where)
    hf_repo = _string(value, "hf_repo", where)
    if _HF_REPO_RE.fullmatch(hf_repo) is None:
        raise ContractError(f"{where}.hf_repo must be an owner/repository name")
    hf_revision = _string(value, "hf_revision", where)
    if _HF_REF_RE.fullmatch(hf_revision) is None:
        raise ContractError(f"{where}.hf_revision must be a lowercase 40-hex immutable ref")
    return DatasetPublication(hf_repo, hf_revision, _digest(value, "payload_digest", where))


def _bundle_path(raw: Mapping[str, object], digest: str, where: str) -> str:
    path = _string(raw, "bundle_path", where)
    match = _BUNDLE_PATH_RE.fullmatch(path)
    if match is None or match.group(1) != digest:
        raise ContractError(f"{where}.bundle_path must equal bundles/<bundle_payload_digest>")
    return path


def _desired_fields(raw: Mapping[str, object], where: str) -> dict[str, object]:
    expected = frozenset(
        {
            "schema_version",
            "model_family",
            "worker_image_digest",
            "bundle_path",
            "bundle_payload_digest",
            "dataset_publication",
            "evaluation_receipt_digest",
            "field_evaluation_receipt_digest",
            "selected_deployment_seed",
            "selected_deployment_seed_rule_digest",
            "calibration_digest",
            "conformance_digest",
            "class_order_digest",
            "input_contract_digest",
            "fall_policy_v2_digest",
            "config_revision",
            "restart_revision",
        }
    )
    _exact_keys(raw, expected, where)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{where}.schema_version must be {SCHEMA_VERSION}")
    if raw.get("model_family") != MODEL_FAMILY:
        raise ContractError(f"{where}.model_family must be {MODEL_FAMILY!r}")
    bundle_payload_digest = _digest(raw, "bundle_payload_digest", where)
    return {
        "worker_image_digest": _digest(raw, "worker_image_digest", where),
        "bundle_path": _bundle_path(raw, bundle_payload_digest, where),
        "bundle_payload_digest": bundle_payload_digest,
        "dataset_publication": _dataset_publication(
            raw.get("dataset_publication"), f"{where}.dataset_publication"
        ),
        "evaluation_receipt_digest": _digest(raw, "evaluation_receipt_digest", where),
        "field_evaluation_receipt_digest": _digest(raw, "field_evaluation_receipt_digest", where),
        "selected_deployment_seed": _string(raw, "selected_deployment_seed", where),
        "selected_deployment_seed_rule_digest": _digest(
            raw, "selected_deployment_seed_rule_digest", where
        ),
        "calibration_digest": _digest(raw, "calibration_digest", where),
        "conformance_digest": _digest(raw, "conformance_digest", where),
        "class_order_digest": _digest(raw, "class_order_digest", where),
        "input_contract_digest": _digest(raw, "input_contract_digest", where),
        "fall_policy_v2_digest": _digest(raw, "fall_policy_v2_digest", where),
        "config_revision": _digest(raw, "config_revision", where),
        "restart_revision": _digest(raw, "restart_revision", where),
    }


def parse_model_selection(raw: object) -> ModelSelection:
    """Validate the sole desired authority; no receipt can replace this document."""
    fields = _desired_fields(_object(raw, "model-selection"), "model-selection")
    return ModelSelection(**fields)  # type: ignore[arg-type]


def parse_applied_model_selection(raw: object) -> AppliedModelSelection:
    """Validate a derived applied proof without treating it as activation input."""
    value = _object(raw, "applied-model-selection")
    expected = frozenset(
        {
            "schema_version",
            "desired_selection_digest",
            "worker_image_digest",
            "bundle_path",
            "bundle_payload_digest",
            "dataset_publication",
            "evaluation_receipt_digest",
            "field_evaluation_receipt_digest",
            "selected_deployment_seed",
            "selected_deployment_seed_rule_digest",
            "calibration_digest",
            "conformance_digest",
            "class_order_digest",
            "input_contract_digest",
            "fall_policy_v2_digest",
            "config_revision",
            "restart_revision",
            "boot_id",
            "restart_id",
            "verified_at",
            "status",
            "reasons",
        }
    )
    _exact_keys(value, expected, "applied-model-selection")
    projected = _desired_fields(
        {
            "schema_version": SCHEMA_VERSION,
            "model_family": MODEL_FAMILY,
            **{
                key: value[key]
                for key in (
                    "worker_image_digest",
                    "bundle_path",
                    "bundle_payload_digest",
                    "dataset_publication",
                    "evaluation_receipt_digest",
                    "field_evaluation_receipt_digest",
                    "selected_deployment_seed",
                    "selected_deployment_seed_rule_digest",
                    "calibration_digest",
                    "conformance_digest",
                    "class_order_digest",
                    "input_contract_digest",
                    "fall_policy_v2_digest",
                    "config_revision",
                    "restart_revision",
                )
            },
        },
        "applied-model-selection",
    )
    reasons_raw = value.get("reasons")
    if not isinstance(reasons_raw, list) or any(
        not isinstance(reason, str) for reason in reasons_raw
    ):
        raise ContractError("applied-model-selection.reasons must be an array of strings")
    reasons = tuple(reasons_raw)
    if len(set(reasons)) != len(reasons):
        raise ContractError("applied-model-selection.reasons must not contain duplicates")
    status = _string(value, "status", "applied-model-selection")
    if status not in {"match", "mismatch"}:
        raise ContractError("applied-model-selection.status must be match or mismatch")
    if status == "match" and reasons:
        raise ContractError("applied-model-selection match status cannot have reasons")
    if status == "mismatch" and not reasons:
        raise ContractError("applied-model-selection mismatch status requires reasons")
    verified_at = _string(value, "verified_at", "applied-model-selection")
    if _TIMESTAMP_RE.fullmatch(verified_at) is None:
        raise ContractError("applied-model-selection.verified_at must be an RFC3339 UTC timestamp")
    return AppliedModelSelection(
        desired_selection_digest=_digest(
            value, "desired_selection_digest", "applied-model-selection"
        ),
        **projected,  # type: ignore[arg-type]
        boot_id=_string(value, "boot_id", "applied-model-selection"),
        restart_id=_string(value, "restart_id", "applied-model-selection"),
        verified_at=verified_at,
        status=status,
        reasons=reasons,
    )


def _receipt_identity(raw: object, where: str, *, field: bool) -> Mapping[str, object]:
    value = _object(raw, where)
    expected = {
        "bundle_payload_digest",
        "dataset_payload_digest",
        "calibration_digest",
        "conformance_digest",
        "class_order_digest",
        "input_contract_digest",
        "fall_policy_v2_digest",
    }
    if field:
        expected |= {
            "evaluation_receipt_digest",
            "status",
            "selected_deployment_seed",
            "selected_deployment_seed_rule_digest",
        }
    _exact_keys(value, frozenset(expected), where)
    for key in expected - {"status", "selected_deployment_seed"}:
        _digest(value, key, where)
    if field:
        if value.get("status") != "green":
            raise ContractError(f"{where}.status must be canonical green")
        _string(value, "selected_deployment_seed", where)
    return value


def _require_receipt_matches(
    desired: ModelSelection, receipt: Mapping[str, object], *, field: bool
) -> None:
    expected = {
        "bundle_payload_digest": desired.bundle_payload_digest,
        "dataset_payload_digest": desired.dataset_publication.payload_digest,
        "calibration_digest": desired.calibration_digest,
        "conformance_digest": desired.conformance_digest,
        "class_order_digest": desired.class_order_digest,
        "input_contract_digest": desired.input_contract_digest,
        "fall_policy_v2_digest": desired.fall_policy_v2_digest,
    }
    if field:
        expected |= {
            "evaluation_receipt_digest": desired.evaluation_receipt_digest,
            "selected_deployment_seed": desired.selected_deployment_seed,
            "selected_deployment_seed_rule_digest": (desired.selected_deployment_seed_rule_digest),
        }
    mismatches = sorted(
        key for key, expected_value in expected.items() if receipt[key] != expected_value
    )
    if mismatches:
        raise ContractError(
            f"receipt identity does not match desired selection: {', '.join(mismatches)}"
        )


def validate_evaluation_receipt_identity(desired: ModelSelection, raw: object) -> None:
    """Require an evaluation receipt to bind every immutable desired identity."""
    receipt = _receipt_identity(raw, "evaluation-receipt", field=False)
    if canonical_digest(receipt) != desired.evaluation_receipt_digest:
        raise ContractError("evaluation receipt external digest does not match desired selection")
    _require_receipt_matches(desired, receipt, field=False)


def validate_field_receipt_identity(desired: ModelSelection, raw: object) -> None:
    """Refuse absent, red, or differently-bound selected-seed field evidence."""
    receipt = _receipt_identity(raw, "field-evaluation-receipt", field=True)
    if canonical_digest(receipt) != desired.field_evaluation_receipt_digest:
        raise ContractError("field receipt external digest does not match desired selection")
    _require_receipt_matches(desired, receipt, field=True)


def compare_desired_to_applied(
    desired: ModelSelection, applied: AppliedModelSelection
) -> tuple[str, ...]:
    """Return exact terminal identity discrepancies; empty means an observed match."""
    expected = {
        "desired_selection_digest": desired.digest,
        "worker_image_digest": desired.worker_image_digest,
        "bundle_path": desired.bundle_path,
        "bundle_payload_digest": desired.bundle_payload_digest,
        "dataset_publication.hf_repo": desired.dataset_publication.hf_repo,
        "dataset_publication.hf_revision": desired.dataset_publication.hf_revision,
        "dataset_publication.payload_digest": (desired.dataset_publication.payload_digest),
        "evaluation_receipt_digest": desired.evaluation_receipt_digest,
        "field_evaluation_receipt_digest": (desired.field_evaluation_receipt_digest),
        "selected_deployment_seed": desired.selected_deployment_seed,
        "selected_deployment_seed_rule_digest": (desired.selected_deployment_seed_rule_digest),
        "calibration_digest": desired.calibration_digest,
        "conformance_digest": desired.conformance_digest,
        "class_order_digest": desired.class_order_digest,
        "input_contract_digest": desired.input_contract_digest,
        "fall_policy_v2_digest": desired.fall_policy_v2_digest,
        "config_revision": desired.config_revision,
        "restart_revision": desired.restart_revision,
    }
    actual = {
        "desired_selection_digest": applied.desired_selection_digest,
        "worker_image_digest": applied.worker_image_digest,
        "bundle_path": applied.bundle_path,
        "bundle_payload_digest": applied.bundle_payload_digest,
        "dataset_publication.hf_repo": applied.dataset_publication.hf_repo,
        "dataset_publication.hf_revision": applied.dataset_publication.hf_revision,
        "dataset_publication.payload_digest": (applied.dataset_publication.payload_digest),
        "evaluation_receipt_digest": applied.evaluation_receipt_digest,
        "field_evaluation_receipt_digest": (applied.field_evaluation_receipt_digest),
        "selected_deployment_seed": applied.selected_deployment_seed,
        "selected_deployment_seed_rule_digest": (applied.selected_deployment_seed_rule_digest),
        "calibration_digest": applied.calibration_digest,
        "conformance_digest": applied.conformance_digest,
        "class_order_digest": applied.class_order_digest,
        "input_contract_digest": applied.input_contract_digest,
        "fall_policy_v2_digest": applied.fall_policy_v2_digest,
        "config_revision": applied.config_revision,
        "restart_revision": applied.restart_revision,
    }
    return tuple(key for key in expected if actual[key] != expected[key])


def validate_applied_against_desired(
    desired: ModelSelection, applied: AppliedModelSelection
) -> tuple[str, ...]:
    """Validate terminal proof semantics; this function never activates anything."""
    mismatches = compare_desired_to_applied(desired, applied)
    if applied.status == "match":
        if mismatches:
            raise ContractError(
                "applied-model-selection claims match despite: " + ", ".join(mismatches)
            )
        return ()
    if applied.reasons != mismatches:
        raise ContractError(
            "applied-model-selection mismatch reasons must equal exact identity mismatches"
        )
    return mismatches


__all__ = [
    "MODEL_FAMILY",
    "SCHEMA_VERSION",
    "AppliedModelSelection",
    "ContractError",
    "DatasetPublication",
    "ModelSelection",
    "canonical_digest",
    "canonical_json_bytes",
    "compare_desired_to_applied",
    "parse_applied_model_selection",
    "parse_model_selection",
    "validate_applied_against_desired",
    "validate_evaluation_receipt_identity",
    "validate_field_receipt_identity",
]

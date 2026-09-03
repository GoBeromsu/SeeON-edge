"""Pure validation for immutable desired and derived applied model identities."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

SCHEMA_VERSION: Final = 2
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_LOCATOR_RE: Final = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_TIMESTAMP_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class ContractError(ValueError):
    """A promotion identity is absent, mutable, or internally inconsistent."""


@dataclass(frozen=True)
class DatasetPublication:
    source_locator: str
    revision: str
    payload_digest: str


@dataclass(frozen=True)
class ModelPublication:
    source_locator: str
    revision: str
    bundle_sha256: str


@dataclass(frozen=True)
class ModelSelection:
    model_publication: ModelPublication
    bundle_members_digest: str
    dataset_publication: DatasetPublication
    evaluation_receipt_digest: str
    field_evaluation_receipt_digest: str
    calibration_digest: str
    conformance_digest: str
    input_observation_schema: str
    output_class_count: int
    output_class_semantics_digest: str
    policy_digest: str
    runtime_format: str
    bundle_format: str
    preprocessing_identity: str
    transition_threshold: float
    threshold_source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_publication": {
                "source_locator": self.model_publication.source_locator,
                "revision": self.model_publication.revision,
                "bundle_sha256": self.model_publication.bundle_sha256,
            },
            "bundle_members_digest": self.bundle_members_digest,
            "dataset_publication": {
                "source_locator": self.dataset_publication.source_locator,
                "revision": self.dataset_publication.revision,
                "payload_digest": self.dataset_publication.payload_digest,
            },
            "evaluation_receipt_digest": self.evaluation_receipt_digest,
            "field_evaluation_receipt_digest": self.field_evaluation_receipt_digest,
            "calibration_digest": self.calibration_digest,
            "conformance_digest": self.conformance_digest,
            "input_observation_schema": self.input_observation_schema,
            "output_class_count": self.output_class_count,
            "output_class_semantics_digest": self.output_class_semantics_digest,
            "policy_digest": self.policy_digest,
            "runtime_format": self.runtime_format,
            "bundle_format": self.bundle_format,
            "preprocessing_identity": self.preprocessing_identity,
            "transition_threshold": self.transition_threshold,
            "threshold_source": self.threshold_source,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.as_dict())


@dataclass(frozen=True)
class AppliedModelSelection:
    desired_selection_digest: str
    model_publication: ModelPublication
    bundle_members_digest: str
    dataset_publication: DatasetPublication
    evaluation_receipt_digest: str
    field_evaluation_receipt_digest: str
    calibration_digest: str
    conformance_digest: str
    input_observation_schema: str
    output_class_count: int
    output_class_semantics_digest: str
    policy_digest: str
    runtime_format: str
    bundle_format: str
    preprocessing_identity: str
    transition_threshold: float
    threshold_source: str
    boot_id: str
    restart_id: str
    verified_at: str
    status: str
    reasons: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    """Return the one JSON encoding used when hashing a contract document."""
    try:
        encoded = json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
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
        raise ContractError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


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


def _publication(
    raw: object,
    where: str,
    *,
    content_key: str,
    publication_type: type[DatasetPublication] | type[ModelPublication],
) -> DatasetPublication | ModelPublication:
    value = _object(raw, where)
    _exact_keys(value, frozenset({"source_locator", "revision", content_key}), where)
    source_locator = _string(value, "source_locator", where)
    if _SOURCE_LOCATOR_RE.fullmatch(source_locator) is None:
        raise ContractError(f"{where}.source_locator must be an owner/repository name")
    revision = _string(value, "revision", where)
    if _REVISION_RE.fullmatch(revision) is None:
        raise ContractError(f"{where}.revision must be a lowercase 40-hex immutable ref")
    return publication_type(source_locator, revision, _digest(value, content_key, where))


def _positive_integer(raw: Mapping[str, object], key: str, where: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{where}.{key} must be a positive integer")
    return value


def _probability(raw: Mapping[str, object], key: str, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}.{key} must be a probability")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ContractError(f"{where}.{key} must be in [0, 1]")
    return parsed


def _threshold_source(raw: Mapping[str, object], key: str, where: str) -> str:
    value = _string(raw, key, where)
    if value not in {"default", "receipt"}:
        raise ContractError(f"{where}.{key} must be default or receipt")
    return value


_SELECTION_KEYS: Final = frozenset(
    {
        "schema_version",
        "model_publication",
        "bundle_members_digest",
        "dataset_publication",
        "evaluation_receipt_digest",
        "field_evaluation_receipt_digest",
        "calibration_digest",
        "conformance_digest",
        "input_observation_schema",
        "output_class_count",
        "output_class_semantics_digest",
        "policy_digest",
        "runtime_format",
        "bundle_format",
        "preprocessing_identity",
        "transition_threshold",
        "threshold_source",
    }
)


def _desired_fields(raw: Mapping[str, object], where: str) -> dict[str, object]:
    _exact_keys(raw, _SELECTION_KEYS, where)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{where}.schema_version must be {SCHEMA_VERSION}")
    return {
        "model_publication": _publication(
            raw.get("model_publication"),
            f"{where}.model_publication",
            content_key="bundle_sha256",
            publication_type=ModelPublication,
        ),
        "bundle_members_digest": _digest(raw, "bundle_members_digest", where),
        "dataset_publication": _publication(
            raw.get("dataset_publication"),
            f"{where}.dataset_publication",
            content_key="payload_digest",
            publication_type=DatasetPublication,
        ),
        "evaluation_receipt_digest": _digest(raw, "evaluation_receipt_digest", where),
        "field_evaluation_receipt_digest": _digest(raw, "field_evaluation_receipt_digest", where),
        "calibration_digest": _digest(raw, "calibration_digest", where),
        "conformance_digest": _digest(raw, "conformance_digest", where),
        "input_observation_schema": _string(raw, "input_observation_schema", where),
        "output_class_count": _positive_integer(raw, "output_class_count", where),
        "output_class_semantics_digest": _digest(raw, "output_class_semantics_digest", where),
        "policy_digest": _digest(raw, "policy_digest", where),
        "runtime_format": _string(raw, "runtime_format", where),
        "bundle_format": _string(raw, "bundle_format", where),
        "preprocessing_identity": _string(raw, "preprocessing_identity", where),
        "transition_threshold": _probability(raw, "transition_threshold", where),
        "threshold_source": _threshold_source(raw, "threshold_source", where),
    }


def parse_model_selection(raw: object) -> ModelSelection:
    """Validate the sole desired authority; no receipt can replace this document."""
    fields = _desired_fields(_object(raw, "model-selection"), "model-selection")
    return ModelSelection(**fields)  # type: ignore[arg-type]


def parse_applied_model_selection(raw: object) -> AppliedModelSelection:
    """Validate a derived applied proof without treating it as activation input."""
    value = _object(raw, "applied-model-selection")
    expected = _SELECTION_KEYS | frozenset(
        {"desired_selection_digest", "boot_id", "restart_id", "verified_at", "status", "reasons"}
    )
    _exact_keys(value, expected, "applied-model-selection")
    projected = _desired_fields(
        {key: value[key] for key in _SELECTION_KEYS}, "applied-model-selection"
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
    if (status == "match" and reasons) or (status == "mismatch" and not reasons):
        raise ContractError("applied-model-selection status and reasons disagree")
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
        "bundle_sha256",
        "bundle_members_digest",
        "dataset_payload_digest",
        "calibration_digest",
        "conformance_digest",
        "input_observation_schema",
        "output_class_count",
        "output_class_semantics_digest",
        "policy_digest",
    }
    if field:
        expected |= {"evaluation_receipt_digest", "status"}
    _exact_keys(value, frozenset(expected), where)
    for key in expected - {"status", "input_observation_schema", "output_class_count"}:
        _digest(value, key, where)
    _string(value, "input_observation_schema", where)
    _positive_integer(value, "output_class_count", where)
    if field and value.get("status") != "green":
        raise ContractError(f"{where}.status must be canonical green")
    return value


def _require_receipt_matches(
    desired: ModelSelection, receipt: Mapping[str, object], *, field: bool
) -> None:
    expected: dict[str, object] = {
        "bundle_sha256": desired.model_publication.bundle_sha256,
        "bundle_members_digest": desired.bundle_members_digest,
        "dataset_payload_digest": desired.dataset_publication.payload_digest,
        "calibration_digest": desired.calibration_digest,
        "conformance_digest": desired.conformance_digest,
        "input_observation_schema": desired.input_observation_schema,
        "output_class_count": desired.output_class_count,
        "output_class_semantics_digest": desired.output_class_semantics_digest,
        "policy_digest": desired.policy_digest,
    }
    if field:
        expected["evaluation_receipt_digest"] = desired.evaluation_receipt_digest
    mismatches = sorted(
        key for key, expected_value in expected.items() if receipt[key] != expected_value
    )
    if mismatches:
        raise ContractError(
            f"receipt identity does not match desired selection: {', '.join(mismatches)}"
        )


def validate_evaluation_receipt_identity(desired: ModelSelection, raw: object) -> None:
    receipt = _receipt_identity(raw, "evaluation-receipt", field=False)
    if canonical_digest(receipt) != desired.evaluation_receipt_digest:
        raise ContractError("evaluation receipt external digest does not match desired selection")
    _require_receipt_matches(desired, receipt, field=False)


def validate_field_receipt_identity(desired: ModelSelection, raw: object) -> None:
    receipt = _receipt_identity(raw, "field-evaluation-receipt", field=True)
    if canonical_digest(receipt) != desired.field_evaluation_receipt_digest:
        raise ContractError("field receipt external digest does not match desired selection")
    _require_receipt_matches(desired, receipt, field=True)


def compare_desired_to_applied(
    desired: ModelSelection, applied: AppliedModelSelection
) -> tuple[str, ...]:
    expected = {
        "desired_selection_digest": desired.digest,
        **desired.as_dict(),
    }
    actual = {
        "desired_selection_digest": applied.desired_selection_digest,
        **ModelSelection(
            model_publication=applied.model_publication,
            bundle_members_digest=applied.bundle_members_digest,
            dataset_publication=applied.dataset_publication,
            evaluation_receipt_digest=applied.evaluation_receipt_digest,
            field_evaluation_receipt_digest=applied.field_evaluation_receipt_digest,
            calibration_digest=applied.calibration_digest,
            conformance_digest=applied.conformance_digest,
            input_observation_schema=applied.input_observation_schema,
            output_class_count=applied.output_class_count,
            output_class_semantics_digest=applied.output_class_semantics_digest,
            policy_digest=applied.policy_digest,
            runtime_format=applied.runtime_format,
            bundle_format=applied.bundle_format,
            preprocessing_identity=applied.preprocessing_identity,
            transition_threshold=applied.transition_threshold,
            threshold_source=applied.threshold_source,
        ).as_dict(),
    }
    return tuple(key for key in expected if actual[key] != expected[key])


def validate_applied_against_desired(
    desired: ModelSelection, applied: AppliedModelSelection
) -> tuple[str, ...]:
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
    "SCHEMA_VERSION",
    "AppliedModelSelection",
    "ContractError",
    "DatasetPublication",
    "ModelPublication",
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

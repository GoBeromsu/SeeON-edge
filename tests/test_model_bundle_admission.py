from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from contracts.model_selection import canonical_digest
from worker.runtime.provenance.model_bundle import (
    ModelBundleAdmissionError,
    admit_model_bundle,
    desired_model_bundle_from_selection_document,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


_IDENTITIES = {
    "dataset": "1" * 64,
    "calibration": hashlib.sha256(b"{}").hexdigest(),
    "conformance": hashlib.sha256(b"{}").hexdigest(),
    "class": "4" * 64,
    "input": "pose-bbox56.v1",
    "policy": canonical_digest(None),
    "members": "6" * 64,
}


def _selection(bundle_sha256: str, evaluation: str, field: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "model_publication": {
            "source_locator": "seeon/fall-model",
            "revision": "a" * 40,
            "bundle_sha256": bundle_sha256,
        },
        "bundle_members_digest": _IDENTITIES["members"],
        "dataset_publication": {
            "source_locator": "seeon/dataset",
            "revision": "b" * 40,
            "payload_digest": _IDENTITIES["dataset"],
        },
        "evaluation_receipt_digest": evaluation,
        "field_evaluation_receipt_digest": field,
        "calibration_digest": _IDENTITIES["calibration"],
        "conformance_digest": _IDENTITIES["conformance"],
        "input_observation_schema": _IDENTITIES["input"],
        "output_class_count": 2,
        "output_class_semantics_digest": _IDENTITIES["class"],
        "policy_digest": _IDENTITIES["policy"],
        "runtime_format": "opaque-bundle-format",
        "bundle_format": "bundle-manifest/proxy-v0",
        "preprocessing_identity": "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        "transition_threshold": 0.5,
        "threshold_source": "default",
    }


def _bundle(
    tmp_path: Path,
    identities: dict[str, str] | None = None,
    members: dict[str, bytes] | None = None,
) -> tuple[Path, object]:
    identities = {**_IDENTITIES, **(identities or {})}
    payload = {"identities": identities}
    members = (
        {
            "model.pt": b"model",
            "arch.json": b"{}",
            "metadata.yaml": b"metadata",
            "input-contract.json": b"{}",
            "fall-policy-v2.json": b"{}",
            "calibration.json": b"{}",
            "conformance.json": b"{}",
        }
        if members is None
        else members
    )
    member_records = [
        {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        for path, content in members.items()
    ]
    bundle_sha256 = hashlib.sha256(
        json.dumps(
            {"members": member_records, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evaluation = {
        "bundle_sha256": bundle_sha256,
        "bundle_members_digest": identities["members"],
        "dataset_payload_digest": identities["dataset"],
        "calibration_digest": identities["calibration"],
        "conformance_digest": identities["conformance"],
        "input_observation_schema": identities["input"],
        "output_class_count": 2,
        "output_class_semantics_digest": identities["class"],
        "policy_digest": identities["policy"],
    }
    field = {
        **evaluation,
        "evaluation_receipt_digest": canonical_digest(evaluation),
        "status": "green",
    }
    root = tmp_path / "models" / "bundles" / bundle_sha256
    root.mkdir(parents=True)
    for path, content in members.items():
        (root / path).write_bytes(content)
    for path, document in (
        ("evaluation-receipt.json", evaluation),
        ("field-evaluation-receipt.json", field),
    ):
        content = _canonical(document)
        (root / path).write_bytes(content)
        member_records.append(
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        )
    manifest = {
        "schema_version": 1,
        "bundle_sha256": bundle_sha256,
        "runtime_format": "opaque-bundle-format",
        "members": member_records[: len(members)],
        "receipts": member_records[len(members) :],
        "payload": payload,
    }
    (root / "manifest.json").write_bytes(_canonical(manifest))
    return tmp_path / "models", desired_model_bundle_from_selection_document(
        _selection(bundle_sha256, canonical_digest(evaluation), canonical_digest(field))
    )


def test_admission_returns_immutable_content_proof(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    proof = admit_model_bundle(models_root, desired)
    assert proof.observed["bundle_sha256"] == desired.bundle_sha256
    assert proof.applied["identities"]["members"] == _IDENTITIES["members"]
    with pytest.raises(TypeError):
        proof.observed["bundle_sha256"] = "x"  # type: ignore[index]


@pytest.mark.parametrize("field", sorted(_IDENTITIES))
def test_each_bundle_identity_mismatch_is_fatal(tmp_path: Path, field: str) -> None:
    observed = dict(_IDENTITIES)
    observed[field] = "different" if field == "input" else "f" * 64
    models_root, desired = _bundle(tmp_path, observed)
    with pytest.raises(ModelBundleAdmissionError, match=rf"{field} identity mismatch"):
        admit_model_bundle(models_root, desired)


def test_admission_rejects_extra_bundle_member(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "unexpected").write_text("unexpected")
    with pytest.raises(ModelBundleAdmissionError, match="bundle tree"):
        admit_model_bundle(models_root, desired)


def test_admission_uses_manifest_declared_member_set(tmp_path: Path) -> None:
    models_root, desired = _bundle(
        tmp_path,
        members={
            "runtime.bin": b"model",
            "contract.json": b"{}",
            "calibration.json": b"{}",
        },
    )
    assert admit_model_bundle(models_root, desired).observed["members"] == (
        "runtime.bin",
        "contract.json",
        "calibration.json",
    )


def test_admission_refuses_selection_calibration_digest_not_matching_member_content(
    tmp_path: Path,
) -> None:
    models_root, desired = _bundle(
        tmp_path,
        members={
            "model.pt": b"model",
            "calibration.json": b"different calibration",
        },
    )

    with pytest.raises(
        ModelBundleAdmissionError,
        match=rf"selection declares {_IDENTITIES['calibration']}.*member content has "
        rf"'{hashlib.sha256(b'different calibration').hexdigest()}'",
    ):
        admit_model_bundle(models_root, desired)


def test_admission_refuses_selection_conformance_digest_not_matching_member_content(
    tmp_path: Path,
) -> None:
    models_root, desired = _bundle(
        tmp_path,
        members={
            "model.pt": b"model",
            "calibration.json": b"{}",
            "conformance.json": b"different conformance",
        },
    )

    with pytest.raises(
        ModelBundleAdmissionError,
        match=rf"selection declares {_IDENTITIES['conformance']}.*member content has "
        rf"'{hashlib.sha256(b'different conformance').hexdigest()}'",
    ):
        admit_model_bundle(models_root, desired)


def test_admission_refuses_policy_digest_not_matching_temporal_rule_content(
    tmp_path: Path,
) -> None:
    models_root, desired = _bundle(tmp_path)
    assert desired.selection is not None
    desired = replace(
        desired,
        selection=replace(desired.selection, policy_digest="5" * 64),
    )

    with pytest.raises(
        ModelBundleAdmissionError,
        match=rf"policy_digest mismatch: selection declares {'5' * 64}, "
        rf"calibration temporal_rule content has {_IDENTITIES['policy']}",
    ):
        admit_model_bundle(models_root, desired)


def test_admission_requires_the_manifest_runtime_format(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["runtime_format"] = "different-format"
    manifest_path.write_bytes(_canonical(manifest))
    with pytest.raises(ModelBundleAdmissionError, match="runtime format"):
        admit_model_bundle(models_root, desired)

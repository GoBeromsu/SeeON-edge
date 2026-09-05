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
    "calibration": hashlib.sha256(b'{"calibration": true}').hexdigest(),
    "conformance": hashlib.sha256(b'{"conformance": true}').hexdigest(),
    "class": "4" * 64,
    "input": "pose-bbox56.v1",
    "policy": canonical_digest(None),
    "members": "6" * 64,
}


def _selection(
    bundle_sha256: str,
    evaluation: str,
    field: str,
    identities: dict[str, str] = _IDENTITIES,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "model_publication": {
            "source_locator": "seeon/fall-model",
            "revision": "a" * 40,
            "bundle_sha256": bundle_sha256,
        },
        "bundle_members_digest": identities["members"],
        "dataset_publication": {
            "source_locator": "seeon/dataset",
            "revision": "b" * 40,
            "payload_digest": identities["dataset"],
        },
        "evaluation_receipt_digest": evaluation,
        "field_evaluation_receipt_digest": field,
        "calibration_digest": identities["calibration"],
        "conformance_digest": identities["conformance"],
        "input_observation_schema": identities["input"],
        "output_class_count": 2,
        "output_class_semantics_digest": identities["class"],
        "policy_digest": identities["policy"],
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
    *,
    payload_identities: dict[str, str] | None = None,
) -> tuple[Path, object]:
    """Write a bundle whose selection and on-disk payload agree, unless a test
    supplies `payload_identities` to make the bundle contradict its selection."""
    members = (
        {
            "model.pt": b"model",
            # Distinct content per member, as in a real bundle: admission binds
            # the conformance document by its content digest, so identical
            # bytes in several members would be an ambiguity, not a bundle.
            "arch.json": b'{"arch": true}',
            "metadata.yaml": b"metadata",
            "input-contract.json": b'{"input": true}',
            "fall-policy-v2.json": b'{"policy": true}',
            "calibration.json": b'{"calibration": true}',
            "conformance.json": b'{"conformance": true}',
        }
        if members is None
        else members
    )
    # The calibration and conformance identities are the content digests of
    # the members actually supplied, unless a test overrides them on purpose.
    derived = {}
    if "calibration.json" in members:
        derived["calibration"] = hashlib.sha256(members["calibration.json"]).hexdigest()
    conformance_member = next(
        (path for path in members if path.split("/")[-1].startswith("conformance")), None
    )
    if conformance_member is not None:
        derived["conformance"] = hashlib.sha256(members[conformance_member]).hexdigest()
    identities = {**_IDENTITIES, **derived, **(identities or {})}
    payload = {"identities": {**identities, **(payload_identities or {})}}
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
        (root / path).parent.mkdir(parents=True, exist_ok=True)
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
        _selection(bundle_sha256, canonical_digest(evaluation), canonical_digest(field), identities)
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
    # The bundle on disk claims one identity for this field; the selection
    # declares another. Every such contradiction is fatal at admission.
    contradiction = {field: "different" if field == "input" else "f" * 64}
    models_root, desired = _bundle(tmp_path, payload_identities=contradiction)
    with pytest.raises(ModelBundleAdmissionError, match=rf"{field} identity mismatch"):
        admit_model_bundle(models_root, desired)


def test_admission_rejects_extra_bundle_member(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "unexpected").write_text("unexpected")
    with pytest.raises(ModelBundleAdmissionError, match="bundle tree"):
        admit_model_bundle(models_root, desired)


def test_admission_uses_manifest_declared_member_set(tmp_path: Path) -> None:
    """Admission walks the members the manifest declares, whatever they are
    named - including a conformance document at a publisher-chosen path."""
    models_root, desired = _bundle(
        tmp_path,
        members={
            "runtime.bin": b"model",
            "contract.json": b'{"contract": true}',
            "calibration.json": b'{"calibration": true}',
            "conformance/pose-bbox56-v1.json": b'{"conformance": true}',
        },
    )
    assert admit_model_bundle(models_root, desired).observed["members"] == (
        "runtime.bin",
        "contract.json",
        "calibration.json",
        "conformance/pose-bbox56-v1.json",
    )


def test_admission_refuses_selection_calibration_digest_not_matching_member_content(
    tmp_path: Path,
) -> None:
    # The selection declares one calibration digest; the member on disk has
    # different content. The fixture would otherwise derive the digest from
    # the member, so the declared value is pinned explicitly.
    declared = hashlib.sha256(b'{"calibration": true}').hexdigest()
    models_root, desired = _bundle(
        tmp_path,
        identities={"calibration": declared},
        members={
            "model.pt": b"model",
            "calibration.json": b"different calibration",
            "conformance/pose-bbox56-v1.json": b'{"conformance": true}',
        },
    )

    with pytest.raises(
        ModelBundleAdmissionError,
        match=rf"selection declares {declared}.*member content has "
        rf"'{hashlib.sha256(b'different calibration').hexdigest()}'",
    ):
        admit_model_bundle(models_root, desired)


def test_admission_refuses_selection_conformance_digest_not_matching_member_content(
    tmp_path: Path,
) -> None:
    """The conformance document is bound by content, whatever the member is
    named: a declared digest that is the content of no member refuses."""
    declared = hashlib.sha256(b"{}").hexdigest()
    models_root, desired = _bundle(
        tmp_path,
        identities={"conformance": declared},
        members={
            "model.pt": b"model",
            "calibration.json": b'{"calibration": true}',
            "conformance/pose-bbox56-v1.json": b"different conformance",
        },
    )

    with pytest.raises(
        ModelBundleAdmissionError,
        match=rf"conformance_digest {declared} matches 0 bundle member",
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

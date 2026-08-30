"""Read-only content-addressed model bundle admission contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from worker.runtime.provenance.model_bundle import (
    DesiredModelBundle,
    ModelBundleAdmissionError,
    RuntimeModelObservations,
    admit_model_bundle,
    desired_model_bundle_from_selection_document,
    runtime_revision_digest,
)

_DATASET_PUBLICATION = MappingProxyType(
    {
        "hf_repo": "seeon/golden73",
        "hf_revision": "0" * 40,
        "payload_digest": "1" * 64,
    }
)

_IDENTITIES = {
    "dataset": _DATASET_PUBLICATION["payload_digest"],
    "evaluation": "2" * 64,
    "field": "3" * 64,
    "seed": "seed-0004",
    "rule": "4" * 64,
    "calibration": "5" * 64,
    "conformance": "6" * 64,
    "class": "7" * 64,
    "input": "8" * 64,
    "policy": "9" * 64,
    "config": "a" * 64,
    "restart": "b" * 64,
    "worker": "c" * 64,
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _bundle(
    tmp_path: Path, identities: dict[str, object] | None = None
) -> tuple[Path, DesiredModelBundle]:
    identities = dict(_IDENTITIES if identities is None else identities)
    content = b"weights"
    member = {
        "path": "weights/model.bin",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    payload = {
        "identities": {
            key: value
            for key, value in identities.items()
            if key not in {"worker", "config", "restart"}
        }
    }
    digest = hashlib.sha256(
        json.dumps(
            {"members": [member], "payload": payload}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    root = tmp_path / "models" / "bundles" / digest
    (root / "weights").mkdir(parents=True)
    (root / member["path"]).write_bytes(content)
    (root / "manifest.json").write_bytes(
        _canonical(
            {"bundle_sha256": digest, "members": [member], "payload": payload, "schema_version": 1}
        )
    )
    return tmp_path / "models", DesiredModelBundle(digest, _IDENTITIES)


def _observations() -> RuntimeModelObservations:
    return RuntimeModelObservations(
        _IDENTITIES["worker"], _IDENTITIES["config"], _IDENTITIES["restart"]
    )


def test_admission_returns_immutable_observed_and_desired_applied_proof(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)

    proof = admit_model_bundle(models_root, desired, _observations())

    assert proof.observed["bundle_sha256"] == desired.bundle_sha256
    assert proof.applied["identities"] == desired.identities
    with pytest.raises(TypeError):
        proof.applied["bundle_sha256"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        proof.observed["identities"]["dataset"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        desired.identities["dataset"] = "other"  # type: ignore[index]


@pytest.mark.parametrize("field", sorted(_IDENTITIES))
def test_each_desired_identity_mismatch_is_fatal_before_admission(
    tmp_path: Path, field: str
) -> None:
    observed = dict(_IDENTITIES)
    observed[field] = "different-seed" if field == "seed" else "f" * 64
    models_root, desired = _bundle(tmp_path, observed)
    observations = RuntimeModelObservations(
        observed["worker"], observed["config"], observed["restart"]
    )

    with pytest.raises(ModelBundleAdmissionError, match=rf"{field} identity mismatch"):
        admit_model_bundle(models_root, desired, observations)


def test_admission_rejects_extra_file_empty_directory_and_noncanonical_manifest(
    tmp_path: Path,
) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "extra").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ModelBundleAdmissionError, match="tree contains"):
        admit_model_bundle(models_root, desired, _observations())

    (root / "extra").unlink()
    (root / "empty").mkdir()
    with pytest.raises(ModelBundleAdmissionError, match="tree contains"):
        admit_model_bundle(models_root, desired, _observations())

    (root / "empty").rmdir()
    raw = (root / "manifest.json").read_text(encoding="utf-8")
    (root / "manifest.json").write_text(raw + " ", encoding="utf-8")
    with pytest.raises(ModelBundleAdmissionError, match="not canonical"):
        admit_model_bundle(models_root, desired, _observations())


def test_admission_rejects_symlink_and_member_path_escape(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "link").symlink_to(root / "weights/model.bin")
    with pytest.raises(ModelBundleAdmissionError, match="unsafe path"):
        admit_model_bundle(models_root, desired, _observations())

    (root / "link").unlink()
    document = json.loads((root / "manifest.json").read_bytes())
    document["members"][0]["path"] = "../escape"
    (root / "manifest.json").write_bytes(_canonical(document))
    with pytest.raises(ModelBundleAdmissionError, match="content identity mismatch"):
        admit_model_bundle(models_root, desired, _observations())


@pytest.mark.parametrize(
    ("field", "attribute"),
    [
        ("worker", "worker_image_digest"),
        ("config", "config_revision"),
        ("restart", "restart_revision"),
    ],
)
def test_runtime_observation_mismatch_is_fatal(tmp_path: Path, field: str, attribute: str) -> None:
    models_root, desired = _bundle(tmp_path)
    values = {
        "worker_image_digest": _IDENTITIES["worker"],
        "config_revision": _IDENTITIES["config"],
        "restart_revision": _IDENTITIES["restart"],
    }
    values[attribute] = "f" * 64
    with pytest.raises(ModelBundleAdmissionError, match=rf"{field} identity mismatch"):
        admit_model_bundle(models_root, desired, RuntimeModelObservations(**values))


def test_selection_parser_and_revision_digest_are_canonical() -> None:
    raw = {
        "schema_version": 1,
        "bundle_path": f"bundles/{'a' * 64}",
        "bundle_payload_digest": "a" * 64,
        "dataset_publication": _DATASET_PUBLICATION,
        "evaluation_receipt_digest": _IDENTITIES["evaluation"],
        "field_evaluation_receipt_digest": _IDENTITIES["field"],
        "selected_deployment_seed": _IDENTITIES["seed"],
        "selected_deployment_seed_rule_digest": _IDENTITIES["rule"],
        "calibration_digest": _IDENTITIES["calibration"],
        "conformance_digest": _IDENTITIES["conformance"],
        "class_order_digest": _IDENTITIES["class"],
        "input_contract_digest": _IDENTITIES["input"],
        "fall_policy_v2_digest": _IDENTITIES["policy"],
        "config_revision": _IDENTITIES["config"],
        "restart_revision": _IDENTITIES["restart"],
        "worker_image_digest": _IDENTITIES["worker"],
    }
    assert desired_model_bundle_from_selection_document(raw).bundle_sha256 == "a" * 64
    assert runtime_revision_digest("config", 7) == runtime_revision_digest("config", 7)

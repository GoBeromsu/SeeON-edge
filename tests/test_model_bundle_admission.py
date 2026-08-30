"""Read-only content-addressed model bundle admission contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worker.runtime.provenance.model_bundle import (
    DesiredModelBundle,
    ModelBundleAdmissionError,
    admit_model_bundle,
)

_IDENTITIES = {
    "dataset": "dataset-v1",
    "evaluation": "evaluation-v1",
    "field": "field-v1",
    "seed": "seed-v1",
    "rule": "rule-v1",
    "calibration": "calibration-v1",
    "conformance": "conformance-v1",
    "class": "class-v1",
    "input": "input-v1",
    "policy": "policy-v1",
    "config": "config-v1",
    "restart": "restart-v1",
    "worker": "worker-v1",
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
    payload = {"identities": identities}
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


def test_admission_returns_immutable_observed_and_desired_applied_proof(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)

    proof = admit_model_bundle(models_root, desired)

    assert proof.observed["bundle_sha256"] == desired.bundle_sha256
    assert proof.applied["identities"] == desired.identities
    with pytest.raises(TypeError):
        proof.applied["bundle_sha256"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        desired.identities["dataset"] = "other"  # type: ignore[index]


@pytest.mark.parametrize("field", sorted(_IDENTITIES))
def test_each_desired_identity_mismatch_is_fatal_before_admission(
    tmp_path: Path, field: str
) -> None:
    observed = dict(_IDENTITIES)
    observed[field] = "wrong"
    models_root, desired = _bundle(tmp_path, observed)

    with pytest.raises(ModelBundleAdmissionError, match=rf"{field} identity mismatch"):
        admit_model_bundle(models_root, desired)


def test_admission_rejects_extra_file_and_noncanonical_manifest(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "extra").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ModelBundleAdmissionError, match="tree contains"):
        admit_model_bundle(models_root, desired)

    (root / "extra").unlink()
    raw = (root / "manifest.json").read_text(encoding="utf-8")
    (root / "manifest.json").write_text(raw + " ", encoding="utf-8")
    with pytest.raises(ModelBundleAdmissionError, match="not canonical"):
        admit_model_bundle(models_root, desired)


def test_admission_rejects_symlink_and_member_path_escape(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "link").symlink_to(root / "weights/model.bin")
    with pytest.raises(ModelBundleAdmissionError, match="unsafe path"):
        admit_model_bundle(models_root, desired)

    (root / "link").unlink()
    document = json.loads((root / "manifest.json").read_bytes())
    document["members"][0]["path"] = "../escape"
    (root / "manifest.json").write_bytes(_canonical(document))
    with pytest.raises(ModelBundleAdmissionError, match="content identity mismatch"):
        admit_model_bundle(models_root, desired)

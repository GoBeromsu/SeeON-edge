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
    observed = dict(_IDENTITIES if identities is None else identities)
    member_bytes = {
        "model.pt": b"weights",
        "arch.json": b"{}",
        "metadata.yaml": b"metadata\n",
        "input-contract.json": b'{"input":"pose-bbox"}\n',
        "fall-policy-v2.json": b'{"policy":"fall-v2"}\n',
        "calibration.json": b'{"calibration":"v1"}\n',
        "conformance.json": b'{"conformance":"v1"}\n',
    }
    members = [
        {"path": path, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for path, content in member_bytes.items()
    ]
    payload = {
        "identities": {
            key: value
            for key, value in observed.items()
            if key not in {"worker", "config", "restart", "evaluation", "field"}
        }
    }
    digest = hashlib.sha256(
        json.dumps(
            {"members": members, "payload": payload}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    evaluation = {
        "bundle_payload_digest": digest,
        "dataset_payload_digest": _IDENTITIES["dataset"],
        "calibration_digest": _IDENTITIES["calibration"],
        "conformance_digest": _IDENTITIES["conformance"],
        "class_order_digest": _IDENTITIES["class"],
        "input_contract_digest": _IDENTITIES["input"],
        "fall_policy_v2_digest": _IDENTITIES["policy"],
    }
    evaluation_bytes = _canonical(evaluation)
    desired_identities = dict(_IDENTITIES)
    desired_identities["evaluation"] = hashlib.sha256(
        evaluation_bytes.removesuffix(b"\n")
    ).hexdigest()
    field = {
        **evaluation,
        "evaluation_receipt_digest": desired_identities["evaluation"],
        "status": "green",
        "selected_deployment_seed": _IDENTITIES["seed"],
        "selected_deployment_seed_rule_digest": _IDENTITIES["rule"],
    }
    field_bytes = _canonical(field)
    desired_identities["field"] = hashlib.sha256(field_bytes.removesuffix(b"\n")).hexdigest()
    receipts = [
        {
            "path": "evaluation-receipt.json",
            "size": len(evaluation_bytes),
            "sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        },
        {
            "path": "field-evaluation-receipt.json",
            "size": len(field_bytes),
            "sha256": hashlib.sha256(field_bytes).hexdigest(),
        },
    ]
    root = tmp_path / "models" / "bundles" / digest
    root.mkdir(parents=True)
    for path, content in member_bytes.items():
        (root / path).write_bytes(content)
    (root / "evaluation-receipt.json").write_bytes(evaluation_bytes)
    (root / "field-evaluation-receipt.json").write_bytes(field_bytes)
    (root / "manifest.json").write_bytes(
        _canonical(
            {
                "bundle_sha256": digest,
                "members": members,
                "payload": payload,
                "receipts": receipts,
                "schema_version": 1,
            }
        )
    )
    selection_raw = {
        "schema_version": 1,
        "model_family": "gru-pose-bbox",
        "worker_image_digest": desired_identities["worker"],
        "bundle_path": f"bundles/{digest}",
        "bundle_payload_digest": digest,
        "dataset_publication": _DATASET_PUBLICATION,
        "evaluation_receipt_digest": desired_identities["evaluation"],
        "field_evaluation_receipt_digest": desired_identities["field"],
        "selected_deployment_seed": desired_identities["seed"],
        "selected_deployment_seed_rule_digest": desired_identities["rule"],
        "calibration_digest": desired_identities["calibration"],
        "conformance_digest": desired_identities["conformance"],
        "class_order_digest": desired_identities["class"],
        "input_contract_digest": desired_identities["input"],
        "fall_policy_v2_digest": desired_identities["policy"],
        "config_revision": desired_identities["config"],
        "restart_revision": desired_identities["restart"],
    }
    return tmp_path / "models", desired_model_bundle_from_selection_document(selection_raw)


def _observations() -> RuntimeModelObservations:
    return RuntimeModelObservations(
        _IDENTITIES["worker"], _IDENTITIES["config"], _IDENTITIES["restart"]
    )


def test_admission_returns_immutable_observed_and_desired_applied_proof(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)

    proof = admit_model_bundle(models_root, desired, _observations())

    assert proof.observed["bundle_sha256"] == desired.bundle_sha256
    assert set(proof.observed["members"]) == {
        "model.pt",
        "arch.json",
        "metadata.yaml",
        "input-contract.json",
        "fall-policy-v2.json",
        "calibration.json",
        "conformance.json",
    }
    assert proof.applied["identities"] == desired.identities
    with pytest.raises(TypeError):
        proof.applied["bundle_sha256"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        proof.observed["identities"]["dataset"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        desired.identities["dataset"] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    "field",
    sorted(set(_IDENTITIES) - {"evaluation", "field"}),
)
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


def test_admission_requires_one_newline_after_canonical_receipt_json(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    receipt = root / "evaluation-receipt.json"
    receipt.write_bytes(receipt.read_bytes() + b"\n")

    with pytest.raises(ModelBundleAdmissionError, match="member mismatch"):
        admit_model_bundle(models_root, desired, _observations())


def test_admission_rejects_missing_gru_loader_member_before_runner(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "calibration.json").unlink()

    with pytest.raises(ModelBundleAdmissionError, match="unavailable"):
        admit_model_bundle(models_root, desired, _observations())


def test_admission_rejects_symlink_and_member_path_escape(tmp_path: Path) -> None:
    models_root, desired = _bundle(tmp_path)
    root = models_root / "bundles" / desired.bundle_sha256
    (root / "link").symlink_to(root / "model.pt")
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
        "model_family": "gru-pose-bbox",
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
    desired = desired_model_bundle_from_selection_document(raw)
    assert desired.bundle_sha256 == "a" * 64
    assert desired.selection is not None
    assert runtime_revision_digest("config", 7) == runtime_revision_digest("config", 7)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.pop("model_family"),
        lambda raw: raw.update(model_family="lstm"),
        lambda raw: raw["dataset_publication"].update(hf_repo="invalid"),
        lambda raw: raw.update(evaluation_receipt_digest="not-a-digest"),
        lambda raw: raw.update(selected_deployment_seed=""),
    ],
)
def test_selection_parser_rejects_every_shared_contract_violation(mutation: object) -> None:
    raw = {
        "schema_version": 1,
        "model_family": "gru-pose-bbox",
        "bundle_path": f"bundles/{'a' * 64}",
        "bundle_payload_digest": "a" * 64,
        "dataset_publication": dict(_DATASET_PUBLICATION),
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
    mutation(raw)  # type: ignore[operator]
    with pytest.raises(ModelBundleAdmissionError):
        desired_model_bundle_from_selection_document(raw)

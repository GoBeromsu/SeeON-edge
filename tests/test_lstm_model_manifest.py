from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.lstm_manifest import LstmFallManifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGED_ARTIFACT_DIR = _REPO_ROOT / "models" / "fall" / "lstm"


def _write_manifest(path: Path) -> Path:
    payload = {
        "type": "lstm",
        "framework": "pytorch",
        "mode": "sequence",
        "artifact_dir": str(path.parent),
        "weights": "model.pt",
        "architecture": "arch.json",
        "metadata": "metadata.yaml",
        "window": 3,
        "stride": 1,
        "input_shape": [3, 51],
        "operating_threshold": 0.5,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_packaged_lstm_metadata_declares_the_artifact_identity() -> None:
    metadata = yaml.safe_load(
        (_PACKAGED_ARTIFACT_DIR / "metadata.yaml").read_text(encoding="utf-8")
    )

    assert isinstance(metadata, dict)
    assert metadata["artifact_digest"] == (
        "889075695884742475b9713e3b86ba67085bb96979b64c51756ea3fd715ab57a"
    )
    assert metadata["preprocessing_identity"] == "legacy-coco17-xyc-frame-normalized-zero-fill-v1"


def test_packaged_lstm_weights_match_the_metadata_digest_when_provisioned() -> None:
    weights_path = _PACKAGED_ARTIFACT_DIR / "model.pt"
    if not weights_path.is_file():
        pytest.skip("packaged LSTM weights are not provisioned in this checkout")
    metadata = yaml.safe_load(
        (_PACKAGED_ARTIFACT_DIR / "metadata.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(metadata, dict)

    with weights_path.open("rb") as weights:
        actual_digest = hashlib.file_digest(weights, "sha256").hexdigest()

    assert actual_digest == metadata["artifact_digest"]


def test_lstm_manifest_accepts_metadata_yaml(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"placeholder")
    (tmp_path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "metadata.yaml")

    manifest = LstmFallManifest.from_yaml(manifest_path)

    assert manifest.artifact_dir == tmp_path
    assert manifest.weights_path == tmp_path / "model.pt"
    assert manifest.architecture_path == tmp_path / "arch.json"
    assert manifest.metadata_path == tmp_path / "metadata.yaml"
    assert manifest.input_shape == (3, 51)


def test_lstm_manifest_ignores_metadata_json_when_yaml_exists(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"placeholder")
    (tmp_path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "metadata.yaml")
    (tmp_path / "metadata.json").write_text("not valid JSON", encoding="utf-8")

    manifest = LstmFallManifest.from_artifact_dir(tmp_path)

    assert manifest.metadata_path == manifest_path


def test_lstm_manifest_rejects_metadata_json_fallback(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="metadata.yaml"):
        LstmFallManifest.from_artifact_dir(tmp_path)


def test_lstm_manifest_rejects_wrong_input_shape(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"placeholder")
    (tmp_path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "metadata.yaml")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["input_shape"] = [3, 45]
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="input_shape"):
        LstmFallManifest.from_yaml(manifest_path)

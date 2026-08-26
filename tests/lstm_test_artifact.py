"""Deterministic real LSTM artifact fixture builder for isolated tests."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import torch

from worker.adapters.model.torch_lstm_fall import build_lstm_module_from_arch

_TRACKED_ARTIFACT = Path(__file__).parents[1] / "models/fall/lstm"


def write_test_lstm_artifact(destination: Path) -> Path:
    """Write valid tracked metadata and real serialized model state under tmp_path."""
    destination.mkdir(parents=True)
    for name in ("arch.json", "metadata.yaml"):
        _ = shutil.copy2(_TRACKED_ARTIFACT / name, destination / name)
    with torch.random.fork_rng(devices=()):
        torch.manual_seed(0)
        module = build_lstm_module_from_arch(destination / "arch.json")
        weights = destination / "model.pt"
        torch.save(module.state_dict(), weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata = destination / "metadata.yaml"
    updated, count = re.subn(
        r'(?m)^artifact_digest: "[0-9a-f]{64}"$',
        f'artifact_digest: "{digest}"',
        metadata.read_text(encoding="utf-8"),
    )
    if count != 1:
        raise AssertionError("tracked LSTM metadata has no unique artifact digest")
    metadata.write_text(updated, encoding="utf-8")
    return destination

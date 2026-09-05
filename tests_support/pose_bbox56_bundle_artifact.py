"""Deterministic synthetic pose+bbox56 proxy bundle for tests.

Mirrors the published ``bundle-manifest/proxy-v0`` layout that
``worker.adapters.model.pose_bbox56_bundle`` verifies: every member is listed
with its sha256 and size, ``arch.json`` describes the binary GRU proxy,
and ``calibration.json`` carries the operational settings. The independent
``evaluation-receipt.json`` is research metadata. Weights are seeded so two
builds with the same arguments are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

PREPROCESSING_IDENTITY_DIGEST = "6ab6d8165fe11a374446e36c8448ff1dae32946a23715e0bb0c22d2a234877bb"
_CONFORMANCE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "models/fall/pose-bbox56-gru/conformance/pose-bbox56-v1.json"
)


class _ProxyGru(nn.Module):
    def __init__(self, encoder: nn.GRU, classifier: nn.Linear) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(window)
        return self.classifier(hidden[-1])


def write_pose_bbox56_bundle(
    root: Path,
    *,
    hidden_size: int = 4,
    num_layers: int = 1,
    temperature: float = 1.0,
    receipt_threshold: float | None = 0.05,
    promotion_eligible: bool = False,
    evaluation_receipt_threshold: float | None = None,
    evaluation_receipt_promotion_eligible: bool | None = None,
    manifest_evaluation_receipt: bool = True,
    seed: int = 7,
) -> Path:
    """Write a verifiable proxy bundle under ``root`` and return ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    encoder = nn.GRU(56, hidden_size, num_layers, batch_first=True)
    classifier = nn.Linear(hidden_size, 1)
    state = {
        **{f"encoder.{key}": value for key, value in encoder.state_dict().items()},
        **{f"classifier.{key}": value for key, value in classifier.state_dict().items()},
    }
    torch.save(state, root / "model.pt")
    torch.onnx.export(
        _ProxyGru(encoder, classifier).eval(),
        torch.zeros((1, 30, 56), dtype=torch.float32),
        root / "model.onnx",
        input_names=["window"],
        opset_version=17,
        dynamo=False,
    )
    arch = {
        "class_order": ["non_fall", "fall_transition_proxy"],
        "dropout": 0.0,
        "fallen_output": False,
        "hidden_size": hidden_size,
        "input_size": 56,
        "num_layers": num_layers,
        "promotion_eligible": promotion_eligible,
        "source_proxy": "binary_source_proxy",
        "task": "binary_source_proxy",
    }
    calibration = {
        "class_order": ["non_fall", "fall_transition_proxy"],
        "preprocessing_identity_digest": PREPROCESSING_IDENTITY_DIGEST,
        "promotion_eligible": promotion_eligible,
        "temperature": temperature,
        "temporal_rule": {"m": 5, "n": 5},
        "threshold": receipt_threshold,
    }
    receipt = {
        "class_order": ["non_fall", "fall_transition_proxy"],
        "preprocessing_identity_digest": PREPROCESSING_IDENTITY_DIGEST,
        "promotion_eligible": (
            promotion_eligible
            if evaluation_receipt_promotion_eligible is None
            else evaluation_receipt_promotion_eligible
        ),
        "schema_version": "evaluation-receipt/proxy-v0",
        "threshold": (
            receipt_threshold
            if evaluation_receipt_threshold is None
            else evaluation_receipt_threshold
        ),
    }
    (root / "arch.json").write_text(json.dumps(arch, sort_keys=True), encoding="utf-8")
    (root / "calibration.json").write_text(
        json.dumps(calibration, sort_keys=True), encoding="utf-8"
    )
    (root / "evaluation-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    (root / "metadata.yaml").write_text(
        "model_family: gru_source_proxy_v0\nresearch_only: true\n", encoding="utf-8"
    )
    conformance_path = root / "conformance/pose-bbox56-v1.json"
    conformance_path.parent.mkdir(parents=True, exist_ok=True)
    conformance_path.write_bytes(_CONFORMANCE_SOURCE.read_bytes())
    members = [
        "arch.json",
        "calibration.json",
        "conformance/pose-bbox56-v1.json",
    ]
    if manifest_evaluation_receipt:
        members.append("evaluation-receipt.json")
    members.extend(
        [
            "metadata.yaml",
            "model.onnx",
            "model.pt",
        ]
    )
    files = []
    for relative in members:
        payload = (root / relative).read_bytes()
        files.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    manifest = {
        "files": files,
        "model_family": "gru_source_proxy_v0",
        "preprocessing_identity_digest": PREPROCESSING_IDENTITY_DIGEST,
        "schema_version": "bundle-manifest/proxy-v0",
    }
    (root / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )
    return root


__all__ = ["PREPROCESSING_IDENTITY_DIGEST", "write_pose_bbox56_bundle"]

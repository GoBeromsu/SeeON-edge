"""CPU-only loader for the published pose+bbox56 proxy bundle."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from contracts.model_selection import POSE_BBOX56_PREPROCESSING_IDENTITY
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle_support import (
    member_digest,
    read_json,
    verify_bundle,
)
from worker.interfaces.fall_model import FallV2Probabilities
from worker.types import FallModelInput

_SHAPE: Final = (30, 56)


class _ProxyGru(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.GRU(56, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(value)
        return self.classifier(hidden[-1])


class PoseBbox56BundleRunner:
    """Verified, binary-source proxy exposed through the V2 three-class seam."""

    device: Final[str] = "cpu"

    def __init__(
        self,
        module: nn.Module,
        temperature: float,
        receipt_threshold: float | None,
        promotion_eligible: bool,
        artifact_digest: str,
        preprocessing_identity: str,
    ) -> None:
        self._module = module.to("cpu").eval()
        self._temperature = temperature
        self.receipt_threshold = receipt_threshold
        self.promotion_eligible = promotion_eligible
        # The composition root verifies these against the registry's pinned
        # component identity before any camera activates, so they must come
        # from the bundle itself, never from a constant in this adapter.
        self.artifact_digest = artifact_digest
        self.preprocessing_identity = preprocessing_identity

    @classmethod
    def from_artifact_dir(
        cls, artifact_dir: str | Path, device: str = "cpu"
    ) -> PoseBbox56BundleRunner:
        if device != "cpu":
            raise ModelLoadError("pose-bbox56 proxy bundle is pinned to cpu")
        root = Path(artifact_dir).expanduser().resolve()
        manifest = read_json(root / "bundle-manifest.json")
        verify_bundle(root, manifest)
        arch = read_json(root / "arch.json")
        calibration = read_json(root / "calibration.json")
        receipt = read_json(root / "evaluation-receipt.json")
        if not isinstance(arch, dict) or (arch.get("input_size"), arch.get("fallen_output")) != (
            56,
            False,
        ):
            raise ModelLoadError("unsupported pose-bbox56 architecture")
        if not isinstance(calibration, dict):
            raise ModelLoadError("invalid calibration.json")
        temperature = calibration.get("temperature")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ModelLoadError("calibration temperature must be numeric")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ModelLoadError("calibration temperature must be finite and positive")
        try:
            module = _ProxyGru(
                _positive_int(arch.get("hidden_size"), "hidden_size"),
                _positive_int(arch.get("num_layers"), "num_layers"),
                _dropout(arch.get("dropout")),
            )
            state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
            module.load_state_dict(state, strict=True)
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            raise ModelLoadError(f"cannot load pose-bbox56 model: {exc}") from exc
        if not isinstance(receipt, dict):
            raise ModelLoadError("invalid evaluation-receipt.json")
        candidate_threshold = receipt.get("threshold", calibration.get("threshold"))
        receipt_threshold = (
            float(candidate_threshold)
            if isinstance(candidate_threshold, (int, float))
            and not isinstance(candidate_threshold, bool)
            and 0.0 <= float(candidate_threshold) <= 1.0
            else None
        )
        promotion_eligible = receipt.get(
            "promotion_eligible", calibration.get("promotion_eligible")
        )
        if not isinstance(promotion_eligible, bool):
            raise ModelLoadError("receipt promotion_eligible must be boolean")
        runner = cls(
            module,
            temperature,
            receipt_threshold,
            promotion_eligible,
            member_digest(manifest, "model.pt"),
            POSE_BBOX56_PREPROCESSING_IDENTITY,
        )
        runner.warmup()
        return runner

    def predict(self, features: FallModelInput) -> FallV2Probabilities:
        values = np.asarray(features, dtype=np.float32)
        if values.shape != _SHAPE or not np.isfinite(values).all():
            raise ModelLoadError("pose-bbox56 input must be finite shape (30, 56)")
        with torch.no_grad():
            logits = self._module(torch.from_numpy(values).unsqueeze(0))
            fall_transition = torch.sigmoid(logits / self._temperature).cpu().numpy()[0, 0]
        return FallV2Probabilities(
            background=float(1.0 - fall_transition),
            fall_transition=float(fall_transition),
            fallen=0.0,
        )

    def warmup(self) -> None:
        self.predict(np.zeros(_SHAPE, dtype=np.float32))


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelLoadError(f"invalid {name}")
    return value


def _dropout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ModelLoadError("invalid dropout")
    return float(value)


__all__ = ["PoseBbox56BundleRunner"]

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, TypeAlias, final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from typing_extensions import override

from worker.adapters.model.artifact import verify_artifact_digest
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.gru_manifest import GRU_INPUT_DIM, GruFallManifest

StateDict: TypeAlias = Mapping[str, torch.Tensor]
_TORCH_LOAD: Final[Callable[..., StateDict]] = torch.load


@final
class _GruNet(nn.Module):
    def __init__(self, *, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=GRU_INPUT_DIM,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 3)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs)
        return self.fc(hidden[-1])


@final
class GruFallRunner:
    """Closed three-class GRU runner; construction verifies every bundle member."""

    name: Final = "fall-detector"
    version: Final = "gru"

    def __init__(self, manifest: GruFallManifest, module: nn.Module, device: str = "cpu") -> None:
        self.manifest = manifest
        self.class_order = manifest.class_order
        try:
            self.device = torch.device(device)
            self.model: nn.Module = module.to(self.device).eval()
        except (RuntimeError, TypeError) as exc:
            raise ModelLoadError(
                f"cannot place GRU fall model on device {device!r}: {exc}"
            ) from exc

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path, device: str = "cpu") -> GruFallRunner:
        manifest = GruFallManifest.from_artifact_dir(artifact_dir)
        _verify_bundle(manifest)
        module = build_gru_module_from_arch(manifest.architecture_path)
        state_dict = _load_state_dict(manifest.weights_path)
        try:
            module.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ModelLoadError(f"cannot load GRU model state_dict: {exc}") from exc
        runner = cls(manifest, module, device=device)
        runner.warmup()
        return runner

    def predict(
        self, features: Sequence[Sequence[float]] | NDArray[np.floating]
    ) -> tuple[float, float, float]:
        sequence = np.asarray(features, dtype=np.float32)
        if sequence.shape != self.manifest.input_shape:
            raise ModelLoadError(
                "unexpected input shape: "
                f"expected {self.manifest.input_shape}, received {sequence.shape}"
            )
        if not np.isfinite(sequence).all():
            raise ModelLoadError("GRU input contains non-finite values")
        with torch.no_grad():
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            if logits.shape != (1, 3):
                raise ModelLoadError(
                    f"GRU model must return shape (1, 3), received {tuple(logits.shape)}"
                )
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
            raise ModelLoadError("GRU model returned non-finite probabilities")
        output = tuple(float(value) for value in probabilities)
        if not math.isfinite(sum(output)):
            raise ModelLoadError("GRU model returned non-finite probabilities")
        return output  # type: ignore[return-value]

    def warmup(self) -> None:
        self.predict(np.zeros(self.manifest.input_shape, dtype=np.float32))


def _verify_bundle(manifest: GruFallManifest) -> None:
    """Verify all identities before deserializing any torch data."""
    for path, digest in (
        (manifest.weights_path, manifest.artifact_digest),
        (manifest.architecture_path, manifest.architecture_digest),
        (manifest.input_contract_path, manifest.input_digest),
        (manifest.policy_path, manifest.policy_digest),
        (manifest.calibration_path, manifest.calibration_digest),
        (manifest.conformance_path, manifest.conformance_digest),
    ):
        verify_artifact_digest(path, digest)


def build_gru_module(*, hidden: int, layers: int, dropout: float) -> _GruNet:
    if hidden <= 0 or layers <= 0 or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ModelLoadError("invalid GRU architecture")
    return _GruNet(hidden=hidden, layers=layers, dropout=dropout)


def build_gru_module_from_arch(path: Path) -> _GruNet:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelLoadError(f"cannot read arch.json at {path}") from exc
    if not isinstance(raw, dict) or set(raw) != {"hidden", "layers", "dropout"}:
        raise ModelLoadError("arch.json must contain exactly hidden, layers, and dropout")
    hidden, layers, dropout = raw["hidden"], raw["layers"], raw["dropout"]
    if (
        isinstance(hidden, bool)
        or not isinstance(hidden, int)
        or isinstance(layers, bool)
        or not isinstance(layers, int)
        or isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
    ):
        raise ModelLoadError("invalid GRU architecture")
    return build_gru_module(hidden=hidden, layers=layers, dropout=float(dropout))


def _load_state_dict(path: Path) -> StateDict:
    try:
        try:
            return _TORCH_LOAD(path, map_location="cpu", weights_only=True)
        except TypeError:
            return _TORCH_LOAD(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ModelLoadError(f"cannot load model.pt at {path}") from exc


__all__ = [
    "GruFallManifest",
    "GruFallRunner",
    "ModelLoadError",
    "build_gru_module",
    "build_gru_module_from_arch",
]

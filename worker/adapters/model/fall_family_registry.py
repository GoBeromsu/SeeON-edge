"""Fall-model family registry: dispatch ``FallModelConfig.type`` to a factory.

Issue #65: a brand-new fall-model AI family (a different architecture, not
merely a retrained weights version-up of an existing one -- GRU, transformer,
a rule-based heuristic, or any future family) must be pluggable purely through
config/artifact metadata. Before this module, ``WorkerRuntime._create_fall_model``
was code-pinned to ``LstmFallRunner`` and ``FallModelConfig.type`` was a
single-value ``Literal["lstm"]``, so onboarding a new family meant editing
both.

This registry is the one-time dispatch mechanism: every family after "lstm"
is additive -- implement ``FallV2ModelProtocol``
(``worker/domains/fall/classifier.py``), write a factory function taking a
``FallModelConfigLike`` and a device string, and ``register()`` it under the
family's ``type`` name. ``_create_fall_model`` never grows a new branch.

Fail-closed per ADR-0002 (same principle as #43's None-config refusal, which
this registry does not touch): an unregistered ``type`` value raises
``UnknownFallModelTypeError`` with the offending value and every known family
name, rather than silently falling back to "lstm" or any other default.

``worker/adapters`` may not import ``worker/runtime`` (runtime is the sole
composition root; see the import-linter contracts in ``pyproject.toml``), so
this module cannot import the real, validated
``worker.runtime.config.worker_models.FallModelConfig``. It instead declares
``FallModelConfigLike``, a structural mirror of that pydantic model's public
fields -- the same "adapters define their own Protocol instead of importing
the higher layer's type" convention already used by
``worker/adapters/model/registry.py``'s ``FallModel``/``FallModelMetadata``.
``FallModelConfig`` satisfies it structurally with no inheritance or import
needed; ``worker.runtime.worker`` (the composition root) passes a real
instance straight through.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeAlias

from typing_extensions import override

from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.adapters.model.registry import FallModel
from worker.adapters.model.torch_gru_fall import GruFallRunner


class FallModelConfigLike(Protocol):
    """Structural mirror of ``worker.runtime.config.worker_models.FallModelConfig``.

    Every field a fall-model factory may need to read, named identically to
    the real pydantic model. See the module docstring for why this Protocol
    exists instead of an import.
    """

    @property
    def type(self) -> str: ...

    @property
    def framework(self) -> str: ...

    @property
    def mode(self) -> str: ...

    @property
    def artifact_dir(self) -> Path: ...

    @property
    def weights(self) -> str: ...

    @property
    def architecture(self) -> str: ...

    @property
    def metadata(self) -> str: ...

    @property
    def window(self) -> int: ...

    @property
    def stride(self) -> int: ...

    @property
    def input_shape(self) -> tuple[int, int]: ...

    @property
    def operating_threshold(self) -> float: ...

    @property
    def schema_version(self) -> int | None: ...

    @property
    def preprocessing_identity(self) -> str | None: ...


FallModelFactory: TypeAlias = Callable[[FallModelConfigLike, str], FallModel]
BundleFallModelFactory: TypeAlias = Callable[[Path, str], FallModel]


@dataclass(slots=True)
class UnknownFallModelTypeError(Exception):
    """Raised when ``FallModelConfig.type`` names an unregistered family."""

    requested_type: str
    known_types: tuple[str, ...]

    @override
    def __str__(self) -> str:
        known = ", ".join(self.known_types) if self.known_types else "(none registered)"
        return (
            f"unknown fall model type {self.requested_type!r}; known fall model families: {known}"
        )


class FallModelFamilyRegistry:
    """Map fall-model family names (``FallModelConfig.type``) to factories."""

    def __init__(self) -> None:
        self._factories: dict[str, FallModelFactory] = {}
        self._bundle_factories: dict[str, BundleFallModelFactory] = {}

    def register(self, model_type: str, factory: FallModelFactory) -> None:
        if not model_type:
            raise ValueError("model_type must be non-empty")
        self._factories[model_type] = factory

    def register_runtime_format(self, runtime_format: str, factory: BundleFallModelFactory) -> None:
        if not runtime_format:
            raise ValueError("runtime_format must be non-empty")
        self._bundle_factories[runtime_format] = factory

    def create_bundle(self, runtime_format: str, artifact_dir: Path, device: str) -> FallModel:
        factory = self._bundle_factories.get(runtime_format)
        if factory is None:
            raise UnknownFallModelTypeError(runtime_format, self.runtime_formats())
        return factory(artifact_dir, device)

    def runtime_formats(self) -> tuple[str, ...]:
        return tuple(sorted(self._bundle_factories))

    def create(self, model_type: str, config: FallModelConfigLike, device: str) -> FallModel:
        return self.get_factory(model_type)(config, device)

    def get_factory(self, model_type: str) -> FallModelFactory:
        factory = self._factories.get(model_type)
        if factory is None:
            raise UnknownFallModelTypeError(model_type, self.types())
        return factory

    def types(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _create_gru_bundle_model(artifact_dir: Path, device: str) -> FallModel:
    return GruFallRunner.from_artifact_dir(artifact_dir, device=device)

def _create_pose_bbox56_bundle_model(artifact_dir: Path, device: str) -> FallModel:
    return PoseBbox56BundleRunner.from_artifact_dir(artifact_dir, device=device)  # type: ignore[return-value]

def _create_pose_bbox56_model(config: FallModelConfigLike, device: str) -> FallModel:
    return _create_pose_bbox56_bundle_model(config.artifact_dir, device)


def default_fall_model_family_registry() -> FallModelFamilyRegistry:
    """Build the packaged-model and admitted-bundle dispatch registry."""
    registry = FallModelFamilyRegistry()
    registry.register("pose-bbox56-proxy-v0", _create_pose_bbox56_model)
    registry.register_runtime_format("torchscript-gru-pose-bbox", _create_gru_bundle_model)
    registry.register_runtime_format("pose-bbox56-proxy-v0", _create_pose_bbox56_bundle_model)
    return registry


DEFAULT_FALL_MODEL_FAMILY_REGISTRY: Final = default_fall_model_family_registry()

__all__ = [
    "DEFAULT_FALL_MODEL_FAMILY_REGISTRY",
    "BundleFallModelFactory",
    "FallModelConfigLike",
    "FallModelFactory",
    "FallModelFamilyRegistry",
    "UnknownFallModelTypeError",
    "default_fall_model_family_registry",
]

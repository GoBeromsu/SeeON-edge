"""Fall-model family registry: dispatch ``FallModelConfig.type`` to a factory.

Issue #65: a brand-new fall-model AI family (a different architecture, not
merely a retrained weights version-up of an existing one -- GRU, transformer,
a rule-based heuristic, or any future family) must be pluggable purely through
config/artifact metadata. Before this module, ``WorkerRuntime._create_fall_model``
was code-pinned to ``LstmFallRunner`` and ``FallModelConfig.type`` was a
single-value ``Literal["lstm"]``, so onboarding a new family meant editing
both.

This registry is the one-time dispatch mechanism: every family after "lstm"
is additive -- implement ``FallModelProtocol``
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

from worker.adapters.model.registry import FallModel
from worker.adapters.model.torch_lstm_fall import LstmFallRunner


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


@dataclass(slots=True)
class DisabledFallModelTypeError(Exception):
    """Raised when a known-but-dark model family is requested for activation."""

    requested_type: str

    @override
    def __str__(self) -> str:
        return f"fall model type {self.requested_type!r} is registered but disabled"


class FallModelFamilyRegistry:
    """Map fall-model family names (``FallModelConfig.type``) to factories."""

    def __init__(self) -> None:
        self._factories: dict[str, FallModelFactory] = {}
        self._disabled_types: set[str] = set()

    def register(self, model_type: str, factory: FallModelFactory) -> None:
        if not model_type:
            raise ValueError("model_type must be non-empty")
        if model_type in self._disabled_types:
            raise ValueError(f"model_type {model_type!r} is disabled")
        self._factories[model_type] = factory

    def register_disabled(self, model_type: str) -> None:
        if not model_type:
            raise ValueError("model_type must be non-empty")
        if model_type in self._factories:
            raise ValueError(f"model_type {model_type!r} is active")
        self._disabled_types.add(model_type)

    def create(self, model_type: str, config: FallModelConfigLike, device: str) -> FallModel:
        return self.get_factory(model_type)(config, device)

    def get_factory(self, model_type: str) -> FallModelFactory:
        if model_type in self._disabled_types:
            raise DisabledFallModelTypeError(model_type)
        factory = self._factories.get(model_type)
        if factory is None:
            raise UnknownFallModelTypeError(model_type, self.types())
        return factory

    def types(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _create_lstm_fall_model(config: FallModelConfigLike, device: str) -> FallModel:
    return LstmFallRunner.from_artifact_dir(
        config.artifact_dir,
        device=device,
        expected_schema_version=config.schema_version,
        expected_preprocessing_identity=config.preprocessing_identity,
        operating_threshold=config.operating_threshold,
    )


def default_fall_model_family_registry() -> FallModelFamilyRegistry:
    """Build the default registry: today, only the "lstm" family."""
    registry = FallModelFamilyRegistry()
    registry.register("lstm", _create_lstm_fall_model)
    registry.register_disabled("gru")
    return registry


DEFAULT_FALL_MODEL_FAMILY_REGISTRY: Final = default_fall_model_family_registry()

__all__ = [
    "DEFAULT_FALL_MODEL_FAMILY_REGISTRY",
    "DisabledFallModelTypeError",
    "FallModelConfigLike",
    "FallModelFactory",
    "FallModelFamilyRegistry",
    "UnknownFallModelTypeError",
    "default_fall_model_family_registry",
]

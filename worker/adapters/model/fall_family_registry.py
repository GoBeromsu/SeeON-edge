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
(``worker/domains/fall/classifier.py``), write a factory function taking the
validated ``FallModelConfig`` and a device string, and ``register()`` it under
the family's ``type`` name. ``_create_fall_model`` never grows a new branch.

Fail-closed per ADR-0002 (same principle as #43's None-config refusal, which
this registry does not touch): an unregistered ``type`` value raises
``UnknownFallModelTypeError`` with the offending value and every known family
name, rather than silently falling back to "lstm" or any other default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeAlias

from typing_extensions import override

from worker.adapters.model.registry import FallModel
from worker.adapters.model.torch_lstm_fall import LstmFallRunner
from worker.runtime.config.worker_models import FallModelConfig

FallModelFactory: TypeAlias = Callable[[FallModelConfig, str], FallModel]


@dataclass(slots=True)
class UnknownFallModelTypeError(Exception):
    """Raised when ``FallModelConfig.type`` names an unregistered family."""

    requested_type: str
    known_types: tuple[str, ...]

    @override
    def __str__(self) -> str:
        known = ", ".join(self.known_types) if self.known_types else "(none registered)"
        return (
            f"unknown fall model type {self.requested_type!r}; "
            f"known fall model families: {known}"
        )


class FallModelFamilyRegistry:
    """Map fall-model family names (``FallModelConfig.type``) to factories."""

    def __init__(self) -> None:
        self._factories: dict[str, FallModelFactory] = {}

    def register(self, model_type: str, factory: FallModelFactory) -> None:
        if not model_type:
            raise ValueError("model_type must be non-empty")
        self._factories[model_type] = factory

    def create(self, model_type: str, config: FallModelConfig, device: str) -> FallModel:
        return self.get_factory(model_type)(config, device)

    def get_factory(self, model_type: str) -> FallModelFactory:
        factory = self._factories.get(model_type)
        if factory is None:
            raise UnknownFallModelTypeError(model_type, self.types())
        return factory

    def types(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _create_lstm_fall_model(config: FallModelConfig, device: str) -> FallModel:
    return LstmFallRunner.from_artifact_dir(
        config.artifact_dir,
        device=device,
        expected_schema_version=config.schema_version,
        expected_preprocessing_identity=config.preprocessing_identity,
    )


def default_fall_model_family_registry() -> FallModelFamilyRegistry:
    """Build the default registry: today, only the "lstm" family."""
    registry = FallModelFamilyRegistry()
    registry.register("lstm", _create_lstm_fall_model)
    return registry


DEFAULT_FALL_MODEL_FAMILY_REGISTRY: Final = default_fall_model_family_registry()

__all__ = [
    "DEFAULT_FALL_MODEL_FAMILY_REGISTRY",
    "FallModelFactory",
    "FallModelFamilyRegistry",
    "UnknownFallModelTypeError",
    "default_fall_model_family_registry",
]

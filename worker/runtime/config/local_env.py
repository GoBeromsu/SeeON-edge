"""Environment-sourced local overrides for ``models.fall`` and ``clip.enabled``.

Issues #66/#68: ``BackendWorkerConfigPayload.to_worker_config()``
(``worker/runtime/config/pull_models.py``) never carried ``models``/``clip``
through a relay pull, so the shipped pull-first production topology
(``compose.edge.yaml``'s default empty ``EDGE_CAMERA_CONFIG``) could never
configure a fall model or turn on clip recording -- both YAML-only fields
always resolved to their pydantic defaults (``fall=None``,
``enabled=False``).

Docker packaging configures the worker via ``.env``/environment variables in
production, so both fields get a real, production-reachable surface here,
read the same way ``RELAY_URL``/``RELAY_TOKEN`` already are
(``worker/runtime/config/config_pull.py``). The fall artifact is edge-local
(mounted into the container), so its env vars live alongside the other
edge-local switches (``ML_WORKER_DEV_MJPEG*``, ``ML_WORKER_PROFILE``) rather
than being pulled from the backend, which only carries fleet-level config
(relay/domains/cameras).

Fail-closed per ADR-0002: a malformed value (a non-integer window/stride, a
non-numeric operating_threshold, or an unrecognized boolean token) raises
``WorkerConfigError`` loudly rather than silently defaulting.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import (
    ClipRecordingConfig,
    FallModelConfig,
    WorkerConfig,
    WorkerModelsConfig,
)

ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV: Final = "ML_WORKER_FALL_MODEL_ARTIFACT_DIR"
ML_WORKER_FALL_MODEL_TYPE_ENV: Final = "ML_WORKER_FALL_MODEL_TYPE"
ML_WORKER_FALL_MODEL_WEIGHTS_ENV: Final = "ML_WORKER_FALL_MODEL_WEIGHTS"
ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV: Final = "ML_WORKER_FALL_MODEL_ARCHITECTURE"
ML_WORKER_FALL_MODEL_WINDOW_ENV: Final = "ML_WORKER_FALL_MODEL_WINDOW"
ML_WORKER_FALL_MODEL_STRIDE_ENV: Final = "ML_WORKER_FALL_MODEL_STRIDE"
ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: Final = "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD"
ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV: Final = "ML_WORKER_FALL_MODEL_SCHEMA_VERSION"
ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV: Final = (
    "ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY"
)
ML_WORKER_CLIP_RECORDING_ENABLED_ENV: Final = "ML_WORKER_CLIP_RECORDING_ENABLED"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off"})
_DEFAULT_TYPE: Final = "lstm"
_DEFAULT_WEIGHTS: Final = "model.pt"
_DEFAULT_ARCHITECTURE: Final = "arch.json"


def _bool_env(name: str, env: Mapping[str, str]) -> bool:
    raw = env.get(name, "").strip().lower()
    if raw == "":
        return False
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    raise WorkerConfigError(
        f"{name} must be a boolean ({sorted(_TRUTHY | _FALSY)}), got {raw!r}"
    )


def _required_str(name: str, env: Mapping[str, str], *, because: str) -> str:
    raw = env.get(name, "").strip()
    if not raw:
        raise WorkerConfigError(f"{name} is required {because}")
    return raw


def _required_int(name: str, env: Mapping[str, str], *, because: str) -> int:
    raw = _required_str(name, env, because=because)
    try:
        return int(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be an integer, got {raw!r}") from error


def _required_float(name: str, env: Mapping[str, str], *, because: str) -> float:
    raw = _required_str(name, env, because=because)
    try:
        return float(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be a number, got {raw!r}") from error


def _optional_int(name: str, env: Mapping[str, str]) -> int | None:
    raw = env.get(name, "").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be an integer, got {raw!r}") from error


def clip_recording_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ClipRecordingConfig:
    env = os.environ if environ is None else environ
    return ClipRecordingConfig(enabled=_bool_env(ML_WORKER_CLIP_RECORDING_ENABLED_ENV, env))


def fall_model_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> FallModelConfig | None:
    """Build ``models.fall`` from env, or ``None`` if unconfigured.

    ``ML_WORKER_FALL_MODEL_ARTIFACT_DIR`` is the on/off switch: unset (the
    out-of-the-box default) means no fall model is configured via env, same
    as an omitted ``models.fall`` in YAML -- the #43 boot gate
    (``WorkerRuntime._create_fall_model``) then refuses to boot, by design.
    Once it is set, the rest of the artifact contract
    (window/stride/operating_threshold) becomes required so a partially
    configured fall model fails loudly at boot rather than silently
    defaulting (ADR-0002). ``framework``/``mode`` are not independent env
    vars: today's ``FallModelConfig`` only has one valid literal for each
    (pytorch/sequence), so there is nothing for an env var to select yet.
    ``type`` (the model family/architecture -- #65) does have an env var,
    ``ML_WORKER_FALL_MODEL_TYPE``, defaulting to "lstm" so existing
    deployments are unaffected; the registry
    (``worker.adapters.model.fall_family_registry``) fails closed at boot on
    an unrecognized value, not here, so a typo'd family name is still
    validated against every registered family rather than silently accepted
    as an opaque string. ``input_shape`` is derived from ``window`` rather
    than given its own var, since ``FallModelConfig`` already requires
    ``input_shape == (window, 51)``.
    """
    env = os.environ if environ is None else environ
    artifact_dir = env.get(ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV, "").strip()
    if not artifact_dir:
        return None
    because = f"when {ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV} is set"
    window = _required_int(ML_WORKER_FALL_MODEL_WINDOW_ENV, env, because=because)
    stride = _required_int(ML_WORKER_FALL_MODEL_STRIDE_ENV, env, because=because)
    operating_threshold = _required_float(
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV, env, because=because
    )
    schema_version = _optional_int(ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV, env)
    model_type = env.get(ML_WORKER_FALL_MODEL_TYPE_ENV, "").strip() or _DEFAULT_TYPE
    weights = env.get(ML_WORKER_FALL_MODEL_WEIGHTS_ENV, "").strip() or _DEFAULT_WEIGHTS
    architecture = (
        env.get(ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV, "").strip() or _DEFAULT_ARCHITECTURE
    )
    preprocessing_identity = (
        env.get(ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV, "").strip() or None
    )
    try:
        return FallModelConfig(
            type=model_type,
            framework="pytorch",
            mode="sequence",
            artifact_dir=Path(artifact_dir),
            weights=weights,
            architecture=architecture,
            window=window,
            stride=stride,
            input_shape=(window, 51),
            operating_threshold=operating_threshold,
            schema_version=schema_version,
            preprocessing_identity=preprocessing_identity,
        )
    except ValidationError as error:
        raise WorkerConfigError(
            f"invalid fall model environment configuration: {error}"
        ) from error


def worker_models_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> WorkerModelsConfig:
    return WorkerModelsConfig(fall=fall_model_config_from_environment(environ))


def resolve_local_overrides(
    yaml_config: WorkerConfig | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[WorkerModelsConfig, ClipRecordingConfig]:
    """Settle ``models``/``clip`` to merge into a pulled or LKG-restored
    ``WorkerConfig``.

    Mirrors the "explicit wins outright, silence defers" precedence
    ``WorkerRuntime._resolve_mjpeg_config`` already uses for
    ``dev_mjpeg``/``ML_WORKER_DEV_MJPEG*`` (worker/runtime/worker.py
    ~535-554): an explicit local YAML value wins outright over env; with the
    YAML silent (no local YAML at all -- the production pull-first default
    -- or the YAML field left at its own default), the environment decides.
    """
    env = os.environ if environ is None else environ
    models = (
        yaml_config.models
        if yaml_config is not None and yaml_config.models.fall is not None
        else worker_models_config_from_environment(env)
    )
    clip = (
        yaml_config.clip
        if yaml_config is not None and yaml_config.clip.enabled
        else clip_recording_config_from_environment(env)
    )
    return models, clip


__all__ = [
    "ML_WORKER_CLIP_RECORDING_ENABLED_ENV",
    "ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV",
    "ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV",
    "ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV",
    "ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV",
    "ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV",
    "ML_WORKER_FALL_MODEL_STRIDE_ENV",
    "ML_WORKER_FALL_MODEL_TYPE_ENV",
    "ML_WORKER_FALL_MODEL_WEIGHTS_ENV",
    "ML_WORKER_FALL_MODEL_WINDOW_ENV",
    "clip_recording_config_from_environment",
    "fall_model_config_from_environment",
    "resolve_local_overrides",
    "worker_models_config_from_environment",
]

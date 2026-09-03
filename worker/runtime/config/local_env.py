"""Environment-sourced local overrides for ``models.fall`` and ``clip.enabled``.

Issues #66/#68: ``BackendWorkerConfigPayload.to_worker_config()``
(``worker/runtime/config/pull_models.py``) never carried ``models``/``clip``
through a relay pull, so the shipped pull-only production topology (no
local YAML roster at all) could never
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

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from contracts.model_selection import POSE_BBOX56_PREPROCESSING_IDENTITY
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import (
    ClipRecordingConfig,
    DevMjpegConfig,
    FallModelConfig,
    SelectedFallBundleConfig,
    WorkerConfig,
    WorkerModelsConfig,
)
from worker.runtime.provenance.model_bundle import (
    ModelBundleAdmissionError,
    desired_model_bundle_from_selection_document,
)

LOGGER: Final = logging.getLogger(__name__)

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
WORKER_REPLAY_TRACE_DIR_ENV: Final = "WORKER_REPLAY_TRACE_DIR"
FALL_SELECTION_PATH: Final = Path("/app/model-selection.json")
FALL_MODELS_ROOT: Final = Path("/models")

_RETIRED_WORKER_ENV: Final = frozenset(
    {
        ML_WORKER_CLIP_RECORDING_ENABLED_ENV,
        ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV,
        ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV,
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV,
        ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV,
        ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV,
        ML_WORKER_FALL_MODEL_STRIDE_ENV,
        ML_WORKER_FALL_MODEL_TYPE_ENV,
        ML_WORKER_FALL_MODEL_WEIGHTS_ENV,
        ML_WORKER_FALL_MODEL_WINDOW_ENV,
        "CLIP_STORE_DIR",
        "EDGE_CAMERA_CONFIG",
        "EDGE_CAMERA_CONFIG_FILE",
        "ML_WORKER_DEV_MJPEG",
        "ML_WORKER_DEV_MJPEG_HOST",
        "ML_WORKER_DEV_MJPEG_PORT",
        "ML_WORKER_EVENT_CLIP_EXPORT_ENABLED",
        "RELAY_URL",
    }
)


def reject_retired_worker_environment(environ: Mapping[str, str]) -> None:
    """Reject removed worker authorities instead of silently overriding config."""
    present = sorted(_RETIRED_WORKER_ENV.intersection(environ))
    if present:
        raise WorkerConfigError(
            "retired edge environment key(s): "
            + ", ".join(present)
            + "; use the versioned worker config authority"
        )


_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off"})
_DEFAULT_TYPE: Final = "pose-bbox56-proxy-v0"
_DEFAULT_WEIGHTS: Final = "model.pt"
_DEFAULT_ARCHITECTURE: Final = "arch.json"
# Packaged default fall model, used when ML_WORKER_FALL_MODEL_ARTIFACT_DIR is
# unset: the published pose+bbox56 proxy bundle pinned in
# worker/tools/fetch_models/manifest.json and provisioned by
# scripts/fetch-models.sh (nothing under models/ is tracked). Values mirror
# worker/ml-worker.example.yaml's models.fall block; the 0.5 transition
# threshold is the owner-fixed default that a promotion-eligible receipt may
# override (worker/domains/registry.py).
_DEFAULT_ARTIFACT_DIR: Final = "models/fall/pose-bbox56-gru"
_DEFAULT_WINDOW: Final = 30
_DEFAULT_STRIDE: Final = 5
_DEFAULT_OPERATING_THRESHOLD: Final = 0.5
_DEFAULT_SCHEMA_VERSION: Final = 2
_DEFAULT_PREPROCESSING_IDENTITY: Final = POSE_BBOX56_PREPROCESSING_IDENTITY
_FETCH_MODELS_HINT: Final = (
    "run scripts/fetch-models.sh to download the packaged pose+bbox56 model "
    "weights (or set ML_WORKER_FALL_MODEL_ARTIFACT_DIR to point at an "
    "already-provisioned artifact directory)"
)


def _bool_env(name: str, env: Mapping[str, str]) -> bool | None:
    """Parse an optional boolean env var.

    Returns ``None`` when unset/blank so callers can distinguish "not set"
    from an explicit ``false`` -- "explicit wins outright, silence defers"
    (mirrored from ``WorkerRuntime._resolve_mjpeg_config``'s ``dev_mjpeg``
    precedent) only works if silence is representable here, not collapsed to
    a hardcoded ``False``.
    """
    raw = env.get(name, "").strip().lower()
    if raw == "":
        return None
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    raise WorkerConfigError(f"{name} must be a boolean ({sorted(_TRUTHY | _FALSY)}), got {raw!r}")


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


def _collect_required_int(
    name: str, env: Mapping[str, str], *, because: str, errors: list[str]
) -> int | None:
    """Like ``_required_int``, but appends to ``errors`` instead of raising.

    Issue #79 (track 2): callers collect every malformed fall-model env var
    into one error report instead of the first ``_required_int`` call
    aborting before the next field is even checked.
    """
    try:
        return _required_int(name, env, because=because)
    except WorkerConfigError as error:
        errors.append(str(error))
        return None


def _collect_required_float(
    name: str, env: Mapping[str, str], *, because: str, errors: list[str]
) -> float | None:
    try:
        return _required_float(name, env, because=because)
    except WorkerConfigError as error:
        errors.append(str(error))
        return None


def _optional_int(name: str, env: Mapping[str, str]) -> int | None:
    raw = env.get(name, "").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be an integer, got {raw!r}") from error


def _optional_float(name: str, env: Mapping[str, str]) -> float | None:
    raw = env.get(name, "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be a number, got {raw!r}") from error


def _warn_if_env_ignored(name: str, env: Mapping[str, str], *, reason: str) -> None:
    """Warn when ``name`` is set in the environment but the current code path
    does not read it.

    Issue #198 (and #191 before it): a "set, documented, silently dead" env
    var is an operator-facing footgun regardless of which var it is, so this
    stays a small generic check rather than one-off handling per variable.
    """
    if env.get(name, "").strip():
        LOGGER.warning("%s is set but ignored: %s", name, reason)


def clip_recording_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ClipRecordingConfig:
    """Build ``clip`` from env, deferring to ``ClipRecordingConfig``'s own
    default when ``ML_WORKER_CLIP_RECORDING_ENABLED`` is unset.

    An explicit env value (true or false) always wins outright. Env silence
    must *not* be read as an explicit "false" -- it defers to whatever
    ``ClipRecordingConfig.enabled`` itself defaults to, so a future change to
    that default (e.g. always-on clip recording) takes effect on an
    unconfigured boot instead of being silently overridden here.
    """
    env = os.environ if environ is None else environ
    explicit = _bool_env(ML_WORKER_CLIP_RECORDING_ENABLED_ENV, env)
    return ClipRecordingConfig() if explicit is None else ClipRecordingConfig(enabled=explicit)


def fall_model_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> FallModelConfig:
    """Build ``models.fall`` from env, defaulting to the packaged pose+bbox56 bundle.

    Issue #133: the worker must boot with zero env vars.
    ``ML_WORKER_FALL_MODEL_ARTIFACT_DIR`` is the on/off switch for *explicit*
    artifact configuration: unset (the out-of-the-box default) no longer
    means "no fall model" -- it now resolves to the packaged default LSTM
    model at ``models/fall/lstm`` (arch.json/metadata.yaml are tracked in
    git; model.pt is fetched separately via ``scripts/fetch-models.sh``
    since weights stay gitignored). An explicit ``ARTIFACT_DIR`` env value
    always overrides the default outright, never blends with it.

    Issue #198: ``ARTIFACT_DIR`` being unset does *not* also gate
    ``window``/``stride``/``operating_threshold`` -- those three are read
    from their own env vars regardless of which artifact is in play, each
    falling back independently to the packaged manifest's value only when
    its own env var is absent. Reading them only inside the "artifact dir is
    explicitly set" branch (the pre-#198 behavior) meant an operator-set
    ``ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD`` was silently discarded on
    the packaged-default path -- the only path the shipped edge topology
    actually takes.

    Once ``ML_WORKER_FALL_MODEL_ARTIFACT_DIR`` is explicitly set, the
    artifact contract tightens: window/stride/operating_threshold become
    *required* (not just respected-if-present) so a partially configured
    fall model still fails loudly at boot rather than silently defaulting
    (ADR-0002) -- issue #79 (track 2): every malformed field is collected
    and reported together, not just the first.
    ``framework``/``mode`` are not independent env vars: today's
    ``FallModelConfig`` only has one valid literal for each
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
    artifact_dir_raw = env.get(ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV, "").strip()
    is_default = not artifact_dir_raw

    if is_default:
        # Issue #198: previously window/stride/operating_threshold were only
        # ever read when ARTIFACT_DIR was also set, so an operator-set
        # ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD was silently discarded on
        # the packaged-default path (the only path the shipped edge topology
        # actually takes) and the field default (0.0007872396381571889, an
        # upstream le2i operating point -- see the module docstring above)
        # was used instead. An explicit env value now wins outright per
        # field; env silence falls back to the packaged manifest default,
        # same "explicit wins outright, silence defers" precedence used
        # elsewhere in this module (see ``clip_recording_config_from_environment``).
        artifact_dir = _DEFAULT_ARTIFACT_DIR
        window_env = _optional_int(ML_WORKER_FALL_MODEL_WINDOW_ENV, env)
        stride_env = _optional_int(ML_WORKER_FALL_MODEL_STRIDE_ENV, env)
        operating_threshold_env = _optional_float(ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV, env)
        window = _DEFAULT_WINDOW if window_env is None else window_env
        stride = _DEFAULT_STRIDE if stride_env is None else stride_env
        operating_threshold = (
            _DEFAULT_OPERATING_THRESHOLD
            if operating_threshold_env is None
            else operating_threshold_env
        )
        operating_threshold_source = (
            "packaged manifest default" if operating_threshold_env is None else "env"
        )
        schema_version: int | None = _DEFAULT_SCHEMA_VERSION
        preprocessing_identity: str | None = _DEFAULT_PREPROCESSING_IDENTITY
        # schema_version/preprocessing_identity have no packaged-default
        # fallback path (unlike window/stride/operating_threshold above) --
        # they always resolve to the packaged manifest's own values here, so
        # an env value for either is unconditionally dead on this branch.
        _warn_if_env_ignored(
            ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV,
            env,
            reason=(
                f"only read when {ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV} is also set; "
                "the packaged default model's own manifest value is used instead"
            ),
        )
        _warn_if_env_ignored(
            ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV,
            env,
            reason=(
                f"only read when {ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV} is also set; "
                "the packaged default model's own manifest value is used instead"
            ),
        )
    else:
        artifact_dir = artifact_dir_raw
        because = f"when {ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV} is set"
        errors: list[str] = []
        window = _collect_required_int(
            ML_WORKER_FALL_MODEL_WINDOW_ENV, env, because=because, errors=errors
        )
        stride = _collect_required_int(
            ML_WORKER_FALL_MODEL_STRIDE_ENV, env, because=because, errors=errors
        )
        operating_threshold = _collect_required_float(
            ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV, env, because=because, errors=errors
        )
        if errors:
            raise WorkerConfigError(
                f"{len(errors)} fall model environment variable(s) invalid: " + "; ".join(errors)
            )
        # Guaranteed non-None: the empty-errors check above already returned
        # (raised) if any of the three collectors above appended a failure.
        assert window is not None
        assert stride is not None
        assert operating_threshold is not None
        # Required (not optional) on this branch, so it is always env-sourced.
        operating_threshold_source = "env"
        schema_version = _optional_int(ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV, env)
        preprocessing_identity = (
            env.get(ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV, "").strip() or None
        )

    model_type = env.get(ML_WORKER_FALL_MODEL_TYPE_ENV, "").strip() or _DEFAULT_TYPE
    weights = env.get(ML_WORKER_FALL_MODEL_WEIGHTS_ENV, "").strip() or _DEFAULT_WEIGHTS
    architecture = (
        env.get(ML_WORKER_FALL_MODEL_ARCHITECTURE_ENV, "").strip() or _DEFAULT_ARCHITECTURE
    )
    # Issue #198: the only prior way to discover the effective operating
    # threshold was dumping an emitted event's `audit` blob out of the SQLite
    # outbox. Logging it once here, at the boot-only call site, makes the
    # resolved value -- and whether it came from env or the packaged
    # manifest default -- visible without waiting for a detection to fire.
    LOGGER.info(
        "fall model operating_threshold resolved to %s (source: %s)",
        operating_threshold,
        operating_threshold_source,
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
            input_shape=(window, 56),
            operating_threshold=operating_threshold,
            schema_version=schema_version,
            preprocessing_identity=preprocessing_identity,
        )
    except ValidationError as error:
        if is_default:
            raise WorkerConfigError(
                "packaged default pose+bbox56 fall model is not fully provisioned at "
                f"{artifact_dir!r} ({error}); {_FETCH_MODELS_HINT}"
            ) from error
        raise WorkerConfigError(f"invalid fall model environment configuration: {error}") from error


def selected_fall_bundle_config_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    selection_path: Path = FALL_SELECTION_PATH,
    models_root: Path = FALL_MODELS_ROOT,
) -> SelectedFallBundleConfig | None:
    """Load the selected fall bundle; absence leaves the packaged model active."""
    if not selection_path.exists():
        return None
    try:
        raw_selection = selection_path.read_bytes()
        selection_document = json.loads(raw_selection)
        canonical_selection = json.dumps(
            selection_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        if raw_selection != canonical_selection:
            raise WorkerConfigError("fall selection must be canonical JSON")
        desired = desired_model_bundle_from_selection_document(selection_document)
        return SelectedFallBundleConfig(
            models_root=models_root,
            desired=desired,
        )
    except (OSError, ValueError, TypeError, ModelBundleAdmissionError) as error:
        raise WorkerConfigError(f"invalid fall selection: {error}") from error


def worker_models_config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> WorkerModelsConfig:
    return WorkerModelsConfig(
        fall=fall_model_config_from_environment(environ),
        selected=selected_fall_bundle_config_from_environment(environ),
    )


def resolve_local_overrides(
    yaml_config: WorkerConfig | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[WorkerModelsConfig, ClipRecordingConfig, DevMjpegConfig | None]:
    """Settle ``models``/``clip``/``dev_mjpeg`` to merge into a pulled or
    LKG-restored ``WorkerConfig``.

    ``models``/``clip`` mirror the "explicit wins outright, silence defers"
    precedence ``WorkerRuntime._resolve_mjpeg_config`` already uses for
    ``dev_mjpeg``/``ML_WORKER_DEV_MJPEG*`` (worker/runtime/worker.py
    ~535-554): an explicit local YAML value wins outright over env; with the
    YAML silent (no local YAML at all -- the production pull-first default
    -- or the YAML field left at its own default), the environment decides.

    ``dev_mjpeg`` only needs the YAML half of that precedence here: unlike
    ``models``/``clip``, ``WorkerRuntime._resolve_mjpeg_config`` already
    falls back to ``ML_WORKER_DEV_MJPEG*`` env vars itself whenever the
    resolved ``WorkerConfig.dev_mjpeg`` comes back disabled, reading
    ``self._env`` directly -- so that half of the precedence already worked
    even before this fix. What was missing (issue #113) was the YAML half:
    ``BackendWorkerConfigPayload.to_worker_config`` never received
    ``dev_mjpeg`` at all, so an explicit local ``dev_mjpeg.enabled: true``
    was silently reset to the pydantic default (disabled) on every
    successful pull -- with no failure and no log line, the operator-facing
    MJPEG diagnostic port would simply never bind. Returning ``None`` here
    when the YAML did not explicitly enable it (rather than synthesizing a
    disabled ``DevMjpegConfig()``) preserves that existing env fallback:
    ``to_worker_config`` only overrides the pulled config's default when the
    caller actually has an explicit answer to give it.
    """
    env = os.environ if environ is None else environ
    environment_models = worker_models_config_from_environment(env)
    models = WorkerModelsConfig(
        fall=(
            yaml_config.models.fall
            if yaml_config is not None and yaml_config.models.fall is not None
            else environment_models.fall
        ),
        box_source=(
            yaml_config.models.box_source
            if yaml_config is not None and yaml_config.models.fall is not None
            else environment_models.box_source
        ),
        selected=environment_models.selected,
    )
    clip = (
        yaml_config.clip
        if yaml_config is not None and yaml_config.clip.enabled
        else clip_recording_config_from_environment(env)
    )
    dev_mjpeg = (
        yaml_config.dev_mjpeg if yaml_config is not None and yaml_config.dev_mjpeg.enabled else None
    )
    return models, clip, dev_mjpeg


def replay_trace_directory_from_environment(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the opt-in local replay trace directory, if configured."""
    env = os.environ if environ is None else environ
    raw = env.get(WORKER_REPLAY_TRACE_DIR_ENV, "").strip()
    return None if not raw else Path(raw)


__all__ = [
    "FALL_MODELS_ROOT",
    "FALL_SELECTION_PATH",
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
    "WORKER_REPLAY_TRACE_DIR_ENV",
    "clip_recording_config_from_environment",
    "fall_model_config_from_environment",
    "reject_retired_worker_environment",
    "replay_trace_directory_from_environment",
    "resolve_local_overrides",
    "selected_fall_bundle_config_from_environment",
    "worker_models_config_from_environment",
]

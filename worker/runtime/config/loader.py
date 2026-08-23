from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import JsonValue, TypeAdapter, ValidationError

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import WorkerConfig

ML_WORKER_DEV_MJPEG_ENV: Final = "ML_WORKER_DEV_MJPEG"
ML_WORKER_DEV_MJPEG_HOST_ENV: Final = "ML_WORKER_DEV_MJPEG_HOST"
ML_WORKER_DEV_MJPEG_PORT_ENV: Final = "ML_WORKER_DEV_MJPEG_PORT"
_YAML_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def load_worker_config(path: str | Path) -> WorkerConfig:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        raise WorkerConfigError(
            f"worker camera config must be YAML, JSON is not supported: {config_path}"
        )
    try:
        raw = _YAML_VALUE_ADAPTER.validate_python(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
        if not isinstance(raw, dict):
            raise WorkerConfigError(
                f"worker camera config must contain a YAML mapping: {config_path}"
            )
        cameras = raw.get("cameras")
        if isinstance(cameras, list) and cameras:
            raise WorkerConfigError(
                "static camera roster is retired; register cameras in the dashboard "
                "and pull the versioned worker config"
            )
        # Reject by field presence before WorkerConfig can parse, normalize, or
        # validate any policy content. Static YAML is never a numeric-policy
        # authority, including when its document is malformed or forged.
        if "detection_policies" in raw:
            raise WorkerConfigError(
                "static detection_policies authority is retired; numeric detection "
                "policies must be pulled from the central service via the versioned "
                "worker config"
            )
        for field in ("models", "domains", "clip"):
            if field in raw:
                raise WorkerConfigError(
                    f"static {field} policy is retired; use the versioned worker "
                    "config authority (Todo 9 owns detailed policy revisions)"
                )
        return WorkerConfig.model_validate(raw)
    except OSError as error:
        raise WorkerConfigError(f"worker camera config not readable: {config_path}") from error
    except yaml.YAMLError as error:
        raise WorkerConfigError(f"worker camera config is not valid YAML: {config_path}") from error
    except ValidationError as error:
        fields = ", ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_url=False, include_input=False)
        )
        raise WorkerConfigError(f"worker camera config invalid: {fields}") from error


def resolve_config_path(value: str | None = None) -> Path:
    """Resolve an explicit ``--config`` YAML roster path.

    Deliberately reads no environment variable. A camera roster must never
    arrive through the environment (and therefore through compose, and
    therefore through Git); on a real Edge the roster is pulled from ml-api,
    which serves the dashboard-entered camera_registry DB. ``--config`` stays
    available as an explicit developer/e2e escape hatch.
    """
    stripped = (value or "").strip()
    if not stripped:
        raise WorkerConfigError("worker camera config path required via --config")
    return Path(stripped)


__all__ = [
    "ML_WORKER_DEV_MJPEG_ENV",
    "ML_WORKER_DEV_MJPEG_HOST_ENV",
    "ML_WORKER_DEV_MJPEG_PORT_ENV",
    "load_worker_config",
    "resolve_config_path",
]

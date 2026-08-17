"""ml-api settings."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import ClassVar, Final

from pydantic_settings import BaseSettings, SettingsConfigDict

_RETIRED_BACKEND_ENV: Final = frozenset(
    {
        "API_ALLOW_LEGACY_DASHBOARD_AUTH",
        "API_BACKEND_CONFIG_URL",
        "API_BACKEND_EVENTS_URL",
        "API_CAMERA_INVENTORY",
        "API_CONNECTION_SETTINGS_PATH",
        "API_FACILITY_ID",
        "API_FACILITY_TOKEN",
        "API_BACKEND_FACILITY_TOKEN",
        "API_EDGE_FACILITY_TOKEN",
        "API_LABEL_STORE",
        "CLIP_STORE_DIR",
        "EDGE_FACILITY_TOKEN",
        "ML_API_DETECTION_TZ",
        "ML_API_EVENT_CLIP_EXPORT_ENABLED",
        "ML_API_WORKER_PROBE_ORIGIN",
        "ML_API_WORKER_STREAM_ORIGIN",
        "ML_DEFAULT_CAMERA_FPS",
        "ML_DEFAULT_FRAME_STRIDE",
        "ML_SERVING_PORT",
    }
)


def reject_retired_backend_environment(environ: Mapping[str, str]) -> None:
    """Reject removed authorities using Settings' case-insensitive env rules."""
    present = sorted(original for original in environ if original.upper() in _RETIRED_BACKEND_ENV)
    if present:
        raise ValueError(
            "retired edge environment key(s): "
            + ", ".join(present)
            + "; use edge-env-inventory.json for the replacement authority"
        )


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="ML_API_",
        extra="ignore",
    )

    api_v1_prefix: str = "/api/v1"
    worker_stream_origin: str = "http://ml-worker:8090"
    worker_stream_timeout_s: float = 3.0
    worker_probe_origin: str = "http://ml-worker:8090"
    worker_probe_timeout_s: float = 5.0
    connection_test_timeout_s: float = 5.0
    # On-demand bed segmentation is a heavier, infrequent user action (not
    # periodic polling): the worker route waits up to ~2s for a fresh frame
    # (see BED_ZONE_FRAME_TIMEOUT_SECONDS) before it even runs inference, so
    # this must stay comfortably above worker_stream_timeout_s.
    worker_bed_zone_timeout_s: float = 8.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

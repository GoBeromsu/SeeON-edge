from __future__ import annotations

from typing import ClassVar, Final
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from worker.runtime.config.errors import ConfigValidationError
from worker.types import CURRENT_TEMPORAL_PROFILE

SUPPORTED_DECODE_BACKENDS: Final = frozenset({"auto", "nvdec", "opencv", "cpu"})


class CameraStreamsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    sub: str = Field(min_length=1)
    main: str | None = None

    @field_validator("sub", "main")
    @classmethod
    def _require_rtsp_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_rtsp_url(value, "streams")


class CameraRuntimeConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    resident_id: str | None = None
    rtsp_url: str | None = Field(default=None, min_length=1)
    streams: CameraStreamsConfig | None = None
    # Declared/relay hint only. CapturePolicy.target_fps is owned by the
    # TemporalProfile passed to compose_camera_ingest_loop, not this field.
    fps: float = Field(default=CURRENT_TEMPORAL_PROFILE.target_fps, gt=0)
    heartbeat_interval_sec: float = Field(default=30.0, gt=0)
    frame_stride: int = Field(default=1, gt=0)
    label: str | None = None
    decode_backend: str | None = None
    # Operator-recognized bed polygon (see the bed-zone recognize endpoint),
    # persisted backend-side and pulled down as part of the worker config.
    # It is the authoritative bed region for bed-exit; segmentation is used
    # only on demand to propose a polygon for persistence.
    bed_zone_polygon: tuple[tuple[int, int], ...] | None = None
    bed_zone_image_width: int | None = Field(default=None, gt=0)
    bed_zone_image_height: int | None = Field(default=None, gt=0)

    @field_validator("bed_zone_polygon")
    @classmethod
    def _validate_bed_zone_polygon(
        cls, value: tuple[tuple[int, int], ...] | None
    ) -> tuple[tuple[int, int], ...] | None:
        if value is None:
            return None
        if len(value) < 3:
            raise ConfigValidationError("bed_zone_polygon must have at least 3 points")
        return value

    @field_validator("camera_id")
    @classmethod
    def _require_opaque_camera_id(cls, value: str) -> str:
        """Reject unresolved/log-unsafe input, but never canonicalize an opaque DB key."""
        if not value.strip():
            raise ConfigValidationError("camera_id must not be blank")
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise ConfigValidationError("camera_id contains unsafe control characters")
        return value

    @field_validator("facility_id")
    @classmethod
    def _strip_facility_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ConfigValidationError("must not be blank")
        return stripped

    @field_validator("rtsp_url")
    @classmethod
    def _validate_rtsp_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_rtsp_url(value, "rtsp_url")

    @field_validator("resident_id")
    @classmethod
    def _normalize_resident_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("decode_backend")
    @classmethod
    def _validate_decode_backend(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_DECODE_BACKENDS:
            raise ConfigValidationError("decode_backend must be one of auto, nvdec, opencv, cpu")
        return normalized

    @model_validator(mode="after")
    def _require_inference_stream(self) -> CameraRuntimeConfig:
        if self.rtsp_url is None and self.streams is None:
            raise ConfigValidationError("camera must define rtsp_url or streams.sub")
        return self

    @property
    def inference_rtsp_url(self) -> str:
        if self.streams is not None:
            return self.streams.sub
        if self.rtsp_url is None:
            raise ConfigValidationError("camera must define rtsp_url or streams.sub")
        return self.rtsp_url

    @property
    def main_rtsp_url(self) -> str | None:
        return None if self.streams is None else self.streams.main


class RelayConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    token: SecretStr = Field(repr=False)

    @field_validator("url")
    @classmethod
    def _require_http_url(cls, value: str) -> str:
        stripped = value.strip()
        parsed = urlsplit(stripped)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ConfigValidationError("relay.url must be absolute HTTP(S)")
        if parsed.query or parsed.fragment:
            raise ConfigValidationError("relay.url must not include query or fragment")
        return urlunsplit(parsed._replace(path=parsed.path.rstrip("/")))


def _normalize_rtsp_url(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped.lower().startswith("rtsp://"):
        raise ConfigValidationError(f"{field_name} must start with rtsp://")
    return stripped


__all__ = [
    "SUPPORTED_DECODE_BACKENDS",
    "CameraRuntimeConfig",
    "CameraStreamsConfig",
    "RelayConfig",
]

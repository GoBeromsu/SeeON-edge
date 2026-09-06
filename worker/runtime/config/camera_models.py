from __future__ import annotations

from typing import ClassVar, Final, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    field_validator,
    model_validator,
)

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


class BedZoneRegionConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, strict=True)
    polygon: tuple[tuple[StrictInt, StrictInt], ...] = Field(min_length=3, max_length=16)
    origin: Literal["manual", "model"]

    @field_validator("id")
    @classmethod
    def _require_nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ConfigValidationError("bed zone region id must not be blank")
        return value


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
    # Persisted operator-approved regions are authoritative for bed-exit.
    # Recognition only proposes regions for explicit persistence.
    bed_zone_regions: tuple[BedZoneRegionConfig, ...] = Field(default=(), max_length=8)
    bed_zone_image_width: int | None = Field(default=None, gt=0)
    bed_zone_image_height: int | None = Field(default=None, gt=0)

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
        if self.bed_zone_regions:
            if self.bed_zone_image_width is None or self.bed_zone_image_height is None:
                raise ConfigValidationError("bed zone regions require image width and height")
            region_ids = tuple(region.id for region in self.bed_zone_regions)
            if len(set(region_ids)) != len(region_ids):
                raise ConfigValidationError("bed zone region ids must be distinct")
            for region in self.bed_zone_regions:
                if any(
                    x < 0
                    or x >= self.bed_zone_image_width
                    or y < 0
                    or y >= self.bed_zone_image_height
                    for x, y in region.polygon
                ):
                    raise ConfigValidationError(
                        "bed zone polygon points must be within source image bounds"
                    )
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
    "BedZoneRegionConfig",
    "CameraRuntimeConfig",
    "CameraStreamsConfig",
    "RelayConfig",
]

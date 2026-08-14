"""Validated schema-v2 evidence manifest models."""

from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction
from typing import ClassVar, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_core import PydanticCustomError

from worker.pipeline.output.evidence.evidence_metadata import (
    validate_runtime_manifest_sha256,
)
from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode


class RemuxStreamFacts(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    index: int
    media_type: str | None = None
    codec_name: str | None = None
    codec_tag: str | None = None
    time_base: str
    extradata_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    packet_count: int
    timestamp_translation_ticks: int | None = None

    @model_validator(mode="after")
    def _typed_values(self) -> Self:
        if self.index < 0 or self.packet_count < 0:
            raise PydanticCustomError("remux_stream", "remux stream counters are invalid")
        _ = _fraction(self.time_base)
        if self.extradata_sha256 is not None:
            _sha256(self.extradata_sha256, "extradata_sha256")
        for value in (self.width, self.height, self.sample_rate, self.channels):
            if value is not None and value <= 0:
                raise PydanticCustomError("remux_stream", "remux stream dimensions are invalid")
        return self


class SourceMediaFacts(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    configuration_id: str | None = None
    selected_start_pts_sec: float | None = None
    selected_end_pts_sec: float | None = None
    packet_count: int | None = None
    remux_method: str | None = None
    remux_version: str | None = None
    timestamp_translation_seconds: str
    streams: tuple[RemuxStreamFacts, ...]

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        translation = _fraction(self.timestamp_translation_seconds)
        if self.packet_count is not None and self.packet_count < 0:
            raise PydanticCustomError("source_media", "source packet count is invalid")
        if not self.streams:
            raise PydanticCustomError("source_media", "source streams must not be empty")
        for stream in self.streams:
            if stream.timestamp_translation_ticks is None:
                continue
            stream_translation = Fraction(stream.timestamp_translation_ticks) * _fraction(
                stream.time_base
            )
            if stream_translation != translation:
                raise PydanticCustomError("source_media", "nonuniform remux timestamp translation")
        return self


class TimeOriginFacts(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    generation: int
    media_origin_pts_sec: float
    event_pts_sec: float
    requested_start_pts_sec: float
    requested_end_pts_sec: float
    event_media_time_ms: float

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if not self.worker_boot_id or not self.camera_id:
            raise PydanticCustomError("time_origin", "time origin identity is missing")
        if self.stream_epoch < 0 or self.generation < 0:
            raise PydanticCustomError("time_origin", "time origin counters are invalid")
        expected = (self.event_pts_sec - self.media_origin_pts_sec) * 1000.0
        if abs(expected - self.event_media_time_ms) > 0.001:
            raise PydanticCustomError("time_origin", "event media time is inconsistent")
        if self.requested_start_pts_sec > self.requested_end_pts_sec:
            raise PydanticCustomError("time_origin", "requested time window is inverted")
        return self


class _ManifestProvenance(BaseModel):
    # Final on-disk manifests may carry forward-compatible keys from older
    # writers; repair and publication both read them with ignore semantics.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    event_ref: str | None = None
    event_type: str | None = None
    domain: str | None = None
    decision_trace_id: str | None = None
    started_at: str | None = None
    duration_s: float | None = None
    encoder: str | None = None
    path: str | None = None
    finalized: bool | None = None
    video_available: bool | None = None
    recovery_state: Literal["MEDIA_VERIFIED", "UNAVAILABLE"] | None = None
    source_media: SourceMediaFacts | None = None
    source_error_reason: str | None = None
    truncation_reasons: tuple[str, ...] = ()
    time_origin: TimeOriginFacts | None = None

    @field_validator("decision_trace_id")
    @classmethod
    def _decision_trace_id(cls, value: str | None) -> str | None:
        if value is not None:
            _sha256(value, "decision_trace_id")
        return value

    @field_validator("started_at")
    @classmethod
    def _started_at(cls, value: str | None) -> str | None:
        return None if value is None else normalized_timestamp(value)

    @field_validator("event_type", "domain", "encoder", "source_error_reason")
    @classmethod
    def _optional_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise PydanticCustomError("manifest_text", "manifest text fact is empty")
        return value


class ReadyClipManifest(_ManifestProvenance):
    manifest_schema_version: Literal[2] = 2
    state: Literal["READY"] = "READY"
    clip_id: str
    camera_id: str
    event_refs: tuple[str, ...]
    clip_start_at: str
    clip_end_at: str
    finalized_at: str
    sha256: str
    size_bytes: int
    mime_type: Literal["video/mp4"] = "video/mp4"
    codec: str = "h264"
    audio_codec: str | None = None
    duration_ms: int
    state_version: Literal[2] = 2
    runtime_manifest_sha256: str | None = None

    @field_validator("runtime_manifest_sha256", mode="before")
    @classmethod
    def _runtime_manifest_sha256(cls, value: object) -> str | None:
        return validate_runtime_manifest_sha256(value)

    @field_validator("clip_id", "camera_id")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        if value.strip() == "":
            raise PydanticCustomError("empty_identity", "identity must not be empty")
        return value

    @field_validator("event_refs")
    @classmethod
    def _ordered_unique_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_event_refs(values)

    @field_validator("clip_start_at", "clip_end_at", "finalized_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        return normalized_timestamp(value)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise PydanticCustomError("sha256", "sha256 must be lowercase hexadecimal")
        return value

    @field_validator("codec", "audio_codec")
    @classmethod
    def _codec_name(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise PydanticCustomError("codec", "codec must not be blank")
        return value

    @field_validator("size_bytes")
    @classmethod
    def _positive_size(cls, value: int) -> int:
        if value <= 0:
            raise PydanticCustomError("size_bytes", "size_bytes must be positive")
        return value

    @field_validator("duration_ms")
    @classmethod
    def _bounded_duration(cls, value: int) -> int:
        if not 1 <= value <= 120_000:
            raise PydanticCustomError("duration_ms", "duration_ms is out of range")
        return value

    @model_validator(mode="after")
    def _ordered_timestamps(self) -> Self:
        start = parse_utc(self.clip_start_at)
        end = parse_utc(self.clip_end_at)
        finalized = parse_utc(self.finalized_at)
        if not start <= end <= finalized:
            raise PydanticCustomError("timestamps", "clip timestamps are not ordered")
        return self


class UnavailableClipManifest(_ManifestProvenance):
    manifest_schema_version: Literal[2] = 2
    state: Literal["UNAVAILABLE"] = "UNAVAILABLE"
    clip_id: str
    camera_id: str
    event_refs: tuple[str, ...]
    clip_start_at: str
    clip_end_at: str
    finalized_at: str
    state_version: Literal[2] = 2
    reason_code: EvidenceReasonCode
    runtime_manifest_sha256: str | None = None

    @field_validator("runtime_manifest_sha256", mode="before")
    @classmethod
    def _runtime_manifest_sha256(cls, value: object) -> str | None:
        return validate_runtime_manifest_sha256(value)

    @field_validator("clip_id", "camera_id")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        if value.strip() == "":
            raise PydanticCustomError("empty_identity", "identity must not be empty")
        return value

    @field_validator("event_refs")
    @classmethod
    def _ordered_unique_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_event_refs(values)

    @field_validator("clip_start_at", "clip_end_at", "finalized_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        return normalized_timestamp(value)

    @model_validator(mode="after")
    def _ordered_timestamps(self) -> Self:
        start = parse_utc(self.clip_start_at)
        end = parse_utc(self.clip_end_at)
        finalized = parse_utc(self.finalized_at)
        if not start <= end <= finalized:
            raise PydanticCustomError("timestamps", "clip timestamps are not ordered")
        return self


ClipManifest: TypeAlias = ReadyClipManifest | UnavailableClipManifest


def _fraction(value: str) -> Fraction:
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise PydanticCustomError("rational", "rational fact is invalid") from exc
    return parsed


def _sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PydanticCustomError(
            "sha256",
            "{field} must be lowercase SHA-256",
            {"field": field},
        )


def validate_event_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise PydanticCustomError("event_refs", "event_refs must not be empty")
    if len(set(values)) != len(values):
        raise PydanticCustomError("event_refs", "event_refs must be unique")
    for value in values:
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError) as exc:
            raise PydanticCustomError(
                "event_refs",
                "event_refs must be canonical UUIDv4",
            ) from exc
        if parsed.version != 4 or str(parsed) != value:
            raise PydanticCustomError(
                "event_refs",
                "event_refs must be canonical UUIDv4",
            )
    return values


def coalesce_event_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def normalized_timestamp(value: str) -> str:
    parsed = parse_utc(value)
    body = value.removesuffix("Z").removesuffix("+00:00")
    if "." in body and len(body.rsplit(".", maxsplit=1)[1]) == 3:
        return rfc3339_milliseconds(parsed)
    return rfc3339_z(parsed)


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PydanticCustomError("utc_timestamp", "timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PydanticCustomError("utc_timestamp", "timestamp must be aware")
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise PydanticCustomError("utc_timestamp", "timestamp must be UTC")
    return parsed.astimezone(UTC)


def rfc3339_milliseconds(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def rfc3339_z(value: datetime) -> str:
    normalized = value.astimezone(UTC).isoformat(timespec="microseconds")
    body = normalized.removesuffix("+00:00")
    if "." in body:
        prefix, fraction = body.split(".", maxsplit=1)
        body = prefix if fraction.rstrip("0") == "" else f"{prefix}.{fraction.rstrip('0')}"
    return f"{body}Z"


__all__ = [
    "ClipManifest",
    "ReadyClipManifest",
    "UnavailableClipManifest",
    "coalesce_event_refs",
    "parse_utc",
    "rfc3339_milliseconds",
]

"""Validated source-media facts embedded in evidence manifests."""

from __future__ import annotations

from fractions import Fraction
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError


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
    input_framing: str | None = None
    output_framing: str | None = None
    normalizer_version: str | None = None
    parser_caps_sha256: str | None = None

    @model_validator(mode="after")
    def _typed_values(self) -> Self:
        if self.index < 0 or self.packet_count < 0:
            raise PydanticCustomError("remux_stream", "remux stream counters are invalid")
        _ = _fraction(self.time_base)
        if self.extradata_sha256 is not None:
            _sha256(self.extradata_sha256, "extradata_sha256")
        if self.parser_caps_sha256 is not None:
            _sha256(self.parser_caps_sha256, "parser_caps_sha256")
        for value in (self.width, self.height, self.sample_rate, self.channels):
            if value is not None and value <= 0:
                raise PydanticCustomError("remux_stream", "remux stream dimensions are invalid")
        return self


class AuIndexFacts(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = Field(alias="schema")
    path: str
    sha256: str
    size_bytes: int
    count: int

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if self.path != "au-index.cbor" or self.size_bytes < 0 or self.count < 0:
            raise PydanticCustomError("au_index", "AU index facts are invalid")
        _sha256(self.sha256, "au_index.sha256")
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
    au_index: AuIndexFacts | None = None

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


def _fraction(value: str) -> Fraction:
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise PydanticCustomError("rational", "rational fact is invalid") from exc


def _sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PydanticCustomError(
            "sha256",
            "{field} must be lowercase SHA-256",
            {"field": field},
        )


__all__ = ["AuIndexFacts", "SourceMediaFacts"]

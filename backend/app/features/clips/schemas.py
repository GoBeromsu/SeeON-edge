"""HTTP query and response schemas for the clips feature."""

from __future__ import annotations

from typing import ClassVar, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

ClipEventType: TypeAlias = Literal["fall", "bed-exit", "other"]


class ClipManifestResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    event_ref: str = Field(min_length=1)
    event_type: str | None = Field(default=None, min_length=1)
    started_at: str = Field(min_length=1)
    duration_s: float = Field(ge=0)
    codec: str = Field(default="")
    path: str | None = Field(default=None)
    video_available: bool
    video_error: str | None = Field(default=None)
    finalized: bool
    size_bytes: int | None = Field(default=None, ge=0)


class ClipsPaginationResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    limit: int | None = Field(default=None, ge=1, le=100)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)
    has_more: bool


class ListClipsResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clips: list[ClipManifestResponse]
    pagination: ClipsPaginationResponse
    event_type_counts: dict[str, int]


class ClipListQuery(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    camera_id: str | None = Field(default=None, min_length=1)
    event_type: ClipEventType | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_limit_for_offset(self) -> Self:
        if self.limit is None and self.offset > 0:
            raise ValueError("offset requires limit")
        return self


class LabelClipRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    label: Literal["TRUE_POSITIVE", "FALSE_POSITIVE"] | None
    reviewer: str | None = Field(default=None, min_length=1)


class LabelClipResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    label: Literal["TRUE_POSITIVE", "FALSE_POSITIVE"] | None
    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)


class AuditResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    entries: list[dict[str, object]]

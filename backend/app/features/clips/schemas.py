"""HTTP query and response schemas for the clips feature."""

from __future__ import annotations

from typing import ClassVar, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

ClipEventType: TypeAlias = Literal["fall", "bed-exit", "other"]
ClipExtensionBoundary: TypeAlias = Literal["none", "extension_bounded", "extension_raced"]


class ClipExtensionContributorResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    event_ref: str = Field(min_length=1)
    detected_at: str = Field(min_length=1)


class ClipExtensionResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    contributors: list[ClipExtensionContributorResponse]
    duration_s: float = Field(ge=0)
    boundary: ClipExtensionBoundary


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
    thumbnail_available: bool = False
    detected_at: str | None = Field(default=None, min_length=1)
    truncation_reasons: list[str] = Field(default_factory=list)
    extension: ClipExtensionResponse | None = None


class ClipsPaginationResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    limit: int | None = Field(default=None, ge=1, le=100)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)
    has_more: bool
    next_cursor: str | None = None


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
    cursor: str | None = Field(default=None, min_length=1, max_length=384)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_limit_for_offset(self) -> Self:
        if self.limit is None and self.cursor is not None:
            raise ValueError("cursor requires limit")
        if self.offset > 0:
            raise ValueError("offset pagination is retired; use cursor")
        return self


CleanArtifactState = Literal["AVAILABLE", "UNAVAILABLE"]
SnapshotArtifactState = Literal[
    "PENDING",
    "AVAILABLE",
    "UNAVAILABLE",
    "CORRUPT",
    "PURGED",
]


class ClipArtifactViewsResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    clean: CleanArtifactState
    snapshot: SnapshotArtifactState | None = None


class AuditResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    entries: list[dict[str, object]]


class DeleteClipRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    # Explicit exact clip-id confirmation -- not a generic query/GET side
    # effect. The router rejects a mismatch against the path's clip_id.
    confirm_clip_id: str = Field(min_length=1)


ClipDeleteStatus = Literal[
    "PURGED",
    "HELD",
    "MISSING",
    "UNVERIFIABLE",
    "DELETE_FAILED",
    "VERIFICATION_FAILED",
]


class DeleteClipResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    status: ClipDeleteStatus

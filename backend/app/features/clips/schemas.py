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
    thumbnail_available: bool = False


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


ArtifactState = Literal[
    "PENDING",
    "RUNNING",
    "AVAILABLE",
    "UNAVAILABLE",
    "CORRUPT",
    "CANCELLED",
    "NOT_REQUESTED",
]
DerivativeKind = Literal["STILL", "VIDEO"]


class ClipDerivativeResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=1)
    kind: DerivativeKind
    request_id: str = Field(min_length=64, max_length=64)
    state: ArtifactState
    reason: str | None
    attempt_count: int = Field(ge=0)
    mime_type: Literal["image/jpeg", "video/mp4"] | None = None
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    size_bytes: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    start_time_ms: int | None = Field(default=None, ge=0)
    end_time_ms: int | None = Field(default=None, ge=0)
    render_backend: str | None = Field(default=None, min_length=1)
    render_version: str | None = Field(default=None, min_length=1)
    scene_id: str | None = Field(default=None, min_length=64, max_length=64)
    primary_clip_id: str | None = Field(default=None, min_length=1)
    decision_trace_id: str | None = Field(default=None, min_length=64, max_length=64)
    runtime_manifest_sha256: str | None = Field(default=None, min_length=64, max_length=64)


class ClipArtifactViewsResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    clean: ArtifactState
    analysis: ArtifactState
    annotated: ArtifactState
    playback_view: Literal["clean", "annotated"]
    annotated_fallback_to_clean: bool


class ClipAnalysisValueResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: float | None
    missing_reason: str | None


class ClipAnalysisResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    decision_trace_id: str = Field(min_length=64, max_length=64)
    module_qualified_id: str = Field(min_length=1)
    policy_qualified_id: str = Field(min_length=1)
    effective_policy_id: str = Field(min_length=64, max_length=64)
    runtime_manifest_sha256: str = Field(min_length=64, max_length=64)
    reason: str = Field(min_length=1)
    previous_state: str = Field(min_length=1)
    current_state: str = Field(min_length=1)
    triggered: bool
    track_id: int | None
    bed_id: int | None
    values: list[ClipAnalysisValueResponse]


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

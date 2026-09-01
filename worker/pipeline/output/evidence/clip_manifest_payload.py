"""Canonical central evidence manifest payload enrichment."""

from __future__ import annotations

from datetime import UTC, datetime

from worker.pipeline.output.evidence.clip_publication_types import (
    ClipPublicationMetadata,
    JsonValue,
)
from worker.pipeline.output.evidence.evidence_manifest import (
    ReadyClipManifest,
    UnavailableClipManifest,
)


def manifest_payload(
    manifest: ReadyClipManifest | UnavailableClipManifest,
    metadata: ClipPublicationMetadata,
    *,
    path: str | None,
    video_available: bool,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = manifest.model_dump(mode="json", exclude_none=True)
    payload.update(
        {
            "event_ref": str(metadata.event_refs[0]),
            "started_at": _utc_iso(metadata.started_at),
            "duration_s": metadata.duration_s,
            "encoder": metadata.encoder,
            "path": path,
            "finalized": True,
            "video_available": video_available,
            "recovery_state": "MEDIA_VERIFIED" if video_available else "UNAVAILABLE",
        }
    )
    optional = {
        "decision_trace_id": metadata.decision_trace_id,
        "event_type": metadata.event_type,
        "domain": metadata.domain,
        "source_media": metadata.source_media,
        "source_error_reason": metadata.source_error_reason,
        "scene_index": (
            None if metadata.scene_index is None else metadata.scene_index.model_dump(by_alias=True)
        ),
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    if metadata.truncation_reasons:
        payload["truncation_reasons"] = list(metadata.truncation_reasons)
    if metadata.time_origin is not None:
        origin = metadata.time_origin
        payload["time_origin"] = {
            "worker_boot_id": origin.worker_boot_id,
            "camera_id": origin.camera_id,
            "stream_epoch": origin.stream_epoch,
            "generation": origin.generation,
            "media_origin_pts_sec": origin.media_origin_pts_sec,
            "event_pts_sec": origin.event_pts_sec,
            "requested_start_pts_sec": origin.requested_start_pts_sec,
            "requested_end_pts_sec": origin.requested_end_pts_sec,
            "event_media_time_ms": origin.event_media_time_ms,
        }
    return payload


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["manifest_payload"]

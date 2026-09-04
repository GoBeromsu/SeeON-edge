"""Manifest metadata construction for finalized incident windows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from worker.adapters.encode.models import ClipArtifact, RemuxStreamFact
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationMetadata,
    ClipTimeOrigin,
    JsonValue,
)
from worker.pipeline.output.evidence.clip_recorder_models import ActiveClip
from worker.pipeline.output.evidence.clip_recording import ClipReasonCode
from worker.pipeline.output.evidence.decision_trace_reference import (
    DECISION_TRACE_ID_KEY,
    validate_decision_trace_id,
)
from worker.pipeline.output.evidence.evidence_metadata import runtime_manifest_sha256_from_audit
from worker.pipeline.output.evidence.evidence_outbox_types import (
    EdgeEventId,
    EvidenceReasonCode,
)


def publication_metadata(
    active: ActiveClip,
    duration_s: float,
    encoder_name: str,
    artifact: ClipArtifact | None = None,
    *,
    source_error_reason: str | None = None,
    truncation_reasons: tuple[str, ...] = (),
) -> ClipPublicationMetadata:
    finalized_at = datetime.now(UTC)
    clip_end_at = min(active.started_at + timedelta(seconds=duration_s), finalized_at)
    clip_start_at = clip_end_at - timedelta(seconds=duration_s)
    time_origin = None
    if artifact is not None and artifact.media_origin_pts_sec is not None:
        time_origin = ClipTimeOrigin(
            artifact.worker_boot_id,
            artifact.camera_id,
            artifact.stream_epoch,
            artifact.generation,
            artifact.media_origin_pts_sec,
            active.event_time_sec,
            active.start_time_sec,
            active.cutoff_time_sec,
        )
    source_media = _source_media(artifact)
    return ClipPublicationMetadata(
        camera_id=active.reservation.camera_id,
        event_refs=tuple(EdgeEventId(value) for value in active.event_refs),
        event_type=active.event_type,
        clip_start_at=clip_start_at,
        clip_end_at=clip_end_at,
        finalized_at=finalized_at,
        started_at=active.started_at,
        detected_at=active.detected_at,
        duration_s=duration_s,
        encoder=encoder_name,
        runtime_manifest_sha256=runtime_manifest_sha256_from_audit(active.event.audit),
        decision_trace_id=validate_decision_trace_id(
            None if active.event.audit is None else active.event.audit.get(DECISION_TRACE_ID_KEY)
        ),
        time_origin=time_origin,
        source_media=source_media,
        source_error_reason=source_error_reason,
        truncation_reasons=(
            artifact.truncation_reasons if artifact is not None else truncation_reasons
        ),
        domain=active.event.domain,
        facility_id=active.event.facility_id,
    )


def _source_media(artifact: ClipArtifact | None) -> dict[str, JsonValue] | None:
    if artifact is None or artifact.remux_method is None:
        return None
    return {
        "configuration_id": artifact.configuration_id,
        "selected_start_pts_sec": artifact.selected_start_pts_sec,
        "selected_end_pts_sec": artifact.selected_end_pts_sec,
        "packet_count": artifact.packet_count,
        "remux_method": artifact.remux_method,
        "remux_version": artifact.remux_version,
        "timestamp_translation_seconds": (
            f"{artifact.timestamp_translation_seconds.numerator}/"
            f"{artifact.timestamp_translation_seconds.denominator}"
        ),
        "au_index": {
            "path": "au-index.cbor",
            "sha256": artifact.au_index_sha256,
            "size_bytes": artifact.au_index_size_bytes,
            "schema": artifact.au_index_schema,
            "count": artifact.au_index_count,
        },
        "streams": [_stream_payload(stream) for stream in artifact.streams],
    }


def _stream_payload(stream: RemuxStreamFact) -> dict[str, JsonValue]:
    return {
        "index": stream.index,
        "media_type": stream.media_type,
        "codec_name": stream.codec_name,
        "codec_tag": stream.codec_tag,
        "time_base": f"{stream.time_base.numerator}/{stream.time_base.denominator}",
        "extradata_sha256": stream.extradata_sha256,
        "width": stream.width,
        "height": stream.height,
        "sample_rate": stream.sample_rate,
        "channels": stream.channels,
        "packet_count": stream.packet_count,
        "timestamp_translation_ticks": stream.timestamp_translation_ticks,
        "input_framing": stream.input_framing,
        "output_framing": stream.output_framing,
        "normalizer_version": stream.normalizer_version,
        "parser_caps_sha256": stream.parser_caps_sha256,
    }


def evidence_reason(reason_code: ClipReasonCode) -> EvidenceReasonCode:
    return {
        ClipReasonCode.ENCODER_FAILED: EvidenceReasonCode.ENCODER_FAILED,
        ClipReasonCode.REMUX_FAILED: EvidenceReasonCode.FINALIZE_FAILED,
        ClipReasonCode.NO_SEGMENTS: EvidenceReasonCode.NO_FRAMES,
        ClipReasonCode.STREAM_EPOCH_MISMATCH: EvidenceReasonCode.STREAM_EPOCH_MISMATCH,
    }[reason_code]


__all__ = ["evidence_reason", "publication_metadata"]

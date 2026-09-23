"""sdk.frame, policy.consume, model.score, and policy.decision payloads."""

from __future__ import annotations

from shared.events.execution_records import WireRecord
from worker.domains.fall.classifier import FALL_WINDOW_FRAMES
from worker.pipeline.diagnostics.record_builder import (
    PRODUCER_MODEL,
    PRODUCER_POLICY,
    PRODUCER_SDK,
    fall_causal_unit_id,
    frame_causal_unit_id,
    make_record,
    monotonic_or,
    observed_time,
)
from worker.types.metadata import MetadataCounters, MetadataFrame, NativeObservationEvidence
from worker.types.trace import DecisionTraceSnapshot


def sdk_frame_record(
    metadata: MetadataFrame,
    *,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    identity = metadata.identity
    payload = _sdk_payload(metadata.native_observation_evidence)
    payload["seq"] = identity.seq
    payload["source_generation"] = metadata.source_generation
    payload["native_publish_sequence"] = metadata.native_publish_sequence
    time_quality, observed = observed_time(metadata, observed_at_ns)
    return make_record(
        record_kind="sdk.frame",
        camera_id=identity.camera_id,
        worker_boot_id=identity.worker_boot_id,
        source_generation=metadata.source_generation,
        stream_epoch=identity.stream_epoch,
        producer=PRODUCER_SDK,
        observed_at_ns=observed,
        time_quality=time_quality,
        causal_unit_id=frame_causal_unit_id(
            identity.camera_id, identity.worker_boot_id, identity.stream_epoch, identity.seq
        ),
        outcome="accepted",
        payload=payload,
        frame_seq=identity.seq,
        source_pts_ns=identity.source_pts,
    )


def policy_consume_record(
    metadata: MetadataFrame,
    *,
    before: MetadataCounters,
    after: MetadataCounters,
    processed_count: int,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    identity = metadata.identity
    time_quality, observed = observed_time(metadata, observed_at_ns)
    return make_record(
        record_kind="policy.consume",
        camera_id=identity.camera_id,
        worker_boot_id=identity.worker_boot_id,
        source_generation=metadata.source_generation,
        stream_epoch=identity.stream_epoch,
        producer=PRODUCER_POLICY,
        observed_at_ns=observed,
        time_quality=time_quality,
        causal_unit_id=frame_causal_unit_id(
            identity.camera_id, identity.worker_boot_id, identity.stream_epoch, identity.seq
        ),
        outcome="consumed",
        payload={
            "accepted_delta": after.accepted - before.accepted,
            "overwritten_delta": after.overwritten - before.overwritten,
            "late_delta": after.late - before.late,
            "processed_count": processed_count,
        },
        frame_seq=identity.seq,
        source_pts_ns=identity.source_pts,
    )


def model_score_record(
    *,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    frame_seq: int,
    source_pts_ns: int | None,
    track_id: int,
    generation: int | None,
    probability: object,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    evidence = getattr(probability, "model_evidence", None)
    payload: dict[str, object] = {
        "track_id": track_id,
        "generation": generation,
        "fall_transition": getattr(probability, "fall_transition", None),
        "background": getattr(probability, "background", None),
        "fallen": getattr(probability, "fallen", None),
        "window_frames": FALL_WINDOW_FRAMES,
    }
    if evidence is not None:
        payload["raw_logit"] = evidence.raw_logit
        payload["applied_temperature"] = evidence.applied_temperature
        payload["class_origins"] = list(evidence.class_origins)
    return make_record(
        record_kind="model.score",
        camera_id=camera_id,
        worker_boot_id=worker_boot_id,
        source_generation=source_generation,
        stream_epoch=stream_epoch,
        producer=PRODUCER_MODEL,
        observed_at_ns=monotonic_or(observed_at_ns),
        time_quality="monotonic",
        causal_unit_id=fall_causal_unit_id(
            camera_id, worker_boot_id, stream_epoch, track_id, generation
        ),
        outcome="scored",
        payload=payload,
        frame_seq=frame_seq,
        source_pts_ns=source_pts_ns,
    )


def policy_decision_record(
    snapshot: DecisionTraceSnapshot,
    *,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    frame_seq: int,
    source_pts_ns: int | None,
    generation: int | None,
    decision_trace_id: str | None = None,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    return make_record(
        record_kind="policy.decision",
        camera_id=camera_id,
        worker_boot_id=worker_boot_id,
        source_generation=source_generation,
        stream_epoch=stream_epoch,
        producer=PRODUCER_POLICY,
        observed_at_ns=monotonic_or(observed_at_ns),
        time_quality="monotonic",
        causal_unit_id=fall_causal_unit_id(
            camera_id, worker_boot_id, stream_epoch, snapshot.track_id, generation
        ),
        outcome="triggered" if snapshot.triggered else snapshot.current_state,
        payload={
            "reason": snapshot.reason,
            "previous_state": snapshot.previous_state,
            "current_state": snapshot.current_state,
            "triggered": snapshot.triggered,
            "track_id": snapshot.track_id,
            "bed_id": snapshot.bed_id,
            "values": dict(snapshot.values),
            "missing_values": dict(snapshot.missing_values),
            # Same id the relayed alert carries in audit.decision_trace_id;
            # None when the pump had no decision identity to attribute to.
            "decision_trace_id": decision_trace_id,
        },
        frame_seq=frame_seq,
        source_pts_ns=source_pts_ns,
    )


def _sdk_payload(evidence: NativeObservationEvidence | None) -> dict[str, object]:
    if evidence is None:
        return {}
    return {
        "sdk_frame_number": evidence.sdk_frame_number,
        "source_id": evidence.source_id,
        "inference_tensor_present": evidence.inference_tensor_present,
        "raw_output_row_count": evidence.raw_output_row_count,
        "eligible_row_count": evidence.eligible_row_count,
        "matched_row_count": evidence.matched_row_count,
    }


__all__ = [
    "model_score_record",
    "policy_consume_record",
    "policy_decision_record",
    "sdk_frame_record",
]

"""Read-only execution-record emission from the native policy pump."""

from __future__ import annotations

from worker.domains.fall import FallDomainDecider
from worker.interfaces.execution_records import ExecutionRecordSink
from worker.pipeline.decision import EventAggregator
from worker.pipeline.diagnostics.emit import (
    model_score_record,
    policy_consume_record,
    policy_decision_record,
    try_emit,
)
from worker.types.metadata import MetadataCounters, MetadataFrame


def emit_policy_consume(
    sink: ExecutionRecordSink | None,
    metadata: MetadataFrame,
    *,
    before: MetadataCounters,
    after: MetadataCounters,
    processed_count: int,
) -> None:
    try_emit(
        sink,
        policy_consume_record(
            metadata,
            before=before,
            after=after,
            processed_count=processed_count,
        ),
    )


def emit_model_and_decision(
    sink: ExecutionRecordSink | None,
    metadata: MetadataFrame,
    decision: EventAggregator,
) -> None:
    if sink is None:
        return
    identity = metadata.identity
    pts = identity.source_pts
    fall = _fall_decider(decision)
    classifier = None if fall is None else getattr(fall, "classifier", None)
    for snapshot in decision.last_trace_snapshots:
        track_id = snapshot.track_id
        generation = _generation(fall, classifier, track_id)
        if track_id is not None and classifier is not None:
            probability = classifier.probabilities_for(track_id)
            if probability is not None:
                try_emit(
                    sink,
                    model_score_record(
                        camera_id=identity.camera_id,
                        worker_boot_id=identity.worker_boot_id,
                        source_generation=metadata.source_generation,
                        stream_epoch=identity.stream_epoch,
                        frame_seq=identity.seq,
                        source_pts_ns=pts,
                        track_id=track_id,
                        generation=0 if generation is None else generation,
                        probability=probability,
                    ),
                )
        try_emit(
            sink,
            policy_decision_record(
                snapshot,
                camera_id=identity.camera_id,
                worker_boot_id=identity.worker_boot_id,
                source_generation=metadata.source_generation,
                stream_epoch=identity.stream_epoch,
                frame_seq=identity.seq,
                source_pts_ns=pts,
                generation=generation,
            ),
        )


def _fall_decider(decision: EventAggregator) -> FallDomainDecider | None:
    for decider in decision.deciders:
        target: object = decider
        while hasattr(target, "decider"):
            target = target.decider
        if isinstance(target, FallDomainDecider):
            return target
    return None


def _generation(
    fall: FallDomainDecider | None, classifier: object, track_id: int | None
) -> int | None:
    if track_id is None:
        return None
    if classifier is not None:
        value = getattr(classifier, "generation_for", None)
        if callable(value):
            generation = value(track_id)
            if isinstance(generation, int):
                return generation
    if fall is not None:
        return fall.policy.generation_for(track_id)
    return None


__all__ = ["emit_model_and_decision", "emit_policy_consume"]

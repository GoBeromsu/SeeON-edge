"""Read-only execution-record emission from the native policy pump."""

from __future__ import annotations

from collections.abc import Mapping

from worker.domains.fall import FallDomainDecider
from worker.interfaces.execution_records import ExecutionRecordSink
from worker.pipeline.decision import EventAggregator, unwrap_decider
from worker.pipeline.diagnostics.emit_policy import (
    model_score_record,
    policy_coast_record,
    policy_consume_record,
    policy_decision_record,
)
from worker.pipeline.diagnostics.record_builder import try_emit
from worker.types.metadata import MetadataCounters, MetadataFrame
from worker.types.trace import DecisionIdentity, decision_trace_id


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
    """Emit model.score and policy.decision for this frame, per producing decider.

    Every snapshot is attributed to the decider that produced it (identity and
    authority role come from the aggregator, never borrowed across deciders).
    model.score is emitted only for snapshots of the fall decider whose track
    the classifier actually scored on this call; bed-exit and window-gated
    snapshots never yield a fall model record.
    """
    if sink is None:
        return
    identity = metadata.identity
    pts = identity.source_pts
    fall = _fall_decider(decision)
    fall_index = None if fall is None else decision.index_of(fall)
    classifier = None if fall is None else getattr(fall, "classifier", None)
    # FallDomainDecider enforces that every classifier reports which live
    # tracks it deliberately did not score on this call; read it plainly.
    # probabilities_for() would return the cached score for those tracks, and
    # a model.score record must mean "this CPU model call".
    not_scored: Mapping[int, object] = (
        {} if classifier is None else classifier.current_call_missing_score_reasons
    )
    coasted_modules: dict[int, DecisionIdentity | None] = {}
    for attributed in decision.attributed_trace_snapshots():
        if not attributed.fresh:
            # The producing decider coasted on this frame: its snapshots are
            # from an earlier frame and must not be stamped with this frame's
            # identity. Record the coast itself once per decider instead.
            coasted_modules.setdefault(attributed.producer_index, attributed.identity)
            continue
        snapshot = attributed.snapshot
        module_identity = attributed.identity
        # Structural attribution: the snapshot came from the fall decider itself,
        # whether or not composition supplied it an identity.
        is_fall = fall_index is not None and attributed.producer_index == fall_index
        track_id = snapshot.track_id
        generation = _generation(fall, classifier, track_id) if is_fall else None
        if (
            is_fall
            and attributed.authority == "authoritative"
            and track_id is not None
            and classifier is not None
            and track_id not in not_scored
        ):
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
                        generation=generation,
                        probability=probability,
                    ),
                )
        try_emit(
            sink,
            policy_decision_record(
                snapshot,
                module_qualified_id=(
                    None if module_identity is None else module_identity.module_qualified_id
                ),
                authority_role=attributed.authority,
                decision_trace_id=(
                    None
                    if module_identity is None
                    else decision_trace_id(
                        snapshot,
                        module_qualified_id=module_identity.module_qualified_id,
                        effective_policy_id=module_identity.effective_policy_id,
                    )
                ),
                camera_id=identity.camera_id,
                worker_boot_id=identity.worker_boot_id,
                source_generation=metadata.source_generation,
                stream_epoch=identity.stream_epoch,
                frame_seq=identity.seq,
                source_pts_ns=pts,
                generation=generation,
            ),
        )

    for module_identity in coasted_modules.values():
        try_emit(
            sink,
            policy_coast_record(
                camera_id=identity.camera_id,
                worker_boot_id=identity.worker_boot_id,
                source_generation=metadata.source_generation,
                stream_epoch=identity.stream_epoch,
                frame_seq=identity.seq,
                source_pts_ns=pts,
                module_qualified_id=(
                    None if module_identity is None else module_identity.module_qualified_id
                ),
            ),
        )


def _fall_decider(decision: EventAggregator) -> FallDomainDecider | None:
    for decider in decision.deciders:
        target = unwrap_decider(decider)
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

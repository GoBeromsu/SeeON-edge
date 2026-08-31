from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from worker.pipeline.analytics.composite import CompositeResult
from worker.pipeline.trace.models import (
    AnalysisTrace,
    DecisionTrace,
    OptionalNumber,
    TraceBed,
    TraceComponent,
    TraceContractError,
    TraceFrame,
    TraceKeypoint,
    TracePersistenceError,
    TracePerson,
    content_id,
    require_component_qualified,
    require_qualified,
    require_sha256,
)
from worker.types import BusinessEvent, DecisionTraceSnapshot, FramePacket
from worker.types.trace import canonical_trace_number

if TYPE_CHECKING:
    from worker.pipeline.trace.writer import BoundedTraceWriter

SnapshotProvider = Callable[[], object]


@dataclass(frozen=True, slots=True)
class TraceIdentity:
    module_qualified_id: str
    component_qualified_ids: tuple[str, ...]
    policy_qualified_id: str
    effective_policy_id: str
    runtime_manifest_sha256: str
    snapshot_provider: SnapshotProvider | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        require_qualified(self.module_qualified_id, "module_qualified_id")
        require_qualified(self.policy_qualified_id, "policy_qualified_id")
        require_sha256(self.effective_policy_id, "effective_policy_id")
        require_sha256(self.runtime_manifest_sha256, "runtime_manifest_sha256")
        if not self.component_qualified_ids:
            raise TraceContractError("component_qualified_ids must not be empty")
        for component in self.component_qualified_ids:
            require_component_qualified(component)

    @property
    def module_id(self) -> str:
        return self.module_qualified_id.rsplit(".v", 1)[0]


@dataclass(frozen=True, slots=True)
class TraceCapture:
    identities: tuple[TraceIdentity, ...]

    def __post_init__(self) -> None:
        if not self.identities:
            raise TraceContractError("trace capture requires at least one module identity")
        modules = tuple(identity.module_qualified_id for identity in self.identities)
        if len(modules) != len(set(modules)):
            raise TraceContractError("trace capture module identities must be unique")

    def build(
        self,
        packet: FramePacket,
        result: CompositeResult,
        events: Sequence[BusinessEvent],
    ) -> TraceFrame:
        snapshots_by_identity = tuple(snapshots_for(identity) for identity in self.identities)
        analysis = self._analysis(packet, result, snapshots_by_identity)
        decisions: list[DecisionTrace] = []
        for identity_index, (identity, snapshots) in enumerate(
            zip(self.identities, snapshots_by_identity, strict=True)
        ):
            for snapshot in snapshots:
                if len(snapshot.values) + len(snapshot.missing_values) > 256:
                    raise TraceContractError("decision trace contains too many numeric values")
                body = {
                    "analysis_trace_id": analysis.trace_id,
                    "identity_index": identity_index,
                    "module": identity.module_qualified_id,
                    "policy": identity.policy_qualified_id,
                    "effective_policy_id": identity.effective_policy_id,
                    "runtime_manifest_sha256": identity.runtime_manifest_sha256,
                    "reason": snapshot.reason,
                    "previous_state": snapshot.previous_state,
                    "current_state": snapshot.current_state,
                    "triggered": snapshot.triggered,
                    "track_id": snapshot.track_id,
                    "bed_id": snapshot.bed_id,
                    "values": dict(snapshot.values),
                    "missing_values": dict(snapshot.missing_values),
                }
                decisions.append(
                    DecisionTrace(
                        trace_id=content_id(body),
                        analysis_trace_id=analysis.trace_id,
                        identity_index=identity_index,
                        module_qualified_id=identity.module_qualified_id,
                        policy_qualified_id=identity.policy_qualified_id,
                        effective_policy_id=identity.effective_policy_id,
                        runtime_manifest_sha256=identity.runtime_manifest_sha256,
                        snapshot=snapshot,
                    )
                )
        return TraceFrame(analysis, tuple(decisions))

    def capture(
        self,
        writer: BoundedTraceWriter,
        packet: FramePacket,
        result: CompositeResult,
        events: Sequence[BusinessEvent],
        *,
        require_persisted: bool = False,
    ) -> tuple[BusinessEvent, ...] | bool:
        frame = self.build(packet, result, events)
        persisted = writer.submit(frame, require_persisted=require_persisted)
        if not events:
            return persisted
        if not persisted:
            raise TracePersistenceError(
                "admitted event trace could not enter the bounded persistence handoff"
            )
        return tuple(_attach_trace(event, frame.decisions, self.identities) for event in events)

    def _analysis(
        self,
        packet: FramePacket,
        result: CompositeResult,
        snapshots_by_identity: tuple[tuple[DecisionTraceSnapshot, ...], ...],
    ) -> AnalysisTrace:
        observation = result.observation
        pts = (
            OptionalNumber(None, "source-not-provided")
            if packet.pts is None
            else OptionalNumber(canonical_trace_number(packet.pts))
        )
        source_time = (
            OptionalNumber(None, "source-not-provided")
            if result.decision_input.time_sec is None
            else OptionalNumber(canonical_trace_number(result.decision_input.time_sec))
        )
        provenance = result.decision_input.bed_region.source.value
        persons = tuple(
            TracePerson(
                ordinal=index,
                track_id=(
                    OptionalNumber(None, "tracker-unmatched")
                    if track_id is None
                    else OptionalNumber(track_id)
                ),
                box=(box.x1, box.y1, box.x2, box.y2),
                confidence=canonical_trace_number(box.confidence),
                keypoints=tuple(
                    TraceKeypoint(
                        point_index,
                        int(point[0]),
                        int(point[1]),
                        canonical_trace_number(point[2]),
                    )
                    for point_index, point in enumerate(
                        observation.keypoints[index] if index < len(observation.keypoints) else ()
                    )
                ),
            )
            for index, (box, track_id) in enumerate(
                zip(observation.boxes, observation.track_ids, strict=True)
            )
        )
        beds = tuple(
            TraceBed(
                ordinal=index,
                box=(bed.x1, bed.y1, bed.x2, bed.y2),
                confidence=canonical_trace_number(bed.confidence),
                provenance=provenance,
                polygon=() if bed.polygon is None else bed.polygon,
            )
            for index, bed in enumerate(observation.bed_boxes)
        )
        observed_components = {item.module_name for item in result.module_results}
        all_components = tuple(
            dict.fromkeys(
                component
                for identity in self.identities
                for component in identity.component_qualified_ids
            )
        )
        components = tuple(
            TraceComponent(
                index,
                qualified,
                _component_state(
                    qualified,
                    observed_components,
                    self.identities,
                    snapshots_by_identity,
                ),
            )
            for index, qualified in enumerate(all_components)
        )
        body = {
            "schema_version": 1,
            "worker_boot_id": packet.worker_boot_id,
            "camera_id": packet.camera_id,
            "stream_epoch": packet.stream_epoch,
            "frame_seq": packet.seq,
            "pts": pts.value,
            "pts_missing_reason": pts.missing_reason,
            "source_time": source_time.value,
            "source_time_missing_reason": source_time.missing_reason,
            "width": packet.width,
            "height": packet.height,
            "bed_region_provenance": provenance,
            "persons": [
                [
                    person.ordinal,
                    person.track_id.value,
                    person.track_id.missing_reason,
                    *person.box,
                    person.confidence,
                    [
                        [point.index, point.x, point.y, point.confidence]
                        for point in person.keypoints
                    ],
                ]
                for person in persons
            ],
            "beds": [
                [bed.ordinal, *bed.box, bed.confidence, bed.provenance, list(bed.polygon)]
                for bed in beds
            ],
            "components": [
                [component.ordinal, component.qualified_id, component.observation_state]
                for component in components
            ],
        }
        return AnalysisTrace(
            trace_id=content_id(body),
            frame_key=(
                packet.worker_boot_id,
                packet.camera_id,
                packet.stream_epoch,
                packet.seq,
            ),
            pts=pts,
            source_time=source_time,
            frame_width=packet.width,
            frame_height=packet.height,
            bed_region_provenance=provenance,
            persons=persons,
            beds=beds,
            components=components,
        )


def _component_state(
    qualified_id: str,
    observed_components: set[str],
    identities: tuple[TraceIdentity, ...],
    snapshots_by_identity: tuple[tuple[DecisionTraceSnapshot, ...], ...],
) -> str:
    component_id = qualified_id.split(".sha256.", 1)[0]
    if component_id in observed_components:
        return "observed"
    # These are the finite extractor ids in the compiled module graph. Unlike
    # camera-local components, an extractor absent from this frame's results was
    # genuinely not scheduled (or was disabled by its compiled activation flag).
    if component_id in {"pose", "person", "bed"}:
        return "not-scheduled"
    # Tracking executes in analytics before a windowed decider can be skipped.
    if component_id == "person-tracker":
        return "executed"
    relevant_snapshots = tuple(
        snapshots
        for identity, snapshots in zip(identities, snapshots_by_identity, strict=True)
        if qualified_id in identity.component_qualified_ids
    )
    if relevant_snapshots and all(
        snapshots and all(snapshot.current_state == "not-evaluated" for snapshot in snapshots)
        for snapshots in relevant_snapshots
    ):
        return "not-applicable"
    return "executed"


def _unavailable(detail: str) -> tuple[DecisionTraceSnapshot, ...]:
    return (
        DecisionTraceSnapshot(
            reason="trace-unavailable",
            previous_state="unknown",
            current_state="unknown",
            triggered=False,
            track_id=None,
            bed_id=None,
            missing_values={"decision_state": detail},
        ),
    )


def snapshots_for(identity: TraceIdentity) -> tuple[DecisionTraceSnapshot, ...]:
    provider = identity.snapshot_provider
    if provider is None:
        return _unavailable("adapter-not-provided")
    value = provider()
    if isinstance(value, DecisionTraceSnapshot):
        return (value,)
    if isinstance(value, tuple):
        if value and all(isinstance(item, DecisionTraceSnapshot) for item in value):
            return value
        if value:
            raise TraceContractError("trace adapter returned an unvalidated value")
    if (isinstance(value, tuple | Mapping)) and not value:
        return _unavailable("adapter-returned-no-data")
    raise TraceContractError("trace adapter returned an unvalidated value")


def _attach_trace(
    event: BusinessEvent,
    decisions: tuple[DecisionTrace, ...],
    identities: tuple[TraceIdentity, ...],
) -> BusinessEvent:
    candidates = tuple(
        decision
        for decision in decisions
        if decision.snapshot.triggered
        and identities[decision.identity_index].module_id == event.domain
        and (
            decision.snapshot.track_id is None
            or event.person_id is None
            or decision.snapshot.track_id == event.person_id
        )
        and (
            decision.snapshot.bed_id is None
            or event.bed_id is None
            or decision.snapshot.bed_id == event.bed_id
        )
    )
    if len(candidates) != 1:
        raise TracePersistenceError("admitted event does not resolve to exactly one decision trace")
    audit = dict(event.audit or {})
    audit["decision_trace_id"] = candidates[0].trace_id
    return replace(event, audit=MappingProxyType(audit))


__all__ = ["TraceCapture", "TraceIdentity", "snapshots_for"]

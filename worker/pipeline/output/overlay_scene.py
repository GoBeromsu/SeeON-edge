from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from contracts.observation import FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.pipeline.output._overlay_primitives import POSE_EDGES
from worker.pipeline.trace.models import AnalysisTrace, DecisionTrace, OptionalNumber
from worker.types import DecisionTraceSnapshot, FramePacket
from worker.types.overlay_scene import (
    CoordinateTransform,
    ObservationSemantics,
    OverlayScene,
    SceneBed,
    SceneComponent,
    SceneContainment,
    SceneDecision,
    SceneFrameIdentity,
    SceneKeypoint,
    SceneLabel,
    ScenePerson,
    SceneValue,
    fit_scene_transform,
    scene_content_id,
    scene_data,
)

PERSON_COLOR: Final = (0, 255, 0)
BED_COLOR: Final = (128, 128, 0)
POSE_COLOR: Final = (255, 255, 255)
POSE_DOT_COLOR: Final = (255, 255, 255)
DANGER_COLOR: Final = (0, 0, 255)
NEUTRAL_COLOR: Final = (180, 180, 180)
MIN_KEYPOINT_CONFIDENCE: Final = 0.2


@dataclass(frozen=True, slots=True)
class AppliedCameraProvenance:
    runtime_manifest_sha256: str
    camera_configuration_id: str


class OverlaySceneBuilder:
    """Build the sole hardware-neutral overlay contract from frozen facts."""

    def from_traces(
        self,
        analysis: AnalysisTrace,
        decisions: tuple[DecisionTrace, ...],
        *,
        provenance: AppliedCameraProvenance,
        transform: CoordinateTransform | None = None,
    ) -> OverlayScene:
        width, height = analysis.frame_width, analysis.frame_height
        resolved_transform = transform or fit_scene_transform(width, height, width, height)
        person_decisions = _decisions_by_track(decisions)
        persons = tuple(
            ScenePerson(
                person.ordinal,
                _optional(person.track_id),
                (
                    float(person.box[0]),
                    float(person.box[1]),
                    float(person.box[2]),
                    float(person.box[3]),
                ),
                person.confidence,
                tuple(
                    _keypoint(point.index, point.x, point.y, point.confidence)
                    for point in person.keypoints
                ),
                PERSON_COLOR,
                20,
            )
            for person in analysis.persons
        )
        beds = tuple(
            SceneBed(
                bed.ordinal,
                (
                    float(bed.box[0]),
                    float(bed.box[1]),
                    float(bed.box[2]),
                    float(bed.box[3]),
                ),
                tuple((float(x), float(y)) for x, y in bed.polygon),
                bed.confidence,
                bed.provenance,
                (
                    ObservationSemantics.STALE
                    if bed.provenance in {"cached", "expired"}
                    else ObservationSemantics.PRESENT
                ),
                tuple(
                    _containment(decision)
                    for decision in decisions
                    if decision.snapshot.bed_id == bed.ordinal
                ),
                BED_COLOR,
                10,
            )
            for bed in analysis.beds
        )
        rendered_decisions = tuple(_decision(value) for value in decisions)
        components = tuple(
            _component(value.qualified_id, value.observation_state) for value in analysis.components
        )
        labels = tuple(
            sorted(
                (
                    label
                    for person in persons
                    for label in _person_labels(
                        person, _track_decisions(person_decisions, person.track_id)
                    )
                ),
                key=lambda item: (item.z_order, item.text, item.anchor),
            )
        )
        frame = SceneFrameIdentity(
            *analysis.frame_key,
            _optional(analysis.pts),
            _optional(analysis.source_time),
            provenance.camera_configuration_id,
        )
        return _scene(
            frame,
            width,
            height,
            resolved_transform,
            persons,
            beds,
            rendered_decisions,
            components,
            labels,
        )

    def from_live(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...],
        *,
        mode: Literal["bedexit", "fall"],
    ) -> OverlayScene:
        frame = SceneFrameIdentity(
            packet.worker_boot_id or "not-recorded",
            packet.camera_id,
            packet.stream_epoch,
            packet.seq,
            _number(packet.pts, "source-not-provided"),
            _number(packet.frame.time_sec, "source-not-provided"),
            "not-recorded",
        )
        persons = tuple(
            ScenePerson(
                index,
                _number(track_id, "tracker-unmatched"),
                (float(box.x1), float(box.y1), float(box.x2), float(box.y2)),
                box.confidence,
                tuple(
                    _keypoint(i, *point)
                    for i, point in enumerate(
                        observation.keypoints[index] if index < len(observation.keypoints) else ()
                    )
                ),
                PERSON_COLOR,
                20,
            )
            for index, (box, track_id) in enumerate(
                zip(
                    observation.boxes,
                    observation.track_ids or (None,) * len(observation.boxes),
                    strict=True,
                )
            )
        )
        occupancy = {
            status.bed_id: status.occupancy for snap in debug_snapshots for status in snap.statuses
        }
        beds = (
            ()
            if mode == "fall"
            else tuple(
                SceneBed(
                    index,
                    (float(box.x1), float(box.y1), float(box.x2), float(box.y2)),
                    tuple((float(x), float(y)) for x, y in (box.polygon or ())),
                    box.confidence,
                    "live-applied",
                    ObservationSemantics.PRESENT,
                    (),
                    BED_COLOR,
                    10,
                )
                for index, box in enumerate(observation.bed_boxes)
            )
        )
        fall_labels = _live_fall_labels(observation)
        labels = [SceneLabel("person", person.box[:2], PERSON_COLOR, 30) for person in persons]
        labels.extend(
            SceneLabel(
                f"bed:{occupancy[index]}" if index in occupancy else "bed",
                bed.box[:2],
                BED_COLOR,
                30,
            )
            for index, bed in enumerate(beds)
        )
        labels.extend(
            SceneLabel(label[0], persons[index].box[:2], label[1], 40)
            for index, label in fall_labels.items()
            if index < len(persons)
        )
        transform = fit_scene_transform(packet.width, packet.height, packet.width, packet.height)
        return _scene(
            frame,
            packet.width,
            packet.height,
            transform,
            persons,
            beds,
            (),
            (),
            tuple(sorted(labels, key=lambda item: (item.z_order, item.text))),
        )

    def from_observation(
        self,
        *,
        identity: SceneFrameIdentity,
        observation: FrameObservation,
        source_width: int,
        source_height: int,
        decisions: tuple[DecisionTrace, ...],
    ) -> OverlayScene:
        """Build an image-free source-pixel scene from one current observation.

        The caller supplies a fully formed frame identity and decisions read
        immediately after the decider update. This path intentionally emits
        person and decision labels only; bed status labels need live debug
        snapshots, which are not part of this hardware-neutral input.
        """
        transform = fit_scene_transform(source_width, source_height, source_width, source_height)
        person_decisions = _decisions_by_track(decisions)
        persons = tuple(
            ScenePerson(
                index,
                _number(track_id, "tracker-unmatched"),
                (float(box.x1), float(box.y1), float(box.x2), float(box.y2)),
                box.confidence,
                tuple(
                    _keypoint(point_index, *point)
                    for point_index, point in enumerate(
                        observation.keypoints[index] if index < len(observation.keypoints) else ()
                    )
                ),
                PERSON_COLOR,
                20,
            )
            for index, (box, track_id) in enumerate(
                zip(
                    observation.boxes,
                    observation.track_ids or (None,) * len(observation.boxes),
                    strict=True,
                )
            )
        )
        beds = tuple(
            SceneBed(
                index,
                (float(box.x1), float(box.y1), float(box.x2), float(box.y2)),
                tuple((float(x), float(y)) for x, y in (box.polygon or ())),
                box.confidence,
                "observed",
                ObservationSemantics.PRESENT,
                tuple(
                    _containment(decision)
                    for decision in decisions
                    if decision.snapshot.bed_id == index
                ),
                BED_COLOR,
                10,
            )
            for index, box in enumerate(observation.bed_boxes)
        )
        labels = tuple(
            sorted(
                (
                    label
                    for person in persons
                    for label in _person_labels(
                        person, _track_decisions(person_decisions, person.track_id)
                    )
                ),
                key=lambda item: (item.z_order, item.text, item.anchor),
            )
        )
        return _scene(
            identity,
            source_width,
            source_height,
            transform,
            persons,
            beds,
            tuple(_decision(value) for value in decisions),
            (),
            labels,
        )


def _scene(
    frame: SceneFrameIdentity,
    width: int,
    height: int,
    transform: CoordinateTransform,
    persons: tuple[ScenePerson, ...],
    beds: tuple[SceneBed, ...],
    decisions: tuple[SceneDecision, ...],
    components: tuple[SceneComponent, ...],
    labels: tuple[SceneLabel, ...],
) -> OverlayScene:
    body = {
        "frame": scene_data(frame),
        "source_dimensions": [width, height],
        "coordinate_space": "source-pixels",
        "transform": scene_data(transform),
        "persons": scene_data(persons),
        "beds": scene_data(beds),
        "decisions": scene_data(decisions),
        "components": scene_data(components),
        "labels": scene_data(labels),
        "schema_version": 1,
    }
    return OverlayScene(
        scene_content_id(body),
        frame,
        (width, height),
        "source-pixels",
        transform,
        persons,
        beds,
        decisions,
        components,
        labels,
    )


def _optional(value: OptionalNumber) -> SceneValue:
    return _number(value.value, value.missing_reason or "not-recorded")


def _number(value: int | float | None, reason: str) -> SceneValue:
    return (
        SceneValue(value, ObservationSemantics.PRESENT)
        if value is not None
        else SceneValue(None, ObservationSemantics.MISSING, reason)
    )


def _keypoint(index: int, x: int, y: int, confidence: float) -> SceneKeypoint:
    if confidence < MIN_KEYPOINT_CONFIDENCE:
        return SceneKeypoint(
            index, None, confidence, ObservationSemantics.MISSING, "below-confidence-threshold"
        )
    return SceneKeypoint(
        index, (float(x), float(y)), confidence, ObservationSemantics.PRESENT, None
    )


def _component(identity: str, state: str) -> SceneComponent:
    if state in {"observed", "executed"}:
        return SceneComponent(identity, ObservationSemantics.PRESENT, None)
    semantics = (
        ObservationSemantics.MISSING if state == "missing" else ObservationSemantics.NOT_EVALUATED
    )
    return SceneComponent(identity, semantics, state)


def _decision(value: DecisionTrace) -> SceneDecision:
    snapshot = value.snapshot
    score_name, threshold_name = _metric_names(value.module_qualified_id)
    score = _trace_metric(snapshot, score_name)
    threshold = _trace_metric(snapshot, threshold_name)
    counters = {
        name: numeric
        for name, numeric in snapshot.values.items()
        if name not in {score_name, threshold_name}
    }
    semantics = (
        ObservationSemantics.NOT_EVALUATED
        if snapshot.current_state == "not-evaluated"
        else ObservationSemantics.PRESENT
    )
    return SceneDecision(
        value.module_qualified_id,
        value.policy_qualified_id,
        value.effective_policy_id,
        value.runtime_manifest_sha256,
        _number(snapshot.track_id, "not-applicable"),
        _number(snapshot.bed_id, "not-applicable"),
        snapshot.previous_state,
        snapshot.current_state,
        snapshot.triggered,
        snapshot.reason,
        score,
        threshold,
        counters,
        semantics,
        DANGER_COLOR if snapshot.triggered else NEUTRAL_COLOR,
        40,
    )


def _metric_names(module: str) -> tuple[str, str]:
    return (
        ("fall_probability", "operating_threshold")
        if module.startswith("fall.")
        else ("containment_ratio", "min_containment")
    )


def _trace_metric(snapshot: DecisionTraceSnapshot, name: str) -> SceneValue:
    values = snapshot.values
    missing = snapshot.missing_values
    if name in values:
        return SceneValue(values[name], ObservationSemantics.PRESENT, name=name)
    return SceneValue(None, ObservationSemantics.MISSING, missing.get(name, "not-evaluated"), name)


def _containment(value: DecisionTrace) -> SceneContainment:
    snap = value.snapshot
    return SceneContainment(
        _number(snap.track_id, "not-applicable"),
        _trace_metric(snap, "containment_ratio"),
        _trace_metric(snap, "min_containment"),
        snap.current_state,
        snap.reason,
    )


def _track_decisions(
    decisions: dict[int, tuple[SceneDecision, ...]], track_id: SceneValue
) -> tuple[SceneDecision, ...]:
    return decisions.get(track_id.value, ()) if isinstance(track_id.value, int) else ()


def _decisions_by_track(values: tuple[DecisionTrace, ...]) -> dict[int, tuple[SceneDecision, ...]]:
    grouped: dict[int, list[SceneDecision]] = {}
    for value in values:
        if value.snapshot.track_id is not None:
            grouped.setdefault(value.snapshot.track_id, []).append(_decision(value))
    return {
        track_id: tuple(
            sorted(
                decisions,
                key=lambda decision: (
                    not decision.triggered,
                    decision.module_qualified_id,
                    decision.current_state,
                ),
            )
        )
        for track_id, decisions in grouped.items()
    }


def _person_labels(
    person: ScenePerson, decisions: tuple[SceneDecision, ...]
) -> tuple[SceneLabel, ...]:
    if not decisions:
        return (SceneLabel("person", person.box[:2], PERSON_COLOR, 30),)
    return tuple(
        SceneLabel(
            decision.current_state.upper(),
            (person.box[0], person.box[1] + index * 18),
            decision.color,
            40,
        )
        for index, decision in enumerate(decisions)
    )


def _live_fall_labels(observation: FrameObservation) -> dict[int, tuple[str, tuple[int, int, int]]]:
    result: dict[int, tuple[str, tuple[int, int, int]]] = {}
    labels = iter(observation.labels)
    for index, track_id in enumerate(observation.track_ids):
        if track_id is None:
            continue
        label = next(labels, None)
        if label is None:
            break
        result[index] = (label.text, DANGER_COLOR if label.is_fall else NEUTRAL_COLOR)
    return result


__all__ = [
    "BED_COLOR",
    "DANGER_COLOR",
    "MIN_KEYPOINT_CONFIDENCE",
    "NEUTRAL_COLOR",
    "PERSON_COLOR",
    "POSE_COLOR",
    "POSE_DOT_COLOR",
    "POSE_EDGES",
    "AppliedCameraProvenance",
    "OverlaySceneBuilder",
]

"""Adapt current Python runner outputs onto PerceptionFrameV1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from contracts.runner import BedBoxOutput, BedRunnerResult, PersonRunnerResult, PoseRunnerResult
from worker.types.perception_frame import (
    LEGACY_ASSOCIATION_STRATEGY,
    PERSON_BOX_CUE_SOURCE,
    AssociationResult,
    BedRegion,
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameFailure,
    PerceptionFrameFailureCode,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
    assemble_perception_frame,
)


def _channel_state(provided: bool, count: int) -> ChannelState:
    if not provided:
        return ChannelState.SKIPPED
    if count == 0:
        return ChannelState.INFERRED_EMPTY
    return ChannelState.INFERRED


def _person_box(row: Sequence[object]) -> PersonBox:
    return PersonBox(
        x1=int(row[0]),  # type: ignore[arg-type]
        y1=int(row[1]),  # type: ignore[arg-type]
        x2=int(row[2]),  # type: ignore[arg-type]
        y2=int(row[3]),  # type: ignore[arg-type]
        confidence=float(row[4]),  # type: ignore[arg-type]
    )


def _keypoint(point: object) -> Keypoint:
    if isinstance(point, Mapping):
        return Keypoint(
            x=int(point["x"]),  # type: ignore[arg-type]
            y=int(point["y"]),  # type: ignore[arg-type]
            score=float(point["score"]),  # type: ignore[arg-type]
        )
    values = cast(Sequence[object], point)
    return Keypoint(
        x=int(values[0]),  # type: ignore[arg-type]
        y=int(values[1]),  # type: ignore[arg-type]
        score=float(values[2]),  # type: ignore[arg-type]
    )


def _pose_row(row: Sequence[object]) -> tuple[Keypoint, ...]:
    if not row:
        return ()
    first = row[0]
    if isinstance(first, (int, float)):
        values = tuple(float(cast(float, value)) for value in row)
        return tuple(
            Keypoint(
                x=int(values[index]),
                y=int(values[index + 1]),
                score=float(values[index + 2]),
            )
            for index in range(0, len(values), 3)
        )
    return tuple(_keypoint(point) for point in row)


def _bed_region(item: BedBoxOutput) -> BedRegion:
    values = tuple(item)
    polygon = None
    if len(values) == 6:
        raw_polygon = values[5]
        if isinstance(raw_polygon, Sequence) and not isinstance(raw_polygon, (str, bytes)):
            polygon = tuple((int(point[0]), int(point[1])) for point in raw_polygon)
    return BedRegion(
        x1=int(values[0]),  # type: ignore[arg-type]
        y1=int(values[1]),  # type: ignore[arg-type]
        x2=int(values[2]),  # type: ignore[arg-type]
        y2=int(values[3]),  # type: ignore[arg-type]
        confidence=float(values[4]),  # type: ignore[arg-type]
        polygon=polygon,
    )


def _identity_mapping(identity: PerceptionFrameIdentity) -> dict[str, object]:
    return {
        "worker_boot_id": identity.worker_boot_id,
        "camera_id": identity.camera_id,
        "stream_epoch": identity.stream_epoch,
        "seq": identity.seq,
        "source_pts": identity.source_pts,
    }


def _parse_identity(payload: object) -> PerceptionFrameIdentity | PerceptionFrameFailure:
    if not isinstance(payload, Mapping):
        return PerceptionFrameFailure(
            code=PerceptionFrameFailureCode.MALFORMED_IDENTITY,
            message="perception identity must be an object",
        )
    try:
        return PerceptionFrameIdentity(
            worker_boot_id=str(payload["worker_boot_id"]),
            camera_id=str(payload["camera_id"]),
            stream_epoch=int(payload["stream_epoch"]),  # type: ignore[arg-type]
            seq=int(payload["seq"]),  # type: ignore[arg-type]
            source_pts=(
                None if payload.get("source_pts") is None else int(payload["source_pts"])  # type: ignore[arg-type]
            ),
        )
    except (KeyError, TypeError, ValueError):
        return PerceptionFrameFailure(
            code=PerceptionFrameFailureCode.MALFORMED_IDENTITY,
            message="perception identity is missing required durable fields",
        )


def _parse_boxes(payload: object) -> tuple[PersonBox, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return ()
    return tuple(
        PersonBox(
            x1=int(item["x1"]),  # type: ignore[arg-type]
            y1=int(item["y1"]),  # type: ignore[arg-type]
            x2=int(item["x2"]),  # type: ignore[arg-type]
            y2=int(item["y2"]),  # type: ignore[arg-type]
            confidence=float(item["confidence"]),  # type: ignore[arg-type]
        )
        for item in payload
        if isinstance(item, Mapping)
    )


def _parse_poses(payload: object) -> tuple[tuple[Keypoint, ...], ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return ()
    return tuple(
        _pose_row(row)
        for row in payload
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
    )


def _parse_regions(payload: object) -> tuple[BedRegion, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return ()
    regions: list[BedRegion] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        raw_polygon = item.get("polygon")
        polygon = None
        if isinstance(raw_polygon, Sequence) and not isinstance(raw_polygon, (str, bytes)):
            polygon = tuple((int(point[0]), int(point[1])) for point in raw_polygon)
        regions.append(
            BedRegion(
                x1=int(item["x1"]),  # type: ignore[arg-type]
                y1=int(item["y1"]),  # type: ignore[arg-type]
                x2=int(item["x2"]),  # type: ignore[arg-type]
                y2=int(item["y2"]),  # type: ignore[arg-type]
                confidence=float(item["confidence"]),  # type: ignore[arg-type]
                polygon=polygon,
            )
        )
    return tuple(regions)


def _parse_channel_state(payload: Mapping[str, object], key: str = "state") -> ChannelState:
    return ChannelState(str(payload.get(key, ChannelState.SKIPPED)))


class PythonInferencePerceptionAdapter:
    """Public adapter from current Python inference outputs to PerceptionFrameV1."""

    def adapt(
        self,
        *,
        identity: PerceptionFrameIdentity,
        pose: PoseRunnerResult | None = None,
        person: PersonRunnerResult | None = None,
        bed: BedRunnerResult | None = None,
        track_ids: tuple[int, ...] | None = None,
        selected_cue_indexes: tuple[int, ...] | None = None,
        association_identity: PerceptionFrameIdentity | None = None,
        association_strategy: str = LEGACY_ASSOCIATION_STRATEGY,
        association_cue_source: str = PERSON_BOX_CUE_SOURCE,
    ) -> PerceptionFrameV1 | PerceptionFrameFailure:
        if person is not None:
            boxes = tuple(_person_box(row) for row in person.boxes)
            person_provided = True
        elif pose is not None:
            boxes = tuple(_person_box(row) for row in pose.boxes)
            person_provided = True
        else:
            boxes = ()
            person_provided = False
        poses = () if pose is None else tuple(_pose_row(row) for row in pose.poses)
        regions = () if bed is None else tuple(_bed_region(item) for item in bed.boxes)
        association = None
        if track_ids is not None or association_identity is not None:
            indexes = (
                selected_cue_indexes
                if selected_cue_indexes is not None
                else tuple(range(len(boxes)))
            )
            association = AssociationResult(
                strategy=association_strategy,
                track_ids=() if track_ids is None else track_ids,
                selected_cue_indexes=indexes,
                identity=identity if association_identity is None else association_identity,
                cue_source=association_cue_source,
            )
        return assemble_perception_frame(
            identity=identity,
            person_box=PersonBoxChannel(
                state=_channel_state(person_provided, len(boxes)),
                boxes=boxes,
            ),
            human_pose=HumanPoseChannel(
                state=_channel_state(pose is not None, len(poses)),
                poses=poses,
            ),
            bed_region=BedRegionChannel(
                state=_channel_state(bed is not None, len(regions)),
                regions=regions,
            ),
            association=association,
        )

    def parse(self, payload: Mapping[str, object]) -> PerceptionFrameV1 | PerceptionFrameFailure:
        identity = _parse_identity(payload.get("identity"))
        if isinstance(identity, PerceptionFrameFailure):
            return identity
        person_payload = payload.get("person_box", {})
        pose_payload = payload.get("human_pose", {})
        bed_payload = payload.get("bed_region", {})
        person_mapping = person_payload if isinstance(person_payload, Mapping) else {}
        pose_mapping = pose_payload if isinstance(pose_payload, Mapping) else {}
        bed_mapping = bed_payload if isinstance(bed_payload, Mapping) else {}
        association_payload = payload.get("association")
        association = None
        if isinstance(association_payload, Mapping):
            association_identity = _parse_identity(association_payload.get("identity"))
            if isinstance(association_identity, PerceptionFrameFailure):
                return association_identity
            raw_tracks = association_payload.get("track_ids", ())
            raw_indexes = association_payload.get("selected_cue_indexes", ())
            association = AssociationResult(
                strategy=str(association_payload.get("strategy", LEGACY_ASSOCIATION_STRATEGY)),
                track_ids=tuple(int(item) for item in raw_tracks)  # type: ignore[arg-type]
                if isinstance(raw_tracks, Sequence) and not isinstance(raw_tracks, (str, bytes))
                else (),
                selected_cue_indexes=tuple(int(item) for item in raw_indexes)  # type: ignore[arg-type]
                if isinstance(raw_indexes, Sequence) and not isinstance(raw_indexes, (str, bytes))
                else (),
                identity=association_identity,
                cue_source=str(association_payload.get("cue_source", PERSON_BOX_CUE_SOURCE)),
            )
        return assemble_perception_frame(
            identity=identity,
            person_box=PersonBoxChannel(
                state=_parse_channel_state(person_mapping),
                boxes=_parse_boxes(person_mapping.get("boxes")),
            ),
            human_pose=HumanPoseChannel(
                state=_parse_channel_state(pose_mapping),
                poses=_parse_poses(pose_mapping.get("poses")),
            ),
            bed_region=BedRegionChannel(
                state=_parse_channel_state(bed_mapping),
                regions=_parse_regions(bed_mapping.get("regions")),
            ),
            association=association,
        )

    def diagnostic(self, frame: PerceptionFrameV1) -> dict[str, object]:
        association = None
        if frame.association is not None:
            association = {
                "strategy": frame.association.strategy,
                "track_ids": list(frame.association.track_ids),
                "selected_cue_indexes": list(frame.association.selected_cue_indexes),
                "cue_source": frame.association.cue_source,
                "identity": _identity_mapping(frame.association.identity),
            }
        return {
            "version": "PerceptionFrameV1",
            "identity": _identity_mapping(frame.identity),
            "person_box": {
                "state": str(frame.person_box.state),
                "boxes": [
                    {
                        "x1": box.x1,
                        "y1": box.y1,
                        "x2": box.x2,
                        "y2": box.y2,
                        "confidence": box.confidence,
                    }
                    for box in frame.person_box.boxes
                ],
            },
            "human_pose": {
                "state": str(frame.human_pose.state),
                "poses": [
                    [{"x": point.x, "y": point.y, "score": point.score} for point in pose]
                    for pose in frame.human_pose.poses
                ],
            },
            "bed_region": {
                "state": str(frame.bed_region.state),
                "regions": [
                    {
                        "x1": region.x1,
                        "y1": region.y1,
                        "x2": region.x2,
                        "y2": region.y2,
                        "confidence": region.confidence,
                        "polygon": (
                            None
                            if region.polygon is None
                            else [list(point) for point in region.polygon]
                        ),
                    }
                    for region in frame.bed_region.regions
                ],
            },
            "association": association,
        }


__all__ = ["PythonInferencePerceptionAdapter"]

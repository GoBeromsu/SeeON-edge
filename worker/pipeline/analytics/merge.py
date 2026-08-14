from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import singledispatch
from typing import TypeAlias, TypeGuard

from contracts.observation import BoundingBox, DetectionResult
from contracts.runner import (
    BedBoxOutput,
    BedRunnerResult,
    BoxOutput,
    DetectionRunnerResult,
    PersonRunnerResult,
    PoseOutput,
    PoseRunnerResult,
    RunnerResult,
)
from worker.types import ModuleResult

RawBox: TypeAlias = tuple[int, int, int, int, float]
Pose: TypeAlias = tuple[tuple[int, int, float], ...]
PosePersonInput: TypeAlias = Sequence[float] | Sequence[Sequence[float]]
PoseRowsInput: TypeAlias = Sequence[PosePersonInput]


class RunnerResultShapeError(ValueError):
    def __init__(self, module_name: str, detail: str) -> None:
        super().__init__(f"invalid {module_name} runner result: {detail}")


@dataclass(frozen=True, slots=True)
class MergedOutputs:
    detections: DetectionResult | None = None
    raw_boxes: tuple[RawBox, ...] | None = None
    poses: tuple[Pose, ...] | None = None
    bed_boxes: tuple[BoundingBox, ...] | None = None


ResultMerger: TypeAlias = Callable[[RunnerResult, MergedOutputs], MergedOutputs]


@singledispatch
def _merge_pose_result(
    _result: RunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return merged


def _merge_pose_output(
    result: PoseRunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return replace(
        merged,
        poses=_pose_rows(result.poses),
        raw_boxes=_raw_boxes(result.boxes),
    )


def _merge_pose_detection(
    result: DetectionRunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return replace(
        merged,
        detections=result.detections,
        poses=result.detections.keypoints,
    )


@singledispatch
def _merge_person_result(
    _result: RunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return merged


def _merge_person_output(
    result: PersonRunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return replace(merged, raw_boxes=_raw_boxes(result.boxes))


def _merge_person_detection(
    result: DetectionRunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return replace(merged, detections=result.detections, raw_boxes=None)


@singledispatch
def _merge_bed_result(
    _result: RunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return merged


def _merge_bed_output(
    result: BedRunnerResult,
    merged: MergedOutputs,
) -> MergedOutputs:
    return replace(
        merged,
        bed_boxes=tuple(_bed_box_from_output(item) for item in result.boxes),
    )


_ = _merge_pose_result.register(PoseRunnerResult, _merge_pose_output)
_ = _merge_pose_result.register(DetectionRunnerResult, _merge_pose_detection)
_ = _merge_person_result.register(PersonRunnerResult, _merge_person_output)
_ = _merge_person_result.register(DetectionRunnerResult, _merge_person_detection)
_ = _merge_bed_result.register(BedRunnerResult, _merge_bed_output)


_RESULT_MERGERS: dict[str, ResultMerger] = {
    "pose": _merge_pose_result,
    "person": _merge_person_result,
    "bed": _merge_bed_result,
}


def result_merger_names() -> frozenset[str]:
    return frozenset(_RESULT_MERGERS)


def merge_module_results(results: Sequence[ModuleResult]) -> MergedOutputs:
    merged = MergedOutputs()
    for module_result in results:
        merger = _RESULT_MERGERS.get(module_result.output_adapter or module_result.module_name)
        if merger is not None:
            merged = merger(module_result.result, merged)
    return merged


def authoritative_boxes(merged: MergedOutputs) -> tuple[BoundingBox, ...]:
    if merged.raw_boxes is not None:
        return tuple(BoundingBox(*box) for box in merged.raw_boxes)
    if merged.detections is not None:
        return merged.detections.boxes
    return ()


def _raw_boxes(boxes: BoxOutput) -> tuple[RawBox, ...]:
    converted: list[RawBox] = []
    for box in boxes:
        if len(box) < 5:
            raise RunnerResultShapeError("box", "expected five coordinates")
        converted.append(
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]))
        )
    return tuple(converted)


def _pose_rows(poses: PoseOutput) -> tuple[Pose, ...]:
    rows: PoseRowsInput = poses
    return tuple(_pose_row(row) for row in rows)


def _pose_row(row: PosePersonInput) -> Pose:
    if _is_flat_pose(row):
        if len(row) != 51:
            raise RunnerResultShapeError("pose", "expected 51 flattened values")
        return tuple(
            (int(row[index]), int(row[index + 1]), float(row[index + 2]))
            for index in range(0, 51, 3)
        )
    if _is_nested_pose(row):
        if len(row) != 17 or any(len(point) != 3 for point in row):
            raise RunnerResultShapeError("pose", "expected 17 x/y/confidence points")
        return tuple(
            (int(point[0]), int(point[1]), float(point[2]))
            for point in row
        )
    raise RunnerResultShapeError("pose", "expected flattened or nested numeric values")


def _is_flat_pose(value: PosePersonInput) -> TypeGuard[Sequence[float]]:
    return not value or isinstance(value[0], (int, float))


def _is_nested_pose(
    value: PosePersonInput,
) -> TypeGuard[Sequence[Sequence[float]]]:
    return bool(value) and isinstance(value[0], Sequence)


def _bed_box_from_output(item: BedBoxOutput) -> BoundingBox:
    values = tuple(item)
    if len(values) not in {5, 6}:
        raise RunnerResultShapeError("bed", "expected five values and optional polygon")
    coordinates = tuple(_numeric_bed_value(value) for value in values[:5])
    polygon = _bed_polygon(values[5]) if len(values) == 6 else None
    return BoundingBox(
        int(coordinates[0]),
        int(coordinates[1]),
        int(coordinates[2]),
        int(coordinates[3]),
        coordinates[4],
        polygon,
    )


def _numeric_bed_value(value: float | Sequence[Sequence[int]]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise RunnerResultShapeError("bed", "box coordinates must be numeric")


def _bed_polygon(
    value: float | Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (int, float)):
        raise RunnerResultShapeError("bed", "polygon must contain x/y pairs")
    if any(len(point) != 2 for point in value):
        raise RunnerResultShapeError("bed", "polygon must contain x/y pairs")
    return tuple((int(point[0]), int(point[1])) for point in value)


__all__ = [
    "MergedOutputs",
    "RunnerResultShapeError",
    "authoritative_boxes",
    "merge_module_results",
    "result_merger_names",
]

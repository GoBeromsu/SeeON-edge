"""Binary comparator for Python-reference vs native parser output.

Parity is exact, not approximate. Counts, order, class, confidence, box,
keypoint and polygon must all agree; the first divergence is reported with the
field that diverged so a reviewer can tell an ordering bug from a threshold bug
from a geometry bug. An empty result tuple is the only pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from worker.native.deepstream.parity.parse import ParsedBed, ParsedPerson, ParsedPose

#: Confidence tolerance: one FP32 unit-in-the-last-place at unit magnitude.
#:
#: Boxes, keypoint coordinates, counts, order and class are compared EXACTLY --
#: this tolerance applies only to score fields. Two FP32 forward passes over the
#: same weights and the same input can differ in the last mantissa bit purely
#: from kernel/accumulation order; the measured spread across the pinned corpus
#: is 3.8e-06 (see ``task-3-model-parity.json``), and it appears only on
#: keypoint scores, never on a coordinate or a row count.
#:
#: The value is deliberately far below every decision boundary it could mask:
#: the strict pose cut is 0.05 and the keypoint/policy threshold is 0.2, so a
#: real threshold, ordering or geometry defect cannot hide inside it.
CONFIDENCE_TOLERANCE: Final = 6e-06


@dataclass(frozen=True, slots=True)
class ParityMismatch:
    """One binary parity failure, named by the field that diverged."""

    channel: str
    field: str
    index: int
    reference: object
    candidate: object

    def __str__(self) -> str:
        return (
            f"{self.channel}[{self.index}].{self.field}: "
            f"reference={self.reference!r} candidate={self.candidate!r}"
        )


def _count_mismatch(
    channel: str, reference: int, candidate: int
) -> tuple[ParityMismatch, ...]:
    if reference == candidate:
        return ()
    return (
        ParityMismatch(
            channel=channel,
            field="count",
            index=-1,
            reference=reference,
            candidate=candidate,
        ),
    )


def _compare_box(
    channel: str,
    index: int,
    reference: tuple[int, int, int, int, float],
    candidate: tuple[int, int, int, int, float],
) -> tuple[ParityMismatch, ...]:
    mismatches: list[ParityMismatch] = []
    for field, position in (("x1", 0), ("y1", 1), ("x2", 2), ("y2", 3)):
        if reference[position] != candidate[position]:
            mismatches.append(
                ParityMismatch(
                    channel=channel,
                    field=field,
                    index=index,
                    reference=reference[position],
                    candidate=candidate[position],
                )
            )
    if abs(float(reference[4]) - float(candidate[4])) > CONFIDENCE_TOLERANCE:
        mismatches.append(
            ParityMismatch(
                channel=channel,
                field="confidence",
                index=index,
                reference=reference[4],
                candidate=candidate[4],
            )
        )
    return tuple(mismatches)


def compare_pose(reference: ParsedPose, candidate: ParsedPose) -> tuple[ParityMismatch, ...]:
    """Compare box count/order/geometry and every COCO-17 keypoint."""
    mismatches = list(_count_mismatch("pose_box", len(reference.boxes), len(candidate.boxes)))
    mismatches.extend(_count_mismatch("pose", len(reference.poses), len(candidate.poses)))
    if mismatches:
        return tuple(mismatches)
    for index, (expected, actual) in enumerate(zip(reference.boxes, candidate.boxes, strict=True)):
        mismatches.extend(_compare_box("pose_box", index, expected, actual))
    for index, (expected_pose, actual_pose) in enumerate(
        zip(reference.poses, candidate.poses, strict=True)
    ):
        if len(expected_pose) != len(actual_pose):
            mismatches.append(
                ParityMismatch(
                    channel="pose",
                    field="keypoint_count",
                    index=index,
                    reference=len(expected_pose),
                    candidate=len(actual_pose),
                )
            )
            continue
        for point_index, (expected, actual) in enumerate(
            zip(expected_pose, actual_pose, strict=True)
        ):
            if (expected.x, expected.y) != (actual.x, actual.y):
                mismatches.append(
                    ParityMismatch(
                        channel="pose",
                        field=f"keypoint[{point_index}].xy",
                        index=index,
                        reference=(expected.x, expected.y),
                        candidate=(actual.x, actual.y),
                    )
                )
            if abs(expected.score - actual.score) > CONFIDENCE_TOLERANCE:
                mismatches.append(
                    ParityMismatch(
                        channel="pose",
                        field=f"keypoint[{point_index}].score",
                        index=index,
                        reference=expected.score,
                        candidate=actual.score,
                    )
                )
    return tuple(mismatches)


def compare_person(
    reference: ParsedPerson, candidate: ParsedPerson
) -> tuple[ParityMismatch, ...]:
    mismatches = list(_count_mismatch("person", len(reference.boxes), len(candidate.boxes)))
    if mismatches:
        return tuple(mismatches)
    for index, (expected, actual) in enumerate(zip(reference.boxes, candidate.boxes, strict=True)):
        mismatches.extend(_compare_box("person", index, expected, actual))
    return tuple(mismatches)


def compare_bed(reference: ParsedBed, candidate: ParsedBed) -> tuple[ParityMismatch, ...]:
    mismatches = list(_count_mismatch("bed", len(reference.regions), len(candidate.regions)))
    if mismatches:
        return tuple(mismatches)
    for index, (expected, actual) in enumerate(
        zip(reference.regions, candidate.regions, strict=True)
    ):
        mismatches.extend(
            _compare_box(
                "bed",
                index,
                (expected.x1, expected.y1, expected.x2, expected.y2, expected.confidence),
                (actual.x1, actual.y1, actual.x2, actual.y2, actual.confidence),
            )
        )
        if expected.polygon != actual.polygon:
            mismatches.append(
                ParityMismatch(
                    channel="bed",
                    field="polygon",
                    index=index,
                    reference=expected.polygon,
                    candidate=actual.polygon,
                )
            )
    return tuple(mismatches)


__all__ = [
    "CONFIDENCE_TOLERANCE",
    "ParityMismatch",
    "compare_bed",
    "compare_person",
    "compare_pose",
]

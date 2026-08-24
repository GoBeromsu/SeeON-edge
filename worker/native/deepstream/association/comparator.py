"""Binary differential comparator: native strategy vs the Python tracker oracle.

Parity is exact, not approximate. Per-frame `track_ids`, `live_ids`, and (when
an `AssociationResult`-shaped trace is compared) `selected_cue_indexes` and
`identity` must match byte/value; the first divergence names the frame index
and field so a reviewer can tell an ordering bug from an eviction bug from a
tie-break bug. `compare_traces` returns an empty tuple only when every frame
agreed.
"""

from __future__ import annotations

from dataclasses import dataclass

from worker.types.perception_frame import PerceptionFrameIdentity

#: Every value type a comparator field can hold. Kept a closed union (never
#: `object`/`Any`) so a new comparable field is a deliberate addition here,
#: not a silent escape hatch.
MismatchValue = int | tuple[int, ...] | frozenset[int] | PerceptionFrameIdentity | None


@dataclass(frozen=True, slots=True)
class AssociationParityMismatch:
    """One binary parity failure, named by the frame and field that diverged."""

    frame_index: int
    field: str
    reference: MismatchValue
    candidate: MismatchValue

    def __str__(self) -> str:
        return (
            f"frame[{self.frame_index}].{self.field}: "
            f"reference={self.reference!r} candidate={self.candidate!r}"
        )


@dataclass(frozen=True, slots=True)
class AssociationFrameTrace:
    """One frame's observable association outcome, from either implementation.

    `selected_cue_indexes` and `identity` default to the oracle-shaped values
    (`()`/`None`) so pre-existing lifecycle/eviction traces that only ever
    populated `track_ids`/`live_ids` keep constructing unchanged; a seam that
    compares the full `AssociationResult` shape supplies both.
    """

    track_ids: tuple[int, ...]
    live_ids: frozenset[int]
    selected_cue_indexes: tuple[int, ...] = ()
    identity: PerceptionFrameIdentity | None = None


def compare_traces(
    reference: tuple[AssociationFrameTrace, ...],
    candidate: tuple[AssociationFrameTrace, ...],
) -> tuple[AssociationParityMismatch, ...]:
    """Compare two full per-frame traces, one frame at a time, in order."""
    if len(reference) != len(candidate):
        return (
            AssociationParityMismatch(
                frame_index=-1,
                field="frame_count",
                reference=len(reference),
                candidate=len(candidate),
            ),
        )
    mismatches: list[AssociationParityMismatch] = []
    for frame_index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        if expected.track_ids != actual.track_ids:
            mismatches.append(
                AssociationParityMismatch(
                    frame_index=frame_index,
                    field="track_ids",
                    reference=expected.track_ids,
                    candidate=actual.track_ids,
                )
            )
        if expected.live_ids != actual.live_ids:
            mismatches.append(
                AssociationParityMismatch(
                    frame_index=frame_index,
                    field="live_ids",
                    reference=expected.live_ids,
                    candidate=actual.live_ids,
                )
            )
        if expected.selected_cue_indexes != actual.selected_cue_indexes:
            mismatches.append(
                AssociationParityMismatch(
                    frame_index=frame_index,
                    field="selected_cue_indexes",
                    reference=expected.selected_cue_indexes,
                    candidate=actual.selected_cue_indexes,
                )
            )
        if expected.identity != actual.identity:
            mismatches.append(
                AssociationParityMismatch(
                    frame_index=frame_index,
                    field="identity",
                    reference=expected.identity,
                    candidate=actual.identity,
                )
            )
    return tuple(mismatches)


__all__ = ["AssociationFrameTrace", "AssociationParityMismatch", "MismatchValue", "compare_traces"]

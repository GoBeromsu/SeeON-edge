"""Pure deterministic PTS resampling for replay and perception inputs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

CADENCE_NS = 66_666_667
DEFAULT_MAX_GAP_ROWS = 900


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResampledRow(Generic[T]):
    """A selected source row or a synthetic invalid cadence row."""

    pts_ns: int
    value: T | None
    valid: int


@dataclass(frozen=True, slots=True)
class PtsGapTooLargeError(ValueError):
    gap_rows: int
    max_gap_rows: int

    def __str__(self) -> str:
        return f"PTS gap requires {self.gap_rows} rows; limit is {self.max_gap_rows}"


def resample_pts(
    rows: Iterable[tuple[int, T]],
    *,
    cadence_ns: int = CADENCE_NS,
    max_gap_rows: int = DEFAULT_MAX_GAP_ROWS,
) -> Iterator[ResampledRow[T]]:
    """Select the first row in each cadence bucket and fill missing buckets.

    The first PTS establishes the epoch-local cadence. Non-monotonic and
    duplicate rows are deterministically dropped; callers reset this function
    at a stream-epoch boundary.
    """
    if cadence_ns <= 0 or max_gap_rows < 0:
        raise ValueError("cadence_ns must be positive and max_gap_rows non-negative")
    origin: int | None = None
    next_slot: int | None = None
    last_pts: int | None = None
    for pts_ns, value in rows:
        if not isinstance(pts_ns, int) or isinstance(pts_ns, bool):
            raise TypeError("pts_ns must be an integer")
        if last_pts is not None and pts_ns <= last_pts:
            continue
        last_pts = pts_ns
        if origin is None:
            origin = pts_ns
            next_slot = pts_ns
        assert next_slot is not None and origin is not None
        slot = origin + ((pts_ns - origin) // cadence_ns) * cadence_ns
        if slot < next_slot:
            continue
        gap_rows = (slot - next_slot) // cadence_ns
        if gap_rows > max_gap_rows:
            raise PtsGapTooLargeError(gap_rows, max_gap_rows)
        while next_slot < slot:
            yield ResampledRow(pts_ns=next_slot, value=None, valid=0)
            next_slot += cadence_ns
        yield ResampledRow(pts_ns=slot, value=value, valid=1)
        next_slot = slot + cadence_ns


__all__ = [
    "CADENCE_NS",
    "DEFAULT_MAX_GAP_ROWS",
    "PtsGapTooLargeError",
    "ResampledRow",
    "resample_pts",
]

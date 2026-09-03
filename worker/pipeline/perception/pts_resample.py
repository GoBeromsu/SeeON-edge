"""Pure deterministic PTS resampling for replay and perception inputs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

CADENCE_NS = 66_666_667


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResampledRow(Generic[T]):
    """A selected source row or a synthetic invalid cadence row."""

    pts_ns: int
    value: T | None
    valid: int


def resample_pts(
    rows: Iterable[tuple[int, T]], *, cadence_ns: int = CADENCE_NS
) -> Iterator[ResampledRow[T]]:
    """Select the first row in each cadence bucket and fill missing buckets.

    The first PTS establishes the epoch-local cadence. Non-monotonic and
    duplicate rows are deterministically dropped; callers reset this function
    at a stream-epoch boundary.
    """
    if cadence_ns <= 0:
        raise ValueError("cadence_ns must be positive")
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
        while next_slot < slot:
            yield ResampledRow(pts_ns=next_slot, value=None, valid=0)
            next_slot += cadence_ns
        yield ResampledRow(pts_ns=slot, value=value, valid=1)
        next_slot = slot + cadence_ns


__all__ = ["CADENCE_NS", "ResampledRow", "resample_pts"]

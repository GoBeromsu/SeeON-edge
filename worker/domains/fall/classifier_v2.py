"""Dark, structural three-class fall-window classification.

This module deliberately has no runtime or registry dependency.  A camera owns one
classifier instance; model instances may be shared by the composition root later.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from worker.types import FallModelInput

_WINDOW_FRAMES = 30
_STRIDE_FRAMES = 5
_ROW_WIDTH = 56
_TRACK_TTL_FRAMES = 45
_ZERO_ROW = (0.0,) * _ROW_WIDTH


@dataclass(frozen=True, slots=True)
class FallV2Probabilities:
    """The ordered model output: background, transition, and fallen."""

    background: float
    fall_transition: float
    fallen: float

    def __post_init__(self) -> None:
        for value in (self.background, self.fall_transition, self.fallen):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("fall v2 probabilities must be finite values in [0, 1]")


@runtime_checkable
class FallV2ModelProtocol(Protocol):
    """The deliberately narrow seam between the pure domain and a GRU adapter."""

    def predict(self, features: FallModelInput) -> FallV2Probabilities: ...


@dataclass(slots=True)
class FallWindowClassifierV2:
    """Maintain independent `[30, 56]` windows for the tracks of one camera."""

    model: FallV2ModelProtocol
    _buffers: dict[int, deque[tuple[float, ...]]] = field(default_factory=dict, init=False)
    _last_rows: dict[int, tuple[float, ...]] = field(default_factory=dict, init=False)
    _last_probabilities: dict[int, FallV2Probabilities] = field(default_factory=dict, init=False)
    _last_seen_frames: dict[int, int] = field(default_factory=dict, init=False)
    _generations: dict[int, int] = field(default_factory=dict, init=False)
    _next_generations: dict[int, int] = field(default_factory=dict, init=False)
    _reconnect_ids: set[int] = field(default_factory=set, init=False)
    _frame_counter: int = field(default=0, init=False)

    def update(
        self,
        rows_by_track: Mapping[int, Sequence[float] | None],
        live_track_ids: Iterable[int],
    ) -> Mapping[int, FallV2Probabilities]:
        """Append one row per live track and return predictions due this tick.

        A missing row coasts by repeating that track's previous valid row. A
        temporarily absent track also coasts through the shared 45-frame TTL:
        it retains its window but is not returned as a live prediction. An
        unknown or malformed row cannot become model input and is treated like a
        missing row. After exact TTL expiry all classifier state is evicted. A
        reused numeric id then immediately receives an inferable window of 29
        zero rows plus its current row; ordinary first startup remains warming.
        """
        self._frame_counter += 1
        live_ids = frozenset(live_track_ids)
        for track_id in live_ids:
            row = _valid_row(rows_by_track.get(track_id))
            if row is not None:
                self._last_rows[track_id] = row
            else:
                row = self._last_rows.get(track_id, _ZERO_ROW)
            self._buffer_for(track_id).append(row)
            self._last_seen_frames[track_id] = self._frame_counter

        for track_id in tuple(self._buffers):
            if track_id in live_ids:
                continue
            if self._frame_counter - self._last_seen_frames[track_id] >= _TRACK_TTL_FRAMES:
                self._evict(track_id)
                continue
            self._buffer_for(track_id).append(self._last_rows.get(track_id, _ZERO_ROW))

        if self._frame_counter % _STRIDE_FRAMES:
            return {}

        due: dict[int, FallV2Probabilities] = {}
        for track_id in sorted(live_ids):
            buffer = self._buffers.get(track_id)
            if buffer is None or len(buffer) != _WINDOW_FRAMES:
                continue
            prediction = self.model.predict(tuple(buffer))
            if not isinstance(prediction, FallV2Probabilities):
                try:
                    prediction = FallV2Probabilities(*prediction)  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "fall v2 model must return three finite probabilities"
                    ) from exc
            self._last_probabilities[track_id] = prediction
            due[track_id] = prediction
        return due

    def probabilities_for(self, track_id: int) -> FallV2Probabilities | None:
        return self._last_probabilities.get(track_id)

    def generation_for(self, track_id: int) -> int | None:
        return self._generations.get(track_id)

    def _buffer_for(self, track_id: int) -> deque[tuple[float, ...]]:
        buffer = self._buffers.get(track_id)
        if buffer is None:
            buffer = deque(maxlen=_WINDOW_FRAMES)
            if track_id in self._reconnect_ids:
                buffer.extend((_ZERO_ROW,) * (_WINDOW_FRAMES - 1))
                self._reconnect_ids.remove(track_id)
            generation = self._next_generations.get(track_id, 0)
            self._next_generations[track_id] = generation + 1
            self._generations[track_id] = generation
            self._buffers[track_id] = buffer
        return buffer

    def _evict(self, track_id: int) -> None:
        del self._buffers[track_id]
        self._last_rows.pop(track_id, None)
        self._last_probabilities.pop(track_id, None)
        del self._last_seen_frames[track_id]
        del self._generations[track_id]
        self._reconnect_ids.add(track_id)


def _valid_row(value: Sequence[float] | None) -> tuple[float, ...] | None:
    if value is None or len(value) != _ROW_WIDTH:
        return None
    row = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in row):
        return None
    return row


__all__ = ["FallV2ModelProtocol", "FallV2Probabilities", "FallWindowClassifierV2"]

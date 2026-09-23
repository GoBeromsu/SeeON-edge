"""Dark, structural three-class fall-window classification.

This module deliberately has no runtime or registry dependency.  A camera owns one
classifier instance; model instances may be shared by the composition root later.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from worker.interfaces.fall_model import FallModelProtocol, FallProbabilities
from worker.types.trace import DecisionTraceMissingReason

FALL_WINDOW_FRAMES = 30
FALL_STRIDE_FRAMES = 5
_ROW_WIDTH = 56
_TRACK_TTL_FRAMES = 45


@dataclass(slots=True)
class FallWindowClassifier:
    """Maintain independent `[30, 56]` windows for the tracks of one camera."""

    model: FallModelProtocol
    _buffers: dict[int, deque[tuple[float, ...]]] = field(default_factory=dict, init=False)
    _last_rows: dict[int, tuple[float, ...]] = field(default_factory=dict, init=False)
    _last_probabilities: dict[int, FallProbabilities] = field(default_factory=dict, init=False)
    _last_seen_frames: dict[int, int] = field(default_factory=dict, init=False)
    _generations: dict[int, int] = field(default_factory=dict, init=False)
    _next_generations: dict[int, int] = field(default_factory=dict, init=False)
    _reconnect_ids: set[int] = field(default_factory=set, init=False)
    _current_call_missing_score_reasons: dict[int, DecisionTraceMissingReason] = field(
        default_factory=dict, init=False
    )
    _frame_counter: int = field(default=0, init=False)
    #: Adoptions by _adopt_lineage; counted next to the episode authority's
    #: track_id_switch_absorbed_total, which this mirrors at the window level.
    buffer_lineage_adopted_total: int = field(default=0, init=False)

    def update(
        self,
        rows_by_track: Mapping[int, Sequence[float] | None],
        live_track_ids: Iterable[int],
    ) -> Mapping[int, FallProbabilities]:
        """Append one row per live track and return predictions due this tick.

        A missing row coasts by repeating that track's previous valid row; a
        track with no valid row yet appends nothing this tick and stays
        warming rather than accepting a synthetic placeholder. A temporarily
        absent track also coasts through the shared 45-frame TTL: it retains
        its window but is not returned as a live prediction. An unknown,
        malformed, or all-zero row cannot become model input and is treated
        like a missing row. After exact TTL expiry all classifier state is
        evicted; a reused numeric id then rebuilds its window by replicating
        its first valid row (never zeros) until real frames replace it. A
        brand-new id that appears while a different track is still within its
        TTL window instead inherits that track's buffer, last row, and
        generation outright -- the same recency criterion the episode
        authority uses to absorb a tracker id switch -- so the window
        continues rather than resetting.
        ``current_call_missing_score_reasons`` describes only this invocation:
        non-due live tracks are stride-skipped, while due tracks without a full
        window are warming.
        """
        self._current_call_missing_score_reasons = {}
        self._frame_counter += 1
        live_ids = frozenset(live_track_ids)
        # Sorted so a tick with more than one new id and more than one
        # vanished donor adopts deterministically, lowest id first -- the
        # same tie-break the episode authority uses for reassociation.
        for track_id in sorted(live_ids):
            valid = _valid_row(rows_by_track.get(track_id))
            if valid is not None:
                self._last_rows[track_id] = valid
            row = self._last_rows.get(track_id)
            self._last_seen_frames[track_id] = self._frame_counter
            if row is None:
                continue
            if track_id not in self._buffers:
                self._adopt_lineage(track_id, live_ids)
            self._buffer_for(track_id, row).append(row)

        for track_id in tuple(self._buffers):
            if track_id in live_ids:
                continue
            if self._frame_counter - self._last_seen_frames[track_id] >= _TRACK_TTL_FRAMES:
                self._evict(track_id)
                continue
            self._buffers[track_id].append(self._last_rows[track_id])

        if self._frame_counter % FALL_STRIDE_FRAMES:
            self._current_call_missing_score_reasons = dict.fromkeys(
                live_ids, DecisionTraceMissingReason.CLASSIFIER_STRIDE_NOT_DUE
            )
            return {}

        due: dict[int, FallProbabilities] = {}
        for track_id in sorted(live_ids):
            buffer = self._buffers.get(track_id)
            if buffer is None or len(buffer) != FALL_WINDOW_FRAMES:
                self._current_call_missing_score_reasons[track_id] = (
                    DecisionTraceMissingReason.CLASSIFIER_WARMUP
                )
                continue
            prediction = self.model.predict(tuple(buffer))
            if not isinstance(prediction, FallProbabilities):
                try:
                    prediction = FallProbabilities(*prediction)  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    raise ValueError("fall model must return three finite probabilities") from exc
            self._last_probabilities[track_id] = prediction
            due[track_id] = prediction
        return due

    @property
    def current_call_missing_score_reasons(
        self,
    ) -> Mapping[int, DecisionTraceMissingReason]:
        """Explain missing scores from the immediately preceding ``update`` only."""
        return MappingProxyType(self._current_call_missing_score_reasons)

    def probabilities_for(self, track_id: int) -> FallProbabilities | None:
        return self._last_probabilities.get(track_id)

    def generation_for(self, track_id: int) -> int | None:
        return self._generations.get(track_id)

    def _buffer_for(self, track_id: int, row: tuple[float, ...]) -> deque[tuple[float, ...]]:
        buffer = self._buffers.get(track_id)
        if buffer is None:
            buffer = deque(maxlen=FALL_WINDOW_FRAMES)
            if track_id in self._reconnect_ids:
                # Replicate this track's own first valid row rather than
                # zero-filling: a repeated real pose is in-distribution, a
                # teleport-to-(0,0) pose is not.
                buffer.extend((row,) * (FALL_WINDOW_FRAMES - 1))
                self._reconnect_ids.remove(track_id)
            generation = self._next_generations.get(track_id, 0)
            self._next_generations[track_id] = generation + 1
            self._generations[track_id] = generation
            self._buffers[track_id] = buffer
        return buffer

    def _adopt_lineage(self, track_id: int, live_ids: frozenset[int]) -> None:
        """Move a still-live-within-TTL vanished track's window onto ``track_id``.

        Reuses the episode authority's own reassociation criterion (recency
        within the TTL window, lowest id first) instead of inventing a new
        one. Like that authority, this does not check spatial proximity, so a
        genuinely different person taking over the id right as the old one
        vanishes would wrongly inherit its window -- the same accepted
        trade-off the authority already makes for the identical reason.
        """
        candidates = sorted(
            candidate
            for candidate in self._buffers
            if candidate not in live_ids
            and self._frame_counter - self._last_seen_frames[candidate] < _TRACK_TTL_FRAMES
        )
        if not candidates:
            return
        donor = candidates[0]
        self._buffers[track_id] = self._buffers.pop(donor)
        self._last_rows[track_id] = self._last_rows.pop(donor)
        self._generations[track_id] = self._generations.pop(donor)
        self._last_seen_frames.pop(donor, None)
        self._next_generations.pop(donor, None)
        self._reconnect_ids.discard(donor)
        self._reconnect_ids.discard(track_id)
        self.buffer_lineage_adopted_total += 1

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
    if all(component == 0.0 for component in row):
        # The all-zero row is the domain's own "no detection" sentinel
        # (worker.domains.fall.pose_bbox56._zero_row). It must never be
        # scored as a real pose from any caller.
        return None
    return row


__all__ = [
    "FALL_STRIDE_FRAMES",
    "FALL_WINDOW_FRAMES",
    "FallProbabilities",
    "FallWindowClassifier",
]

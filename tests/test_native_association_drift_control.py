"""Adversarial: intentional strategy drift must produce a real, catchable divergence.

Every case here mutates a REAL `AssociationResult` the active strategy
returned (or wraps the real strategy via composition) and proves
`compare_traces`/`association_failure` -- the actual C1/comparator gates --
reject the drift. None of these are hand-built proxy shapes disconnected from
what the strategy emits.
"""

from __future__ import annotations

from contracts.observation import BoundingBox
from worker.native.deepstream.association import (
    LEGACY_GREEDY_BBOX_IOU_V1,
    AssociationFrameTrace,
    build_active_association_strategy,
    compare_traces,
)
from worker.native.deepstream.association.legacy_greedy_iou import LegacyGreedyBboxIouStrategy
from worker.pipeline.perception.tracker import GreedyIouTracker
from worker.types.perception_frame import (
    AssociationResult,
    ChannelState,
    PerceptionFrameIdentity,
    PersonBox,
    PersonBoxChannel,
    association_failure,
)

_IDENTITY = PerceptionFrameIdentity(
    worker_boot_id="boot-1", camera_id="camera-1", stream_epoch=0, seq=0
)


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def _person_box_channel(boxes: tuple[BoundingBox, ...]) -> PersonBoxChannel:
    state = ChannelState.INFERRED if boxes else ChannelState.INFERRED_EMPTY
    cues = tuple(
        PersonBox(x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, confidence=box.confidence)
        for box in boxes
    )
    return PersonBoxChannel(state=state, boxes=cues)


def test_comparator_catches_intentional_track_id_drift() -> None:
    reference = (AssociationFrameTrace(track_ids=(0,), live_ids=frozenset({0})),)
    drifted = (AssociationFrameTrace(track_ids=(1,), live_ids=frozenset({1})),)

    mismatches = compare_traces(reference, drifted)

    assert len(mismatches) == 2
    fields = {mismatch.field for mismatch in mismatches}
    assert fields == {"track_ids", "live_ids"}


def test_comparator_catches_intentional_equal_iou_tie_break_drift() -> None:
    """Mutating the tie-break rule (favor the higher index) must fail parity."""
    left = _box(0, 0, 100, 100)
    right = _box(50, 0, 150, 100)
    center = _box(25, 0, 125, 100)

    oracle = GreedyIouTracker()
    reference = (
        AssociationFrameTrace(
            track_ids=oracle.observe((left, right)),
            live_ids=oracle.live_ids,
            selected_cue_indexes=(0, 1),
            identity=_IDENTITY,
        ),
        AssociationFrameTrace(
            track_ids=oracle.observe((center,)),
            live_ids=oracle.live_ids,
            selected_cue_indexes=(0,),
            identity=_IDENTITY,
        ),
    )

    class _DriftedStrategy:
        """Wraps the real strategy but flips the tie-break on a single match.

        Composition, not subclassing: `LegacyGreedyBboxIouStrategy` is
        `@final` (the shipped candidate must not grow subclass seams).
        """

        def __init__(self) -> None:
            self._inner = LegacyGreedyBboxIouStrategy()
            self.identity = self._inner.identity

        @property
        def live_ids(self) -> frozenset[int]:
            return self._inner.live_ids

        def observe(
            self, identity: PerceptionFrameIdentity, person_box: PersonBoxChannel
        ) -> AssociationResult:
            result = self._inner.observe(identity, person_box)
            if len(person_box.boxes) == 1 and len(result.track_ids) == 1:
                # Intentional mutation: report the wrong (higher) existing
                # track index on a tie, instead of the shipped lower-index win.
                return AssociationResult(
                    strategy=result.strategy,
                    track_ids=(result.track_ids[0] + 1,),
                    selected_cue_indexes=result.selected_cue_indexes,
                    identity=result.identity,
                )
            return result

        def coast(self) -> None:
            self._inner.coast()

        def reset(self) -> None:
            self._inner.reset()

    drifted_strategy = _DriftedStrategy()
    candidate: list[AssociationFrameTrace] = []
    for boxes in ((left, right), (center,)):
        result = drifted_strategy.observe(_IDENTITY, _person_box_channel(boxes))
        candidate.append(
            AssociationFrameTrace(
                track_ids=result.track_ids,
                live_ids=drifted_strategy.live_ids,
                selected_cue_indexes=result.selected_cue_indexes,
                identity=result.identity,
            )
        )

    mismatches = compare_traces(reference, tuple(candidate))
    assert mismatches != ()
    assert any(mismatch.field == "track_ids" for mismatch in mismatches)


def test_comparator_catches_missing_reconnect_reset() -> None:
    """Intentional drift: a strategy that forgets to reset on reconnect.

    Two stale pre-reconnect tracks exist; the second sits where the first
    post-reconnect box reappears. A properly reset strategy clears the track
    list, so `reset()` followed by one observation mints a fresh id 0. A
    strategy that forgets to reset instead greedy-matches the second stale
    track and reports its old id 1 -- the mismatch the comparator must catch.
    """
    stale_first = _box(0, 0, 20, 20)
    stale_second = _box(200, 200, 220, 220)
    reappearing = _box(202, 202, 222, 222)

    reference_strategy = build_active_association_strategy()
    _ = reference_strategy.observe(_IDENTITY, _person_box_channel((stale_first, stale_second)))
    reference_strategy.reset()
    reference_result = reference_strategy.observe(_IDENTITY, _person_box_channel((reappearing,)))
    reference = (
        AssociationFrameTrace(
            track_ids=reference_result.track_ids, live_ids=reference_strategy.live_ids
        ),
    )

    never_reset_strategy = build_active_association_strategy()
    _ = never_reset_strategy.observe(_IDENTITY, _person_box_channel((stale_first, stale_second)))
    # Intentional drift: skip reset() entirely across the reconnect boundary.
    drifted_result = never_reset_strategy.observe(_IDENTITY, _person_box_channel((reappearing,)))
    candidate = (
        AssociationFrameTrace(
            track_ids=drifted_result.track_ids, live_ids=never_reset_strategy.live_ids
        ),
    )

    mismatches = compare_traces(reference, candidate)
    assert mismatches != ()
    assert any(mismatch.field == "track_ids" for mismatch in mismatches)


def test_comparator_catches_malformed_cue_index_drift() -> None:
    """Intentional drift: track_ids/selected_cue_indexes length mismatch.

    Built from a REAL `AssociationResult` the active strategy returned, then
    mutated to drop one `selected_cue_indexes` entry -- not a hand-built
    proxy shape unrelated to what the strategy actually emits.
    """
    strategy = build_active_association_strategy()
    person_box = _person_box_channel((_box(0, 0, 40, 40), _box(100, 100, 140, 140)))
    good = strategy.observe(_IDENTITY, person_box)
    assert good.track_ids == (0, 1)
    assert association_failure(_IDENTITY, good, person_box_count=2) is None

    drifted = AssociationResult(
        strategy=good.strategy,
        track_ids=good.track_ids,
        selected_cue_indexes=(good.selected_cue_indexes[0],),
        identity=good.identity,
    )
    failure = association_failure(_IDENTITY, drifted, person_box_count=2)
    assert failure is not None
    assert failure.code == "invalid_cue_index"


def test_bed_identity_cue_source_is_rejected_by_the_real_association_failure() -> None:
    """A result claiming `cue_source="bed_region"` is rejected regardless of
    shape correctness -- the C1 gate, not this package, owns that refusal.
    """
    strategy = build_active_association_strategy()
    person_box = _person_box_channel((_box(0, 0, 40, 40),))
    result = strategy.observe(_IDENTITY, person_box)
    assert result.strategy == LEGACY_GREEDY_BBOX_IOU_V1

    bed_tagged = AssociationResult(
        strategy=result.strategy,
        track_ids=result.track_ids,
        selected_cue_indexes=result.selected_cue_indexes,
        identity=result.identity,
        cue_source="bed_region",
    )
    failure = association_failure(_IDENTITY, bed_tagged, person_box_count=1)
    assert failure is not None
    assert failure.code == "bed_identity_cue"

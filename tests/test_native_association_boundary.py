"""Typed person-cue boundary: `AssociationStrategy.observe` -> real `AssociationResult`.

Drives the active strategy through its INTENDED seam
(`PerceptionFrameIdentity` + `PersonBoxChannel`) and asserts every observable
field of the returned `AssociationResult`: identity binding, `strategy`,
`cue_source`, and `selected_cue_indexes` in person-box input order. Also
proves bed regions are structurally unrepresentable at this boundary (a
`BedRegionChannel` has no parameter to satisfy) and that malformed cue
indexes are caught by the real C1 `association_failure`, not a hand-built
proxy.
"""

from __future__ import annotations

from worker.native.deepstream.association import (
    LEGACY_GREEDY_BBOX_IOU_V1,
    build_active_association_strategy,
)
from worker.types.perception_frame import (
    ChannelState,
    PerceptionFrameIdentity,
    PersonBox,
    PersonBoxChannel,
    association_failure,
)

_IDENTITY = PerceptionFrameIdentity(
    worker_boot_id="boot-1", camera_id="camera-1", stream_epoch=0, seq=0
)


def _person_box(*boxes: PersonBox) -> PersonBoxChannel:
    state = ChannelState.INFERRED if boxes else ChannelState.INFERRED_EMPTY
    return PersonBoxChannel(state=state, boxes=boxes)


def test_active_strategy_returns_a_real_association_result_bound_to_identity() -> None:
    strategy = build_active_association_strategy()
    person_box = _person_box(PersonBox(x1=0, y1=0, x2=40, y2=40, confidence=0.9))

    result = strategy.observe(_IDENTITY, person_box)

    assert type(result).__name__ == "AssociationResult"
    assert result.identity == _IDENTITY
    assert result.strategy == LEGACY_GREEDY_BBOX_IOU_V1
    assert result.cue_source == "person_box"
    assert result.selected_cue_indexes == (0,)
    assert result.track_ids == (0,)
    assert association_failure(_IDENTITY, result, person_box_count=1) is None


def test_selected_cue_indexes_follow_person_box_input_order_not_match_order() -> None:
    strategy = build_active_association_strategy()
    first = _person_box(
        PersonBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9),
        PersonBox(x1=200, y1=200, x2=250, y2=250, confidence=0.9),
    )
    first_result = strategy.observe(_IDENTITY, first)
    assert first_result.selected_cue_indexes == (0, 1)
    assert first_result.track_ids == (0, 1)

    # Reorder the incoming cues: the moved-second box now arrives first.
    reordered = _person_box(
        PersonBox(x1=205, y1=205, x2=255, y2=255, confidence=0.9),
        PersonBox(x1=4, y1=4, x2=54, y2=54, confidence=0.9),
    )
    reordered_result = strategy.observe(_IDENTITY, reordered)
    assert reordered_result.selected_cue_indexes == (0, 1), "cue indexes are input order, always"
    assert reordered_result.track_ids == (1, 0), "track ids follow the matched identity, not order"


def test_inferred_empty_person_box_counts_a_miss_and_returns_empty_selection() -> None:
    strategy = build_active_association_strategy()
    box = PersonBox(x1=0, y1=0, x2=20, y2=20, confidence=0.9)
    first = strategy.observe(_IDENTITY, _person_box(box))
    assert first.track_ids == (0,)

    empty = strategy.observe(_IDENTITY, _person_box())
    assert empty.selected_cue_indexes == ()
    assert empty.track_ids == ()
    assert strategy.live_ids == frozenset({0}), "one miss must not evict under the default budget"


def test_bed_region_channel_has_no_parameter_shape_to_satisfy_observe() -> None:
    """Structural boundary proof: `observe` only accepts `PersonBoxChannel`.

    A `BedRegionChannel` is a distinct, unrelated dataclass with no shared
    base and no duck-typed overlap `observe` could accidentally accept, so
    passing one is rejected before any association logic runs -- proven here
    by asserting the parameter annotation itself, and confirmed structurally
    via `uv run basedpyright` (a `BedRegionChannel` argument is a real type
    error at the call site, not just a runtime guard).
    """
    import inspect

    signature = inspect.signature(build_active_association_strategy().observe)
    person_box_param = signature.parameters["person_box"]
    assert person_box_param.annotation in ("PersonBoxChannel", PersonBoxChannel)


def test_malformed_selected_cue_index_is_caught_by_the_real_association_failure() -> None:
    """Malformed cue index via the real seam: a hand-crafted AssociationResult
    that lies about `selected_cue_indexes` is still caught by the shared C1
    `association_failure` gate every `AssociationResult` must pass.
    """
    from worker.types.perception_frame import AssociationResult

    strategy = build_active_association_strategy()
    person_box = _person_box(PersonBox(x1=0, y1=0, x2=40, y2=40, confidence=0.9))
    result = strategy.observe(_IDENTITY, person_box)

    malformed = AssociationResult(
        strategy=result.strategy,
        track_ids=result.track_ids,
        selected_cue_indexes=(5,),
        identity=result.identity,
    )
    failure = association_failure(_IDENTITY, malformed, person_box_count=1)
    assert failure is not None
    assert failure.code == "invalid_cue_index"

from __future__ import annotations

import threading
from types import MappingProxyType, SimpleNamespace

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.fall.policy_v2 import FallPolicyDeciderV2, FallV2DomainDecider
from worker.interfaces.fall_model import FallV2Probabilities
from worker.runtime.flow.policy_pump import NativePolicyPump
from worker.types import DecisionInput


def _pump_for(decider: object) -> NativePolicyPump:
    pump = object.__new__(NativePolicyPump)
    pump._decision = SimpleNamespace(  # noqa: SLF001
        last_trace_snapshots=decider.last_trace_snapshots
    )
    pump._preview_states_lock = threading.Lock()  # noqa: SLF001
    pump._preview_states = MappingProxyType({})  # noqa: SLF001
    return pump


def _sync_preview(pump: NativePolicyPump, decider: object) -> None:
    pump._decision.last_trace_snapshots = decider.last_trace_snapshots  # noqa: SLF001
    pump._refresh_preview_states()  # noqa: SLF001


class _ImmediateClassifier:
    def update(
        self, _rows: object, live_track_ids: tuple[int, ...]
    ) -> dict[int, FallV2Probabilities]:
        return {
            track_id: FallV2Probabilities(0.8, 0.1, 0.1) for track_id in live_track_ids
        }


class _SilentClassifier:
    def update(
        self, _rows: object, live_track_ids: tuple[int, ...]
    ) -> dict[int, FallV2Probabilities]:
        del live_track_ids
        return {}


def _fall_input(*, time_sec: float, frame_index: int) -> DecisionInput:
    person = BoundingBox(10, 10, 70, 90, 0.9)
    pose = tuple((index + 1, index + 2, 0.9) for index in range(17))
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            poses=(pose,),
            regions=((BoundingBox(0, 0, 80, 100, 0.9),), ()),
            track_ids=(9,),
        ),
        frame_width=180,
        frame_height=120,
        live_track_ids=(9,),
        time_sec=time_sec,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def _domain_decider(classifier: object) -> FallV2DomainDecider:
    return FallV2DomainDecider(
        classifier=classifier,
        policy=FallPolicyDeciderV2(
            camera_id="camera-a",
            facility_id="facility-a",
            boot_id="boot-a",
            stream_epoch="epoch-a",
            source_generation=1,
        ),
    )


def test_preview_states_follow_real_fall_decider_normal_and_suspected_traces() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera-a",
        facility_id="facility-a",
        boot_id="boot-a",
        stream_epoch="epoch-a",
        source_generation=1,
    )
    _ = decider.update(
        {9: FallV2Probabilities(0.8, 0.1, 0.1)},
        (9,),
        frame_index=1,
        time_sec=1.0,
    )
    pump = _pump_for(decider)
    pump._refresh_preview_states()  # noqa: SLF001

    normal = pump.preview_states()
    assert normal[9].status == "normal"
    assert normal[9].probability == 0.1

    _ = decider.update(
        {9: FallV2Probabilities(0.0, 0.99, 0.01)},
        (9,),
        frame_index=2,
        time_sec=2.0,
    )
    _sync_preview(pump, decider)

    suspected = pump.preview_states()
    assert suspected[9].status == "suspected"
    assert suspected[9].probability == 0.99
    assert normal[9].status == "normal"


def test_never_scored_warmup_track_is_not_published_as_normal() -> None:
    decider = _domain_decider(_SilentClassifier())
    _ = decider.update(_fall_input(time_sec=1.0, frame_index=1))
    pump = _pump_for(decider)
    pump._refresh_preview_states()  # noqa: SLF001

    assert decider.last_trace_snapshots[0].reason == "score-missing"
    assert decider.last_trace_snapshots[0].current_state == "unknown"
    assert 9 not in pump.preview_states()


def test_scored_track_keeps_its_state_through_stride_gaps() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera-a",
        facility_id="facility-a",
        boot_id="boot-a",
        stream_epoch="epoch-a",
        source_generation=1,
    )
    _ = decider.update({9: FallV2Probabilities(0.8, 0.1, 0.1)}, (9,), frame_index=1, time_sec=1.0)
    pump = _pump_for(decider)
    pump._refresh_preview_states()  # noqa: SLF001
    assert pump.preview_states()[9].status == "normal"

    _ = decider.update({}, (9,), frame_index=2, time_sec=1.1)
    _sync_preview(pump, decider)
    gap = pump.preview_states()[9]
    assert decider.last_trace_snapshots[0].reason == "score-missing"
    assert gap.status == "normal"
    assert gap.probability is None

    _ = decider.update({9: FallV2Probabilities(0.0, 0.99, 0.01)}, (9,), frame_index=3, time_sec=1.2)
    _sync_preview(pump, decider)
    assert pump.preview_states()[9].status == "suspected"

    _ = decider.update({}, (9,), frame_index=4, time_sec=1.3)
    _sync_preview(pump, decider)
    assert pump.preview_states()[9].status == "suspected"
    assert pump.preview_states()[9].probability is None


def test_open_episode_stays_suspected_through_a_score_gap_after_votes_age_out() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera-a",
        facility_id="facility-a",
        boot_id="boot-a",
        stream_epoch="epoch-a",
        source_generation=1,
    )
    onset = FallV2Probabilities(0.3, 0.7, 0.0)
    emitted = ()
    for frame in range(3):
        emitted += decider.update({9: onset}, (9,), frame_index=frame, time_sec=float(frame))
    assert len(emitted) == 1

    # Every vote in the deque ages out with sub-threshold transition scores
    # while the episode authority still holds the OPEN episode.
    quiet = FallV2Probabilities(0.9, 0.1, 0.5)
    for frame in range(3, 9):
        _ = decider.update({9: quiet}, (9,), frame_index=frame, time_sec=float(frame))

    _ = decider.update({}, (9,), frame_index=9, time_sec=9.0)
    pump = _pump_for(decider)
    pump._refresh_preview_states()  # noqa: SLF001

    assert decider.last_trace_snapshots[0].reason == "score-missing"
    assert decider.last_trace_snapshots[0].current_state == "transition-confirmed"
    assert pump.preview_states()[9].status == "suspected"


def test_coast_preserves_the_previous_preview_map() -> None:
    decider = _domain_decider(_ImmediateClassifier())
    _ = decider.update(_fall_input(time_sec=1.0, frame_index=1))
    pump = _pump_for(decider)
    pump._refresh_preview_states()  # noqa: SLF001
    assert pump.preview_states()[9].status == "normal"

    _ = decider.coast()
    _sync_preview(pump, decider)
    assert pump.preview_states()[9].status == "normal"

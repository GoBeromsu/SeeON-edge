from __future__ import annotations

import threading
from types import MappingProxyType, SimpleNamespace

from worker.domains.fall.policy_v2 import FallPolicyDeciderV2
from worker.interfaces.fall_model import FallV2Probabilities
from worker.runtime.flow.policy_pump import NativePolicyPump


def _pump_for(decider: FallPolicyDeciderV2) -> NativePolicyPump:
    pump = object.__new__(NativePolicyPump)
    pump._decision = SimpleNamespace(  # noqa: SLF001
        last_trace_snapshots=decider.last_trace_snapshots
    )
    pump._preview_states_lock = threading.Lock()  # noqa: SLF001
    pump._preview_states = MappingProxyType({})  # noqa: SLF001
    return pump


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
    pump._decision.last_trace_snapshots = decider.last_trace_snapshots  # noqa: SLF001
    pump._refresh_preview_states()  # noqa: SLF001

    suspected = pump.preview_states()
    assert suspected[9].status == "suspected"
    assert suspected[9].probability == 0.99
    assert normal[9].status == "normal"

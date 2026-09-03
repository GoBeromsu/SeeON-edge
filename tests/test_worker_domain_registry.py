from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.domains import DOMAIN_REGISTRY, BedExitDomainDependencies, enabled_domains, list_domains
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor
from worker.domains.fall import FallV2DomainDecider, FallV2Probabilities
from worker.interfaces.decision import Decider
from worker.types import DecisionInput


class _FallModel:
    def predict(self, _features: object) -> FallV2Probabilities:
        return FallV2Probabilities(0.1, 0.9, 0.0)


def _clock() -> datetime:
    return datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))


def _empty_input() -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=640,
        frame_height=360,
        live_track_ids=(),
        time_sec=2.5,
        frame_index=4,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_registry_contains_enabled_fall_v2_and_bed_exit() -> None:
    assert set(DOMAIN_REGISTRY) == {"fall", "bed_exit"}
    assert set(enabled_domains()) == {"fall", "bed_exit"}
    assert list_domains(enabled=False) == ()


def test_registry_wires_v2_fall_and_bed_exit_deciders() -> None:
    fall = DOMAIN_REGISTRY["fall"].factory(
        {
            "model": _FallModel(),
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "boot_id": "boot-1",
            "stream_epoch": "1",
            "source_generation": 0,
        }
    )
    bed_exit = DOMAIN_REGISTRY["bed_exit"].factory(
        BedExitDomainDependencies(
            config=BedExitConfig(camera_id="camera-1", facility_id="facility-1"), clock=_clock
        )
    )
    assert isinstance(fall, FallV2DomainDecider)
    assert isinstance(bed_exit, BedExitMonitor)
    assert isinstance(fall, Decider)
    assert isinstance(bed_exit, Decider)


def test_bed_exit_debug_adapter_returns_snapshot() -> None:
    detector = DOMAIN_REGISTRY["bed_exit"].factory(
        BedExitDomainDependencies(
            config=BedExitConfig(camera_id="camera-1", facility_id="facility-1"), clock=_clock
        )
    )
    _ = detector.update(_empty_input())
    adapter = DOMAIN_REGISTRY["bed_exit"].debug_snapshot_adapter
    assert adapter is not None
    assert adapter(detector, 9) is not None

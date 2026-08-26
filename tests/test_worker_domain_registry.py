from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    FrameObservation,
)
from worker.domains import (
    DOMAIN_REGISTRY,
    AuditContext,
    BedExitDomainDependencies,
    DomainAuditSnapshot,
    FallDomainDependencies,
    enabled_domains,
    list_domains,
)
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor
from worker.domains.fall import FallEventLatch
from worker.interfaces.decision import Decider
from worker.types import DecisionInput


@dataclass(frozen=True, slots=True)
class _Metadata:
    window: int = 1
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@dataclass(slots=True)  # policy: MUTABLE_OK - records model calls for assertions
class _FallModel:
    metadata: _Metadata = field(default_factory=_Metadata)
    operating_threshold: float = 0.5
    inputs: list[NDArray[np.float32]] = field(default_factory=list)

    def predict(self, features: NDArray[np.float32]) -> float:
        self.inputs.append(features.copy())
        return 0.91


def _clock() -> datetime:
    return datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))


def _empty_input(frame_index: int = 4) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=640,
        frame_height=360,
        live_track_ids=(),
        time_sec=2.5,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_registry_contains_only_enabled_fall_and_bed_exit_domains() -> None:
    # Given: the shared worker domain registry.
    expected = {"fall", "bed_exit"}

    # When: registrations are filtered by enabled state.
    all_domains = set(list_domains())

    # Then: no disabled legacy scaffold becomes executable.
    assert set(DOMAIN_REGISTRY) == expected
    assert all_domains == expected
    assert set(enabled_domains()) == expected
    assert list_domains(enabled=False) == ()


def test_registry_wires_real_camera_local_fall_and_bed_exit_deciders() -> None:
    # Given: concrete dependencies for each enabled domain.
    fall_model = _FallModel()
    bed_config = BedExitConfig(camera_id="camera-1", facility_id="facility-1")

    # When: the registrations build their deciders.
    fall = DOMAIN_REGISTRY["fall"].factory(
        FallDomainDependencies(
            model=fall_model,
            camera_id="camera-1",
            facility_id="facility-1",
        )
    )
    bed_exit = DOMAIN_REGISTRY["bed_exit"].factory(
        BedExitDomainDependencies(config=bed_config, clock=_clock)
    )

    # Then: both public domain APIs satisfy the frozen decision Protocol.
    assert isinstance(fall, FallEventLatch)
    assert isinstance(bed_exit, BedExitMonitor)
    assert isinstance(fall, Decider)
    assert isinstance(bed_exit, Decider)


def test_registry_exposes_event_views_and_immutable_audit_metadata() -> None:
    # Given: model audit context supplied by the future composition root.
    context = AuditContext(model_version="fall-v3", operating_threshold=0.73)

    # When: registry metadata is inspected without running a detector.
    fall = DOMAIN_REGISTRY["fall"]
    bed_exit = DOMAIN_REGISTRY["bed_exit"]
    audit_provider = fall.audit_metadata_provider

    # Then: event allowlists, input views, and audit data have one source of truth.
    assert fall.input_view == "fall_window"
    assert fall.event_types == frozenset({"fall"})
    assert bed_exit.input_view == "bed_regions"
    assert bed_exit.event_types == frozenset({"bed-exit"})
    assert audit_provider is not None
    assert audit_provider(context) == DomainAuditSnapshot(
        model_version="fall-v3",
        operating_threshold=0.73,
    )


def test_bed_exit_debug_adapter_returns_the_single_update_snapshot() -> None:
    # Given: a registry-built bed monitor that has processed one numeric input.
    factory = DOMAIN_REGISTRY["bed_exit"].factory
    detector = factory(
        BedExitDomainDependencies(
            config=BedExitConfig(camera_id="camera-1", facility_id="facility-1"),
            clock=_clock,
        )
    )
    _ = detector.update(_empty_input())
    adapter = DOMAIN_REGISTRY["bed_exit"].debug_snapshot_adapter

    # When: output asks for the already-computed frame snapshot.
    assert adapter is not None
    snapshot = adapter(detector, 9)

    # Then: the adapter does not invoke the detector a second time.
    assert snapshot is not None
    assert snapshot.frame_index == 9
    assert snapshot.bed_region == BedRegionDebugSnapshot(BedRegionCacheState.EMPTY)

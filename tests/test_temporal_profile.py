"""Characterization of the production fps/interval contract.

Written against the pre-TemporalProfile tree so the identity refactor has a
live pin: these assertions must keep passing, unmodified, after the contract
owner is introduced.
"""

from __future__ import annotations

from math import inf, nan

import pytest

from worker.pipeline.bus.scheduler import Scheduler
from worker.pipeline.ingest.lifecycle import CapturePolicy
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.runtime.config.camera_models import CameraRuntimeConfig
from worker.types import CURRENT_TEMPORAL_PROFILE, TemporalProfile, TemporalProfileError


def test_current_15fps_derived_values_match_production_literals() -> None:
    """Pin today's derived production constants.

    target_fps, pose interval, bed interval, and frame_interval_sec are the
    values a 15fps worker actually ships. The identity moved 5.0 -> 15.0 when
    the profile was raised; the bed interval moved 30 -> 90 in the same edit
    so the bed decision rate stayed 1/6 Hz (6.0s wall clock) rather than
    tripling to 1/2 Hz.
    """
    policy = CapturePolicy()
    camera = CameraRuntimeConfig(
        camera_id="camera-a",
        facility_id="facility-a",
        rtsp_url="rtsp://192.0.2.1/camera-a",
    )
    scheduler = Scheduler()
    source = RTSPSource(object(), object())

    assert policy.target_fps == 15.0
    assert camera.fps == 15.0
    assert scheduler.task_intervals["pose"] == 1
    assert scheduler.task_intervals["bed"] == 90
    assert source._frame_interval_sec == pytest.approx(1.0 / 15.0)  # noqa: SLF001


def test_temporal_profile_at_5fps_computes_current_derived_constants() -> None:
    profile = TemporalProfile(ingest_fps=5.0)
    camera = CameraRuntimeConfig(
        camera_id="camera-a",
        facility_id="facility-a",
        rtsp_url="rtsp://192.0.2.1/camera-a",
        fps=profile.target_fps,
    )
    policy = CapturePolicy(target_fps=profile.target_fps)
    scheduler = Scheduler(profile.task_intervals())
    source = RTSPSource(object(), object(), target_fps=profile.target_fps)

    assert profile.target_fps == 5.0
    assert profile.pose_fps == 5.0
    assert profile.decision_hz["bed"] == pytest.approx(1.0 / 6.0)
    assert profile.pose_interval_frames() == 1
    assert profile.decision_interval_frames("bed") == 30
    assert profile.frame_interval_sec == 0.2
    assert camera.fps == 5.0
    assert policy.target_fps == 5.0
    assert scheduler.task_intervals == {"pose": 1, "bed": 30}
    assert source._frame_interval_sec == 0.2  # noqa: SLF001
    assert CURRENT_TEMPORAL_PROFILE.ingest_fps == 15.0
    assert CURRENT_TEMPORAL_PROFILE.task_intervals() == {"pose": 1, "bed": 90}


@pytest.mark.parametrize("ingest_fps", [0, -1.0, nan, inf, True, "5"])
def test_temporal_profile_rejects_malformed_ingest_fps(ingest_fps: object) -> None:
    with pytest.raises(TemporalProfileError, match="ingest_fps"):
        TemporalProfile(ingest_fps=ingest_fps)  # type: ignore[arg-type]


def test_temporal_profile_rejects_nonpositive_decision_hz() -> None:
    with pytest.raises(TemporalProfileError, match="decision_hz"):
        TemporalProfile(ingest_fps=5.0, decision_hz={"bed": 0.0})


def test_schedule_rule_resolve_requires_an_explicit_temporal_profile() -> None:
    """The implicit CURRENT fallback is gone: resolve() names its owner."""
    from worker.domains.module_definition import ScheduleRule

    bed = ScheduleRule("bed", "temporal-profile")
    fifteen = TemporalProfile(ingest_fps=15.0)

    with pytest.raises(TypeError):
        bed.resolve(1)  # type: ignore[call-arg]
    assert bed.resolve(1, CURRENT_TEMPORAL_PROFILE) == 90
    assert bed.resolve(1, fifteen) == 90
    assert fifteen.decision_interval_frames("bed") == 90


def test_raising_ingest_fps_preserves_the_bed_decision_wall_clock() -> None:
    """The bed Hz is the invariant; the frame count is the derived value.

    Raising ingest fps must re-denominate the bed interval in the same edit.
    If a future raise changes only ``_CURRENT_INGEST_FPS`` and leaves the
    frame count alone, the decision rate silently multiplies and bed-exit
    behaviour changes. This pins the wall clock instead of the frame count so
    that regression fails loudly here.
    """
    assert CURRENT_TEMPORAL_PROFILE.decision_hz["bed"] == pytest.approx(1.0 / 6.0)

    bed_frames = CURRENT_TEMPORAL_PROFILE.decision_interval_frames("bed")
    wall_clock_sec = bed_frames / CURRENT_TEMPORAL_PROFILE.ingest_fps
    assert wall_clock_sec == pytest.approx(6.0)

    legacy = TemporalProfile(ingest_fps=5.0)
    legacy_wall_clock = legacy.decision_interval_frames("bed") / legacy.ingest_fps
    assert legacy_wall_clock == pytest.approx(wall_clock_sec)


def test_compile_time_and_activation_bed_interval_agree_at_15fps() -> None:
    """Compile-time validation and live activation must see the same bed interval.

    Before the implicit-fallback fix, compile calls ``rule.resolve(1)`` and
    gets CURRENT's 30 while activation with an injected 15fps profile gets 90.
    """
    from worker.domains.module_compiler import compile_detection_module_registry
    from worker.domains.registry import (
        AVAILABLE_OBSERVATION_CHANNELS,
        DETECTION_MODULE_DEFINITIONS,
        DETECTION_MODULE_REGISTRY,
    )
    from worker.pipeline.analytics.merge import result_merger_names

    profile = TemporalProfile(ingest_fps=15.0)
    bed_rule = next(
        rule
        for definition in DETECTION_MODULE_DEFINITIONS
        for rule in definition.schedule_rules
        if rule.component_id == "bed"
    )
    # Pre-fix compile-time call was ``rule.resolve(1)``. That form must not
    # exist: if the optional CURRENT fallback returns, this test fails and
    # the 30-vs-90 split is back.
    with pytest.raises(TypeError):
        bed_rule.resolve(1)  # type: ignore[call-arg]
    compiled = compile_detection_module_registry(
        DETECTION_MODULE_DEFINITIONS,
        available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
        output_adapter_ids=result_merger_names(),
        temporal_profile=profile,
    )
    flags = {"person-box-source": True, "persisted-bed-region": False}
    components = compiled.shared_component_ids(("fall", "bed_exit"), flags=flags)
    activation = compiled.activation(
        module_ids=("fall", "bed_exit"),
        available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
        available_component_ids=components,
        warmed_component_ids=components,
        output_adapter_ids=result_merger_names(),
        camera_frame_stride=1,
        flags=flags,
        temporal_profile=profile,
    )
    # The bed extractor is provisioned on-demand: its rule resolves to no
    # interval and it never appears in the per-frame activation schedule, so
    # compile-time validation and live activation still agree exactly.
    assert bed_rule.resolve(1, profile) is None
    assert "bed" not in activation.schedule
    # The profile still declares the bed cadence for the on-demand owner.
    assert profile.decision_interval_frames("bed") == 90
    # Production registry stays on the 5fps identity; this test must not
    # mutate the process-wide compiled graph.
    assert DETECTION_MODULE_REGISTRY is not compiled


def test_temporal_profile_governs_ingest_over_relay_declared_fps() -> None:
    """Design B: TemporalProfile is the ingest-fps owner.

    A relay-style CameraRuntimeConfig.fps is a declared hint. Effective
    CapturePolicy.target_fps comes from the injected profile, so raising
    the profile for a measurement run actually raises fps.
    """
    from worker.pipeline.bus import BoundedFrameBus
    from worker.runtime.ingest_composition import (
        build_camera_source_registry,
        compose_camera_ingest_loop,
    )

    camera = CameraRuntimeConfig(
        camera_id="camera-a",
        facility_id="facility-a",
        rtsp_url="rtsp://192.0.2.1/camera-a",
        fps=15.0,
    )
    registry = build_camera_source_registry((camera,))
    five = TemporalProfile(ingest_fps=5.0)
    fifteen = TemporalProfile(ingest_fps=15.0)
    clamped = compose_camera_ingest_loop(
        camera,
        BoundedFrameBus(),
        _IngestReporter(),
        decode="opencv",
        registry=registry,
        temporal_profile=five,
    )
    raised = compose_camera_ingest_loop(
        camera.model_copy(update={"fps": 5.0}),
        BoundedFrameBus(),
        _IngestReporter(),
        decode="opencv",
        registry=registry,
        temporal_profile=fifteen,
    )

    assert camera.fps == 15.0
    assert clamped._spec.policy.target_fps == 5.0  # noqa: SLF001
    assert raised._spec.policy.target_fps == 15.0  # noqa: SLF001


class _IngestReporter:
    def mark_starting(self, camera_id: str) -> None:
        del camera_id

    def mark_ready(self, camera_id: str) -> None:
        del camera_id

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        del camera_id, category

    def emit(self, event: object) -> None:
        del event

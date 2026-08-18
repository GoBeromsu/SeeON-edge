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


def test_current_5fps_derived_values_match_production_literals() -> None:
    """Pin today's derived production constants.

    target_fps, pose interval, bed interval, and frame_interval_sec are the
    values a 5fps worker actually ships. The TemporalProfile refactor is
    identity-preserving only if this test still passes unchanged.
    """
    policy = CapturePolicy()
    camera = CameraRuntimeConfig(
        camera_id="camera-a",
        facility_id="facility-a",
        rtsp_url="rtsp://192.0.2.1/camera-a",
    )
    scheduler = Scheduler()
    source = RTSPSource(object(), object())

    assert policy.target_fps == 5.0
    assert camera.fps == 5.0
    assert scheduler.task_intervals["pose"] == 1
    assert scheduler.task_intervals["bed"] == 30
    assert source._frame_interval_sec == 0.2  # noqa: SLF001
    assert source._frame_interval_sec == 1.0 / 5.0  # noqa: SLF001


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
    assert CURRENT_TEMPORAL_PROFILE.ingest_fps == 5.0
    assert CURRENT_TEMPORAL_PROFILE.task_intervals() == {"pose": 1, "bed": 30}


@pytest.mark.parametrize("ingest_fps", [0, -1.0, nan, inf, True, "5"])
def test_temporal_profile_rejects_malformed_ingest_fps(ingest_fps: object) -> None:
    with pytest.raises(TemporalProfileError, match="ingest_fps"):
        TemporalProfile(ingest_fps=ingest_fps)  # type: ignore[arg-type]


def test_temporal_profile_rejects_nonpositive_decision_hz() -> None:
    with pytest.raises(TemporalProfileError, match="decision_hz"):
        TemporalProfile(ingest_fps=5.0, decision_hz={"bed": 0.0})

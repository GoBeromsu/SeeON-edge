"""Focused contracts for local metrics and the frozen relay projection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from contracts.decode_diagnostics import DecodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.inference_coordinator import (
    CameraInferenceTelemetry,
    InferenceTelemetrySnapshot,
)
from worker.pipeline.perception.scene_state import BedRegionCacheCounters
from worker.runtime.telemetry.runtime_diagnostics import (
    EncoderLifecycleSnapshot,
    WorkerDiagnostics,
)
from worker.runtime.telemetry.status_store import CameraStatus, StatusStore


@dataclass(frozen=True, slots=True)
class _BusMetrics:
    published: int
    taken: int
    dropped: int
    queue_age_sec: float


@dataclass(frozen=True, slots=True)
class _Bus:
    values: dict[str, _BusMetrics]

    def metrics(self, name: str) -> _BusMetrics:
        return self.values[name]


def _selection() -> DecodeSelection:
    return DecodeSelection(
        requested="opencv",
        selected="opencv",
        fallback_count=0,
        last_reason=None,
        updated_at_sec=10.0,
    )


def test_local_snapshot_includes_stage_bus_encoder_and_failure_metrics() -> None:
    # Given
    statuses = StatusStore()
    _ = statuses.set_status(
        "camera-1",
        "facility-1",
        CameraStatus.DEGRADED,
        error_category="auth",
        timestamp=10.0,
    )
    diagnostics = WorkerDiagnostics(statuses)
    diagnostics.register_decode("camera-1", "opencv")
    diagnostics.update_decode("camera-1", _selection())
    diagnostics.record_stage_timing("camera-1", "decode", 0.1)
    diagnostics.record_stage_timing("camera-1", "decode", 0.2)
    diagnostics.register_bus(
        "camera-1",
        _Bus({"inference": _BusMetrics(5, 3, 2, 0.4)}),
        ("inference",),
    )
    diagnostics.update_encoder_lifecycle(
        EncoderLifecycleSnapshot(
            process_starts=4,
            recreates=2,
            failures=1,
            active_sessions=3,
            finalized_segments=8,
            unavailable_cameras=("camera-1",),
        )
    )
    counters = BedRegionCacheCounters(fresh=3, cached=1, expired=1, reset=0, scheduled_empty=2)
    diagnostics.record_bed_region("camera-1", BedRegionCacheState.CACHED, counters.snapshot())

    # When
    snapshot = diagnostics.snapshot()

    # Then
    camera = snapshot.cameras[0]
    assert camera.failure_category == "auth"
    assert camera.stage_timings[0].samples == 2
    assert abs(camera.stage_timings[0].total_sec - 0.3) < 1e-12
    assert camera.stage_timings[0].last_sec == 0.2
    assert camera.bus[0].dropped == 2
    assert camera.bus[0].queue_age_sec == 0.4
    assert snapshot.encoder.failures == 1
    assert snapshot.encoder.finalized_segments == 8
    assert snapshot.encoder.unavailable_cameras == ("camera-1",)
    assert camera.bed_region is not None
    assert camera.bed_region.freshness == BedRegionCacheState.CACHED
    assert camera.bed_region.counters == {
        "fresh": 3,
        "cached": 1,
        "expired": 1,
        "reset": 0,
        "scheduled_empty": 2,
    }


def test_local_metric_is_excluded_from_frozen_wire_payload() -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-1", _selection())
    diagnostics.record_stage_timing("camera-1", "local_metric_must_not_leak", 1.0)

    # When
    payload = diagnostics.to_payload("facility-1", None, 1)

    # Then
    assert set(payload) == {
        "facility_id",
        "generation",
        "seq",
        "cameras",
        "clip_export",
        "clip_recorder",
    }
    assert set(payload["cameras"][0]) == {"camera_id", "decode", "detection"}
    assert payload["cameras"][0].get("detection") == {
        "expected": False,
        "inference_admitted": 0,
        "inference_succeeded": 0,
        "inference_overwritten": 0,
        "decision_completed": 0,
    }
    assert "local_metric_must_not_leak" not in repr(payload)


def test_bed_region_is_excluded_from_frozen_wire_payload() -> None:
    """Bed-region diagnostics never cross the relay boundary (issue #207).

    Same reasoning as ``encode`` (#53, see the comment on
    ``CameraDiagnosticsSnapshot.bed_region``): the strict backend contract in
    worker/runtime/telemetry/wire.py has no field for it, so it must stay
    local-only, same as encode/stage_timings/bus already do.
    """
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-1", _selection())
    counters = BedRegionCacheCounters(fresh=1)
    diagnostics.record_bed_region("camera-1", BedRegionCacheState.FRESH, counters.snapshot())

    # When
    payload = diagnostics.to_payload("facility-1", None, 1)

    # Then
    assert set(payload["cameras"][0]) == {"camera_id", "decode", "detection"}
    assert payload["cameras"][0].get("detection") == {
        "expected": False,
        "inference_admitted": 0,
        "inference_succeeded": 0,
        "inference_overwritten": 0,
        "decision_completed": 0,
    }
    assert "bed_region" not in repr(payload)


def test_structured_log_contains_local_metrics(caplog: pytest.LogCaptureFixture) -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-1", _selection())
    diagnostics.record_stage_timing("camera-1", "inference", 0.25)

    # When
    with caplog.at_level(logging.INFO):
        diagnostics.log_snapshot()

    # Then
    record = caplog.records[-1]
    assert record.getMessage().startswith("worker.runtime.telemetry ")
    assert "camera_id=camera-1" in record.getMessage()
    assert vars(record).get("camera_id") == "camera-1"
    assert vars(record).get("stage_timings") == {
        "inference": {
            "samples": 1,
            "total_sec": 0.25,
            "last_sec": 0.25,
            "max_sec": 0.25,
        }
    }


def test_structured_log_survives_the_entrypoints_basicconfig_format(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Values must appear in the *formatted* line, not just on the LogRecord.

    ``worker/__main__.py`` configures the root logger with
    ``logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s -
    %(message)s")``. That format string never references an ``extra`` key, so
    anything passed only through ``extra=`` -- which is what every field here
    used to be -- is silently absent from what an operator actually reads,
    even though ``caplog``/``vars(record)`` still see it as a LogRecord
    attribute. Every other test in this file asserts against
    ``vars(record)``, which is exactly why this gap went unnoticed: it proves
    the call was made, never that the value survives formatting.
    """
    # Given -- the exact format string worker/__main__.py's basicConfig uses.
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    diagnostics = WorkerDiagnostics()
    diagnostics.record_stage_timing("camera-1", "inference", 0.25)
    counters = BedRegionCacheCounters(fresh=1, expired=3)
    diagnostics.record_bed_region("camera-1", BedRegionCacheState.EXPIRED, counters.snapshot())
    # Issue #238: same trap as bed_region -- a value that only reaches
    # `extra=` renders invisibly, and this is the field tonight's redeploy
    # depends on to tell (b) "never scored" from (c) "scored, exit counter
    # never crossed" apart. Must be asserted against *rendered* output, not
    # `vars(record)`, or a regression here goes uncaught the same way #224's
    # did.
    diagnostics.record_bed_exit_scoring(
        "camera-1",
        max_containment_observed=0.82,
        grace_positive_transitions=2,
        assignments_made=1,
    )

    # When
    with caplog.at_level(logging.INFO):
        diagnostics.log_snapshot()
    rendered = formatter.format(caplog.records[-1])

    # Then
    assert "camera_id=camera-1" in rendered
    assert "stage_timings=" in rendered
    assert "inference" in rendered
    assert "bed_region=" in rendered
    assert "'freshness': 'expired'" in rendered
    assert "bed_exit_scoring=" in rendered
    assert "'max_containment_observed': 0.82" in rendered
    assert "'grace_positive_transitions': 2" in rendered
    assert "'assignments_made': 1" in rendered


def test_structured_log_lets_an_operator_read_bed_region_liveness_without_ssh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A log line alone answers "is this camera's bed region alive?" (#207).

    No SSH + sqlite required: `camera_id` plus `bed_region.freshness` is
    sufficient, and the counters give the history behind that state.
    """
    # Given
    diagnostics = WorkerDiagnostics()
    counters = BedRegionCacheCounters(fresh=10, cached=4, expired=2, reset=1, scheduled_empty=6)
    diagnostics.record_bed_region("camera-9", BedRegionCacheState.EXPIRED, counters.snapshot())

    # When
    with caplog.at_level(logging.INFO):
        diagnostics.log_snapshot()

    # Then
    record = caplog.records[-1]
    assert vars(record).get("camera_id") == "camera-9"
    bed_region = vars(record).get("bed_region")
    assert bed_region is not None
    assert bed_region["freshness"] == "expired"
    assert bed_region["counters"] == {
        "fresh": 10,
        "cached": 4,
        "expired": 2,
        "reset": 1,
        "scheduled_empty": 6,
    }


def test_structured_log_bed_region_never_carries_a_url_ip_or_credential_shaped_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The bed-region log payload is closed to freeform strings (issue #207).

    ``BedRegionCacheState`` is a four-member enum and the counters are plain
    ints (`BedRegionCacheCounterSnapshot`) -- there is no string field an RTSP
    URL, camera IP, or credential could ever be written into, so this asserts
    the *shape* stays closed rather than pattern-matching for one leak.
    """
    # Given
    diagnostics = WorkerDiagnostics()
    counters = BedRegionCacheCounters(fresh=1, cached=0, expired=0, reset=0, scheduled_empty=0)
    diagnostics.record_bed_region("camera-1", BedRegionCacheState.FRESH, counters.snapshot())

    # When
    with caplog.at_level(logging.INFO):
        diagnostics.log_snapshot()

    # Then
    record = caplog.records[-1]
    bed_region = vars(record).get("bed_region")
    assert bed_region is not None
    assert set(bed_region) == {"freshness", "counters", "updated_at_sec"}
    assert bed_region["freshness"] in {"fresh", "cached", "empty", "expired"}
    assert set(bed_region["counters"]) == {
        "fresh",
        "cached",
        "expired",
        "reset",
        "scheduled_empty",
    }
    assert all(isinstance(value, int) for value in bed_region["counters"].values())


def _mixed_geometry_inference() -> InferenceTelemetrySnapshot:
    return InferenceTelemetrySnapshot(
        cameras={
            "camera-a": CameraInferenceTelemetry(
                admitted=1,
                overwritten=0,
                inferred=1,
                queue_age_sec=0.1,
                observed_geometry=(640, 360),
            ),
            "camera-b": CameraInferenceTelemetry(
                admitted=1,
                overwritten=0,
                inferred=1,
                queue_age_sec=0.2,
                observed_geometry=(640, 480),
            ),
        },
        batch_sizes={1: 2},
        forward_p50_sec=0.02,
        forward_p95_sec=0.02,
        geometry_batch_sizes={(640, 360): {1: 1}, (640, 480): {1: 1}},
    )


class _InferenceSource:
    def __init__(self, snapshot: InferenceTelemetrySnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> InferenceTelemetrySnapshot:
        return self._snapshot


def test_diagnostics_snapshot_carries_local_geometry_fields() -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.register_inference(_InferenceSource(_mixed_geometry_inference()))
    # When
    snapshot = diagnostics.snapshot()
    # Then
    cameras = {camera.camera_id: camera for camera in snapshot.cameras}
    assert cameras["camera-a"].inference is not None
    assert cameras["camera-b"].inference is not None
    assert cameras["camera-a"].inference.observed_geometry == (640, 360)
    assert cameras["camera-b"].inference.observed_geometry == (640, 480)
    assert cameras["camera-a"].failure_category is None
    assert cameras["camera-b"].failure_category is None
    histograms = {
        item.geometry: dict(item.batch_sizes)
        for item in cameras["camera-a"].geometry_batch_sizes
    }
    assert histograms == {(640, 360): {1: 1}, (640, 480): {1: 1}}
    assert cameras["camera-a"].geometry_batch_sizes == cameras["camera-b"].geometry_batch_sizes


def test_stable_mixed_geometries_are_not_a_health_failure() -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.register_inference(_InferenceSource(_mixed_geometry_inference()))
    # When
    snapshot = diagnostics.snapshot()
    # Then
    assert all(camera.failure_category is None for camera in snapshot.cameras)
    assert {camera.inference.observed_geometry for camera in snapshot.cameras} == {
        (640, 360),
        (640, 480),
    }


def test_log_snapshot_renders_geometry_key_distribution_in_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    diagnostics = WorkerDiagnostics()
    diagnostics.register_inference(_InferenceSource(_mixed_geometry_inference()))
    # When
    with caplog.at_level(logging.INFO):
        diagnostics.log_snapshot()
    rendered = formatter.format(caplog.records[-1])
    message = caplog.records[-1].getMessage()
    # Then
    assert "640x360" in message
    assert "640x480" in message
    assert "geometry_batch_sizes" in message
    assert "640x360" in rendered
    assert "640x480" in rendered
    assert "geometry_batch_sizes" in rendered


def test_geometry_fields_stay_out_of_relay_payload() -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_inference(_InferenceSource(_mixed_geometry_inference()))
    # When
    payload = diagnostics.to_payload("facility-1", None, 1)
    # Then
    assert set(payload) == {
        "facility_id",
        "generation",
        "seq",
        "cameras",
        "clip_export",
        "clip_recorder",
    }
    assert set(payload["cameras"][0]) == {"camera_id", "decode"}
    dumped = repr(payload)
    assert "640x360" not in dumped
    assert "geometry" not in dumped
    assert "observed_geometry" not in dumped


def test_geometry_log_excludes_rtsp_credentials_and_frame_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.register_inference(_InferenceSource(_mixed_geometry_inference()))
    # When
    with caplog.at_level(logging.INFO):
        diagnostics.log_snapshot()
    message = caplog.records[-1].getMessage()
    extras = vars(caplog.records[-1])
    # Then
    dumped = message + repr(extras.get("inference"))
    assert "rtsp://" not in dumped
    assert "password" not in dumped
    assert "array(" not in dumped
    assert "uint8" not in dumped

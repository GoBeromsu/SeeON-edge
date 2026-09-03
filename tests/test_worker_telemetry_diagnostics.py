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
        item.geometry: dict(item.batch_sizes) for item in cameras["camera-a"].geometry_batch_sizes
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
    camera_keys = set(payload["cameras"][0])
    assert "camera_id" in camera_keys
    assert not camera_keys & {"geometry", "observed_geometry", "geometry_batch_sizes"}
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


def _detection_of(payload: object, camera_id: str) -> dict[str, object]:
    cameras = payload["cameras"]  # pyright: ignore[reportIndexIssue]
    for camera in cameras:
        if camera["camera_id"] == camera_id:
            return camera["detection"]
    raise AssertionError(f"camera {camera_id} missing from relay payload")


def test_camera_without_any_detection_producer_reports_expected_false() -> None:
    """No producer at all is the only case that may claim 'disabled'.

    The backend short-circuits to ``state="disabled"`` on ``expected=False``
    before reading a single counter, so this flag must mean "nobody is
    producing detections for this camera".
    """
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    # Then
    assert detection["expected"] is False
    assert detection["decision_completed"] == 0


def test_native_producer_reports_expected_true_and_real_decision_count() -> None:
    """Regression: the nvidia pump's progress must survive the relay projection.

    ``_detection_for_camera`` used to infer "no producer" from the host
    inference source being ``None``, which is always the case under
    ``ML_WORKER_PROFILE=nvidia``. Every nvidia camera therefore reported
    ``expected=False`` with ``decision_completed`` forced to 0, and the
    dashboard rendered "detection disabled" no matter how well the
    ``NativePolicyPump`` was running.
    """
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_native_detection("camera-a")
    diagnostics.record_detection_completed("camera-a")
    diagnostics.record_detection_completed("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    # Then
    assert detection["expected"] is True
    assert detection["decision_completed"] == 2
    # The wire contract enforces decision_completed <= inference_succeeded <=
    # inference_admitted and 422s otherwise. The child only publishes frames it
    # already inferred and the pump completes a decision for each accepted
    # frame, so for this producer the three counts are equal.
    assert detection["inference_admitted"] == 2
    assert detection["inference_succeeded"] == 2


def test_native_producer_with_no_completions_still_reports_expected_true() -> None:
    """A registered-but-idle producer is not the same as no producer.

    Reporting ``expected=False`` here would hide a dead pump behind the same
    badge as an intentionally disabled camera.
    """
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_native_detection("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    # Then
    assert detection["expected"] is True
    assert detection["decision_completed"] == 0


def test_host_inference_path_is_unchanged_by_native_registration() -> None:
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_inference(_InferenceSource(_mixed_geometry_inference()))
    diagnostics.record_detection_completed("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    # Then
    assert detection["expected"] is True
    assert detection["decision_completed"] == 1


def test_native_producer_counters_satisfy_the_wire_ordering_invariant() -> None:
    """Regression: the backend 422s a payload that breaks counter ordering.

    ``RelayDetectionStatus.counters_are_ordered`` requires
    ``decision_completed <= inference_succeeded <= inference_admitted``.
    Reporting a real ``decision_completed`` alongside zero admitted/succeeded
    made every runtime-status POST fail with HTTP 422 once the native pump
    completed its first decision, which silently stopped all telemetry.
    """
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_native_detection("camera-a")
    for _ in range(5):
        diagnostics.record_detection_completed("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    # Then
    assert detection["decision_completed"] <= detection["inference_succeeded"]
    assert detection["inference_succeeded"] <= detection["inference_admitted"]
    assert detection["decision_completed"] == 5


def test_native_producer_payload_validates_against_the_backend_relay_model() -> None:
    """Cross-boundary contract: the worker payload must satisfy the backend model.

    The per-field unit assertions above cannot catch a schema rejection. This
    validates the real emitted payload against the backend's own Pydantic model
    so a counter-ordering or extra-field regression fails here instead of
    silently 422-ing every runtime-status POST in production.
    """
    # Given
    from backend.app.features.relay.router import RelayDetectionStatus

    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_native_detection("camera-a")
    for _ in range(3):
        diagnostics.record_detection_completed("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    validated = RelayDetectionStatus.model_validate(detection)
    # Then
    assert validated.expected is True
    assert validated.decision_completed == 3


def test_idle_native_producer_payload_also_validates() -> None:
    # Given
    from backend.app.features.relay.router import RelayDetectionStatus

    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_native_detection("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    validated = RelayDetectionStatus.model_validate(detection)
    # Then
    assert validated.expected is True
    assert validated.decision_completed == 0


def test_metadata_slot_counters_render_into_the_log_message_not_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rejection reasons must be greppable in the rendered message.

    The worker's ``basicConfig`` format is ``%(message)s`` only, so anything
    passed via ``extra=`` never reaches an operator. Regression guard for the
    observability gap that made "child never publishes" and "child publishes
    but every frame is rejected" indistinguishable.
    """
    # Given
    import logging

    from worker.native.deepstream.metadata_slot import LatestMetadataSlot
    from worker.runtime import worker as worker_module

    slot = LatestMetadataSlot()
    slot.mark_malformed()
    slot.mark_pull_failure()
    slot.mark_pull_failure()

    class _Child:
        metadata = slot

    class _MediaPlane:
        child = _Child()

    class _Runtime:
        _nvidia_media_plane = _MediaPlane()
        _log_native_metadata_counters = worker_module.WorkerRuntime._log_native_metadata_counters

    # When
    with caplog.at_level(logging.INFO, logger=worker_module.LOGGER.name):
        _Runtime._log_native_metadata_counters(_Runtime())  # pyright: ignore[reportArgumentType]
    # Then
    rendered = [record.getMessage() for record in caplog.records]
    assert any("native metadata slot:" in message for message in rendered)
    line = next(message for message in rendered if "native metadata slot:" in message)
    assert "malformed=1" in line
    assert "pull_failures=2" in line
    for field in (
        "accepted",
        "overwritten",
        "late",
        "unknown_source",
        "generation_mismatch",
        "epoch_mismatch",
        "boot_mismatch",
        "child_mismatch",
        "transform_mismatch",
    ):
        assert f"{field}=" in line


def test_metadata_counter_logging_is_a_no_op_without_the_nvidia_media_plane(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given
    import logging

    from worker.runtime import worker as worker_module

    class _Runtime:
        _nvidia_media_plane = None
        _log_native_metadata_counters = worker_module.WorkerRuntime._log_native_metadata_counters

    # When
    with caplog.at_level(logging.INFO, logger=worker_module.LOGGER.name):
        _Runtime._log_native_metadata_counters(_Runtime())  # pyright: ignore[reportArgumentType]
    # Then
    assert not [r for r in caplog.records if "native metadata slot:" in r.getMessage()]


def test_native_producer_reports_real_attempts_so_success_rate_is_meaningful() -> None:
    """Regression: synthesising admitted == completed pinned success rate at 1.0.

    The backend computes ``recent_success_rate`` as completed_delta over
    admitted_delta, so reporting the two as equal made every native camera look
    perfectly healthy no matter how many frames failed. ``admitted`` must be the
    real attempt count.
    """
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_native_detection("camera-a")
    for _ in range(5):
        diagnostics.record_native_detection_attempt("camera-a")
    for _ in range(3):
        diagnostics.record_detection_completed("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    # Then
    assert detection["inference_admitted"] == 5
    assert detection["decision_completed"] == 3
    # ordering invariant the backend enforces still holds
    assert detection["decision_completed"] <= detection["inference_succeeded"]
    assert detection["inference_succeeded"] <= detection["inference_admitted"]


def test_native_attempts_never_fall_below_completions() -> None:
    """A completion without a recorded attempt must not break the wire contract."""
    # Given
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.register_native_detection("camera-a")
    diagnostics.record_detection_completed("camera-a")
    # When
    detection = _detection_of(diagnostics.to_payload("facility-1", None, 1), "camera-a")
    # Then
    assert detection["inference_admitted"] >= detection["decision_completed"]


def test_runtime_status_reports_cpu_fall_inference_and_real_resample_gap_rows() -> None:
    """P1a-AC6b/AC4: the runtime status names the fall inference device and the
    resampler's actual dropped-bucket count per camera.

    ``fall_inference_device`` defaults to ``unknown`` and is only ever recorded
    from a runner that declares CPU placement, so the field is evidence rather
    than a hardcoded label; a non-CPU device is refused outright.
    """
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.record_fall_inference_device("camera-a", "cpu")
    diagnostics.record_resample_gap_rows("camera-a", 2)
    diagnostics.record_resample_gap_rows("camera-a")

    cameras = {camera.camera_id: camera for camera in diagnostics.snapshot().cameras}
    assert cameras["camera-a"].fall_inference_device == "cpu"
    assert cameras["camera-a"].resample_gap_rows_total == 3

    diagnostics.update_decode("camera-b", _selection())
    unreported = {camera.camera_id: camera for camera in diagnostics.snapshot().cameras}
    assert unreported["camera-b"].fall_inference_device == "unknown"
    assert unreported["camera-b"].resample_gap_rows_total == 0

    with pytest.raises(ValueError, match="cpu"):
        diagnostics.record_fall_inference_device("camera-a", "cuda")


def test_runtime_status_reports_absorbed_track_id_switches() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode("camera-a", _selection())
    diagnostics.record_track_id_switch_absorbed_total("camera-a", 3)

    camera = {camera.camera_id: camera for camera in diagnostics.snapshot().cameras}["camera-a"]

    assert camera.track_id_switch_absorbed_total == 3

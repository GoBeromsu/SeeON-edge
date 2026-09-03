"""Focused contracts for worker-local camera lifecycle telemetry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from worker.pipeline.ingest.lifecycle import IngestEvent
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.status_store import (
    CameraStatus,
    IngestStatusReporter,
    StatusStore,
)


def test_snapshot_preserves_camera_failure_category_and_recovery() -> None:
    # Given
    store = StatusStore()

    # When
    store.record_camera_reconnecting(
        "camera-1",
        "facility-1",
        "rtsp_reconnecting",
        timestamp=10.0,
    )
    store.record_camera_recovered("camera-1", "facility-1", timestamp=11.0)

    # Then
    snapshot = store.snapshot()
    assert snapshot.cameras[0].status is CameraStatus.READY
    assert snapshot.cameras[0].error_category is None
    assert [(event.event_type, event.category) for event in snapshot.ops_events] == [
        ("camera.offline", "rtsp_reconnecting"),
        ("camera.recovered", "rtsp_recovered"),
    ]


def test_status_updates_are_thread_safe_and_camera_scoped() -> None:
    # Given
    store = StatusStore()
    camera_ids = tuple(f"camera-{index}" for index in range(16))

    def update(camera_id: str) -> None:
        _ = store.set_status(
            camera_id,
            "facility-1",
            CameraStatus.DEGRADED,
            error_category="decode",
            timestamp=20.0,
        )

    # When
    with ThreadPoolExecutor(max_workers=8) as executor:
        _ = tuple(executor.map(update, camera_ids))

    # Then
    snapshot = store.snapshot()
    assert tuple(record.camera_id for record in snapshot.cameras) == tuple(sorted(camera_ids))
    assert all(record.error_category == "decode" for record in snapshot.cameras)


def test_camera_status_and_runtime_wire_preserve_opaque_identity_bytes() -> None:
    camera_id = " \u2007camera/\u1100\u1161/e\u0301 "
    store = StatusStore()
    reporter = IngestStatusReporter(store, {camera_id: "facility-1"})
    diagnostics = WorkerDiagnostics(status_store=store)

    reporter.mark_ready(camera_id)
    diagnostics.register_decode(camera_id, "opencv")

    record = store.get_status(camera_id)
    assert record is not None
    assert record.camera_id.encode("utf-8") == camera_id.encode("utf-8")
    assert store.get_status(camera_id.strip()) is None
    payload = diagnostics.to_payloads({camera_id: "facility-1"}, None, 1)[0]
    wire_camera_id = payload["cameras"][0]["camera_id"]
    assert wire_camera_id.encode("utf-8") == camera_id.encode("utf-8")


def test_ingest_reporter_preserves_source_and_processing_failure_categories() -> None:
    # Given
    store = StatusStore()
    reporter = IngestStatusReporter(store, {"camera-1": "facility-1"})

    # When
    reporter.mark_ready("camera-1")
    reporter.emit(
        IngestEvent("camera-1", "frame.processing_error", "RuntimeError", "frame dropped")
    )
    status_after_processing = store.get_status("camera-1")
    reporter.mark_degraded("camera-1", category="rtsp_reconnecting")
    reporter.emit(IngestEvent("camera-1", "camera.offline", "rtsp_reconnecting", "retrying"))

    # Then
    assert status_after_processing is not None
    assert status_after_processing.status is CameraStatus.READY
    degraded = store.get_status("camera-1")
    assert degraded is not None
    assert degraded.error_category == "rtsp_reconnecting"
    assert [(event.event_type, event.category) for event in store.ops_events()] == [
        ("frame.processing_error", "RuntimeError"),
        ("camera.offline", "rtsp_reconnecting"),
    ]

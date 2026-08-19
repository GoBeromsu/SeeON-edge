"""Local-only requested-versus-actual decode observability contracts."""

from __future__ import annotations

import logging

import pytest

from worker.runtime.telemetry.models import DecodeBackendObservability
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics


def test_local_snapshot_contains_requested_resolved_and_actual_decode_backend() -> None:
    diagnostics = WorkerDiagnostics()

    diagnostics.record_decode_backend(
        "camera-a",
        requested_profile_decode="nvdec",
        resolved_backend="nvdec",
        actual_adapter_class="PyAvPreservingAdapter",
    )

    snapshot = diagnostics.snapshot()

    assert len(snapshot.cameras) == 1
    assert snapshot.cameras[0].camera_id == "camera-a"
    assert snapshot.cameras[0].decode_backend == DecodeBackendObservability(
        requested_profile_decode="nvdec",
        resolved_backend="nvdec",
        actual_adapter_class="PyAvPreservingAdapter",
    )


def test_local_snapshot_log_renders_decode_backend_only_when_recorded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = WorkerDiagnostics()
    recorded.record_decode_backend(
        "camera-a",
        requested_profile_decode="nvdec",
        resolved_backend="nvdec",
        actual_adapter_class="PyAvPreservingAdapter",
    )

    with caplog.at_level(logging.INFO, logger="worker.runtime.telemetry.local_metrics"):
        recorded.log_snapshot()

    message = caplog.records[-1].getMessage()
    assert "decode_backend=" in message
    assert "'requested_profile_decode': 'nvdec'" in message
    assert "'resolved_backend': 'nvdec'" in message
    assert "'actual_adapter_class': 'PyAvPreservingAdapter'" in message

    caplog.clear()
    absent = WorkerDiagnostics()
    absent.record_stage_timing("camera-b", "ingest", 0.1)

    with caplog.at_level(logging.INFO, logger="worker.runtime.telemetry.local_metrics"):
        absent.log_snapshot()

    assert "decode_backend=" not in caplog.records[-1].getMessage()


def test_decode_backend_observability_does_not_change_the_relay_wire_payload() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.record_decode_backend(
        "camera-a",
        requested_profile_decode="nvdec",
        resolved_backend="nvdec",
        actual_adapter_class="PyAvPreservingAdapter",
    )

    payload = diagnostics.to_payload("facility-a", None, 1)

    assert payload["cameras"] == []
    assert "PyAvPreservingAdapter" not in repr(payload)

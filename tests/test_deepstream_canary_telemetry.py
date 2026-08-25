from __future__ import annotations

import json
from pathlib import Path

import pytest

import worker.runtime.deepstream.canary_telemetry as telemetry_module
from worker.runtime.deepstream.canary_telemetry import NativeCanaryTelemetry
from worker.tools.deepstream_canary.collector import CollectionRequest, collect_recorded_telemetry
from worker.tools.deepstream_canary.models import CanaryMode
from worker.tools.deepstream_canary.telemetry import NativeWindowSample, RuntimeGpuSample


def test_native_telemetry_seals_non_overlapping_window_when_boundary_arrives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: deterministic source PTS/wall mapping and one skipped native publication.
    monotonic_values = iter((0.0, 5.0, 10.1, 10.1))
    wall_values = iter((1_000_000_000, 1_100_000_000, 11_200_000_000))
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(telemetry_module.time, "time_ns", lambda: next(wall_values))
    output = tmp_path / "native.jsonl"
    telemetry = NativeCanaryTelemetry("loop-01", output)

    # When: the second decision crosses the exact 10-second window boundary.
    telemetry.record(0, 1_000_000_000, 1)
    telemetry.record(10_000_000_000, 11_000_000_000, 3)

    # Then: one immutable raw window carries measured latency and overwrite counts.
    record = json.loads(output.read_text())
    assert record["decision_count"] == 2
    assert record["latency_samples_ms"] == [100.0, 200.0]
    assert record["metadata_published"] == 3
    assert record["metadata_overwritten"] == 1
    assert record["timestamp_discontinuities"] == 0


def test_collector_flushes_partial_camera_registry_on_abort(tmp_path: Path) -> None:
    # Given: source telemetry is absent but the expected camera registry is known.
    (tmp_path / "raw").mkdir()
    gpu = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )

    # When: the first-fault path seals the rung before teardown.
    receipt = collect_recorded_telemetry(
        CollectionRequest(
            evidence_dir=tmp_path,
            rung="loopback",
            mode=CanaryMode.SHARED_HOST_SMOKE,
            clean_steady_seconds=900,
            camera_ids=("loop-01",),
            gpu_samples=(gpu,),
            native_windows=(),
            allow_partial=True,
            fault_windows=("native_source_ready_timeout",),
        )
    )

    # Then: declared identity and fault survive even though camera samples are empty.
    recorded = json.loads(receipt.read_text())
    assert recorded["camera_count"] == 1
    assert recorded["cameras"] == []
    assert recorded["fault_windows"] == ["native_source_ready_timeout"]


def test_collector_rejects_missing_expected_camera_identity(tmp_path: Path) -> None:
    # Given: a workload declares loop-01 but native telemetry belongs to no such camera.
    window = NativeWindowSample(
        schema_version=1,
        camera_id="wrong-camera",
        window_started_ns=1,
        window_ended_ns=10_000_000_001,
        decision_count=150,
        latency_samples_ms=(1.0,),
        metadata_published=150,
        metadata_overwritten=0,
        timestamp_discontinuities=0,
    )
    gpu = RuntimeGpuSample(
        child_pid=1,
        child_memory_mib=1.0,
        global_used_mib=1.0,
        total_mib=10.0,
        utilization=1.0,
    )

    # When / Then: receipt construction fails instead of emitting cameras=[].
    with pytest.raises(ValueError, match="native camera identities mismatch"):
        collect_recorded_telemetry(
            CollectionRequest(
                evidence_dir=tmp_path,
                rung="loopback",
                mode=CanaryMode.SHARED_HOST_SMOKE,
                clean_steady_seconds=900,
                camera_ids=("loop-01",),
                gpu_samples=(gpu,),
                native_windows=(window,),
            )
        )

"""Operator-gated observability load measurement (Gate M/V).

Skipped, never errored, when mediamtx/ffmpeg/pyservicemaker or OBS_STREAM_PATH
are missing. Asserts document structure and the order of the measurement
window only; never a numeric threshold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from observability_load_harness import ObservabilityLoadSkip, run_measurement, skip_reason

pytestmark = pytest.mark.real_stack
pytest_plugins = (
    "tests_support.postgres_sandbox",
    "tests_support.postgres_diagnostics_sandbox",
)


def test_observability_load_measurement_writes_structural_document(
    postgres_product_sandbox, postgres_audit_runtime, postgres_lifespan_diagnostics_schema
) -> None:
    reason = skip_reason()
    if reason is not None:
        pytest.skip(reason)

    streams = int(os.environ.get("OBS_STREAMS", "1"))
    duration_sec = float(os.environ.get("OBS_DURATION_SEC", "30"))
    camera_fps = float(os.environ.get("OBS_CAMERA_FPS", "15"))
    output_raw = os.environ.get("OBS_OUTPUT_DIR", "").strip()
    output_dir = Path(output_raw) if output_raw else Path(".omo/evidence/observability")

    try:
        document_path = run_measurement(
            streams=streams,
            duration_sec=duration_sec,
            camera_fps=camera_fps,
            output_dir=output_dir,
            sandbox=postgres_product_sandbox,
            audit_runtime=postgres_audit_runtime,
            diagnostics_schema=postgres_lifespan_diagnostics_schema,
        )
    except ObservabilityLoadSkip as skipped:
        pytest.skip(str(skipped))

    assert document_path.is_file()
    payload = json.loads(document_path.read_text(encoding="utf-8"))
    assert "backlog_slope" in payload
    assert payload["streams"] == streams
    assert payload["offered_fps"] == camera_fps
    assert payload.get("exporter_exceptions") == []
    # The window opens only after the worker published its alive status, so
    # bootstrap and warmup never count as measured throughput.
    assert payload["worker_alive_at_unix_sec"] is not None
    assert payload["measurement_started_at_unix_sec"] >= payload["worker_alive_at_unix_sec"]

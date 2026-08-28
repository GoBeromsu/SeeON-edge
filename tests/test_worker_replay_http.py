from __future__ import annotations

import json
import sqlite3
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from backend.app.edge_db.bootstrap import bootstrap_database
from shared.detection_policies import BedExitPolicyV1, FallPolicyV1, make_effective_policy
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig

_TOKEN = "relay-token"


def _trace() -> dict[str, object]:
    return {
        "camera_id": "camera-replay-http",
        "frames": [
            {
                "trace_id": "a" * 64,
                "frame_key": ["boot-replay-http", "camera-replay-http", 1, 1],
                "pts": {"value": 1.0, "missing_reason": None},
                "source_time": {"value": 1.0, "missing_reason": None},
                "frame_width": 640,
                "frame_height": 480,
                "bed_region_provenance": "empty",
                "persons": [],
                "beds": [],
                "components": [],
                "schema_version": 1,
            }
        ],
        "truncation": {
            "handoff_dropped_frames": 0,
            "pruned_frames": 0,
            "persistence_failed_frames": 0,
            "retention_blocked_frames": 0,
            "oldest_retained_seq": 1,
            "newest_retained_seq": 1,
            "oldest_retained_key": ["boot-replay-http", "camera-replay-http", 1, 1],
            "newest_retained_key": ["boot-replay-http", "camera-replay-http", 1, 1],
            "detail_unavailable_reason": None,
        },
    }


def _payload() -> dict[str, object]:
    policy = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.5, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    return {"trace": _trace(), "module_id": "bed_exit", "policy": policy.as_dict()}


def _request(base: str, payload: object, token: str | None = _TOKEN) -> Request:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Edge-Relay-Token"] = token
    return Request(
        f"{base}/replay",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )


def test_replay_requires_relay_token_and_rejects_malformed_body() -> None:
    server = MjpegServer(LatestFrameStore(), MjpegServerConfig(port=0, probe_token=_TOKEN))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with pytest.raises(HTTPError) as missing:
            urlopen(_request(base, _payload(), None), timeout=1)
        assert missing.value.code == 403
        with pytest.raises(HTTPError) as wrong:
            urlopen(_request(base, _payload(), "wrong-token"), timeout=1)
        assert wrong.value.code == 403
        with pytest.raises(HTTPError) as malformed:
            urlopen(_request(base, {"trace": {"camera_id": "camera"}}, _TOKEN), timeout=1)
        assert malformed.value.code == 400
    finally:
        server.stop()


def test_replay_returns_deterministic_result_without_a_fall_model_for_bed_exit() -> None:
    server = MjpegServer(LatestFrameStore(), MjpegServerConfig(port=0, probe_token=_TOKEN))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urlopen(_request(base, _payload()), timeout=1) as response:
            result = json.loads(response.read())
    finally:
        server.stop()
    assert result["reproducible"] is True
    assert result["event_count"] == 0
    assert result["frames"][0]["analysis_trace_id"] == "a" * 64


def test_fall_replay_without_the_running_model_is_a_typed_refusal() -> None:
    payload = _payload()
    payload["module_id"] = "fall"
    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    payload["policy"] = policy.as_dict()
    server = MjpegServer(LatestFrameStore(), MjpegServerConfig(port=0, probe_token=_TOKEN))
    server.start()
    try:
        with pytest.raises(HTTPError) as refused:
            urlopen(_request(f"http://127.0.0.1:{server.port}", payload), timeout=1)
        assert refused.value.code == 422
        assert json.loads(refused.value.read()) == {
            "detail": "module 'fall.v1' requires a fall_model for replay",
            "status": "refused",
        }
    finally:
        server.stop()


def test_replay_reports_truncated_input_without_silently_completing_it() -> None:
    payload = _payload()
    trace = payload["trace"]
    assert isinstance(trace, dict)
    truncation = trace["truncation"]
    assert isinstance(truncation, dict)
    truncation["pruned_frames"] = 1
    server = MjpegServer(LatestFrameStore(), MjpegServerConfig(port=0, probe_token=_TOKEN))
    server.start()
    try:
        with urlopen(_request(f"http://127.0.0.1:{server.port}", payload), timeout=1) as response:
            result = json.loads(response.read())
    finally:
        server.stop()
    assert result["reproducible"] is False
    assert "pruned_frames=1" in result["non_reproducible_reason"]
    assert result["truncation"]["pruned_frames"] == 1


def _replay_command() -> object:
    script = Path(__file__).parents[1] / "scripts/ops/replay-runtime-analysis.py"
    spec = spec_from_file_location("replay_runtime_analysis", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_no_replay_tables(database: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        connection.close()
    assert "qa_replay_runs" not in tables
    assert not any(name.startswith("runtime_analysis_") for name in tables)


def test_packaged_replay_command_posts_without_persisting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    trace_path = tmp_path / "trace.json"
    _ = trace_path.write_text(json.dumps(_trace()), encoding="utf-8")
    server = MjpegServer(LatestFrameStore(), MjpegServerConfig(port=0, probe_token=_TOKEN))
    server.start()
    try:
        status = _replay_command().main(  # type: ignore[attr-defined]
            [
                "--database", str(database),
                "--camera-id", "camera-replay-http",
                "--worker-url", f"http://127.0.0.1:{server.port}",
                "--relay-token", _TOKEN,
                "--module-id", "bed_exit",
                "--policy-json", json.dumps(_payload()["policy"]),
                "--requested-by", "test-operator",
                "--trace-json", str(trace_path),
            ]
        )
    finally:
        server.stop()
    printed = json.loads(capsys.readouterr().out)
    assert status == 0
    assert printed["reproducible"] is True
    assert printed["event_count"] == 0
    _assert_no_replay_tables(database)


def test_packaged_replay_command_missing_input_does_not_open_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    status = _replay_command().main(  # type: ignore[attr-defined]
        [
            "--database", str(database),
            "--camera-id", "missing",
            "--worker-url", "http://127.0.0.1:1",
            "--relay-token", _TOKEN,
            "--module-id", "bed_exit",
            "--policy-json", json.dumps(_payload()["policy"]),
            "--requested-by", "test-operator",
        ]
    )
    assert status == 2
    _assert_no_replay_tables(database)


def test_packaged_replay_command_refuses_truncated_input_without_persisting(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    trace = _trace()
    truncation = trace["truncation"]
    assert isinstance(truncation, dict)
    truncation["handoff_dropped_frames"] = 1
    trace_path = tmp_path / "trace.json"
    _ = trace_path.write_text(json.dumps(trace), encoding="utf-8")
    server = MjpegServer(LatestFrameStore(), MjpegServerConfig(port=0, probe_token=_TOKEN))
    server.start()
    try:
        status = _replay_command().main(  # type: ignore[attr-defined]
            [
                "--database", str(database),
                "--camera-id", "camera-replay-http",
                "--worker-url", f"http://127.0.0.1:{server.port}",
                "--relay-token", _TOKEN,
                "--module-id", "bed_exit",
                "--policy-json", json.dumps(_payload()["policy"]),
                "--requested-by", "test-operator",
                "--trace-json", str(trace_path),
            ]
        )
    finally:
        server.stop()
    assert status == 2
    _assert_no_replay_tables(database)

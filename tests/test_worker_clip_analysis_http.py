from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig

_TOKEN = "relay-token"
_SHA256 = "a" * 64


@dataclass(frozen=True)
class _Status:
    state: Literal["idle", "running", "available", "failed"]
    reason: str | None = None


@dataclass
class _Supervisor:
    admit: bool = True
    reject_reason: str | None = None
    status_value: _Status = _Status("idle")
    calls: list[tuple[str, Path, str, int, int, int, int]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    def status(self, clip_id: str) -> _Status:
        del clip_id
        return self.status_value

    def trigger(
        self,
        clip_id: str,
        clip_path: Path,
        clip_sha256: str,
        *,
        size_bytes: int,
        duration_ms: int,
        width: int,
        height: int,
    ) -> bool:
        self.calls.append((clip_id, clip_path, clip_sha256, size_bytes, duration_ms, width, height))
        if self.reject_reason is not None:
            raise ClipAnalysisRejected(self.reject_reason)
        return self.admit

    def cancel(self, clip_id: str) -> bool:
        self.cancelled.append(clip_id)
        return True


def _request(base: str, path: str, body: object | None = None) -> urllib.request.Request:
    data = None if body is None else json.dumps(body).encode("utf-8")
    return urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"X-Edge-Relay-Token": _TOKEN},
        method="GET" if body is None else "POST",
    )


def _write_ready_manifest(clip: Path, *, sha256: str = _SHA256) -> None:
    clip.with_name("manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 2,
                "state": "READY",
                "clip_id": clip.parent.name,
                "camera_id": "camera-1",
                "event_refs": ["11111111-1111-4111-8111-111111111111"],
                "clip_start_at": "2026-01-01T00:00:00Z",
                "clip_end_at": "2026-01-01T00:00:01Z",
                "finalized_at": "2026-01-01T00:00:02Z",
                "sha256": sha256,
                "size_bytes": 4,
                "duration_ms": 1000,
                "state_version": 2,
                "source_media": {
                    "timestamp_translation_seconds": "0",
                    "streams": [
                        {
                            "index": 0,
                            "media_type": "video",
                            "time_base": "1/90000",
                            "width": 640,
                            "height": 360,
                            "packet_count": 1,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_clip_analysis_trigger_status_and_cancel(tmp_path: Path) -> None:
    clip = tmp_path / "clips" / "camera-20260101-abc" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"clip")
    _write_ready_manifest(clip)
    supervisor = _Supervisor(status_value=_Status("available", "published"))
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token=_TOKEN),
        clip_analysis_supervisor=supervisor,
        clip_store_dir=tmp_path,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(
            _request(base, "/clips/camera-20260101-abc/analysis", {"clip_sha256": _SHA256}),
            timeout=1,
        ) as response:
            assert response.status == 202
            assert json.loads(response.read()) == {"state": "running"}
        assert supervisor.calls == [("camera-20260101-abc", clip, _SHA256, 4, 1000, 640, 360)]

        with urllib.request.urlopen(
            _request(base, "/clips/camera-20260101-abc/analysis"), timeout=1
        ) as response:
            assert json.loads(response.read()) == {"reason": "published", "state": "available"}

        with urllib.request.urlopen(
            _request(base, "/clips/camera-20260101-abc/analysis/cancel", {}), timeout=1
        ) as response:
            assert json.loads(response.read()) == {"cancelled": True}
        assert supervisor.cancelled == ["camera-20260101-abc"]
    finally:
        server.stop()


def test_clip_analysis_rejects_concurrent_missing_and_invalid_requests(tmp_path: Path) -> None:
    supervisor = _Supervisor(admit=False)
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token=_TOKEN),
        clip_analysis_supervisor=supervisor,
        clip_store_dir=tmp_path,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        for path, body, expected in (
            ("/clips/camera-1/analysis", {"clip_sha256": _SHA256}, 404),
            ("/clips/../analysis", {"clip_sha256": _SHA256}, 400),
            ("/clips/camera-1/analysis", {"clip_sha256": "bad"}, 400),
        ):
            try:
                urllib.request.urlopen(_request(base, path, body), timeout=1)
            except urllib.error.HTTPError as error:
                assert error.code == expected
            else:  # pragma: no cover
                raise AssertionError(f"expected {expected}")

        clip = tmp_path / "clips" / "camera-1" / "clip.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"clip")
        _write_ready_manifest(clip)
        try:
            urllib.request.urlopen(
                _request(base, "/clips/camera-1/analysis", {"clip_sha256": _SHA256}), timeout=1
            )
        except urllib.error.HTTPError as error:
            assert error.code == 409
            assert json.loads(error.read()) == {"state": "running"}
        else:  # pragma: no cover
            raise AssertionError("expected 409")
    finally:
        server.stop()


def test_clip_analysis_rejects_sha_mismatch_and_over_cap_facts(tmp_path: Path) -> None:
    clip = tmp_path / "clips" / "camera-1" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"clip")
    _write_ready_manifest(clip)
    supervisor = _Supervisor(reject_reason="input_bytes")
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token=_TOKEN),
        clip_analysis_supervisor=supervisor,
        clip_store_dir=tmp_path,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(
                _request(base, "/clips/camera-1/analysis", {"clip_sha256": "b" * 64}), timeout=1
            )
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert json.loads(error.read()) == {"error": "sha_mismatch"}
        else:  # pragma: no cover
            raise AssertionError("expected 400")
        assert supervisor.calls == []

        try:
            urllib.request.urlopen(
                _request(base, "/clips/camera-1/analysis", {"clip_sha256": _SHA256}), timeout=1
            )
        except urllib.error.HTTPError as error:
            assert error.code == 422
            assert json.loads(error.read()) == {"error": "rejected", "reason": "input_bytes"}
        else:  # pragma: no cover
            raise AssertionError("expected 422")
        assert supervisor.calls == [("camera-1", clip, _SHA256, 4, 1000, 640, 360)]
    finally:
        server.stop()


def test_clip_analysis_resolves_nested_historical_layout(tmp_path: Path) -> None:
    clip = tmp_path / "old" / "archive" / "clips" / "camera-1" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"clip")
    _write_ready_manifest(clip)
    supervisor = _Supervisor()
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token=_TOKEN),
        clip_analysis_supervisor=supervisor,
        clip_store_dir=tmp_path,
    )
    server.start()
    try:
        with urllib.request.urlopen(
            _request(
                f"http://127.0.0.1:{server.port}",
                "/clips/camera-1/analysis",
                {"clip_sha256": _SHA256},
            ),
            timeout=1,
        ):
            pass
        assert supervisor.calls[0][1] == clip
    finally:
        server.stop()

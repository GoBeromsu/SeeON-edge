"""Stored-clip analysis relay and artifact identity coverage."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app, no_lifespan
from shared.events.clip_analysis_wire import (
    ClipAnalysisBox,
    ClipAnalysisFrame,
    ClipAnalysisResult,
    ClipAnalysisTimeBase,
    encode_clip_analysis,
)

CLIP_ID = "clip-1"
CLIP_SHA256 = "a" * 64


class _WorkerServer(ThreadingHTTPServer):
    response_status: int = 202
    response_body: dict[str, object] = {"state": "running"}
    requests: list[tuple[str, str, bytes, str | None]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _WorkerHandler)
        self.requests = []

    @property
    def origin(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class _WorkerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._respond()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.server.requests.append(
            (
                self.command,
                self.path,
                self.rfile.read(length),
                self.headers.get("X-Edge-Relay-Token"),
            )
        )
        self._respond()

    def _respond(self) -> None:
        body = json.dumps(self.server.response_body).encode()
        self.send_response(self.server.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def worker_server() -> Iterator[_WorkerServer]:
    server = _WorkerServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture(autouse=True)
def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    clip_store = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(clip_store))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("ML_API_WORKER_STREAM_TIMEOUT_S", "0.1")
    get_settings.cache_clear()
    yield clip_store
    get_settings.cache_clear()


def _login(client: TestClient) -> None:
    assert (
        client.post(
            "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
        ).status_code
        == 204
    )


def _write_clip(clip_store: Path) -> Path:
    clip_dir = clip_store / "clips" / CLIP_ID
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"original")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": CLIP_ID,
                "camera_id": "camera-1",
                "event_ref": "event-1",
                "started_at": "2026-09-01T00:00:00Z",
                "duration_s": 1.0,
                "codec": "hevc",
                "path": f"clips/{CLIP_ID}",
                "finalized": True,
                "sha256": CLIP_SHA256,
            }
        ),
        encoding="utf-8",
    )
    return clip_dir


def _write_analysis(clip_dir: Path, *, clip_sha256: str = CLIP_SHA256) -> None:
    result = ClipAnalysisResult(
        source="clip_reanalysis",
        clip_id=CLIP_ID,
        clip_sha256=clip_sha256,
        pose_model_sha256="b" * 64,
        bed_model_sha256="c" * 64,
        decoder_identity="ffmpeg",
        analysis_profile_sha256="d" * 64,
        image_width=640,
        image_height=480,
        time_base=ClipAnalysisTimeBase(1, 12000),
        frames=(
            ClipAnalysisFrame(
                pts=0,
                status="available",
                boxes=(ClipAnalysisBox(1, 1, 10, 10, 0.9),),
            ),
        ),
    )
    payload = encode_clip_analysis(result)
    (clip_dir / "clip.analysis.0123456789abcdef.json").write_bytes(payload)
    (clip_dir / "clip.analysis.0123456789abcdef.json.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )


def _write_playback(
    clip_dir: Path,
    *,
    pts_identical: bool,
    source_sha256: str = CLIP_SHA256,
    rendition_sha256: str | None = None,
) -> str:
    playback = clip_dir / "clip.playback-h264.mp4"
    playback.write_bytes(b"playback")
    digest = hashlib.sha256(b"playback").hexdigest()
    (clip_dir / "clip.playback-h264.mp4.sha256").write_text(digest + "\n", encoding="ascii")
    (clip_dir / "clip.playback-h264.timing.json").write_text(
        json.dumps(
            {
                "source_sha256": source_sha256,
                "rendition_sha256": digest if rendition_sha256 is None else rendition_sha256,
                "pts_identical": pts_identical,
                "time_base": "1/1000",
                "frames": 1,
                "source_frames": 1,
            }
        ),
        encoding="utf-8",
    )
    return digest


@pytest.mark.parametrize("worker_status", [202, 409, 404])
def test_trigger_relays_worker_status_and_original_manifest_identity(
    _environment: Path,
    worker_server: _WorkerServer,
    monkeypatch: pytest.MonkeyPatch,
    worker_status: int,
) -> None:
    _write_clip(_environment)
    worker_server.response_status = worker_status
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.post(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == worker_status
    assert worker_server.requests == [
        (
            "POST",
            f"/clips/{CLIP_ID}/analysis",
            b'{"clip_sha256":"' + CLIP_SHA256.encode() + b'"}',
            "relay-token",
        )
    ]


def test_available_analysis_reports_identical_served_timing(_environment: Path) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir)
    _write_playback(clip_dir, pts_identical=True)
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json()["state"] == "available"
    assert response.json()["served_timing_identical"] is True
    assert response.json()["result"]["clip_sha256"] == CLIP_SHA256


def test_available_analysis_reports_nonidentical_served_timing(_environment: Path) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir)
    _write_playback(clip_dir, pts_identical=False)
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {
        "state": "unavailable",
        "served_media_sha256": hashlib.sha256(b"playback").hexdigest(),
        "reason": "timing_unverified",
        "served_timing_identical": False,
    }


def test_analysis_identity_mismatch_is_unavailable(_environment: Path) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir, clip_sha256="e" * 64)
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json() == {
        "state": "unavailable",
        "served_media_sha256": CLIP_SHA256,
        "reason": "identity_mismatch",
    }


def test_worker_unreachable_is_an_honest_available_status(
    _environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_clip(_environment)
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", "http://127.0.0.1:1")
    get_settings.cache_clear()
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.status_code == 200
    assert response.json()["state"] == "unavailable"
    assert response.json()["reason"] == "worker_unreachable"


@pytest.mark.parametrize(
    ("source_sha256", "rendition_sha256"),
    [("e" * 64, None), (CLIP_SHA256, "e" * 64)],
)
def test_analysis_rejects_timing_attestation_bound_to_other_media(
    _environment: Path,
    source_sha256: str,
    rendition_sha256: str | None,
) -> None:
    clip_dir = _write_clip(_environment)
    _write_analysis(clip_dir)
    _write_playback(
        clip_dir,
        pts_identical=True,
        source_sha256=source_sha256,
        rendition_sha256=rendition_sha256,
    )
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.json()["state"] == "unavailable"
    assert response.json()["reason"] == "timing_unverified"
    assert "result" not in response.json()


@pytest.mark.parametrize("worker_state", ["idle", "running", "failed"])
def test_worker_states_include_served_media_identity(
    _environment: Path,
    worker_server: _WorkerServer,
    monkeypatch: pytest.MonkeyPatch,
    worker_state: str,
) -> None:
    _write_clip(_environment)
    worker_server.response_status = 200
    worker_server.response_body = {"state": worker_state}
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert response.json()["state"] == worker_state
    assert response.json()["served_media_sha256"] == CLIP_SHA256


def test_cancel_relays_cancelled_envelope_and_worker_auth_is_unreachable(
    _environment: Path, worker_server: _WorkerServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_clip(_environment)
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", worker_server.origin)
    get_settings.cache_clear()
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        worker_server.response_status = 200
        worker_server.response_body = {"cancelled": True}
        cancelled = client.post(f"/api/v1/clips/{CLIP_ID}/analysis/cancel")
        worker_server.response_status = 403
        unreachable = client.get(f"/api/v1/clips/{CLIP_ID}/analysis")
    assert cancelled.json() == {"cancelled": True}
    assert unreachable.status_code == 200
    assert unreachable.json()["reason"] == "worker_unreachable"


def test_analysis_rejects_clip_id_outside_worker_grammar(_environment: Path) -> None:
    _write_clip(_environment)
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/clip%3Alegacy/analysis")
    assert response.status_code == 400

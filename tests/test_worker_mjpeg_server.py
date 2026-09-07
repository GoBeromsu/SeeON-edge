from __future__ import annotations

import json
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytest

from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.live_view_api import (
    BedZoneRecognizeRegion,
    BedZoneRecognizeResponse,
)
from worker.pipeline.output.mjpeg_server import (
    BedZoneNotFoundError,
    MjpegProbeError,
    MjpegServer,
    MjpegServerConfig,
    dev_mjpeg_enabled,
    dev_mjpeg_host,
)
from worker.types.preview import OverlaySelection

# A real, cv2-decodable JPEG for the clean snapshot provider used by
# bed-zone recognition.
_REAL_JPEG = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))[1].tobytes()
_RELAY_TOKEN = "relay-token"
_AUTH_HEADERS = {"X-Edge-Relay-Token": _RELAY_TOKEN}


@dataclass(frozen=True)
class _ClipAnalysisStatus:
    state: Literal["idle", "running", "available", "failed"] = "idle"
    reason: str | None = None


class _ClipAnalysisSupervisor:
    def status(self, clip_id: str) -> _ClipAnalysisStatus:
        del clip_id
        return _ClipAnalysisStatus()

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
        del clip_id, clip_path, clip_sha256, size_bytes, duration_ms, width, height
        return True

    def cancel(self, clip_id: str) -> bool:
        del clip_id
        return False


_RealMjpegServer = MjpegServer


def MjpegServer(
    store: LatestFrameStore,
    config: MjpegServerConfig,
    *,
    clip_analysis_supervisor: _ClipAnalysisSupervisor | None = None,
    clip_store_dir: Path | None = None,
    **kwargs: object,
) -> _RealMjpegServer:
    return _RealMjpegServer(
        store,
        config,
        clip_analysis_supervisor=(
            _ClipAnalysisSupervisor()
            if clip_analysis_supervisor is None
            else clip_analysis_supervisor
        ),
        clip_store_dir=Path(".") if clip_store_dir is None else clip_store_dir,
        **kwargs,
    )


def _bed_zone_response(
    polygon: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (1, 1)),
) -> BedZoneRecognizeResponse:
    return BedZoneRecognizeResponse(
        regions=(BedZoneRecognizeRegion(id="region-1", polygon=polygon),),
        image_width=16,
        image_height=16,
    )


def _authed_get(url: str, *, timeout: float = 1) -> urllib.request.Request:
    return urllib.request.Request(url, headers=_AUTH_HEADERS)


def _assert_forbidden(request: urllib.request.Request) -> bytes:
    try:
        urllib.request.urlopen(request, timeout=1)
    except urllib.error.HTTPError as exc:
        assert exc.code == 403
        body = exc.read()
        # Credential non-disclosure: rejection body must never echo the token.
        assert _RELAY_TOKEN.encode() not in body
        assert b"relay-token" not in body
        return body
    raise AssertionError("expected 403 Forbidden")  # pragma: no cover


# worker's MjpegServer takes an injected `probe` callable (worker/pipeline/
# output/mjpeg_server.py:36-52) instead of owning an internal RTSP-probing
# helper wired to the legacy sources/probe module's probe_first_frame. The legacy
# test_mjpeg_probe_response_keeps_selected_backend monkeypatched that internal
# helper (edge/runtime/mjpeg_server.py:268-275); there is no worker-side
# equivalent to monkeypatch because backend-selection is now the injected
# probe's responsibility, decoupled from this module entirely (see
# start_optional_mjpeg_server's `probe` parameter). That assertion is
# impossible-with-reason here; RTSP-backend-selection coverage belongs to the
# camera-probe/ingest layer, not this HTTP-server module.
#
# The buffer-level "camera-keyed, non-consuming" assertion
# (test_mjpeg_buffer_is_camera_keyed_non_consuming) is superseded by
# tests/test_runtime_latest_frame.py::test_latest_frame_store_is_camera_keyed_and_non_consuming
# since worker unified OverlayFrameBuffer and LatestFrameBuffer into the same
# LatestFrameStore class — it is not re-asserted here.


def test_mjpeg_server_defaults_loopback_and_disabled() -> None:
    assert dev_mjpeg_enabled({}) is False
    assert dev_mjpeg_host({}) == "127.0.0.1"
    store = LatestFrameStore()
    server = MjpegServer(store, MjpegServerConfig(host="127.0.0.1", port=0))
    try:
        assert server.host == "127.0.0.1"
    finally:
        server.stop()


def test_mjpeg_server_unknown_empty_and_stream_response() -> None:
    store = LatestFrameStore()
    store.register_camera("empty")
    store.publish_jpeg("camera-a", b"\xff\xd8jpeg\xff\xd9", frame_index=1)
    server = MjpegServer(store, MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(_authed_get(f"{base}/stream/missing"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")
        try:
            urllib.request.urlopen(_authed_get(f"{base}/stream/empty"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("empty camera should 503")
        with urllib.request.urlopen(_authed_get(f"{base}/stream/camera-a"), timeout=1) as response:
            body = response.read(64)
            assert response.status == 200
            assert b"multipart" in response.headers["Content-Type"].encode()
            assert b"\xff\xd8jpeg" in body
    finally:
        server.stop()


def test_snapshot_unknown_empty_and_happy_path() -> None:
    store = LatestFrameStore()
    store.register_camera("empty")
    store.publish_jpeg("camera-a", b"\xff\xd8jpeg\xff\xd9", frame_index=1)
    server = MjpegServer(store, MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(_authed_get(f"{base}/snapshot/missing"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")
        try:
            urllib.request.urlopen(_authed_get(f"{base}/snapshot/empty"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("empty camera should 503")
        with urllib.request.urlopen(
            _authed_get(f"{base}/snapshot/camera-a"), timeout=1
        ) as response:
            body = response.read()
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            assert body == b"\xff\xd8jpeg\xff\xd9"
    finally:
        server.stop()


def test_mjpeg_server_probe_requires_token_and_returns_sanitized_result() -> None:
    store = LatestFrameStore()
    seen_urls: list[str] = []

    def probe(url: str) -> dict[str, object]:
        seen_urls.append(url)
        return {"ok": True, "backend": "opencv", "width": 640, "height": 360}

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token="relay-token"),
        probe=probe,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    # Public literal IP so admission DNS pinning succeeds without depending on
    # local resolver / special-use hostnames; the injected probe is what the
    # test actually exercises after the gate.
    probe_url = "rtsp://user:secret@8.8.8.8/trackID=2"
    body = json.dumps({"rtsp_url": probe_url}).encode()
    try:
        request = urllib.request.Request(f"{base}/probe", data=body, method="POST")
        try:
            urllib.request.urlopen(request, timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:  # pragma: no cover
            raise AssertionError("probe should require relay token")

        request = urllib.request.Request(
            f"{base}/probe",
            data=body,
            headers={"X-Edge-Relay-Token": "relay-token"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload == {"backend": "opencv", "height": 360, "ok": True, "width": 640}
        assert seen_urls == [probe_url]
        assert "secret" not in json.dumps(payload)
        assert "8.8.8.8" not in json.dumps(payload)
    finally:
        server.stop()


def test_mjpeg_server_probe_normalizes_auth_failure_without_leaking_url() -> None:
    store = LatestFrameStore()

    def probe(url: str) -> dict[str, object]:
        del url
        raise MjpegProbeError("auth")

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token="relay-token"),
        probe=probe,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    probe_url = "rtsp://user:secret@8.8.8.8/trackID=2"
    body = json.dumps({"rtsp_url": probe_url}).encode()
    try:
        request = urllib.request.Request(
            f"{base}/probe",
            data=body,
            headers={"X-Edge-Relay-Token": "relay-token"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload == {"error_class": "auth", "ok": False}
        assert "secret" not in json.dumps(payload)
        assert "8.8.8.8" not in json.dumps(payload)
    finally:
        server.stop()


def test_mjpeg_server_default_probe_reports_unavailable_capability() -> None:
    store = LatestFrameStore()
    server = MjpegServer(store, MjpegServerConfig(port=0, probe_token="relay-token"))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    body = json.dumps({"rtsp_url": "rtsp://8.8.8.8/trackID=2"}).encode()
    try:
        request = urllib.request.Request(
            f"{base}/probe",
            data=body,
            headers={"X-Edge-Relay-Token": "relay-token"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload == {"error_class": "unavailable", "ok": False}
    finally:
        server.stop()


def test_mjpeg_stream_emits_multiple_camera_keyed_non_consuming_parts() -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", b"\xff\xd8camera-a-1\xff\xd9", frame_index=1)
    store.publish_jpeg("camera-b", b"\xff\xd8camera-b-1\xff\xd9", frame_index=1)
    server = MjpegServer(store, MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN))
    server.start()
    base = f"http://127.0.0.1:{server.port}"

    def publish_updates() -> None:
        time.sleep(0.05)
        store.publish_jpeg("camera-b", b"\xff\xd8camera-b-2\xff\xd9", frame_index=2)
        store.publish_jpeg("camera-a", b"\xff\xd8camera-a-2\xff\xd9", frame_index=2)
        time.sleep(0.05)
        store.publish_jpeg("camera-a", b"\xff\xd8camera-a-3\xff\xd9", frame_index=3)

    publisher = threading.Thread(target=publish_updates)
    try:
        with urllib.request.urlopen(
            _authed_get(f"{base}/stream/camera-a", timeout=2), timeout=2
        ) as response:
            publisher.start()
            body = bytearray()
            deadline = time.monotonic() + 2.0
            while b"camera-a-3" not in body and time.monotonic() < deadline:
                body.extend(response.read(1))
            assert response.status == 200
            assert body.count(b"Content-Type: image/jpeg") >= 3
            assert body.count(b"--frame\r\n") >= 3
            assert b"camera-a-1" in body
            assert b"camera-a-2" in body
            assert b"camera-a-3" in body
            assert b"camera-b" not in body
    finally:
        publisher.join(timeout=1)
        server.stop()

    latest = store.get_latest("camera-a")
    assert latest is not None
    assert latest.jpeg == b"\xff\xd8camera-a-3\xff\xd9"


def test_mjpeg_stream_requests_bounded_refreshes_and_emits_new_frames() -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    refresh_times: list[float] = []

    def publish_on_demand(
        camera_id: str,
        viewers: int,
        _selection: OverlaySelection,
        snapshot_requested: bool,
    ) -> None:
        if viewers <= 0 and not snapshot_requested:
            return
        refresh_times.append(time.monotonic())
        frame_index = len(refresh_times)
        store.publish_jpeg(
            camera_id,
            f"\xff\xd8refresh-{frame_index}\xff\xd9".encode(),
            frame_index=frame_index,
        )

    store.set_demand_listener(publish_on_demand)
    server = MjpegServer(store, MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN))
    server.start()
    try:
        with urllib.request.urlopen(
            _authed_get(f"http://127.0.0.1:{server.port}/stream/camera-a"),
            timeout=2,
        ) as response:
            body = bytearray()
            deadline = time.monotonic() + 2.0
            while b"refresh-3" not in body and time.monotonic() < deadline:
                body.extend(response.read(1))

        assert b"refresh-1" in body
        assert b"refresh-2" in body
        assert b"refresh-3" in body
        assert len(refresh_times) >= 3
        assert refresh_times[1] - refresh_times[0] >= 0.04
    finally:
        server.stop()


def test_stream_connect_and_disconnect_track_the_viewer_counter() -> None:
    """Viewer gating (#48): the counter must clear even on a broken pipe.

    A raw socket (not ``http.client``, which duplicates the fd behind a
    ``makefile()`` the moment a response is read, making ``SO_LINGER`` a
    no-op) lets this force an RST instead of a graceful FIN close. That makes
    the server's next ``wfile.write`` raise ``ConnectionResetError`` -- the
    exception return path in ``_handle_stream`` -- exercising the
    ``finally: store.mark_viewer_disconnected(...)`` on that path, not just
    normal completion. A graceful close can otherwise leave the socket
    writable for a while on localhost, so this cannot rely on that.
    """
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", b"\xff\xd8jpeg\xff\xd9", frame_index=1)
    server = MjpegServer(store, MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN))
    server.start()
    try:
        assert store.has_viewers("camera-a") is False
        client = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        try:
            client.sendall(
                b"GET /stream/camera-a HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"X-Edge-Relay-Token: relay-token\r\n"
                b"Connection: keep-alive\r\n\r\n"
            )
            response_head = client.recv(4096)
            assert b"200" in response_head.split(b"\r\n", 1)[0]
            assert store.has_viewers("camera-a") is True

            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        finally:
            client.close()

        deadline = time.monotonic() + 2.0
        while store.has_viewers("camera-a") and time.monotonic() < deadline:
            time.sleep(0.02)
        assert store.has_viewers("camera-a") is False
    finally:
        server.stop()


def test_pose_get_and_set_round_trip_and_defaults_enabled() -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(f"{base}/overlay/camera-a/pose", timeout=1) as response:
            assert json.loads(response.read()) == {"person": True, "bed": True}

        request = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps({"person": False, "bed": True}).encode(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            assert json.loads(response.read()) == {"person": False, "bed": True}

        with urllib.request.urlopen(f"{base}/overlay/camera-a/pose", timeout=1) as response:
            assert json.loads(response.read()) == {"person": False, "bed": True}

        assert store.get_selection("camera-a") == OverlaySelection(person=False, bed=True)

        request = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps({"person": True, "bed": False}).encode(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            assert json.loads(response.read()) == {"person": True, "bed": False}
        assert store.get_selection("camera-a") == OverlaySelection(person=True, bed=False)
    finally:
        server.stop()


def test_pose_get_and_set_open_when_no_relay_token_configured() -> None:
    """Issue #71: a standalone dev MJPEG server (no relay token wired at
    all, i.e. ``MjpegServerConfig(probe_token=None)`` as used above) must
    keep working unauthenticated -- preserving pre-existing dev ergonomics
    -- even though pose GET/POST now gate on the relay token once one *is*
    configured (see the paired ``..._require_token_when_configured`` test).
    """
    store = LatestFrameStore()
    store.register_camera("camera-a")
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(f"{base}/overlay/camera-a/pose", timeout=1) as response:
            assert response.status == 200

        request = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps({"person": False, "bed": False}).encode(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.status == 200
        assert store.get_selection("camera-a") == OverlaySelection(person=False, bed=False)
    finally:
        server.stop()


def test_pose_get_and_set_require_token_when_configured() -> None:
    """Issue #71: once a relay token is configured, pose GET/POST require the
    same ``X-Edge-Relay-Token`` header as ``/probe`` (``_authorized_probe``).
    """
    store = LatestFrameStore()
    store.register_camera("camera-a")
    server = MjpegServer(store, MjpegServerConfig(port=0, probe_token="relay-token"))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(f"{base}/overlay/camera-a/pose", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:  # pragma: no cover
            raise AssertionError("pose GET without a token should 403")

        unauthorized_post = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps({"person": False, "bed": False}).encode(),
            method="POST",
        )
        try:
            urllib.request.urlopen(unauthorized_post, timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:  # pragma: no cover
            raise AssertionError("pose POST without a token should 403")

        wrong_token_post = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps({"person": False, "bed": False}).encode(),
            headers={"X-Edge-Relay-Token": "wrong-token"},
            method="POST",
        )
        try:
            urllib.request.urlopen(wrong_token_post, timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:  # pragma: no cover
            raise AssertionError("pose POST with a mismatched token should 403")

        assert store.get_selection("camera-a") == OverlaySelection()

        authorized_get = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        with urllib.request.urlopen(authorized_get, timeout=1) as response:
            assert json.loads(response.read()) == {"person": True, "bed": True}

        authorized_post = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps({"person": True, "bed": False}).encode(),
            headers={"X-Edge-Relay-Token": "relay-token"},
            method="POST",
        )
        with urllib.request.urlopen(authorized_post, timeout=1) as response:
            assert json.loads(response.read()) == {"person": True, "bed": False}
        assert store.get_selection("camera-a") == OverlaySelection(person=True, bed=False)
    finally:
        server.stop()


def test_pose_unknown_camera_and_malformed_body_are_rejected() -> None:
    store = LatestFrameStore()
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(f"{base}/overlay/missing/pose", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")

        store.register_camera("camera-a")
        request = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=b"not-json",
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:  # pragma: no cover
            raise AssertionError("malformed body should 400")
        assert store.get_selection("camera-a") == OverlaySelection()
    finally:
        server.stop()


def _post_bed_zone_recognize(
    base: str,
    camera_id: str,
    *,
    token: str | None = _RELAY_TOKEN,
    payload: object | None = None,
) -> urllib.request.Request:
    headers = {} if token is None else {"X-Edge-Relay-Token": token}
    return urllib.request.Request(
        f"{base}/overlay/{camera_id}/bed-zone/recognize",
        data=b"" if payload is None else json.dumps(payload).encode(),
        method="POST",
        headers=headers,
    )


def test_bed_zone_recognize_unknown_camera_returns_404() -> None:
    store = LatestFrameStore()
    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=lambda image, confidence: _bed_zone_response(),
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(_post_bed_zone_recognize(base, "missing"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")
    finally:
        server.stop()


def test_bed_zone_recognize_without_recognizer_configured_returns_503(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_snapshot=lambda _camera_id: _REAL_JPEG,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(_post_bed_zone_recognize(base, "camera-a"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("missing recognizer should 503")
        assert "stage=mising_wiring" in caplog.text
        assert "camera_id=camera-a" in caplog.text
        assert "exception_class=RuntimeError" in caplog.text
    finally:
        server.stop()


def test_bed_zone_recognize_without_clean_snapshot_provider_returns_503() -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    recognizer_calls: list[object] = []

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        recognizer_calls.append(image)
        return _bed_zone_response()

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(_post_bed_zone_recognize(base, "camera-a"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("missing clean snapshot provider should 503")
        assert recognizer_calls == []
    finally:
        server.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {"confidence": True},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"confidence": 0.049},
        {"confidence": 0.951},
        {"confidence": "0.5"},
        {"confidence": 0.5, "extra": 1},
        [],
    ],
    ids=(
        "boolean",
        "nan",
        "infinite",
        "below-range",
        "above-range",
        "string",
        "unknown-key",
        "non-object",
    ),
)
def test_bed_zone_recognize_rejects_invalid_request_before_snapshot(payload: object) -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    snapshot_calls: list[str] = []

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=lambda image, confidence: _bed_zone_response(),
        bed_zone_snapshot=lambda camera_id: snapshot_calls.append(camera_id) or _REAL_JPEG,
    )
    server.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(
                _post_bed_zone_recognize(
                    f"http://127.0.0.1:{server.port}",
                    "camera-a",
                    payload=payload,
                ),
                timeout=1,
            )
    finally:
        server.stop()

    assert raised.value.code == 400
    assert snapshot_calls == []


@pytest.mark.parametrize(
    ("payload", "expected"),
    [(None, 0.25), ({}, 0.25), ({"confidence": 0.05}, 0.05), ({"confidence": 0.95}, 0.95)],
)
def test_bed_zone_recognize_forwards_valid_confidence(
    payload: object | None,
    expected: float,
) -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    seen_confidences: list[float] = []

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        seen_confidences.append(confidence)
        return _bed_zone_response()

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=lambda _camera_id: _REAL_JPEG,
    )
    server.start()
    try:
        with urllib.request.urlopen(
            _post_bed_zone_recognize(
                f"http://127.0.0.1:{server.port}",
                "camera-a",
                payload=payload,
            ),
            timeout=1,
        ) as response:
            assert response.status == 200
    finally:
        server.stop()

    assert seen_confidences == [expected]


@pytest.mark.parametrize(
    "provider",
    [
        lambda _camera_id: b"",
        lambda _camera_id: b"not-a-jpeg",
        lambda _camera_id: (_ for _ in ()).throw(RuntimeError("snapshot unavailable")),
    ],
    ids=("empty", "invalid-jpeg", "unavailable"),
)
def test_bed_zone_recognize_snapshot_failure_does_not_call_recognizer(
    provider: Callable[[str], bytes],
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", _REAL_JPEG, frame_index=1)
    recognizer_calls: list[object] = []

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        recognizer_calls.append(image)
        return _bed_zone_response()

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=provider,
    )
    server.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(
                _post_bed_zone_recognize(
                    f"http://127.0.0.1:{server.port}",
                    "camera-a",
                ),
                timeout=1,
            )
    finally:
        server.stop()

    assert raised.value.code == 503
    assert recognizer_calls == []
    assert "stage=snapshot_decode" in caplog.text
    assert "camera_id=camera-a" in caplog.text
    assert "exception_class=" in caplog.text
    assert "snapshot unavailable" not in caplog.text


@pytest.mark.parametrize(
    "decoded",
    [
        np.zeros((16, 16), dtype=np.uint8),
        np.zeros((16, 16, 4), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.float32),
        np.zeros((0, 16, 3), dtype=np.uint8),
    ],
    ids=("two-dimensional", "four-channel", "wrong-dtype", "empty-height"),
)
def test_bed_zone_recognize_rejects_images_outside_the_image_contract(
    monkeypatch: pytest.MonkeyPatch,
    decoded: np.ndarray,
) -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", _REAL_JPEG, frame_index=1)
    recognizer_calls: list[object] = []

    def imdecode(_buffer: np.ndarray, _flags: int) -> np.ndarray:
        return decoded

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        recognizer_calls.append(image)
        return _bed_zone_response()

    monkeypatch.setattr(cv2, "imdecode", imdecode)
    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=lambda _camera_id: _REAL_JPEG,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(_post_bed_zone_recognize(base, "camera-a"), timeout=1)
    finally:
        server.stop()

    assert raised.value.code == 503
    assert recognizer_calls == []


def test_bed_zone_recognize_uses_clean_snapshot_not_annotated_preview() -> None:
    store = LatestFrameStore()
    annotated_jpeg = cv2.imencode(".jpg", np.full((16, 16, 3), 255, dtype=np.uint8))[1].tobytes()
    clean_jpeg = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))[1].tobytes()
    store.publish_jpeg("camera-a", annotated_jpeg, frame_index=1)
    snapshot_camera_ids: list[str] = []
    seen_images: list[np.ndarray] = []

    def clean_snapshot(camera_id: str) -> bytes:
        snapshot_camera_ids.append(camera_id)
        return clean_jpeg

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        seen_images.append(image)
        return _bed_zone_response(((1, 2), (3, 2), (3, 4), (1, 4)))

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=clean_snapshot,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(
            _post_bed_zone_recognize(base, "camera-a"), timeout=1
        ) as response:
            assert response.status == 200
            payload = json.loads(response.read())
        assert payload == {
            "regions": [
                {
                    "id": "region-1",
                    "polygon": [[1, 2], [3, 2], [3, 4], [1, 4]],
                    "origin": "model",
                }
            ],
            "image_width": 16,
            "image_height": 16,
        }
        assert len(seen_images) == 1
        assert snapshot_camera_ids == ["camera-a"]
        assert seen_images[0].dtype == np.dtype(np.uint8)
        assert seen_images[0].shape == (16, 16, 3)
        assert int(seen_images[0].max()) == 0
    finally:
        server.stop()


def test_bed_zone_recognize_converts_decoded_bgr_jpeg_to_rgb() -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    bgr_red = np.zeros((16, 16, 3), dtype=np.uint8)
    bgr_red[:, :] = (0, 0, 255)
    red_jpeg = cv2.imencode(".jpg", bgr_red)[1].tobytes()
    seen_images: list[np.ndarray] = []

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        seen_images.append(image)
        return _bed_zone_response()

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=lambda _camera_id: red_jpeg,
    )
    server.start()
    try:
        with urllib.request.urlopen(
            _post_bed_zone_recognize(
                f"http://127.0.0.1:{server.port}",
                "camera-a",
            ),
            timeout=1,
        ) as response:
            assert response.status == 200
    finally:
        server.stop()

    assert len(seen_images) == 1
    red, green, blue = (int(channel) for channel in seen_images[0][0, 0])
    assert red > 250
    assert green < 5
    assert blue < 5


def test_bed_zone_recognize_not_found_maps_to_structured_404() -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", _REAL_JPEG, frame_index=1)

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        del image
        raise BedZoneNotFoundError("no bed detected")

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=lambda _camera_id: _REAL_JPEG,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(_post_bed_zone_recognize(base, "camera-a"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            payload = json.loads(exc.read())
            assert payload == {"error_class": "bed_not_found"}
        else:  # pragma: no cover
            raise AssertionError("no bed detected should 404 with error_class")
    finally:
        server.stop()


def test_bed_zone_recognize_runner_failure_returns_503(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", _REAL_JPEG, frame_index=1)

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        del image
        raise RuntimeError("model exploded")

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=lambda _camera_id: _REAL_JPEG,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(_post_bed_zone_recognize(base, "camera-a"), timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("recognizer runtime error should 503")
        assert "stage=model" in caplog.text
        assert "camera_id=camera-a" in caplog.text
        assert "exception_class=RuntimeError" in caplog.text
        assert "model exploded" not in caplog.text
    finally:
        server.stop()


@pytest.mark.parametrize(
    "path_builder",
    [
        lambda base: f"{base}/stream/camera-a",
        lambda base: f"{base}/snapshot/camera-a",
        lambda base: f"{base}/overlay/camera-a/bed-zone/recognize",
    ],
    ids=("stream", "snapshot", "bed-zone"),
)
def test_media_endpoints_require_relay_token_no_wrong_correct(
    path_builder: object,
) -> None:
    """Security finding #3: stream/snapshot/bed-zone match probe/delete
    constant-time relay-token auth (``_authorized_probe``).
    """
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", _REAL_JPEG, frame_index=1)

    def recognizer(image: np.ndarray, confidence: float) -> BedZoneRecognizeResponse:
        del image
        return _bed_zone_response()

    server = MjpegServer(
        store,
        MjpegServerConfig(port=0, probe_token=_RELAY_TOKEN),
        bed_zone_recognizer=recognizer,
        bed_zone_snapshot=lambda _camera_id: _REAL_JPEG,
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    path = path_builder(base)  # type: ignore[operator]
    is_post = path.endswith("/bed-zone/recognize")
    method = "POST" if is_post else "GET"
    body = b"" if is_post else None
    try:
        missing = urllib.request.Request(path, data=body, method=method)
        _assert_forbidden(missing)

        wrong = urllib.request.Request(
            path,
            data=body,
            method=method,
            headers={"X-Edge-Relay-Token": "wrong-token"},
        )
        _assert_forbidden(wrong)

        correct = urllib.request.Request(
            path,
            data=body,
            method=method,
            headers=_AUTH_HEADERS,
        )
        with urllib.request.urlopen(correct, timeout=1) as response:
            assert response.status == 200
            body = response.read(64)
            assert _RELAY_TOKEN.encode() not in body
    finally:
        server.stop()


def test_media_endpoints_fail_closed_when_no_token_configured() -> None:
    """Unlike pose (fail-open for standalone dev), media routes match probe:
    missing configured token rejects every caller.
    """
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", _REAL_JPEG, frame_index=1)
    server = MjpegServer(
        store,
        MjpegServerConfig(port=0),
        bed_zone_recognizer=lambda image, confidence: _bed_zone_response(),
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        for path, method in (
            (f"{base}/stream/camera-a", "GET"),
            (f"{base}/snapshot/camera-a", "GET"),
            (f"{base}/overlay/camera-a/bed-zone/recognize", "POST"),
        ):
            request = urllib.request.Request(
                path,
                data=b"" if method == "POST" else None,
                method=method,
                headers=_AUTH_HEADERS,
            )
            _assert_forbidden(request)
    finally:
        server.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"person": True},
        {"bed": True},
        {"person": True, "bed": True, "extra": 1},
        {"person": 1, "bed": True},
        {"person": True, "bed": "true"},
    ],
)
def test_pose_rejects_missing_extra_and_non_boolean_fields(payload: object) -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        request = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps(payload).encode(),
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:  # pragma: no cover
            raise AssertionError("invalid overlay selection should 400")

        assert store.get_selection("camera-a") == OverlaySelection()
    finally:
        server.stop()

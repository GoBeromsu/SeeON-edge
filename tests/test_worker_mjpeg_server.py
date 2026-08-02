from __future__ import annotations

import json
import socket
import struct
import threading
import time
import urllib.error
import urllib.request

from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import (
    MjpegProbeError,
    MjpegServer,
    MjpegServerConfig,
    dev_mjpeg_enabled,
    dev_mjpeg_host,
)

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
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(f"{base}/stream/missing", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")
        try:
            urllib.request.urlopen(f"{base}/stream/empty", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("empty camera should 503")
        with urllib.request.urlopen(f"{base}/stream/camera-a", timeout=1) as response:
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
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(f"{base}/snapshot/missing", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("unknown camera should 404")
        try:
            urllib.request.urlopen(f"{base}/snapshot/empty", timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
        else:  # pragma: no cover
            raise AssertionError("empty camera should 503")
        with urllib.request.urlopen(f"{base}/snapshot/camera-a", timeout=1) as response:
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
    body = json.dumps({"rtsp_url": "rtsp://user:secret@camera.local/trackID=2"}).encode()
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
        assert seen_urls == ["rtsp://user:secret@camera.local/trackID=2"]
        assert "secret" not in json.dumps(payload)
        assert "camera.local" not in json.dumps(payload)
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
    body = json.dumps({"rtsp_url": "rtsp://user:secret@camera.local/trackID=2"}).encode()
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
        assert "camera.local" not in json.dumps(payload)
    finally:
        server.stop()


def test_mjpeg_stream_emits_multiple_camera_keyed_non_consuming_parts() -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", b"\xff\xd8camera-a-1\xff\xd9", frame_index=1)
    store.publish_jpeg("camera-b", b"\xff\xd8camera-b-1\xff\xd9", frame_index=1)
    server = MjpegServer(store, MjpegServerConfig(port=0))
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
        with urllib.request.urlopen(f"{base}/stream/camera-a", timeout=2) as response:
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
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    try:
        assert store.has_viewers("camera-a") is False
        client = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        try:
            client.sendall(
                b"GET /stream/camera-a HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Connection: keep-alive\r\n\r\n"
            )
            response_head = client.recv(4096)
            assert b"200" in response_head.split(b"\r\n", 1)[0]
            assert store.has_viewers("camera-a") is True

            client.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        finally:
            client.close()

        deadline = time.monotonic() + 2.0
        while store.has_viewers("camera-a") and time.monotonic() < deadline:
            time.sleep(0.02)
        assert store.has_viewers("camera-a") is False
    finally:
        server.stop()


def test_pose_get_and_set_round_trip_and_defaults_off() -> None:
    store = LatestFrameStore()
    store.register_camera("camera-a")
    server = MjpegServer(store, MjpegServerConfig(port=0))
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(f"{base}/overlay/camera-a/pose", timeout=1) as response:
            assert json.loads(response.read()) == {"show_pose": False}

        request = urllib.request.Request(
            f"{base}/overlay/camera-a/pose",
            data=json.dumps({"show_pose": True}).encode(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            assert json.loads(response.read()) == {"show_pose": True}

        with urllib.request.urlopen(f"{base}/overlay/camera-a/pose", timeout=1) as response:
            assert json.loads(response.read()) == {"show_pose": True}

        assert store.get_show_pose("camera-a") is True
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
        assert store.get_show_pose("camera-a") is False
    finally:
        server.stop()

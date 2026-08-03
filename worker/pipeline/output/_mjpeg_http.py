from __future__ import annotations

import hmac
import json
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Final, Literal, Required, TypedDict
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from contracts.runner import Image
from worker.pipeline.output.live_view import LatestFrame, LatestFrameStore
from worker.pipeline.output.overlay import OverlayMode

BOUNDARY: Final = b"frame"
POLL_INTERVAL_SECONDS: Final = 0.05
HEARTBEAT_INTERVAL_SECONDS: Final = 1.0
MAX_PROBE_BODY_BYTES: Final = 8192
MAX_POSE_BODY_BYTES: Final = 256
_OVERLAY_MODES: Final[frozenset[str]] = frozenset({"none", "bedexit", "fall"})
# Bounded wait for the first frame after a stream connects. Viewer gating
# (#48) means encoding does not start until this connection's counter
# increment makes `has_viewers` true, so the first frame is not already
# cached the way it always was pre-gating; this must stay comfortably under
# a client's read timeout while giving the pump a real chance to publish.
STREAM_FIRST_FRAME_TIMEOUT_SECONDS: Final = 0.5
# On-demand bed-zone recognition is a deliberate, infrequent user action (not
# periodic polling), so it is worth waiting a bit longer than a stream's first
# frame for a genuinely fresh capture before falling back to whatever is
# already cached.
BED_ZONE_FRAME_TIMEOUT_SECONDS: Final = 2.0

OVERLAY_PREFIX: Final = "/overlay/"
POSE_SUFFIX: Final = "/pose"
BED_ZONE_SUFFIX: Final = "/bed-zone/recognize"

ProbeErrorClass = Literal["auth", "timeout", "decode"]


class BedZonePayload(TypedDict):
    polygon: list[list[int]]
    image_width: int
    image_height: int


class BedZoneNotFoundError(RuntimeError):
    """Recognition ran successfully but found no bed in the frame."""


# Given the best available (raw-ish) frame, run bed segmentation once and
# return the highest-confidence bed's polygon, or raise ``BedZoneNotFoundError``
# when the model finds no bed. Injected from ``worker.runtime`` -- the
# composition root -- so this output-layer module never imports
# ``worker.adapters`` directly, mirroring the existing ``MjpegProbe`` seam.
BedZoneRecognizer = Callable[[Image], BedZonePayload]


class MjpegProbePayload(TypedDict, total=False):
    ok: Required[bool]
    backend: str
    width: int
    height: int
    error_class: ProbeErrorClass


MjpegProbe = Callable[[str], MjpegProbePayload]


class MjpegProbeError(RuntimeError):
    """Carry a bounded probe failure category without URL details."""

    __slots__ = ("error_class",)
    error_class: ProbeErrorClass

    def __init__(self, error_class: str) -> None:
        self.error_class = _normalize_error_class(error_class)
        super().__init__(self.error_class)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_http_server(
    store: LatestFrameStore,
    *,
    host: str,
    port: int,
    probe_token: str | None,
    probe: MjpegProbe,
    bed_zone_recognizer: BedZoneRecognizer | None = None,
    bed_zone_frame_timeout_s: float = BED_ZONE_FRAME_TIMEOUT_SECONDS,
) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            path = urlsplit(self.path).path
            if path.startswith("/stream/"):
                self._handle_stream(unquote(path[len("/stream/") :]))
                return
            if path.startswith("/snapshot/"):
                self._handle_snapshot(unquote(path[len("/snapshot/") :]))
                return
            pose_camera_id = _pose_camera_id(path)
            if pose_camera_id is not None:
                self._handle_get_pose(pose_camera_id)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
            path = urlsplit(self.path).path
            if path != "/probe":
                bed_zone_camera_id = _bed_zone_camera_id(path)
                if bed_zone_camera_id is not None:
                    self._handle_bed_zone_recognize(bed_zone_camera_id)
                    return
                pose_camera_id = _pose_camera_id(path)
                if pose_camera_id is not None:
                    self._handle_set_pose(pose_camera_id)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not _authorized_probe(
                self.headers.get("X-Edge-Relay-Token"),
                probe_token,
            ):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            rtsp_url = self._read_probe_url()
            if rtsp_url is None:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            payload: MjpegProbePayload
            try:
                payload = _sanitize_probe_payload(probe(rtsp_url))
            except MjpegProbeError as exc:
                payload = {"ok": False, "error_class": exc.error_class}
            except (OSError, RuntimeError, TypeError, ValueError):
                payload = {"ok": False, "error_class": "decode"}
            self._write_json(payload)

        def _resolve_frame(self, camera_id: str) -> LatestFrame | None:
            if camera_id == "" or not store.is_known(camera_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return None
            frame = store.get_latest(camera_id)
            if frame is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return None
            return frame

        def _handle_stream(self, camera_id: str) -> None:
            # Viewer gating (#48): count this connection *before* waiting for
            # a frame so `has_viewers` opens the encode gate in time for the
            # pump to actually publish one -- and always uncount it, on every
            # return path (normal completion or a broken/reset pipe), via
            # `finally`.
            if camera_id == "" or not store.is_known(camera_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            store.mark_viewer_connected(camera_id)
            try:
                frame = store.wait_for_latest(
                    camera_id,
                    previous=None,
                    timeout=STREAM_FIRST_FRAME_TIMEOUT_SECONDS,
                )
                if frame is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                previous: LatestFrame | None = None
                last_send_at = 0.0
                while True:
                    frame = store.wait_for_latest(
                        camera_id,
                        previous=previous,
                        timeout=POLL_INTERVAL_SECONDS,
                    )
                    now = time.monotonic()
                    heartbeat_due = now - last_send_at >= HEARTBEAT_INTERVAL_SECONDS
                    if frame is None or (frame is previous and not heartbeat_due):
                        continue
                    try:
                        self._write_part(frame)
                        self.wfile.flush()
                    except (TimeoutError, BrokenPipeError, ConnectionResetError):
                        return
                    previous = frame
                    last_send_at = now
            finally:
                store.mark_viewer_disconnected(camera_id)

        def _handle_snapshot(self, camera_id: str) -> None:
            # Viewer gating (#48) means the cache can go stale with no stream
            # viewer connected; treat this request as a momentary viewer so
            # the next `publish()` encodes one fresh frame, while still
            # serving whatever is cached right now (bounded cost, no wait).
            if store.is_known(camera_id):
                store.request_snapshot_refresh(camera_id)
            frame = self._resolve_frame(camera_id)
            if frame is None:
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", frame.content_type)
            self.send_header("Content-Length", str(len(frame.jpeg)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(frame.jpeg)
            except (TimeoutError, BrokenPipeError, ConnectionResetError):
                return

        def _handle_get_pose(self, camera_id: str) -> None:
            if not _authorized_pose(self.headers.get("X-Edge-Relay-Token"), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if camera_id == "" or not store.is_known(camera_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._write_mode_json(store.get_mode(camera_id))

        def _handle_set_pose(self, camera_id: str) -> None:
            if not _authorized_pose(self.headers.get("X-Edge-Relay-Token"), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if camera_id == "" or not store.is_known(camera_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mode = self._read_mode_body()
            if mode is None:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            store.set_mode(camera_id, mode)
            self._write_mode_json(mode)

        def _handle_bed_zone_recognize(self, camera_id: str) -> None:
            if camera_id == "" or not store.is_known(camera_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if bed_zone_recognizer is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            # Force as fresh a capture as the live-view store can provide:
            # note whatever is already cached, ask for a refresh (viewer
            # gating -- #48 -- means encoding is otherwise paused with no
            # stream viewer connected), then wait for a frame that is not the
            # one we already had. Fall back to the stale cached frame if the
            # wait times out rather than failing the whole request.
            previous = store.get_latest(camera_id)
            store.request_snapshot_refresh(camera_id)
            frame = store.wait_for_latest(
                camera_id, previous=previous, timeout=bed_zone_frame_timeout_s
            )
            if frame is None:
                frame = previous
            if frame is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            image = cv2.imdecode(
                np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            try:
                payload = bed_zone_recognizer(image)
            except BedZoneNotFoundError:
                self._write_status_json(
                    HTTPStatus.NOT_FOUND, {"error_class": "bed_not_found"}
                )
                return
            except (OSError, RuntimeError, TypeError, ValueError, cv2.error):
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self._write_status_json(HTTPStatus.OK, payload)

        def _write_status_json(self, http_status: HTTPStatus, payload: object) -> None:
            body = json.dumps(
                payload, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            self.send_response(http_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_mode_body(self) -> OverlayMode | None:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return None
            try:
                length = int(raw_length)
            except ValueError:
                return None
            if length <= 0 or length > MAX_POSE_BODY_BYTES:
                return None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            if not isinstance(payload, dict) or set(payload) != {"mode"}:
                return None
            mode = payload.get("mode")
            if isinstance(mode, str) and mode in _OVERLAY_MODES:
                return mode  # type: ignore[return-value]
            return None

        def _write_mode_json(self, mode: OverlayMode) -> None:
            body = json.dumps(
                {"mode": mode}, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_probe_url(self) -> str | None:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return None
            try:
                length = int(raw_length)
            except ValueError:
                return None
            if length <= 0 or length > MAX_PROBE_BODY_BYTES:
                return None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            if not isinstance(payload, dict):
                return None
            rtsp_url = payload.get("rtsp_url")
            if not isinstance(rtsp_url, str) or rtsp_url.strip() == "":
                return None
            return rtsp_url

        def _write_json(self, payload: MjpegProbePayload) -> None:
            body = json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_part(self, frame: LatestFrame) -> None:
            self.wfile.write(b"--" + BOUNDARY + b"\r\n")
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(
                f"Content-Length: {len(frame.jpeg)}\r\n\r\n".encode("ascii")
            )
            self.wfile.write(frame.jpeg)
            self.wfile.write(b"\r\n")

        def log_message(self, format: str, *args: str) -> None:  # noqa: A002
            del format, args

    return _ThreadingHTTPServer((host, port), Handler)


def _pose_camera_id(path: str) -> str | None:
    """Match ``/overlay/{camera_id}/pose`` and return the decoded camera id.

    Returns ``None`` for any other path (including an empty camera id),
    letting callers fall through to their existing 404 handling.
    """
    if not path.startswith(OVERLAY_PREFIX) or not path.endswith(POSE_SUFFIX):
        return None
    camera_id = unquote(path[len(OVERLAY_PREFIX) : -len(POSE_SUFFIX)])
    return camera_id or None


def _bed_zone_camera_id(path: str) -> str | None:
    """Match ``/overlay/{camera_id}/bed-zone/recognize`` and return the camera id.

    Returns ``None`` for any other path (including an empty camera id),
    letting callers fall through to their existing 404 handling.
    """
    if not path.startswith(OVERLAY_PREFIX) or not path.endswith(BED_ZONE_SUFFIX):
        return None
    camera_id = unquote(path[len(OVERLAY_PREFIX) : -len(BED_ZONE_SUFFIX)])
    return camera_id or None


def _authorized_probe(supplied: str | None, expected: str | None) -> bool:
    if expected is None or expected.strip() == "" or supplied is None:
        return False
    return hmac.compare_digest(
        supplied.encode("utf-8"),
        expected.strip().encode("utf-8"),
    )


def _authorized_pose(supplied: str | None, expected: str | None) -> bool:
    """Gate the pose-overlay GET/POST routes with the same relay token as
    ``/probe`` (issue #71), but -- unlike ``_authorized_probe`` -- fail open
    when no token is configured. A standalone ``ML_WORKER_DEV_MJPEG`` server
    (no relay token wired at all) must keep working unauthenticated exactly
    as it did before this endpoint had any auth, preserving dev ergonomics;
    once a relay token *is* configured, it is enforced just like ``/probe``.
    """
    if expected is None or expected.strip() == "":
        return True
    return _authorized_probe(supplied, expected)


def _sanitize_probe_payload(payload: MjpegProbePayload) -> MjpegProbePayload:
    if payload.get("ok") is not True:
        return {
            "ok": False,
            "error_class": _normalize_error_class(payload.get("error_class")),
        }
    sanitized: MjpegProbePayload = {"ok": True}
    backend = payload.get("backend")
    width = payload.get("width")
    height = payload.get("height")
    if isinstance(backend, str):
        sanitized["backend"] = backend
    if isinstance(width, int):
        sanitized["width"] = width
    if isinstance(height, int):
        sanitized["height"] = height
    return sanitized


def _normalize_error_class(error_class: str | None) -> ProbeErrorClass:
    if error_class == "auth":
        return "auth"
    if error_class == "timeout":
        return "timeout"
    return "decode"


__all__ = [
    "BED_ZONE_FRAME_TIMEOUT_SECONDS",
    "BedZoneNotFoundError",
    "BedZonePayload",
    "BedZoneRecognizer",
    "MjpegProbe",
    "MjpegProbeError",
    "MjpegProbePayload",
    "build_http_server",
]

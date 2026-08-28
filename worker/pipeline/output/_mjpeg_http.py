from __future__ import annotations

import hmac
import json
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Final, Protocol, TypeAlias
from urllib.parse import urlsplit

import cv2
import numpy as np

from contracts.runner import Image
from shared.detection_policies import parse_effective_policy
from shared.events.replay_wire import MAX_REPLAY_BODY_BYTES as _REPLAY_BODY_LIMIT
from shared.events.replay_wire import ReplayWireError, decode_replay_trace
from worker.domains.fall import FallModelProtocol
from worker.pipeline.output.live_view import LatestFrame, LatestFrameStore
from worker.pipeline.output.live_view_api import (
    BED_ZONE_NOT_FOUND_BODY,
    MJPEG_BOUNDARY,
    MJPEG_MEDIA_TYPE,
    PROBE_PATH,
    RELAY_TOKEN_HEADER,
    REPLAY_PATH,
    BedZoneRecognizeResponse,
    OverlayMode,
    ProbeErrorClass,
    ProbeResponse,
    bed_zone_camera_id,
    clip_deletion_clip_id,
    clip_deletion_preflight_clip_id,
    normalize_probe_error_class,
    parse_pose_body,
    parse_probe_request,
    pose_body,
    pose_camera_id,
    snapshot_camera_id,
    stream_camera_id,
)
from worker.pipeline.trace.models import (
    AnalysisTrace,
    OptionalNumber,
    RecoveredCameraTrace,
    TraceBed,
    TraceComponent,
    TraceKeypoint,
    TracePerson,
    TraceTruncation,
)
from worker.replay.engine import ReplayConfigurationError, ReplayRun, replay_recovered

POLL_INTERVAL_SECONDS: Final = 0.05
HEARTBEAT_INTERVAL_SECONDS: Final = 1.0
MAX_PROBE_BODY_BYTES: Final = 8192
MAX_POSE_BODY_BYTES: Final = 256
# Derived from the trace retention bound in shared/events/replay_wire.py so a
# full retained timeline can actually be transferred. A bare constant here
# refused exactly the long windows replay exists for.
MAX_REPLAY_BODY_BYTES: Final = _REPLAY_BODY_LIMIT
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


class BedZoneNotFoundError(RuntimeError):
    """Recognition ran successfully but found no bed in the frame."""


# Given the best available (raw-ish) frame, run bed segmentation once and
# return the highest-confidence bed's polygon, or raise ``BedZoneNotFoundError``
# when the model finds no bed. Injected from ``worker.runtime`` -- the
# composition root -- so this output-layer module never imports
# ``worker.adapters`` directly, mirroring the existing ``MjpegProbe`` seam.
BedZoneRecognizer = Callable[[Image], BedZoneRecognizeResponse]


class ClipDeletionControl(Protocol):
    def preflight(self, clip_id: str) -> dict[str, object]: ...

    def delete(self, clip_id: str) -> dict[str, object]: ...


# Raw probe result as the runtime's probe callable returns it (it may carry
# the masked URL and a message); only ``ProbeResponse.sanitized`` reaches the wire.
MjpegProbePayload: TypeAlias = dict[str, bool | str | int]


MjpegProbe = Callable[[str], MjpegProbePayload]


class MjpegProbeError(RuntimeError):
    """Carry a bounded probe failure category without URL details."""

    __slots__ = ("error_class",)
    error_class: ProbeErrorClass

    def __init__(self, error_class: str) -> None:
        self.error_class = normalize_probe_error_class(error_class)
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
    clip_deletion_control: ClipDeletionControl | None = None,
    replay_fall_model: FallModelProtocol | None = None,
    bed_zone_frame_timeout_s: float = BED_ZONE_FRAME_TIMEOUT_SECONDS,
) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            path = urlsplit(self.path).path
            stream_id = stream_camera_id(path)
            if stream_id is not None:
                self._handle_stream(stream_id)
                return
            snapshot_id = snapshot_camera_id(path)
            if snapshot_id is not None:
                self._handle_snapshot(snapshot_id)
                return
            pose_id = pose_camera_id(path)
            if pose_id is not None:
                self._handle_get_pose(pose_id)
                return
            clip_preflight_id = clip_deletion_preflight_clip_id(path)
            if clip_preflight_id is not None:
                self._handle_clip_delete_preflight(clip_preflight_id)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
            path = urlsplit(self.path).path
            if path == REPLAY_PATH:
                self._handle_replay()
                return
            if path != PROBE_PATH:
                bed_zone_id = bed_zone_camera_id(path)
                if bed_zone_id is not None:
                    self._handle_bed_zone_recognize(bed_zone_id)
                    return
                pose_id = pose_camera_id(path)
                if pose_id is not None:
                    self._handle_set_pose(pose_id)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not _authorized_probe(
                self.headers.get(RELAY_TOKEN_HEADER),
                probe_token,
            ):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            rtsp_url = self._read_probe_url()
            if rtsp_url is None:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            payload: ProbeResponse
            try:
                # Re-validate destination (including DNS answers) so forged
                # internal callers cannot skip API admission and open an
                # arbitrary RTSP target (SSRF). Pinning happens inside the
                # worker probe/open path; this gate only admits.
                from shared.rtsp_url_policy import assert_rtsp_endpoint_allowed

                try:
                    endpoint = assert_rtsp_endpoint_allowed(rtsp_url)
                except ValueError:
                    payload = ProbeResponse(ok=False, error_class="unsupported")
                else:
                    payload = ProbeResponse.sanitized(probe(endpoint.original_url))
            except MjpegProbeError as exc:
                payload = ProbeResponse(ok=False, error_class=exc.error_class)
            except (OSError, RuntimeError, TypeError, ValueError):
                payload = ProbeResponse(ok=False, error_class="decode")
            self._write_json(payload)

        def _handle_replay(self) -> None:
            if not _authorized_probe(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            request = self._read_replay_body()
            if request is None:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            try:
                trace = decode_replay_trace(request["trace"])
                recovered = _recovered_trace(trace.frames, trace.truncation)
                policy = parse_effective_policy(request["policy"])
                result = replay_recovered(
                    camera_id=trace.camera_id,
                    recovered=recovered,
                    module_id=_replay_module_id(request),
                    policy=policy,
                    fall_model=replay_fall_model,
                )
            except ReplayWireError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            except (ReplayConfigurationError, TypeError, ValueError) as error:
                self._write_status_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"status": "refused", "detail": str(error)},
                )
                return
            self._write_status_json(HTTPStatus.OK, _replay_run_payload(result, trace.truncation))

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib hook name
            path = urlsplit(self.path).path
            clip_id = clip_deletion_clip_id(path)
            if clip_id is not None:
                self._handle_clip_delete(clip_id)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _handle_clip_delete_preflight(self, clip_id: str) -> None:
            if not _authorized_probe(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if clip_deletion_control is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            try:
                payload = clip_deletion_control.preflight(clip_id)
            except (OSError, RuntimeError, TypeError, ValueError):
                self.send_error(HTTPStatus.CONFLICT)
                return
            self._write_status_json(HTTPStatus.OK, payload)

        def _handle_clip_delete(self, clip_id: str) -> None:
            if not _authorized_probe(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if clip_deletion_control is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            try:
                payload = clip_deletion_control.delete(clip_id)
            except (OSError, RuntimeError, TypeError, ValueError):
                self.send_error(HTTPStatus.CONFLICT)
                return
            self._write_status_json(HTTPStatus.ACCEPTED, payload)

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
            # Same constant-time relay-token gate as /probe,
            # and clip delete (security finding #3). Auth runs before any
            # viewer-counter side effect so a rejected caller never opens the
            # encode gate.
            if not _authorized_probe(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
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
                    MJPEG_MEDIA_TYPE,
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
            if not _authorized_probe(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
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
            if not _authorized_pose(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if camera_id == "" or not store.is_known(camera_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._write_mode_json(store.get_mode(camera_id))

        def _handle_set_pose(self, camera_id: str) -> None:
            if not _authorized_pose(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
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
            if not _authorized_probe(self.headers.get(RELAY_TOKEN_HEADER), probe_token):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
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
            decoded = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if not isinstance(decoded, np.ndarray) or decoded.dtype != np.dtype(np.uint8):
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            image = np.asarray(decoded, dtype=np.uint8)
            if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0 or image.shape[2] != 3:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            try:
                payload = bed_zone_recognizer(image)
            except BedZoneNotFoundError:
                self._write_status_json(HTTPStatus.NOT_FOUND, BED_ZONE_NOT_FOUND_BODY)
                return
            except (OSError, RuntimeError, TypeError, ValueError, cv2.error):
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self._write_status_json(HTTPStatus.OK, payload.as_dict())

        def _read_json_object(self, limit: int) -> dict[str, object] | None:
            """Bounded JSON object body, or None when absent, oversized, or malformed."""
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return None
            try:
                length = int(raw_length)
            except ValueError:
                return None
            if length <= 0 or length > limit:
                return None
            try:
                payload: object = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None

        def _write_status_json(self, http_status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            self.send_response(http_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_mode_body(self) -> OverlayMode | None:
            payload = self._read_json_object(MAX_POSE_BODY_BYTES)
            if payload is None:
                return None
            return parse_pose_body(payload)

        def _read_replay_body(self) -> dict[str, object] | None:
            payload = self._read_json_object(MAX_REPLAY_BODY_BYTES)
            if payload is None or set(payload) != {"trace", "module_id", "policy"}:
                return None
            return payload

        def _write_mode_json(self, mode: OverlayMode) -> None:
            self._write_status_json(HTTPStatus.OK, pose_body(mode))

        def _read_probe_url(self) -> str | None:
            payload = self._read_json_object(MAX_PROBE_BODY_BYTES)
            if payload is None:
                return None
            return parse_probe_request(payload)

        def _write_json(self, payload: ProbeResponse) -> None:
            self._write_status_json(HTTPStatus.OK, payload.as_dict())

        def _write_part(self, frame: LatestFrame) -> None:
            self.wfile.write(b"--" + MJPEG_BOUNDARY + b"\r\n")
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(frame.jpeg)}\r\n\r\n".encode("ascii"))
            self.wfile.write(frame.jpeg)
            self.wfile.write(b"\r\n")

        def log_message(self, format: str, *args: str) -> None:  # noqa: A002
            del format, args

    return _ThreadingHTTPServer((host, port), Handler)


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


def _recovered_trace(
    frames: tuple[dict[str, object], ...], truncation: dict[str, object]
) -> RecoveredCameraTrace:
    return RecoveredCameraTrace(
        frames=tuple(_analysis_trace(frame) for frame in frames),
        decisions=(),
        truncation=TraceTruncation(
            handoff_dropped_frames=_integer(truncation, "handoff_dropped_frames"),
            pruned_frames=_integer(truncation, "pruned_frames"),
            persistence_failed_frames=_integer(truncation, "persistence_failed_frames"),
            retention_blocked_frames=_integer(truncation, "retention_blocked_frames"),
            oldest_retained_seq=_nullable_integer(truncation, "oldest_retained_seq"),
            newest_retained_seq=_nullable_integer(truncation, "newest_retained_seq"),
            oldest_retained_key=_trace_key(truncation.get("oldest_retained_key")),
            newest_retained_key=_trace_key(truncation.get("newest_retained_key")),
            detail_unavailable_reason=None,
        ),
    )


def _analysis_trace(frame: dict[str, object]) -> AnalysisTrace:
    required = {
        "trace_id",
        "frame_key",
        "pts",
        "source_time",
        "frame_width",
        "frame_height",
        "bed_region_provenance",
        "persons",
        "beds",
        "components",
        "schema_version",
    }
    if set(frame) != required:
        raise ReplayWireError("analysis frame has undeclared or missing fields")
    persons = _objects(frame["persons"], "persons")
    beds = _objects(frame["beds"], "beds")
    components = _objects(frame["components"], "components")
    return AnalysisTrace(
        trace_id=_text(frame, "trace_id"),
        frame_key=_required_trace_key(frame["frame_key"]),
        pts=_optional_number(frame["pts"], "pts"),
        source_time=_optional_number(frame["source_time"], "source_time"),
        frame_width=_integer(frame, "frame_width"),
        frame_height=_integer(frame, "frame_height"),
        bed_region_provenance=_text(frame, "bed_region_provenance"),
        persons=tuple(_trace_person(person) for person in persons),
        beds=tuple(_trace_bed(bed) for bed in beds),
        components=tuple(_trace_component(component) for component in components),
        schema_version=_integer(frame, "schema_version"),
    )


def _trace_person(value: dict[str, object]) -> TracePerson:
    if set(value) != {"ordinal", "track_id", "box", "confidence", "keypoints"}:
        raise ReplayWireError("person has undeclared or missing fields")
    return TracePerson(
        ordinal=_integer(value, "ordinal"),
        track_id=_optional_number(value["track_id"], "track_id"),
        box=_box(value["box"], "person box"),
        confidence=_number(value, "confidence"),
        keypoints=tuple(
            _trace_keypoint(item) for item in _objects(value["keypoints"], "keypoints")
        ),
    )


def _trace_keypoint(value: dict[str, object]) -> TraceKeypoint:
    if set(value) != {"index", "x", "y", "confidence"}:
        raise ReplayWireError("keypoint has undeclared or missing fields")
    return TraceKeypoint(
        index=_integer(value, "index"),
        x=_integer(value, "x"),
        y=_integer(value, "y"),
        confidence=_number(value, "confidence"),
    )


def _trace_bed(value: dict[str, object]) -> TraceBed:
    if set(value) != {"ordinal", "box", "confidence", "provenance", "polygon"}:
        raise ReplayWireError("bed has undeclared or missing fields")
    polygon = value["polygon"]
    if not isinstance(polygon, list):
        raise ReplayWireError("bed polygon must be a list")
    return TraceBed(
        ordinal=_integer(value, "ordinal"),
        box=_box(value["box"], "bed box"),
        confidence=_number(value, "confidence"),
        provenance=_text(value, "provenance"),
        polygon=tuple(_point(point) for point in polygon),
    )


def _trace_component(value: dict[str, object]) -> TraceComponent:
    if set(value) != {"ordinal", "qualified_id", "observation_state"}:
        raise ReplayWireError("component has undeclared or missing fields")
    return TraceComponent(
        ordinal=_integer(value, "ordinal"),
        qualified_id=_text(value, "qualified_id"),
        observation_state=_text(value, "observation_state"),
    )


def _objects(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ReplayWireError(f"{name} must be a list of objects")
    return value


def _optional_number(value: object, name: str) -> OptionalNumber:
    if not isinstance(value, dict) or set(value) != {"value", "missing_reason"}:
        raise ReplayWireError(f"{name} must carry value and missing_reason")
    present, missing = value["value"], value["missing_reason"]
    if present is not None and (isinstance(present, bool) or not isinstance(present, int | float)):
        raise ReplayWireError(f"{name} value must be numeric")
    if missing is not None and not isinstance(missing, str):
        raise ReplayWireError(f"{name} missing_reason must be text")
    return OptionalNumber(present, missing)


def _trace_key(value: object, *, required: bool = False) -> tuple[str, str, int, int] | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not isinstance(value[0], str)
        or not isinstance(value[1], str)
        or isinstance(value[2], bool)
        or not isinstance(value[2], int)
        or isinstance(value[3], bool)
        or not isinstance(value[3], int)
    ):
        raise ReplayWireError("frame key is invalid")
    return value[0], value[1], value[2], value[3]


def _required_trace_key(value: object) -> tuple[str, str, int, int]:
    key = _trace_key(value, required=True)
    if key is None:
        raise ReplayWireError("frame key is required")
    return key


def _box(value: object, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ReplayWireError(f"{name} is invalid")
    return value[0], value[1], value[2], value[3]


def _point(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ReplayWireError("bed polygon point is invalid")
    return value[0], value[1]


def _integer(value: dict[str, object], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ReplayWireError(f"{name} must be an integer")
    return item


def _nullable_integer(value: dict[str, object], name: str) -> int | None:
    item = value.get(name)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int):
        raise ReplayWireError(f"{name} must be an integer or null")
    return item


def _number(value: dict[str, object], name: str) -> float:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int | float):
        raise ReplayWireError(f"{name} must be numeric")
    return float(item)


def _text(value: dict[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str):
        raise ReplayWireError(f"{name} must be text")
    return item


def _replay_module_id(request: dict[str, object]) -> str:
    module_id = request["module_id"]
    if not isinstance(module_id, str) or not module_id:
        raise ReplayWireError("module_id is required")
    return module_id


def _replay_run_payload(run: ReplayRun, truncation: dict[str, object]) -> dict[str, object]:
    return {
        "camera_id": run.camera_id,
        "module_qualified_id": run.module_qualified_id,
        "policy_qualified_id": run.policy_qualified_id,
        "effective_policy_id": run.effective_policy_id,
        "event_count": run.event_count,
        "reproducible": run.reproducible,
        "non_reproducible_reason": run.non_reproducible_reason,
        "boot_ids": list(run.boot_ids),
        "truncation": truncation,
        "frames": [
            {
                "frame_key": list(frame.frame_key),
                "analysis_trace_id": frame.analysis_trace_id,
                "events": [
                    {
                        "domain": event.domain,
                        "event_type": event.event_type,
                        "identity": event.identity,
                        "camera_id": event.camera_id,
                        "facility_id": event.facility_id,
                        "time_sec": event.time_sec,
                        "probability": event.probability,
                        "person_id": event.person_id,
                        "bed_id": event.bed_id,
                        "audit": dict(event.audit) if event.audit is not None else None,
                        "snapshot_jpeg": None,
                    }
                    for event in frame.events
                ],
                "snapshots": [
                    {
                        "reason": snapshot.reason,
                        "previous_state": snapshot.previous_state,
                        "current_state": snapshot.current_state,
                        "triggered": snapshot.triggered,
                        "track_id": snapshot.track_id,
                        "bed_id": snapshot.bed_id,
                        "values": dict(snapshot.values),
                        "missing_values": dict(snapshot.missing_values),
                    }
                    for snapshot in frame.snapshots
                ],
            }
            for frame in run.frames
        ],
    }


__all__ = [
    "BED_ZONE_FRAME_TIMEOUT_SECONDS",
    "BedZoneNotFoundError",
    "BedZoneRecognizer",
    "ClipDeletionControl",
    "MjpegProbe",
    "MjpegProbeError",
    "MjpegProbePayload",
    "build_http_server",
]

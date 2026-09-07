"""Relay-protected controls for bounded stored-clip reanalysis."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.interfaces.clip_analysis import ClipAnalysisSupervisor
from worker.pipeline.output.evidence.clip_identity import is_clip_id
from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    parse_manifest_content,
)
from worker.pipeline.output.evidence.manifest_models import ReadyClipManifest

MAX_ANALYSIS_BODY_BYTES = 256


def clip_analysis_path(path: str) -> tuple[str, str] | None:
    parts = path.split("/")
    if len(parts) == 4 and parts[1] == "clips" and parts[3] == "analysis":
        return parts[2], "status"
    if len(parts) == 5 and parts[1] == "clips" and parts[3] == "analysis" and parts[4] == "cancel":
        return parts[2], "cancel"
    return None


def handle_get(
    handler: BaseHTTPRequestHandler,
    clip_id: str,
    *,
    supervisor: ClipAnalysisSupervisor,
    authorized: bool,
) -> None:
    if not authorized:
        handler.send_error(HTTPStatus.FORBIDDEN)
        return
    if not is_clip_id(clip_id):
        handler.send_error(HTTPStatus.BAD_REQUEST)
        return
    status = supervisor.status(clip_id)
    _write_json(handler, HTTPStatus.OK, {"state": status.state, "reason": status.reason})


def handle_post(
    handler: BaseHTTPRequestHandler,
    clip_id: str,
    action: str,
    *,
    store_dir: Path,
    supervisor: ClipAnalysisSupervisor,
    authorized: bool,
) -> None:
    if not authorized:
        handler.send_error(HTTPStatus.FORBIDDEN)
        return
    if not is_clip_id(clip_id):
        handler.send_error(HTTPStatus.BAD_REQUEST)
        return
    if action == "cancel":
        _write_json(handler, HTTPStatus.OK, {"cancelled": supervisor.cancel(clip_id)})
        return
    clip_sha256 = _read_clip_sha256(handler)
    if clip_sha256 is None:
        handler.send_error(HTTPStatus.BAD_REQUEST)
        return
    clip_path = store_dir / "clips" / clip_id / "clip.mp4"
    if not clip_path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    facts = _manifest_facts(clip_path)
    if facts is None:
        handler.send_error(HTTPStatus.BAD_REQUEST)
        return
    manifest_sha256, size_bytes, duration_ms, width, height = facts
    if clip_sha256 != manifest_sha256:
        _write_json(handler, HTTPStatus.BAD_REQUEST, {"error": "sha_mismatch"})
        return
    try:
        accepted = supervisor.trigger(
            clip_id,
            clip_path,
            clip_sha256,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            width=width,
            height=height,
        )
    except ClipAnalysisRejected as exc:
        _write_json(
            handler,
            HTTPStatus.UNPROCESSABLE_ENTITY,
            {"error": "rejected", "reason": str(exc)},
        )
        return
    if not accepted:
        _write_json(handler, HTTPStatus.CONFLICT, {"state": "running"})
        return
    _write_json(handler, HTTPStatus.ACCEPTED, {"state": "running"})


def _read_clip_sha256(handler: BaseHTTPRequestHandler) -> str | None:
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        return None
    try:
        length = int(raw_length)
    except ValueError:
        return None
    if length <= 0 or length > MAX_ANALYSIS_BODY_BYTES:
        return None
    try:
        payload: object = json.loads(handler.rfile.read(length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"clip_sha256"}:
        return None
    sha256 = payload["clip_sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        return None
    return sha256


def _manifest_facts(clip_path: Path) -> tuple[str, int, int, int, int] | None:
    try:
        manifest, _, payload = parse_manifest_content(clip_path.with_name("manifest.json"))
    except ClipEvidenceError:
        return None
    if not isinstance(manifest, ReadyClipManifest) or manifest.clip_id != clip_path.parent.name:
        return None
    dimensions = _manifest_dimensions(manifest, payload)
    if dimensions is None:
        dimensions = _probe_dimensions(clip_path)
    if dimensions is None:
        return None
    width, height = dimensions
    return manifest.sha256, manifest.size_bytes, manifest.duration_ms, width, height


def _manifest_dimensions(
    manifest: ReadyClipManifest, payload: dict[str, object]
) -> tuple[int, int] | None:
    width = payload.get("width")
    height = payload.get("height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
    ):
        return width, height
    if manifest.source_media is None:
        return None
    for stream in manifest.source_media.streams:
        if stream.media_type == "video" and stream.width is not None and stream.height is not None:
            return stream.width, stream.height
    return None


def _probe_dimensions(clip_path: Path) -> tuple[int, int] | None:
    try:
        import av

        with av.open(str(clip_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            stream.thread_count = 1
            if stream.width <= 0 or stream.height <= 0:
                return None
            return stream.width, stream.height
    except Exception:  # noqa: BLE001 - malformed sealed media is a bad request
        return None


def _write_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: object) -> None:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)

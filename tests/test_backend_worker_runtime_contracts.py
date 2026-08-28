"""Provider/consumer drift guard for the two backend<->worker runtime seams.

``backend`` and ``worker`` never import each other (import-linter enforces
it) and each owns its own definition of what crosses the seam: the worker,
as provider, owns the routes it serves and the ``manifest.json`` it writes
(``worker/pipeline/output/live_view_api.py``, ``.../evidence/``); the backend,
as consumer, owns the paths it calls and the parsers it reads with
(``backend/app/features/cameras/*``, ``backend/app/features/clips/*``). There
is deliberately no shared edge-internal contract module -- ``contracts/`` is
the byte-mirrored ML vocabulary (ADR-0006), not an interface package.

This file is the sanctioned meeting point: it imports both packages and
round-trips real provider output through the real consumer parser, so drift
fails here rather than on an edge node.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.app.features.cameras import bed_zone_router, router, streams_router
from backend.app.features.cameras.store import ProbeResult
from backend.app.features.clips import catalog, deletion_control
from backend.app.features.clips.manifest import read_manifest_file
from backend.app.features.clips.store import ClipStore
from worker.pipeline.output import live_view_api
from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationMetadata,
    ClipPublisher,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    EdgeEventId,
    EvidenceReasonCode,
)
from worker.pipeline.output.evidence.manifest_models import (
    ReadyClipManifest,
    UnavailableClipManifest,
)

EVENT_ONE = EdgeEventId("00000000-0000-4000-8000-000000000001")
START = datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC)
ORIGIN = "http://ml-worker:8090"


def _metadata() -> ClipPublicationMetadata:
    return ClipPublicationMetadata(
        camera_id="camera-1",
        event_refs=(EVENT_ONE,),
        event_type="fall",
        clip_start_at=START,
        clip_end_at=START + timedelta(seconds=1),
        finalized_at=START + timedelta(seconds=2),
        started_at=START,
        duration_s=1.0,
        encoder="libx264",
        # The real pipeline always sets this (BusinessEvent.domain is required);
        # the strict backend reader must accept the key.
        domain="fall",
    )


def _publish_unavailable(root: Path) -> Path:
    reservation = ClipReservation(
        ClipId("clip-a"), "camera-1", root / "clips/.staging/clip-a", root / "clips/clip-a"
    )
    published = ClipPublisher(root).publish_unavailable(
        reservation, _metadata(), EvidenceReasonCode.NO_FRAMES
    )
    return published.manifest_path


# --- worker writes manifest.json, backend reads it ---------------------------


def test_worker_manifest_is_served_by_the_backend_lenient_parser(tmp_path: Path) -> None:
    manifest_path = _publish_unavailable(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["domain"] == "fall"

    served = read_manifest_file(manifest_path)
    assert served is not None
    assert served.clip_id == "clip-a"
    assert served.camera_id == "camera-1"
    assert served.event_ref == str(EVENT_ONE)
    assert served.event_type == "fall"
    assert served.started_at == "2026-07-16T01:02:03Z"
    assert served.duration_s == 1.0
    assert served.finalized is True
    assert served.video_available is False
    assert served.path is None
    assert served.video_error == "NO_FRAMES"

    store = ClipStore(tmp_path)
    assert store.get_manifest("clip-a") == served
    assert [m.clip_id for m in store.list_manifests()] == ["clip-a"]


def test_worker_manifest_with_domain_passes_the_backend_strict_reader(tmp_path: Path) -> None:
    manifest_path = _publish_unavailable(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(raw) <= catalog._MANIFEST_FIELDS  # noqa: SLF001
    assert "domain" in raw

    records = catalog.strict_manifest_records(ClipStore(tmp_path))
    assert [record.manifest.clip_id for record in records] == ["clip-a"]
    assert records[0].payload["domain"] == "fall"


def test_worker_manifest_vocabulary_is_within_the_backend_strict_sets() -> None:
    writer_fields = set(ReadyClipManifest.model_fields) | set(UnavailableClipManifest.model_fields)
    assert writer_fields <= catalog._MANIFEST_FIELDS  # noqa: SLF001
    assert ReadyClipManifest.model_fields["manifest_schema_version"].default == (
        catalog._MANIFEST_SCHEMA_VERSION  # noqa: SLF001
    )
    assert UnavailableClipManifest.model_fields["manifest_schema_version"].default == (
        catalog._MANIFEST_SCHEMA_VERSION  # noqa: SLF001
    )
    for state in ("READY", "UNAVAILABLE"):
        assert state in catalog._MANIFEST_STATES  # noqa: SLF001
    assert {code.value for code in EvidenceReasonCode} == catalog._UNAVAILABLE_REASON_CODES  # noqa: SLF001


# --- backend calls the worker's routes ---------------------------------------


def _path(url: str) -> str:
    assert url.startswith(ORIGIN)
    return url[len(ORIGIN) :]


@pytest.mark.parametrize(
    ("backend_path", "worker_match", "identity"),
    [
        (
            lambda i: _path(streams_router._stream_url(ORIGIN, i)),
            live_view_api.stream_camera_id,
            "cam/one two",
        ),  # noqa: SLF001
        (
            lambda i: _path(streams_router._snapshot_url(ORIGIN, i)),
            live_view_api.snapshot_camera_id,
            "cam/one two",
        ),  # noqa: SLF001
        (
            lambda i: _path(streams_router._pose_url(ORIGIN, i)),
            live_view_api.pose_camera_id,
            "cam/one two",
        ),  # noqa: SLF001
        (
            lambda i: _path(bed_zone_router._bed_zone_url(ORIGIN, i)),
            live_view_api.bed_zone_camera_id,
            "cam/one two",
        ),  # noqa: SLF001
        (
            lambda i: deletion_control._clip_path(i, ""),
            live_view_api.clip_deletion_clip_id,
            "clip:a/b",
        ),  # noqa: SLF001
        (
            lambda i: deletion_control._clip_path(i, deletion_control._PREFLIGHT_SUFFIX),
            live_view_api.clip_deletion_preflight_clip_id,
            "clip:a/b",
        ),  # noqa: SLF001
    ],
    ids=["stream", "snapshot", "pose", "bed-zone", "clip-delete", "clip-delete-preflight"],
)
def test_backend_built_paths_are_matched_by_the_worker_route(
    backend_path, worker_match, identity
) -> None:
    path = backend_path(identity)
    assert worker_match(path) == identity
    others = {
        live_view_api.stream_camera_id,
        live_view_api.snapshot_camera_id,
        live_view_api.pose_camera_id,
        live_view_api.bed_zone_camera_id,
        live_view_api.clip_deletion_clip_id,
        live_view_api.clip_deletion_preflight_clip_id,
    } - {worker_match}
    assert all(other(path) is None for other in others)


def test_fixed_routes_headers_and_media_type_agree() -> None:
    assert router.PROBE_PATH == live_view_api.PROBE_PATH == "/probe"
    assert router.RELAY_TOKEN_HEADER == live_view_api.RELAY_TOKEN_HEADER
    assert streams_router._RELAY_TOKEN_HEADER == live_view_api.RELAY_TOKEN_HEADER  # noqa: SLF001
    assert bed_zone_router._RELAY_TOKEN_HEADER == live_view_api.RELAY_TOKEN_HEADER  # noqa: SLF001
    assert deletion_control._RELAY_TOKEN_HEADER == live_view_api.RELAY_TOKEN_HEADER  # noqa: SLF001
    assert streams_router._DEFAULT_MEDIA_TYPE == live_view_api.MJPEG_MEDIA_TYPE  # noqa: SLF001


# --- worker response bodies through the backend parsers ---------------------


def test_probe_response_round_trips_worker_sanitizer_to_backend_reader() -> None:
    raw_success = {
        "ok": True,
        "url": "rtsp://masked",
        "requested_backend": "cpu_av",
        "backend": "cpu_av",
        "width": 640,
        "height": 480,
        "channels": 3,
    }
    wire = json.loads(json.dumps(live_view_api.ProbeResponse.sanitized(raw_success).as_dict()))
    assert wire == {"ok": True, "backend": "cpu_av", "width": 640, "height": 480}
    assert router._probe_result_from_worker(wire) == ProbeResult(  # noqa: SLF001
        ok=True, width=640, height=480
    )

    for raw_class, wire_class in (("auth", "auth"), ("timeout", "timeout"), ("bogus", "decode")):
        wire = live_view_api.ProbeResponse.sanitized(
            {"ok": False, "error_class": raw_class}
        ).as_dict()
        assert wire == {"ok": False, "error_class": wire_class}
        assert router._probe_result_from_worker(wire) == ProbeResult(  # noqa: SLF001
            ok=False, error_class=wire_class
        )
    assert live_view_api.parse_probe_request({"rtsp_url": "rtsp://x"}) == "rtsp://x"


def test_pose_overlay_body_round_trips_both_ways() -> None:
    for mode in ("none", "bedexit", "fall"):
        wire = json.dumps(live_view_api.pose_body(mode)).encode("utf-8")
        assert streams_router._parse_pose_payload(wire).mode == mode  # noqa: SLF001
        assert live_view_api.parse_pose_body(json.loads(json.dumps({"mode": mode}))) == mode
    with pytest.raises(HTTPException):
        streams_router._parse_pose_payload(b'{"mode": "sideways"}')  # noqa: SLF001
    assert live_view_api.parse_pose_body({"mode": "sideways"}) is None
    assert live_view_api.parse_pose_body({"mode": "fall", "extra": 1}) is None


def test_bed_zone_response_round_trips_to_the_backend_parser() -> None:
    bed = live_view_api.BedZoneRecognizeResponse(
        polygon=((1, 2), (3, 2), (3, 4)), image_width=16, image_height=9
    )
    wire = json.dumps(bed.as_dict()).encode("utf-8")
    assert bed_zone_router._parse_worker_payload(wire) == (  # noqa: SLF001
        [[1, 2], [3, 2], [3, 4]],
        16,
        9,
    )
    not_found = json.dumps(live_view_api.BED_ZONE_NOT_FOUND_BODY).encode("utf-8")
    assert bed_zone_router._is_bed_not_found(not_found)  # noqa: SLF001
    assert not bed_zone_router._is_bed_not_found(wire)  # noqa: SLF001

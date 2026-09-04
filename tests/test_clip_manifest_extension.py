from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.features.clips.manifest import read_manifest_file
from worker.pipeline.output.evidence.clip_manifest_payload import manifest_payload
from worker.pipeline.output.evidence.clip_publication_types import ClipPublicationMetadata
from worker.pipeline.output.evidence.evidence_manifest import ReadyClipManifest
from worker.pipeline.output.evidence.manifest_models import ClipExtension, ExtensionContributor

_EVENT_REF = "123e4567-e89b-42d3-a456-426614174000"
_SHA256 = "a" * 64
_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _extension(boundary: str = "extension_bounded") -> ClipExtension:
    return ClipExtension(
        contributors=(
            ExtensionContributor(event_ref=_EVENT_REF, detected_at="2026-01-01T00:00:00Z"),
        ),
        duration_s=30.0,
        boundary=boundary,
    )


def _payload() -> dict[str, object]:
    manifest = ReadyClipManifest(
        clip_id="clip-1",
        camera_id="camera-a",
        event_refs=(_EVENT_REF,),
        clip_start_at="2026-01-01T00:00:00Z",
        clip_end_at="2026-01-01T00:00:10Z",
        finalized_at="2026-01-01T00:00:11Z",
        sha256=_SHA256,
        size_bytes=1,
        duration_ms=10_000,
    )
    metadata = ClipPublicationMetadata(
        camera_id="camera-a",
        event_refs=(_EVENT_REF,),
        event_type="fall",
        clip_start_at=_TIME,
        clip_end_at=_TIME,
        finalized_at=_TIME,
        started_at=_TIME,
        detected_at=_TIME,
        duration_s=10.0,
        encoder="h264",
        extension=_extension(),
    )
    return manifest_payload(manifest, metadata, path="/clips/clip-1.mp4", video_available=True)


def test_worker_extension_payload_round_trips_through_backend_reader(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    manifest = read_manifest_file(path)
    assert manifest is not None
    assert manifest.extension is not None
    assert manifest.extension.boundary == "extension_bounded"
    assert manifest.extension.contributors[0].event_ref == _EVENT_REF


def test_unknown_extension_boundary_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        _extension("not-a-boundary")
    payload = _payload()
    payload["extension"] = {
        "contributors": [{"event_ref": _EVENT_REF, "detected_at": "2026-01-01T00:00:00Z"}],
        "duration_s": 30.0,
        "boundary": "not-a-boundary",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_manifest_file(path) is None

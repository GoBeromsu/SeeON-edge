from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    EvidenceReasonCode,
    ReadyClipManifest,
    UnavailableClipManifest,
    finalize_ready_manifest,
    parse_manifest,
    unavailable_manifest,
    verify_ready_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox import (
    ClipId,
    ClipLocalState,
    EdgeEventId,
    EvidenceOutbox,
    StagedEvent,
)
from worker.pipeline.output.evidence.evidence_reconciliation import reconcile_event_evidence

EVENT_ONE = EdgeEventId("00000000-0000-4000-8000-000000000001")
EVENT_TWO = EdgeEventId("00000000-0000-4000-8000-000000000002")
READY_CLIP_ID = ClipId("clip-ready")
START = datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC)
END = START + timedelta(seconds=1)
FINALIZED = END + timedelta(milliseconds=250)
RUNTIME_MANIFEST_SHA256 = "b" * 64


def _event(edge_event_id: EdgeEventId, sequence: int) -> StagedEvent:
    return StagedEvent(
        edge_event_id=edge_event_id,
        detected_at=f"2026-07-16T00:00:{sequence:02d}Z",
        payload_json=f'{{"sequence":{sequence}}}',
        queued_at=float(sequence),
    )


def _bind(outbox: EvidenceOutbox, clip_id: ClipId, *event_ids: EdgeEventId) -> None:
    for sequence, event_id in enumerate(event_ids, start=1):
        outbox.stage(_event(event_id, sequence))
        outbox.bind_clip(event_id, clip_id)


def _make_video(path: Path, *, codec: str = "libx264", audio: bool = False) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=10:d=1",
    ]
    if audio:
        command.extend(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest"])
    command.extend(
        [
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if not audio:
        command.append("-an")
    command.append(str(path))
    subprocess.run(command, check=True)


def _ready_manifest(video_path: Path, clip_id: ClipId = READY_CLIP_ID) -> ReadyClipManifest:
    return finalize_ready_manifest(
        video_path=video_path,
        clip_id=clip_id,
        camera_id="camera-1",
        event_refs=(EVENT_ONE, EVENT_ONE, EVENT_TWO),
        clip_start_at=START,
        clip_end_at=END,
        finalized_at=FINALIZED,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_manifest_v2_uses_finalized_descriptor_and_exact_ready_contract(tmp_path: Path) -> None:
    # Given: one finalized browser-compatible MP4 and duplicate coalesced refs.
    video_path = tmp_path / "clip.mp4"
    _make_video(video_path)

    # When: the immutable READY manifest is derived from the finalized file.
    manifest = _ready_manifest(video_path)
    payload = manifest.model_dump(mode="json")

    # Then: contract fields are exact, refs are ordered unique, and bytes match.
    required = {
        "clip_id",
        "camera_id",
        "event_refs",
        "clip_start_at",
        "clip_end_at",
        "finalized_at",
        "sha256",
        "size_bytes",
        "mime_type",
        "codec",
        "duration_ms",
        "state_version",
    }
    assert required <= payload.keys()
    assert payload["manifest_schema_version"] == 2
    assert payload["state"] == "READY"
    assert payload["event_refs"] == [EVENT_ONE, EVENT_TWO]
    assert payload["clip_start_at"] == "2026-07-16T01:02:03.000Z"
    assert payload["clip_end_at"] == "2026-07-16T01:02:04.000Z"
    assert payload["finalized_at"] == "2026-07-16T01:02:04.250Z"
    assert payload["sha256"] == hashlib.sha256(video_path.read_bytes()).hexdigest()
    assert payload["size_bytes"] == video_path.stat().st_size
    assert payload["mime_type"] == "video/mp4"
    assert payload["codec"] == "h264"
    assert 900 <= payload["duration_ms"] <= 1100
    assert payload["state_version"] == 2


def test_unavailable_manifest_uses_canonical_milliseconds() -> None:
    manifest = unavailable_manifest(
        clip_id=ClipId("clip-unavailable"),
        camera_id="camera-1",
        event_refs=(EVENT_ONE,),
        clip_start_at=START,
        clip_end_at=END,
        finalized_at=FINALIZED,
        reason_code=EvidenceReasonCode.NO_FRAMES,
    )

    assert manifest.clip_start_at == "2026-07-16T01:02:03.000Z"
    assert manifest.clip_end_at == "2026-07-16T01:02:04.000Z"
    assert manifest.finalized_at == "2026-07-16T01:02:04.250Z"


@pytest.mark.parametrize("invalid", ("B" * 64, "b" * 63, "b" * 65))
def test_manifest_parser_rejects_noncanonical_runtime_manifest_hash(
    tmp_path: Path,
    invalid: str,
) -> None:
    payload = unavailable_manifest(
        clip_id=ClipId("clip-invalid-provenance"),
        camera_id="camera-1",
        event_refs=(EVENT_ONE,),
        clip_start_at=START,
        clip_end_at=END,
        finalized_at=FINALIZED,
        reason_code=EvidenceReasonCode.NO_FRAMES,
    ).model_dump(mode="json")
    payload["runtime_manifest_sha256"] = invalid
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ClipEvidenceError, match="manifest invalid"):
        parse_manifest(manifest_path)


def test_manifest_parser_accepts_shipped_v2_without_runtime_manifest_reference(
    tmp_path: Path,
) -> None:
    legacy = unavailable_manifest(
        clip_id=ClipId("clip-v2-without-runtime-manifest"),
        camera_id="camera-1",
        event_refs=(EVENT_ONE,),
        clip_start_at=START,
        clip_end_at=END,
        finalized_at=FINALIZED,
        reason_code=EvidenceReasonCode.NO_FRAMES,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        legacy.model_dump_json(exclude_none=True),
        encoding="utf-8",
    )

    parsed = parse_manifest(manifest_path)

    assert parsed.manifest_schema_version == 2
    assert parsed.runtime_manifest_sha256 is None


def test_unavailable_manifest_accepts_legacy_utc_precision() -> None:
    manifest = UnavailableClipManifest.model_validate(
        {
            "clip_id": "clip-legacy",
            "camera_id": "camera-1",
            "event_refs": [EVENT_ONE],
            "clip_start_at": "2026-07-16T01:02:03Z",
            "clip_end_at": "2026-07-16T01:02:04.1Z",
            "finalized_at": "2026-07-16T01:02:04.25Z",
            "reason_code": EvidenceReasonCode.NO_FRAMES,
        }
    )

    assert manifest.clip_start_at == "2026-07-16T01:02:03Z"
    assert manifest.clip_end_at == "2026-07-16T01:02:04.1Z"
    assert manifest.finalized_at == "2026-07-16T01:02:04.25Z"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.parametrize("field", ("sha256", "size_bytes"))
def test_verify_ready_manifest_refuses_immutable_byte_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    # Given: a valid manifest whose immutable hash or size is changed.
    video_path = tmp_path / "clip.mp4"
    _make_video(video_path)
    payload = _ready_manifest(video_path).model_dump(mode="json")
    payload[field] = "f" * 64 if field == "sha256" else payload["size_bytes"] + 1
    manifest = ReadyClipManifest.model_validate(payload)

    # When/Then: verification classifies the artifact as corrupt.
    with pytest.raises(ClipEvidenceError) as raised:
        verify_ready_manifest(manifest, video_path)
    assert raised.value.reason_code is EvidenceReasonCode.CORRUPT


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.parametrize(
    ("codec", "audio", "expected_video", "expected_audio"),
    (
        ("mpeg4", False, "mpeg4", None),
        ("libx264", True, "h264", "aac"),
    ),
)
def test_finalize_ready_manifest_records_preserved_source_codecs_and_optional_audio(
    tmp_path: Path,
    codec: str,
    audio: bool,
    expected_video: str,
    expected_audio: str | None,
) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_video(video_path, codec=codec, audio=audio)

    manifest = _ready_manifest(video_path)

    assert manifest.codec == expected_video
    assert manifest.audio_codec == expected_audio


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_reconcile_repairs_manifest_relations_and_persists_verified_outcome(
    tmp_path: Path,
) -> None:
    # Given: a crash left a valid atomic manifest after event staging but before binding.
    store = tmp_path / "store"
    clip_id = ClipId("clip-callback-crash")
    clip_dir = store / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    video_path = clip_dir / "clip.mp4"
    _make_video(video_path)
    manifest = _ready_manifest(video_path, clip_id)
    payload = manifest.model_dump(mode="json")
    payload["runtime_manifest_sha256"] = RUNTIME_MANIFEST_SHA256
    manifest_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    manifest_path = clip_dir / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_content_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    database = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(_event(EVENT_ONE, 1))
        outbox.stage(_event(EVENT_TWO, 2))

        # When: boot reconciliation scans the finalized manifest.
        report = reconcile_event_evidence(store, outbox)
        outcome = outbox.clip_outcome(clip_id)
        references = outbox.ordered_event_ids(clip_id)

    # Then: relations and VERIFIED state survive independently of the callback.
    assert report.verified == 1
    assert outcome is not None
    assert outcome.local_state is ClipLocalState.VERIFIED
    assert outcome.sha256 == manifest.sha256
    assert references == (EVENT_ONE, EVENT_TWO)
    assert parse_manifest(manifest_path).runtime_manifest_sha256 == RUNTIME_MANIFEST_SHA256
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_content_sha256


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_reconcile_marks_mutated_final_media_corrupt_without_rewriting_manifest(
    tmp_path: Path,
) -> None:
    # Given: a valid finalized artifact is mutated after its manifest is committed.
    store = tmp_path / "store"
    clip_id = ClipId("clip-mutated")
    clip_dir = store / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    video_path = clip_dir / "clip.mp4"
    _make_video(video_path)
    manifest = _ready_manifest(video_path, clip_id)
    manifest_bytes = manifest.model_dump_json().encode()
    (clip_dir / "manifest.json").write_bytes(manifest_bytes)
    with video_path.open("ab") as media:
        media.write(b"corrupt")
    database = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database) as outbox:
        _bind(outbox, clip_id, EVENT_ONE, EVENT_TWO)

        # When: restart reconciliation verifies immutable bytes.
        report = reconcile_event_evidence(store, outbox)
        outcome = outbox.clip_outcome(clip_id)

    # Then: local READY is never fabricated and the source manifest is unchanged.
    assert report.corrupt == 1
    assert outcome is not None
    assert outcome.local_state is ClipLocalState.CORRUPT
    assert outcome.unavailable_reason == EvidenceReasonCode.CORRUPT
    assert (clip_dir / "manifest.json").read_bytes() == manifest_bytes


def test_reconcile_marks_malformed_manifest_corrupt_with_durable_relations(
    tmp_path: Path,
) -> None:
    # Given: a clip relation exists but restart finds malformed final metadata.
    store = tmp_path / "store"
    clip_id = ClipId("clip-bad-manifest")
    clip_dir = store / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "manifest.json").write_text("{broken", encoding="utf-8")
    database = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database) as outbox:
        _bind(outbox, clip_id, EVENT_ONE)

        # When: reconciliation parses the corrupt trust boundary.
        report = reconcile_event_evidence(store, outbox)
        outcome = outbox.clip_outcome(clip_id)
        references = outbox.ordered_event_ids(clip_id)

    # Then: the clip is typed CORRUPT and its event relation is preserved.
    assert report.corrupt == 1
    assert outcome is not None
    assert outcome.local_state is ClipLocalState.CORRUPT
    assert references == (EVENT_ONE,)


def test_reconcile_preserves_shipped_v2_manifest_without_runtime_manifest_reference(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    clip_id = ClipId("clip-v2-without-runtime-manifest")
    clip_dir = store / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    manifest = unavailable_manifest(
        clip_id=clip_id,
        camera_id="camera-1",
        event_refs=(EVENT_ONE,),
        clip_start_at=START,
        clip_end_at=END,
        finalized_at=FINALIZED,
        reason_code=EvidenceReasonCode.NO_FRAMES,
    )
    manifest_path = clip_dir / "manifest.json"
    manifest_bytes = manifest.model_dump_json(exclude_none=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    database = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(_event(EVENT_ONE, 1))

        report = reconcile_event_evidence(store, outbox)
        outcome = outbox.clip_outcome(clip_id)

    assert report.unavailable == 1
    assert outcome is not None
    assert outcome.local_state is ClipLocalState.UNAVAILABLE
    assert parse_manifest(manifest_path).runtime_manifest_sha256 is None
    assert manifest_path.read_bytes() == manifest_bytes


def test_reconcile_interrupted_staging_and_missing_artifact_to_unavailable(
    tmp_path: Path,
) -> None:
    # Given: one crash left staging bytes and another left only an outbox placeholder.
    store = tmp_path / "store"
    interrupted = ClipId("clip-interrupted")
    missing = ClipId("clip-missing")
    (store / "clips" / ".staging" / interrupted).mkdir(parents=True)
    database = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database) as outbox:
        _bind(outbox, interrupted, EVENT_ONE)
        _bind(outbox, missing, EVENT_TWO)

        # When: startup reconciles every awaiting clip.
        report = reconcile_event_evidence(store, outbox)
        interrupted_outcome = outbox.clip_outcome(interrupted)
        missing_outcome = outbox.clip_outcome(missing)

    # Then: both become explicit UNAVAILABLE, never VERIFIED/READY.
    assert report.unavailable == 2
    assert interrupted_outcome is not None
    assert interrupted_outcome.local_state is ClipLocalState.UNAVAILABLE
    assert interrupted_outcome.unavailable_reason is EvidenceReasonCode.INTERRUPTED_FINALIZE
    assert missing_outcome is not None
    assert missing_outcome.local_state is ClipLocalState.UNAVAILABLE
    assert missing_outcome.unavailable_reason is EvidenceReasonCode.MISSING


def test_outbox_clip_outcome_survives_restart(tmp_path: Path) -> None:
    # Given: an interrupted clip is reconciled to UNAVAILABLE.
    store = tmp_path / "store"
    clip_id = ClipId("clip-restart")
    database = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database) as outbox:
        _bind(outbox, clip_id, EVENT_ONE)
        reconcile_event_evidence(store, outbox)

    # When: a fresh process reopens the WAL outbox.
    with EvidenceOutbox.open(database) as reopened:
        outcome = reopened.clip_outcome(clip_id)
        references = reopened.ordered_event_ids(clip_id)

    # Then: the typed unavailable outcome and relation remain durable.
    assert outcome is not None
    assert outcome.local_state is ClipLocalState.UNAVAILABLE
    assert outcome.unavailable_reason is EvidenceReasonCode.MISSING
    assert references == (EVENT_ONE,)

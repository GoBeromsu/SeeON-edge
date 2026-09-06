"""Adversarial tests for finalized evidence trust boundaries."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from shared.events.delivery_queue import DeliveryQueue, EventEntry
from worker.pipeline.output.evidence.evidence_manifest import (
    MAX_MANIFEST_BYTES,
    ClipEvidenceError,
    parse_manifest,
)
from worker.pipeline.output.evidence.evidence_media import (
    _media_length_ms,
    inspect_finalized_media,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    EdgeEventId,
    EvidenceReasonCode,
)

EVENT_ONE = EdgeEventId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
EVENT_TWO = EdgeEventId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


def _box(kind: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def _fake_faststart_mp4(marker: bytes) -> bytes:
    return _box(b"ftyp", b"isom") + _box(b"moov") + _box(b"mdat", marker)


def _manifest_payload(state: str, event_refs: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_schema_version": 2,
        "state": state,
        "clip_id": "clip-trust",
        "camera_id": "camera-1",
        "event_refs": event_refs,
        "clip_start_at": "2026-07-16T01:02:03Z",
        "clip_end_at": "2026-07-16T01:02:04Z",
        "finalized_at": "2026-07-16T01:02:04.25Z",
        "state_version": 2,
    }
    if state == "READY":
        payload.update(
            {
                "sha256": "a" * 64,
                "size_bytes": 100,
                "mime_type": "video/mp4",
                "codec": "h264",
                "duration_ms": 1000,
            }
        )
    else:
        payload["reason_code"] = "MISSING"
    return payload


def test_media_probe_uses_same_open_inode_when_path_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an attacker can replace the pathname while ffprobe is starting.
    media = tmp_path / "clip.mp4"
    replacement = tmp_path / "replacement.mp4"
    original_bytes = _fake_faststart_mp4(b"original-inode")
    replacement_bytes = _fake_faststart_mp4(b"replacement-inode")
    media.write_bytes(original_bytes)
    replacement.write_bytes(replacement_bytes)
    observed_probe_bytes: list[bytes] = []
    descriptor_root = "/proc/self/fd" if Path("/proc/self/fd").is_dir() else "/dev/fd"

    def _swap_during_probe(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        probe_target = command[-1]
        os.replace(replacement, media)
        observed_probe_bytes.append(Path(probe_target).read_bytes())
        assert probe_target.startswith(f"{descriptor_root}/")
        descriptor = int(probe_target.rsplit("/", maxsplit=1)[1])
        assert kwargs.get("pass_fds") == (descriptor,)
        stdout = json.dumps(
            {
                "streams": [{"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p"}],
                "format": {"duration": "1.000"},
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_media.subprocess.run", _swap_during_probe
    )

    # When: immutable facts are collected across hash and codec validation.
    facts = inspect_finalized_media(media)

    # Then: all facts came from the original open inode, not the swapped path.
    assert observed_probe_bytes == [original_bytes]
    assert facts.sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert facts.size_bytes == len(original_bytes)
    assert media.read_bytes() == replacement_bytes


def test_parse_manifest_rejects_external_symlink(tmp_path: Path) -> None:
    # Given: a valid-looking manifest is outside the trusted clip directory.
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps(_manifest_payload("UNAVAILABLE", [EVENT_ONE])),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.symlink_to(external)

    # When/Then: parsing does not follow the symlink.
    with pytest.raises(ClipEvidenceError) as raised:
        parse_manifest(manifest)
    assert raised.value.reason_code is EvidenceReasonCode.CORRUPT


def test_parse_manifest_rejects_oversized_valid_json(tmp_path: Path) -> None:
    # Given: an otherwise valid manifest exceeds the bounded trust input.
    payload = _manifest_payload("UNAVAILABLE", [EVENT_ONE])
    payload["padding"] = "x" * MAX_MANIFEST_BYTES
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    # When/Then: the parser rejects it before unbounded JSON processing.
    with pytest.raises(ClipEvidenceError) as raised:
        parse_manifest(manifest)
    assert raised.value.reason_code is EvidenceReasonCode.CORRUPT


@pytest.mark.parametrize("state", ("READY", "UNAVAILABLE"))
@pytest.mark.parametrize(
    "event_refs",
    (
        ["not-a-uuid"],
        ["aaaaaaaa-aaaa-1aaa-8aaa-aaaaaaaaaaaa"],
        ["AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"],
        [""],
        [EVENT_ONE, EVENT_ONE],
    ),
)
def test_parse_manifest_rejects_noncanonical_or_duplicate_event_refs(
    tmp_path: Path,
    state: str,
    event_refs: list[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_payload(state, event_refs)), encoding="utf-8")

    with pytest.raises(ClipEvidenceError) as raised:
        parse_manifest(manifest)
    assert raised.value.reason_code is EvidenceReasonCode.CORRUPT


def test_conflicting_event_admission_preserves_the_original_durable_fact(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    original = EventEntry(
        str(EVENT_ONE), "fall", "2026-07-16T01:02:03Z", "camera-a", "facility-a", b"{}", b"{}"
    )
    conflict = EventEntry(
        str(EVENT_ONE), "fall", "2026-07-16T01:02:04Z", "camera-a", "facility-a", b"{}", b"{}"
    )
    assert queue.try_admit(original).accepted
    refused = queue.try_admit(conflict)

    assert not refused.accepted
    assert tuple(queue.entries()) == (next(DeliveryQueue(tmp_path).entries()),)


@contextlib.contextmanager
def _passthrough_scope():
    """A context manager that does not suppress — it only re-raises."""
    yield


def test_evidence_error_survives_contextlib_reraise_with_message_intact() -> None:
    """Regression guard for frozen+slots on an Exception subclass (CPython 3.13).

    ``dataclass(slots=True)`` cannot add ``__slots__`` to an existing class, so it
    builds a replacement class object. The ``__setattr__`` that ``frozen=True``
    generates closes over the *old* class and calls ``super(cls, self)``, while
    ``self`` is an instance of the *new* class. ``contextlib`` trips this on
    re-raise because it assigns ``exc.__traceback__``, and the operator then sees
    ``TypeError: super(type, obj)`` instead of the real failure reason.

    Dropping ``slots=True`` (keeping ``frozen=True``) is the fix. This test fails
    loudly if ``slots=True`` is reintroduced on the evidence exceptions.
    """
    error = ClipEvidenceError(EvidenceReasonCode.FINALIZE_FAILED, "ffprobe unavailable")

    with pytest.raises(ClipEvidenceError) as captured, _passthrough_scope():
        raise error

    # The original exception object propagates, not a TypeError from __setattr__.
    assert captured.value is error
    assert type(captured.value) is ClipEvidenceError
    assert captured.value.reason_code is EvidenceReasonCode.FINALIZE_FAILED
    assert captured.value.detail == "ffprobe unavailable"
    assert str(captured.value) == "FINALIZE_FAILED: ffprobe unavailable"


def test_a_clip_whose_timeline_starts_late_is_not_called_corrupt() -> None:
    """Container duration is measured from zero; a clip's length is not.

    These are the exact numbers ffprobe returned for a real clip on this
    deployment: the camera's uptime origin puts the first sample 8236.957s into
    the timeline, so ``format.duration`` reads 8270.974s while the footage runs
    34.016s. Measured against the 120-second ceiling, that marked every clip
    the system produced CORRUPT, which suppressed the manifest, which kept
    every playable clip out of the catalogue -- the operator saw nothing while
    good footage sat on disk.
    """
    payload = {
        "streams": [{"codec_type": "video", "codec_name": "hevc", "duration": "34.016000"}],
        "format": {"duration": "8270.974000", "start_time": "8236.957000"},
    }

    assert _media_length_ms(payload, payload["streams"]) == 34_016


def test_the_length_survives_a_container_that_reports_no_stream_duration() -> None:
    """Some containers omit stream duration; the origin must still be removed."""
    payload = {
        "streams": [{"codec_type": "video", "codec_name": "hevc"}],
        "format": {"duration": "8270.974000", "start_time": "8236.957000"},
    }

    assert _media_length_ms(payload, payload["streams"]) == 34_017


def test_a_zero_origin_container_keeps_its_own_duration() -> None:
    """The ordinary case must not be distorted by the origin correction."""
    payload = {
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
        "format": {"duration": "34.000000", "start_time": "0.000000"},
    }

    assert _media_length_ms(payload, payload["streams"]) == 34_000

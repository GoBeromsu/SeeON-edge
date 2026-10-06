"""Route-owned descriptor acquisition; no PostgreSQL or HTTP delivery claim."""

import errno
import hashlib
import os
import stat
from types import SimpleNamespace

import pytest
from fastapi import Request

from backend.app.features.evidence import router as evidence
from backend.app.features.evidence.receipt_store import ArtifactReceiptVerificationError


def _input(tmp_path):
    root = tmp_path / "clip-store"
    media = root / "clips" / "clip-1" / "clip.mp4"
    media.parent.mkdir(parents=True)
    content = b"synthetic verified media"
    media.write_bytes(content)
    request = Request(
        {"type": "http", "app": SimpleNamespace(state=SimpleNamespace(clip_store_root=root))}
    )
    payload = evidence.ReadyClipPayload(
        state="READY",
        camera_id="camera-1",
        facility_id="facility-1",
        event_refs=["00000000-0000-4000-8000-000000000001"],
        state_version=1,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        mime_type="video/mp4",
        codec="h264",
        duration_ms=1000,
        clip_start_at="2026-07-06T00:00:00Z",
        clip_end_at="2026-07-06T00:00:01Z",
        finalized_at="2026-07-06T00:00:02Z",
    )
    return request, payload, media, content


def _observe(monkeypatch):
    opened, handles = [], []
    real_open, real_fdopen, real_fstat = os.open, os.fdopen, os.fstat

    def open_file(path, flags, *args, **kwargs):
        if path == "clip.mp4":
            # A flag regression fails before opening a writer-less FIFO; no
            # timing assertion or hung test is needed for that negative control.
            assert flags & os.O_NONBLOCK
        descriptor = real_open(path, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def wrap(descriptor, *args, **kwargs):
        handle = real_fdopen(descriptor, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(evidence.os, "open", open_file)
    monkeypatch.setattr(evidence.os, "fdopen", wrap)
    return opened, handles, real_fstat


def _closed(opened, real_fstat):
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError) as caught:
            real_fstat(descriptor)
        assert caught.value.errno == errno.EBADF


def test_success_transfers_only_verified_media_to_caller(tmp_path, monkeypatch):
    request, payload, _, content = _input(tmp_path)
    opened, handles, real_fstat = _observe(monkeypatch)
    verified = evidence._verified_media(request, "clip-1", payload)
    try:
        assert len(opened) == 4 and handles == [verified.handle]
        _closed(opened[:-1], real_fstat)
        assert stat.S_ISREG(real_fstat(verified.handle.fileno()).st_mode)
        assert verified.handle.read() == content
        assert (verified.sha256, verified.size_bytes) == (payload.sha256, len(content))
    finally:
        verified.handle.close()
    _closed(opened, real_fstat)


@pytest.mark.parametrize("phase", ["fdopen", "fstat", "hash"])
@pytest.mark.parametrize("cancel", [False, True])
def test_acquisition_fault_closes_every_owned_descriptor(tmp_path, monkeypatch, phase, cancel):
    request, payload, _, _ = _input(tmp_path)
    opened, handles, real_fstat = _observe(monkeypatch)
    error = (
        KeyboardInterrupt("descriptor cancellation") if cancel else OSError("descriptor failure")
    )

    def fail(*args, **kwargs):
        raise error

    def fail_file_stat(descriptor):
        value = real_fstat(descriptor)
        if stat.S_ISREG(value.st_mode):
            raise error
        return value

    original_hash = evidence.verified_artifact

    def hash_with_read_failure(handle):
        class InterruptedReader:
            def fileno(self):
                return handle.fileno()

            def seek(self, *args):
                return handle.seek(*args)

            def read(self, *args):
                raise error

        return original_hash(InterruptedReader())

    if phase == "fdopen":
        monkeypatch.setattr(evidence.os, "fdopen", fail)
    elif phase == "fstat":
        monkeypatch.setattr(evidence.os, "fstat", fail_file_stat)
    else:
        monkeypatch.setattr(evidence, "verified_artifact", hash_with_read_failure)
    expected = KeyboardInterrupt if cancel else ArtifactReceiptVerificationError
    with pytest.raises(expected) as caught:
        evidence._verified_media(request, "clip-1", payload)
    if cancel:
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error
    assert len(opened) == 4 and all(handle.closed for handle in handles)
    assert len(handles) == (0 if phase == "fdopen" else 1)
    _closed(opened, real_fstat)


@pytest.mark.parametrize("kind", ["fifo", "directory", "hash_mismatch"])
def test_nonregular_or_changed_media_closes_handle(tmp_path, monkeypatch, kind):
    request, payload, media, _ = _input(tmp_path)
    if kind == "hash_mismatch":
        media.write_bytes(b"different synthetic bytes")
    else:
        media.unlink()
        if kind == "fifo":
            os.mkfifo(media)
        else:
            media.mkdir()
    opened, handles, real_fstat = _observe(monkeypatch)
    with pytest.raises(ArtifactReceiptVerificationError):
        evidence._verified_media(request, "clip-1", payload)
    assert all(handle.closed for handle in handles)
    _closed(opened, real_fstat)


@pytest.mark.parametrize("component", ["clips", "clip-1", "clip.mp4"])
def test_every_path_component_remains_nofollow(tmp_path, monkeypatch, component):
    request, payload, media, _ = _input(tmp_path)
    path = (
        media
        if component == "clip.mp4"
        else (media.parent if component == "clip-1" else media.parent.parent)
    )
    moved = path.with_name(path.name + "-actual")
    path.rename(moved)
    path.symlink_to(moved.name, target_is_directory=component != "clip.mp4")
    opened, handles, real_fstat = _observe(monkeypatch)
    with pytest.raises(ArtifactReceiptVerificationError, match="artifact is unavailable"):
        evidence._verified_media(request, "clip-1", payload)
    assert not handles
    _closed(opened, real_fstat)

from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from worker.pipeline.output.evidence.snapshot_store import SnapshotConflictError, SnapshotStore

# This file's use of the fd table is *instrumentation only* -- it resolves a
# descriptor back to a path to assert fsync/replace ordering. ``snapshot_store.py``
# itself never touches ``/proc``, so there is no reason for these tests to be
# Linux-only.
#
# ``/proc/self/fd/N`` is a symlink on Linux, so ``os.readlink`` resolves it.
# macOS has no ``/proc``; its ``/dev/fd/N`` is a character device, not a
# symlink, so ``readlink`` fails with ``EINVAL`` -- which is why these tests
# used to fail here. ``fcntl(F_GETPATH)`` is the macOS way to ask the same
# question, and ``/dev/fd`` still enumerates the process's descriptors.
#
# Note this is *not* the same situation as `tests/test_clip_recorder.py`, where
# `/proc/self/fd` is used by production code
# (`worker/pipeline/output/evidence/evidence_media.py`) to hand ffprobe a
# TOCTOU-safe reference to an already-open inode. That one is a genuine
# Linux-only runtime dependency and its tests pin that floor deliberately.

_FD_DIR = Path("/proc/self/fd") if Path("/proc/self/fd").exists() else Path("/dev/fd")
_DescriptorIdentity = tuple[int, int, int]


def _path_of_fd(descriptor: int) -> str:
    """Resolve an open descriptor back to its path, on Linux or macOS."""
    if _FD_DIR == Path("/proc/self/fd"):
        return os.readlink(f"/proc/self/fd/{descriptor}")
    raw: bytes = fcntl.fcntl(descriptor, fcntl.F_GETPATH, bytes(1024))
    return raw.rstrip(b"\x00").decode()


def _store(store: SnapshotStore, **overrides: object) -> object:
    values: dict[str, object] = {
        "snapshot_id": "event-1",
        "captured_at": "2026-07-28T12:00:00Z",
        "camera_id": "camera-1",
        "edge_event_id": "event-1",
    }
    values.update(overrides)
    return store.store(b"jpeg-bytes", **values)  # type: ignore[arg-type]


def test_snapshot_store_is_idempotent_for_one_event(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    first = _store(store)
    second = _store(store)

    assert first == second
    assert list(tmp_path.rglob("*.jpg")) == [tmp_path / first.path]  # type: ignore[union-attr]


def test_snapshot_store_fsyncs_each_file_before_replace_and_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)
    calls: list[tuple[str, str, str | None]] = []
    replace = os.replace
    fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        target = _path_of_fd(descriptor)
        calls.append(("fsync", target, None))
        fsync(descriptor)

    def record_replace(source: str, destination: str, **kwargs: object) -> None:
        directory = Path(_path_of_fd(int(kwargs["src_dir_fd"])))  # type: ignore[call-overload]
        calls.append(("replace", str(directory / source), str(directory / destination)))
        replace(source, destination, **kwargs)

    monkeypatch.setattr("worker.pipeline.output.evidence.snapshot_files.os.fsync", record_fsync)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.snapshot_files.os.replace", record_replace
    )

    _store(store)

    replacements = [
        (index, source, destination)
        for index, (operation, source, destination) in enumerate(calls)
        if operation == "replace" and destination is not None
    ]
    assert len(replacements) == 4
    for index, source, destination in replacements:
        if source.endswith(".tmp"):
            assert ("fsync", source, None) in calls[:index]
        assert ("fsync", str(Path(destination).parent), None) in calls[index + 1 :]


def test_snapshot_store_rejects_same_id_with_different_bytes(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    _store(store)

    with pytest.raises(SnapshotConflictError, match="event-1.*existing bytes"):
        store.store(
            b"second-jpeg",
            snapshot_id="event-1",
            captured_at="2026-07-28T12:00:00Z",
            camera_id="camera-1",
            edge_event_id="event-1",
        )

    stored = next(tmp_path.rglob("*.jpg"))
    assert stored.read_bytes() == b"jpeg-bytes"


def test_snapshot_store_rejects_identity_rebinding(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    _store(store)

    with pytest.raises(SnapshotConflictError, match="event-1"):
        _store(store, camera_id="camera-2")


def test_snapshot_store_commits_identity_before_blob_and_prevents_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)
    original = SnapshotStore._write_atomic

    def interrupt_blob(self: SnapshotStore, destination: Path, content: bytes) -> None:
        if destination.suffix == ".jpg":
            raise OSError("blob write interrupted")
        original(self, destination, content)

    monkeypatch.setattr(SnapshotStore, "_write_atomic", interrupt_blob)
    with pytest.raises(OSError, match="blob write interrupted"):
        _store(store)

    with pytest.raises(SnapshotConflictError):
        _store(store, camera_id="camera-2")

    monkeypatch.setattr(SnapshotStore, "_write_atomic", original)
    assert _store(store)


@pytest.mark.parametrize("failure_at", range(1, 14))
def test_snapshot_store_retries_interrupted_directory_barriers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    store = SnapshotStore(tmp_path)
    original = store._fsync_directory
    calls = 0
    failed = False

    def interrupt_once(descriptor: int) -> None:
        nonlocal calls, failed
        calls += 1
        if calls == failure_at:
            failed = True
            raise OSError("directory barrier interrupted")
        original(descriptor)

    monkeypatch.setattr(SnapshotStore, "_fsync_directory", staticmethod(interrupt_once))
    with pytest.raises(OSError, match="directory barrier interrupted"):
        _store(store)

    assert _store(store)
    assert failed


def test_snapshot_store_retries_post_replace_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)
    original = store._fsync_directory
    failed = False

    def fail_once(descriptor: int) -> None:
        nonlocal failed
        directory = Path(_path_of_fd(descriptor))
        if directory.name == "2026-07-28" and not failed:
            failed = True
            raise OSError("durability barrier interrupted")
        original(descriptor)

    monkeypatch.setattr(SnapshotStore, "_fsync_directory", staticmethod(fail_once))
    with pytest.raises(OSError, match="durability barrier interrupted"):
        _store(store)

    assert _store(store)
    assert failed


@pytest.mark.parametrize("case", range(20))
def test_snapshot_store_does_not_leak_descriptors_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: int
) -> None:
    owned_descriptors: dict[int, _DescriptorIdentity] = {}
    opened: set[_DescriptorIdentity] = set()
    closed: set[_DescriptorIdentity] = set()
    real_open = os.open
    real_close = os.close
    real_open_store_root = SnapshotStore._open_store_root

    def identity(descriptor: int) -> _DescriptorIdentity:
        descriptor_stat = os.fstat(descriptor)
        return descriptor, descriptor_stat.st_dev, descriptor_stat.st_ino

    def record_store_root(self: SnapshotStore) -> int:
        descriptor = real_open_store_root(self)
        resource = identity(descriptor)
        owned_descriptors[descriptor] = resource
        opened.add(resource)
        return descriptor

    def record_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd in owned_descriptors:
            resource = identity(descriptor)
            owned_descriptors[descriptor] = resource
            opened.add(resource)
        return descriptor

    def record_close(descriptor: int) -> None:
        resource = owned_descriptors.pop(descriptor, None)
        if resource is not None:
            assert identity(descriptor) == resource
        real_close(descriptor)
        if resource is not None:
            closed.add(resource)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("forced directory fsync failure")

    monkeypatch.setattr(SnapshotStore, "_open_store_root", record_store_root)
    monkeypatch.setattr("worker.pipeline.output.evidence.snapshot_files.os.open", record_open)
    monkeypatch.setattr("worker.pipeline.output.evidence.snapshot_files.os.close", record_close)
    monkeypatch.setattr(SnapshotStore, "_fsync_directory", staticmethod(fail_fsync))

    with pytest.raises(OSError, match="forced directory fsync failure"):
        _store(SnapshotStore(tmp_path / str(case)))

    assert opened
    assert closed == opened


def test_snapshot_store_tolerates_concurrent_shared_directory_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(2)
    shared_root = tmp_path / "snapshot-store"
    original_mkdir = os.mkdir

    def racing_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        if path == shared_root or path == ".snapshot-locks":
            barrier.wait(timeout=5)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr("worker.pipeline.output.evidence.snapshot_files.os.mkdir", racing_mkdir)
    stores = [SnapshotStore(shared_root), SnapshotStore(shared_root)]

    def write(index: int) -> object:
        return _store(
            stores[index],
            snapshot_id=f"event-{index}",
            edge_event_id=f"event-{index}",
            camera_id=f"camera-{index}",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, range(2)))

    assert len(results) == 2
    assert len(list(shared_root.rglob("*.jpg"))) == 2


def test_snapshot_store_serializes_same_identity_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = SnapshotStore._write_atomic
    calls = 0

    def delayed(self: SnapshotStore, destination: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if destination.suffix == ".jpg":
            entered.set()
            assert release.wait(timeout=2)
        original(self, destination, content)

    monkeypatch.setattr(SnapshotStore, "_write_atomic", delayed)
    results: list[object] = []

    def save() -> None:
        results.append(_store(store))

    first = threading.Thread(target=save)
    second = threading.Thread(target=save)
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert len(results) == 2
    assert calls == 2  # JPEG plus immutable metadata; the second caller did not write either.


def test_snapshot_store_serializes_conflicting_identity_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = SnapshotStore._write_atomic

    def delayed(self: SnapshotStore, destination: Path, content: bytes) -> None:
        if destination.suffix == ".json":
            entered.set()
            assert release.wait(timeout=2)
        original(self, destination, content)

    monkeypatch.setattr(SnapshotStore, "_write_atomic", delayed)
    results: list[object] = []

    def save(content: bytes) -> None:
        try:
            results.append(
                store.store(
                    content,
                    snapshot_id="event-1",
                    captured_at="2026-07-28T12:00:00Z",
                    camera_id="camera-1",
                    edge_event_id="event-1",
                )
            )
        except SnapshotConflictError as exc:
            results.append(exc)

    first = threading.Thread(target=save, args=(b"first",))
    second = threading.Thread(target=save, args=(b"second",))
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert sum(isinstance(result, SnapshotConflictError) for result in results) == 1
    assert sum(not isinstance(result, SnapshotConflictError) for result in results) == 1
    assert len(list(tmp_path.rglob("*.jpg"))) == 1


def test_snapshot_store_uses_bounded_lock_shards(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    for index in range(300):
        store.store(
            f"jpeg-{index}".encode(),
            snapshot_id=f"event-{index}",
            captured_at="2026-07-28T12:00:00Z",
            camera_id="camera-1",
            edge_event_id=f"event-{index}",
        )

    lock_files = list((tmp_path / ".snapshot-locks").glob("*.lock"))
    assert len(lock_files) <= 256


def test_snapshot_store_preserves_primary_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)

    def interrupted(source: str, destination: str, **kwargs: object) -> None:
        del source, destination, kwargs
        raise OSError("replace interrupted")

    def cleanup_failed(name: str, *, dir_fd: int) -> None:
        del name, dir_fd
        raise OSError("cleanup interrupted")

    monkeypatch.setattr(
        "worker.pipeline.output.evidence.snapshot_files.os.replace", interrupted
    )
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.snapshot_files.os.unlink", cleanup_failed
    )

    with pytest.raises(OSError, match="replace interrupted") as raised:
        _store(store)
    assert raised.value.__notes__ == ["temporary snapshot cleanup failed: cleanup interrupted"]

@pytest.mark.parametrize(
    "path_kind",
    [
        "lock_directory",
        "lock_file",
        "identity_directory",
        "metadata_file",
        "snapshots_directory",
        "camera_directory",
        "date_directory",
        "destination_file",
    ],
)
def test_snapshot_store_rejects_symlinked_internal_paths_without_outside_writes(
    tmp_path: Path, path_kind: str
) -> None:
    store = SnapshotStore(tmp_path)
    relative = store._relative_path("camera-1", "2026-07-28T12:00:00Z", "event-1")
    identity_key = hashlib.sha256(b"event-1").hexdigest()
    outside = tmp_path.parent / f"{tmp_path.name}-{path_kind}-outside"
    outside.mkdir()
    target = outside / "must-not-be-written"
    lock_shard = identity_key[:2]

    if path_kind == "lock_directory":
        (tmp_path / ".snapshot-locks").symlink_to(target)
    elif path_kind == "lock_file":
        lock_directory = tmp_path / ".snapshot-locks"
        lock_directory.mkdir()
        (lock_directory / f"{lock_shard}.lock").symlink_to(target)
    elif path_kind == "identity_directory":
        (tmp_path / ".snapshot-identities").symlink_to(target)
    elif path_kind == "metadata_file":
        identity_directory = tmp_path / ".snapshot-identities"
        identity_directory.mkdir()
        (identity_directory / f"{identity_key}.json").symlink_to(target)
    elif path_kind == "snapshots_directory":
        (tmp_path / "snapshots").symlink_to(target)
    elif path_kind == "camera_directory":
        snapshots = tmp_path / "snapshots"
        snapshots.mkdir()
        (snapshots / relative.parts[1]).symlink_to(target)
    elif path_kind == "date_directory":
        camera = tmp_path / "snapshots" / relative.parts[1]
        camera.mkdir(parents=True)
        (camera / relative.parts[2]).symlink_to(target)
    else:
        destination_directory = tmp_path / relative.parent
        destination_directory.mkdir(parents=True)
        (destination_directory / relative.name).symlink_to(target)

    with pytest.raises((OSError, ValueError)):
        _store(store)

    assert not target.exists()


def test_snapshot_store_rejects_symlinked_temporary_file_without_outside_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)
    identity_key = hashlib.sha256(b"event-1").hexdigest()
    outside = tmp_path.parent / f"{tmp_path.name}-temporary-outside"
    outside.mkdir()
    target = outside / "must-not-be-written"
    staging_directory = tmp_path / ".snapshot-staging"
    staging_directory.mkdir(parents=True)

    class KnownUuid:
        hex = "known"

    monkeypatch.setattr(
        "worker.pipeline.output.evidence.snapshot_files.uuid4", lambda: KnownUuid()
    )
    (staging_directory / f".{identity_key}.json.known.tmp").symlink_to(target)

    with pytest.raises(FileExistsError):
        _store(store)

    assert not target.exists()

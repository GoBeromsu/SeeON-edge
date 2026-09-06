from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from worker.pipeline.decision.event_identity import EventIdentityStore, EventIdentityStoreError

MAX_JOURNAL_BYTES = 16 * 1024 * 1024
RETENTION_SEC = 90 * 24 * 3600


@dataclass(slots=True)
class _Clock:
    now: float

    def __call__(self) -> float:
        return self.now


def _line(source_key: str, edge_event_id: str, recorded_at: float) -> str:
    return json.dumps(
        {
            "source_key": source_key,
            "edge_event_id": edge_event_id,
            "recorded_at": recorded_at,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def test_recent_identity_is_reused_across_store_restart(tmp_path: Path) -> None:
    path = tmp_path / "identities.jsonl"
    clock = _Clock(1_000.0)
    first = EventIdentityStore(path, clock=clock)
    event_id = first.resolve("camera-a:fall:1")

    restarted = EventIdentityStore(path, clock=_Clock(1_000.0 + RETENTION_SEC - 1))

    assert restarted.resolve("camera-a:fall:1") == event_id
    assert UUID(event_id).version == 4


def test_entries_older_than_ninety_days_are_removed(tmp_path: Path) -> None:
    path = tmp_path / "identities.jsonl"
    old_id = str(uuid4())
    recent_id = str(uuid4())
    path.write_text(
        "\n".join(
            [
                _line("old", old_id, 0.0),
                _line("recent", recent_id, RETENTION_SEC),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store = EventIdentityStore(path, clock=_Clock(RETENTION_SEC + 1))

    assert store.resolve("recent") == recent_id
    assert store.resolve("old") != old_id
    retained = path.read_text(encoding="utf-8")
    assert old_id not in retained
    assert recent_id in retained


def test_flood_journal_compacts_to_max_bytes_keeping_newest(tmp_path: Path) -> None:
    path = tmp_path / "identities.jsonl"
    clock = _Clock(10_000.0)
    payload = []
    for index in range(400):
        key = f"k-{index:04d}-{'x' * 80_000}"
        payload.append(_line(key, str(uuid4()), float(index)))
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")
    assert path.stat().st_size > MAX_JOURNAL_BYTES

    store = EventIdentityStore(path, clock=clock)
    newest_key = f"k-0399-{'x' * 80_000}"
    oldest_key = f"k-0000-{'x' * 80_000}"

    assert path.stat().st_size <= MAX_JOURNAL_BYTES
    assert store.resolve(newest_key) == json.loads(payload[-1])["edge_event_id"]
    assert store.resolve(oldest_key) != json.loads(payload[0])["edge_event_id"]


@pytest.mark.parametrize(
    "bad_line",
    [
        "{not-json",
        '{"source_key":"truncated","edge_event_id":"',
        _line("valid", str(uuid4()), 60.0),
        _line("future", str(uuid4()), 10_000.0),
    ],
)
def test_mixed_valid_and_invalid_journal_fails_closed_without_rewrite(
    tmp_path: Path, bad_line: str
) -> None:
    path = tmp_path / "identities.jsonl"
    valid_id = str(uuid4())
    original = "\n".join([_line("valid", valid_id, 50.0), bad_line]) + "\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(EventIdentityStoreError):
        EventIdentityStore(path, clock=_Clock(100.0))

    assert path.read_text(encoding="utf-8") == original
    assert valid_id in original


def test_malformed_only_journal_fails_closed_without_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "identities.jsonl"
    original = "{ this is not valid json\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(EventIdentityStoreError):
        EventIdentityStore(path, clock=_Clock(100.0))

    assert path.read_text(encoding="utf-8") == original


def test_legacy_untimestamped_line_is_retained(tmp_path: Path) -> None:
    path = tmp_path / "identities.jsonl"
    event_id = str(uuid4())
    path.write_text(
        json.dumps({"source_key": "legacy", "edge_event_id": event_id}) + "\n",
        encoding="utf-8",
    )

    store = EventIdentityStore(path, clock=_Clock(100.0))

    assert store.resolve("legacy") == event_id


def test_concurrent_appends_are_stable_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "identities.jsonl"
    store = EventIdentityStore(path, clock=_Clock(100.0))
    barrier = threading.Barrier(17)
    keys = [f"cam:{index}" for index in range(16)]

    def resolve(key: str) -> None:
        barrier.wait()
        store.resolve(key)

    threads = [threading.Thread(target=resolve, args=(key,)) for key in keys]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    restarted = EventIdentityStore(path, clock=_Clock(100.0))
    for key in keys:
        first = store.resolve(key)
        assert restarted.resolve(key) == first
    assert path.stat().st_size <= MAX_JOURNAL_BYTES


@pytest.mark.parametrize("seam", ["write", "fsync", "replace", "dir_fsync"])
def test_interrupted_compaction_leaves_complete_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str
) -> None:
    path = tmp_path / "identities.jsonl"
    clock = _Clock(100.0)
    store = EventIdentityStore(path, clock=clock)
    original_id = store.resolve("keep")
    original = path.read_bytes()

    if seam == "write":

        def boom(*_args: object, **_kwargs: object) -> int:
            raise OSError("write failed")

        monkeypatch.setattr("worker.pipeline.decision.event_identity._write_text", boom)
    elif seam == "fsync":
        real_fsync = os.fsync

        def boom_fsync(fd: int) -> None:
            raise OSError("fsync failed")

        monkeypatch.setattr(os, "fsync", boom_fsync)
        del real_fsync
    elif seam == "replace":

        def boom_replace(*_args: object, **_kwargs: object) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(os, "replace", boom_replace)
    else:
        calls = {"n": 0}
        real_fsync = os.fsync

        def counted_fsync(fd: int) -> None:
            calls["n"] += 1
            if calls["n"] >= 2:
                raise OSError("dir fsync failed")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", counted_fsync)

    with pytest.raises(EventIdentityStoreError):
        store.resolve("new")
    monkeypatch.undo()

    leftover = [
        child
        for child in path.parent.iterdir()
        if child.suffix == ".tmp" or child.name.startswith(f".{path.name}.")
    ]
    if seam == "dir_fsync":
        restarted = EventIdentityStore(path, clock=_Clock(100.0))
        assert restarted.resolve("keep") == original_id
        assert path.read_bytes() != original
        assert path.stat().st_size > 0
    else:
        assert path.read_bytes() == original
        restarted = EventIdentityStore(path, clock=_Clock(100.0))
        assert restarted.resolve("keep") == original_id
    assert leftover == []

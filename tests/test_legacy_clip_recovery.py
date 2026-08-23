from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import patch

import pytest

from backend.app.edge_db.legacy_clip_recovery import (
    LegacyClipRecovery,
    LegacyClipStoreUnavailableError,
)
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import MIGRATIONS, SchemaV17MigrationError
from backend.app.features.clips import consistency_ops
from backend.app.features.clips.consistency_ops import ClipConsistencyError, FinalizedClipEvidence


def _recovery_cli() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/ops/recover-legacy-clips.py"
    spec = importlib.util.spec_from_file_location("recover_legacy_clips", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema16_database(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:16])
    return database


def _insert_clip(database: Path, clip_id: str, *, publish_state: str = "WAITING") -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, publish_state) VALUES (?, ?)",
            (clip_id, publish_state),
        )


def _ready_manifest(clip_id: str, media: bytes) -> dict[str, object]:
    return {
        "manifest_schema_version": 2,
        "state": "READY",
        "clip_id": clip_id,
        "camera_id": "cam-1",
        "event_refs": ["11111111-1111-4111-8111-111111111111"],
        "event_ref": "11111111-1111-4111-8111-111111111111",
        "started_at": "2026-01-01T00:00:00Z",
        "clip_start_at": "2026-01-01T00:00:00Z",
        "clip_end_at": "2026-01-01T00:00:01Z",
        "finalized_at": "2026-01-01T00:00:02Z",
        "duration_s": 1.0,
        "path": f"clips/{clip_id}/clip.mp4",
        "finalized": True,
        "video_available": True,
        "state_version": 2,
        "sha256": hashlib.sha256(media).hexdigest(),
        "size_bytes": len(media),
        "mime_type": "video/mp4",
        "codec": "h264",
        "duration_ms": 1000,
    }


def _mounted_store(root: Path) -> Path:
    """Build a structurally real clip store.

    `<root>/clips/.staging` is the marker the clip consistency ops treat as
    authoritative. Recovery refuses without it, because a mistyped or unmounted
    root would otherwise be indistinguishable from a store whose media is gone
    and would classify every clip UNAVAILABLE, opening the migration gate.
    """
    (root / "clips" / ".staging").mkdir(parents=True, exist_ok=True)
    return root


def _write_manifest(store: Path, clip_id: str, media: bytes, *, corrupt: bool = False) -> None:
    _mounted_store(store)
    directory = store / "clips" / clip_id
    directory.mkdir(parents=True)
    (directory / "clip.mp4").write_bytes(media)
    payload = _ready_manifest(clip_id, media)
    if corrupt:
        payload["sha256"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_legacy_clip_recovery_verifies_intact_media_and_terminalizes_publication(
    tmp_path: Path,
) -> None:
    database = _schema16_database(tmp_path)
    clip_id = "clip-intact"
    _insert_clip(database, clip_id)
    store = tmp_path / "clip-store"
    _write_manifest(store, clip_id, b"complete-media")

    result = LegacyClipRecovery(database, store).run()

    assert result.verified == 1
    assert result.unavailable == result.corrupt == result.unresolved == 0
    assert result.publication_terminalized == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT local_state, sha256, publish_state, last_error_code FROM evidence_clips"
        ).fetchone() == (
            "VERIFIED",
            hashlib.sha256(b"complete-media").hexdigest(),
            "PERMANENT",
            "LEGACY_CLIP_PUBLICATION_UNSUPPORTED",
        )


def test_legacy_clip_recovery_classifies_missing_and_corrupt_media_without_deletion(
    tmp_path: Path,
) -> None:
    database = _schema16_database(tmp_path)
    _insert_clip(database, "clip-missing")
    _insert_clip(database, "clip-corrupt")
    store = tmp_path / "clip-store"
    _write_manifest(store, "clip-corrupt", b"damaged-media", corrupt=True)

    result = LegacyClipRecovery(database, store).run()

    assert (result.unavailable, result.corrupt, result.unresolved) == (1, 1, 0)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evidence_clips").fetchone() == (2,)
        assert connection.execute(
            "SELECT clip_id, local_state, unavailable_reason FROM evidence_clips ORDER BY clip_id"
        ).fetchall() == [
            ("clip-corrupt", "CORRUPT", "CORRUPT"),
            ("clip-missing", "UNAVAILABLE", "MISSING"),
        ]


def test_legacy_clip_recovery_terminalizes_abandoned_inflight_publication(
    tmp_path: Path,
) -> None:
    database = _schema16_database(tmp_path)
    _insert_clip(database, "clip-inflight", publish_state="IN_FLIGHT")

    assert (
        _recovery_cli().main(
            ["--database", str(database), "--clip-store", str(_mounted_store(tmp_path))]
        )
        == 0
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT publish_state, last_error_code, publish_lease_owner, "
            "publish_lease_expires_at FROM evidence_clips"
        ).fetchone() == (
            "PERMANENT",
            "LEGACY_CLIP_PUBLISHER_RETIRED",
            None,
            None,
        )


def test_legacy_clip_recovery_unblocks_real_schema17_migrator(tmp_path: Path) -> None:
    database = _schema16_database(tmp_path)
    clip_id = "clip-missing"
    _insert_clip(database, clip_id)

    with pytest.raises(SchemaV17MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        migrate_database(database)

    assert (
        LegacyClipRecovery(database, _mounted_store(tmp_path / "clip-store")).run().unresolved == 0
    )
    assert migrate_database(database).current_version == 17


def test_a_wrong_clip_store_root_refuses_before_mutating_anything(tmp_path: Path) -> None:
    """A mistyped --clip-store must not write off every clip.

    Without this precondition a wrong or unmounted root is indistinguishable
    from a store whose media is genuinely gone: every AWAITING_FINALIZE clip is
    classified UNAVAILABLE, its publication terminalized, the command exits 0,
    and the forward-only schema-17 migration proceeds behind it. One typo would
    discard all 1053 live clips.
    """
    database = _schema16_database(tmp_path)
    for clip_id in ("clip:a", "clip:b"):
        _insert_clip(database, clip_id)

    with pytest.raises(LegacyClipStoreUnavailableError):
        LegacyClipRecovery(database, tmp_path / "typo-not-a-store").run()

    with sqlite3.connect(database) as connection:
        states = connection.execute(
            "SELECT local_state, COUNT(*) FROM evidence_clips GROUP BY 1"
        ).fetchall()
    assert states == [("AWAITING_FINALIZE", 2)], "recovery mutated rows despite refusing"


def test_a_mounted_but_empty_store_is_distinguished_from_a_wrong_root(tmp_path: Path) -> None:
    """A real store whose media is genuinely gone must still be processable."""
    database = _schema16_database(tmp_path)
    _insert_clip(database, "clip:a")

    result = LegacyClipRecovery(database, _mounted_store(tmp_path / "store")).run()

    assert result.verified == 0
    assert result.unavailable == 1
    assert result.unresolved == 0


def test_a_mid_scan_read_fault_refuses_without_mutating_any_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _schema16_database(tmp_path)
    store = tmp_path / "clip-store"
    for clip_id in ("clip-a", "clip-b"):
        _insert_clip(database, clip_id)
        _write_manifest(store, clip_id, clip_id.encode())

    with sqlite3.connect(database) as connection:
        before = connection.execute("SELECT * FROM evidence_clips ORDER BY clip_id").fetchall()

    original_inspect = consistency_ops._inspect_finalized_directory_at

    def fail_mid_scan_read(
        directory: Path, directory_fd: int, uid: int
    ) -> tuple[object, tuple[str, ...]]:
        if directory.name == "clip-b":
            raise ClipConsistencyError("final_read_error", "simulated EIO")
        return cast(
            tuple[object, tuple[str, ...]],
            original_inspect(directory, directory_fd, uid),
        )

    monkeypatch.setattr(consistency_ops, "_inspect_finalized_directory_at", fail_mid_scan_read)

    assert (
        _recovery_cli().main(["--database", str(database), "--clip-store", str(store)]) == 3
    )

    with sqlite3.connect(database) as connection:
        after = connection.execute("SELECT * FROM evidence_clips ORDER BY clip_id").fetchall()
    assert after == before


def test_the_cli_reports_a_distinct_status_for_an_unavailable_store(tmp_path: Path) -> None:
    """Exit 3 must not be confused with exit 2 'ran and found work remaining'."""
    database = _schema16_database(tmp_path)
    _insert_clip(database, "clip:a")

    exit_code = _recovery_cli().main(
        ["--database", str(database), "--clip-store", str(tmp_path / "typo")]
    )

    assert exit_code == 3


def test_cli_reports_partial_completion_when_transitional_recovery_loses_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = _schema16_database(tmp_path)
    clip_id = "clip-intact"
    _insert_clip(database, clip_id)
    store = tmp_path / "clip-store"
    _write_manifest(store, clip_id, b"complete-media")

    from backend.app.edge_db.legacy_transitional_recovery import LegacyTransitionalRecovery

    def _lose_store(self: object) -> object:
        raise LegacyClipStoreUnavailableError("store lost after clip recovery")

    monkeypatch.setattr(LegacyTransitionalRecovery, "run", _lose_store)

    assert _recovery_cli().main(["--database", str(database), "--clip-store", str(store)]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "corrupt": 0,
        "detail": "store lost after clip recovery",
        "error": "clip_store_unavailable_after_clip_recovery",
        "publication_terminalized": 1,
        "unavailable": 0,
        "verified": 1,
    }
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT local_state, publish_state FROM evidence_clips WHERE clip_id = ?",
            (clip_id,),
        ).fetchone() == ("VERIFIED", "PERMANENT")


def test_a_mount_lost_mid_scan_refuses_instead_of_writing_off_the_rest(
    tmp_path: Path,
) -> None:
    """A mount that vanishes partway must not become 1053 UNAVAILABLE verdicts.

    The marker check only guards the start. If the store disappears during the
    scan, later `lstat()` calls raise FileNotFoundError, which
    `inspect_finalized_clip` legitimately reads as `missing` -- so every
    remaining clip would be committed UNAVAILABLE and the migration gate opened,
    on evidence nothing ever examined. Store identity is therefore revalidated
    immediately before the write transaction.
    """
    database = _schema16_database(tmp_path)
    for clip_id in ("clip:a", "clip:b"):
        _insert_clip(database, clip_id)
    store = _mounted_store(tmp_path / "store")

    before = _clip_rows(database)
    real_lstat = Path.lstat
    seen: list[str] = []

    def vanishing(self: Path, *args: object, **kwargs: object) -> object:
        if self.name.startswith("clip:"):
            seen.append(self.name)
            if len(seen) > 1:  # the mount disappears after the first clip
                raise FileNotFoundError(2, "No such file or directory")
        return real_lstat(self, *args, **kwargs)

    with (
        patch.object(Path, "lstat", vanishing),
        patch.object(
            LegacyClipRecovery,
            "_require_same_store",
            side_effect=LegacyClipStoreUnavailableError("mount vanished mid-scan"),
        ),
        pytest.raises(LegacyClipStoreUnavailableError),
    ):
        LegacyClipRecovery(database, store).run()

    assert _clip_rows(database) == before, "recovery recorded verdicts despite refusing"


def _clip_rows(database: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(database) as connection:
        return list(
            connection.execute(
                "SELECT clip_id, local_state, publish_state FROM evidence_clips ORDER BY 1"
            )
        )


def test_a_transient_mount_loss_that_restores_the_same_inode_still_refuses(
    tmp_path: Path,
) -> None:
    """Identity equality at two instants does not prove continuous presence.

    If the store vanishes while clips are classified `missing` and the very same
    mount returns before the post-scan check, an `(st_dev, st_ino)` comparison
    matches and every verdict gathered during the gap commits. Recovery
    therefore holds an open directory descriptor for the whole scan, so a
    disappearance is detectable even when the identity is restored.
    """
    database = _schema16_database(tmp_path)
    for clip_id in ("clip:a", "clip:b"):
        _insert_clip(database, clip_id)
    store = _mounted_store(tmp_path / "store")
    before = _clip_rows(database)

    marker = store / "clips" / ".staging"
    real_is_dir = Path.is_dir

    def marker_gone(self: Path, *args: object, **kwargs: object) -> bool:
        if self == marker:
            return False  # the mount was absent when the scan finished
        return bool(real_is_dir(self, *args, **kwargs))

    with (
        patch.object(Path, "is_dir", marker_gone),
        pytest.raises(LegacyClipStoreUnavailableError),
    ):
        LegacyClipRecovery(database, store).run()

    assert _clip_rows(database) == before


def test_descriptor_scan_does_not_commit_missing_when_store_path_is_replaced_mid_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Classifications remain bound to the held directory even when its path moves."""
    database = _schema16_database(tmp_path)
    store = tmp_path / "store"
    for clip_id in ("clip:a", "clip:b"):
        _insert_clip(database, clip_id)
        _write_manifest(store, clip_id, clip_id.encode())

    clips = store / "clips"
    detached = store / "clips-detached"
    replacement = store / "clips-replacement"
    original_classify = LegacyClipRecovery._classify
    calls = 0

    def replace_path_mid_scan(
        recovery: LegacyClipRecovery, clip_id: str, handle: int
    ) -> tuple[str, str | None, FinalizedClipEvidence | None]:
        nonlocal calls
        outcome = original_classify(recovery, clip_id, handle)
        calls += 1
        if calls == 1:
            os.rename(clips, detached)
            replacement.mkdir()
            (replacement / ".staging").mkdir()
            os.rename(replacement, clips)
        elif calls == 2:
            os.rename(clips, replacement)
            os.rename(detached, clips)
        return outcome

    monkeypatch.setattr(LegacyClipRecovery, "_classify", replace_path_mid_scan)

    result = LegacyClipRecovery(database, store).run()

    assert (result.verified, result.unavailable, result.corrupt) == (2, 0, 0)

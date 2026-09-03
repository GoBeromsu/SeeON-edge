from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips import catalog as catalog_module
from backend.app.features.clips.catalog import (
    SCHEMA_VERSION,
    CatalogConflictError,
    CatalogSchemaNewerThanSupportedError,
    CatalogStore,
    get_catalog_store,
)
from backend.app.features.clips.store import ClipStore
from backend.app.shared.dashboard_credentials import DashboardCredentialsStore

_EVENT_ID = "11111111-1111-4111-8111-111111111111"


def _ready_manifest(clip_id: str = "clip-1") -> dict[str, object]:
    return {
        "manifest_schema_version": 2,
        "clip_id": clip_id,
        "camera_id": "cam-1",
        "event_ref": _EVENT_ID,
        "event_refs": [_EVENT_ID],
        "started_at": "2026-01-01T00:00:00Z",
        "clip_start_at": "2026-01-01T00:00:00Z",
        "clip_end_at": "2026-01-01T00:00:01Z",
        "finalized_at": "2026-01-01T00:00:02Z",
        "duration_s": 1.0,
        "codec": "h264",
        "path": f"clips/{clip_id}/clip.mp4",
        "finalized": True,
        "video_available": True,
        "duration_ms": 1000,
        "sha256": "a" * 64,
        "size_bytes": 1,
        "mime_type": "video/mp4",
        "state": "READY",
        "state_version": 2,
    }


def _write_manifest(root, payload: dict[str, object]) -> None:
    path = root / "clips" / str(payload["clip_id"]) / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_catalog_from_env_resolves_under_state_dir_and_is_queryable(tmp_path, monkeypatch) -> None:
    """``from_env()`` resolves through the central database seam."""
    catalog_path = tmp_path / "clip-catalog.sqlite3"
    monkeypatch.setattr(catalog_module, "EDGE_DATABASE_PATH", catalog_path)

    store = CatalogStore.from_env()
    try:
        assert store.record("clips", "clip-1", {"clip_id": "clip-1", "camera_id": "cam-1"})
        assert store.list_clips() == [{"camera_id": "cam-1", "clip_id": "clip-1"}]
    finally:
        store.close()

    assert catalog_path.exists()


def test_catalog_open_failure_is_recorded_without_raising(monkeypatch) -> None:
    app = FastAPI()

    def unavailable(_: object) -> CatalogStore:
        raise PermissionError("catalog mount unavailable")

    monkeypatch.setattr(CatalogStore, "open", unavailable)

    assert get_catalog_store(app) is None
    assert "catalog mount unavailable" in app.state.catalog_error


def test_catalog_open_refuses_newer_schema_without_mutation(tmp_path) -> None:
    """Mirrors the worker-side ``test_newer_schema_is_refused_without_mutation``
    (``tests/test_evidence_outbox.py``): opening a catalog written by a newer
    ml-api must refuse rather than silently rewriting/corrupting it."""
    catalog_path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(catalog_path) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    before = os.stat(catalog_path).st_size

    with pytest.raises(CatalogSchemaNewerThanSupportedError) as raised:
        CatalogStore.open(catalog_path)

    assert raised.value.found == SCHEMA_VERSION + 1
    assert raised.value.supported == SCHEMA_VERSION
    assert os.stat(catalog_path).st_size == before


def test_catalog_downgrade_degrades_instead_of_crashing_startup(tmp_path, monkeypatch) -> None:
    """The parallel case to the worker's ``NewerSchemaVersionError`` handling
    in ``_STORE_UNAVAILABLE_ERRORS`` (``worker/runtime/faults/record.py``,
    ``worker/runtime/config/lkg_store.py``): a downgraded ml-api binary
    pointed at a newer ``catalog.sqlite3`` must degrade at startup instead of
    crashing, matching ``get_catalog_store``'s existing OSError/sqlite3.Error
    degradation behavior."""
    catalog_path = tmp_path / "catalog.sqlite3"
    monkeypatch.setattr(catalog_module, "EDGE_DATABASE_PATH", catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(catalog_path) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    app = FastAPI()

    assert get_catalog_store(app) is None
    assert "catalog unavailable" in app.state.catalog_error


def test_catalog_queries_promoted_clip_columns_in_sql(tmp_path) -> None:
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        for clip_id, camera_id, event_type, started_at in (
            ("clip-1", "cam-1", "fall", "2026-01-01T00:00:00Z"),
            ("clip-2", "cam-1", "bed-exit", "2026-01-02T00:00:00Z"),
            ("clip-3", "cam-2", "fall", "2026-01-03T00:00:00Z"),
        ):
            store.record(
                "clips",
                clip_id,
                {
                    "clip_id": clip_id,
                    "camera_id": camera_id,
                    "event_type": event_type,
                    "started_at": started_at,
                    "path": f"clips/{clip_id}/clip.mp4",
                },
            )
        assert [row["clip_id"] for row in store.list_clips("cam-1")] == ["clip-2", "clip-1"]
        assert [
            row["clip_id"]
            for row in store.list_clips(
                started_at_from="2026-01-02T00:00:00Z", started_at_to="2026-01-03T00:00:00Z"
            )
        ] == ["clip-3", "clip-2"]
        assert [row["clip_id"] for row in store.list_clips(event_type="fall")] == [
            "clip-3",
            "clip-1",
        ]
        plan = store._connection.execute(
            "EXPLAIN QUERY PLAN SELECT clip_id FROM clips WHERE camera_id = ? AND started_at >= ?",
            ("cam-1", "2026-01-01T00:00:00Z"),
        ).fetchall()
        assert any("clips_camera_started_at_idx" in row[-1] for row in plan)
    finally:
        store.close()


@pytest.mark.parametrize(
    "detected_at",
    [None, "2026-01-01T00:00:00Z"],
)
def test_strict_manifest_tolerates_both_detected_at_generations(
    tmp_path, detected_at: str | None
) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    if detected_at is not None:
        payload["detected_at"] = detected_at
    _write_manifest(root, payload)
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        [record] = catalog_module.strict_manifest_records(ClipStore(root))
    finally:
        store.close()

    assert record.manifest.detected_at == detected_at


def test_strict_manifest_rejects_non_utc_detected_at(tmp_path) -> None:
    root = tmp_path / "clip-store"
    payload = {**_ready_manifest(), "detected_at": "2026-01-01T00:00:00+00:00"}
    _write_manifest(root, payload)

    with pytest.raises(TypeError, match="invalid manifest timestamp"):
        catalog_module.strict_manifest_records(ClipStore(root))


def test_catalog_migrates_legacy_payload_schema_losslessly_and_idempotently(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    payload = {
        "clip_id": "clip-1",
        "camera_id": "cam-1",
        "event_type": "fall",
        "started_at": "2026-01-01T00:00:00Z",
        "path": "clips/clip-1",
        "sha256": "abc",
        "size_bytes": 12,
        "mime_type": "video/mp4",
        "encoder": "ffmpeg",
        "state": "finalized",
    }
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE clips (clip_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL) STRICT"
    )
    for table, key in (
        ("snapshots", "snapshot_id"),
        ("events", "edge_event_id"),
        ("labels", "clip_id"),
        ("audit", "audit_id"),
        ("cameras", "camera_id"),
    ):
        connection.execute(
            f"CREATE TABLE {table} ({key} TEXT PRIMARY KEY, payload_json TEXT NOT NULL) STRICT"
        )
    connection.execute(
        "INSERT INTO clips VALUES (?, ?)",
        ("clip-1", json.dumps(payload, sort_keys=True, separators=(",", ":"))),
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    store = CatalogStore.open(path)
    try:
        row = store._connection.execute(
            "SELECT camera_id, event_type, started_at, path, sha256, size_bytes, "
            "mime_type, encoder, state, payload_json FROM clips"
        ).fetchone()
        assert row[:-1] == (
            "cam-1",
            "fall",
            "2026-01-01T00:00:00Z",
            "clips/clip-1",
            "abc",
            12,
            "video/mp4",
            "ffmpeg",
            "finalized",
        )
        assert json.loads(row[-1]) == payload
    finally:
        store.close()
    store = CatalogStore.open(path)
    store.close()


def test_catalog_conflict_preserves_existing_record(tmp_path) -> None:
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        payload = {"clip_id": "clip-1", "camera_id": "cam-1", "path": "one.mp4"}
        assert store.record("clips", "clip-1", payload)
        with pytest.raises(CatalogConflictError):
            store.record("clips", "clip-1", {**payload, "path": "two.mp4"})
        assert store.records("clips") == [payload]
    finally:
        store.close()


def test_catalog_records_with_columns_returns_sql_values_with_payload(tmp_path) -> None:
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    payload = {
        "snapshot_id": "snapshot-1",
        "camera_id": "cam-1",
        "edge_event_id": _EVENT_ID,
        "captured_at": "2026-01-01T00:00:00Z",
        "path": "snapshots/cam-1/snapshot.jpg",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "mime_type": "image/jpeg",
    }
    try:
        store.record("snapshots", "snapshot-1", payload)
        [record] = store.records_with_columns("snapshots")
        assert record.key == "snapshot-1"
        assert record.columns == {
            "camera_id": "cam-1",
            "edge_event_id": _EVENT_ID,
            "captured_at": "2026-01-01T00:00:00Z",
            "path": "snapshots/cam-1/snapshot.jpg",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "mime_type": "image/jpeg",
        }
        assert record.payload == payload
    finally:
        store.close()


def test_catalog_uncommitted_transaction_is_not_recovered_as_a_record(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = CatalogStore.open(path)
    store._connection.execute("BEGIN IMMEDIATE")
    store._connection.execute(
        "INSERT INTO clips (clip_id, payload_json) VALUES (?, ?)", ("uncommitted", "{}")
    )
    store._connection.close()

    reopened = CatalogStore.open(path)
    try:
        assert reopened.integrity_check() == "ok"
        assert reopened.records("clips") == []
    finally:
        reopened.close()


def test_catalog_backfill_is_idempotent_and_preserves_raw_manifest(tmp_path) -> None:
    clip_root = tmp_path / "clip-store"
    payload = {
        **_ready_manifest(),
        "event_type": "fall",
        "encoder": "ffmpeg",
        "size_bytes": 12,
    }
    _write_manifest(clip_root, payload)
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        store.backfill(ClipStore(clip_root))
        store.backfill(ClipStore(clip_root))
        assert store.records("clips") == [payload]
    finally:
        store.close()


def test_catalog_backfill_rejects_corrupt_or_mismatched_sidecars(tmp_path) -> None:
    root = tmp_path / "clip-store"
    bad = root / "clips" / "bad" / "manifest.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises(ValueError, match="unable to read manifest"):
            store.backfill(ClipStore(root))
        assert store.records("clips") == []
    finally:
        store.close()


def test_catalog_backfill_rejects_directory_clip_id_mismatch(tmp_path) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    path = root / "clips" / "wrong-id" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises(ValueError, match="does not match directory"):
            store.backfill(ClipStore(root))
    finally:
        store.close()


def test_catalog_connections_are_thread_local_and_concurrent(tmp_path) -> None:
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    errors: list[Exception] = []

    def write(index: int) -> None:
        try:
            store.record("clips", f"clip-{index}", {"clip_id": f"clip-{index}"})
        except sqlite3.Error as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert errors == []
        assert len(store.records("clips")) == 8
        assert os.stat(tmp_path / "catalog.sqlite3").st_mode & 0o777 == 0o600
    finally:
        store.close()


def test_catalog_batch_rolls_back_when_later_record_conflicts(tmp_path) -> None:
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        store.record("events", "event-1", {"edge_event_id": "event-1", "camera_id": "cam-1"})
        with pytest.raises(CatalogConflictError):
            store.record_many(
                (
                    ("snapshots", "snapshot-1", {"snapshot_id": "snapshot-1"}),
                    ("events", "event-1", {"edge_event_id": "event-1", "camera_id": "cam-2"}),
                )
            )
        assert store.records("snapshots") == []
    finally:
        store.close()


def test_camera_backfill_projects_only_allowed_scalar_fields_and_excludes_secrets(tmp_path) -> None:
    class Registry:
        def snapshot(self):
            return {
                "cameras": [
                    {
                        "id": "cam-1",
                        "backend_camera_id": "backend-cam-1",
                        "label": "Camera",
                        "decode_backend": "nvdec",
                        "created_at": "2026-01-01T00:00:00Z",
                        "mapping_pending": False,
                        "rtsp_url": ("rtsp://operator:fixture-password@example.test/live"),
                        "username": "operator",
                        "password": "fixture-password",
                        "meta": {"auth": {"token": "synthetic-token"}},
                        "creds": [{"password": "DEEP-PASSWORD"}],
                        "future_field": "ignored",
                    }
                ]
            }

    path = tmp_path / "catalog.sqlite3"
    store = CatalogStore.open(path)
    try:
        store.backfill(ClipStore(tmp_path / "empty"), camera_registry=Registry())
        assert store.records("cameras") == [
            {
                "id": "cam-1",
                "backend_camera_id": "backend-cam-1",
                "label": "Camera",
                "decode_backend": "nvdec",
                "created_at": "2026-01-01T00:00:00Z",
                "mapping_pending": False,
            }
        ]
        for database_path in (path, path.with_name(f"{path.name}-wal")):
            assert database_path.exists()
            encoded = database_path.read_bytes()
            # Checked as VALUES, not generic key-name substrings: since issue
            # #35's SQLite consolidation, the shared catalog.sqlite3 file
            # legitimately has schema-level `username`/`password_hash`
            # columns (the unrelated `credentials` table), so bare "user" or
            # "password" now appear in every fresh database's own DDL text
            # regardless of any camera secret leak. `rtsp_url` and `secret`
            # aren't used as column/table names anywhere, so those two key
            # names are still meaningful literal checks.
            for secret in (
                b"rtsp_url",
                b"secret",
                b"NESTED-SECRET-42",
                b"DEEP-PASSWORD",
                b"operator",
                b"fixture-password",
                b"synthetic-token",
            ):
                assert secret not in encoded
    finally:
        store.close()


@pytest.mark.parametrize(
    "changes",
    (
        {"state": "anything"},
        {"finalized": False},
    ),
)
def test_catalog_backfill_rejects_non_evidence_manifest_states(tmp_path, changes) -> None:
    root = tmp_path / "clip-store"
    payload = {**_ready_manifest(), **changes}
    _write_manifest(root, payload)
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises(ValueError):
            store.backfill(ClipStore(root))
        assert store.records("clips") == []
    finally:
        store.close()


def test_catalog_backfill_requires_manifest_schema_v2(tmp_path) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    payload.pop("manifest_schema_version")
    _write_manifest(root, payload)
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises(ValueError, match="missing manifest schema"):
            store.backfill(ClipStore(root))
    finally:
        store.close()


@pytest.mark.parametrize(
    ("duration_s", "raises"),
    ((0, False), (-0.001, True)),
)
def test_catalog_backfill_requires_non_negative_duration(tmp_path, duration_s, raises) -> None:
    root = tmp_path / "clip-store"
    payload = {**_ready_manifest(), "duration_s": duration_s}
    _write_manifest(root, payload)
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        if raises:
            with pytest.raises(ValueError, match="invalid manifest duration"):
                store.backfill(ClipStore(root))
        else:
            store.backfill(ClipStore(root))
            assert store.records("clips") == [payload]
    finally:
        store.close()


def test_catalog_backfill_rejects_symlinked_manifest_file(tmp_path: Path) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    outside = tmp_path / "outside-manifest.json"
    outside.write_text(json.dumps(payload), encoding="utf-8")
    manifest = root / "clips" / str(payload["clip_id"]) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.symlink_to(outside)

    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            store.backfill(ClipStore(root))
    finally:
        store.close()


def test_catalog_backfill_rejects_symlinked_clip_directory(tmp_path: Path) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    outside = tmp_path / "outside-clip"
    outside.mkdir()
    (outside / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    clips = root / "clips"
    clips.mkdir(parents=True)
    (clips / str(payload["clip_id"])).symlink_to(outside, target_is_directory=True)

    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises(ValueError, match="contains a symlink"):
            store.backfill(ClipStore(root))
    finally:
        store.close()


def test_catalog_backfill_rejects_symlinked_clips_root(tmp_path: Path) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    outside = tmp_path / "outside-clips"
    manifest = outside / str(payload["clip_id"]) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    root.mkdir()
    (root / "clips").symlink_to(outside, target_is_directory=True)

    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises(ValueError, match="clips directory must not be a symlink"):
            store.backfill(ClipStore(root))
    finally:
        store.close()


def test_catalog_migration_from_v1_adds_v3_tables_and_promotes_columns(tmp_path) -> None:
    """Schema-version-3 migration (PR E, issue #35): a database still at
    version 1 (pre-column-promotion -- only ``key``/``payload_json`` columns,
    matching ``test_catalog_migrates_legacy_payload_schema_losslessly_and_idempotently``'s
    shape) must land at ``user_version == 3`` with BOTH the column-promotion
    ALTERs applied AND the three new v3 tables present. The single-dispatch
    ``_migrate`` only ever runs the branch matching the *found* version once,
    so a v1-origin database would silently skip the v3 tables unless the
    ``elif version == 1:`` branch also executes ``_V3_TABLE_STATEMENTS``."""
    path = tmp_path / "legacy-v1.sqlite3"
    camera_payload = {
        "camera_id": "cam-1",
        "label": "Lobby",
        "decode_backend": "software",
    }
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE clips (clip_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL) STRICT"
    )
    for table, key in (
        ("snapshots", "snapshot_id"),
        ("events", "edge_event_id"),
        ("labels", "clip_id"),
        ("audit", "audit_id"),
        ("cameras", "camera_id"),
    ):
        connection.execute(
            f"CREATE TABLE {table} ({key} TEXT PRIMARY KEY, payload_json TEXT NOT NULL) STRICT"
        )
    connection.execute(
        "INSERT INTO cameras VALUES (?, ?)",
        ("cam-1", json.dumps(camera_payload, sort_keys=True, separators=(",", ":"))),
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    store = CatalogStore.open(path)
    try:
        version = store._connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == 3

        # (b) the version==1 column-promotion ALTERs ran: the real columns
        # (not just payload_json) are populated.
        row = store._connection.execute(
            "SELECT camera_id, label, decode_backend, payload_json FROM cameras"
        ).fetchone()
        assert row[:-1] == ("cam-1", "Lobby", "software")
        assert json.loads(row[-1]) == camera_payload

        # (c) all three new v3 tables exist and are empty.
        tables = {
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"credentials", "camera_registry"} <= tables
        assert "runtime_latency" not in tables
        for table in ("credentials", "camera_registry"):
            count = store._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0, f"expected {table} to start empty, found {count} row(s)"

        # (d) legacy data untouched (beyond the expected column promotion).
        assert store.records("cameras") == [camera_payload]
    finally:
        store.close()

    # Idempotent: reopening an already-migrated database is a no-op.
    reopened = CatalogStore.open(path)
    reopened.close()


def test_catalog_migration_from_v2_adds_v3_tables_without_touching_legacy_cameras_table(
    tmp_path,
) -> None:
    """Schema-version-3 migration (PR E, issue #35): a database already at
    version 2 (the pre-existing 6-table schema) must gain the three new
    ``credentials``/``camera_registry``/``runtime_latency`` tables on open,
    while its existing data -- including the pre-existing ``cameras`` cache
    table, a completely distinct table from the new ``camera_registry`` --
    is left untouched."""
    path = tmp_path / "legacy-v2.sqlite3"
    camera_payload = {
        "camera_id": "cam-1",
        "label": "Lobby",
        "decode_backend": "software",
    }
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE clips (clip_id TEXT PRIMARY KEY, camera_id TEXT, event_type TEXT, "
        "state TEXT, started_at TEXT, path TEXT, sha256 TEXT, size_bytes INTEGER, "
        "mime_type TEXT, encoder TEXT, payload_json TEXT NOT NULL) STRICT"
    )
    connection.execute(
        "CREATE TABLE snapshots (snapshot_id TEXT PRIMARY KEY, camera_id TEXT, "
        "edge_event_id TEXT, captured_at TEXT, path TEXT, sha256 TEXT, "
        "size_bytes INTEGER, mime_type TEXT, payload_json TEXT NOT NULL) STRICT"
    )
    connection.execute(
        "CREATE TABLE events (edge_event_id TEXT PRIMARY KEY, camera_id TEXT, "
        "event_type TEXT, detected_at TEXT, clip_id TEXT, payload_json TEXT NOT NULL) STRICT"
    )
    connection.execute(
        "CREATE TABLE labels (clip_id TEXT PRIMARY KEY, label TEXT, reviewer TEXT, "
        "reviewed_at TEXT, payload_json TEXT NOT NULL) STRICT"
    )
    connection.execute(
        "CREATE TABLE cameras (camera_id TEXT PRIMARY KEY, label TEXT, "
        "decode_backend TEXT, payload_json TEXT NOT NULL) STRICT"
    )
    connection.execute(
        "CREATE TABLE audit (audit_id TEXT PRIMARY KEY, occurred_at TEXT, action TEXT, "
        "payload_json TEXT NOT NULL) STRICT"
    )
    connection.execute(
        "INSERT INTO cameras (camera_id, label, decode_backend, payload_json) VALUES (?, ?, ?, ?)",
        (
            "cam-1",
            "Lobby",
            "software",
            json.dumps(camera_payload, sort_keys=True, separators=(",", ":")),
        ),
    )
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    store = CatalogStore.open(path)
    try:
        version = store._connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == 3

        tables = {
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"credentials", "camera_registry"} <= tables
        assert "runtime_latency" not in tables

        for table in ("credentials", "camera_registry"):
            count = store._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0, f"expected {table} to start empty, found {count} row(s)"

        # The pre-existing `cameras` cache table (distinct from the new
        # `camera_registry` table) survives the migration untouched.
        assert store.records("cameras") == [camera_payload]
    finally:
        store.close()

    # Idempotent: reopening an already-migrated database is a no-op.
    reopened = CatalogStore.open(path)
    reopened.close()


def test_camera_registry_and_credentials_share_externally_migrated_schema18(tmp_path) -> None:
    path = tmp_path / "edge.sqlite3"
    bootstrap_database(path)

    camera_store = CameraRegistryStore(path)
    camera_store.create(
        camera_id="cam-1",
        label="Lobby",
        rtsp_url="rtsp://camera/live",
        space_id=None,
        status="online",
    )
    DashboardCredentialsStore(path).save(username="admin", password="admin")

    reloaded = CameraRegistryStore(path).snapshot()
    credentials = DashboardCredentialsStore(path).load()
    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert reloaded["registry_version"] == 1
    assert [record["id"] for record in reloaded["cameras"]] == ["cam-1"]
    assert credentials is not None and credentials.verify_password("admin")
    assert "camera_registry" not in tables
    assert "runtime_latency" not in tables


def test_schema18_authorities_do_not_export_feature_local_ddl() -> None:
    import backend.app.features.cameras.store as camera_store_module

    assert not hasattr(camera_store_module, "_CREATE_CAMERA_REGISTRY_TABLE")

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.features.clips.catalog import CatalogStore

ROOT = Path(__file__).resolve().parents[1]


_EVENT_ID = "11111111-1111-4111-8111-111111111111"
_SNAPSHOT_ID_1 = "22222222-2222-4222-8222-222222222222"
_SNAPSHOT_ID_2 = "33333333-3333-4333-8333-333333333333"


def _payload(clip_id: str, *, state: str, media: bytes | None) -> dict[str, object]:
    path = f"clips/{clip_id}/clip.mp4" if media is not None else None
    payload: dict[str, object] = {
        "manifest_schema_version": 2,
        "clip_id": clip_id,
        "camera_id": "cam-1",
        "event_ref": _EVENT_ID,
        "event_refs": [_EVENT_ID],
        "event_type": "fall",
        "started_at": "2026-01-01T00:00:00Z",
        "clip_start_at": "2026-01-01T00:00:00Z",
        "clip_end_at": "2026-01-01T00:00:01Z",
        "finalized_at": "2026-01-01T00:00:02Z",
        "duration_s": 1.0,
        "path": path,
        "finalized": True,
        "video_available": media is not None,
        "state_version": 2,
        "state": state,
    }
    if media is not None:
        payload.update(
            {
                "sha256": hashlib.sha256(media).hexdigest(),
                "size_bytes": len(media),
                "mime_type": "video/mp4",
                "codec": "h264",
                "duration_ms": 1000,
                "encoder": "ffmpeg",
            }
        )
    else:
        payload.update(
            {
                "sha256": None,
                "size_bytes": None,
                "mime_type": None,
                "codec": None,
                "duration_ms": None,
                "reason_code": "ENCODER_FAILED",
                "encoder": "ffmpeg",
            }
        )
    return payload


def _snapshot(
    snapshot_id: str,
    content: bytes,
    *,
    path: str | None = None,
    camera_id: str = "cam-1",
) -> dict[str, object]:
    captured_at = "2026-01-01T00:00:00.000Z"
    camera_key = hashlib.sha256(camera_id.encode()).hexdigest()[:16]
    snapshot_key = hashlib.sha256(snapshot_id.encode()).hexdigest()
    return {
        "snapshot_id": snapshot_id,
        "camera_id": camera_id,
        "edge_event_id": snapshot_id,
        "captured_at": captured_at,
        "path": path or f"snapshots/{camera_key}/2026-01-01/{snapshot_key}.jpg",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "mime_type": "image/jpeg",
    }


def _run_verify(
    tmp_path: Path,
    payload: dict[str, object],
    media: bytes | None,
    snapshots: tuple[tuple[dict[str, object], bytes | None], ...] = (),
) -> subprocess.CompletedProcess[str]:
    clip_id = str(payload["clip_id"])
    clip_root = tmp_path / "clip-store"
    clip_dir = clip_root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    if media is not None:
        (clip_dir / "clip.mp4").write_bytes(media)
    catalog = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        catalog.record("clips", clip_id, payload)
        for snapshot, snapshot_bytes in snapshots:
            if snapshot_bytes is not None:
                snapshot_path = clip_root / str(snapshot["path"])
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_path.write_bytes(snapshot_bytes)
            catalog.record("snapshots", str(snapshot["snapshot_id"]), snapshot)
    finally:
        catalog.close()
    return _verify_catalog(tmp_path)


def _verify_catalog(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/catalog_verify.py",
            "--catalog",
            str(tmp_path / "catalog.sqlite3"),
            "--clip-store",
            str(tmp_path / "clip-store"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        capture_output=True,
    )


def _camera_source() -> dict[str, object]:
    return {
        "cameras": [
            {
                "backend_camera_id": "cam_a",
                "id": "11111111-1111-1111-1111-111111111111",
                "label": "A",
                "decode_backend": "nvdec",
                "rtsp_url": "rtsp://operator:fixture-password@192.0.2.10/s",
                "created_at": "2026-01-01T00:00:00Z",
                "mapping_pending": False,
                "future_field": {"auth": {"token": "ignored-by-catalog"}},
            },
            {
                "backend_camera_id": "cam_b",
                "id": "22222222-2222-2222-2222-222222222222",
                "label": "B",
                "decode_backend": "nvdec",
                "rtsp_url": "rtsp://operator:fixture-password@192.0.2.11/s",
                "created_at": "2026-01-02T00:00:00Z",
                "mapping_pending": True,
                "future_field": ["ignored-by-catalog"],
            },
        ],
        "registry_version": 1,
    }


def _run_backfill(
    tmp_path: Path, *, camera_source: Path | None
) -> subprocess.CompletedProcess[str]:
    # An explicit nonexistent path stands in for "absent" so the result does
    # not depend on whether ~/.local/state/ml-api/cameras.json happens to
    # exist on the host running the test (no env-var override exists to
    # force absence anymore).
    cameras = camera_source if camera_source is not None else tmp_path / "absent"
    return subprocess.run(
        [
            sys.executable,
            "scripts/catalog_backfill.py",
            "--catalog",
            str(tmp_path / "catalog.sqlite3"),
            "--clip-store",
            str(tmp_path / "clip-store"),
            "--cameras",
            str(cameras),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        capture_output=True,
    )


def test_catalog_backfill_and_verify_ignore_unknown_camera_fields(tmp_path) -> None:
    camera_source = tmp_path / "cameras.json"
    camera_source.write_text(json.dumps(_camera_source()), encoding="utf-8")

    result = _run_backfill(tmp_path, camera_source=camera_source)

    assert result.returncode == 0
    assert f"camera_source={camera_source}" in result.stdout
    assert "cameras=2" in result.stdout
    catalog = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        rows = catalog._connection.execute("SELECT payload_json FROM cameras")
        encoded = "".join(row[0] for row in rows)
    finally:
        catalog.close()
    for secret in ("rtsp://", "user", "pw", "username", "password"):
        assert secret not in encoded
    verify = subprocess.run(
        [
            sys.executable,
            "scripts/catalog_verify.py",
            "--catalog",
            str(tmp_path / "catalog.sqlite3"),
            "--clip-store",
            str(tmp_path / "clip-store"),
            "--cameras",
            str(camera_source),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        capture_output=True,
    )
    assert verify.returncode == 0
    assert f"camera_source={camera_source}" in verify.stdout
    assert "cameras=2" in verify.stdout


def test_catalog_tools_report_absent_camera_source_without_failing_clip_verification(
    tmp_path,
) -> None:
    result = _run_backfill(tmp_path, camera_source=None)

    assert result.returncode == 0
    assert "camera_source=absent cameras=0" in result.stdout
    verify = subprocess.run(
        [
            sys.executable,
            "scripts/catalog_verify.py",
            "--catalog",
            str(tmp_path / "catalog.sqlite3"),
            "--clip-store",
            str(tmp_path / "clip-store"),
            "--cameras",
            str(tmp_path / "absent"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        capture_output=True,
    )
    assert verify.returncode == 0
    assert "camera_source=absent cameras=0" in verify.stdout


def test_catalog_backfill_rejects_corrupt_camera_source(tmp_path) -> None:
    camera_source = tmp_path / "cameras.json"
    camera_source.write_text("{broken", encoding="utf-8")

    result = _run_backfill(tmp_path, camera_source=camera_source)

    assert result.returncode != 0
    assert "unable to read camera source" in result.stderr


@pytest.mark.parametrize(
    "camera_source_payload",
    [
        {"registry_version": "1", "cameras": []},
        {"registry_version": 1, "cameras": [None]},
        {"registry_version": 1, "cameras": [{"id": ""}]},
        {"registry_version": 1, "cameras": [{"id": "cam-1", "label": 42}]},
    ],
)
def test_catalog_backfill_rejects_invalid_known_camera_schema(
    tmp_path: Path, camera_source_payload: dict[str, object]
) -> None:
    camera_source = tmp_path / "cameras.json"
    camera_source.write_text(json.dumps(camera_source_payload), encoding="utf-8")

    result = _run_backfill(tmp_path, camera_source=camera_source)

    assert result.returncode != 0


def test_catalog_verify_allows_declared_unavailable_media(tmp_path) -> None:
    result = _run_verify(tmp_path, _payload("clip-1", state="UNAVAILABLE", media=None), None)

    assert result.returncode == 0
    assert "states=UNAVAILABLE:1" in result.stdout
    assert "media_absent_declared=1" in result.stdout
    assert "media_errors=0" in result.stdout


def test_catalog_verify_rejects_ready_media_missing(tmp_path) -> None:
    payload = _payload("clip-1", state="READY", media=b"expected")
    clip_root = tmp_path / "clip-store"
    clip_dir = clip_root / "clips" / "clip-1"
    clip_dir.mkdir(parents=True)
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    catalog = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        catalog.record("clips", "clip-1", payload)
    finally:
        catalog.close()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/catalog_verify.py",
            "--catalog",
            str(tmp_path / "catalog.sqlite3"),
            "--clip-store",
            str(clip_root),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "media_errors=1" in result.stdout


def test_catalog_verify_rejects_sha256_mismatch(tmp_path) -> None:
    payload = _payload("clip-1", state="READY", media=b"expected")
    result = _run_verify(tmp_path, payload, b"changed!")

    assert result.returncode == 1
    assert "media_sha256_mismatches=1" in result.stdout


def test_catalog_verify_rejects_size_mismatch(tmp_path) -> None:
    payload = _payload("clip-1", state="READY", media=b"expected")
    result = _run_verify(tmp_path, payload, b"much longer media")

    assert result.returncode == 1
    assert "media_size_mismatches=1" in result.stdout


def test_catalog_verify_rejects_media_for_unavailable_manifest(tmp_path) -> None:
    result = _run_verify(
        tmp_path, _payload("clip-1", state="UNAVAILABLE", media=None), b"unexpected"
    )

    assert result.returncode == 1
    assert "unavailable_media_present=1" in result.stdout


def test_catalog_verify_reports_corrupt_raw_sidecar(tmp_path) -> None:
    payload = _payload("clip-1", state="UNAVAILABLE", media=None)
    result = _run_verify(tmp_path, payload, None)
    manifest = tmp_path / "clip-store" / "clips" / "clip-1" / "manifest.json"
    manifest.write_text("{broken", encoding="utf-8")
    result = _verify_catalog(tmp_path)
    assert result.returncode == 1
    assert "manifest_errors=1" in result.stdout


def test_catalog_verify_rejects_snapshot_path_escape(tmp_path) -> None:
    content = b"snapshot"
    snapshot = _snapshot(_SNAPSHOT_ID_1, content, path="../outside.jpg")

    result = _run_verify(
        tmp_path, _payload("clip-1", state="UNAVAILABLE", media=None), None, ((snapshot, None),)
    )

    assert result.returncode == 1
    assert "malformed_snapshot_paths=1" in result.stdout


def test_catalog_verify_rejects_missing_or_mutated_snapshot(tmp_path) -> None:
    content = b"expected"
    missing = _snapshot(_SNAPSHOT_ID_1, content)
    mutated = _snapshot(_SNAPSHOT_ID_2, content)

    result = _run_verify(
        tmp_path,
        _payload("clip-1", state="UNAVAILABLE", media=None),
        None,
        ((missing, None), (mutated, b"changed")),
    )

    assert result.returncode == 1
    assert "missing_snapshots=1" in result.stdout
    assert "snapshot_sha256_mismatches=1" in result.stdout
    assert "snapshot_size_mismatches=1" in result.stdout


def test_catalog_verify_rejects_consistent_but_invalid_snapshot_payload(tmp_path) -> None:
    content = b"snapshot"
    snapshot = _snapshot(_SNAPSHOT_ID_1, content)
    snapshot["mime_type"] = "image/png"

    result = _run_verify(
        tmp_path,
        _payload("clip-1", state="UNAVAILABLE", media=None),
        None,
        ((snapshot, content),),
    )

    assert result.returncode == 1
    assert "malformed_snapshot_paths=1" in result.stdout
    assert "snapshot_promoted_mismatches=0" in result.stdout


def test_catalog_verify_rejects_orphan_snapshot_file(tmp_path) -> None:
    content = b"expected"
    snapshot = _snapshot(_SNAPSHOT_ID_1, content)

    orphan = tmp_path / "clip-store" / "snapshots" / "cam-1" / "orphan.jpg"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")

    result = _run_verify(
        tmp_path,
        _payload("clip-1", state="UNAVAILABLE", media=None),
        None,
        ((snapshot, content),),
    )

    assert result.returncode == 1
    assert "orphan_snapshot_files=1" in result.stdout


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("snapshot_id", "snapshot-corrupt"),
        ("path", "snapshots/cam-1/corrupt.jpg"),
        ("sha256", "b" * 64),
        ("size_bytes", 999),
        ("camera_id", "cam-corrupt"),
        ("edge_event_id", _SNAPSHOT_ID_2),
        ("captured_at", "2026-01-01T00:00:01Z"),
        ("mime_type", "image/png"),
    ),
)
def test_catalog_verify_rejects_direct_sql_snapshot_column_corruption(
    tmp_path, column, value
) -> None:
    content = b"snapshot"
    snapshot = _snapshot(_SNAPSHOT_ID_1, content)

    _run_verify(
        tmp_path,
        _payload("clip-1", state="UNAVAILABLE", media=None),
        None,
        ((snapshot, content),),
    )
    catalog = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        catalog._connection.execute(
            f"UPDATE snapshots SET {column} = ? WHERE snapshot_id = ?",
            (value, _SNAPSHOT_ID_1),
        )
    finally:
        catalog.close()

    result = _verify_catalog(tmp_path)

    assert result.returncode == 1
    assert "snapshot_promoted_mismatches=1" in result.stdout

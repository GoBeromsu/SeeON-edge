"""Local camera registry storage and lightweight probes for ml-api.

Storage lives in the single-row ``camera_registry`` table of the shared
``catalog.sqlite3`` database (see ``backend/app/features/clips/catalog.py``),
distinct from the pre-existing ``cameras`` table there (a derived clip/audit
denormalization read cache, not this registry's source of truth). This
module bootstraps its own table independently via an idempotent
``CREATE TABLE IF NOT EXISTS`` (identical statement text to ``catalog.py``'s
``_V3_TABLE_STATEMENTS``) rather than depending on ``CatalogStore``, so
construction order relative to the catalog store never matters, and never
touches ``PRAGMA user_version`` (that stays exclusively ``CatalogStore``'s
responsibility).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from backend.app.shared.state_dir import resolve_state_dir

CameraStatus = Literal["online", "offline", "starting", "unknown"]
ProbeErrorClass = Literal["timeout", "decode", "auth"]

_CREATE_CAMERA_REGISTRY_TABLE = (
    "CREATE TABLE IF NOT EXISTS camera_registry (id INTEGER PRIMARY KEY CHECK (id = 1), "
    "registry_version INTEGER NOT NULL, cameras_json TEXT NOT NULL) STRICT"
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    error_class: ProbeErrorClass | None = None
    width: int | None = None
    height: int | None = None


class DuplicateCameraError(Exception):
    """Raised when a create/update would register a second camera for the
    same physical RTSP stream (see ``normalize_stream_identity``)."""

    def __init__(self, existing_record: dict[str, object]) -> None:
        super().__init__("a camera is already registered for this stream")
        self.existing_record = existing_record


class CameraRegistryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_env(cls) -> CameraRegistryStore:
        return cls(resolve_state_dir("ml-api") / "catalog.sqlite3")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._read_unlocked()

    def create(
        self,
        *,
        camera_id: str | None = None,
        label: str,
        rtsp_url: str,
        space_id: str | None,
        status: CameraStatus,
        backend_camera_id: str | None = None,
        mapping_pending: bool = False,
        decode_backend: str | None = None,
        fps: float | None = None,
        floor: str | None = None,
        last_probed_at: str | None = None,
        last_ok_at: str | None = None,
        never_connected: bool = True,
    ) -> dict[str, object]:
        with self._lock:
            data = self._read_unlocked()
            # Duplicate check runs INSIDE the lock (not as a router pre-check):
            # a pre-check outside the lock is TOCTOU-racy, since two concurrent
            # POSTs could both pass it before either writes.
            duplicate = _find_duplicate(data["cameras"], rtsp_url)
            if duplicate is not None:
                raise DuplicateCameraError(dict(duplicate))
            record = {
                "id": camera_id or str(uuid.uuid4()),
                "label": label,
                "rtsp_url": rtsp_url,
                "space_id": space_id,
                "backend_camera_id": backend_camera_id,
                "mapping_pending": mapping_pending,
                "status": status,
                "decode_backend": decode_backend,
                "fps": fps,
                "floor": floor,
                "created_at": utc_now_iso(),
                "last_probed_at": last_probed_at,
                "last_ok_at": last_ok_at,
                "never_connected": never_connected,
            }
            data["cameras"].append(record)
            data["registry_version"] += 1
            self._write_unlocked(data)
            return dict(record)

    def update(self, camera_id: str, updates: dict[str, object]) -> dict[str, object] | None:
        with self._lock:
            data = self._read_unlocked()
            for index, record in enumerate(data["cameras"]):
                if record.get("id") != camera_id:
                    continue
                new_rtsp_url = updates.get("rtsp_url")
                if isinstance(new_rtsp_url, str):
                    duplicate = _find_duplicate(
                        data["cameras"], new_rtsp_url, exclude_camera_id=camera_id
                    )
                    if duplicate is not None:
                        raise DuplicateCameraError(dict(duplicate))
                updated = {**record, **updates}
                data["cameras"][index] = updated
                data["registry_version"] += 1
                self._write_unlocked(data)
                return dict(updated)
            return None

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            data = self._read_unlocked()
            cameras = data["cameras"]
            kept = [record for record in cameras if record.get("id") != camera_id]
            if len(kept) == len(cameras):
                return False
            data["cameras"] = kept
            data["registry_version"] += 1
            self._write_unlocked(data)
            return True

    def get(self, camera_id: str) -> dict[str, object] | None:
        with self._lock:
            data = self._read_unlocked()
            for record in data["cameras"]:
                if record.get("id") == camera_id:
                    return dict(record)
            return None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            # RTSP credentials are stored in the registry by design; API
            # responses and logs must use mask_rtsp_url(), and the database
            # file is best-effort 0600 (see connect_catalog_store).
            self._connection = connect_catalog_store(self.path, (_CREATE_CAMERA_REGISTRY_TABLE,))
        return self._connection

    def _read_unlocked(self) -> dict[str, object]:
        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT registry_version, cameras_json FROM camera_registry WHERE id = 1"
            ).fetchone()
        except (OSError, sqlite3.Error):
            return {"registry_version": 0, "cameras": []}
        if row is None:
            return {"registry_version": 0, "cameras": []}
        registry_version, cameras_json = row
        try:
            cameras = json.loads(cameras_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"registry_version": 0, "cameras": []}
        if isinstance(registry_version, bool) or not isinstance(registry_version, int):
            registry_version = 0
        if not isinstance(cameras, list):
            cameras = []
        return {
            "registry_version": max(0, registry_version),
            "cameras": [record for record in cameras if isinstance(record, dict)],
        }

    def _write_unlocked(self, data: dict[str, object]) -> None:
        connection = self._connect()
        cameras_json = json.dumps(
            data["cameras"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        connection.execute(
            "INSERT INTO camera_registry (id, registry_version, cameras_json) "
            "VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "registry_version = excluded.registry_version, cameras_json = excluded.cameras_json",
            (data["registry_version"], cameras_json),
        )


def public_camera(record: dict[str, object]) -> dict[str, object]:
    return {
        "id": str(record.get("id", "")),
        "label": str(record.get("label", "")),
        "rtsp_url_masked": mask_rtsp_url(str(record.get("rtsp_url", ""))),
        "space_id": _optional_str(record.get("space_id")),
        "backend_camera_id": _optional_str(record.get("backend_camera_id")),
        "status": _status(record.get("status")),
        "decode_backend": _optional_str(record.get("decode_backend")),
        "fps": _optional_float(record.get("fps")),
        # User-set floor override (issue #85): distinct from the space-sync-owned
        # "floor_name" merged in later by _public_snapshot from the external
        # roster pull. Kept in a separate key on purpose -- that merge only ever
        # assigns "space_name"/"floor_name" and never touches this field, so a
        # locally-set floor survives every roster re-sync instead of being
        # clobbered by it. See CameraResponse.floor in router.py for the
        # display-precedence contract (floor ?? floor_name).
        "floor": _optional_str(record.get("floor")),
        "created_at": str(record.get("created_at", "")),
        # Probe-history fields (see CameraRegistryStore.create/update): surfaced
        # so the UI can render "never connected" / "last connected at" text.
        # Records written before this field existed simply omit the key, which
        # record.get(...) reports as None -- the same as an explicit null.
        "never_connected": _optional_bool(record.get("never_connected")),
        "last_ok_at": _optional_str(record.get("last_ok_at")),
        "last_probed_at": _optional_str(record.get("last_probed_at")),
    }


def normalize_stream_identity(rtsp_url: str) -> str:
    """Return a comparable identity key for an RTSP stream URL.

    Used to detect duplicate camera registrations (same physical stream
    registered twice). Deliberately excludes username/password so rotating a
    default ``admin/admin`` password does not spawn a zombie duplicate, and
    excludes the fragment (not meaningful for RTSP).

    The query string IS kept (parsed, sorted by key): some vendors (Dahua)
    encode main-vs-sub stream selection in the query (``?subtype=0`` vs
    ``?subtype=1``) while others (Hikvision) encode it in the path. Dropping
    the query would silently merge distinct Dahua streams while leaving
    Hikvision correctly distinguished -- a vendor-dependent bug.

    The path is case-SENSITIVE (only trailing-slash normalized): vendor paths
    like ``/Streaming/Channels/101`` are case-sensitive on the wire.
    """
    try:
        parsed = urlsplit(rtsp_url.strip())
    except ValueError:
        return rtsp_url.strip().lower()
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port is None:
        port_part = ""
    elif scheme == "rtsp" and port == 554:
        # Elide only the well-known rtsp:// default; rtsps:// (and any other
        # scheme) has no default worth guessing, so an explicit port stays.
        port_part = ""
    else:
        port_part = f":{port}"
    path = parsed.path
    if len(path) > 1:
        path = path.rstrip("/")
    query_pairs = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    query_part = "&".join(f"{key}={value}" for key, value in query_pairs)
    identity = f"{scheme}://{host}{port_part}{path}"
    if query_part:
        identity = f"{identity}?{query_part}"
    return identity


def _find_duplicate(
    cameras: list[object], rtsp_url: str, *, exclude_camera_id: str | None = None
) -> dict[str, object] | None:
    target = normalize_stream_identity(rtsp_url)
    for record in cameras:
        if not isinstance(record, dict):
            continue
        if exclude_camera_id is not None and record.get("id") == exclude_camera_id:
            continue
        existing_rtsp = record.get("rtsp_url")
        if not isinstance(existing_rtsp, str):
            continue
        if normalize_stream_identity(existing_rtsp) == target:
            return record
    return None


def mask_rtsp_url(rtsp_url: str) -> str:
    try:
        parsed = urlsplit(rtsp_url)
        port = parsed.port
    except ValueError:
        return "RTSP URL masked"
    if not parsed.hostname:
        return "RTSP URL masked"
    host = "redacted-camera"
    if port is not None:
        host = f"{host}:{port}"
    if parsed.username is None and parsed.password is None:
        return urlunsplit(parsed._replace(netloc=host))
    user = "***" if parsed.username is not None else ""
    password = ":***" if parsed.password is not None else ""
    return urlunsplit(parsed._replace(netloc=f"{user}{password}@{host}"))


def status_from_probe(result: ProbeResult) -> CameraStatus:
    return "online" if result.ok else "offline"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _status(value: object) -> CameraStatus:
    return value if value in {"online", "offline", "starting", "unknown"} else "unknown"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "CameraRegistryStore",
    "CameraStatus",
    "DuplicateCameraError",
    "ProbeResult",
    "mask_rtsp_url",
    "normalize_stream_identity",
    "public_camera",
    "status_from_probe",
    "utc_now_iso",
]

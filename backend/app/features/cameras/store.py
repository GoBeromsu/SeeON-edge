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
import logging
import re
import sqlite3
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal, TypedDict
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from backend.app.features.cameras.topology import CameraTopologyStore, RegistryTopologySnapshot
from backend.app.features.cameras.topology_schema import TOPOLOGY_SCHEMA
from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from shared.edge_db import EDGE_DATABASE_PATH

CameraStatus = Literal["online", "offline", "starting", "unknown"]
ProbeErrorClass = Literal["timeout", "decode", "auth", "unsupported"]


class CameraRegistryData(TypedDict):
    registry_version: int
    cameras: list[dict[str, object]]


logger = logging.getLogger(__name__)

_CREATE_CAMERA_REGISTRY_TABLE = (
    "CREATE TABLE IF NOT EXISTS camera_registry (id INTEGER PRIMARY KEY CHECK (id = 1), "
    "registry_version INTEGER NOT NULL, cameras_json TEXT NOT NULL) STRICT"
)

# Fixed floor catalog (issue #155): B1 through 10층, encoded as an integer
# (basement negative) instead of a free-text chip label. Display strings are
# generated at render time from this integer -- see floor_label() -- rather
# than persisted, so "2층"/"2"/"B1"/"2 층" style drift is structurally
# impossible for anything written from here on.
FLOOR_MIN = -1
FLOOR_MAX = 10
DEFAULT_FLOOR = 1
FLOOR_VALUES: tuple[int, ...] = (FLOOR_MIN, *range(1, FLOOR_MAX + 1))

_FLOOR_BASEMENT_RE = re.compile(r"^B\s*(\d+)$", re.IGNORECASE)
_FLOOR_LEVEL_RE = re.compile(r"^(-?\d+)\s*층$")
_FLOOR_PLAIN_INT_RE = re.compile(r"^-?\d+$")


def is_valid_floor(value: int) -> bool:
    return value in FLOOR_VALUES


def floor_label(value: int) -> str:
    """Render-time display string for a canonical floor integer (issue #155).

    The integer is the only thing persisted; this is deliberately not stored
    anywhere, so relabeling never requires a data migration.
    """
    return f"B{-value}" if value < 0 else f"{value}층"


def parse_legacy_floor(value: object, *, camera_id: str | None = None) -> int | None:
    """Coerce a possibly pre-#155 floor value into the canonical integer.

    Pre-#155 records store floor as a free-text chip label the facility
    typed in by hand ('2층', '2', 'B1', '2 층', ...), which never round-trips
    reliably: string sort puts '10층' before '2층', and the same floor could
    be spelled four different ways. ``None`` means "not set" and passes
    through untouched (falls back to the space-sync ``floor_name``).
    Anything else that can't be parsed -- or parses to a value outside the
    fixed B1..10층 catalog -- is never silently dropped (dropping it would
    read as "the override was cleared"); it defaults to ``DEFAULT_FLOOR``
    and is logged, so a stray legacy value doesn't 500 every request from
    here on.
    """
    if value is None:
        return None
    if not isinstance(value, bool) and isinstance(value, int):
        if is_valid_floor(value):
            return value
        return _floor_parse_failed(value, camera_id)
    if isinstance(value, str):
        text = value.strip()
        basement = _FLOOR_BASEMENT_RE.match(text)
        if basement is not None:
            candidate = -int(basement.group(1))
        else:
            level = _FLOOR_LEVEL_RE.match(text)
            if level is not None:
                candidate = int(level.group(1))
            elif _FLOOR_PLAIN_INT_RE.match(text):
                candidate = int(text)
            else:
                return _floor_parse_failed(value, camera_id)
        if is_valid_floor(candidate):
            return candidate
        return _floor_parse_failed(value, camera_id)
    return _floor_parse_failed(value, camera_id)


def _floor_parse_failed(value: object, camera_id: str | None) -> int:
    logger.warning(
        "camera floor value could not be parsed, defaulting to %s (camera_id=%s, value=%r)",
        DEFAULT_FLOOR,
        camera_id,
        value,
    )
    return DEFAULT_FLOOR


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    error_class: ProbeErrorClass | None = None
    width: int | None = None
    height: int | None = None
    # worker의 /probe에 아예 닿지 못했음을 나타낸다 (origin 미설정, relay
    # 토큰 없음, 연결 거부/타임아웃 이전 단계의 OS 레벨 실패 등) -- worker가
    # 살아서 응답했지만 그 응답이 실패로 판정한 것(error_class)과는 다른
    # 축이다. 이슈 #151: 이 둘이 구분 안 되면 "검사 자체를 못 함"이 전부
    # "디코드 실패"로 오표시된다.
    probe_unavailable: bool = False


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
        self._topology = CameraTopologyStore()

    @classmethod
    def from_env(cls) -> CameraRegistryStore:
        return cls(EDGE_DATABASE_PATH)

    def snapshot(self) -> CameraRegistryData:
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
        floor: int | None = None,
        last_probed_at: str | None = None,
        last_ok_at: str | None = None,
        never_connected: bool = True,
        edge_ref: str | None = None,
        room_edge_ref: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            with self._transaction_unlocked() as connection:
                data = self._read_unlocked()
                # Duplicate check runs INSIDE the lock (not as a router pre-check):
                # a pre-check outside the lock is TOCTOU-racy, since two concurrent
                # POSTs could both pass it before either writes.
                duplicate = _find_duplicate(data["cameras"], rtsp_url)
                if duplicate is not None:
                    raise DuplicateCameraError(dict(duplicate))
                record: dict[str, object] = {
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
                    "edge_ref": edge_ref,
                    "room_edge_ref": room_edge_ref,
                }
                data["cameras"].append(record)
                data["registry_version"] += 1
                self._topology.bind_camera(
                    connection,
                    camera_id=str(record["id"]),
                    edge_ref=edge_ref,
                    room_edge_ref=room_edge_ref,
                )
                self._write_mutation_unlocked(connection, data)
                return dict(record)

    def update(self, camera_id: str, updates: dict[str, object]) -> dict[str, object] | None:
        with self._lock:
            with self._transaction_unlocked() as connection:
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
                    if "edge_ref" in updates or "room_edge_ref" in updates:
                        edge_ref = updated.get("edge_ref")
                        room_edge_ref = updated.get("room_edge_ref")
                        self._topology.delete_camera(connection, camera_id)
                        self._topology.bind_camera(
                            connection,
                            camera_id=camera_id,
                            edge_ref=edge_ref if isinstance(edge_ref, str) else None,
                            room_edge_ref=(
                                room_edge_ref if isinstance(room_edge_ref, str) else None
                            ),
                        )
                    data["cameras"][index] = updated
                    data["registry_version"] += 1
                    self._write_mutation_unlocked(connection, data)
                    return dict(updated)
            return None

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            with self._transaction_unlocked() as connection:
                data = self._read_unlocked()
                cameras = data["cameras"]
                kept = [record for record in cameras if record.get("id") != camera_id]
                if len(kept) == len(cameras):
                    return False
                data["cameras"] = kept
                data["registry_version"] += 1
                self._topology.delete_camera(connection, camera_id)
                self._write_mutation_unlocked(connection, data)
                return True

    def get(self, camera_id: str) -> dict[str, object] | None:
        with self._lock:
            data = self._read_unlocked()
            for record in data["cameras"]:
                if record.get("id") == camera_id:
                    return dict(record)
            return None

    def create_floor(self, *, edge_ref: str, name: str, order_index: int) -> None:
        with self._lock, self._transaction_unlocked() as connection:
            data = self._read_unlocked()
            self._topology.create_floor(
                connection, edge_ref=edge_ref, name=name, order_index=order_index
            )
            data["registry_version"] += 1
            self._write_mutation_unlocked(connection, data)

    def update_floor(self, edge_ref: str, *, name: str, order_index: int) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            data = self._read_unlocked()
            changed = self._topology.update_floor(
                connection, edge_ref, name=name, order_index=order_index
            )
            if changed:
                data["registry_version"] += 1
                self._write_mutation_unlocked(connection, data)
            return changed

    def delete_floor(self, edge_ref: str) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            data = self._read_unlocked()
            changed = self._topology.delete_floor(connection, edge_ref)
            if changed:
                data["registry_version"] += 1
                self._write_mutation_unlocked(connection, data)
            return changed

    def create_room(
        self,
        *,
        edge_ref: str,
        floor_edge_ref: str,
        name: str,
        legacy_canonical_space_id: str | None = None,
    ) -> None:
        with self._lock, self._transaction_unlocked() as connection:
            data = self._read_unlocked()
            self._topology.create_room(
                connection,
                edge_ref=edge_ref,
                floor_edge_ref=floor_edge_ref,
                name=name,
                legacy_canonical_space_id=legacy_canonical_space_id,
            )
            data["registry_version"] += 1
            self._write_mutation_unlocked(connection, data)

    def update_room(self, edge_ref: str, *, name: str) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            data = self._read_unlocked()
            changed = self._topology.update_room(connection, edge_ref, name=name)
            if changed:
                data["registry_version"] += 1
                self._write_mutation_unlocked(connection, data)
            return changed

    def delete_room(self, edge_ref: str) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            data = self._read_unlocked()
            changed = self._topology.delete_room(connection, edge_ref)
            if changed:
                data["registry_version"] += 1
                self._write_mutation_unlocked(connection, data)
            return changed

    def topology_snapshot(self) -> RegistryTopologySnapshot:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN")
            try:
                data = self._read_unlocked()
                camera_ids = tuple(
                    str(record["id"])
                    for record in data["cameras"]
                    if isinstance(record, dict) and isinstance(record.get("id"), str)
                )
                snapshot = self._topology.snapshot(
                    connection,
                    registry_version=data["registry_version"],
                    camera_ids=camera_ids,
                )
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return snapshot

    def migrate_legacy_string_floors(self) -> list[dict[str, object]]:
        """One-time data migration (issue #155): rewrite any pre-existing
        string floor values ('2층', 'B1', ...) to the canonical integer
        in-place, so the persisted JSON itself is clean rather than relying
        on read-time normalization (``public_camera``'s ``parse_legacy_floor``
        call) forever. Safe to run more than once -- a floor that is already
        an int (or unset) is left untouched. Returns the list of changes
        made, for the caller (see scripts/migrate_camera_floor_to_int.py) to
        log/print; callers are responsible for backing up the database file
        before calling this.
        """
        changes: list[dict[str, object]] = []
        with self._lock:
            data = self._read_unlocked()
            changed = False
            for record in data["cameras"]:
                old = record.get("floor")
                if old is None or (isinstance(old, int) and not isinstance(old, bool)):
                    continue
                new = parse_legacy_floor(old, camera_id=_optional_str(record.get("id")))
                record["floor"] = new
                changed = True
                changes.append({"camera_id": record.get("id"), "old": old, "new": new})
            if changed:
                with self._transaction_unlocked() as connection:
                    data["registry_version"] += 1
                    self._write_mutation_unlocked(connection, data)
        return changes

    def _connect(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            # RTSP credentials are stored in the registry by design; API
            # responses and logs must use mask_rtsp_url(), and the database
            # file is best-effort 0600 (see connect_catalog_store).
            connection = connect_catalog_store(
                self.path, (_CREATE_CAMERA_REGISTRY_TABLE, *TOPOLOGY_SCHEMA)
            )
            self._connection = connection
        return connection

    def _read_unlocked(self) -> CameraRegistryData:
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
            loaded = json.loads(str(cameras_json))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"registry_version": 0, "cameras": []}
        if isinstance(registry_version, bool) or not isinstance(registry_version, int):
            registry_version = 0
        cameras = loaded if isinstance(loaded, list) else []
        return {
            "registry_version": max(0, registry_version),
            "cameras": [
                {str(key): value for key, value in record.items()}
                for record in cameras
                if isinstance(record, dict)
            ],
        }

    def _write_unlocked(
        self, data: CameraRegistryData, connection: sqlite3.Connection | None = None
    ) -> None:
        connection = connection or self._connect()
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

    def _write_mutation_unlocked(
        self, connection: sqlite3.Connection, data: CameraRegistryData
    ) -> None:
        self._write_unlocked(data, connection)
        connection.execute(
            "INSERT INTO topology_dirty (id, registry_version, created_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET registry_version = excluded.registry_version, "
            "created_at = excluded.created_at",
            (data["registry_version"], utc_now_iso()),
        )

    @contextmanager
    def _transaction_unlocked(self) -> Generator[sqlite3.Connection]:
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        with connection:
            yield connection


def public_camera(record: dict[str, object]) -> dict[str, object]:
    response: dict[str, object] = {
        "id": str(record.get("id", "")),
        "label": str(record.get("label", "")),
        "rtsp_url_masked": mask_rtsp_url(str(record.get("rtsp_url", ""))),
        "space_id": _optional_str(record.get("space_id")),
        "backend_camera_id": _optional_str(record.get("backend_camera_id")),
        # 클라우드 매핑이 아직 안 끝났는지. 기사님이 현장을 떠나기 전에
        # "이 카메라가 클라우드에 제대로 붙었는가"를 한 화면에서 확인해야
        # 하는데, 예전에는 이 값이 응답에서 빠져 로컬 상태만 보였다.
        "mapping_pending": _optional_bool(record.get("mapping_pending")),
        "status": _status(record.get("status")),
        "decode_backend": _optional_str(record.get("decode_backend")),
        "fps": _optional_float(record.get("fps")),
        # User-set floor override (issue #85, integer since issue #155): distinct
        # from the space-sync-owned "floor_name" merged in later by
        # _public_snapshot from the external roster pull. Kept in a separate key
        # on purpose -- that merge only ever assigns "space_name"/"floor_name"
        # and never touches this field, so a locally-set floor survives every
        # roster re-sync instead of being clobbered by it. See CameraResponse.floor
        # in router.py for the display-precedence contract (floor ?? floor_name).
        # parse_legacy_floor self-heals any pre-#155 string value on read (see
        # migrate_legacy_string_floors for the one-time rewrite of stored data).
        "floor": parse_legacy_floor(record.get("floor"), camera_id=_optional_str(record.get("id"))),
        "created_at": str(record.get("created_at", "")),
        # Probe-history fields (see CameraRegistryStore.create/update): surfaced
        # so the UI can render "never connected" / "last connected at" text.
        # Records written before this field existed simply omit the key, which
        # record.get(...) reports as None -- the same as an explicit null.
        "never_connected": _optional_bool(record.get("never_connected")),
        "last_ok_at": _optional_str(record.get("last_ok_at")),
        "last_probed_at": _optional_str(record.get("last_probed_at")),
    }
    edge_ref = _optional_str(record.get("edge_ref"))
    room_edge_ref = _optional_str(record.get("room_edge_ref"))
    if edge_ref is not None:
        response["edge_ref"] = edge_ref
    if room_edge_ref is not None:
        response["room_edge_ref"] = room_edge_ref
    return response


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
    cameras: Sequence[dict[str, object]],
    rtsp_url: str,
    *,
    exclude_camera_id: str | None = None,
) -> dict[str, object] | None:
    target = normalize_stream_identity(rtsp_url)
    for record in cameras:
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
    if value == "online":
        return "online"
    if value == "offline":
        return "offline"
    if value == "starting":
        return "starting"
    return "unknown"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def registry_expected_cameras(
    store: CameraRegistryStore | None,
) -> dict[str, dict[str, str | None]]:
    """Build the camera-id index used by status/heartbeat enumeration.

    INBOUND LOOKUP ONLY. Both the Hub-issued backend_camera_id and the local
    registry id are indexed on purpose, because a worker may heartbeat under
    either one; accepting both on the way in is deliberate tolerance. This index
    must never be used to build an outbound payload -- emitting the local id
    where a Hub-issued id belongs is what the Hub rejects with
    FACILITY_BINDING_MISMATCH (issue #308).
    Values are thin binding dicts compatible with HeartbeatStore.snapshot.
    """
    if store is None:
        return {}
    snapshot = store.snapshot()
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return {}
    index: dict[str, dict[str, str | None]] = {}
    for record in cameras:
        if not isinstance(record, dict):
            continue
        local_id = record.get("id")
        backend_id = record.get("backend_camera_id")
        canonical = backend_id or local_id
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        binding = {
            "camera_id": canonical,
            "facility_id": None,
            "resident_id": None,
        }
        index[canonical] = binding
        if isinstance(local_id, str) and local_id and local_id != canonical:
            index[local_id] = binding
    return index


__all__ = [
    "CameraRegistryStore",
    "CameraStatus",
    "DEFAULT_FLOOR",
    "DuplicateCameraError",
    "FLOOR_MAX",
    "FLOOR_MIN",
    "FLOOR_VALUES",
    "ProbeResult",
    "floor_label",
    "is_valid_floor",
    "mask_rtsp_url",
    "normalize_stream_identity",
    "parse_legacy_floor",
    "public_camera",
    "registry_expected_cameras",
    "status_from_probe",
    "utc_now_iso",
]

"""Compact schema persistence for edge enrollment."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias
from uuid import uuid4

from backend.app.edge_db.configuration import (
    ensure_edge_site,
    open_configuration_database,
)
from backend.app.edge_db.configuration import (
    utc_now as utc_now_iso,
)

ConnectionValue: TypeAlias = str | int | None
ConnectionData: TypeAlias = dict[str, ConnectionValue]
ConnectionWriteHook: TypeAlias = Callable[[sqlite3.Connection], None]

SAVE_FIELDS: Final = (
    "facility_code",
    "client_installation_ref",
    "facility_id",
    "facility_token",
    "edge_installation_id",
    "enrollment_generation",
)
COLUMNS: Final = (*SAVE_FIELDS, "enrollment_created_at", "enrollment_updated_at", "updated_at")
ENROLLMENT_MARKER_FIELDS: Final = (
    "facility_code",
    "client_installation_ref",
    "edge_installation_id",
    "enrollment_generation",
)
REQUIRED_ENROLLMENT_FIELDS: Final = SAVE_FIELDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConnectionStoreBackup:
    path: Path
    sha256: str
    size_bytes: int


class ConnectionStoreDatabase:
    """Column-scoped access to the schema-18 ``edge_site`` singleton."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rollback_directory = path.parent / "connection-settings-rollback"

    def read(self) -> ConnectionData:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT facility_code,client_installation_ref,facility_id,facility_token,"
                    "edge_installation_id,enrollment_generation,enrollment_created_at,"
                    "enrollment_updated_at,updated_at FROM edge_site WHERE id=1"
                ).fetchone()
        except sqlite3.DatabaseError as error:
            logger.warning("connection authority unreadable at %s: %r", self.path, error)
            return _empty_data()
        return _empty_data() if row is None else dict(zip(COLUMNS, row, strict=True))

    def write(
        self, data: ConnectionData, after_write: ConnectionWriteHook | None = None
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                ensure_edge_site(connection)
                previous_principal = connection.execute(
                    "SELECT edge_installation_id,enrollment_generation FROM edge_site WHERE id=1"
                ).fetchone()
                connection.execute(
                    "UPDATE edge_site SET facility_code=:facility_code,"
                    "client_installation_ref=:client_installation_ref,facility_id=:facility_id,"
                    "facility_token=:facility_token,edge_installation_id=:edge_installation_id,"
                    "enrollment_generation=:enrollment_generation,"
                    "enrollment_created_at=:enrollment_created_at,"
                    "enrollment_updated_at=:enrollment_updated_at,updated_at=:updated_at "
                    "WHERE id=1",
                    data,
                )
                current_principal = (
                    data["edge_installation_id"],
                    data["enrollment_generation"],
                )
                if previous_principal != current_principal:
                    connection.execute(_RESET_TOPOLOGY_SQL, (data["updated_at"],))
                if after_write is not None:
                    after_write(connection)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def create_pre_v1_backup(self) -> ConnectionStoreBackup:
        self.rollback_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.rollback_directory / f".{self.path.name}.{uuid4().hex}.tmp"
        try:
            with closing(self._connect()) as source, closing(sqlite3.connect(temporary)) as target:
                source.backup(target)
            os.chmod(temporary, 0o600)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            final = self.rollback_directory / f"{self.path.name}.enrollment.{digest}"
            os.replace(temporary, final)
            return ConnectionStoreBackup(final, digest, final.stat().st_size)
        except (OSError, sqlite3.Error, KeyboardInterrupt):
            temporary.unlink(missing_ok=True)
            raise

    def integrity_check(self, path: Path) -> str:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        if row is None:
            raise sqlite3.DatabaseError("connection database integrity check returned no result")
        return str(row[0])

    def restore_pre_v1_backup(self, backup: Path) -> None:
        if self.integrity_check(backup) != "ok":
            raise sqlite3.DatabaseError("connection backup integrity check failed")
        with closing(sqlite3.connect(f"file:{backup}?mode=ro", uri=True)) as source:
            row = source.execute(
                "SELECT facility_code,client_installation_ref,facility_id,facility_token,"
                "edge_installation_id,enrollment_generation,enrollment_created_at,"
                "enrollment_updated_at,updated_at FROM edge_site WHERE id=1"
            ).fetchone()
        self.write(_empty_data() if row is None else dict(zip(COLUMNS, row, strict=True)))

    def _connect(self) -> sqlite3.Connection:
        return open_configuration_database(self.path)


_RESET_TOPOLOGY_SQL: Final = (
    "UPDATE edge_site SET topology_snapshot_registry_version=0,topology_client_revision=0,"
    "topology_server_revision=0,topology_pending_snapshot_id=NULL,topology_pending_body=NULL,"
    "topology_pending_registry_version=NULL,topology_pending_client_revision=NULL,"
    "topology_pending_expected_server_revision=NULL,topology_consecutive_failures=0,"
    "topology_next_retry_at=NULL,topology_pause_reason=NULL,topology_last_accepted_at=NULL,"
    "topology_confirmation_id=NULL,topology_confirmation_digest=NULL,"
    "topology_confirmation_expires_at=NULL,topology_confirmation_snapshot_id=NULL,"
    "topology_confirmation_client_revision=NULL,topology_confirmation_server_revision=NULL,"
    "topology_confirmation_registry_version=NULL,topology_confirmation_cameras=NULL,"
    "topology_confirmation_rooms=NULL,topology_confirmation_floors=NULL,"
    "topology_confirmation_confirmed=NULL,topology_confirmation_result=NULL,updated_at=? WHERE id=1"
)


def _empty_data() -> ConnectionData:
    return {key: None for key in COLUMNS}


__all__ = [
    "COLUMNS",
    "ENROLLMENT_MARKER_FIELDS",
    "REQUIRED_ENROLLMENT_FIELDS",
    "SAVE_FIELDS",
    "ConnectionData",
    "ConnectionStoreBackup",
    "ConnectionStoreDatabase",
    "ConnectionWriteHook",
    "ConnectionValue",
    "utc_now_iso",
]

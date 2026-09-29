"""Column-scoped enrollment SQL on a borrowed PostgreSQL connection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, TypeAlias

import psycopg
from psycopg.rows import dict_row

from backend.app.edge_db.postgres import PostgresError

ConnectionValue: TypeAlias = str | int | None
ConnectionData: TypeAlias = dict[str, ConnectionValue]
ConnectionWriteHook: TypeAlias = Callable[[psycopg.Connection], None]

SAVE_FIELDS: Final = (
    "facility_code",
    "client_installation_ref",
    "facility_id",
    "facility_token",
    "edge_installation_id",
    "enrollment_generation",
)
COLUMNS: Final = (*SAVE_FIELDS, "enrollment_created_at", "enrollment_updated_at", "updated_at")
REQUIRED_ENROLLMENT_FIELDS: Final = SAVE_FIELDS
# Character counts and signed BIGINT range from postgres_product.sql, not bytes.
TEXT_FIELD_LIMITS: Final = {
    "facility_code": 64,
    "client_installation_ref": 128,
    "facility_id": 128,
    "facility_token": 512,
    "edge_installation_id": 128,
}
MAX_ENROLLMENT_GENERATION: Final = 2**63 - 1


class ConnectionSettingsNotInitialized(PostgresError):
    """Serving requires the singleton installed by bootstrap or migration."""

    def __init__(self) -> None:
        super().__init__("connection settings bootstrap row is missing")


_SELECT_SQL: Final = (
    "SELECT facility_code,client_installation_ref,facility_id,facility_token,"
    "edge_installation_id,enrollment_generation,enrollment_created_at,"
    "enrollment_updated_at,updated_at FROM edge_site WHERE id=1"
)
_WRITE_SQL: Final = (
    "UPDATE edge_site SET facility_code=%s,client_installation_ref=%s,facility_id=%s,"
    "facility_token=%s,edge_installation_id=%s,enrollment_generation=%s,"
    "enrollment_created_at=%s,enrollment_updated_at=%s,updated_at=%s WHERE id=1"
)
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
    "topology_confirmation_confirmed=NULL,topology_confirmation_result=NULL WHERE id=1"
)


def read_settings(connection: psycopg.Connection, *, for_update: bool = False) -> ConnectionData:
    """Read the bootstrapped singleton, optionally locking through the caller's commit."""
    query = _SELECT_SQL + (" FOR UPDATE" if for_update else "")
    # Audit and authority consumers require tuple rows on this same connection.
    with connection.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute(query).fetchone()
    if row is None:
        raise ConnectionSettingsNotInitialized()
    return row


def write_settings(
    connection: psycopg.Connection, data: ConnectionData, *, reset_topology: bool
) -> None:
    """Write a validated candidate under the caller's singleton row lock."""
    with connection.cursor() as cursor:
        cursor.execute(_WRITE_SQL, tuple(data[column] for column in COLUMNS))
        if cursor.rowcount != 1:
            raise ConnectionSettingsNotInitialized()
        if reset_topology:
            cursor.execute(_RESET_TOPOLOGY_SQL)


__all__ = [
    "COLUMNS",
    "MAX_ENROLLMENT_GENERATION",
    "REQUIRED_ENROLLMENT_FIELDS",
    "SAVE_FIELDS",
    "TEXT_FIELD_LIMITS",
    "ConnectionData",
    "ConnectionSettingsNotInitialized",
    "ConnectionValue",
    "ConnectionWriteHook",
    "read_settings",
    "write_settings",
]

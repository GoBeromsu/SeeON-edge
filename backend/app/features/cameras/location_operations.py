"""Location and topology operations mixed into the camera registry authority."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import psycopg
from psycopg.pq import TransactionStatus

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase, PostgresTransactionStateError
from backend.app.features.cameras.camera_repository import (
    CameraRegistryNotInitialized,
    CameraRegistryWriteError,
    lock_registry,
    migrate_legacy_floors,
    record_registry_mutation,
)
from backend.app.features.cameras.topology import CameraTopologyStore, RegistryTopologySnapshot

TransactionHook = Callable[[psycopg.Connection], None]
_Result = TypeVar("_Result")


class CameraLocationOperations:
    database: PostgresDatabase
    authority: AuthorityToken
    _topology: CameraTopologyStore

    def _mutate(self, operation: Callable[[psycopg.Connection], _Result]) -> _Result:
        """The common admission order for every camera/location mutation."""

        def persist(connection: psycopg.Connection) -> _Result:
            require_authority(connection, self.authority)
            lock_registry(connection)
            return operation(connection)

        try:
            return self.database.transact(persist)
        except (psycopg.IntegrityError, psycopg.DataError):
            # PostgreSQL constraint DETAIL can contain the full credential-bearing row.
            raise CameraRegistryWriteError() from None

    def create_floor(
        self,
        *,
        edge_ref: str,
        name: str,
        order_index: int,
        after_write: TransactionHook | None = None,
    ) -> None:
        def persist(connection: psycopg.Connection) -> None:
            self._topology.create_floor(
                connection, edge_ref=edge_ref, name=name, order_index=order_index
            )
            self._finish_mutation(connection, True, after_write)

        self._mutate(persist)

    def update_floor(
        self,
        edge_ref: str,
        *,
        name: str,
        order_index: int,
        after_write: TransactionHook | None = None,
    ) -> bool:
        def persist(connection: psycopg.Connection) -> bool:
            changed = self._topology.update_floor(
                connection, edge_ref, name=name, order_index=order_index
            )
            self._finish_mutation(connection, changed, after_write)
            return changed

        return self._mutate(persist)

    def delete_floor(self, edge_ref: str, *, after_write: TransactionHook | None = None) -> bool:
        return self._location_mutation(self._topology.delete_floor, edge_ref, after_write)

    def create_room(
        self,
        *,
        edge_ref: str,
        floor_edge_ref: str,
        name: str,
        legacy_canonical_space_id: str | None = None,
        after_write: TransactionHook | None = None,
    ) -> None:
        def persist(connection: psycopg.Connection) -> None:
            self._topology.create_room(
                connection,
                edge_ref=edge_ref,
                floor_edge_ref=floor_edge_ref,
                name=name,
                legacy_canonical_space_id=legacy_canonical_space_id,
            )
            self._finish_mutation(connection, True, after_write)

        self._mutate(persist)

    def update_room(
        self, edge_ref: str, *, name: str, after_write: TransactionHook | None = None
    ) -> bool:
        def persist(connection: psycopg.Connection) -> bool:
            changed = self._topology.update_room(connection, edge_ref, name=name)
            self._finish_mutation(connection, changed, after_write)
            return changed

        return self._mutate(persist)

    def delete_room(self, edge_ref: str, *, after_write: TransactionHook | None = None) -> bool:
        return self._location_mutation(self._topology.delete_room, edge_ref, after_write)

    def topology_snapshot(
        self, *, connection: psycopg.Connection | None = None
    ) -> RegistryTopologySnapshot:
        return self._read(self._topology.snapshot, connection)

    def camera_count(self, *, connection: psycopg.Connection | None = None) -> int:
        def count(current: psycopg.Connection) -> int:
            row = current.execute(
                "SELECT count(c.camera_id) FROM edge_site AS s "
                "LEFT JOIN cameras AS c ON true WHERE s.id=1 GROUP BY s.id"
            ).fetchone()
            if row is None:
                raise CameraRegistryNotInitialized()
            return row[0]

        return self._read(count, connection)

    def _read(
        self,
        query: Callable[[psycopg.Connection], _Result],
        connection: psycopg.Connection | None,
    ) -> _Result:
        if connection is None:
            return self.database.read(query)
        if connection.info.transaction_status is not TransactionStatus.INTRANS:
            raise PostgresTransactionStateError(
                "registry projection requires an active transaction"
            )
        return query(connection)

    def migrate_legacy_string_floors(self) -> list[dict[str, object]]:
        return self._mutate(migrate_legacy_floors)

    def _location_mutation(
        self,
        operation: Callable[[psycopg.Connection, str], bool],
        edge_ref: str,
        after_write: TransactionHook | None,
    ) -> bool:
        def persist(connection: psycopg.Connection) -> bool:
            changed = operation(connection, edge_ref)
            self._finish_mutation(connection, changed, after_write)
            return changed

        return self._mutate(persist)

    @staticmethod
    def _finish_mutation(
        connection: psycopg.Connection, changed: bool, after_write: TransactionHook | None
    ) -> None:
        if changed:
            record_registry_mutation(connection)
            if after_write is not None:
                after_write(connection)


__all__ = ["CameraLocationOperations"]

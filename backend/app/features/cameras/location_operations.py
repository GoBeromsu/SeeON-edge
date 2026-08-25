"""Location and topology operations mixed into the camera registry authority."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from threading import Lock

from backend.app.features.cameras.camera_repository import (
    camera_transaction,
    migrate_legacy_floors,
    read_registry,
    record_registry_mutation,
)
from backend.app.features.cameras.camera_values import CameraStatus
from backend.app.features.cameras.topology import CameraTopologyStore, RegistryTopologySnapshot

TransactionHook = Callable[[sqlite3.Connection], None]


class CameraLocationOperations:
    _lock: Lock
    _connection: sqlite3.Connection
    _topology: CameraTopologyStore
    _statuses: dict[str, CameraStatus]

    def create_floor(
        self, *, edge_ref: str, name: str, order_index: int,
        after_write: TransactionHook | None = None,
    ) -> None:
        with self._lock, camera_transaction(self._connection) as connection:
            self._topology.create_floor(
                connection, edge_ref=edge_ref, name=name, order_index=order_index
            )
            record_registry_mutation(connection)
            if after_write is not None:
                after_write(connection)

    def update_floor(
        self, edge_ref: str, *, name: str, order_index: int,
        after_write: TransactionHook | None = None,
    ) -> bool:
        with self._lock, camera_transaction(self._connection) as connection:
            changed = self._topology.update_floor(
                connection, edge_ref, name=name, order_index=order_index
            )
            self._finish_mutation(connection, changed, after_write)
            return changed

    def delete_floor(
        self, edge_ref: str, *, after_write: TransactionHook | None = None
    ) -> bool:
        return self._location_mutation(self._topology.delete_floor, edge_ref, after_write)

    def create_room(
        self, *, edge_ref: str, floor_edge_ref: str, name: str,
        legacy_canonical_space_id: str | None = None,
        after_write: TransactionHook | None = None,
    ) -> None:
        with self._lock, camera_transaction(self._connection) as connection:
            self._topology.create_room(
                connection, edge_ref=edge_ref, floor_edge_ref=floor_edge_ref, name=name,
                legacy_canonical_space_id=legacy_canonical_space_id,
            )
            record_registry_mutation(connection)
            if after_write is not None:
                after_write(connection)

    def update_room(
        self, edge_ref: str, *, name: str, after_write: TransactionHook | None = None
    ) -> bool:
        with self._lock, camera_transaction(self._connection) as connection:
            changed = self._topology.update_room(connection, edge_ref, name=name)
            self._finish_mutation(connection, changed, after_write)
            return changed

    def delete_room(
        self, edge_ref: str, *, after_write: TransactionHook | None = None
    ) -> bool:
        return self._location_mutation(self._topology.delete_room, edge_ref, after_write)

    def topology_snapshot(self) -> RegistryTopologySnapshot:
        with self._lock:
            data = read_registry(self._connection, self._statuses)
            return self._topology.snapshot(
                self._connection,
                registry_version=data["registry_version"],
                camera_ids=tuple(str(record["id"]) for record in data["cameras"]),
            )

    def migrate_legacy_string_floors(self) -> list[dict[str, object]]:
        with self._lock, camera_transaction(self._connection) as connection:
            return migrate_legacy_floors(connection)

    def _location_mutation(
        self, operation: Callable[[sqlite3.Connection, str], bool], edge_ref: str,
        after_write: TransactionHook | None,
    ) -> bool:
        with self._lock, camera_transaction(self._connection) as connection:
            changed = operation(connection, edge_ref)
            self._finish_mutation(connection, changed, after_write)
            return changed

    @staticmethod
    def _finish_mutation(
        connection: sqlite3.Connection, changed: bool, after_write: TransactionHook | None
    ) -> None:
        if changed:
            record_registry_mutation(connection)
            if after_write is not None:
                after_write(connection)


__all__ = ["CameraLocationOperations"]

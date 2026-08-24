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


class CameraLocationOperations:
    _lock: Lock
    _connection: sqlite3.Connection
    _topology: CameraTopologyStore
    _statuses: dict[str, CameraStatus]

    def create_floor(self, *, edge_ref: str, name: str, order_index: int) -> None:
        with self._lock, camera_transaction(self._connection) as connection:
            self._topology.create_floor(
                connection, edge_ref=edge_ref, name=name, order_index=order_index
            )
            record_registry_mutation(connection)

    def update_floor(self, edge_ref: str, *, name: str, order_index: int) -> bool:
        with self._lock, camera_transaction(self._connection) as connection:
            changed = self._topology.update_floor(
                connection, edge_ref, name=name, order_index=order_index
            )
            if changed:
                record_registry_mutation(connection)
            return changed

    def delete_floor(self, edge_ref: str) -> bool:
        return self._location_mutation(self._topology.delete_floor, edge_ref)

    def create_room(
        self,
        *,
        edge_ref: str,
        floor_edge_ref: str,
        name: str,
        legacy_canonical_space_id: str | None = None,
    ) -> None:
        with self._lock, camera_transaction(self._connection) as connection:
            self._topology.create_room(
                connection,
                edge_ref=edge_ref,
                floor_edge_ref=floor_edge_ref,
                name=name,
                legacy_canonical_space_id=legacy_canonical_space_id,
            )
            record_registry_mutation(connection)

    def update_room(self, edge_ref: str, *, name: str) -> bool:
        with self._lock, camera_transaction(self._connection) as connection:
            changed = self._topology.update_room(connection, edge_ref, name=name)
            if changed:
                record_registry_mutation(connection)
            return changed

    def delete_room(self, edge_ref: str) -> bool:
        return self._location_mutation(self._topology.delete_room, edge_ref)

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
        self,
        operation: Callable[[sqlite3.Connection, str], bool],
        edge_ref: str,
    ) -> bool:
        with self._lock, camera_transaction(self._connection) as connection:
            changed = operation(connection, edge_ref)
            if changed:
                record_registry_mutation(connection)
            return changed


__all__ = ["CameraLocationOperations"]

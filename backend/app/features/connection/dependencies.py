"""Controller-owned connection store and topology coordinator injection."""

from fastapi import FastAPI

from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import TopologyClient
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from backend.app.shared.backend_client_bundle import backend_client_bundle


def get_connection_settings_store(app: FastAPI) -> ConnectionSettingsStore:
    store = getattr(app.state, "connection_settings_store", None)
    if store is None:
        raise RuntimeError("connection settings store is not injected")
    if not isinstance(store, ConnectionSettingsStore):
        raise TypeError("connection settings store has invalid type")
    return store


def topology_retry_coordinator(app: FastAPI) -> TopologyRetryCoordinator:
    existing = getattr(app.state, "topology_retry_coordinator", None)
    if isinstance(existing, TopologyRetryCoordinator):
        return existing
    candidate = getattr(app.state, "camera_registry", None)
    if candidate is None:
        raise RuntimeError("camera registry is not injected")
    if not isinstance(candidate, CameraRegistryStore):
        raise TypeError("camera registry has invalid type")
    registry = candidate

    def client_provider() -> TopologyClient | None:
        bundle = backend_client_bundle(app)
        return None if bundle is None else TopologyClient.from_bundle(bundle)

    coordinator = TopologyRetryCoordinator(
        registry,
        EdgeTopologySyncStateStore(registry.database, registry.authority),
        client_provider,
    )
    app.state.topology_retry_coordinator = coordinator
    return coordinator


__all__ = ["get_connection_settings_store", "topology_retry_coordinator"]

"""Controller-owned runtime settings injection."""

from fastapi import FastAPI

from backend.app.features.runtime_settings.store import RuntimeSettingsStore


def get_runtime_settings_store(app: FastAPI) -> RuntimeSettingsStore:
    store = getattr(app.state, "runtime_settings_store", None)
    if store is None:
        raise RuntimeError("runtime settings store is not injected")
    if not isinstance(store, RuntimeSettingsStore):
        raise TypeError("runtime settings store has invalid type")
    return store


__all__ = ["get_runtime_settings_store"]

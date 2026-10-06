from __future__ import annotations

import importlib

CAMERA_STORE_EXPORTS = [
    "DEFAULT_FLOOR",
    "FLOOR_MAX",
    "FLOOR_MIN",
    "FLOOR_VALUES",
    "CameraRegistryStore",
    "CameraStatus",
    "DuplicateCameraError",
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


def test_camera_store_preserves_curated_public_exports() -> None:
    module = importlib.import_module("backend.app.features.cameras.store")
    namespace: dict[str, object] = {}

    exec("from backend.app.features.cameras.store import *", {}, namespace)

    assert module.__all__ == CAMERA_STORE_EXPORTS
    assert list(namespace) == CAMERA_STORE_EXPORTS
    assert all(namespace[name] is getattr(module, name) for name in CAMERA_STORE_EXPORTS)

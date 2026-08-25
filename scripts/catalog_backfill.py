#!/usr/bin/env python3
"""Idempotently load existing local clip-store sidecars into ml-api's catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.features.clips.audit_log import AuditLogStore
from backend.app.features.clips.catalog import (
    CatalogStore,
    sanitized_camera_payload,
    strict_camera_snapshot,
)
from backend.app.features.clips.store import ClipStore
from backend.app.shared.state_dir import resolve_state_dir

parser = argparse.ArgumentParser()
parser.add_argument(
    "--catalog",
    type=Path,
    default=resolve_state_dir("ml-api") / "catalog.sqlite3",
)
parser.add_argument("--clip-store", type=Path, required=True)
parser.add_argument("--audit", type=Path)
parser.add_argument(
    "--cameras",
    type=Path,
    default=resolve_state_dir("ml-api") / "cameras.json",
)
args = parser.parse_args()
store = CatalogStore.open(args.catalog)
try:
    camera_snapshot = strict_camera_snapshot(args.cameras)
    store.backfill(
        ClipStore(args.clip_store),
        audit_log=AuditLogStore(args.audit) if args.audit else None,
    )
    if camera_snapshot is not None:
        for camera in camera_snapshot["cameras"]:
            if isinstance(camera.get("id"), str) and camera["id"]:
                store.record("cameras", camera["id"], sanitized_camera_payload(camera))
    camera_source = str(args.cameras) if camera_snapshot is not None else "absent"
    print(
        f"catalog backfill complete camera_source={camera_source} "
        f"cameras={len(camera_snapshot['cameras']) if camera_snapshot else 0}"
    )
finally:
    store.close()

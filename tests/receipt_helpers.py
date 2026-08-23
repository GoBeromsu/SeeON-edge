from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import FastAPI

from backend.app.features.clips.store import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR, ClipStore
from backend.app.features.evidence.receipt_store import ArtifactReceipt


class MediaReceiptStore:
    def __init__(self) -> None:
        self.receipts: dict[str, ArtifactReceipt] = {}

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt:
        existing = self.receipts.setdefault(receipt.artifact_id, receipt)
        if existing != receipt:
            raise RuntimeError("test receipt conflict")
        return existing

    def get(self, artifact_id: str) -> ArtifactReceipt | None:
        return self.receipts.get(artifact_id)


def add_accepted_media_receipts(app: FastAPI) -> None:
    store = MediaReceiptStore()
    root = Path(os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))
    clips = ClipStore(root)
    for manifest in clips.list_manifests():
        if not manifest.video_available or manifest.path is None:
            continue
        located = clips.locate_manifest(manifest.clip_id)
        if located is None:
            continue
        try:
            path = clips._resolve_video_path(located.manifest, located.recording_root)
            descriptor = os.open(path, os.O_RDONLY)
            try:
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                content = b"".join(chunks)
            finally:
                os.close(descriptor)
        except (FileNotFoundError, ValueError):
            continue
        store.commit(
            ArtifactReceipt(manifest.clip_id, hashlib.sha256(content).hexdigest(), len(content))
        )
    app.state.artifact_receipt_store = store


__all__ = ["add_accepted_media_receipts"]

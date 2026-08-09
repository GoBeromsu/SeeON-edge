"""Application lifecycle for the derived clip listing projection."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
from anyio.to_thread import run_sync
from fastapi import FastAPI

from backend.app.features.clips.listing_index import (
    ClipListingIndex,
    ClipListingReconcileError,
)
from backend.app.features.clips.store import ClipStore

RECONCILE_INTERVAL_SEC = 5.0
logger = logging.getLogger(__name__)


@asynccontextmanager
async def maintain_clip_listing(app: FastAPI) -> AsyncGenerator[None]:
    store = getattr(app.state, "clip_store", None)
    if not isinstance(store, ClipStore):
        store = ClipStore.from_env()
        app.state.clip_store = store
    await _sync_once(app, store)
    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_reconcile_loop, app, store)
            try:
                yield
            finally:
                task_group.cancel_scope.cancel()
    finally:
        index = getattr(app.state, "clip_listing_index", None)
        if isinstance(index, ClipListingIndex):
            index.close()
            if getattr(app.state, "clip_listing_index", None) is index:
                del app.state.clip_listing_index


async def _reconcile_loop(app: FastAPI, store: ClipStore) -> None:
    while True:
        await anyio.sleep(RECONCILE_INTERVAL_SEC)
        await _sync_once(app, store)


async def _sync_once(app: FastAPI, store: ClipStore) -> None:
    index = getattr(app.state, "clip_listing_index", None)
    if isinstance(index, ClipListingIndex) and index.is_closed:
        del app.state.clip_listing_index
        index = None
    if not isinstance(index, ClipListingIndex):
        try:
            index = ClipListingIndex.from_env()
        except (OSError, sqlite3.Error):
            logger.exception("clip listing index open failed")
            return
        app.state.clip_listing_index = index
    try:
        _ = await run_sync(index.reconcile, store)
    except ClipListingReconcileError:
        logger.exception("clip listing reconciliation failed")


__all__ = ["RECONCILE_INTERVAL_SEC", "maintain_clip_listing"]

"""Application lifecycle adapters for the clip catalogue indexer."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI

from backend.app.features.clips.catalog_indexer import (
    CLIP_CATALOG_SHUTDOWN_WAIT_SEC,
    ClipCatalogIndexer,
    clip_catalog_interval_sec,
    run_clip_catalog_indexer,
)
from backend.app.features.clips.store import ClipStore


def _clip_store(app: FastAPI) -> ClipStore:
    store = getattr(app.state, "clip_store", None)
    if isinstance(store, ClipStore):
        return store
    store = ClipStore.from_env()
    app.state.clip_store = store
    return store


async def start_clip_catalog_indexer(app: FastAPI) -> None:
    """Index the store once (refusing startup on failure), then keep it indexed."""
    indexer = getattr(app.state, "clip_catalog_indexer", None)
    if indexer is None:
        raise RuntimeError("clip catalog indexer is not injected")
    if not isinstance(indexer, ClipCatalogIndexer):
        raise TypeError("clip catalog indexer has invalid type")
    interval = clip_catalog_interval_sec()
    store = _clip_store(app)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-catalog-indexer")
    try:
        outcome = await asyncio.get_running_loop().run_in_executor(
            executor, indexer.reconcile, store
        )
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    stop = asyncio.Event()
    app.state.clip_catalog_indexer_stop = stop
    app.state.clip_catalog_indexer_executor = executor
    app.state.clip_catalog_indexer_task = asyncio.create_task(
        run_clip_catalog_indexer(indexer, store, stop, executor, interval, outcome.remaining),
        name="clip-catalog-indexer",
    )


async def stop_clip_catalog_indexer(app: FastAPI) -> None:
    stop = getattr(app.state, "clip_catalog_indexer_stop", None)
    task = getattr(app.state, "clip_catalog_indexer_task", None)
    executor = getattr(app.state, "clip_catalog_indexer_executor", None)
    if isinstance(stop, asyncio.Event):
        stop.set()
    if isinstance(task, asyncio.Task):
        try:
            await asyncio.wait_for(asyncio.shield(task), CLIP_CATALOG_SHUTDOWN_WAIT_SEC)
        except TimeoutError:
            task.cancel()
    if isinstance(executor, ThreadPoolExecutor):
        executor.shutdown(wait=False, cancel_futures=True)
    app.state.clip_catalog_indexer_stop = None
    app.state.clip_catalog_indexer_task = None
    app.state.clip_catalog_indexer_executor = None


__all__ = ["start_clip_catalog_indexer", "stop_clip_catalog_indexer"]

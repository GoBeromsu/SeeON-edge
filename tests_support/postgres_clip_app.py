"""Index a no-lifespan app's clip store into its PostgreSQL clip catalogue.

GET /clips reads only the catalogue; the lifespan indexes the store at startup
and then on an interval. Route tests that skip the lifespan run the same
``reconcile`` the lifespan runs, explicitly and without a clock, after writing
manifests and before listing.
"""

from __future__ import annotations

from fastapi import FastAPI

from backend.app.features.clips.catalog_indexer import ClipCatalogIndexer, ReconcileOutcome
from backend.app.features.clips.store import ClipStore

MAX_INDEX_PASSES = 64


def app_clip_store(app: FastAPI) -> ClipStore:
    """The clip store the routes serve, built from the environment on first use."""
    store = getattr(app.state, "clip_store", None)
    if not isinstance(store, ClipStore):
        store = ClipStore.from_env()
        app.state.clip_store = store
    return store


def index_clips(app: FastAPI) -> tuple[ReconcileOutcome, ...]:
    """Reconcile until nothing is left to examine; return every pass's outcome."""
    indexer = app.state.clip_catalog_indexer
    assert isinstance(indexer, ClipCatalogIndexer)
    store = app_clip_store(app)
    outcomes: list[ReconcileOutcome] = []
    for _ in range(MAX_INDEX_PASSES):
        outcome = indexer.reconcile(store)
        outcomes.append(outcome)
        if outcome.remaining == 0:
            return tuple(outcomes)
    raise AssertionError(f"clip catalogue did not converge in {MAX_INDEX_PASSES} passes")

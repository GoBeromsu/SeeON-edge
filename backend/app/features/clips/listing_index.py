"""Concurrency-safe facade for derived clip listing generations."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import final

from pydantic import ValidationError

from backend.app.features.clips.listing import ClipPage
from backend.app.features.clips.listing_generation import (
    ClipListingPreparationError,
    ReconcileStats,
    prepare_generation,
)
from backend.app.features.clips.listing_queries import QueryPlans
from backend.app.features.clips.listing_repository import (
    ListingRepository,
    ListingRepositoryClosedError,
)
from backend.app.features.clips.schemas import ClipListQuery
from backend.app.features.clips.store import ClipStore
from shared.edge_db import EDGE_DATABASE_PATH


@final
class ClipListingReconcileError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class ClipListingIndex:
    def __init__(self, repository: ListingRepository) -> None:
        self._repository = repository
        self._reconcile_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._ready = False
        self._closed = False

    @classmethod
    def open(cls, path: Path | str) -> ClipListingIndex:
        return cls(ListingRepository.open(path))

    @classmethod
    def from_env(cls) -> ClipListingIndex:
        return cls.open(EDGE_DATABASE_PATH)

    @property
    def is_closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def reconcile(self, clip_store: ClipStore) -> ReconcileStats:
        with self._reconcile_lock:
            self._require_open()
            try:
                existing = self._repository.active_clips()
                prepared = prepare_generation(clip_store, existing)
                self._repository.publish(prepared)
            except (
                OSError,
                sqlite3.Error,
                ValidationError,
                ClipListingPreparationError,
                ListingRepositoryClosedError,
            ) as exc:
                self._repository.rollback()
                self._set_ready(False)
                raise ClipListingReconcileError(str(exc)) from exc
            self._set_ready(True)
            return prepared.stats

    def page(self, query: ClipListQuery) -> ClipPage:
        self._require_ready(query)
        try:
            return self._repository.page(query)
        except (sqlite3.Error, ValidationError, ListingRepositoryClosedError) as exc:
            self._set_ready(False)
            raise ClipListingReconcileError(str(exc)) from exc

    def explain(self, query: ClipListQuery) -> QueryPlans:
        self._require_ready(query)
        try:
            return self._repository.explain(query)
        except (sqlite3.Error, ValidationError, ListingRepositoryClosedError) as exc:
            self._set_ready(False)
            raise ClipListingReconcileError(str(exc)) from exc

    def close(self) -> None:
        with self._reconcile_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                self._ready = False
            self._repository.close()

    def _require_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise ClipListingReconcileError("clip listing index is closed")

    def _require_ready(self, query: ClipListQuery) -> None:
        with self._state_lock:
            if self._closed or not self._ready or query.limit is None:
                raise ClipListingReconcileError("clip listing index has not synchronized")

    def _set_ready(self, ready: bool) -> None:
        with self._state_lock:
            if not self._closed:
                self._ready = ready


__all__ = ["ClipListingIndex", "ClipListingReconcileError", "ReconcileStats"]

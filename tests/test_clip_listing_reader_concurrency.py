"""Schema-17 listing-reader concurrency fixtures target a removed repository."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_legacy_listing_reader_fixtures_have_no_production_target() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_queries")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_repository")
    assert not (
        Path(__file__).resolve().parents[1]
        / "backend/app/features/clips/listing_repository.py"
    ).exists()

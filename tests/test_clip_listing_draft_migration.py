"""Draft schema-17 listing migration helpers are absent."""

from __future__ import annotations

import importlib

import pytest


def test_listing_schema_migration_module_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_schema_migration")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_repository")

"""Derived listing snapshots are not a serving surface."""

from __future__ import annotations

import importlib

import pytest


def test_listing_generation_helpers_are_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_generation")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_schema")

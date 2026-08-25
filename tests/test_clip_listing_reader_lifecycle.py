"""Schema-17 listing repository lifecycle is absent."""

from __future__ import annotations

import importlib

import pytest


def test_listing_repository_module_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_repository")

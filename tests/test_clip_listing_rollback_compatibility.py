"""Schema-17 listing generations are not a serving compatibility surface."""

from __future__ import annotations

import importlib

import pytest


def test_schema17_listing_cluster_is_absent() -> None:
    for module_name in (
        "backend.app.features.clips.listing_index",
        "backend.app.features.clips.listing_repository",
        "backend.app.features.clips.listing_generation",
        "backend.app.features.clips.listing_schema",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)

from __future__ import annotations

import importlib

import pytest


def test_qa_comparison_store_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.store")

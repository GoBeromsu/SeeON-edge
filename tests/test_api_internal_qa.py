from __future__ import annotations

import importlib

import pytest


def test_qa_store_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.store")


def test_qa_runtime_trace_store_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.runtime_trace_store")

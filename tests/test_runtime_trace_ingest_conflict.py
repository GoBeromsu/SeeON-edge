from __future__ import annotations

import importlib

import pytest

from backend.app.edge_db.migration.mapping import DIAGNOSTICS_TARGET_TABLES, EXPECTED_TARGET_TABLES


def test_runtime_analysis_store_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.runtime_trace_store")


def test_product_schema_has_no_runtime_analysis_tables() -> None:
    # Provisioning checks the PostgreSQL product and diagnostics schemas against these sets.
    tables = EXPECTED_TARGET_TABLES | DIAGNOSTICS_TARGET_TABLES
    assert not any(name.startswith("runtime_analysis_") for name in tables)
    assert not any(name.startswith("qa_") for name in tables)

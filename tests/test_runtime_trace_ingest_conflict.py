from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.migrator import migrate_database


def test_runtime_analysis_store_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.runtime_trace_store")


def test_schema18_has_no_runtime_analysis_tables(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert not any(name.startswith("runtime_analysis_") for name in tables)
    assert not any(name.startswith("qa_") for name in tables)

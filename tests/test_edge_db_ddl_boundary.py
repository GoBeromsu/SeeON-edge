"""Serving/runtime compatibility must validate schema 18 without the DDL ledger.

The runtime schema check (`backend.app.edge_db.compatibility`) is reachable from
every API and worker process that opens the edge database. It must be able to
prove the schema-18 identity/table manifest without importing any historical
v1-v18 DDL ledger module. The migrator is the sole owner and reachable consumer
of that ledger.
"""

from __future__ import annotations

import subprocess
import sys

# Historical v1-v18 DDL ledger modules. Serving compatibility must reach none of
# them; only the migrator may.
_HISTORICAL_DDL_MODULES = (
    "backend.app.edge_db.schema",
    "backend.app.edge_db.migrator",
    "backend.app.edge_db.application_schema",
    "backend.app.edge_db.evidence_backfill",
    "backend.app.edge_db.review_migration",
)


def _modules_loaded_by(import_target: str) -> frozenset[str]:
    """Return the DDL ledger modules a fresh interpreter loads importing target."""
    probe = (
        "import sys\n"
        f"import {import_target}\n"
        "import json\n"
        f"loaded = [m for m in {_HISTORICAL_DDL_MODULES!r} if m in sys.modules]\n"
        "print(json.dumps(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return frozenset(json.loads(completed.stdout.strip().splitlines()[-1]))


def test_compatibility_import_does_not_reach_historical_ddl_ledger() -> None:
    # Given: a clean interpreter that imports only serving compatibility.
    # When: the runtime schema-compatibility module is imported.
    loaded = _modules_loaded_by("backend.app.edge_db.compatibility")
    # Then: no historical v1-v18 DDL ledger module is pulled into the process.
    assert loaded == frozenset()


def test_schema18_manifest_import_does_not_reach_historical_ddl_ledger() -> None:
    # Given: a clean interpreter that imports only the schema-18 manifest leaf.
    # When: the manifest module is imported.
    loaded = _modules_loaded_by("backend.app.edge_db.schema18_manifest")
    # Then: it compiles the schema-18 identity from current-schema DDL alone.
    assert loaded == frozenset()


def test_migrator_remains_a_ddl_ledger_consumer() -> None:
    # Given: a clean interpreter that imports the migrator.
    # When: the migrator module is imported.
    loaded = _modules_loaded_by("backend.app.edge_db.migrator")
    # Then: the migrator still owns and reaches the historical DDL ledger,
    # proving the boundary test above is not vacuously green.
    assert "backend.app.edge_db.schema" in loaded

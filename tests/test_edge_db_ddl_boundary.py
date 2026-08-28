"""Serving/runtime compatibility must validate schema 18 without the DDL owner.

The runtime schema check (`backend.app.edge_db.compatibility`) is reachable from
every API process that opens the edge database. It proves the schema-18
structural manifest from the current-schema DDL alone and never imports the
create-only bootstrap, which is the sole module allowed to execute DDL.
"""

from __future__ import annotations

import json
import subprocess
import sys

_DDL_OWNER_MODULES = ("backend.app.edge_db.bootstrap",)


def _modules_loaded_by(import_target: str) -> frozenset[str]:
    """Return the DDL-owner modules a fresh interpreter loads importing target."""
    probe = (
        "import sys\n"
        f"import {import_target}\n"
        "import json\n"
        f"loaded = [m for m in {_DDL_OWNER_MODULES!r} if m in sys.modules]\n"
        "print(json.dumps(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(json.loads(completed.stdout.strip().splitlines()[-1]))


def test_compatibility_import_does_not_reach_the_bootstrap() -> None:
    assert _modules_loaded_by("backend.app.edge_db.compatibility") == frozenset()


def test_schema18_manifest_import_does_not_reach_the_bootstrap() -> None:
    assert _modules_loaded_by("backend.app.edge_db.schema18_manifest") == frozenset()


def test_package_entrypoint_reaches_the_bootstrap() -> None:
    # Proves the boundary tests above are not vacuously green.
    assert _modules_loaded_by("backend.app.edge_db.bootstrap") == frozenset(_DDL_OWNER_MODULES)

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLLBACK_RUNBOOK = ROOT / "docs" / "runbooks" / "worker-migration-rollback.md"


def _shell_commands(path: Path) -> str:
    return "\n".join(
        re.findall(r"```sh\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    )


def test_migration_commands_enforce_migrator_api_worker_order() -> None:
    commands = _shell_commands(ROLLBACK_RUNBOOK)
    inventory = "$DC up --pull always edge-filesystem-inventory"
    migrator = "$DC up --pull always edge-db-migrator"
    api = "$DC up -d --wait ml-api"
    worker = "$DC up -d --wait ml-worker"

    assert "/var/lib/seeon-state/edge.sqlite3" not in commands
    assert inventory in commands
    assert migrator in commands
    assert api in commands
    assert worker in commands
    assert commands.index(inventory) < commands.index(migrator) < commands.index(api) < (
        commands.index(worker)
    )
    assert "down -v" not in commands
    assert not re.search(r"\b(?:CREATE|ALTER|DROP)\s+(?:TABLE|INDEX)\b", commands, re.IGNORECASE)

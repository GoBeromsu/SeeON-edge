"""Every worker module must import.

A module that raises on import is invisible to the rest of the suite once
nothing imports it any more. That is exactly how two landmines survived the P2
deletion: `worker/runtime/clip_deletion_control.py` imported a deleted retention
module, and its own test had been deleted alongside the tree it covered, so
nothing ever loaded it again. This sweep is the check that catches the next one.

`worker.tools` is excluded deliberately: those modules are build-time entry
points that do real work at import (fetching and verifying model weights), so
importing them here would turn a fast structural check into a network-dependent
one.
"""

from __future__ import annotations

import importlib
import pkgutil

EXCLUDED_PREFIX = "worker.tools"


def test_every_worker_module_imports_cleanly() -> None:
    failures: list[tuple[str, str]] = []
    for module in pkgutil.walk_packages(["worker"], "worker."):
        if module.name.startswith(EXCLUDED_PREFIX):
            continue
        try:
            importlib.import_module(module.name)
        except Exception as error:  # noqa: BLE001 - report every breakage, not the first
            failures.append((module.name, f"{type(error).__name__}: {error}"))

    assert not failures, "worker modules that fail to import:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in failures
    )

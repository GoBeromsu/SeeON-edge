"""Cross-platform-writable state directory resolution for ml-api.

Single rule, no environment-variable override: state lives under
``~/.local/state/<runtime>/`` (XDG State convention) on every OS. This
absorbs the 4 duplicated ``API_*_STORE`` env-var/default-path pairs that
previously lived one per store module.

This module is itself a leaf: it imports nothing from
``backend.app.features``/``routes``/``main``/``lifespan``, satisfying the
"backend base (core/shared) does not import upper layers" import-linter
contract (``pyproject.toml``) from the other direction -- callers in
``features``/``lifespan`` are free to import this module.
"""

from __future__ import annotations

from pathlib import Path


def resolve_state_dir(runtime: str = "ml-api") -> Path:
    """Return the default state directory for ``runtime``."""
    return Path.home() / ".local" / "state" / runtime


__all__ = ["resolve_state_dir"]

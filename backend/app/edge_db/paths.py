"""Location of the retired edge SQLite file, read only by the one-time migration."""

from __future__ import annotations

from pathlib import Path
from typing import Final

EDGE_STATE_DIRECTORY: Final = Path("/var/lib/seeon-state")
EDGE_DATABASE_PATH: Final = EDGE_STATE_DIRECTORY / "edge.sqlite3"

__all__ = ["EDGE_DATABASE_PATH", "EDGE_STATE_DIRECTORY"]

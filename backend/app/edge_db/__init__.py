"""SQLite-free public surface of the edge database package.

Only ``backend.app.edge_db.migration`` reads the retired ``edge.sqlite3``.
"""

from backend.app.edge_db.paths import EDGE_DATABASE_PATH, EDGE_STATE_DIRECTORY

__all__ = ["EDGE_DATABASE_PATH", "EDGE_STATE_DIRECTORY"]

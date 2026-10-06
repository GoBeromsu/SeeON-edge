"""SQLite-free public surface of the edge database package.

Only ``backend.app.edge_db.migration`` reads the retired ``edge.sqlite3``.
"""

from psycopg import DataError
from psycopg import Error as DatabaseDriverError
from psycopg.errors import CheckViolation, NotNullViolation

from backend.app.edge_db.paths import EDGE_DATABASE_PATH, EDGE_STATE_DIRECTORY

__all__ = [
    "EDGE_DATABASE_PATH",
    "EDGE_STATE_DIRECTORY",
    "CheckViolation",
    "DataError",
    "DatabaseDriverError",
    "NotNullViolation",
]

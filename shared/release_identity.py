"""Release-wide identities consumed by independently deployed runtimes."""

from __future__ import annotations

from typing import Final

# This is the schema release this software was built to interoperate with. It
# deliberately does not claim the version currently installed on an edge unit.
EDGE_DATABASE_SCHEMA_VERSION: Final = 17


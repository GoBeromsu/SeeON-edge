"""Release-wide identities consumed by independently deployed runtimes."""

from __future__ import annotations

from typing import Final

# This is the schema release this software was built to interoperate with. It
# deliberately does not claim the version currently installed on an edge unit.
EDGE_DATABASE_FORMAT_IDENTITY: Final = "seeon-edge-v1"
EDGE_DATABASE_SCHEMA_VERSION: Final = 18


class ReleaseIdentityMismatchError(RuntimeError):
    """Two independently deployed images do not share one schema identity."""

    def __init__(self, found: int, expected: int = EDGE_DATABASE_SCHEMA_VERSION) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"edge database schema identity {found} does not match required {expected}"
        )


def require_peer_schema_identity(
    found: int,
    *,
    expected: int = EDGE_DATABASE_SCHEMA_VERSION,
) -> None:
    """Refuse mixed 17/18 image pairs before either runtime serves traffic."""
    if found != expected:
        raise ReleaseIdentityMismatchError(found, expected)


__all__ = [
    "EDGE_DATABASE_FORMAT_IDENTITY",
    "EDGE_DATABASE_SCHEMA_VERSION",
    "ReleaseIdentityMismatchError",
    "require_peer_schema_identity",
]

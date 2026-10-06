"""Transactional product writer/claimer fence; process/network quiescence is separate."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import psycopg
from psycopg.pq import TransactionStatus

from backend.app.edge_db.postgres import PostgresDatabase


class AuthorityFenced(RuntimeError):
    """The caller no longer owns durable acceptance or delivery admission."""


@dataclass(frozen=True, slots=True, repr=False)
class AuthorityToken:
    generation: int
    writer_token: UUID

    def __post_init__(self) -> None:
        if type(self.generation) is not int or not 0 < self.generation < 2**63:
            raise ValueError("authority generation must be a positive signed 64-bit integer")
        if not isinstance(self.writer_token, UUID):
            raise TypeError("authority writer token must be a UUID")

    def __repr__(self) -> str:
        return f"AuthorityToken(generation={self.generation}, writer_token=<redacted>)"


def require_authority(
    connection: psycopg.Connection, token: AuthorityToken, *, sender: bool = False
) -> None:
    """Hold a shared authority row lock through the caller's actual COMMIT.

    The migration owner takes an exclusive lock, so an old transaction either
    commits before the fence or observes it and refuses. This does not cancel
    an HTTP request already sent after an earlier claim commit.
    """
    if connection.info.transaction_status is not TransactionStatus.INTRANS:
        raise AuthorityFenced("authority checks require an owned transaction")
    row = connection.execute(
        "SELECT generation, writer_token, accepting, egress_enabled "
        "FROM deployment_authority WHERE singleton = 1 FOR SHARE"
    ).fetchone()
    if row is None or row[:2] != (token.generation, token.writer_token):
        raise AuthorityFenced("persistence authority no longer belongs to this caller")
    if not row[2] or (sender and not row[3]):
        raise AuthorityFenced("durable acceptance or egress is fenced")


def freeze_authority(database: PostgresDatabase, token: AuthorityToken) -> int:
    """Block later writers/claimers and wait for earlier DB transactions.

    Does not transfer authority or certify sender quiescence. The deployment
    owner must stop/drain processes, account for every in-flight delivery, and
    snapshot/reconcile before issuing a different generation/token.
    """

    def freeze(connection: psycopg.Connection) -> int:
        row = connection.execute(
            "SELECT generation, writer_token FROM deployment_authority "
            "WHERE singleton = 1 FOR UPDATE"
        ).fetchone()
        if row != (token.generation, token.writer_token):
            raise AuthorityFenced("cannot fence a different persistence authority")
        connection.execute(
            "UPDATE deployment_authority SET accepting = false, egress_enabled = false "
            "WHERE singleton = 1"
        )
        return token.generation

    return database.transact(freeze)

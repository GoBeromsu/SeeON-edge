"""Native PostgreSQL transactions; no schema bootstrap or application ACK policy.

Callbacks receive a borrowed connection and must not retain it, change session
settings, or manage transactions themselves. A returned value is released only
after COMMIT and the pool connection context have both exited successfully.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import TypeVar

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout, TooManyRequests

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class PoolBudget:
    max_connections: int
    max_waiting: int
    acquire_timeout_sec: float
    statement_timeout_ms: int
    lock_timeout_ms: int
    startup_timeout_sec: float

    def __post_init__(self) -> None:
        for name in (
            "max_connections",
            "max_waiting",
            "statement_timeout_ms",
            "lock_timeout_ms",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= 2_147_483_647:
                raise ValueError(f"{name} must be a positive 32-bit integer")
        for name in ("acquire_timeout_sec", "startup_timeout_sec"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value <= 2_147_483_647:
                raise ValueError(f"{name} must be positive and finite")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")


class PostgresError(RuntimeError):
    """A privacy-safe database owner failure."""


class PostgresStartupError(PostgresError):
    """The bounded startup did not establish a safe connection."""


class PostgresUnavailable(PostgresError):
    """No usable database connection is available."""


class PostgresPoolBusy(PostgresUnavailable):
    """The bounded acquisition queue or acquisition deadline was exhausted."""


class PostgresTransactionStateError(PostgresError):
    """A callback left its transaction aborted or managed it itself."""


class CommitOutcomeUnknown(PostgresError):
    """COMMIT lost its connection; neither success nor rollback is established."""

    def __init__(self) -> None:
        super().__init__("PostgreSQL commit outcome is unknown; automatic retry is forbidden")


class _PrivateConnection(psycopg.Connection):
    """Keep pool retry/discard logging from printing libpq connection details."""

    def __repr__(self) -> str:
        return "<PostgresConnection redacted>"

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs) -> _PrivateConnection:
        try:
            return super().connect(conninfo, **kwargs)
        except (psycopg.Error, OSError, ValueError, TypeError):
            # psycopg_pool logs connection exceptions during background retries.
            # In particular, libpq errors can include host/user/database strings.
            raise psycopg.OperationalError("PostgreSQL connection failed") from None


class PostgresDatabase:
    def __init__(self, conninfo: str, schema: str, budget: PoolBudget) -> None:
        if not isinstance(conninfo, str):
            raise TypeError("PostgreSQL connection information must be text")
        if not conninfo.strip() or "\x00" in conninfo:
            raise ValueError("PostgreSQL connection information must be nonblank without NUL bytes")
        if (
            not isinstance(schema, str)
            or not schema
            or "\x00" in schema
            or len(schema.encode("utf-8")) > 63
        ):
            raise ValueError("PostgreSQL schema must be a nonempty identifier of at most 63 bytes")
        self._schema = schema
        self._budget = budget
        self._lock = RLock()
        self._started = False
        self._closed = False
        try:
            self._pool = ConnectionPool(
                conninfo=conninfo,
                connection_class=_PrivateConnection,
                kwargs={
                    "autocommit": True,
                    "connect_timeout": max(2, math.ceil(budget.startup_timeout_sec)),
                    "application_name": "seeon-edge",
                },
                min_size=1,
                max_size=budget.max_connections,
                max_waiting=budget.max_waiting,
                timeout=budget.acquire_timeout_sec,
                num_workers=1,
                open=False,
                name="seeon-postgres",
                configure=self._configure,
                reset=self._configure,
                reconnect_timeout=budget.startup_timeout_sec,
            )
        except (psycopg.Error, OSError, ValueError, TypeError):
            raise PostgresStartupError("PostgreSQL pool configuration failed") from None

    def __repr__(self) -> str:
        return "<PostgresDatabase redacted>"

    @property
    def schema(self) -> str:
        """Return the configured namespace, never connection information."""
        return self._schema

    def _configure(self, connection: psycopg.Connection) -> None:
        try:
            connection.execute(
                "SELECT pg_catalog.set_config('TimeZone', 'UTC', false), "
                "pg_catalog.set_config('synchronous_commit', 'on', false), "
                "pg_catalog.set_config('statement_timeout', %s, false), "
                "pg_catalog.set_config('lock_timeout', %s, false), "
                "pg_catalog.set_config('idle_in_transaction_session_timeout', %s, false)",
                (
                    str(self._budget.statement_timeout_ms),
                    str(self._budget.lock_timeout_ms),
                    str(self._budget.statement_timeout_ms),
                ),
            )
            # An omitted pg_temp is implicitly searched before permanent relations.
            connection.execute(
                sql.SQL("SET search_path TO {}, pg_catalog, pg_temp").format(
                    sql.Identifier(self._schema)
                )
            )
            row = connection.execute(
                "SELECT pg_catalog.current_setting('fsync'), "
                "pg_catalog.current_setting('full_page_writes'), "
                "pg_catalog.current_setting('synchronous_commit'), "
                "pg_catalog.current_schema(), "
                "pg_catalog.current_setting('server_encoding'), "
                "pg_catalog.current_setting('session_replication_role')"
            ).fetchone()
        except (psycopg.Error, OSError, ValueError, TypeError):
            # This callback also runs on the pool's background worker.
            raise PostgresStartupError("PostgreSQL connection configuration is unsafe") from None
        if row != ("on", "on", "on", self._schema, "UTF8", "origin"):
            raise PostgresStartupError("PostgreSQL durability or namespace configuration is unsafe")

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise PostgresStartupError("PostgreSQL database owner is closed")
            if self._started:
                return
            try:
                self._pool.open(wait=False)
                # Pool.wait() performs its own close with a separate default
                # timeout on failure. Acquisition lets this owner bound both
                # the startup wait and the cleanup explicitly.
                with self._pool.connection(timeout=self._budget.startup_timeout_sec):
                    pass
            except BaseException as error:
                self._closed = True
                self._pool.close(timeout=self._budget.startup_timeout_sec)
                if not isinstance(error, Exception):
                    raise
                raise PostgresStartupError("PostgreSQL bounded startup failed") from None
            self._started = True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
            self._pool.close(timeout=self._budget.startup_timeout_sec)

    def stats(self) -> dict[str, int]:
        """Return pool counters and gauges, never connection information."""
        return self._pool.get_stats()

    def _require_started(self) -> None:
        with self._lock:
            if not self._started or self._closed:
                raise PostgresUnavailable("PostgreSQL database owner is not running")

    @staticmethod
    def _rollback(connection: psycopg.Connection) -> None:
        try:
            connection.rollback()
        except (psycopg.Error, OSError):
            # Preserve the callback's exception, but never reuse an uncertain
            # connection or allow context exit to commit its partial work.
            connection.close()

    @staticmethod
    def _require_active_transaction(connection: psycopg.Connection) -> None:
        if connection.info.transaction_status is not TransactionStatus.INTRANS:
            raise PostgresTransactionStateError(
                "PostgreSQL callback did not leave an active, successful transaction"
            )

    def _run(self, callback: Callable[[psycopg.Connection], _Result], *, begin: str) -> _Result:
        self._require_started()
        try:
            with self._pool.connection(timeout=self._budget.acquire_timeout_sec) as connection:
                connection.execute(begin)
                try:
                    result = callback(connection)
                    self._require_active_transaction(connection)
                except BaseException:
                    self._rollback(connection)
                    raise
                try:
                    connection.commit()
                except (psycopg.OperationalError, psycopg.InterfaceError):
                    connection.close()
                    raise CommitOutcomeUnknown() from None
        except (PoolTimeout, TooManyRequests):
            raise PostgresPoolBusy("PostgreSQL acquisition budget exhausted") from None
        except (PoolClosed, psycopg.OperationalError, psycopg.InterfaceError):
            raise PostgresUnavailable("PostgreSQL connection unavailable") from None
        else:
            # Never return from inside either context manager.
            return result

    def read(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        """Run one READ COMMITTED, READ ONLY transaction without replay."""
        return self._run(callback, begin="BEGIN ISOLATION LEVEL READ COMMITTED, READ ONLY")

    def read_snapshot(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        """Run one stable REPEATABLE READ, READ ONLY transaction without replay."""
        return self._run(callback, begin="BEGIN ISOLATION LEVEL REPEATABLE READ, READ ONLY")

    def transact(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        """Run one READ COMMITTED, READ WRITE transaction without replay.

        Return only after known successful commit and pool release.
        IntegrityError (including a deferred constraint rejected at COMMIT) is a
        known failure. CommitOutcomeUnknown must not be interpreted as rollback
        or retried blindly. Application idempotency/fencing belongs to callers.
        """
        # Admission serializes with advisory locks, then reads committed usage.
        # A deployment's REPEATABLE READ default would retain a pre-lock snapshot
        # and defeat that serialization. Snapshot reads are explicitly separate.
        return self._run(callback, begin="BEGIN ISOLATION LEVEL READ COMMITTED, READ WRITE")


__all__ = [
    "CommitOutcomeUnknown",
    "PoolBudget",
    "PostgresDatabase",
    "PostgresError",
    "PostgresPoolBusy",
    "PostgresStartupError",
    "PostgresTransactionStateError",
    "PostgresUnavailable",
]

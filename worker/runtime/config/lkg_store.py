from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from pydantic import JsonValue

from worker.pipeline.output.evidence.evidence_outbox_database import open_connection
from worker.pipeline.output.evidence.evidence_outbox_types import NewerSchemaVersionError
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction
from worker.runtime.config.restart import RestartDirective
from worker.runtime.state_dir import resolve_state_dir

# open_connection can fail three ways that must all degrade the LKG to
# "unavailable" rather than crash the worker: OSError (e.g. an unwritable/
# uncreatable state dir -- `path.parent.mkdir(...)` is unguarded), sqlite3.Error
# (PRAGMA/migration failure), and NewerSchemaVersionError (a plain Exception
# subclass, not a sqlite3.Error -- raised when a downgraded worker binary opens
# a database a newer binary already migrated past).
_STORE_UNAVAILABLE_ERRORS: Final = (OSError, sqlite3.Error, NewerSchemaVersionError)

# Same DB file the evidence outbox opens (worker/runtime/worker.py's
# `_evidence_outbox_path`, worker/pipeline/output/evidence/evidence_runtime.py) --
# config_current/config_history are tables in that shared worker-state.sqlite3,
# not a separate file, so they are locally joinable against evidence_events.
WORKER_STATE_DB_FILENAME: Final = "worker-state.sqlite3"

# config_history retention: rows still referenced by an evidence_events row
# that hasn't reached delivery_state='ACKED' are always kept (their
# audit.config_version must stay locally joinable until that evidence is
# acknowledged), plus the most recent N rows overall regardless of ack state,
# so routine operation doesn't unboundedly grow the table.
CONFIG_HISTORY_RETENTION_COUNT: Final = 50

JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StoredConfigPayload:
    payload: JsonObject
    directive: RestartDirective
    registry_version: int


class WorkerConfigLkgStore:
    """Last-known-good worker config, backed by the `config_current` /
    `config_history` tables in worker-state.sqlite3 (see
    worker/pipeline/output/evidence/evidence_outbox_schema.py's
    SCHEMA_V5_STATEMENTS). Replaces the prior JSON-file LKG store: no JSON
    import happens anywhere -- a worker without a pre-existing
    worker-state.sqlite3 simply has no config_current row until its first
    successful save, matching first-boot behavior before this change."""

    def __init__(self, database_path: Path | None = None, *, state_dir: Path | None = None) -> None:
        self.database_path: Path = (
            _worker_state_db_path(state_dir) if database_path is None else database_path
        )
        self._lock: threading.Lock = threading.Lock()

    def save(self, payload: JsonObject, directive: RestartDirective) -> bool:
        registry_version = _registry_version(payload)
        payload_text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        saved_at = time.time()
        with self._lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = open_connection(self.database_path)
                with ImmediateTransaction(connection):
                    current = _read_current(connection)
                    if current is not None and (directive, registry_version) < (
                        current.directive,
                        current.registry_version,
                    ):
                        return False
                    connection.execute(
                        """
                        INSERT INTO config_current
                            (id, generation, config_version, registry_version,
                             payload_json, saved_at)
                        VALUES (1, ?, ?, ?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET
                            generation = excluded.generation,
                            config_version = excluded.config_version,
                            registry_version = excluded.registry_version,
                            payload_json = excluded.payload_json,
                            saved_at = excluded.saved_at
                        """,
                        (
                            directive.generation,
                            directive.version,
                            registry_version,
                            payload_text,
                            saved_at,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO config_history
                            (config_version, generation, registry_version, payload_json, saved_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (config_version) DO UPDATE SET
                            generation = excluded.generation,
                            registry_version = excluded.registry_version,
                            payload_json = excluded.payload_json,
                            saved_at = excluded.saved_at
                        """,
                        (
                            directive.version,
                            directive.generation,
                            registry_version,
                            payload_text,
                            saved_at,
                        ),
                    )
                    _prune_config_history(connection, keep_last=CONFIG_HISTORY_RETENTION_COUNT)
            except _STORE_UNAVAILABLE_ERRORS as error:
                print(
                    f"worker config LKG store unavailable at {self.database_path}: {error}",
                    file=sys.stderr,
                )
                return False
            else:
                return True
            finally:
                if connection is not None:
                    connection.close()

    def load(self) -> StoredConfigPayload | None:
        with self._lock:
            connection = None
            try:
                connection = open_connection(self.database_path)
                return _read_current(connection)
            except _STORE_UNAVAILABLE_ERRORS as error:
                print(
                    f"worker config LKG store unavailable at {self.database_path}: {error}",
                    file=sys.stderr,
                )
                return None
            finally:
                if connection is not None:
                    connection.close()

    def clear(self) -> bool:
        """Delete the current LKG snapshot (the `config_current` row) without
        touching `config_history`, which stays needed for local joins against
        already-emitted evidence_events. Used when the stored LKG no longer
        re-validates and must stop winning the revision race against every
        future legitimate pull (issue #34)."""
        with self._lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = open_connection(self.database_path)
                with ImmediateTransaction(connection):
                    connection.execute("DELETE FROM config_current WHERE id = 1")
            except _STORE_UNAVAILABLE_ERRORS as error:
                print(
                    f"worker config LKG store unavailable at {self.database_path}: {error}",
                    file=sys.stderr,
                )
                return False
            else:
                return True
            finally:
                if connection is not None:
                    connection.close()


def _read_current(connection: sqlite3.Connection) -> StoredConfigPayload | None:
    row = connection.execute(
        """
        SELECT generation, config_version, registry_version, payload_json
        FROM config_current
        WHERE id = 1
        """
    ).fetchone()
    if row is None:
        return None
    generation, config_version, registry_version, payload_json = row
    return StoredConfigPayload(
        payload=json.loads(payload_json),
        directive=RestartDirective(generation=generation, version=config_version),
        registry_version=registry_version,
    )


def _prune_config_history(connection: sqlite3.Connection, *, keep_last: int) -> None:
    connection.execute(
        """
        DELETE FROM config_history
        WHERE config_version NOT IN (
            SELECT config_version FROM config_history
            ORDER BY saved_at DESC, config_version DESC
            LIMIT ?
        )
        AND config_version NOT IN (
            SELECT DISTINCT CAST(json_extract(payload_json, '$.audit.config_version') AS INTEGER)
            FROM evidence_events
            WHERE delivery_state != 'ACKED'
              AND json_extract(payload_json, '$.audit.config_version') IS NOT NULL
        )
        """,
        (keep_last,),
    )


def _worker_state_db_path(state_dir: Path | None = None) -> Path:
    resolved = state_dir if state_dir is not None else resolve_state_dir()
    return resolved / WORKER_STATE_DB_FILENAME


def _registry_version(payload: JsonObject) -> int:
    value = payload.get("registry_version", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "CONFIG_HISTORY_RETENTION_COUNT",
    "WORKER_STATE_DB_FILENAME",
    "JsonObject",
    "JsonValue",
    "StoredConfigPayload",
    "WorkerConfigLkgStore",
]

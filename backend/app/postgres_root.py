"""PostgreSQL composition root: one pool, one authority, every product store.

ADR-0002: missing wiring refuses startup. The DSN and the persistence
authority come only from deployment-issued secret files; neither value is
logged, echoed in an error, or chained into a traceback.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import psycopg
from fastapi import FastAPI

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, require_authority
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase, PostgresError
from backend.app.features.cameras.bed_zone_store import BedZoneStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.artifacts import CentralClipArtifactQuery
from backend.app.features.clips.catalog_indexer import ClipCatalogIndexer, PostgresClipCatalog
from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.clips.store import ClipStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.detection_settings.policy_store import DetectionPolicyStore
from backend.app.features.detection_settings.store import DetectionSettingsStore
from backend.app.features.evidence.outbox_delivery import OutboxDelivery
from backend.app.features.evidence.outbox_dispatch import RELAY_DELIVERY_BUDGET
from backend.app.features.evidence.postgres_receipts import PostgresArtifactReceiptStore
from backend.app.features.evidence.postgres_relay_projection import (
    PostgresRelayEvidenceProjection,
)
from backend.app.features.evidence.record_store import (
    CentralEvidenceQuery,
    CentralEvidenceReviewStore,
)
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore

API_POSTGRES_DSN_FILE_ENV = "API_POSTGRES_DSN_FILE"
API_POSTGRES_AUTHORITY_FILE_ENV = "API_POSTGRES_AUTHORITY_FILE"
API_POSTGRES_SCHEMA_ENV = "API_POSTGRES_SCHEMA"
DEFAULT_POSTGRES_SCHEMA = "seeon_edge"
DEFAULT_POOL_BUDGET = PoolBudget(
    max_connections=8,
    max_waiting=32,
    acquire_timeout_sec=2.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=3000,
    startup_timeout_sec=10.0,
)
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 10.0


class PostgresRootError(RuntimeError):
    """Startup refusal whose text never carries connection or authority secrets."""


@dataclass(frozen=True, slots=True)
class PostgresRoot:
    database: PostgresDatabase
    authority: AuthorityToken


def _required_file(environ: Mapping[str, str], name: str, label: str) -> str:
    raw_path = environ.get(name, "").strip()
    if not raw_path:
        raise PostgresRootError(f"{label} file is not configured; set {name}")
    try:
        text = Path(raw_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        text = None
    if text is None:
        raise PostgresRootError(f"{label} file named by {name} is unreadable")
    if not text:
        raise PostgresRootError(f"{label} file named by {name} is empty")
    return text


def _parse_authority(text: str) -> AuthorityToken:
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    token: AuthorityToken | None = None
    if isinstance(payload, dict) and set(payload) == {"generation", "writer_token"}:
        try:
            token = AuthorityToken(
                generation=payload["generation"], writer_token=UUID(str(payload["writer_token"]))
            )
        except (ValueError, TypeError):
            token = None
    if token is None:
        raise PostgresRootError(
            f"persistence authority file named by {API_POSTGRES_AUTHORITY_FILE_ENV} is invalid"
        )
    return token


def open_postgres_root(
    environ: Mapping[str, str] | None = None, *, budget: PoolBudget = DEFAULT_POOL_BUDGET
) -> PostgresRoot:
    """Open the pool and prove this process holds the live persistence authority."""
    env = os.environ if environ is None else environ
    conninfo = _required_file(env, API_POSTGRES_DSN_FILE_ENV, "PostgreSQL DSN")
    authority = _parse_authority(
        _required_file(env, API_POSTGRES_AUTHORITY_FILE_ENV, "persistence authority")
    )
    schema = env.get(API_POSTGRES_SCHEMA_ENV, "").strip() or DEFAULT_POSTGRES_SCHEMA
    try:
        database: PostgresDatabase | None = PostgresDatabase(conninfo, schema, budget)
    except (TypeError, ValueError, PostgresError):
        database = None
    del conninfo
    if database is None:
        raise PostgresRootError("PostgreSQL connection settings are invalid")
    try:
        database.start()
        started = True
    except PostgresError:
        started = False
    if not started:
        raise PostgresRootError("PostgreSQL is unreachable within the startup budget")
    try:
        database.transact(lambda connection: require_authority(connection, authority))
        verified = True
    except (AuthorityFenced, PostgresError, psycopg.Error):
        verified = False
    if not verified:
        with suppress(PostgresError):
            close_postgres_database(database)
        raise PostgresRootError(
            "PostgreSQL schema is not provisioned for this deployment's persistence authority"
        )
    return PostgresRoot(database, authority)


def _clip_root(app: FastAPI) -> Path:
    """Receipts verify media under the same clip root the API serves clips from."""
    clip_store = getattr(app.state, "clip_store", None)
    return clip_store.root if isinstance(clip_store, ClipStore) else ClipStore.from_env().root


def install_postgres_stores(app: FastAPI, root: PostgresRoot) -> tuple[str, ...]:
    """Publish every PostgreSQL product store; a pre-injected store wins."""
    database, authority = root.database, root.authority
    installed: list[str] = []
    stores = {
        "camera_registry": lambda: CameraRegistryStore(database, authority),
        "bed_zone_store": lambda: BedZoneStore(database, authority),
        "clip_storage_location_store": lambda: ClipStorageLocationStore(database, authority),
        "clip_catalog": lambda: PostgresClipCatalog(database),
        "clip_catalog_indexer": lambda: ClipCatalogIndexer(database, authority),
        "central_clip_artifact_query": lambda: CentralClipArtifactQuery(database),
        "connection_settings_store": lambda: ConnectionSettingsStore(database, authority),
        "detection_policy_store": lambda: DetectionPolicyStore(database, authority),
        "detection_settings_store": lambda: DetectionSettingsStore(database, authority),
        "runtime_settings_store": lambda: RuntimeSettingsStore(database, authority),
        "dashboard_credentials_store": lambda: PostgresDashboardCredentialsStore(
            database, authority
        ),
        "artifact_receipt_store": lambda: PostgresArtifactReceiptStore(
            database, authority, _clip_root(app)
        ),
        "central_evidence_review_store": lambda: CentralEvidenceReviewStore(database, authority),
        "central_evidence_query": lambda: CentralEvidenceQuery(database),
        "event_outbox_delivery": lambda: OutboxDelivery(database, authority, RELAY_DELIVERY_BUDGET),
        "relay_snapshot_projection": lambda: PostgresRelayEvidenceProjection(database, authority),
    }
    for name, build in stores.items():
        if getattr(app.state, name, None) is None:
            setattr(app.state, name, build())
            installed.append(name)
    return tuple(installed)


def close_postgres_database(
    database: PostgresDatabase, *, timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC
) -> None:
    """Refuse new work, drain admitted scopes, then dispose the pool."""
    database.stop_admission()
    database.close(timeout_sec=timeout_sec)


__all__ = [
    "API_POSTGRES_AUTHORITY_FILE_ENV",
    "API_POSTGRES_DSN_FILE_ENV",
    "API_POSTGRES_SCHEMA_ENV",
    "DEFAULT_POOL_BUDGET",
    "DEFAULT_POSTGRES_SCHEMA",
    "PostgresRoot",
    "PostgresRootError",
    "close_postgres_database",
    "install_postgres_stores",
    "open_postgres_root",
]

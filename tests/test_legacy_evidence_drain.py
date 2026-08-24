from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import tempfile
import threading
from pathlib import Path
from types import ModuleType

import pytest
import uvicorn

from backend.app.edge_db.compatibility import MigrationRequiredError
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.edge_db.legacy_drain import LegacyEvidenceDrain
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import MIGRATIONS, SchemaV17MigrationError
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.evidence.relay_projection import RelayEvidenceProjection
from backend.app.main import create_app, no_lifespan
from shared.events.edge_ingest_client import EdgeIngestClient
from shared.events.evidence_export_client import RelayEvidenceClient
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from tests_support.alert_amplification_runtime import (
    CAMERA_ID,
    FACILITY_ID,
    RELAY_TOKEN,
    ServedFixture,
    free_port,
)
from tests_support.compact_authority_db import prepare_compact_database

_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000e2"
_DETECTED_AT = "2026-08-22T00:00:00Z"


def _drain_cli() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/ops/drain-legacy-evidence.py"
    spec = importlib.util.spec_from_file_location("drain_legacy_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clip_recovery_cli() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/ops/recover-legacy-clips.py"
    spec = importlib.util.spec_from_file_location("recover_legacy_clips", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RelayServer:
    """Serve the authenticated relay route over loopback HTTP."""

    def __init__(self, tmp_path: Path, database: Path, hub_origin: str) -> None:
        app = create_app(lifespan=no_lifespan)
        app.state.edge_relay_token = RELAY_TOKEN
        registry_path = Path(tempfile.mkdtemp()) / "registry.sqlite3"
        prepare_compact_database(registry_path)
        registry = CameraRegistryStore(registry_path)
        registry.create(
            camera_id=CAMERA_ID,
            label=CAMERA_ID,
            rtsp_url=f"rtsp://role-gateway:8554/{CAMERA_ID}",
            space_id=None,
            status="online",
            backend_camera_id=CAMERA_ID,
        )
        app.state.camera_registry = registry
        app.state.catalog_store = CatalogStore.open(tmp_path / "relay-catalog.sqlite3")
        app.state.relay_evidence_projection = RelayEvidenceProjection(database)
        app.state.backend_ingest_client = EdgeIngestClient(
            events_url=f"{hub_origin}/api/v1/events",
            bearer_token="fixture-token",
            camera_id=CAMERA_ID,
            timeout_sec=5.0,
        )
        self.port = free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", lifespan="off")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> _RelayServer:
        self._thread.start()
        waiter = threading.Event()
        for _ in range(200):
            if self._server.started:
                return self
            waiter.wait(0.05)
        raise RuntimeError("relay server did not start")

    def __exit__(self, *_args: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def _schema16_database(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:16])
    return database


def _insert_pending(database: Path, *, valid: bool = True) -> None:
    payload: dict[str, object] = {
        "edge_event_id": _EDGE_EVENT_ID,
        "camera_id": CAMERA_ID,
        "event_type": "fall",
        "probability": 0.9,
    }
    if valid:
        payload.update(detected_at=_DETECTED_AT, facility_id=FACILITY_ID)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO evidence_events
                (edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at)
            VALUES (?, ?, ?, 'READY', 1, 1)
            """,
            (_EDGE_EVENT_ID, _DETECTED_AT, json.dumps(payload, separators=(",", ":"))),
        )


def _relay(origin: str) -> RelayEvidenceClient:
    return RelayEvidenceClient(origin, RELAY_TOKEN, timeout_sec=2.0)


class _FixedTransport:
    def __init__(self, result: EventReceipt | DeliveryFailure) -> None:
        self._result = result

    def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt | DeliveryFailure:
        del payload_json, edge_event_id
        return self._result


def test_legacy_drain_real_relay_delivers_once_and_unblocks_schema17_migrator(
    tmp_path: Path,
) -> None:
    database = _schema16_database(tmp_path)
    _insert_pending(database)
    with pytest.raises(MigrationRequiredError):
        open_runtime_database(database, actor=RuntimeActor.API)
    relay_database = tmp_path / "relay.sqlite3"
    migrate_database(relay_database)

    with ServedFixture() as hub, _RelayServer(tmp_path, relay_database, hub.origin) as relay:
        result = _drain_cli().main(
            [
                "--database",
                str(database),
                "--relay-url",
                relay.origin,
                "--relay-token",
                RELAY_TOKEN,
            ]
        )

    assert result == 0
    assert len(hub.fixture.accepted_event_ids(_EDGE_EVENT_ID)) == 1
    receipt = hub.fixture.event_for_edge_id(_EDGE_EVENT_ID)
    assert receipt is not None
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, delivery_state, backend_event_id FROM evidence_events"
        ).fetchone() == ("ACKED", "ACKED", receipt.event_id)
    assert migrate_database(database).current_version == 17


def test_documented_schema16_drains_clear_every_schema17_gate_blocker(
    tmp_path: Path,
) -> None:
    """The documented drain order reaches the real forward-only migration."""
    database = _schema16_database(tmp_path)
    event_ids = [
        "00000000-0000-4000-8000-000000000011",
        "00000000-0000-4000-8000-000000000012",
        "00000000-0000-4000-8000-000000000013",
    ]
    awaiting_clip_id = "clip-awaiting"
    inflight_clip_id = "clip-inflight"
    with sqlite3.connect(database) as connection:
        for state, event_id in zip(("STAGED", "READY", "IN_FLIGHT"), event_ids, strict=True):
            payload = {
                "edge_event_id": event_id,
                "camera_id": CAMERA_ID,
                "detected_at": _DETECTED_AT,
                "event_type": "fall",
                "facility_id": FACILITY_ID,
                "probability": 0.9,
            }
            connection.execute(
                """
                INSERT INTO evidence_events (
                    edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at,
                    lease_owner, lease_expires_at
                ) VALUES (?, ?, ?, ?, 1, 1, ?, ?)
                """,
                (
                    event_id,
                    _DETECTED_AT,
                    json.dumps(payload, separators=(",", ":")),
                    state,
                    "legacy-uploader" if state == "IN_FLIGHT" else None,
                    2.0 if state == "IN_FLIGHT" else None,
                ),
            )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id) VALUES (?)",
            (awaiting_clip_id,),
        )
        connection.execute(
            """
            INSERT INTO evidence_clips (
                clip_id, local_state, publish_state, publish_lease_owner,
                publish_lease_expires_at
            ) VALUES (?, 'VERIFIED', 'IN_FLIGHT', 'legacy-uploader', 2)
            """,
            (inflight_clip_id,),
        )
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_state, provenance_missing_reason, lifecycle_state, failure_reason,
                created_at, updated_at
            ) VALUES ('incident:one', ?, 'camera:one', 'fall', ?,
                      'MISSING', 'LEGACY', 'FAILED', 'MISSING', ?, ?)
            """,
            (event_ids[0], _DETECTED_AT, _DETECTED_AT, _DETECTED_AT),
        )
        for kind, state, request_id in (
            ("STILL", "PENDING", "a" * 64),
            ("VIDEO", "RUNNING", "b" * 64),
        ):
            connection.execute(
                """
                INSERT INTO derivative_jobs (
                    incident_id, derivative_kind, request_id, state, created_at, updated_at
                ) VALUES ('incident:one', ?, ?, ?, ?, ?)
                """,
                (kind, request_id, state, _DETECTED_AT, _DETECTED_AT),
            )
        connection.execute(
            """
            INSERT INTO derivative_evidence_slots (
                incident_id, derivative_kind, state, created_at, updated_at
            ) VALUES ('incident:one', 'ANNOTATED_CLIP', 'PENDING', ?, ?)
            """,
            (_DETECTED_AT, _DETECTED_AT),
        )
        connection.execute(
            """
            INSERT INTO evidence_retention_states (clip_id, state, requested_at, updated_at)
            VALUES (?, 'PENDING', ?, ?), ('orphan-clip', 'PENDING', ?, ?)
            """,
            (
                awaiting_clip_id,
                _DETECTED_AT,
                _DETECTED_AT,
                _DETECTED_AT,
                _DETECTED_AT,
            ),
        )

    store = tmp_path / "clip-store"
    (store / "clips" / ".staging").mkdir(parents=True)
    media = b"verified legacy media"
    clip_directory = store / "clips" / awaiting_clip_id
    clip_directory.mkdir()
    (clip_directory / "clip.mp4").write_bytes(media)
    (clip_directory / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 2,
                "state": "READY",
                "clip_id": awaiting_clip_id,
                "camera_id": "cam-1",
                "event_refs": [event_ids[0]],
                "event_ref": event_ids[0],
                "started_at": _DETECTED_AT,
                "clip_start_at": _DETECTED_AT,
                "clip_end_at": "2026-08-22T00:00:01Z",
                "finalized_at": "2026-08-22T00:00:02Z",
                "duration_s": 1.0,
                "path": f"clips/{awaiting_clip_id}/clip.mp4",
                "finalized": True,
                "video_available": True,
                "state_version": 2,
                "sha256": hashlib.sha256(media).hexdigest(),
                "size_bytes": len(media),
                "mime_type": "video/mp4",
                "codec": "h264",
                "duration_ms": 1000,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    relay_database = tmp_path / "relay.sqlite3"
    migrate_database(relay_database)
    with ServedFixture() as hub, _RelayServer(tmp_path, relay_database, hub.origin) as relay:
        assert (
            _drain_cli().main(
                [
                    "--database",
                    str(database),
                    "--relay-url",
                    relay.origin,
                    "--relay-token",
                    RELAY_TOKEN,
                ]
            )
            == 0
        )
    assert _clip_recovery_cli().main(["--database", str(database), "--clip-store", str(store)]) == 0

    assert migrate_database(database).current_version == 17
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_legacy_drain_mismatched_receipt_keeps_row_pending(tmp_path: Path) -> None:
    database = _schema16_database(tmp_path)
    _insert_pending(database)

    result = LegacyEvidenceDrain(
        database,
        _FixedTransport(
            EventReceipt(
                status="accepted",
                edge_event_id="00000000-0000-4000-8000-0000000000e3",
                event_id="backend-event",
            )
        ),
    ).run()

    assert result.delivered == 0
    assert result.retryable == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, delivery_state, backend_event_id, attempt_count FROM evidence_events"
        ).fetchone() == ("READY", "PENDING", None, 0)


def test_legacy_drain_compatibility_failure_keeps_row_pending_and_classified(
    tmp_path: Path,
) -> None:
    database = _schema16_database(tmp_path)
    _insert_pending(database)

    result = LegacyEvidenceDrain(
        database,
        _FixedTransport(
            DeliveryFailure(DeliveryDisposition.COMPATIBILITY, "UNSUPPORTED_RELAY_PAYLOAD")
        ),
    ).run()

    assert result.delivered == 0
    assert result.permanent == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, delivery_state, last_error_code, attempt_count FROM evidence_events"
        ).fetchone() == ("READY", "COMPATIBILITY", "UNSUPPORTED_RELAY_PAYLOAD", 1)


def test_legacy_drain_422_keeps_row_pending_cli_nonzero_and_blocks_migration(
    tmp_path: Path,
) -> None:
    database = _schema16_database(tmp_path)
    _insert_pending(database, valid=False)
    relay_database = tmp_path / "relay.sqlite3"
    migrate_database(relay_database)

    with ServedFixture() as hub, _RelayServer(tmp_path, relay_database, hub.origin) as relay:
        result = _drain_cli().main(
            [
                "--database",
                str(database),
                "--relay-url",
                relay.origin,
                "--relay-token",
                RELAY_TOKEN,
            ]
        )

    assert result == 2
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, delivery_state, last_error_code FROM evidence_events"
        ).fetchone() == ("READY", "PERMANENT", "HTTP_422")
    with pytest.raises(SchemaV17MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        migrate_database(database)


def test_legacy_drain_cli_coexists_with_relay_runtime_lock(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _insert_pending(database)

    with ServedFixture() as hub, _RelayServer(tmp_path, database, hub.origin) as relay:
        result = _drain_cli().main(
            [
                "--database",
                str(database),
                "--relay-url",
                relay.origin,
                "--relay-token",
                RELAY_TOKEN,
            ]
        )

    assert result == 0
    assert len(hub.fixture.accepted_event_ids(_EDGE_EVENT_ID)) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, delivery_state FROM evidence_events"
        ).fetchone() == ("ACKED", "ACKED")

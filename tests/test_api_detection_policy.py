from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.detection_settings.policy_store import DetectionPolicyStore
from backend.app.main import create_app, no_lifespan

DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}
RELAY_HEADERS = {"X-Edge-Relay-Token": "relay-token"}
FACILITY_ID = "facility/non-uuid:seoul"
LOCAL_CAMERA_ID = "local/camera:room-1"
CANONICAL_CAMERA_ID = "hub-camera|opaque|A-17"


def _app(database: Path):
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(database)
    app.state.connection_settings_store = ConnectionSettingsStore(database)
    app.state.detection_policy_store = DetectionPolicyStore(database)
    app.state.connection_settings_store.save(
        {
            "facility_code": "NH-0123456789",
            "client_installation_ref": "install-ref",
            "facility_id": FACILITY_ID,
            "facility_token": "facility-token",
            "edge_installation_id": "edge/opaque:1",
            "enrollment_generation": 1,
        }
    )
    app.state.camera_registry.create(
        camera_id=LOCAL_CAMERA_ID,
        label="Room 1",
        rtsp_url="rtsp://camera.invalid/stream",
        space_id=None,
        status="offline",
        backend_camera_id=CANONICAL_CAMERA_ID,
    )
    return app


def _login(client: TestClient) -> None:
    assert client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN).status_code == 204


def _request(
    *,
    module_id: str,
    schema_id: str,
    values: dict[str, object] | None,
    camera_id: str | None = None,
    expected_revision_id: int | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "module_id": module_id,
        "module_version": 1,
        "schema_id": schema_id,
        "schema_version": 1,
        "camera_id": camera_id,
        "values": values,
    }
    if expected_revision_id is not None:
        body["expected_revision_id"] = expected_revision_id
    return body


def _diff_token(client: TestClient, payload: dict[str, object]) -> int:
    response = client.post("/api/v1/detection-policies/diff", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "concurrency_token" in body
    assert "compared_payload" in body
    assert body["compared_payload"]["module_id"] == payload["module_id"]
    assert body["compared_payload"]["module_version"] == payload["module_version"]
    assert body["compared_payload"]["schema_id"] == payload["schema_id"]
    assert body["compared_payload"]["schema_version"] == payload["schema_version"]
    assert body["compared_payload"]["camera_id"] == payload["camera_id"]
    assert body["compared_payload"]["values"] == payload["values"]
    return int(body["concurrency_token"])


def _apply(
    client: TestClient,
    payload: dict[str, object],
    *,
    expected_revision_id: int | None = None,
):
    body = dict(payload)
    token = (
        expected_revision_id if expected_revision_id is not None else _diff_token(client, payload)
    )
    body["expected_revision_id"] = token
    return client.post("/api/v1/detection-policies/apply", json=body)


def _rollback(
    client: TestClient,
    *,
    module_id: str,
    module_version: int = 1,
    camera_id: str | None,
    expected_revision_id: int,
):
    return client.post(
        "/api/v1/detection-policies/rollback",
        json={
            "module_id": module_id,
            "module_version": module_version,
            "camera_id": camera_id,
            "expected_revision_id": expected_revision_id,
        },
    )


def test_policy_diff_apply_precedence_revision_activation_and_rollback(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)
    facility_fall = _request(
        module_id="fall",
        schema_id="fall.policy",
        values={"operating_threshold": 0.62},
    )
    camera_fall = _request(
        module_id="fall",
        schema_id="fall.policy",
        camera_id=CANONICAL_CAMERA_ID,
        values={"operating_threshold": 0.81},
    )

    with TestClient(app) as client:
        _login(client)
        diff = client.post("/api/v1/detection-policies/diff", json=facility_fall)
        assert diff.status_code == 200
        assert diff.json()["changed"] is True
        assert diff.json()["current"]["source"] == "image-default"
        assert diff.json()["concurrency_token"] == 0
        assert diff.json()["compared_payload"] == {
            "module_id": "fall",
            "module_version": 1,
            "schema_id": "fall.policy",
            "schema_version": 1,
            "camera_id": None,
            "values": {"operating_threshold": 0.62},
        }
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT count(*) FROM control_detection_policy_revisions"
            ).fetchone() == (0,)

        first = _apply(client, facility_fall, expected_revision_id=0)
        assert first.status_code == 202
        assert first.json()["status"] == "pending"
        first_revision = first.json()["active_revision_id"]

        override = _apply(client, camera_fall, expected_revision_id=0)
        assert override.status_code == 202
        override_revision = override.json()["active_revision_id"]
        assert override_revision > first_revision

        config = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
        assert config.status_code == 200
        body = config.json()
        assert body["cameras"][0]["camera_id"] == CANONICAL_CAMERA_ID
        assert body["cameras"][0]["facility_id"] == FACILITY_ID
        default = body["detection_policies"]["defaults"]["fall"]
        effective = body["detection_policies"]["cameras"][CANONICAL_CAMERA_ID]["fall"]
        assert default["values"] == {"operating_threshold": 0.62}
        assert default["source"] == "facility-default"
        assert effective["values"] == {"operating_threshold": 0.81}
        assert effective["source"] == "camera-override"
        assert effective["facility_revision_id"] == first_revision
        assert effective["camera_revision_id"] == override_revision
        config_version = body["config_version"]

        heartbeat = client.post(
            "/api/v1/relay/heartbeat",
            headers=RELAY_HEADERS,
            json={
                "camera_id": CANONICAL_CAMERA_ID,
                "facility_id": FACILITY_ID,
                "config_version": config_version,
            },
        )
        assert heartbeat.status_code == 202
        status = client.get("/api/v1/detection-policies")
        assert status.status_code == 200
        activations = status.json()["activations"]
        assert {activation["status"] for activation in activations} == {"applied"}

        second_payload = facility_fall | {"values": {"operating_threshold": 0.67}}
        second = _apply(client, second_payload, expected_revision_id=first_revision)
        assert second.status_code == 202
        second_revision = second.json()["active_revision_id"]
        assert second_revision > first_revision
        rolled_back = _rollback(
            client,
            module_id="fall",
            camera_id=None,
            expected_revision_id=second_revision,
        )
        assert rolled_back.status_code == 202
        assert rolled_back.json()["active_revision_id"] == first_revision

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT revision_id, values_json FROM control_detection_policy_revisions "
            "WHERE facility_id = ? AND camera_id IS NULL ORDER BY revision_id",
            (FACILITY_ID,),
        ).fetchall()
    assert [row[0] for row in rows] == [first_revision, second_revision]
    assert "0.62" in rows[0][1]
    assert "0.67" in rows[1][1]


def test_policy_resolution_uses_only_worker_camera_ids_when_namespaces_collide(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)
    second_worker_id = "worker-camera/second:opaque"
    app.state.camera_registry.create(
        camera_id=CANONICAL_CAMERA_ID,
        label="Cross-namespace collision",
        rtsp_url="rtsp://camera.invalid/second",
        space_id=None,
        status="offline",
        backend_camera_id=second_worker_id,
    )

    with TestClient(app) as client:
        _login(client)
        applied = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                camera_id=CANONICAL_CAMERA_ID,
                values={"operating_threshold": 0.81},
            ),
            expected_revision_id=0,
        )
        assert applied.status_code == 202
        config = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS).json()

    policies = config["detection_policies"]["cameras"]
    assert set(policies) == {CANONICAL_CAMERA_ID, second_worker_id}
    assert policies[CANONICAL_CAMERA_ID]["fall"]["source"] == "camera-override"
    assert policies[CANONICAL_CAMERA_ID]["fall"]["values"] == {"operating_threshold": 0.81}
    assert policies[second_worker_id]["fall"]["source"] == "image-default"
    assert policies[second_worker_id]["fall"]["values"] == {"operating_threshold": 0.5}


def test_policy_diff_reports_equal_numeric_values_with_new_source_as_changed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)
    facility_default = _request(
        module_id="fall",
        schema_id="fall.policy",
        values={"operating_threshold": 0.5},
    )
    camera_override = facility_default | {"camera_id": CANONICAL_CAMERA_ID}

    with TestClient(app) as client:
        _login(client)
        facility_diff = client.post("/api/v1/detection-policies/diff", json=facility_default)
        assert facility_diff.status_code == 200
        assert facility_diff.json()["current"]["source"] == "image-default"
        assert facility_diff.json()["proposed"]["source"] == "facility-default"
        assert facility_diff.json()["changed"] is True
        assert _apply(client, facility_default, expected_revision_id=0).status_code == 202

        camera_diff = client.post("/api/v1/detection-policies/diff", json=camera_override)
        assert camera_diff.status_code == 200
        assert camera_diff.json()["current"]["source"] == "facility-default"
        assert camera_diff.json()["proposed"]["source"] == "camera-override"
        assert camera_diff.json()["changed"] is True
        assert camera_diff.json()["concurrency_token"] == 0


def test_repeated_rollback_walks_revision_history_without_toggling_forward(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)

    with TestClient(app) as client:
        _login(client)
        revision_ids: list[int] = []
        expected = 0
        for threshold in (0.61, 0.62, 0.63):
            response = _apply(
                client,
                _request(
                    module_id="fall",
                    schema_id="fall.policy",
                    values={"operating_threshold": threshold},
                ),
                expected_revision_id=expected,
            )
            assert response.status_code == 202
            expected = response.json()["active_revision_id"]
            revision_ids.append(expected)

        first = _rollback(
            client,
            module_id="fall",
            camera_id=None,
            expected_revision_id=revision_ids[2],
        )
        replacement = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.70},
            ),
            expected_revision_id=revision_ids[1],
        )
        replacement_revision = replacement.json()["active_revision_id"]
        after_replacement = _rollback(
            client,
            module_id="fall",
            camera_id=None,
            expected_revision_id=replacement_revision,
        )
        second = _rollback(
            client,
            module_id="fall",
            camera_id=None,
            expected_revision_id=revision_ids[1],
        )
        exhausted = _rollback(
            client,
            module_id="fall",
            camera_id=None,
            expected_revision_id=revision_ids[0],
        )

    assert first.status_code == 202
    assert first.json()["active_revision_id"] == revision_ids[1]
    assert replacement.status_code == 202
    assert after_replacement.status_code == 202
    assert after_replacement.json()["active_revision_id"] == revision_ids[1]
    assert after_replacement.json()["active_revision_id"] != revision_ids[2]
    assert second.status_code == 202
    assert second.json()["active_revision_id"] == revision_ids[0]
    assert exhausted.status_code == 409


def test_nullable_camera_override_returns_to_facility_default(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)

    with TestClient(app) as client:
        _login(client)
        facility_apply = _apply(
            client,
            _request(
                module_id="bed_exit",
                schema_id="bed_exit.policy",
                values={"min_containment": 0.4, "hold_frames": 3, "grace_frames": 5},
            ),
            expected_revision_id=0,
        )
        assert facility_apply.status_code == 202
        override_apply = _apply(
            client,
            _request(
                module_id="bed_exit",
                schema_id="bed_exit.policy",
                camera_id=CANONICAL_CAMERA_ID,
                values={"min_containment": 0.6, "hold_frames": 4, "grace_frames": 6},
            ),
            expected_revision_id=0,
        )
        assert override_apply.status_code == 202
        override_revision = override_apply.json()["active_revision_id"]
        inherit_request = _request(
            module_id="bed_exit",
            schema_id="bed_exit.policy",
            camera_id=CANONICAL_CAMERA_ID,
            values=None,
        )
        diff = client.post("/api/v1/detection-policies/diff", json=inherit_request)
        assert diff.status_code == 200
        assert diff.json()["changed"] is True
        assert diff.json()["proposed"]["source"] == "facility-default"
        assert diff.json()["concurrency_token"] == override_revision
        assert diff.json()["compared_payload"]["values"] is None
        cleared = _apply(client, inherit_request, expected_revision_id=override_revision)
        assert cleared.status_code == 202
        config = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS).json()

    effective = config["detection_policies"]["cameras"][CANONICAL_CAMERA_ID]["bed_exit"]
    assert effective["source"] == "facility-default"
    assert effective["values"] == {
        "min_containment": 0.4,
        "hold_frames": 3,
        "grace_frames": 5,
    }
    assert effective["camera_revision_id"] is None


def test_api_rejects_malformed_nonfinite_unknown_cross_field_and_unknown_camera(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)

    invalid = (
        _request(
            module_id="fall",
            schema_id="fall.policy",
            values={"operating_threshold": 0.5, "unknown": 1},
        ),
        _request(
            module_id="bed_exit",
            schema_id="bed_exit.policy",
            values={"min_containment": 0.4, "hold_frames": 200, "grace_frames": 101},
        ),
        _request(
            module_id="fall",
            schema_id="fall.policy",
            camera_id="missing/opaque-camera",
            values={"operating_threshold": 0.5},
        ),
    )
    with TestClient(app) as client:
        _login(client)
        missing_token = client.post(
            "/api/v1/detection-policies/apply",
            json=_request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.5},
            ),
        )
        nonfinite = client.post(
            "/api/v1/detection-policies/apply",
            headers={"Content-Type": "application/json"},
            content=(
                '{"module_id":"fall","module_version":1,'
                '"schema_id":"fall.policy","schema_version":1,'
                '"camera_id":null,"expected_revision_id":0,'
                '"values":{"operating_threshold":NaN}}'
            ),
        )
        responses = [
            client.post(
                "/api/v1/detection-policies/apply",
                json=payload | {"expected_revision_id": 0},
            )
            for payload in invalid
        ]
        drift = client.post(
            "/api/v1/detection-policies/apply",
            json=_request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.5},
                expected_revision_id=0,
            )
            | {"schema_version": 99},
        )

    assert missing_token.status_code == 422
    assert nonfinite.status_code == 422
    assert [response.status_code for response in responses] == [422, 422, 404]
    assert drift.status_code == 422


def test_corrupt_revision_is_refused_and_failed_status_persists(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)

    with TestClient(app) as client:
        _login(client)
        applied = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.62},
            ),
            expected_revision_id=0,
        )
        assert applied.status_code == 202
        revision_id = applied.json()["active_revision_id"]

        with sqlite3.connect(database) as connection:
            try:
                connection.execute(
                    "UPDATE control_detection_policy_revisions SET content_sha256=? "
                    "WHERE revision_id=?",
                    ("0" * 64, revision_id),
                )
            except sqlite3.IntegrityError as error:
                assert "immutable" in str(error)
            else:
                raise AssertionError("immutable policy revision was updated")
            connection.execute("DROP TRIGGER control_detection_policy_revisions_immutable_update")
            connection.execute(
                "UPDATE control_detection_policy_revisions SET content_sha256=? "
                "WHERE revision_id=?",
                ("0" * 64, revision_id),
            )

        refused = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
        assert refused.status_code == 503
        assert "content hash mismatch" in refused.json()["detail"]
        status_response = client.get("/api/v1/detection-policies")
        assert status_response.status_code == 200
        activation = status_response.json()["activations"][0]
        assert activation["status"] == "failed"
        assert activation["refusal_reason"] == "policy revision content hash mismatch"

        recovered = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.74},
            ),
            expected_revision_id=revision_id,
        )
        assert recovered.status_code == 202
        assert recovered.json()["active_revision_id"] > revision_id
        recovered_config = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
        assert recovered_config.status_code == 200
        assert recovered_config.json()["detection_policies"]["defaults"]["fall"]["values"] == {
            "operating_threshold": 0.74
        }

    with sqlite3.connect(database) as connection:
        corrupt_history = connection.execute(
            "SELECT content_sha256 FROM control_detection_policy_revisions WHERE revision_id=?",
            (revision_id,),
        ).fetchone()
    assert corrupt_history == ("0" * 64,)


def test_fresh_apply_recovers_corrupt_active_without_prior_read(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)

    with TestClient(app) as client:
        _login(client)
        applied = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.62},
            ),
            expected_revision_id=0,
        )
        assert applied.status_code == 202
        corrupt_revision_id = applied.json()["active_revision_id"]
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TRIGGER control_detection_policy_revisions_immutable_update")
            connection.execute(
                "UPDATE control_detection_policy_revisions SET content_sha256=? "
                "WHERE revision_id=?",
                ("0" * 64, corrupt_revision_id),
            )

        recovered = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.75},
            ),
            expected_revision_id=corrupt_revision_id,
        )
        assert recovered.status_code == 202
        assert recovered.json()["active_revision_id"] > corrupt_revision_id
        assert recovered.json()["previous_revision_id"] is None
        config = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
        assert config.status_code == 200
        assert config.json()["detection_policies"]["defaults"]["fall"]["values"] == {
            "operating_threshold": 0.75
        }


def test_two_operator_first_facility_apply_race_uses_generation_zero_token(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)
    operator_a = _request(
        module_id="fall",
        schema_id="fall.policy",
        values={"operating_threshold": 0.61},
    )
    operator_b = _request(
        module_id="fall",
        schema_id="fall.policy",
        values={"operating_threshold": 0.71},
    )

    with TestClient(app) as client:
        _login(client)
        a_diff = client.post("/api/v1/detection-policies/diff", json=operator_a)
        b_diff = client.post("/api/v1/detection-policies/diff", json=operator_b)
        assert a_diff.status_code == 200
        assert b_diff.status_code == 200
        assert a_diff.json()["concurrency_token"] == 0
        assert b_diff.json()["concurrency_token"] == 0

        first = _apply(client, operator_a, expected_revision_id=0)
        second = _apply(client, operator_b, expected_revision_id=0)

    assert first.status_code == 202
    assert second.status_code == 409
    assert first.json()["active_revision_id"] >= 1
    config = TestClient(app)
    with config:
        _login(config)
        body = config.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS).json()
    assert body["detection_policies"]["defaults"]["fall"]["values"] == {"operating_threshold": 0.61}


def test_two_operator_inherited_camera_apply_race_uses_token_zero(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)
    facility = _request(
        module_id="fall",
        schema_id="fall.policy",
        values={"operating_threshold": 0.55},
    )
    operator_a = _request(
        module_id="fall",
        schema_id="fall.policy",
        camera_id=CANONICAL_CAMERA_ID,
        values={"operating_threshold": 0.66},
    )
    operator_b = _request(
        module_id="fall",
        schema_id="fall.policy",
        camera_id=CANONICAL_CAMERA_ID,
        values={"operating_threshold": 0.77},
    )

    with TestClient(app) as client:
        _login(client)
        assert _apply(client, facility, expected_revision_id=0).status_code == 202
        a_diff = client.post("/api/v1/detection-policies/diff", json=operator_a)
        b_diff = client.post("/api/v1/detection-policies/diff", json=operator_b)
        assert a_diff.json()["concurrency_token"] == 0
        assert b_diff.json()["concurrency_token"] == 0
        assert a_diff.json()["current"]["source"] == "facility-default"

        first = _apply(client, operator_a, expected_revision_id=0)
        second = _apply(client, operator_b, expected_revision_id=0)

    assert first.status_code == 202
    assert second.status_code == 409
    with TestClient(app) as client:
        _login(client)
        body = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS).json()
    effective = body["detection_policies"]["cameras"][CANONICAL_CAMERA_ID]["fall"]
    assert effective["values"] == {"operating_threshold": 0.66}
    assert effective["source"] == "camera-override"


def test_two_operator_rollback_race_requires_cas_token(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    app = _app(database)

    with TestClient(app) as client:
        _login(client)
        first = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.61},
            ),
            expected_revision_id=0,
        )
        second = _apply(
            client,
            _request(
                module_id="fall",
                schema_id="fall.policy",
                values={"operating_threshold": 0.72},
            ),
            expected_revision_id=first.json()["active_revision_id"],
        )
        assert first.status_code == 202
        assert second.status_code == 202
        current_revision = second.json()["active_revision_id"]

        winner = _rollback(
            client,
            module_id="fall",
            camera_id=None,
            expected_revision_id=current_revision,
        )
        loser = _rollback(
            client,
            module_id="fall",
            camera_id=None,
            expected_revision_id=current_revision,
        )

    assert winner.status_code == 202
    assert winner.json()["active_revision_id"] == first.json()["active_revision_id"]
    assert loser.status_code == 409


def test_control_policy_tables_are_backend_written_after_schema17(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    api = open_runtime_database(database, actor=RuntimeActor.API)
    worker = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        api.execute("BEGIN IMMEDIATE")
        api.execute(
            "INSERT INTO control_detection_policy_state "
            "(facility_id, activation_generation) VALUES (?, ?)",
            (FACILITY_ID, 1),
        )
        api.commit()
        assert worker.execute(
            "SELECT activation_generation FROM control_detection_policy_state "
            "WHERE facility_id = ?",
            (FACILITY_ID,),
        ).fetchone() == (1,)
        worker.execute(
            "UPDATE control_detection_policy_state SET activation_generation = 2 "
            "WHERE facility_id = ?",
            (FACILITY_ID,),
        )
        assert api.execute(
            "SELECT activation_generation FROM control_detection_policy_state "
            "WHERE facility_id = ?",
            (FACILITY_ID,),
        ).fetchone() == (2,)
    finally:
        api.close()
        worker.close()

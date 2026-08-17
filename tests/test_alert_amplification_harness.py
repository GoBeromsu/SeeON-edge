from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import tests_support.alert_amplification_harness as harness
from tests_support.alert_amplification_harness import (
    INSECURE_HTTP_ENV,
    ClockSample,
    CorrelationRow,
    DiagnosticOutcome,
    IncidentProjection,
    NamespacePolicy,
    NetworkFlow,
    RunOwnedTeardown,
    SessionBudgetExceeded,
    SessionLatch,
    attest_gateway,
    build_diagnostic_overlay,
    classify_rows,
    collect_stable_incidents,
    evaluate_clock,
    rows_from_relations,
    scan_retained_text,
    validate_insecure_http_assignments,
    validate_insecure_http_source_declarations,
    validate_namespace_policy,
    validate_network_canaries,
    validate_no_insecure_http_assignments,
    validate_route_ledger,
    validate_temporal_order,
    verify_tmpfs_destroyed,
)
from tests_support.local_backend_fixture import RouteRecord


def _row(
    *,
    transition: str | None = "transition-1",
    edge: str | None = "edge-1",
    attempts: tuple[int, ...] = (1,),
    backend: tuple[str, ...] = ("backend-1",),
    incidents: tuple[str, ...] = ("incident-1",),
    state: str = "ACKED",
    clock: bool = True,
) -> CorrelationRow:
    return CorrelationRow(transition, edge, attempts, backend, incidents, state, clock)


def test_single_api_delivery_is_not_model_attribution() -> None:
    result = classify_rows([_row()])

    assert result.outcome is DiagnosticOutcome.SINGLE_API_DELIVERY
    assert result.model_policy_cause == "판정 불가"


def test_lost_response_retry_converges_without_amplification() -> None:
    result = classify_rows([_row(attempts=(1, 2))])

    assert result.outcome is DiagnosticOutcome.TRANSPORT_RETRY
    assert "converged" in result.reason


def test_restart_attempts_merge_across_rows() -> None:
    result = classify_rows([_row(attempts=(1,)), _row(attempts=(2,))])

    assert result.outcome is DiagnosticOutcome.TRANSPORT_RETRY


def test_backend_identity_multiplication_is_detected_across_rows() -> None:
    result = classify_rows(
        [_row(backend=("backend-1",)), _row(backend=("backend-2",))]
    )

    assert result.outcome is DiagnosticOutcome.BACKEND_IDENTITY_DUPLICATION


def test_api_incident_multiplication_is_detected_across_rows() -> None:
    result = classify_rows(
        [_row(incidents=("incident-1",)), _row(incidents=("incident-2",))]
    )

    assert result.outcome is DiagnosticOutcome.API_INCIDENT_DUPLICATION


def test_worker_refire_requires_same_transition_and_distinct_edge_ids() -> None:
    result = classify_rows(
        [
            _row(edge="edge-1"),
            _row(edge="edge-2", backend=("backend-2",), incidents=("incident-2",)),
        ]
    )

    assert result.outcome is DiagnosticOutcome.WORKER_REFIRE


def test_distinct_machine_transitions_are_always_model_inconclusive() -> None:
    result = classify_rows(
        [
            _row(transition="transition-1", edge="edge-1"),
            _row(
                transition="transition-2",
                edge="edge-2",
                backend=("backend-2",),
                incidents=("incident-2",),
            ),
        ]
    )

    assert result.outcome is DiagnosticOutcome.REPEATED_MACHINE_POSITIVE
    assert result.model_policy_cause == "판정 불가"


@pytest.mark.parametrize(
    "row",
    [
        _row(transition=None),
        _row(edge=None),
        _row(attempts=()),
        _row(backend=()),
        _row(incidents=()),
        _row(state="PENDING"),
    ],
)
def test_missing_or_unstable_identity_is_inconclusive(row: CorrelationRow) -> None:
    assert classify_rows([row]).outcome is DiagnosticOutcome.INCONCLUSIVE


def test_ordered_convergence_is_blocked_without_order_evidence() -> None:
    row = CorrelationRow(
        "transition-1",
        "edge-1",
        (1, 2),
        ("backend-1",),
        ("incident-1",),
        "ACKED",
        True,
        False,
    )

    result = classify_rows([row])

    assert result.outcome is DiagnosticOutcome.INCONCLUSIVE
    assert "temporal-order evidence" in result.reason


def test_identity_findings_survive_missing_order_evidence() -> None:
    def _row_without_order(**kwargs: object) -> CorrelationRow:
        base = _row(**kwargs)  # type: ignore[arg-type]
        return CorrelationRow(
            base.transition_id,
            base.edge_event_id,
            base.attempt_ordinals,
            base.backend_event_ids,
            base.incident_ids,
            base.terminal_state,
            base.clock_order_valid,
            False,
        )

    duplication = classify_rows(
        [
            _row_without_order(backend=("backend-1",)),
            _row_without_order(backend=("backend-2",)),
        ]
    )
    refire = classify_rows(
        [
            _row_without_order(edge="edge-1"),
            _row_without_order(
                edge="edge-2", backend=("backend-2",), incidents=("incident-2",)
            ),
        ]
    )

    assert duplication.outcome is DiagnosticOutcome.BACKEND_IDENTITY_DUPLICATION
    assert refire.outcome is DiagnosticOutcome.WORKER_REFIRE


def test_temporal_order_requires_a_real_forward_relation() -> None:
    assert validate_temporal_order(
        attempt_times=[1.0, 2.0],
        receipt_times=[1.5, 2.5],
        projection_time=3.0,
        uncertainty_ms=10.0,
    )
    # Reversed projection, reversed attempts, missing timestamp, and malformed
    # values must all fail closed rather than pass on mere presence.
    assert not validate_temporal_order(
        attempt_times=[1.0],
        receipt_times=[2.0],
        projection_time=0.5,
        uncertainty_ms=10.0,
    )
    assert not validate_temporal_order(
        attempt_times=[2.0, 1.0],
        receipt_times=[2.5, 3.0],
        projection_time=4.0,
        uncertainty_ms=10.0,
    )
    assert not validate_temporal_order(
        attempt_times=[1.0],
        receipt_times=[2.0],
        projection_time=None,
        uncertainty_ms=10.0,
    )
    assert not validate_temporal_order(
        attempt_times=[float("nan")],
        receipt_times=[2.0],
        projection_time=3.0,
        uncertainty_ms=10.0,
    )


def test_api_duplication_requires_exactly_one_backend_identity() -> None:
    without_backend = CorrelationRow(
        "transition-1", "edge-1", (1,), (), ("incident-1", "incident-2"), "ACKED"
    )

    assert classify_rows([without_backend]).outcome is DiagnosticOutcome.INCONCLUSIVE


def test_identity_findings_survive_a_failed_clock_gate() -> None:
    rows = [
        _row(edge="edge-1", clock=False),
        _row(edge="edge-2", backend=("backend-2",), incidents=("incident-2",), clock=False),
    ]

    assert classify_rows(rows).outcome is DiagnosticOutcome.WORKER_REFIRE


def test_noncontiguous_attempts_are_inconclusive() -> None:
    assert classify_rows([_row(attempts=(1, 3))]).outcome is DiagnosticOutcome.INCONCLUSIVE


def test_incident_identity_cannot_cross_edge_ids() -> None:
    result = classify_rows(
        [
            _row(edge="edge-1"),
            _row(edge="edge-2", backend=("backend-2",)),
        ]
    )

    assert result.outcome is DiagnosticOutcome.INCONCLUSIVE


def test_empty_input_is_inconclusive() -> None:
    assert classify_rows([]).outcome is DiagnosticOutcome.INCONCLUSIVE


def test_relation_builder_joins_fixture_and_api_by_edge_id() -> None:
    rows = rows_from_relations(
        transitions={"edge-1": "transition-1"},
        attempts={"edge-1": [1, 2]},
        backend_event_ids={"edge-1": ["backend-1"]},
        incidents=[
            IncidentProjection(
                "incident-1", "edge-1", "2026-08-16T00:00:00Z", "OPEN", "ACKED"
            )
        ],
        terminal_states={"edge-1": "ACKED"},
        clock_order_valid=True,
    )

    assert classify_rows(rows).outcome is DiagnosticOutcome.TRANSPORT_RETRY


def test_session_latch_allows_one_attempt_session_and_teardown() -> None:
    latch = SessionLatch("room")

    latch.attempt()
    latch.connected()
    latch.closed()
    latch.teardown()

    assert latch.counts.attempts == 1
    assert latch.counts.successes == 1
    assert latch.counts.closes == 1
    assert latch.counts.reopen_attempts == 0


def test_session_latch_blocks_reconnect_after_disconnect() -> None:
    latch = SessionLatch("corridor")
    latch.attempt()
    latch.connected()
    latch.closed()

    with pytest.raises(SessionBudgetExceeded):
        latch.attempt()

    assert latch.counts.reopen_attempts == 1


def test_initial_failure_consumes_the_only_attempt() -> None:
    latch = SessionLatch("room")
    latch.attempt()

    with pytest.raises(SessionBudgetExceeded):
        latch.attempt()

    latch.teardown()
    assert latch.counts.successes == 0


def test_session_deadline_and_open_teardown_fail_closed() -> None:
    now = [0.0]
    latch = SessionLatch("room", maximum_seconds=1.0, monotonic=lambda: now[0])
    latch.attempt()
    latch.connected()
    with pytest.raises(SessionBudgetExceeded):
        latch.teardown()
    latch.closed()
    latch.teardown()

    expired = SessionLatch("corridor", maximum_seconds=1.0, monotonic=lambda: now[0])
    now[0] = 2.0
    with pytest.raises(SessionBudgetExceeded):
        expired.attempt()

    now[0] = 0.0
    connected = SessionLatch("room", maximum_seconds=1.0, monotonic=lambda: now[0])
    connected.attempt()
    connected.connected()
    now[0] = 2.0
    with pytest.raises(SessionBudgetExceeded):
        connected.closed()
    assert connected.counts.closes == 1
    with pytest.raises(SessionBudgetExceeded):
        connected.teardown()


def test_clock_gate_requires_five_bounded_samples() -> None:
    result = evaluate_clock([ClockSample(20.0, 10.0, 40.0) for _ in range(5)])

    assert result.passed
    assert result.median_offset_ms == 20.0


@pytest.mark.parametrize(
    "samples",
    [
        [ClockSample(0.0, 0.0, 0.0)] * 4,
        [ClockSample(245.0, 10.0, 40.0)] * 5,
        [ClockSample(0.0, 10.0, 201.0)] * 5,
        [ClockSample(float("nan"), 0.0, 0.0)] * 5,
        [ClockSample(0.0, -1.0, 0.0)] * 5,
    ],
)
def test_clock_gate_rejects_invalid_or_unbounded_samples(
    samples: list[ClockSample],
) -> None:
    assert not evaluate_clock(samples).passed


def test_semantic_cleartext_assignment_is_exactly_one_ml_api() -> None:
    validate_insecure_http_assignments(
        {
            "ml-api": {INSECURE_HTTP_ENV: "1"},
            "ml-worker": {},
            "diagnostic-hub": {},
        }
    )


@pytest.mark.parametrize(
    "environments",
    [
        {"ml-api": {}, "ml-worker": {}},
        {
            "ml-api": {INSECURE_HTTP_ENV: "1"},
            "ml-worker": {INSECURE_HTTP_ENV: "1"},
        },
        {"ml-api": {INSECURE_HTTP_ENV: "true"}},
    ],
)
def test_semantic_cleartext_assignment_rejects_missing_or_leaked_values(
    environments: dict[str, dict[str, str]],
) -> None:
    with pytest.raises(ValueError):
        validate_insecure_http_assignments(environments)


def test_package_services_require_complete_opt_in_absence() -> None:
    validate_no_insecure_http_assignments({"ml-api": {}, "ml-worker": {}})
    with pytest.raises(ValueError):
        validate_no_insecure_http_assignments(
            {"ml-api": {INSECURE_HTTP_ENV: "0"}}
        )


def test_source_declaration_allowlist_rejects_extra_reference(tmp_path: Path) -> None:
    required = {
        "backend/app/features/connection/hub_url.py",
        "backend/app/features/connection/store.py",
        "compose.edge.yaml",
        "scripts/edge-preflight/check-env.sh",
        "tests/conftest.py",
        "tests/test_hub_url_transport_policy.py",
    }
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {INSECURE_HTTP_ENV}\n")
    validate_insecure_http_source_declarations(tmp_path)

    extra = tmp_path / "backend/app/unapproved.py"
    extra.write_text(f"# {INSECURE_HTTP_ENV}\n")
    with pytest.raises(ValueError, match="unexpected cleartext policy references"):
        validate_insecure_http_source_declarations(tmp_path)


def test_gateway_attestation_binds_file_identity_and_capabilities(tmp_path: Path) -> None:
    gateway = tmp_path / "mediamtx"
    gateway.write_bytes(b"synthetic gateway binary")
    gateway.chmod(0o755)
    digest = hashlib.sha256(gateway.read_bytes()).hexdigest()
    probed_inodes: list[int] = []

    def probe(descriptor: int) -> tuple[str, set[str]]:
        probed_inodes.append(os.fstat(descriptor).st_ino)
        return "v1.19.3", {"rtsp-publish", "rtsp-read", "config-stdin"}

    result = attest_gateway(
        gateway.resolve(),
        expected_version="v1.19.3",
        expected_sha256=digest,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        expected_mode=0o755,
        probe=probe,
        required_capabilities=frozenset(
            {"rtsp-publish", "rtsp-read", "config-stdin"}
        ),
    )

    assert result.sha256 == digest
    assert result.mode == 0o755
    assert probed_inodes == [result.inode]


def test_gateway_attestation_rejects_symlink_and_capability_drift(tmp_path: Path) -> None:
    gateway = tmp_path / "gateway"
    gateway.write_bytes(b"binary")
    gateway.chmod(0o755)
    link = tmp_path / "link"
    link.symlink_to(gateway)
    digest = hashlib.sha256(gateway.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="following links"):
        attest_gateway(
            link.absolute(),
            expected_version="v1",
            expected_sha256=digest,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            expected_mode=0o755,
            probe=lambda _fd: ("v1", set()),
            required_capabilities=frozenset(),
        )
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested_gateway = real_parent / "gateway"
    nested_gateway.write_bytes(b"nested")
    nested_gateway.chmod(0o755)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        attest_gateway(
            (alias_parent / "gateway").absolute(),
            expected_version="v1",
            expected_sha256=hashlib.sha256(b"nested").hexdigest(),
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            expected_mode=0o755,
            probe=lambda _fd: ("v1", set()),
            required_capabilities=frozenset(),
        )

    with pytest.raises(ValueError, match="capabilities"):
        attest_gateway(
            gateway.resolve(),
            expected_version="v1",
            expected_sha256=digest,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            expected_mode=0o755,
            probe=lambda _fd: ("v1", {"unexpected"}),
            required_capabilities=frozenset(),
        )


def test_overlay_is_internal_and_scopes_opt_in_to_ml_api() -> None:
    overlay = build_diagnostic_overlay("run-1", "http://diagnostic-hub:8080")
    services = overlay["services"]
    assert isinstance(services, dict)
    assert services["ml-api"]["environment"] == {
        "API_BACKEND_BASE_URL": "http://diagnostic-hub:8080",
        INSECURE_HTTP_ENV: "1",
    }
    assert services["ml-worker"]["environment"] == {}
    assert overlay["networks"] == {"diagnostic": {"internal": True}}
    assert "token" not in repr(overlay).lower()


def test_namespace_policy_and_canaries_are_exact() -> None:
    policy = NamespacePolicy(
        services=frozenset(
            {"diagnostic-runner", "diagnostic-hub", "ml-api", "ml-worker", "role-gateway"}
        ),
        allowed_flows=frozenset(
            {
                NetworkFlow("ml-worker", "ml-api", 8000),
                NetworkFlow("ml-api", "diagnostic-hub", 8080),
                NetworkFlow("diagnostic-runner", "ml-api", 8000),
                NetworkFlow("ml-worker", "role-gateway", 8554),
            }
        ),
        default_drop_input=True,
        default_drop_output=True,
        default_drop_forward=True,
        resolver_disabled=True,
    )
    validate_namespace_policy(policy)
    validate_network_canaries(
        {
            "production_dns_blocked": True,
            "production_tcp_blocked": True,
            "local_hub_reachable": True,
            "local_api_reachable": True,
            "unexpected_egress_zero": True,
        }
    )
    with pytest.raises(ValueError):
        validate_network_canaries(
            {
                "production_dns_blocked": False,
                "production_tcp_blocked": True,
                "local_hub_reachable": True,
                "local_api_reachable": True,
                "unexpected_egress_zero": True,
            }
        )


def test_run_owned_teardown_requires_reverse_complete_destruction() -> None:
    teardown = RunOwnedTeardown("run-1")
    teardown.acquire("run-1:namespace", "namespace")
    teardown.acquire("run-1:mount", "mount")
    with pytest.raises(ValueError, match="stopped in reverse order"):
        teardown.stop("run-1:namespace")
    teardown.stop("run-1:mount")
    teardown.stop("run-1:namespace")
    with pytest.raises(ValueError, match="not repeatable"):
        teardown.stop("run-1:namespace")
    with pytest.raises(ValueError, match="reverse order"):
        teardown.destroy("run-1:namespace")
    teardown.destroy("run-1:mount")
    teardown.destroy("run-1:namespace")
    with pytest.raises(ValueError, match="not repeatable"):
        teardown.destroy("run-1:namespace")
    teardown.finalize()


def test_route_ledger_accepts_only_exact_contract_routes() -> None:
    validate_route_ledger(
        [
            RouteRecord("POST", "/api/v1/events", 201, 100, "application/json", 100),
            RouteRecord(
                "PUT",
                "/api/v1/events/c4444444-4444-4444-8444-444444444444/snapshot",
                201,
                20,
                "image/jpeg",
                20,
            ),
            RouteRecord("GET", "/api/v1/events/capabilities", 200, 0, "", 0),
            RouteRecord("POST", "/v1/events", 404, 100, "application/json", 100),
        ]
    )


@pytest.mark.parametrize(
    "record",
    [
        RouteRecord("GET", "/api/v1/clips/clip-1/video", 200, 0, "", 0),
        RouteRecord("PUT", "/api/v1/incident-reviews/i-1", 200, 20, "application/json", 20),
        RouteRecord("GET", "/api/v1/unknown", 404, 0, "", 0),
        RouteRecord(
            "PUT",
            "/api/v1/events/c4444444-4444-4444-8444-444444444444/snapshot",
            201,
            20,
            "image/png",
            20,
        ),
    ],
)
def test_route_ledger_rejects_unknown_media_or_review_routes(record: RouteRecord) -> None:
    with pytest.raises(ValueError):
        validate_route_ledger([record])


class _Response:
    def __init__(self, body: dict[str, object], status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._body


class _IncidentClient:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, object],
    ) -> _Response:
        self.calls.append((url, headers, params))
        page_index = 1 if params.get("cursor") == "next" else 0
        return _Response(self.pages[page_index])


def _incident(incident_id: str, edge_id: str) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "edge_event_id": edge_id,
        "detected_at": "2026-08-16T00:00:00Z",
        "lifecycle_state": "OPEN",
        "event_delivery_state": "ACKED",
        "review": None,
    }


def test_incident_collector_is_authenticated_paginated_and_stable() -> None:
    client = _IncidentClient(
        [
            {
                "incidents": [_incident("incident-1", "edge-1")],
                "pagination": {"limit": 100, "next_cursor": "next", "has_more": True},
            },
            {
                "incidents": [_incident("incident-2", "edge-2")],
                "pagination": {"limit": 100, "next_cursor": None, "has_more": False},
            },
        ]
    )

    result = collect_stable_incidents(client, bearer_token="dashboard-token")

    assert [item.incident_id for item in result] == ["incident-1", "incident-2"]
    assert len(client.calls) == 4
    assert all(call[1] == {"Authorization": "Bearer dashboard-token"} for call in client.calls)


def test_incident_collector_rejects_human_review_and_unstable_poll() -> None:
    reviewed = _incident("incident-1", "edge-1")
    reviewed["review"] = {"disposition": "FALSE_POSITIVE"}
    client = _IncidentClient(
        [
            {
                "incidents": [reviewed],
                "pagination": {"limit": 100, "next_cursor": None, "has_more": False},
            }
        ]
    )
    with pytest.raises(ValueError):
        collect_stable_incidents(client, bearer_token="dashboard-token")


class _UnstableIncidentClient(_IncidentClient):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, object],
    ) -> _Response:
        self.calls.append((url, headers, params))
        suffix = len(self.calls)
        return _Response(
            {
                "incidents": [_incident(f"incident-{suffix}", "edge-1")],
                "pagination": {"limit": 100, "next_cursor": None, "has_more": False},
            }
        )


def test_incident_collector_rejects_unstable_poll() -> None:
    with pytest.raises(ValueError):
        collect_stable_incidents(
            _UnstableIncidentClient([]), bearer_token="dashboard-token"
        )


def test_tmpfs_destruction_requires_prior_identity_to_be_unreachable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mount"
    path.mkdir()
    metadata = path.stat()

    with pytest.raises(ValueError):
        verify_tmpfs_destroyed(
            path,
            prior_device=metadata.st_dev,
            prior_inode=metadata.st_ino,
        )
    path.rmdir()
    verify_tmpfs_destroyed(
        path,
        prior_device=metadata.st_dev,
        prior_inode=metadata.st_ino,
    )


def test_open_descriptor_on_prior_path_blocks_destruction_proof(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mount"
    path.mkdir()
    metadata = path.stat()
    held = path / "held"
    held.write_text("x")

    with held.open() as handle:
        assert handle.readable()
        assert harness._open_references_for_path(path.resolve())

    assert harness._open_references_for_path(path.resolve()) == ()
    assert metadata.st_ino > 0


def test_procfs_visibility_is_provable_for_own_processes() -> None:
    assert harness.probe_procfs_visibility() is True


def test_retained_text_scanner_rejects_credentials_and_rtsp_urls() -> None:
    with pytest.raises(ValueError):
        scan_retained_text(["token=secret-value"], forbidden=["secret-value"])
    with pytest.raises(ValueError):
        scan_retained_text(["rtsp://user:pass@camera/live"], forbidden=[])


def test_retained_text_scanner_accepts_privacy_safe_metadata() -> None:
    scan_retained_text(
        ["transition=transition-1 edge=edge-1 attempts=2 backend=backend-1 incident=incident-1"],
        forbidden=["secret-value", "camera.example"],
    )


def test_no_browser_or_human_adjudication_dependency_is_imported() -> None:
    source = Path("tests_support/alert_amplification_harness.py").read_text()

    assert "playwright" not in source.lower()
    assert "false_positive" not in source.lower()
    assert "true_positive" not in source.lower()

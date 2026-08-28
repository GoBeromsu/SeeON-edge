#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from backend.app.features.connection import hub_url  # noqa: E402
from tests_support.alert_amplification_harness import (  # noqa: E402
    INSECURE_HTTP_ENV,
    ClockSample,
    CorrelationRow,
    DescriptorScanScope,
    build_diagnostic_overlay,
    classify_rows,
    collect_stable_incidents,
    descriptor_scan_scope,
    evaluate_clock,
    inspect_tmpfs,
    probe_procfs_visibility,
    rows_from_relations,
    validate_insecure_http_assignments,
    validate_insecure_http_source_declarations,
    validate_no_insecure_http_assignments,
    validate_route_ledger,
    validate_temporal_order,
    verify_tmpfs_destroyed,
)
from tests_support.local_backend_fixture import LocalBackendFixture  # noqa: E402

_AUTH = {"Authorization": "Bearer fixture-token"}
_SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de801"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Privacy-safe API alert-amplification diagnostic preflight"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "fixture-preflight",
        help="exercise exact local Hub routes without external network or credentials",
    )
    overlay = subcommands.add_parser(
        "overlay", help="render the secret-free internal diagnostic Compose overlay"
    )
    overlay.add_argument("--run-id", required=True)
    overlay.add_argument("--fixture-origin", default="http://diagnostic-hub:8080")
    probe = subcommands.add_parser(
        "host-probe", help="emit privacy-safe prerequisite availability booleans"
    )
    probe.add_argument("--probe-envelope-fd", required=True, type=int)
    classify = subcommands.add_parser(
        "classify", help="classify an offline privacy-safe T/E/A/B/I test artifact"
    )
    classify.add_argument("artifact", type=Path)
    projection = subcommands.add_parser(
        "incident-projection",
        help="query the authenticated ml-api incident projection twice",
    )
    projection.add_argument("--base-url", required=True)
    projection.add_argument("--token-fd", required=True, type=int)
    measured = subcommands.add_parser(
        "measured-run",
        help="join measured T/E/A/B evidence to the authenticated API projection",
    )
    measured.add_argument("evidence_json", type=Path)
    measured.add_argument("--run-receipt", required=True, type=Path)
    measured.add_argument("--base-url", required=True)
    measured.add_argument("--token-fd", required=True, type=int)
    semantic = subcommands.add_parser(
        "semantic-http",
        help="validate a complete Docker Compose JSON render",
    )
    semantic.add_argument("compose_json", type=Path)
    semantic.add_argument("--package-compose-json", required=True, type=Path)
    semantic.add_argument("--image-inspect-json", required=True, action="append", type=Path)
    semantic.add_argument("--container-inspect-json", required=True, type=Path)
    semantic.add_argument("--run-receipt", required=True, type=Path)
    semantic.add_argument("--repo-root", type=Path, default=ROOT)
    tmpfs = subcommands.add_parser(
        "verify-tmpfs", help="verify already-mounted dedicated clip/state tmpfs paths"
    )
    tmpfs.add_argument("--clip-path", required=True, type=Path)
    tmpfs.add_argument("--state-path", required=True, type=Path)
    destroyed = subcommands.add_parser(
        "verify-tmpfs-destroyed",
        help="prove a prior tmpfs mount identity is no longer reachable",
    )
    destroyed.add_argument("--path", required=True, type=Path)
    destroyed.add_argument("--prior-device", required=True, type=int)
    destroyed.add_argument("--prior-inode", required=True, type=int)
    args = parser.parse_args()

    try:
        if args.command == "fixture-preflight":
            result = _fixture_preflight()
        elif args.command == "overlay":
            result = {"overlay": build_diagnostic_overlay(args.run_id, args.fixture_origin)}
        elif args.command == "host-probe":
            result = _host_probe_from_fd(args.probe_envelope_fd)
        elif args.command == "classify":
            result = _classify(args.artifact)
        elif args.command == "incident-projection":
            result = _incident_projection(args.base_url, args.token_fd)
        elif args.command == "measured-run":
            result = _measured_run(
                args.evidence_json,
                args.base_url,
                args.token_fd,
                run_receipt_path=args.run_receipt,
            )
        elif args.command == "semantic-http":
            result = _semantic_http(
                args.compose_json,
                package_path=args.package_compose_json,
                image_paths=args.image_inspect_json,
                container_path=args.container_inspect_json,
                run_receipt_path=args.run_receipt,
                repo_root=args.repo_root,
            )
        elif args.command == "verify-tmpfs":
            result = _verify_tmpfs(args.clip_path, args.state_path)
        else:
            verify_tmpfs_destroyed(
                args.path,
                prior_device=args.prior_device,
                prior_inode=args.prior_inode,
            )
            result = {"destroyed": True}
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "passed", **result}, ensure_ascii=False, sort_keys=True))
    return 0


def _host_probe_from_fd(fd: int) -> dict[str, Any]:
    envelope = _read_json_fd(fd)
    if set(envelope) != {"clip_path", "expected_clip_sha256"}:
        raise ValueError("host probe envelope has invalid fields")
    clip_path = envelope["clip_path"]
    expected = envelope["expected_clip_sha256"]
    if not isinstance(clip_path, str) or not clip_path.startswith("/"):
        raise ValueError("host probe clip path must be absolute")
    if not isinstance(expected, str):
        raise TypeError("host probe clip digest must be text")
    return _host_probe(Path(clip_path), expected)


def _host_probe(clip_path: Path, expected_clip_sha256: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_clip_sha256) is None:
        raise ValueError("expected clip digest must be lowercase SHA-256")
    clip_present = clip_path.is_file()
    clip_pin_match = clip_present and _sha256_file(clip_path) == expected_clip_sha256
    gateway_candidates = (
        shutil.which("mediamtx"),
        "/usr/local/bin/mediamtx" if Path("/usr/local/bin/mediamtx").is_file() else None,
    )
    # The approved plan isolates through disposable Compose services on an
    # internal network with tmpfs mounts, not raw unprivileged user namespaces,
    # so container-runtime namespace/mount capability is the property to probe.
    namespace_available, _ = _availability_command(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            "type=tmpfs,destination=/probe,tmpfs-size=1m,tmpfs-mode=0700",
            "alpine:3.22",
            "true",
        ],
        timeout=60,
    )
    swap_active, swap_text = _availability_command(
        ["swapon", "--noheadings", "--show=NAME"], timeout=5, capture=True
    )
    scan_scope: DescriptorScanScope = descriptor_scan_scope()
    docker_available, image_text = _availability_command(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        timeout=10,
        capture=True,
    )
    image_text = image_text.lower() if docker_available else ""
    return {
        "schemaVersion": 1,
        "kind": "alert-amplification-host-prerequisite-probe",
        "observed_at": datetime.now(UTC).isoformat(),
        "gateway_candidate_present": any(gateway_candidates),
        "gateway_image_present": "mediamtx" in image_text or "bluenviron" in image_text,
        "pinned_clip_present": clip_present,
        "pinned_clip_match": clip_pin_match,
        "container_isolation_available": namespace_available,
        "host_swap_enabled": bool(swap_active and swap_text.strip()),
        "procfs_visibility_usable": probe_procfs_visibility(),
        "descriptor_scan_skipped_protected": scan_scope.skipped_protected_own_uid,
        "contains_paths": False,
        "contains_process_inventory": False,
        "contains_secrets": False,
        "contains_media_hashes": False,
    }


def _availability_command(
    command: list[str],
    *,
    timeout: int,
    capture: bool = False,
) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=capture,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""
    output = result.stdout if capture and isinstance(result.stdout, str) else ""
    return result.returncode == 0, output


def _fixture_preflight() -> dict[str, Any]:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)
    enrollment = client.post(
        "/api/v1/edge/enrollments/verify",
        headers=_AUTH,
        json={
            "schemaVersion": 1,
            "facilityCode": "NH-7H2K9M4QXP",
            "clientInstallationRef": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
        },
    )
    _require_status(enrollment.status_code, 200, "enrollment")
    config = client.get(f"/api/v1/ml-config/{fixture.facility_id}", headers=_AUTH)
    _require_status(config.status_code, 200, "config")
    capabilities = client.get(
        "/api/v1/events/capabilities",
        headers=_AUTH,
        params={"camera_id": "room-camera"},
    )
    _require_status(capabilities.status_code, 200, "capabilities")
    heartbeat = client.post(
        "/api/v1/events/heartbeat", headers=_AUTH, json={"camera_id": "room-camera"}
    )
    _require_status(heartbeat.status_code, 200, "heartbeat")
    event = {
        "edge_event_id": "preflight-edge-event",
        "camera_id": "room-camera",
        "type": "fall",
        "detected_at": "2026-08-16T00:00:00Z",
        "confidence": 0.9,
    }
    first = client.post("/api/v1/events", headers=_AUTH, json=event)
    replay = client.post("/api/v1/events", headers=_AUTH, json=event)
    _require_status(first.status_code, 201, "event")
    _require_status(replay.status_code, 201, "event replay")
    if first.json() != replay.json():
        raise RuntimeError("event replay did not converge to a stable receipt")
    receipt = first.json()
    snapshot = client.put(
        f"/api/v1/events/{receipt['event_id']}/snapshot",
        headers={**_AUTH, "Content-Type": "image/jpeg"},
        content=b"\xff\xd8diagnostic\xff\xd9",
    )
    _require_status(snapshot.status_code, 201, "snapshot discard")
    topology = client.put(
        f"/api/v1/edge/topology-snapshots/{_SNAPSHOT_ID}",
        headers=_AUTH,
        json={
            "schemaVersion": 1,
            "edgeInstallationId": fixture.edge_installation_id,
            "enrollmentGeneration": 1,
            "clientRevision": 1,
            "expectedServerRevision": 0,
            "floors": [],
        },
    )
    _require_status(topology.status_code, 200, "topology")
    confirm = client.post(
        f"/api/v1/edge/topology-snapshots/{_SNAPSHOT_ID}/confirm",
        headers=_AUTH,
        json={
            "schemaVersion": 1,
            "confirmationId": topology.json()["omissions"]["confirmationId"],
            "digest": topology.json()["omissions"]["digest"],
            "expectedServerRevision": topology.json()["serverRevision"],
        },
    )
    _require_status(confirm.status_code, 200, "topology confirmation")
    obsolete = client.post("/v1/events", headers=_AUTH, json=event)
    _require_status(obsolete.status_code, 404, "obsolete path rejection")
    validate_route_ledger(fixture.route_ledger)
    if fixture.retained_media_bytes != 0:
        raise RuntimeError("fixture retained media bytes")
    return {
        "evidence_scope": "synthetic_fixture_preflight_only",
        "event_idempotency": capabilities.json()["event_idempotency"],
        "clip_export": capabilities.json()["clip_export"],
        "stable_event_receipt": True,
        "snapshot_bytes_retained": fixture.retained_media_bytes,
        "topology_revision": confirm.json()["serverRevision"],
        "route_count": len(fixture.route_ledger),
        "obsolete_paths_rejected": True,
    }


def _classify(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise TypeError("classification artifact must be a JSON array")
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("classification rows must be objects")
        rows.append(
            CorrelationRow(
                _optional_text(item.get("transition_id")),
                _optional_text(item.get("edge_event_id")),
                _integer_tuple(item.get("attempt_ordinals")),
                _text_tuple(item.get("backend_event_ids")),
                _text_tuple(item.get("incident_ids")),
                str(item.get("terminal_state", "")),
                item.get("clock_order_valid") is True,
                # Fail closed: absent or false order evidence can never enable
                # an order-dependent claim from an offline artifact.
                item.get("order_evidence_valid") is True,
            )
        )
    result = classify_rows(rows)
    return {
        "evidence_scope": "offline_fixture_only",
        "outcome": result.outcome.value,
        "model_policy_cause": result.model_policy_cause,
        "reason": result.reason,
        "row_count": len(rows),
        "frontend_display_tested": False,
        "human_adjudication_used": False,
    }


def _incident_projection(base_url: str, token_fd: int) -> dict[str, Any]:
    _validate_incident_origin(base_url)
    token = _read_secret_fd(token_fd)
    try:
        with httpx.Client(base_url=base_url, timeout=5.0, follow_redirects=False) as client:
            incidents = collect_stable_incidents(client, bearer_token=token)
    finally:
        token = ""
    return {
        "projection_count": len(incidents),
        "projections": [
            {
                "incident_id": incident.incident_id,
                "edge_event_id": incident.edge_event_id,
                "detected_at": incident.detected_at,
                "lifecycle_state": incident.lifecycle_state,
                "event_delivery_state": incident.event_delivery_state,
                "projection_timestamp": incident.projection_timestamp,
            }
            for incident in incidents
        ],
        "stable_poll": True,
        "human_adjudication_used": False,
    }


def _measured_run(
    path: Path,
    base_url: str,
    token_fd: int,
    *,
    run_receipt_path: Path,
) -> dict[str, Any]:
    _validate_incident_origin(base_url)
    receipt = _load_run_receipt(run_receipt_path)
    if receipt["evidence_sha256"] != _sha256_file(path):
        raise ValueError("measured evidence does not match the run receipt")
    evidence = _parse_measured_evidence(path)
    if (
        evidence["run_id"] != receipt["run_id"]
        or evidence["source_revision"] != receipt["source_revision"]
    ):
        raise ValueError("measured evidence identity does not match the run receipt")
    token = _read_secret_fd(token_fd)
    try:
        with httpx.Client(base_url=base_url, timeout=5.0, follow_redirects=False) as client:
            incidents = collect_stable_incidents(client, bearer_token=token)
    finally:
        token = ""
    clock_gate = evaluate_clock(evidence["clock_samples"])
    projection_timestamp_complete = bool(incidents) and all(
        incident.projection_timestamp is not None for incident in incidents
    )
    # Presence is not ordering: the full attempt -> receipt -> projection
    # relation must hold within the measured clock uncertainty.
    temporal_order_valid = projection_timestamp_complete and all(
        validate_temporal_order(
            attempt_times=evidence["attempt_times"].get(incident.edge_event_id, ()),
            receipt_times=evidence["receipt_times"].get(incident.edge_event_id, ()),
            projection_time=_epoch_seconds(incident.projection_timestamp),
            uncertainty_ms=clock_gate.max_uncertainty_ms,
        )
        for incident in incidents
    )
    order_evidence_valid = clock_gate.passed and temporal_order_valid
    terminal_states: dict[str, str] = {}
    for edge_id in evidence["transitions"]:
        states = {
            incident.event_delivery_state
            for incident in incidents
            if incident.edge_event_id == edge_id
        }
        terminal_states[edge_id] = "ACKED" if states == {"ACKED"} else ""
    rows = rows_from_relations(
        transitions=evidence["transitions"],
        attempts=evidence["attempts"],
        backend_event_ids=evidence["backend_event_ids"],
        incidents=incidents,
        terminal_states=terminal_states,
        clock_order_valid=clock_gate.passed,
        order_evidence_valid=order_evidence_valid,
    )
    result = classify_rows(rows)
    return {
        "run_id": evidence["run_id"],
        "source_revision": evidence["source_revision"],
        "outcome": result.outcome.value,
        "model_policy_cause": result.model_policy_cause,
        "reason": result.reason,
        "row_count": len(rows),
        "clock_gate_passed": clock_gate.passed,
        "projection_timestamp_complete": projection_timestamp_complete,
        "temporal_order_valid": temporal_order_valid,
        # False blocks every order-dependent claim and is promotion-ineligible,
        # even when an identity-derived finding remains reportable.
        "order_dependent_claims_eligible": order_evidence_valid,
        "promotion_eligible": order_evidence_valid
        and result.outcome.value != "판정 불가",
        "frontend_display_tested": False,
        "human_adjudication_used": False,
    }


def _parse_measured_evidence(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    required = {
        "schemaVersion",
        "run_id",
        "source_revision",
        "transitions",
        "attempts",
        "backend_event_ids",
        "attempt_times",
        "receipt_times",
        "clock_samples",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw["schemaVersion"] != 1:
        raise TypeError("measured evidence has an invalid envelope")
    run_id = raw["run_id"]
    source_revision = raw["source_revision"]
    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        raise ValueError("measured evidence has an invalid run_id")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("measured evidence has an invalid source_revision")
    transitions = _string_mapping(raw["transitions"])
    attempts = _integer_sequence_mapping(raw["attempts"])
    backend_event_ids = _string_sequence_mapping(raw["backend_event_ids"])
    samples_raw = raw["clock_samples"]
    if not isinstance(samples_raw, list):
        raise TypeError("clock_samples must be an array")
    samples: list[ClockSample] = []
    for sample in samples_raw:
        if not isinstance(sample, dict) or set(sample) != {
            "offset_ms",
            "uncertainty_ms",
            "rtt_ms",
        }:
            raise TypeError("clock sample has an invalid shape")
        values = (sample["offset_ms"], sample["uncertainty_ms"], sample["rtt_ms"])
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
            raise TypeError("clock sample values must be numbers")
        samples.append(ClockSample(*(float(value) for value in values)))
    return {
        "run_id": run_id,
        "source_revision": source_revision,
        "transitions": transitions,
        "attempts": attempts,
        "backend_event_ids": backend_event_ids,
        "attempt_times": _float_sequence_mapping(raw["attempt_times"]),
        "receipt_times": _float_sequence_mapping(raw["receipt_times"]),
        "clock_samples": tuple(samples),
    }


def _validate_incident_origin(base_url: str) -> None:
    parsed = urlsplit(base_url)
    allowed_hosts = {"ml-api", "127.0.0.1", "localhost", "::1"}
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in allowed_hosts
    ):
        raise ValueError("incident origin must be the isolated ml-api alias or loopback")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a path, query, or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("incident origin contains an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("incident origin contains an invalid port")


def _semantic_http(
    path: Path,
    *,
    package_path: Path,
    image_paths: list[Path],
    container_path: Path,
    run_receipt_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    receipt = _load_run_receipt(run_receipt_path)
    diagnostic_environments = _compose_environments(path)
    package_environments = _compose_environments(package_path)
    if receipt["diagnostic_compose_sha256"] != _sha256_file(path):
        raise ValueError("diagnostic Compose render does not match the run receipt")
    if receipt["package_compose_sha256"] != _sha256_file(package_path):
        raise ValueError("package Compose render does not match the run receipt")
    image_hashes = [_sha256_file(image_path) for image_path in image_paths]
    if image_hashes != receipt["image_inspect_sha256"]:
        raise ValueError("image inspect artifacts do not match the run receipt")
    if receipt["container_inspect_sha256"] != _sha256_file(container_path):
        raise ValueError("container inspect artifact does not match the run receipt")
    if sorted(diagnostic_environments) != sorted(receipt["diagnostic_services"]):
        raise ValueError("diagnostic service set does not match the run receipt")
    if sorted(package_environments) != sorted(receipt["package_services"]):
        raise ValueError("package service set does not match the run receipt")
    validate_insecure_http_assignments(diagnostic_environments)
    validate_no_insecure_http_assignments(package_environments)
    validate_insecure_http_source_declarations(repo_root.resolve(strict=True))
    observed_image_ids: list[str] = []
    for image_path in image_paths:
        observed_image_ids.extend(_validate_image_environment(image_path))
    if sorted(observed_image_ids) != sorted(receipt["image_ids"]):
        raise ValueError("image identities do not match the run receipt")
    if set(receipt["running_container_ids"]) != set(diagnostic_environments):
        raise ValueError("running container services do not cover the diagnostic topology")
    if set(receipt["service_image_ids"]) != set(diagnostic_environments):
        raise ValueError("service image bindings do not cover the diagnostic topology")
    _validate_container_identities(
        container_path,
        running_ids=receipt["running_container_ids"],
        service_image_ids=receipt["service_image_ids"],
        image_ids=set(receipt["image_ids"]),
    )
    _validate_companion_rejection()
    return {
        "run_id": receipt["run_id"],
        "source_revision": receipt["source_revision"],
        "diagnostic_service_count": len(diagnostic_environments),
        "package_service_count": len(package_environments),
        "image_count": len(image_paths),
        "effective_opt_in_service": "ml-api",
        "companion_rejection": True,
        "source_declaration_allowlist": True,
        "runtime_identity_bound": True,
    }


def _compose_environments(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("services"), dict):
        raise TypeError("Compose render must contain a services object")
    environments: dict[str, dict[str, str]] = {}
    for service, config in raw["services"].items():
        if not isinstance(service, str) or not isinstance(config, dict):
            raise TypeError("Compose service entries must be objects")
        environments[service] = _normalize_environment(config.get("environment"))
    return environments


def _validate_image_environment(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    images = raw if isinstance(raw, list) else [raw]
    if not images or any(not isinstance(image, dict) for image in images):
        raise TypeError("image inspect artifact must contain image objects")
    image_ids: list[str] = []
    for image in images:
        image_id = image.get("Id")
        if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
            raise ValueError("image inspect artifact is missing an immutable image ID")
        image_ids.append(image_id)
        config = image.get("Config")
        if not isinstance(config, dict):
            raise TypeError("image inspect artifact is missing Config")
        environment = config.get("Env")
        if environment is None:
            continue
        if not isinstance(environment, list) or any(
            not isinstance(item, str) for item in environment
        ):
            raise TypeError("image Config.Env must be a string list")
        if any(
            item.partition("=")[0] == INSECURE_HTTP_ENV
            for item in environment
        ):
            raise ValueError("cleartext opt-in leaked into an image environment")
    return image_ids


def _validate_container_identities(
    path: Path,
    *,
    running_ids: dict[str, str],
    service_image_ids: dict[str, str],
    image_ids: set[str],
) -> None:
    raw = json.loads(path.read_text())
    containers = raw if isinstance(raw, list) else [raw]
    observed: dict[str, str] = {}
    observed_images: dict[str, str] = {}
    for container in containers:
        if not isinstance(container, dict):
            raise TypeError("container inspect artifact must contain objects")
        container_id = container.get("Id")
        image_id = container.get("Image")
        config = container.get("Config")
        if (
            not isinstance(container_id, str)
            or not container_id
            or not isinstance(image_id, str)
            or image_id not in image_ids
            or not isinstance(config, dict)
        ):
            raise ValueError("container inspect identity is invalid")
        labels = config.get("Labels")
        service = labels.get("com.docker.compose.service") if isinstance(labels, dict) else None
        if not isinstance(service, str) or not service:
            raise ValueError("container inspect is missing its Compose service identity")
        if service in observed:
            raise ValueError("container inspect contains duplicate service identities")
        observed[service] = container_id
        observed_images[service] = image_id
    if observed != running_ids:
        raise ValueError("running container identities do not match the run receipt")
    if observed_images != service_image_ids:
        raise ValueError("running service image identities do not match the run receipt")


def _load_run_receipt(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    required = {
        "schemaVersion",
        "receipt_id",
        "run_id",
        "source_revision",
        "evidence_sha256",
        "diagnostic_compose_sha256",
        "package_compose_sha256",
        "image_inspect_sha256",
        "container_inspect_sha256",
        "diagnostic_services",
        "package_services",
        "image_ids",
        "running_container_ids",
        "service_image_ids",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw["schemaVersion"] != 1:
        raise TypeError("run receipt has an invalid envelope")
    canonical = json.dumps(
        {key: value for key, value in raw.items() if key != "receipt_id"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if raw["receipt_id"] != hashlib.sha256(canonical).hexdigest():
        raise ValueError("run receipt identity is invalid")
    for key in (
        "evidence_sha256",
        "diagnostic_compose_sha256",
        "package_compose_sha256",
        "container_inspect_sha256",
    ):
        if not _is_sha256(raw[key]):
            raise ValueError(f"run receipt contains invalid {key}")
    image_hashes = raw["image_inspect_sha256"]
    if not isinstance(image_hashes, list) or not image_hashes or any(
        not _is_sha256(value) for value in image_hashes
    ):
        raise ValueError("run receipt contains invalid image inspect hashes")
    if (
        not isinstance(raw["run_id"], str)
        or not raw["run_id"]
        or not isinstance(raw["source_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", raw["source_revision"]) is None
    ):
        raise ValueError("run receipt contains invalid run/source identity")
    for key in ("diagnostic_services", "package_services", "image_ids"):
        values = raw[key]
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"run receipt contains invalid {key}")
        if len(values) != len(set(values)):
            raise ValueError(f"run receipt contains duplicate {key}")
    if any(
        re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None for value in raw["image_ids"]
    ):
        raise ValueError("run receipt contains invalid image IDs")
    for key in ("running_container_ids", "service_image_ids"):
        mapping = raw[key]
        if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in mapping.items()
        ):
            raise ValueError(f"run receipt contains invalid {key}")
    if any(
        value not in raw["image_ids"] for value in raw["service_image_ids"].values()
    ):
        raise ValueError("run receipt service image binding is not in image IDs")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_companion_rejection() -> None:
    if hub_url.allow_insecure_http_from_env():
        raise ValueError("semantic gate process inherited the diagnostic cleartext opt-in")
    if hub_url.hub_url_transport_allowed("http://diagnostic-hub:8080"):
        raise ValueError("companion process accepted diagnostic cleartext without opt-in")


def _verify_tmpfs(clip_path: Path, state_path: Path) -> dict[str, Any]:
    clip = inspect_tmpfs(clip_path, maximum_bytes=2 * 1024**3)
    state = inspect_tmpfs(state_path, maximum_bytes=512 * 1024**2)
    return {
        "clip": _tmpfs_result(clip),
        "state": _tmpfs_result(state),
        "destruction_required_after_session": True,
    }


def _tmpfs_result(evidence: Any) -> dict[str, object]:
    return {
        "size_bytes": evidence.size_bytes,
        "free_bytes": evidence.free_bytes,
        "total_inodes": evidence.total_inodes,
        "free_inodes": evidence.free_inodes,
        "mode": f"{evidence.mode:04o}",
        "device": evidence.device,
        "inode": evidence.inode,
    }


def _read_json_fd(fd: int) -> dict[str, object]:
    if fd < 3:
        raise ValueError("private descriptor must not be stdin/stdout/stderr")
    with os.fdopen(os.dup(fd), encoding="utf-8") as stream:
        raw = stream.read(8193)
    if not raw or len(raw) > 8192:
        raise ValueError("private descriptor contains an invalid envelope")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("private descriptor contains invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("private descriptor JSON must be an object")
    return value


def _read_secret_fd(fd: int) -> str:
    if fd < 3:
        raise ValueError("token descriptor must not be stdin/stdout/stderr")
    with os.fdopen(os.dup(fd), encoding="utf-8") as stream:
        token = stream.read(4097)
    if not token or len(token) > 4096 or "\n" in token.strip("\n"):
        raise ValueError("token descriptor contains an invalid secret")
    return token.strip()


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError("identity mapping must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or not item:
            raise ValueError("identity mapping contains invalid values")
        result[key] = item
    return result


def _integer_sequence_mapping(value: object) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, dict):
        raise TypeError("attempt mapping must be an object")
    result: dict[str, tuple[int, ...]] = {}
    for key, item in value.items():
        parsed = _integer_tuple(item)
        if not isinstance(key, str) or not key or not parsed:
            raise ValueError("attempt mapping contains invalid values")
        result[key] = parsed
    return result


def _float_sequence_mapping(value: object) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, dict):
        raise TypeError("timestamp mapping must be an object")
    result: dict[str, tuple[float, ...]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, list) or not item:
            raise ValueError("timestamp mapping contains invalid values")
        if any(isinstance(v, bool) or not isinstance(v, int | float) for v in item):
            raise ValueError("timestamp mapping contains non-numeric values")
        result[key] = tuple(float(v) for v in item)
    return result


def _epoch_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _string_sequence_mapping(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise TypeError("backend identity mapping must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for key, item in value.items():
        parsed = _text_tuple(item)
        if not isinstance(key, str) or not key or not parsed:
            raise ValueError("backend identity mapping contains invalid values")
        result[key] = parsed
    return result


def _normalize_environment(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): "" if item is None else str(item) for key, item in value.items()}
    if isinstance(value, list):
        result: dict[str, str] = {}
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Compose environment list values must be strings")
            key, separator, content = item.partition("=")
            result[key] = content if separator else ""
        return result
    raise TypeError("Compose environment must be an object or list")


def _require_status(actual: int, expected: int, step: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{step} returned {actual}, expected {expected}")


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        return ()
    return tuple(value)


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return ()
    return tuple(value)


if __name__ == "__main__":
    raise SystemExit(main())

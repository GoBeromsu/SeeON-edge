from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import stat
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from tests_support.local_backend_fixture import RouteRecord

INSECURE_HTTP_ENV = "API_BACKEND_" "ALLOW_INSECURE_HTTP"


class DiagnosticOutcome(StrEnum):
    SINGLE_API_DELIVERY = "single_api_delivery"
    TRANSPORT_RETRY = "transport_retry"
    BACKEND_IDENTITY_DUPLICATION = "backend_identity_duplication"
    API_INCIDENT_DUPLICATION = "api_incident_duplication"
    WORKER_REFIRE = "worker_refire"
    REPEATED_MACHINE_POSITIVE = "repeated_machine_positive"
    INCONCLUSIVE = "판정 불가"


@dataclass(frozen=True, slots=True)
class CorrelationRow:
    transition_id: str | None
    edge_event_id: str | None
    attempt_ordinals: tuple[int, ...]
    backend_event_ids: tuple[str, ...]
    incident_ids: tuple[str, ...]
    terminal_state: str
    clock_order_valid: bool = True
    # Separate from identity completeness: temporal-order evidence (for example
    # an API projection timestamp) is required only for order-dependent claims.
    order_evidence_valid: bool = True


@dataclass(frozen=True, slots=True)
class IncidentProjection:
    incident_id: str
    edge_event_id: str
    detected_at: str
    lifecycle_state: str
    event_delivery_state: str | None
    projection_timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class Classification:
    outcome: DiagnosticOutcome
    model_policy_cause: str
    reason: str


@dataclass(frozen=True, slots=True)
class ClockSample:
    offset_ms: float
    uncertainty_ms: float
    rtt_ms: float


@dataclass(frozen=True, slots=True)
class ClockGate:
    median_offset_ms: float
    max_uncertainty_ms: float
    max_rtt_ms: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SessionCounts:
    attempts: int
    successes: int
    closes: int
    reopen_attempts: int
    reopen_successes: int


class SessionBudgetExceeded(RuntimeError):
    pass


class SessionLatch:
    """Run-owned five-minute latch for one attempt and at most one session."""

    def __init__(
        self,
        role: str,
        *,
        maximum_seconds: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if role not in {"room", "corridor"}:
            raise ValueError("role must be room or corridor")
        if not math.isfinite(maximum_seconds) or maximum_seconds <= 0 or maximum_seconds > 300:
            raise ValueError("session duration must be within (0, 300] seconds")
        self.role = role
        self._monotonic = monotonic
        self._started_at = monotonic()
        self._deadline = self._started_at + maximum_seconds
        self._attempts = 0
        self._successes = 0
        self._closes = 0
        self._reopen_attempts = 0
        self._reopen_successes = 0
        self._torn_down = False

    @property
    def counts(self) -> SessionCounts:
        return SessionCounts(
            self._attempts,
            self._successes,
            self._closes,
            self._reopen_attempts,
            self._reopen_successes,
        )

    def attempt(self) -> None:
        self._require_active()
        if self._attempts:
            self._reopen_attempts += 1
            raise SessionBudgetExceeded("upstream attempt budget exhausted")
        self._attempts = 1

    def connected(self) -> None:
        self._require_active()
        if self._attempts != 1 or self._successes:
            self._reopen_successes += 1
            raise SessionBudgetExceeded("upstream session budget exhausted")
        self._successes = 1

    def closed(self) -> None:
        if self._successes != 1 or self._closes:
            raise SessionBudgetExceeded("no open upstream session")
        self._closes = 1
        self._require_within_deadline()

    def teardown(self) -> None:
        if self._successes and self._closes != 1:
            raise SessionBudgetExceeded("connected session must close before teardown")
        self._torn_down = True
        self._require_within_deadline()

    def _require_within_deadline(self) -> None:
        if self._monotonic() > self._deadline:
            raise SessionBudgetExceeded("five-minute session deadline exceeded")

    def _require_active(self) -> None:
        if self._torn_down:
            raise SessionBudgetExceeded("session was already torn down")
        self._require_within_deadline()


class IncidentResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...


class IncidentClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, object],
    ) -> IncidentResponse: ...


def collect_stable_incidents(
    client: IncidentClient,
    *,
    bearer_token: str,
    path: str = "/api/v1/incidents",
    maximum_pages: int = 10,
) -> tuple[IncidentProjection, ...]:
    if not bearer_token or maximum_pages < 1 or maximum_pages > 10:
        raise ValueError("invalid incident collector configuration")
    first = _collect_incidents(client, path, bearer_token, maximum_pages)
    second = _collect_incidents(client, path, bearer_token, maximum_pages)
    if first != second:
        raise ValueError("incident projection changed across stable polling")
    return first


def rows_from_relations(
    *,
    transitions: Mapping[str, str],
    attempts: Mapping[str, Sequence[int]],
    backend_event_ids: Mapping[str, Sequence[str]],
    incidents: Sequence[IncidentProjection],
    terminal_states: Mapping[str, str],
    clock_order_valid: bool,
    order_evidence_valid: bool = True,
) -> tuple[CorrelationRow, ...]:
    incident_ids: dict[str, list[str]] = {}
    for incident in incidents:
        incident_ids.setdefault(incident.edge_event_id, []).append(incident.incident_id)
    edge_ids = set(transitions) | set(attempts) | set(backend_event_ids) | set(incident_ids)
    return tuple(
        CorrelationRow(
            transitions.get(edge_id),
            edge_id,
            tuple(attempts.get(edge_id, ())),
            tuple(backend_event_ids.get(edge_id, ())),
            tuple(incident_ids.get(edge_id, ())),
            terminal_states.get(edge_id, ""),
            clock_order_valid,
            order_evidence_valid,
        )
        for edge_id in sorted(edge_ids)
    )


def classify_rows(rows: Sequence[CorrelationRow]) -> Classification:
    if not rows:
        return _inconclusive("no correlation rows")

    # A positive multiplication finding is sound even when the chain is
    # otherwise incomplete: two accepted backend identities for one immutable
    # edge_event_id is already a defect, and a later missing join cannot make
    # it healthy. Missing identities only block concluding *health*.
    duplication = _multiplication_defect(rows)
    if duplication is not None:
        return duplication

    transition_to_edges: dict[str, set[str]] = {}
    edge_to_transitions: dict[str, set[str]] = {}
    edge_to_attempts: dict[str, set[int]] = {}
    edge_to_backends: dict[str, set[str]] = {}
    edge_to_incidents: dict[str, set[str]] = {}
    incident_to_edges: dict[str, set[str]] = {}
    edge_terminal_states: dict[str, set[str]] = {}
    for row in rows:
        if not _row_has_identities(row):
            return _inconclusive("missing or unstable machine/API identity")
        assert row.transition_id is not None
        assert row.edge_event_id is not None
        transition_to_edges.setdefault(row.transition_id, set()).add(row.edge_event_id)
        edge_to_transitions.setdefault(row.edge_event_id, set()).add(row.transition_id)
        edge_to_attempts.setdefault(row.edge_event_id, set()).update(row.attempt_ordinals)
        edge_to_backends.setdefault(row.edge_event_id, set()).update(row.backend_event_ids)
        edge_to_incidents.setdefault(row.edge_event_id, set()).update(row.incident_ids)
        edge_terminal_states.setdefault(row.edge_event_id, set()).add(row.terminal_state)
        for incident_id in row.incident_ids:
            incident_to_edges.setdefault(incident_id, set()).add(row.edge_event_id)

    if any(len(values) != 1 for values in edge_to_transitions.values()):
        return _inconclusive("one edge_event_id mapped to inconsistent transition identities")
    if any(len(values) != 1 for values in edge_to_backends.values()):
        return Classification(
            DiagnosticOutcome.BACKEND_IDENTITY_DUPLICATION,
            "판정 불가",
            "one edge_event_id mapped to multiple accepted backend event IDs",
        )
    if any(len(values) != 1 for values in edge_to_incidents.values()):
        return Classification(
            DiagnosticOutcome.API_INCIDENT_DUPLICATION,
            "판정 불가",
            "one edge/backend identity mapped to multiple API incident IDs",
        )
    if any(len(values) != 1 for values in incident_to_edges.values()):
        return _inconclusive("one API incident identity mapped to multiple edge_event_id values")
    if any(values != {"ACKED"} for values in edge_terminal_states.values()):
        return _inconclusive("delivery did not reach one stable ACKED state")
    for ordinals in edge_to_attempts.values():
        if not ordinals or sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            return _inconclusive("delivery attempt ordinals are missing or non-contiguous")
    if any(len(edge_ids) > 1 for edge_ids in transition_to_edges.values()):
        return Classification(
            DiagnosticOutcome.WORKER_REFIRE,
            "판정 불가",
            "one decision transition mapped to multiple edge_event_id values",
        )
    if len(transition_to_edges) > 1:
        return Classification(
            DiagnosticOutcome.REPEATED_MACHINE_POSITIVE,
            "판정 불가",
            "distinct machine-positive transitions cannot establish model/policy truth",
        )
    if any(len(ordinals) > 1 for ordinals in edge_to_attempts.values()):
        # "Ordered attempts converged" is an order-dependent claim, so it is
        # enforced-blocked without temporal-order evidence. Identity-derived
        # findings above remain reportable.
        if not all(row.order_evidence_valid for row in rows):
            return _inconclusive(
                "ordered-attempt convergence requires temporal-order evidence"
            )
        return Classification(
            DiagnosticOutcome.TRANSPORT_RETRY,
            "판정 불가",
            "ordered attempts converged to one backend event and one API incident",
        )
    return Classification(
        DiagnosticOutcome.SINGLE_API_DELIVERY,
        "판정 불가",
        "one transition converged through one API incident",
    )


def evaluate_clock(samples: Sequence[ClockSample]) -> ClockGate:
    if len(samples) < 5 or any(not _valid_clock_sample(sample) for sample in samples):
        return ClockGate(0.0, 0.0, 0.0, False)
    offsets = sorted(sample.offset_ms for sample in samples)
    median = offsets[len(offsets) // 2]
    uncertainty = max(sample.uncertainty_ms for sample in samples)
    max_rtt = max(sample.rtt_ms for sample in samples)
    return ClockGate(
        median,
        uncertainty,
        max_rtt,
        abs(median) + uncertainty <= 250.0 and max_rtt <= 200.0,
    )


def validate_temporal_order(
    *,
    attempt_times: Sequence[float],
    receipt_times: Sequence[float],
    projection_time: float | None,
    uncertainty_ms: float,
) -> bool:
    """True only when attempt -> receipt -> projection holds within uncertainty.

    Presence of a timestamp is never sufficient: a reversed or malformed
    relation must not enable an order-dependent claim.
    """

    if projection_time is None or not attempt_times or not receipt_times:
        return False
    values = (*attempt_times, *receipt_times, projection_time, uncertainty_ms)
    if any(not isinstance(value, int | float) or not math.isfinite(value) for value in values):
        return False
    if uncertainty_ms < 0:
        return False
    if len(receipt_times) != len(attempt_times):
        return False
    slack = uncertainty_ms / 1000.0
    if list(attempt_times) != sorted(attempt_times):
        return False
    if list(receipt_times) != sorted(receipt_times):
        return False
    for attempt, receipt in zip(attempt_times, receipt_times, strict=True):
        if receipt + slack < attempt:
            return False
        if projection_time + slack < receipt:
            return False
    return True


def validate_insecure_http_assignments(
    service_environments: Mapping[str, Mapping[str, str]],
) -> None:
    enabled = [
        service
        for service, environment in service_environments.items()
        if environment.get(INSECURE_HTTP_ENV) == "1"
    ]
    if enabled != ["ml-api"]:
        raise ValueError(
            "diagnostic cleartext opt-in must be effectively assigned exactly once to ml-api"
        )
    for service, environment in service_environments.items():
        value = environment.get(INSECURE_HTTP_ENV)
        if service != "ml-api" and value not in (None, "", "0"):
            raise ValueError(f"unexpected cleartext opt-in on {service}")


def validate_no_insecure_http_assignments(
    service_environments: Mapping[str, Mapping[str, str]],
) -> None:
    leaked = [
        service
        for service, environment in service_environments.items()
        if INSECURE_HTTP_ENV in environment
    ]
    if leaked:
        raise ValueError(
            "cleartext opt-in must be absent from package services: "
            + ",".join(sorted(leaked))
        )


def validate_insecure_http_source_declarations(repo_root: Path) -> None:
    frozen_reference_allowlist = {
        "backend/app/features/connection/hub_url.py",
        "backend/app/features/connection/store.py",
        "compose.edge.yaml",
        "scripts/edge-preflight/check-env.sh",
        "tests/conftest.py",
        "tests/test_hub_url_transport_policy.py",
    }
    effective_assignment_allowlist = {
        "tests/conftest.py",
        "tests/test_hub_url_transport_policy.py",
    }
    skipped_dirs = {".git", ".gjc", ".venv", "node_modules", "dist"}
    observed_references: set[str] = set()
    effective_assignments: set[str] = set()
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root)
        if any(part in skipped_dirs for part in relative.parts):
            continue
        if path.suffix not in {".py", ".sh", ".yaml", ".yml"} and not path.name.startswith(
            "Dockerfile"
        ):
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        if INSECURE_HTTP_ENV not in content:
            continue
        relative_text = relative.as_posix()
        observed_references.add(relative_text)
        if _contains_effective_insecure_http_assignment(path, content):
            effective_assignments.add(relative_text)
    unexpected_assignments = effective_assignments - effective_assignment_allowlist
    if unexpected_assignments:
        raise ValueError(
            "unexpected cleartext opt-in source assignments: "
            + ",".join(sorted(unexpected_assignments))
        )
    missing_references = frozen_reference_allowlist - observed_references
    unexpected_references = observed_references - frozen_reference_allowlist
    if missing_references:
        raise ValueError(
            "frozen cleartext policy references are missing: "
            + ",".join(sorted(missing_references))
        )
    if unexpected_references:
        raise ValueError(
            "unexpected cleartext policy references: "
            + ",".join(sorted(unexpected_references))
        )


def _contains_effective_insecure_http_assignment(path: Path, content: str) -> bool:
    if path.suffix == ".py":
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            raise ValueError(f"cannot parse cleartext policy source {path}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "setenv" and node.args:
                    first = node.args[0]
                    if (
                        isinstance(first, ast.Constant)
                        and first.value == INSECURE_HTTP_ENV
                    ) or isinstance(first, ast.Name):
                        return True
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and "environ" in ast.unparse(target):
                        if INSECURE_HTTP_ENV in ast.unparse(target):
                            return True
        return False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(rf"^(?:export\s+)?{INSECURE_HTTP_ENV}=", stripped):
            return True
        if re.match(rf"^{INSECURE_HTTP_ENV}\s*:", stripped):
            return True
        if re.match(rf"^(?:ENV|ARG)\s+{INSECURE_HTTP_ENV}\b", stripped):
            return True
    return False


@dataclass(frozen=True, slots=True)
class GatewayAttestation:
    path: str
    version: str
    sha256: str
    uid: int
    gid: int
    mode: int
    device: int
    inode: int
    capabilities: frozenset[str]


def attest_gateway(
    path: Path,
    *,
    expected_version: str,
    expected_sha256: str,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
    probe: Callable[[int], tuple[str, Iterable[str]]],
    required_capabilities: frozenset[str],
) -> GatewayAttestation:
    if not path.is_absolute():
        raise ValueError("gateway path must be absolute")
    _reject_symlink_ancestors(path)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("gateway cannot be opened without following links") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("gateway must be a regular file")
        mode = stat.S_IMODE(metadata.st_mode)
        parent_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            parent_flags |= os.O_NOFOLLOW
        parent_descriptor = os.open(path.parent, parent_flags)
        try:
            parent_mode = stat.S_IMODE(os.fstat(parent_descriptor).st_mode)
        finally:
            os.close(parent_descriptor)
        if parent_mode & 0o022:
            raise ValueError("gateway parent directory is group/other writable")
        if (
            metadata.st_uid != expected_uid
            or metadata.st_gid != expected_gid
            or mode != expected_mode
        ):
            raise ValueError("gateway ownership or mode does not match the approved tuple")
        digest_builder = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed_version, observed_capabilities = probe(descriptor)
        if digest != expected_sha256 or observed_version != expected_version:
            raise ValueError("gateway version or digest does not match the approved tuple")
        capabilities = frozenset(observed_capabilities)
        if capabilities != required_capabilities:
            raise ValueError("gateway capabilities do not match the approved tuple")
        return GatewayAttestation(
            str(path),
            observed_version,
            digest,
            metadata.st_uid,
            metadata.st_gid,
            mode,
            metadata.st_dev,
            metadata.st_ino,
            capabilities,
        )
    finally:
        os.close(descriptor)


def _reject_symlink_ancestors(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("gateway ancestor path must not contain symlinks")


def build_diagnostic_overlay(run_id: str, fixture_origin: str) -> dict[str, object]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id) is None:
        raise ValueError("run_id is unsafe for a diagnostic overlay")
    parsed = urlsplit(fixture_origin)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "diagnostic-hub"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("fixture origin must be the credential-free diagnostic-hub alias")
    if parsed.port is None or not 1 <= parsed.port <= 65535:
        raise ValueError("fixture origin must include a valid port")
    return {
        "name": f"seeon-diagnostic-{run_id}",
        "services": {
            "diagnostic-hub": {"networks": ["diagnostic"]},
            "ml-api": {
                "environment": {
                    "API_BACKEND_BASE_URL": fixture_origin.rstrip("/"),
                    INSECURE_HTTP_ENV: "1",
                },
                "networks": ["diagnostic"],
            },
            "ml-worker": {"environment": {}, "networks": ["diagnostic"]},
        },
        "networks": {"diagnostic": {"internal": True}},
    }


@dataclass(frozen=True, slots=True)
class NetworkFlow:
    source: str
    destination: str
    port: int


@dataclass(frozen=True, slots=True)
class NamespacePolicy:
    services: frozenset[str]
    allowed_flows: frozenset[NetworkFlow]
    default_drop_input: bool
    default_drop_output: bool
    default_drop_forward: bool
    resolver_disabled: bool


def validate_namespace_policy(policy: NamespacePolicy) -> None:
    expected_services = frozenset(
        {"diagnostic-runner", "diagnostic-hub", "ml-api", "ml-worker", "role-gateway"}
    )
    expected_flows = frozenset(
        {
            NetworkFlow("ml-worker", "ml-api", 8000),
            NetworkFlow("ml-api", "diagnostic-hub", 8080),
            NetworkFlow("diagnostic-runner", "ml-api", 8000),
            NetworkFlow("ml-worker", "role-gateway", 8554),
        }
    )
    if policy.services != expected_services or policy.allowed_flows != expected_flows:
        raise ValueError("namespace service/flow allowlist does not match the approved topology")
    if not (
        policy.default_drop_input
        and policy.default_drop_output
        and policy.default_drop_forward
        and policy.resolver_disabled
    ):
        raise ValueError("namespace defaults and resolver must fail closed")


def validate_network_canaries(results: Mapping[str, bool]) -> None:
    expected = {
        "production_dns_blocked": True,
        "production_tcp_blocked": True,
        "local_hub_reachable": True,
        "local_api_reachable": True,
        "unexpected_egress_zero": True,
    }
    if dict(results) != expected:
        raise ValueError("network canary result does not match the fail-closed contract")


@dataclass(slots=True)
class _OwnedResource:
    kind: str
    stopped: bool = False
    destroyed: bool = False


class RunOwnedTeardown:
    def __init__(self, run_id: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id) is None:
            raise ValueError("invalid run_id")
        self.run_id = run_id
        self._resources: dict[str, _OwnedResource] = {}
        self._order: list[str] = []

    def acquire(self, resource_id: str, kind: str) -> None:
        if any(resource.stopped for resource in self._resources.values()):
            raise ValueError("resources cannot be acquired after teardown starts")
        if not resource_id.startswith(f"{self.run_id}:") or resource_id in self._resources:
            raise ValueError("resource is not uniquely owned by this run")
        if kind not in {"process", "container", "namespace", "mount", "firewall"}:
            raise ValueError("unsupported run-owned resource kind")
        self._resources[resource_id] = _OwnedResource(kind)
        self._order.append(resource_id)

    def stop(self, resource_id: str) -> None:
        resource = self._require(resource_id)
        if resource.stopped or resource.destroyed:
            raise ValueError("resource stop transition is not repeatable")
        remaining = [
            item
            for item in self._order[self._order.index(resource_id) + 1 :]
            if not self._resources[item].stopped
        ]
        if remaining:
            raise ValueError("run-owned resources must be stopped in reverse order")
        resource.stopped = True

    def destroy(self, resource_id: str) -> None:
        resource = self._require(resource_id)
        if resource.destroyed:
            raise ValueError("resource destroy transition is not repeatable")
        if not resource.stopped:
            raise ValueError("resource must be stopped before destruction")
        remaining = [
            item
            for item in self._order[self._order.index(resource_id) + 1 :]
            if not self._resources[item].destroyed
        ]
        if remaining:
            raise ValueError("run-owned resources must be destroyed in reverse order")
        resource.destroyed = True

    def finalize(self) -> None:
        if not self._resources or any(
            not resource.stopped or not resource.destroyed
            for resource in self._resources.values()
        ):
            raise ValueError("run-owned teardown is incomplete")

    def _require(self, resource_id: str) -> _OwnedResource:
        resource = self._resources.get(resource_id)
        if resource is None:
            raise ValueError("resource is not owned by this run")
        return resource


def validate_route_ledger(records: Iterable[RouteRecord]) -> None:
    for record in records:
        expected = _expected_route_status(record.method, record.path)
        if record.status_code not in expected:
            raise ValueError(
                f"unexpected route/status: {record.method} {record.path} {record.status_code}"
            )
        if record.path.endswith("/snapshot") and record.method == "PUT":
            if record.content_type != "image/jpeg":
                raise ValueError("snapshot exception must be image/jpeg")
            if record.actual_body_bytes <= 0 or record.actual_body_bytes > 512 * 1024:
                raise ValueError("snapshot exception exceeded its actual byte bound")


@dataclass(frozen=True, slots=True)
class TmpfsEvidence:
    path: str
    size_bytes: int
    free_bytes: int
    total_inodes: int
    free_inodes: int
    mode: int
    device: int
    inode: int
    mount_options: frozenset[str]


def inspect_tmpfs(path: Path, *, maximum_bytes: int) -> TmpfsEvidence:
    resolved = path.resolve(strict=True)
    mount = _mount_for(resolved)
    if mount is None:
        raise ValueError(f"no mount entry for {resolved}")
    mount_point, filesystem, options = mount
    if filesystem != "tmpfs" or mount_point != resolved:
        raise ValueError(f"{resolved} must be a dedicated tmpfs mount")
    required = {"nosuid", "nodev", "noexec", "noswap"}
    if not required.issubset(options):
        missing = ",".join(sorted(required - options))
        raise ValueError(f"tmpfs is missing required options: {missing}")
    metadata = resolved.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o700:
        raise ValueError(f"tmpfs mode must be 0700, got {mode:04o}")
    filesystem_stats = os.statvfs(resolved)
    size = filesystem_stats.f_frsize * filesystem_stats.f_blocks
    free = filesystem_stats.f_frsize * filesystem_stats.f_bavail
    if size > maximum_bytes:
        raise ValueError(f"tmpfs size {size} exceeds cap {maximum_bytes}")
    if size and (size - free) / size >= 0.80:
        raise ValueError("tmpfs reached the 80% stop threshold")
    if filesystem_stats.f_files <= 0 or filesystem_stats.f_favail <= 0:
        raise ValueError("tmpfs inode capacity is unavailable")
    return TmpfsEvidence(
        str(resolved),
        size,
        free,
        filesystem_stats.f_files,
        filesystem_stats.f_favail,
        mode,
        metadata.st_dev,
        metadata.st_ino,
        frozenset(options),
    )


@dataclass(frozen=True, slots=True)
class DescriptorScanScope:
    inspected_processes: int
    skipped_foreign_uid: int
    skipped_protected_own_uid: int


def descriptor_scan_scope() -> DescriptorScanScope:
    own_uid = os.getuid()
    inspected = foreign = protected = 0
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != own_uid:
                foreign += 1
                continue
        except (FileNotFoundError, PermissionError):
            foreign += 1
            continue
        try:
            tuple((process / "fd").iterdir())
        except PermissionError:
            protected += 1
            continue
        except FileNotFoundError:
            continue
        inspected += 1
    return DescriptorScanScope(inspected, foreign, protected)


def probe_procfs_visibility() -> bool:
    """True when this principal's own processes can be enumerated at all."""
    return descriptor_scan_scope().inspected_processes > 0


def verify_tmpfs_destroyed(path: Path, *, prior_device: int, prior_inode: int) -> None:
    resolved = path.resolve(strict=False)
    mount = _mount_for(resolved)
    if mount is not None and mount[0] == resolved:
        raise ValueError("tmpfs mount still exists after teardown")
    if path.exists():
        raise ValueError("prior tmpfs path still exists after teardown")
    if _open_references_for_path(resolved):
        raise ValueError("a process descriptor still references the prior tmpfs path")
    if prior_device == 0 or prior_inode <= 0:
        raise ValueError("prior tmpfs identity is invalid")


def scan_retained_text(values: Iterable[str], *, forbidden: Iterable[str]) -> None:
    needles = tuple(item for item in forbidden if item)
    for value in values:
        for needle in needles:
            if needle in value:
                raise ValueError("retained output contains a forbidden sentinel")
        lowered = value.lower()
        if "rtsp://" in lowered or "rtsps://" in lowered:
            raise ValueError("retained output contains a camera URL")


def _collect_incidents(
    client: IncidentClient,
    path: str,
    bearer_token: str,
    maximum_pages: int,
) -> tuple[IncidentProjection, ...]:
    cursor: str | None = None
    projections: list[IncidentProjection] = []
    seen_cursors: set[str] = set()
    for _ in range(maximum_pages):
        params: dict[str, object] = {"limit": 100}
        if cursor is not None:
            params["cursor"] = cursor
        response = client.get(
            path,
            headers={"Authorization": f"Bearer {bearer_token}"},
            params=params,
        )
        if response.status_code != 200:
            raise ValueError(f"incident API returned {response.status_code}")
        body = response.json()
        if not isinstance(body, dict) or set(body) != {"incidents", "pagination"}:
            raise ValueError("incident API returned an invalid envelope")
        incidents = body["incidents"]
        pagination = body["pagination"]
        if not isinstance(incidents, list) or not isinstance(pagination, dict):
            raise TypeError("incident API returned invalid pagination")
        for incident in incidents:
            projections.append(_parse_incident(incident))
        has_more = pagination.get("has_more")
        next_cursor = pagination.get("next_cursor")
        if has_more is False and next_cursor is None:
            return tuple(sorted(projections, key=lambda item: item.incident_id))
        if has_more is not True or not isinstance(next_cursor, str) or not next_cursor:
            raise ValueError("incident API pagination is inconsistent")
        if next_cursor in seen_cursors:
            raise ValueError("incident API pagination cursor repeated")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise ValueError("incident API exceeded the page bound")


def _parse_incident(value: object) -> IncidentProjection:
    if not isinstance(value, dict):
        raise TypeError("incident entry must be an object")
    required = {
        "incident_id",
        "edge_event_id",
        "detected_at",
        "lifecycle_state",
        "event_delivery_state",
        "review",
    }
    if not required.issubset(value):
        raise ValueError("incident entry is missing required fields")
    if value["review"] is not None:
        raise ValueError("human adjudication is forbidden in this diagnostic")
    text_fields = ("incident_id", "edge_event_id", "detected_at", "lifecycle_state")
    if any(not isinstance(value[field], str) or not value[field] for field in text_fields):
        raise ValueError("incident entry contains invalid identities")
    delivery_state = value["event_delivery_state"]
    if delivery_state is not None and not isinstance(delivery_state, str):
        raise ValueError("incident entry contains invalid delivery state")
    projection_timestamp = value.get("projection_timestamp")
    if projection_timestamp is not None and (
        not isinstance(projection_timestamp, str) or not projection_timestamp
    ):
        raise ValueError("incident entry contains invalid projection timestamp")
    return IncidentProjection(
        value["incident_id"],
        value["edge_event_id"],
        value["detected_at"],
        value["lifecycle_state"],
        delivery_state,
        projection_timestamp,
    )


def _multiplication_defect(rows: Sequence[CorrelationRow]) -> Classification | None:
    backends: dict[str, set[str]] = {}
    incidents: dict[str, set[str]] = {}
    for row in rows:
        if not row.edge_event_id:
            continue
        backends.setdefault(row.edge_event_id, set()).update(row.backend_event_ids)
        incidents.setdefault(row.edge_event_id, set()).update(row.incident_ids)
    if any(len(values) > 1 for values in backends.values()):
        return Classification(
            DiagnosticOutcome.BACKEND_IDENTITY_DUPLICATION,
            "판정 불가",
            "one edge_event_id mapped to multiple accepted backend event IDs",
        )
    for edge_id, incident_ids in incidents.items():
        if len(incident_ids) <= 1:
            continue
        # The oracle is one E and ONE stable B mapping to many I. Without a
        # single accepted backend identity the multiplication cannot be
        # attributed to the API projection layer.
        if len(backends.get(edge_id, set())) != 1:
            return _inconclusive(
                "API incident multiplication requires exactly one accepted backend identity"
            )
        return Classification(
            DiagnosticOutcome.API_INCIDENT_DUPLICATION,
            "판정 불가",
            "one edge/backend identity mapped to multiple API incident IDs",
        )
    return None


def _row_has_identities(row: CorrelationRow) -> bool:
    return (
        bool(row.transition_id)
        and bool(row.edge_event_id)
        and bool(row.attempt_ordinals)
        and bool(row.backend_event_ids)
        and bool(row.incident_ids)
    )


def _valid_clock_sample(sample: ClockSample) -> bool:
    values = (sample.offset_ms, sample.uncertainty_ms, sample.rtt_ms)
    return (
        all(math.isfinite(value) for value in values)
        and sample.uncertainty_ms >= 0
        and sample.rtt_ms >= 0
    )


def _expected_route_status(method: str, path: str) -> set[int]:
    exact: dict[tuple[str, str], set[int]] = {
        ("POST", "/api/v1/edge/enrollments/verify"): {200},
        ("GET", "/api/v1/events/capabilities"): {200},
        ("POST", "/api/v1/events"): {201, 409},
        ("POST", "/api/v1/events/heartbeat"): {200},
        ("POST", "/v1/events"): {404},
    }
    if (method, path) in exact:
        return exact[(method, path)]
    patterns: tuple[tuple[str, re.Pattern[str], set[int]], ...] = (
        ("GET", re.compile(r"^/api/v1/ml-config/[^/]+$"), {200, 404}),
        ("GET", re.compile(r"^/v1/ml-config/[^/]+$"), {404}),
        ("PUT", re.compile(r"^/api/v1/events/[0-9a-f-]+/snapshot$"), {201}),
        ("PUT", re.compile(r"^/api/v1/edge/topology-snapshots/[0-9a-f-]+$"), {200}),
        (
            "POST",
            re.compile(r"^/api/v1/edge/topology-snapshots/[0-9a-f-]+/confirm$"),
            {200},
        ),
    )
    for expected_method, pattern, statuses in patterns:
        if method == expected_method and pattern.fullmatch(path):
            return statuses
    return set()


def _inconclusive(reason: str) -> Classification:
    return Classification(DiagnosticOutcome.INCONCLUSIVE, "판정 불가", reason)


def _open_references_for_path(mount_path: Path) -> tuple[str, ...]:
    """Descriptor references to ``device`` held by this principal's processes.

    A run-owned tmpfs is mode 0700 owned by this uid, so only this uid and root
    can open it. This uid's own processes are enumerable through the normal
    ``/proc`` the host already provides, and they are the only principal this
    run can be responsible for. Two classes are deliberately out of scope
    rather than gate failures, because neither can be changed by an operator
    and neither can hold a descriptor this run handed out: processes owned by a
    different uid, and this uid's own non-dumpable helpers whose ``fd``
    directory the kernel reassigns to root (for example PAM's ``(sd-pam)``).
    ``descriptor_scan_scope`` reports the residual skipped count so the
    assumption stays explicit instead of silent.
    """
    own_uid = os.getuid()
    prefix = f"{mount_path}/"
    references: list[str] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != own_uid:
                continue
        except (FileNotFoundError, PermissionError):
            continue
        descriptors = process / "fd"
        try:
            entries = tuple(descriptors.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in entries:
            # readlink resolves the descriptor's own target name and, unlike
            # stat, never requires permission on the target itself.
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError):
                continue
            target = target.removesuffix(" (deleted)")
            if target == str(mount_path) or target.startswith(prefix):
                references.append(f"{process.name}/{descriptor.name}")
    return tuple(references)


def _mount_for(path: Path) -> tuple[Path, str, set[str]] | None:
    selected: tuple[Path, str, set[str]] | None = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        mount_point = Path(left_fields[4].replace("\\040", " ")).resolve()
        try:
            path.relative_to(mount_point)
        except ValueError:
            continue
        options = set(left_fields[5].split(",")) | set(right_fields[2].split(","))
        candidate = (mount_point, right_fields[0], options)
        if selected is None or len(mount_point.parts) > len(selected[0].parts):
            selected = candidate
    return selected


__all__ = [
    "Classification",
    "ClockGate",
    "DescriptorScanScope",
    "ClockSample",
    "CorrelationRow",
    "DiagnosticOutcome",
    "GatewayAttestation",
    "IncidentProjection",
    "INSECURE_HTTP_ENV",
    "NamespacePolicy",
    "NetworkFlow",
    "RunOwnedTeardown",
    "SessionBudgetExceeded",
    "SessionCounts",
    "SessionLatch",
    "TmpfsEvidence",
    "attest_gateway",
    "build_diagnostic_overlay",
    "classify_rows",
    "collect_stable_incidents",
    "descriptor_scan_scope",
    "evaluate_clock",
    "inspect_tmpfs",
    "probe_procfs_visibility",
    "rows_from_relations",
    "scan_retained_text",
    "validate_insecure_http_assignments",
    "validate_insecure_http_source_declarations",
    "validate_namespace_policy",
    "validate_network_canaries",
    "validate_no_insecure_http_assignments",
    "validate_route_ledger",
    "validate_temporal_order",
    "verify_tmpfs_destroyed",
]

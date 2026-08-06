"""Single source of truth for which domains are wired into the pipeline.

`DOMAIN_REGISTRY` maps a domain name to a `DomainRegistration` describing how
to build its camera-local decider from typed dependencies, which input view
and event types it owns, and how to read its audit/debug metadata without a
second detector pass. Registering a domain here is necessary but not
sufficient for its events to reach output: `WorkerRuntime._build_decider`
(worker/runtime/worker.py) resolves each domain's typed `*DomainDependencies`
object through its own `if`/`elif` chain keyed on domain name, and fails
closed (`raise RuntimeError`) for any registered-but-unhandled name. A new
domain needs both a `DOMAIN_REGISTRY` entry here and a matching branch in
`_build_decider`; see `worker/domains/AGENTS.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any

from worker.domains.base import AuditContext, DomainAuditSnapshot, DomainDetector
from worker.domains.bed_exit import (
    BedExitConfig,
    BedExitDebugSnapshot,
    BedExitMonitor,
    BedExitScoringRecorder,
)
from worker.domains.fall import FallEventLatch, FallModelProtocol


@dataclass(frozen=True, slots=True)
class FallDomainDependencies:
    model: FallModelProtocol
    camera_id: str
    facility_id: str


@dataclass(frozen=True, slots=True)
class BedExitDomainDependencies:
    config: BedExitConfig
    clock: Callable[[], datetime]
    scoring_recorder: BedExitScoringRecorder | None = None


@dataclass(frozen=True, slots=True)
class DomainRegistration:
    domain: str
    input_view: str
    event_types: frozenset[str]
    factory: Callable[[Any], DomainDetector]
    # The observations (extractor module names, e.g. "pose"/"bed") this
    # domain requires from the analytics stage. `WorkerRuntime._build_camera`
    # unions `requires` across every active domain to build the extraction
    # schedule -- an extractor whose name is required by no active domain is
    # never scheduled or loaded (issue #47). No default: every registration
    # must declare what it needs.
    requires: frozenset[str]
    enabled: bool = True
    audit_metadata_provider: Callable[[AuditContext], DomainAuditSnapshot] | None = None
    debug_snapshot_adapter: Callable[[Any, int], Any] | None = None


def _build_fall(dependencies: FallDomainDependencies) -> FallEventLatch:
    return FallEventLatch(
        dependencies.model,
        camera_id=dependencies.camera_id,
        facility_id=dependencies.facility_id,
    )


def _build_bed_exit(dependencies: BedExitDomainDependencies) -> BedExitMonitor:
    return BedExitMonitor(
        config=dependencies.config,
        clock=dependencies.clock,
        scoring_recorder=dependencies.scoring_recorder,
    )


def _audit_snapshot(context: AuditContext) -> DomainAuditSnapshot:
    return DomainAuditSnapshot(
        model_version=context.model_version,
        operating_threshold=context.operating_threshold,
    )


def _bed_exit_debug_snapshot(
    detector: BedExitMonitor,
    frame_index: int,
) -> BedExitDebugSnapshot | None:
    snapshot = detector.last_debug_snapshot
    if snapshot is None:
        return None
    return replace(snapshot, frame_index=frame_index)


DOMAIN_REGISTRY: Mapping[str, DomainRegistration] = MappingProxyType(
    {
        "fall": DomainRegistration(
            domain="fall",
            input_view="fall_window",
            event_types=frozenset({"fall"}),
            factory=_build_fall,
            requires=frozenset({"pose"}),
            audit_metadata_provider=_audit_snapshot,
        ),
        "bed_exit": DomainRegistration(
            domain="bed_exit",
            input_view="bed_regions",
            event_types=frozenset({"bed-exit"}),
            factory=_build_bed_exit,
            requires=frozenset({"pose", "bed"}),
            audit_metadata_provider=_audit_snapshot,
            debug_snapshot_adapter=_bed_exit_debug_snapshot,
        ),
    }
)


def list_domains(*, enabled: bool = True) -> tuple[str, ...]:
    return tuple(
        name for name, registration in DOMAIN_REGISTRY.items() if registration.enabled is enabled
    )


def enabled_domains() -> tuple[str, ...]:
    return list_domains(enabled=True)


__all__ = [
    "DOMAIN_REGISTRY",
    "BedExitDomainDependencies",
    "DomainRegistration",
    "FallDomainDependencies",
    "enabled_domains",
    "list_domains",
]

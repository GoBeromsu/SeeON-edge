"""Worker-side successor of the domain-registration / no-bypass characterization.

Ruling R2 (todo30b-disposition-table.md): the AST no-domain-specific-bypass
check at old edge/runtime/camera_worker.py:233 MUST be ported against the new
composition root, worker/runtime/worker.py. This file cannot be deleted.

Original edge coverage (edge/domains/__init__.py, edge/runtime/edge_worker.py,
edge/runtime/camera_worker.py) and its worker-side disposition, test by test:

1. ``test_registry_only_domain_can_be_enabled_in_yaml_and_composed`` -- PARTIAL.
   Worker's ``DOMAIN_REGISTRY`` (worker/domains/registry.py) is a
   ``types.MappingProxyType``, not a plain mutable dict: it has no
   ``__setitem__`` at all, so edge's ``monkeypatch.setitem(DOMAIN_REGISTRY, ...)``
   technique cannot even be expressed against worker's registry (a stronger
   runtime-immutability guarantee than edge had). The downstream half of the
   guarantee -- that a third ``DomainRegistration``, built with no special
   casing, composes end to end through ``EventAggregator``/``IncidentManager``
   -- is ported below (whole-module monkeypatch of the registry object). The
   upstream half -- routing a config-declared domain name through YAML config
   validation and having ``WorkerRuntime`` build it -- does NOT port as-is:
   see the observation on ``_build_decider`` below.
2. ``test_registry_domain_event_reaches_relay_payload`` -- OUT OF SCOPE for
   this file. It exercises the old worker's ``_RelayClient`` (formerly
   ``edge/runtime/edge_worker.py``), whose worker-side decomposition
   (``HeartbeatReporter`` /
   ``RelayRuntimeStatusTransport`` / ``EvidenceSender``) is Phase B named
   symbol #3, assigned to a separate migration wave. Not ported here to avoid
   dueling ownership of that cluster's tests.
3. ``test_relay_client_rejects_unregistered_event_type`` -- OUT OF SCOPE, same
   ``_RelayClient`` cluster as above.
4. ``test_enabled_domains_rejects_unregistered_name`` -- superseded 1:1 by
   ``tests/test_ml_worker_yaml_config.py::test_ml_worker_yaml_rejects_unknown_domain``
   (line 182) and ``::test_ml_worker_yaml_rejects_removed_domains`` (line 192,
   parametrized over former domain names). Not duplicated here.
5. ``test_enabled_domains_rejects_duplicate_and_variant_names`` -- the
   unknown-name-variant cases are covered by the same two tests cited above.
   The *duplicate*-name case was not: ``worker/runtime/config/domain_models.py``
   has had a working ``"domains.enabled contains duplicate domain: ..."``
   validator (``DomainsConfig._validate_enabled``) with zero test coverage
   (confirmed: no "duplicate" hit anywhere in
   tests/test_ml_worker_yaml_config.py pre-port). Ported below, closing that
   gap.
6. ``test_registry_domain_config_is_passed_to_factory`` -- obsolete by
   architecture change, not ported. Edge's ``DomainRegistration`` carried a
   generic ``config_schema``/``input_preparer`` pair so an arbitrary
   pydantic-validated dict flowed from YAML into a domain's factory. Worker's
   ``DomainRegistration`` (worker/domains/registry.py:36-44) has neither
   field: each domain instead takes a typed dependency dataclass
   (``FallDomainDependencies``, ``BedExitDomainDependencies``) built
   explicitly by ``WorkerRuntime._build_decider``. There is no generic
   dict-config-to-factory passthrough left to characterize.
7. ``test_camera_worker_has_no_domain_specific_bypasses`` -- MANDATORY per
   ruling R2, ported below against worker/runtime/worker.py with a forbidden
   set re-derived for worker's architecture (edge's exact set was specific to
   CameraWorker's internals; see the test body for the re-derivation
   rationale).

Observation for the composition-root owner (not a bug, not patched --
production code is out of scope for this migration): ``WorkerRuntime.
_build_decider`` (worker/runtime/worker.py) dispatches "fall"/"bed_exit"
through a closed ``if`` / ``elif`` chain to build each domain's typed
dependency object before calling ``registration.factory(dependencies)``
generically. A domain newly added to ``DOMAIN_REGISTRY`` is therefore not
automatically composable end to end through ``WorkerRuntime`` the way test 1
above characterizes at the registry level -- it also needs a new ``elif``
branch and a new typed dependencies dataclass. This is ordinary heterogeneous
dependency injection, not a business-logic bypass (the chain never does
anything but build a dependency object and hand off to the registry's own
factory -- exactly what test 7 below guards against), so it does not violate
R2. It does mean the registry module's docstring claim ("Registering a domain
here is the only step needed for its events to reach output") is true only up
to dependency construction, worth flagging for whoever owns that comment.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest
from pydantic import ValidationError

import worker.domains.registry as registry_module
from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.domains.registry import DomainRegistration
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.runtime.config import WorkerConfig
from worker.types import BusinessEvent, DecisionInput


def _worker_config_payload(**domains: object) -> dict[str, object]:
    return {
        "relay": {"url": "http://relay.test", "token": "relay-token"},
        "domains": domains,
        "cameras": [
            {"camera_id": "camera-1", "facility_id": "facility-1", "rtsp_url": "rtsp://example.test/camera-1"}
        ],
    }


def test_worker_config_rejects_duplicate_enabled_domain_names() -> None:
    with pytest.raises(ValidationError, match="domains.enabled contains duplicate domain: fall"):
        WorkerConfig.model_validate(_worker_config_payload(enabled=["fall", "fall"]))


def test_domain_registry_extension_composes_through_event_aggregator_generically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A third DomainRegistration, with no special-casing anywhere in
    EventAggregator/IncidentManager, is admitted end to end. This proves the
    downstream composition machinery is genuinely registry-shape-driven and
    not hardcoded to "fall"/"bed_exit" by name -- the guarantee ruling R2
    exists to protect, exercised at the layer that is actually generic.
    """
    seen: list[DecisionInput] = []

    @dataclass(slots=True)
    class _DummyDetector:
        def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
            seen.append(input_value)
            return (
                BusinessEvent(
                    domain="dummy",
                    event_type="dummy-alert",
                    identity=1,
                    camera_id="camera-1",
                    facility_id="facility-1",
                    time_sec=input_value.time_sec,
                    probability=1.0,
                ),
            )

    dummy_registration = DomainRegistration(
        domain="dummy",
        input_view="dummy-view",
        event_types=frozenset({"dummy-alert"}),
        factory=lambda _dependencies: _DummyDetector(),
        requires=frozenset(),
    )
    extended = dict(registry_module.DOMAIN_REGISTRY)
    extended["dummy"] = dummy_registration
    monkeypatch.setattr(registry_module, "DOMAIN_REGISTRY", MappingProxyType(extended))

    detector = registry_module.DOMAIN_REGISTRY["dummy"].factory(None)
    aggregator = EventAggregator(
        deciders=(detector,),
        incidents=IncidentManager(identity_path=tmp_path / "identities.jsonl"),
    )
    frame = DecisionInput(
        observation=FrameObservation(),
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=1.0,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )

    admitted = aggregator.update(frame)

    assert len(seen) == 1
    assert seen[0] is frame
    assert len(admitted) == 1
    assert admitted[0].domain == "dummy"
    assert admitted[0].event_type == "dummy-alert"
    assert "dummy" in registry_module.list_domains()
    assert "dummy" in registry_module.enabled_domains()


def test_camera_worker_has_no_domain_specific_bypasses() -> None:
    """Re-derived forbidden set for worker/runtime/worker.py.

    Edge's forbidden set ({"fall_classifier", "observation_enricher"} attrs,
    {"resolve_bed_regions"} calls, {"BedExitDebugSnapshot"} names,
    {"fall", "bed-exit"} literals) was specific to CameraWorker's internals,
    where per-frame decision logic and observation mutation lived in the same
    class as the composition wiring -- so *any* direct reference to a
    domain's internals from that class was suspect.

    Worker splits that concern structurally: WorkerRuntime only ever builds
    each domain's typed Dependencies dataclass and calls
    ``registration.factory(dependencies)`` -- confirmed by grep that
    worker/runtime/worker.py never imports or names the concrete decider
    classes (FallEventLatch, BedExitMonitor) or reaches into their internals
    (classifier, _previous_fall, last_debug_snapshot) directly; it only knows
    about the registry, the two dependency dataclasses, and the Decider
    protocol. That is the meaningful analog of R2's "no domain-specific
    bypass" guarantee for this architecture: the composition root must never
    construct a decider directly or read its private state, only go through
    DOMAIN_REGISTRY[name].factory(...) and the registry's own
    audit_metadata_provider/debug_snapshot_adapter seams.

    "fall"/"bed_exit" string literals are NOT forbidden here (unlike edge):
    they appear legitimately in _build_decider purely to select which typed
    Dependencies dataclass to construct, never to special-case business
    output -- see the module docstring's observation above.
    """
    source = Path("worker/runtime/worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_names = {"FallEventLatch", "BedExitMonitor"}
    forbidden_attributes = {
        "classifier",
        "_previous_fall",
        "last_debug_snapshot",
        "_assignments",
        "_latch",
    }

    name_hits = [
        node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in forbidden_names
    ]
    attr_hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
    ]

    assert not name_hits, [ast.dump(node) for node in name_hits]
    assert not attr_hits, [ast.dump(node) for node in attr_hits]

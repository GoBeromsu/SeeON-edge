"""The compiled-artifact identity gate must stay fail-closed.

`worker/domains/registry.py::_COMPONENT_ARTIFACT_DIGESTS` pins one sha256 per
component and `worker/runtime/model_composition.py` refuses activation when a
provisioned runner reports a different identity. `tests/test_runtime_manifest.py`
only asserts manifest consistency, so it cannot prove that refusal. These tests
exercise the gate itself: correct identities compose, and any tampered,
missing, or placeholder identity raises before a camera graph can be built.
"""

from __future__ import annotations

from typing import final

import pytest

from contracts.runner import Image, RunnerResult, pose_result
from worker.domains.module_definition import (
    ComponentBinding,
    DetectionModuleActivationError,
    RuntimeResolvedArtifactDigest,
)
from worker.domains.registry import DETECTION_MODULE_REGISTRY
from worker.runtime.model_composition import (
    SharedComponentPool,
    compose_shared_components,
)

ServingOption = object
_SELECTION = {"fall": 2, "bed_exit": 1}


@final
class _IdentityRunner:
    """Reports whatever identity the test asks it to claim."""

    def __init__(self, artifact_digest: str, preprocessing_identity: str) -> None:
        self.artifact_digest = artifact_digest
        self.preprocessing_identity = preprocessing_identity

    def __call__(self, _image: Image) -> RunnerResult:
        return pose_result((), ())

    def warmup(self) -> None:
        return None


def _bindings_by_task() -> dict[str, ComponentBinding]:
    bindings = DETECTION_MODULE_REGISTRY.shared_bindings(_SELECTION, flags={})
    by_task: dict[str, ComponentBinding] = {}
    for binding in bindings:
        task = getattr(binding, "task", None) or binding.component_id
        by_task[str(task)] = binding
    return by_task


def _serving(tamper: dict[str, str] | None = None) -> object:
    """Serving client whose runners claim compiled identities, optionally tampered."""

    by_task = _bindings_by_task()
    overrides = tamper or {}

    @final
    class _Serving:
        def create(self, task: str, **_options: object) -> _IdentityRunner:
            binding = by_task[task]
            digest = binding.artifact_digest
            preprocessing = binding.preprocessing_identity
            assert isinstance(digest, str)
            assert isinstance(preprocessing, str)
            component_id = str(binding.component_id)
            return _IdentityRunner(overrides.get(component_id, digest), preprocessing)

    return _Serving()


def _compose(serving: object, tamper: dict[str, str] | None = None) -> object:
    overrides = tamper or {}

    def _fall_provisioner(binding: ComponentBinding, _device: object) -> _IdentityRunner:
        digest = binding.artifact_digest
        preprocessing = binding.preprocessing_identity
        if isinstance(digest, RuntimeResolvedArtifactDigest):
            digest = "f" * 64
        assert isinstance(digest, str)
        assert isinstance(preprocessing, str)
        component_id = str(binding.component_id)
        return _IdentityRunner(overrides.get(component_id, digest), preprocessing)

    return compose_shared_components(
        DETECTION_MODULE_REGISTRY,
        module_versions=_SELECTION,
        serving_client=serving,
        runtime="cpu",
        device="cpu",
        flags={},
        pool=SharedComponentPool(),
        provisioners={"fall-model-family-registry": _fall_provisioner},
    )


def test_compiled_identities_compose_without_error() -> None:
    graph = _compose(_serving())

    identities = {identity.component_id: identity.artifact_digest for identity in graph.identities}
    assert identities, "composition produced no component identities"
    assert all(isinstance(value, str) and value for value in identities.values())


@pytest.mark.parametrize("component_id", ["person", "bed", "pose"])
def test_tampered_component_digest_is_refused(component_id: str) -> None:
    """A single wrong digest must block activation for every pinned component."""

    by_task = _bindings_by_task()
    pinned = {
        str(b.component_id): b.artifact_digest
        for b in by_task.values()
        if isinstance(b.artifact_digest, str)
    }
    if component_id not in pinned:
        pytest.skip(f"component {component_id} is not part of the current selection")

    tampered = "0" * 64
    assert tampered != pinned[component_id]

    with pytest.raises(DetectionModuleActivationError, match="identity mismatch"):
        _compose(_serving({component_id: tampered}), {component_id: tampered})


@pytest.mark.parametrize("bogus", ["", "runtime-resolved", "runtime-resolved:person"])
def test_missing_or_placeholder_identity_is_refused(bogus: str) -> None:
    """The gate must not accept an empty or placeholder identity as a pass."""

    by_task = _bindings_by_task()
    target = next(
        (
            str(b.component_id)
            for b in by_task.values()
            if isinstance(b.artifact_digest, str) and b.artifact_digest
        ),
        None,
    )
    assert target is not None, "no pinned component available to tamper"

    with pytest.raises(DetectionModuleActivationError):
        _compose(_serving({target: bogus}), {target: bogus})


def test_every_selected_component_declares_its_identity_source() -> None:
    """Every component is pinned or explicitly resolved from its verified bundle."""

    for binding in _bindings_by_task().values():
        digest = binding.artifact_digest
        assert (
            isinstance(digest, RuntimeResolvedArtifactDigest)
            or (isinstance(digest, str) and len(digest) == 64)
        ), (
            f"component {binding.component_id!r} has no declared artifact identity source"
        )

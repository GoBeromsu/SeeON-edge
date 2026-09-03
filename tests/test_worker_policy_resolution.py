from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from shared.detection_policies import (
    BED_EXIT_POLICY_V1_DEFAULT,
    FALL_POLICY_V2_DEFAULT,
    PolicyDocumentError,
    default_policy_bundle,
    make_effective_policy,
)
from worker.domains.bed_exit import BedExitMonitor
from worker.domains.fall import FallV2DomainDecider
from worker.interfaces.fall_model import FallV2Probabilities
from worker.runtime.config.config_pull import ConfigSource, load_worker_config_from_relay
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.lkg_store import JsonObject, WorkerConfigLkgStore
from worker.runtime.config.pull_models import BackendWorkerConfigPayload
from worker.runtime.config.restart import RestartDirective


@dataclass(frozen=True, slots=True)
class _Metadata:
    window: int = 1
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


class _FallModel:
    def predict(self, features: NDArray[np.float32]) -> FallV2Probabilities:
        del features
        return FallV2Probabilities(background=0.3, fall_transition=0.7, fallen=0.0)


class _Response:
    status = 200

    def __init__(self, payload: JsonObject) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback

    def read(self) -> bytes:
        return self._body


def _payload(
    *,
    policy_wire: dict[str, object],
    camera_id: str = "cam/opaque:alpha",
) -> JsonObject:
    return cast(
        JsonObject,
        {
            "registry_version": 1,
            "config_version": 17,
            "restart_epoch": 3,
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": "facility:opaque",
                    "rtsp_url": "rtsp://camera.invalid/stream",
                }
            ],
            "detection_policies": policy_wire,
        },
    )


def test_worker_parses_effective_camera_override_and_keeps_modules_independent() -> None:
    bundle = default_policy_bundle(("cam/opaque:alpha",))
    camera_policies = dict(bundle.cameras["cam/opaque:alpha"])
    camera_policies["fall"] = make_effective_policy(
        module_id="fall",
        module_version=2,
        values=type(FALL_POLICY_V2_DEFAULT)(transition_threshold=0.73),
        source="camera-override",
        facility_revision_id=4,
        camera_revision_id=9,
    )
    overridden = bundle.with_camera("cam/opaque:alpha", camera_policies)

    config = BackendWorkerConfigPayload.model_validate(
        _payload(policy_wire=overridden.as_dict())
    ).to_worker_config("http://relay.invalid", "relay-token")

    assert config.version == 17
    assert config.detection_policies.resolve("cam/opaque:alpha", "fall", 2).values == type(
        FALL_POLICY_V2_DEFAULT
    )(transition_threshold=0.73)
    assert (
        config.detection_policies.resolve("cam/opaque:alpha", "bed_exit", 1).values
        == BED_EXIT_POLICY_V1_DEFAULT
    )


def test_camera_policy_override_uses_the_byte_exact_pulled_camera_identity() -> None:
    camera_id = " \u2007camera/\u1100\u1161/e\u0301 "
    bundle = default_policy_bundle((camera_id,))
    camera_policies = dict(bundle.cameras[camera_id])
    camera_policies["fall"] = make_effective_policy(
        module_id="fall",
        module_version=2,
        values=type(FALL_POLICY_V2_DEFAULT)(transition_threshold=0.81),
        source="camera-override",
        facility_revision_id=4,
        camera_revision_id=10,
    )
    wire = bundle.with_camera(camera_id, camera_policies).as_dict()

    config = BackendWorkerConfigPayload.model_validate(
        _payload(policy_wire=wire, camera_id=camera_id)
    ).to_worker_config("http://relay.invalid", "relay-token")

    parsed_id = config.cameras[0].camera_id
    assert parsed_id.encode("utf-8") == camera_id.encode("utf-8")
    assert config.detection_policies.resolve(parsed_id, "fall", 2).values == type(
        FALL_POLICY_V2_DEFAULT
    )(transition_threshold=0.81)
    assert config.detection_policies.resolve(camera_id.strip(), "fall", 2).values == (
        FALL_POLICY_V2_DEFAULT
    )


def test_runtime_camera_module_receives_policy_not_model_or_profile_threshold() -> None:
    bundle = default_policy_bundle(("cam/opaque:alpha",))
    camera_policies = dict(bundle.cameras["cam/opaque:alpha"])
    camera_policies["fall"] = make_effective_policy(
        module_id="fall",
        module_version=2,
        values=type(FALL_POLICY_V2_DEFAULT)(transition_threshold=0.73),
        source="camera-override",
        facility_revision_id=4,
        camera_revision_id=9,
    )
    camera_policies["bed_exit"] = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=type(BED_EXIT_POLICY_V1_DEFAULT)(
            min_containment=0.44,
            hold_frames=5,
            grace_frames=8,
        ),
        source="facility-default",
        facility_revision_id=10,
        camera_revision_id=None,
    )
    config = BackendWorkerConfigPayload.model_validate(
        _payload(policy_wire=bundle.with_camera("cam/opaque:alpha", camera_policies).as_dict())
    ).to_worker_config("http://relay.invalid", "relay-token")

    # The production module factories are exercised through the runtime's
    # registered definitions; profile/device never participates in resolution.
    from worker.domains import DETECTION_MODULE_REGISTRY, CameraModuleContext

    fall_definition = DETECTION_MODULE_REGISTRY.get("fall", 2)
    fall_module = fall_definition.create_camera_module(
        CameraModuleContext(
            camera_id="cam/opaque:alpha",
            facility_id="facility:opaque",
            shared_components={"fall-classifier": _FallModel()},
            camera_components={"episode-identity": ("boot", "1", 0)},
            detection_window=None,
            clock=lambda: pytest.fail("fall clock should not be called"),
            diagnostics=None,
            policy=config.detection_policies.resolve("cam/opaque:alpha", "fall", 2),
        )
    )
    bed_definition = DETECTION_MODULE_REGISTRY.get("bed_exit", 1)
    bed_module = bed_definition.create_camera_module(
        CameraModuleContext(
            camera_id="cam/opaque:alpha",
            facility_id="facility:opaque",
            shared_components={},
            camera_components={"episode-identity": ("boot", "1", 0)},
            detection_window=None,
            clock=lambda: pytest.fail("bed clock should not be called during construction"),
            diagnostics=None,
            policy=config.detection_policies.resolve("cam/opaque:alpha", "bed_exit", 1),
        )
    )

    assert isinstance(fall_module.decider, FallV2DomainDecider)
    assert fall_module.decider.policy.policy.transition_threshold == 0.73
    assert isinstance(bed_module.decider, BedExitMonitor)
    assert bed_module.decider.config.min_containment == 0.44
    assert bed_module.decider.config.hold_frames == 5
    assert bed_module.decider.config.grace_frames == 8


@pytest.mark.parametrize(
    "mutation",
    [
        lambda wire: wire.update({"unknown": 1}),
        lambda wire: wire["defaults"]["fall"].update({"schema_version": 99}),
        lambda wire: wire["defaults"]["fall"]["values"].update({"extra": 1}),
    ],
)
def test_worker_policy_boundary_fails_closed_on_unknown_or_drift(mutation) -> None:
    wire = default_policy_bundle(("cam/opaque:alpha",)).as_dict()
    mutation(wire)

    with pytest.raises(
        (PolicyDocumentError, WorkerConfigError, ValueError),
        match="unknown|schema|policy",
    ):
        BackendWorkerConfigPayload.model_validate(_payload(policy_wire=wire)).to_worker_config(
            "http://relay.invalid", "relay-token"
        )


def test_malformed_fresh_policy_retains_previous_valid_lkg(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = WorkerConfigLkgStore(state_dir=tmp_path)
    valid = _payload(policy_wire=default_policy_bundle(("cam/opaque:alpha",)).as_dict())
    assert store.save(valid, RestartDirective(generation=3, version=17))
    malformed = _payload(policy_wire=default_policy_bundle(("cam/opaque:alpha",)).as_dict())
    malformed["config_version"] = 18
    policy_wire = cast(dict[str, object], malformed["detection_policies"])
    defaults = cast(dict[str, object], policy_wire["defaults"])
    fall = cast(dict[str, object], defaults["fall"])
    fall["schema_version"] = 99

    snapshot = load_worker_config_from_relay(
        "http://relay.invalid",
        "relay-token",
        store=store,
        urlopen=lambda _request, _timeout: _Response(malformed),
    )

    assert snapshot is not None
    assert snapshot.source is ConfigSource.LKG
    assert snapshot.directive == RestartDirective(generation=3, version=17)
    assert "detection policy refused" in capsys.readouterr().err

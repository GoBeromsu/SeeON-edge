from __future__ import annotations

import math

import pytest

from shared.detection_policies import (
    BED_EXIT_POLICY_V1_DEFAULT,
    FALL_POLICY_V1_DEFAULT,
    BedExitPolicyV1,
    FallPolicyV1,
    PolicyDocumentError,
    make_effective_policy,
    parse_effective_policy,
    parse_policy_values,
)


def test_versioned_policy_parser_returns_typed_numeric_values() -> None:
    fall = parse_policy_values(
        module_id="fall",
        module_version=1,
        schema_id="fall.policy",
        schema_version=1,
        values={"operating_threshold": 0.73},
    )
    bed_exit = parse_policy_values(
        module_id="bed_exit",
        module_version=1,
        schema_id="bed_exit.policy",
        schema_version=1,
        values={"min_containment": 0.42, "hold_frames": 4, "grace_frames": 7},
    )

    assert fall == FallPolicyV1(operating_threshold=0.73)
    assert bed_exit == BedExitPolicyV1(
        min_containment=0.42,
        hold_frames=4,
        grace_frames=7,
    )
    assert FALL_POLICY_V1_DEFAULT == FallPolicyV1(operating_threshold=0.5)
    assert BED_EXIT_POLICY_V1_DEFAULT == BedExitPolicyV1(
        min_containment=0.35,
        hold_frames=2,
        grace_frames=3,
    )


@pytest.mark.parametrize(
    ("module_id", "module_version", "schema_id", "schema_version", "values", "message"),
    [
        ("fall", 2, "fall.policy", 1, {"operating_threshold": 0.5}, "module version"),
        ("fall", 1, "fall.policy", 2, {"operating_threshold": 0.5}, "schema"),
        ("fall", 1, "fall.policy", 1, {"operating_threshold": 0.5, "extra": 1}, "unknown"),
        ("fall", 1, "fall.policy", 1, {"operating_threshold": True}, "numeric"),
        ("fall", 1, "fall.policy", 1, {"operating_threshold": math.inf}, "finite"),
        (
            "bed_exit",
            1,
            "bed_exit.policy",
            1,
            {"min_containment": 0.0, "hold_frames": 2, "grace_frames": 3},
            "min_containment",
        ),
        (
            "bed_exit",
            1,
            "bed_exit.policy",
            1,
            {"min_containment": 0.4, "hold_frames": 200, "grace_frames": 101},
            "combined",
        ),
        (
            "bed_exit",
            1,
            "bed_exit.policy",
            1,
            {"min_containment": 0.4, "hold_frames": 2.0, "grace_frames": 3},
            "integer",
        ),
    ],
)
def test_policy_parser_rejects_numeric_and_identity_drift(
    module_id: str,
    module_version: int,
    schema_id: str,
    schema_version: int,
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PolicyDocumentError, match=message):
        parse_policy_values(
            module_id=module_id,
            module_version=module_version,
            schema_id=schema_id,
            schema_version=schema_version,
            values=values,
        )


def test_effective_policy_identity_is_deterministic_and_profile_independent() -> None:
    first = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.61),
        source="camera-override",
        facility_revision_id=8,
        camera_revision_id=13,
    )
    cpu_wire = first.as_dict()
    nvidia_wire = first.as_dict()

    assert cpu_wire == nvidia_wire
    assert parse_effective_policy(cpu_wire) == first
    assert len(first.effective_policy_id) == 64

    forged = dict(cpu_wire)
    forged["effective_policy_id"] = "0" * 64
    with pytest.raises(PolicyDocumentError, match="identity"):
        parse_effective_policy(forged)


def test_effective_policy_document_rejects_unknown_fields() -> None:
    document = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BED_EXIT_POLICY_V1_DEFAULT,
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    ).as_dict()
    document["runtime_profile"] = "cpu-host"

    with pytest.raises(PolicyDocumentError, match="unknown"):
        parse_effective_policy(document)

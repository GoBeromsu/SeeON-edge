from __future__ import annotations

from worker.runtime.profile.registry import (
    DEFAULT_PROFILE_NAME,
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    ProfileError,
    requested_profile_name,
    select_profile,
)


def test_ml_worker_profile_env_name() -> None:
    assert ML_WORKER_PROFILE_ENV == "ML_WORKER_PROFILE"


def test_flow_is_the_only_production_profile() -> None:
    assert DEFAULT_PROFILE_NAME == "flow"
    assert tuple(PROFILE_REGISTRY) == ("flow",)
    assert requested_profile_name({}) == "flow"
    assert select_profile({}).spec is PROFILE_REGISTRY["flow"]


def test_retired_profile_refuses_to_start_with_adr_0002_diagnostic() -> None:
    try:
        select_profile({ML_WORKER_PROFILE_ENV: "cpu"})
    except ProfileError as error:
        assert str(error) == "ADR-0002: unsupported ML_WORKER_PROFILE 'cpu'; set flow"
    else:
        raise AssertionError("retired profile must refuse rather than fall back to flow")

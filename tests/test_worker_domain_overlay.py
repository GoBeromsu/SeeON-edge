"""Unit coverage for the domain enable/disable overlay: ``DomainsConfig
.resolved_overrides`` and ``WorkerConfig.enabled_domains``.

Before this overlay, config *replaced* ``DOMAIN_REGISTRY``'s set of active
domains instead of layering on top of it -- an empty/unconfigured signal was
indistinguishable from "everything off" (issue #191's root cause, and the
reason ``worker_models.WorkerConfig`` used to carry a ``model_fields_set``
fail-open marker). Under the overlay, ``DOMAIN_REGISTRY`` is always the floor:
``resolved(name) = overrides.get(name, DOMAIN_REGISTRY[name].enabled)``,
iterating the registry -- never config -- for the set of domains that exist.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from worker.domains.registry import DOMAIN_REGISTRY
from worker.runtime.config.camera_models import RelayConfig
from worker.runtime.config.domain_models import (
    BedExitDomainConfig,
    DomainsConfig,
    FallDomainConfig,
)
from worker.runtime.config.worker_models import WorkerConfig
from worker.runtime.worker import WorkerRuntime, _required_extractor_names


def _relay() -> RelayConfig:
    return RelayConfig.model_validate({"url": "http://relay.test", "token": "relay-token"})


# --- DomainsConfig.resolved_overrides ---------------------------------------


def test_resolved_overrides_is_empty_when_nothing_configured() -> None:
    """No config at all -> an empty overrides map, the unambiguous "defer to
    the registry for every domain" signal."""
    assert DomainsConfig().resolved_overrides() == {}


def test_resolved_overrides_leaves_registry_defaults_intact() -> None:
    """An empty overrides map, resolved through WorkerConfig.enabled_domains,
    yields exactly the registry's own defaults -- not an empty set."""
    config = WorkerConfig(relay=_relay())

    assert config.domains.resolved_overrides() == {}
    assert config.enabled_domains == tuple(
        name for name, registration in DOMAIN_REGISTRY.items() if registration.enabled
    )
    assert config.enabled_domains == ("fall", "bed_exit")


def test_resolved_overrides_explicit_false_disables_one_domain_others_stay_on() -> None:
    """The actual defect this overlay fixes: an override naming only one
    domain must not force every *other* known domain off. The domain it
    never mentions (bed_exit) still resolves through its registry default."""
    domains = DomainsConfig(fall=FallDomainConfig(enabled=False))

    assert domains.resolved_overrides() == {"fall": False}

    config = WorkerConfig(relay=_relay(), domains=domains)
    assert config.enabled_domains == ("bed_exit",)


def test_resolved_overrides_explicit_true_and_false_both_apply() -> None:
    domains = DomainsConfig(
        fall=FallDomainConfig(enabled=True),
        bed_exit=BedExitDomainConfig(enabled=False),
    )

    assert domains.resolved_overrides() == {"fall": True, "bed_exit": False}


def test_resolved_overrides_legacy_list_form_still_works_and_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The deprecated ``enabled`` replace-list -- still the live relay's wire
    shape for per-camera-declared domains -- is still honored, converted
    into a full overlay: listed names on, every other known domain off. It
    logs a deprecation warning rather than failing or staying silent."""
    domains = DomainsConfig(enabled=("fall",))

    overrides = domains.resolved_overrides()

    assert overrides == {"fall": True, "bed_exit": False}
    assert "deprecated" in capsys.readouterr().err.lower()


def test_resolved_overrides_legacy_empty_list_disables_every_known_domain(
    capsys: pytest.CaptureFixture[str],
) -> None:
    domains = DomainsConfig(enabled=())

    overrides = domains.resolved_overrides()

    assert overrides == {"fall": False, "bed_exit": False}
    assert capsys.readouterr().err


# --- WorkerConfig.enabled_domains: the boot floor ---------------------------


def test_enabled_domains_boot_floor_with_zero_config_activates_registry_defaults() -> None:
    """The user's central ask: with the relay unreachable and zero domain
    config, the worker must still resolve fall and bed_exit active."""
    config = WorkerConfig(relay=_relay())

    assert config.enabled_domains == ("fall", "bed_exit")


def test_enabled_domains_boot_floor_schedules_pose_and_bed_extractors() -> None:
    """The production consequence of the boot floor: with zero domain
    config, the runtime must derive both extractors ``DOMAIN_REGISTRY``
    requires for fall (``pose``) and bed_exit (``pose``, ``bed``), not
    silently schedule nothing."""
    config = WorkerConfig(relay=_relay())
    runtime = SimpleNamespace(
        config=config,
        _module_versions=config.selected_module_versions,
    )

    domain_names = WorkerRuntime._active_domain_names(runtime)
    required = _required_extractor_names(domain_names)

    assert domain_names == ("fall", "bed_exit")
    assert set(required) >= {"pose", "bed"}


def test_enabled_domains_never_returns_none() -> None:
    """There is no fail-open sentinel left to check for: every WorkerConfig,
    however constructed, resolves to a concrete tuple."""
    assert WorkerConfig(relay=_relay()).enabled_domains is not None

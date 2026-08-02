"""Unit coverage for ``DomainsConfig.resolved_detection_window`` (issue #24).

``detection_windows`` is deliberately lenient (unlike ``enabled``): unknown
domain names are accepted rather than rejected, since it must stay
forward-compatible with domains a given worker build doesn't know about yet.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from worker.runtime.config.domain_models import (
    BedExitDomainConfig,
    DomainsConfig,
    NightWindowConfig,
)


def test_resolved_detection_window_returns_none_when_nothing_configured() -> None:
    assert DomainsConfig().resolved_detection_window("bed_exit") is None
    assert DomainsConfig().resolved_detection_window("fall") is None


def test_resolved_detection_window_falls_back_to_bed_exit_alias() -> None:
    window = NightWindowConfig(start="21:00", end="06:00", tz="UTC")
    domains = DomainsConfig(bed_exit=BedExitDomainConfig(night_window=window))

    assert domains.resolved_detection_window("bed_exit") == window


def test_resolved_detection_window_explicit_entry_wins_over_bed_exit_alias() -> None:
    alias_window = NightWindowConfig(start="21:00", end="06:00", tz="UTC")
    explicit_window = NightWindowConfig(start="22:00", end="05:00", tz="Asia/Seoul")
    domains = DomainsConfig(
        bed_exit=BedExitDomainConfig(night_window=alias_window),
        detection_windows={"bed_exit": explicit_window},
    )

    assert domains.resolved_detection_window("bed_exit") == explicit_window


def test_resolved_detection_window_explicit_none_entry_clears_the_alias() -> None:
    alias_window = NightWindowConfig(start="21:00", end="06:00", tz="UTC")
    domains = DomainsConfig(
        bed_exit=BedExitDomainConfig(night_window=alias_window),
        detection_windows={"bed_exit": None},
    )

    assert domains.resolved_detection_window("bed_exit") is None


def test_resolved_detection_window_supports_domains_with_no_legacy_alias() -> None:
    window = NightWindowConfig(start="22:00", end="05:00", tz="UTC")
    domains = DomainsConfig(detection_windows={"fall": window})

    assert domains.resolved_detection_window("fall") == window


def test_resolved_detection_window_accepts_unknown_domain_names() -> None:
    """Forward-compatible: an unrecognized domain name in detection_windows
    is accepted rather than rejected (unlike ``enabled``)."""
    window = NightWindowConfig(start="22:00", end="05:00", tz="UTC")
    domains = DomainsConfig(detection_windows={"wander": window})

    assert domains.resolved_detection_window("wander") == window


def test_night_window_config_rejects_equal_start_and_end() -> None:
    """start == end would make DetectionWindow.contains's
    ``start <= t < end`` range match nothing, permanently disabling the
    domain. Unlike the fail-open pulled-config path, locally-authored YAML
    config raises loudly (crashes boot) instead of silently degrading to
    24/7, since a bad YAML value is under direct operator control."""
    with pytest.raises(ValidationError, match="must not be equal"):
        NightWindowConfig(start="09:00", end="09:00", tz="UTC")

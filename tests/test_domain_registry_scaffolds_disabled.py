from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from worker.domains import DOMAIN_REGISTRY, BedExitDomainDependencies
from worker.domains.bed_exit import BedExitConfig
from worker.interfaces.decision import Decider
from worker.interfaces.fall_model import FallV2Probabilities

# Membership guarantee (DOMAIN_REGISTRY == {"fall", "bed_exit"}, no disabled
# scaffolds via list_domains(enabled=False)) is superseded 1:1 by
# tests/test_worker_domain_registry.py:65-76
# (test_registry_contains_only_enabled_fall_and_bed_exit_domains); not
# duplicated here.
#
# The per-detector `.enabled` half of the old
# test_enabled_domain_factories_are_active is impossible to port as-is:
# worker.domains.bed_exit.detector.BedExitMonitor carries no `.enabled`
# attribute at all (worker/domains/bed_exit/detector.py has no such field).
# Confirmed intentional, not a gap:
#   - worker.interfaces.decision.Decider (worker/interfaces/decision.py) is a
#     minimal structural Protocol requiring only `update()`; it has no
#     `enabled` member.
#   - worker.domains.base.DomainDetector (worker/domains/base.py:8) is a
#     plain `TypeAlias` for `Decider`, not the edge-side
#     `DomainDetector(ABC)` that declared `enabled: bool` as an abstract
#     attribute (edge/domains/base.py:33-35).
#   - worker.domains.registry.DomainRegistration.enabled
#     (worker/domains/registry.py:42) is a static dataclass default, never
#     read from a built detector instance; `list_domains`/`enabled_domains`
#     (worker/domains/registry.py:97-104) filter on it directly.
#   - worker.domains.fall.detector.FallEventLatch keeps a vestigial
#     `enabled: ClassVar[bool] = True` (worker/domains/fall/detector.py:15)
#     that no worker code reads (`rg -n "\.enabled\b" worker/` turns up no
#     call site reading a detector's `.enabled`); BedExitMonitor was never
#     given the equivalent attribute. That asymmetry is the tell that
#     enablement fully moved to the registration and nothing depends on the
#     per-instance flag anymore.
# The guarantee this file exists to protect -- "no disabled domain scaffold
# is reachable" -- now lives entirely at the registration level, which the
# test below exercises against real, factory-built detectors.


class _FallModel:
    def predict(self, window: object) -> FallV2Probabilities:
        del window
        return FallV2Probabilities(background=1.0, fall_transition=0.0, fallen=0.0)


def test_domain_detectors_no_longer_require_an_enabled_attribute() -> None:
    fall = DOMAIN_REGISTRY["fall"].factory(
        {
            "model": _FallModel(),
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "boot_id": "boot-1",
            "stream_epoch": "1",
            "source_generation": 0,
        }
    )
    bed_exit = DOMAIN_REGISTRY["bed_exit"].factory(
        BedExitDomainDependencies(
            config=BedExitConfig(camera_id="camera-1", facility_id="facility-1"),
            clock=lambda: datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")),
            boot_id="boot-1",
            stream_epoch="1",
            source_generation=0,
        )
    )

    # Both satisfy the (enabled-attribute-free) Decider protocol structurally.
    assert isinstance(fall, Decider)
    assert isinstance(bed_exit, Decider)

    # bed_exit's detector has no per-instance enabled flag to check -- the
    # registration is the sole authority.
    assert not hasattr(bed_exit, "enabled")

    for name in ("fall", "bed_exit"):
        assert DOMAIN_REGISTRY[name].enabled is True

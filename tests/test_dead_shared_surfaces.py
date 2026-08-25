"""Reachability, export, package, and image contracts for retired shared event surfaces.

The shared in-memory ``Outbox`` and its logging ``EventPublisher`` were
production-unreachable: nothing in ``backend``, ``worker``, ``shared``,
``contracts``, or packaged operator scripts constructed them. They shipped only
to satisfy a test. These contracts prove they are gone and stay gone, while the
active ``DeliveryQueue`` and ``build_audit_envelope`` seams remain.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEAD_RELATIVE_PATHS = (
    "shared/events/outbox.py",
    "shared/events/local_publisher.py",
)
DEAD_MODULES = (
    "shared.events.outbox",
    "shared.events.local_publisher",
)
DEAD_EXPORTS = (
    "Outbox",
    "EventPublisher",
    "LoggingEventPublisher",
    "StubEventPublisher",
)


def test_dead_shared_event_modules_are_absent_from_the_package() -> None:
    for relative in DEAD_RELATIVE_PATHS:
        assert not (ROOT / relative).exists(), relative
    for module_name in DEAD_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_shared_events_package_no_longer_re_exports_dead_publisher_or_outbox() -> None:
    package = importlib.import_module("shared.events")
    for name in DEAD_EXPORTS:
        assert name not in getattr(package, "__all__", ()), name
        with pytest.raises(AttributeError):
            getattr(package, name)


def test_shared_image_copy_tree_cannot_ship_dead_shared_modules() -> None:
    for dockerfile_name in ("Dockerfile.backend", "Dockerfile.edge"):
        dockerfile = (ROOT / dockerfile_name).read_text(encoding="utf-8")
        assert "COPY shared ./shared" in dockerfile, dockerfile_name
    present = [relative for relative in DEAD_RELATIVE_PATHS if (ROOT / relative).exists()]
    assert present == []


def test_active_shared_event_seams_remain_importable() -> None:
    delivery = importlib.import_module("shared.events.delivery_queue")
    assert hasattr(delivery, "DeliveryQueue")
    schemas = importlib.import_module("shared.events.schemas")
    assert hasattr(schemas, "build_audit_envelope")

from __future__ import annotations

from collections.abc import Mapping

from contracts.worker_config import PulledNightWindow, PulledWorkerConfig
from worker.runtime.config.domain_models import KNOWN_DOMAIN_NAMES, NightWindowConfig
from worker.runtime.config.worker_models import WorkerConfig


def resolve_runtime_config(
    yaml_config: WorkerConfig,
    pulled: PulledWorkerConfig | None,
) -> Mapping[str, NightWindowConfig | None]:
    """Resolve each known domain's detection window.

    Covers every domain the worker build knows about (``KNOWN_DOMAIN_NAMES``)
    plus any domain the backend pulled a window for, even one this build
    doesn't recognize yet (forward-compatible).

    A pulled config that supplied *any* window info (a non-empty
    ``detection_windows`` map, or the deprecated single ``night_window``) is
    authoritative for every domain: a domain it didn't mention resolves to
    ALWAYS (24/7), not the local YAML alias -- mirroring the backend-payload
    rule that a present ``detectionWindows`` map fully supersedes legacy
    aliases, so a cleared window is never resurrected by a stale one. Only
    when the pulled config said nothing at all about windows does the local
    YAML config's own window/alias apply (the offline/dev path).
    """
    domain_names = set(KNOWN_DOMAIN_NAMES)
    if pulled is not None:
        domain_names |= set(pulled.detection_windows)
    pulled_supplied_window_info = _pulled_supplied_window_info(pulled)
    return {
        name: _resolve_domain_window(yaml_config, pulled, name, pulled_supplied_window_info)
        for name in domain_names
    }


def _pulled_supplied_window_info(pulled: PulledWorkerConfig | None) -> bool:
    if pulled is None:
        return False
    return bool(pulled.detection_windows) or pulled.night_window is not None


def _resolve_domain_window(
    yaml_config: WorkerConfig,
    pulled: PulledWorkerConfig | None,
    name: str,
    pulled_supplied_window_info: bool,
) -> NightWindowConfig | None:
    if pulled_supplied_window_info:
        pulled_window = _pulled_domain_window(pulled, name)
        if pulled_window is None:
            return None
        return NightWindowConfig(
            start=pulled_window.start,
            end=pulled_window.end,
            tz=pulled_window.tz,
        )
    return yaml_config.domains.resolved_detection_window(name)


def _pulled_domain_window(
    pulled: PulledWorkerConfig | None,
    name: str,
) -> PulledNightWindow | None:
    if pulled is None:
        return None
    window = pulled.detection_windows.get(name)
    if window is not None:
        return window
    # Deprecated single-window field; treated as a "bed_exit" pull only.
    return pulled.night_window if name == "bed_exit" else None


__all__ = ["resolve_runtime_config"]

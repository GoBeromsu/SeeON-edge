"""Worker-side successor of the PR-9 night-window characterization.

Original edge coverage (edge/runtime/config_resolver.py, edge/runtime/edge_worker.py,
edge/domains/bed_exit/detector.py) and its worker-side disposition:

- ``resolve_runtime_config`` (pulled-config-overrides-yaml precedence, fall back
  to yaml, ``None`` when neither supplies a window) -> ported below against
  ``worker.runtime.config.config_resolver.resolve_runtime_config``, which has
  identical precedence semantics. This closes a real coverage gap: the worker
  function had zero direct tests before this file (confirmed via
  ``rg -n "resolve_runtime_config" tests/`` returning no hits pre-port).
- ``BedExitMonitor.update_night_window`` live in-place flip -> ported as a
  direct characterization of the still-present method, with an explicit note
  that it is currently an unwired capability in production: worker replaced
  edge's push-based config hot-reload with restart-on-config-change
  (``worker/runtime/config/restart.py``, fully tested in
  ``tests/test_worker_restart_directive.py``). ``WorkerRuntime._bed_exit_config``
  (worker/runtime/worker.py) reads ``self.config.domains.bed_exit.night_window``
  directly at camera build time and never calls ``resolve_runtime_config`` or
  ``update_night_window`` (confirmed via
  ``rg -n "update_night_window" worker/`` returning zero call sites outside
  this detector's own definition). Worth keeping so a future rewiring of a
  live-reload path does not silently regress this method's own contract, but
  the "live flip without reconstruction" *use case* itself is currently dead
  code, not exercised by any production caller. Report-worthy observation, not
  a bug to patch.
- ``_config_refresh`` (pull-apply-or-keep-last-on-exception orchestration) and
  ``EdgeWorkerSupervisor``'s heartbeat-cadence ``config_refresh`` callback ->
  superseded by architecture change, not ported: worker has no analogous
  "mutate a live decider on a timer" path. The equivalent resilience and
  cadence guarantees now live in the restart-check mechanism:
  ``tests/test_worker_ingest_lifecycle.py::test_restart_check_returning_true_stops_the_supervisor_cleanly``,
  ``::test_omitting_restart_check_preserves_default_clean_completion``,
  ``::test_restart_check_always_false_is_consulted_but_never_stops_early``
  (``IngestSupervisor._watch_restart`` polling ``restart_check`` on a timer,
  worker/pipeline/ingest/lifecycle.py) and the pull/apply/last-known-good
  semantics in ``tests/test_worker_config_pull_lkg.py``
  (``resolve_effective_config`` / ``WorkerConfigLkgStore``). A pull failure
  under worker's architecture simply defers the next restart rather than
  reaching into a running camera's decider state.
"""

from __future__ import annotations

from datetime import UTC, datetime

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor, NightWindow
from worker.runtime.config import WorkerConfig
from worker.runtime.config.config_resolver import resolve_runtime_config
from worker.runtime.config.domain_models import NightWindowConfig
from worker.types import BusinessEvent, DecisionInput


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1, y1, x2, y2, confidence)


def _observation_for(person: BoundingBox, bed: BoundingBox) -> FrameObservation:
    return FrameObservation(detections=((person,), ()), regions=((bed,), ()))


def _decision_input(observation: FrameObservation, time_sec: float) -> DecisionInput:
    return DecisionInput(
        observation=observation,
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=time_sec,
        frame_index=int(time_sec),
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )


def _exit_events(monitor: BedExitMonitor, *, bed: BoundingBox) -> tuple[BusinessEvent, ...]:
    monitor.update(_decision_input(_observation_for(_box(20, 0, 120, 100), bed), 0.0))
    return monitor.update(_decision_input(_observation_for(_box(60, 0, 160, 100), bed), 1.0))


def _yaml_config(night_window: dict[str, str] | None) -> WorkerConfig:
    bed_exit: dict[str, object] = {"enabled": True}
    if night_window is not None:
        bed_exit["night_window"] = night_window
    return WorkerConfig.model_validate(
        {
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "domains": {"bed_exit": bed_exit},
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1.local/trackID=2",
                }
            ],
        }
    )


def _pulled(night_window: PulledNightWindow | None) -> PulledWorkerConfig:
    return PulledWorkerConfig(
        config_version=7, restart_epoch=0, night_window=night_window, cameras=()
    )


def test_resolve_night_window_prefers_pulled_over_yaml() -> None:
    yaml_config = _yaml_config({"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"})
    pulled = _pulled(PulledNightWindow(start="22:00", end="06:00", tz="UTC"))

    resolved = resolve_runtime_config(yaml_config, pulled)

    assert resolved["bed_exit"] == NightWindowConfig(start="22:00", end="06:00", tz="UTC")


def test_resolve_night_window_falls_back_to_yaml() -> None:
    yaml_config = _yaml_config({"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"})

    resolved = resolve_runtime_config(yaml_config, None)

    assert resolved["bed_exit"] == NightWindowConfig(start="21:00", end="05:00", tz="Asia/Seoul")


def test_resolve_night_window_returns_none_without_pulled_or_yaml_window() -> None:
    resolved = resolve_runtime_config(_yaml_config(None), None)

    assert resolved["bed_exit"] is None


def test_resolve_runtime_config_reaches_non_bed_exit_domains_via_pulled_detection_windows() -> None:
    """Generalizing night_window (issue #24): resolve_runtime_config now
    resolves every KNOWN_DOMAIN_NAMES entry, not just bed_exit, from the
    pulled config's per-domain detection_windows map."""
    yaml_config = _yaml_config(None)
    pulled = PulledWorkerConfig(
        config_version=7,
        restart_epoch=0,
        night_window=None,
        cameras=(),
        detection_windows={
            "fall": PulledNightWindow(start="22:00", end="05:00", tz="UTC"),
        },
    )

    resolved = resolve_runtime_config(yaml_config, pulled)

    assert resolved["fall"] == NightWindowConfig(start="22:00", end="05:00", tz="UTC")
    assert resolved["bed_exit"] is None


def test_resolve_runtime_config_explicit_bed_exit_pull_wins_over_yaml_alias() -> None:
    """An explicit pulled detection_windows["bed_exit"] entry takes
    precedence over the yaml domains.bed_exit.night_window alias, matching
    detection_windows' precedence over the deprecated night_window field."""
    yaml_config = _yaml_config({"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"})
    pulled = PulledWorkerConfig(
        config_version=7,
        restart_epoch=0,
        night_window=None,
        cameras=(),
        detection_windows={
            "bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="UTC"),
        },
    )

    resolved = resolve_runtime_config(yaml_config, pulled)

    assert resolved["bed_exit"] == NightWindowConfig(start="22:00", end="06:00", tz="UTC")


def test_resolve_runtime_config_pulled_supplying_any_window_is_authoritative_for_all() -> None:
    """Plan-file rule: once pulled supplied window info at all (any
    detection_windows entry, or the deprecated night_window), it is
    authoritative for EVERY domain -- a domain it didn't mention resolves to
    ALWAYS (None), NOT the yaml alias, even when yaml configured a window for
    that domain. This is what makes a cleared window impossible to
    resurrect via a stale local alias once the backend has an opinion."""
    yaml_config = WorkerConfig.model_validate(
        {
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "domains": {
                "detection_windows": {
                    "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
                }
            },
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1.local/trackID=2",
                }
            ],
        }
    )
    pulled = PulledWorkerConfig(
        config_version=7,
        restart_epoch=0,
        night_window=None,
        cameras=(),
        detection_windows={
            "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
        },
    )

    resolved = resolve_runtime_config(yaml_config, pulled)

    assert resolved["bed_exit"] == NightWindowConfig(start="21:00", end="06:00", tz="UTC")
    assert resolved["fall"] is None


def test_resolve_runtime_config_yaml_alias_applies_when_pulled_supplies_nothing() -> None:
    """The inverse of the above: when pulled said nothing about windows at
    all (None, or empty detection_windows and no legacy night_window), the
    local yaml config's own per-domain windows/aliases fully apply."""
    yaml_config = WorkerConfig.model_validate(
        {
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "domains": {
                "detection_windows": {
                    "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
                }
            },
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1.local/trackID=2",
                }
            ],
        }
    )

    resolved = resolve_runtime_config(yaml_config, None)

    assert resolved["fall"] == NightWindowConfig(start="22:00", end="05:00", tz="UTC")


def test_bed_exit_monitor_live_night_window_flip_without_reconstruction() -> None:
    # Characterizes BedExitMonitor.update_night_window as a still-correct direct
    # mutator; see module docstring for why no production caller reaches it today.
    bed = _box(0, 0, 100, 100)

    def clock() -> datetime:
        return datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    monitor = BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
            night_window=NightWindow(start="21:00", end="05:00", tz="UTC"),
        ),
        clock=clock,
        boot_id="test-boot",
        stream_epoch="test-epoch",
        source_generation=0,
    )

    # Noon is outside the configured 21:00-05:00 night window: gated, no emission.
    assert _exit_events(monitor, bed=bed) == ()

    monitor.update_night_window(NightWindow(start="11:00", end="13:00", tz="UTC"))

    events = _exit_events(monitor, bed=bed)
    assert len(events) == 1
    assert events[0] == BusinessEvent(
        domain="bed_exit",
        event_type="bed-exit",
        identity="test-boot:test-epoch:bed-exit:0:0:0:0:1",
        camera_id="camera-1",
        facility_id="facility-1",
        time_sec=1.0,
        probability=1.0,
        person_id=0,
        bed_id=0,
    )

    monitor.update_night_window(NightWindow(start="21:00", end="05:00", tz="UTC"))
    assert _exit_events(monitor, bed=bed) == ()

    monitor.update_night_window(None)
    assert len(_exit_events(monitor, bed=bed)) == 1

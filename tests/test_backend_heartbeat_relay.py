from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.features.status.backend_heartbeat_relay import (
    HeartbeatRelayState,
    get_heartbeat_relay_state,
    relay_heartbeats_once,
)
from backend.app.features.status.heartbeat_store import HeartbeatStore
from backend.app.lifespan import API_BACKEND_HEARTBEAT_RELAY_SEC_ENV
from backend.app.main import create_app


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "API_EDGE_RELAY_TOKEN",
        "API_CAMERA_INVENTORY",
        "API_FACILITY_ID",
        "API_BACKEND_CONFIG_URL",
        "API_BACKEND_EVENTS_URL",
        "EDGE_FACILITY_TOKEN",
        API_BACKEND_HEARTBEAT_RELAY_SEC_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


class FakeIngestClient:
    """Records per-camera heartbeat calls; ``.for_camera`` clones like the real client."""

    def __init__(self, *, failing_camera_ids: frozenset[str] = frozenset()) -> None:
        self.calls: list[str] = []
        self.failing_camera_ids = failing_camera_ids
        self.camera_id: str | None = None

    def for_camera(self, camera_id: str) -> FakeIngestClient:
        clone = FakeIngestClient(failing_camera_ids=self.failing_camera_ids)
        clone.calls = self.calls  # shared call log across clones, like a real relay tick
        clone.camera_id = camera_id
        return clone

    def send_heartbeat(self) -> bool:
        assert self.camera_id is not None
        self.calls.append(self.camera_id)
        return self.camera_id not in self.failing_camera_ids


def _make_app(*, client: object | None, inventory: dict[str, dict[str, str | None]]) -> object:
    state = SimpleNamespace(
        heartbeat_store=HeartbeatStore(stale_after_sec=90.0),
        camera_inventory=inventory,
    )
    if client is not None:
        state.backend_ingest_client = client
    return SimpleNamespace(state=state)


def test_online_camera_relayed_stale_and_never_seen_not_called() -> None:
    inventory = {
        "cam-online": {"camera_id": "cam-online", "facility_id": "fac-1"},
        "cam-stale": {"camera_id": "cam-stale", "facility_id": "fac-1"},
        "cam-never-seen": {"camera_id": "cam-never-seen", "facility_id": "fac-1"},
    }
    client = FakeIngestClient()
    app = _make_app(client=client, inventory=inventory)
    app.state.heartbeat_store.record("cam-online", "fac-1", received_at=1000.0)
    app.state.heartbeat_store.record("cam-stale", "fac-1", received_at=0.0)

    result = relay_heartbeats_once(app, now=1010.0)

    assert client.calls == ["cam-online"]
    assert result.attempted == 1
    assert result.sent == 1
    assert result.failed == 0
    assert result.skipped_reason is None


def test_no_client_attr_is_noop() -> None:
    app = _make_app(client=None, inventory={"cam-a": {"camera_id": "cam-a", "facility_id": "f"}})

    result = relay_heartbeats_once(app, now=0.0)

    assert result.skipped_reason == "no_client"
    assert result.attempted == 0


def test_client_without_send_heartbeat_or_for_camera_is_noop_no_exception() -> None:
    # Bare sentinel like the fakes injected by other tests (e.g. object()) --
    # missing .for_camera entirely must be treated as unusable, not crash.
    app = _make_app(client=object(), inventory={})
    app.state.heartbeat_store.record("cam-a", "fac-1", received_at=0.0)

    result = relay_heartbeats_once(app, now=1.0)

    assert result.skipped_reason == "no_client"


def test_client_with_for_camera_but_no_send_heartbeat_on_clone_is_noop() -> None:
    class NoSendHeartbeat:
        def for_camera(self, camera_id: str) -> object:
            return object()  # clone has neither send_heartbeat nor for_camera

    app = _make_app(
        client=NoSendHeartbeat(),
        inventory={"cam-a": {"camera_id": "cam-a", "facility_id": "fac-1"}},
    )
    app.state.heartbeat_store.record("cam-a", "fac-1", received_at=0.0)

    result = relay_heartbeats_once(app, now=1.0)

    assert result.skipped_reason is None
    assert result.attempted == 1
    assert result.sent == 0
    assert result.failed == 1


def test_no_online_cameras_is_noop() -> None:
    app = _make_app(client=FakeIngestClient(), inventory={})

    result = relay_heartbeats_once(app, now=0.0)

    assert result.skipped_reason == "no_online_cameras"


def test_per_camera_failure_does_not_stop_others() -> None:
    inventory = {
        "cam-good": {"camera_id": "cam-good", "facility_id": "fac-1"},
        "cam-bad": {"camera_id": "cam-bad", "facility_id": "fac-1"},
    }
    client = FakeIngestClient(failing_camera_ids=frozenset({"cam-bad"}))
    app = _make_app(client=client, inventory=inventory)
    app.state.heartbeat_store.record("cam-good", "fac-1", received_at=1000.0)
    app.state.heartbeat_store.record("cam-bad", "fac-1", received_at=1000.0)

    result = relay_heartbeats_once(app, now=1010.0)

    assert set(client.calls) == {"cam-good", "cam-bad"}
    assert result.attempted == 2
    assert result.sent == 1
    assert result.failed == 1


def test_backoff_doubles_on_consecutive_all_fail_ticks_and_resets_on_success() -> None:
    inventory = {"cam-a": {"camera_id": "cam-a", "facility_id": "fac-1"}}
    failing_client = FakeIngestClient(failing_camera_ids=frozenset({"cam-a"}))
    app = _make_app(client=failing_client, inventory=inventory)
    app.state.heartbeat_store.record("cam-a", "fac-1", received_at=1000.0)

    relay_heartbeats_once(app, now=1010.0)
    state = get_heartbeat_relay_state(app)
    assert state.backoff_multiplier == 2
    assert state.consecutive_all_fail_ticks == 1

    relay_heartbeats_once(app, now=1020.0)
    assert state.backoff_multiplier == 4
    assert state.consecutive_all_fail_ticks == 2

    relay_heartbeats_once(app, now=1030.0)
    assert state.backoff_multiplier == 8

    relay_heartbeats_once(app, now=1040.0)
    assert state.backoff_multiplier == 8  # capped

    # Swap in a healthy client: the very next successful tick resets backoff.
    app.state.backend_ingest_client = FakeIngestClient()
    relay_heartbeats_once(app, now=1050.0)
    assert state.backoff_multiplier == 1
    assert state.consecutive_all_fail_ticks == 0


def test_skipped_tick_does_not_touch_backoff_state() -> None:
    app = _make_app(client=None, inventory={})
    initial_state = get_heartbeat_relay_state(app)
    initial_state.backoff_multiplier = 4
    initial_state.consecutive_all_fail_ticks = 2

    relay_heartbeats_once(app, now=0.0)

    state = get_heartbeat_relay_state(app)
    assert state.backoff_multiplier == 4
    assert state.consecutive_all_fail_ticks == 2


def test_get_heartbeat_relay_state_self_heals_and_is_cached() -> None:
    app = _make_app(client=None, inventory={})

    first = get_heartbeat_relay_state(app)
    second = get_heartbeat_relay_state(app)

    assert isinstance(first, HeartbeatRelayState)
    assert first is second


def test_disabled_via_env_zero_never_schedules_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_BACKEND_HEARTBEAT_RELAY_SEC_ENV, "0")

    with TestClient(create_app()) as client:
        app = client.app
        assert app.state.backend_heartbeat_relay_task is None
        assert app.state.backend_heartbeat_relay_executor is None


def test_disabled_via_invalid_env_never_schedules_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_BACKEND_HEARTBEAT_RELAY_SEC_ENV, "not-a-number")

    with TestClient(create_app()) as client:
        assert client.app.state.backend_heartbeat_relay_task is None


def test_real_lifespan_boots_and_shuts_down_cleanly_with_relay_enabled_and_empty_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default env (unset) -> relay enabled at 30s; nothing in this test waits
    # 30s, so the loop task starts, waits, and gets cleanly cancelled/stopped
    # at shutdown without ever ticking -- mirrors the existing lifespan boot
    # smoke coverage in tests/test_ml_api_config_pull.py.
    with TestClient(create_app()) as client:
        app = client.app
        assert app.state.backend_heartbeat_relay_task is not None
        assert app.state.camera_inventory == {}

    assert app.state.backend_heartbeat_relay_task is None
    assert app.state.backend_heartbeat_relay_executor is None

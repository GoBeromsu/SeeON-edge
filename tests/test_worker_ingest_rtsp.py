from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from contracts.frame import Frame
from worker.pipeline.ingest.probe import RTSPProbeError, probe_first_frame
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.pipeline.ingest.rtsp_url import mask_rtsp_url
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class _DecodeConfig:
    url: str


class _Session:
    def __init__(
        self,
        packets: list[FramePacket | None],
        close_error: OSError | None = None,
    ) -> None:
        self._packets = packets
        self._close_error = close_error
        self.closed = False
        self.close_count = 0

    def read(self) -> FramePacket | None:
        return self._packets.pop(0) if self._packets else None

    def close(self) -> None:
        self.close_count += 1
        if self._close_error is not None:
            raise self._close_error
        self.closed = True


class _Adapter:
    def __init__(self, outcomes: list[_Session | Exception]) -> None:
        self._outcomes = outcomes
        self.configs: list[_DecodeConfig] = []

    def open(self, config: _DecodeConfig) -> _Session:
        self.configs.append(config)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _packet(seq: int, value: int = 0) -> FramePacket:
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=float(seq), image=image)
    return FramePacket("camera-a", frame, float(seq), seq, 3, 2, 0.25)


def test_rtsp_source_opens_injected_decoder_with_exact_config_and_closes() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    session = _Session([_packet(0, 10), _packet(1, 20), None])
    adapter = _Adapter([session])

    packets = list(
        RTSPSource(
            config,
            adapter,
            max_failures=1,
            max_total_reconnects=0,
            pace_wait=lambda _delay: False,
        )
    )

    assert adapter.configs == [config]
    assert [packet.seq for packet in packets] == [0, 1]
    assert packets[0].frame.image.tolist() == np.full((2, 3, 3), 10).tolist()
    assert session.closed is True


def test_rtsp_source_reconnects_after_read_failure_and_reports_recovery() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    first = _Session([None])
    second = _Session([_packet(0), None])
    adapter = _Adapter([first, second])
    sleeps: list[float] = []
    liveness: list[tuple[str, str]] = []
    source = RTSPSource(
        config,
        adapter,
        max_failures=1,
        max_total_reconnects=1,
        sleep=sleeps.append,
        pace_wait=lambda _delay: True,
    )
    source.set_liveness_callbacks(
        on_reconnecting=lambda reason: liveness.append(("degraded", reason)),
        on_recovered=lambda reason: liveness.append(("ready", reason)),
    )

    packets = list(source)

    assert [packet.seq for packet in packets] == [0]
    assert adapter.configs == [config, config]
    assert first.closed is True
    assert second.closed is True
    assert sleeps == [0.25]
    assert liveness == [("degraded", "read_failure"), ("ready", "read_recovered")]


def test_rtsp_source_continues_after_reconnect_close_failure() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    first = _Session([None], close_error=OSError("close failed"))
    replacement = _Session([_packet(0), None])
    adapter = _Adapter([first, replacement])
    waits: list[float] = []

    packets = list(
        RTSPSource(
            config,
            adapter,
            max_failures=1,
            max_total_reconnects=1,
            backoff_wait=lambda delay: waits.append(delay) or False,
            pace_wait=lambda _delay: True,
        )
    )

    assert [packet.stream_epoch for packet in packets] == [2]
    assert first.close_count == 1
    assert adapter.configs == [config, config]
    assert waits == [0.25]


def test_rtsp_source_stops_after_close_and_replacement_open_fail_once() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    first = _Session([None], close_error=OSError("close failed"))
    adapter = _Adapter([first, OSError("replacement unavailable")])
    failures: list[str] = []
    waits: list[float] = []

    packets = list(
        RTSPSource(
            config,
            adapter,
            max_failures=1,
            max_total_reconnects=1,
            backoff_wait=lambda delay: waits.append(delay) or False,
            on_open_failure=failures.append,
            pace_wait=lambda _delay: True,
        )
    )

    assert packets == []
    assert first.close_count == 1
    assert failures == ["spawn_failed"]
    assert waits == [0.25]


def test_rtsp_source_retries_external_open_failure_but_propagates_programming_error() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    recovered = _Session([_packet(0), None])
    recovering_adapter = _Adapter([OSError("unavailable"), recovered])

    packet = next(
        iter(
            RTSPSource(
                config,
                recovering_adapter,
                max_total_reconnects=1,
                sleep=lambda _delay: None,
                pace_wait=lambda _delay: True,
            )
        )
    )

    assert packet.seq == 0
    assert recovering_adapter.configs == [config, config]

    broken_adapter = _Adapter([TypeError("bad decoder wiring")])
    with pytest.raises(TypeError, match="bad decoder wiring"):
        next(iter(RTSPSource(config, broken_adapter)))


def test_rtsp_source_paces_packets_with_injected_clock() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    adapter = _Adapter([_Session([_packet(0), _packet(1), None])])
    now = 10.0
    waits: list[float] = []

    def clock() -> float:
        return now

    def pace_wait(delay_sec: float) -> bool:
        nonlocal now
        waits.append(delay_sec)
        now += delay_sec
        return False

    packets = list(
        RTSPSource(
            config,
            adapter,
            max_failures=1,
            max_total_reconnects=0,
            target_fps=2.0,
            clock=clock,
            pace_wait=pace_wait,
        )
    )

    assert [packet.seq for packet in packets] == [0, 1]
    assert waits == [0.5, 0.5]


def test_probe_first_frame_reports_geometry_and_closes_session() -> None:
    raw_url = "rtsp://user:secret@camera.local/live?token=abc"
    config = _DecodeConfig(raw_url)
    session = _Session([_packet(0)])
    adapter = _Adapter([session])

    result = probe_first_frame(
        raw_url,
        decoder=adapter,
        config=config,
        requested_backend="cpu",
        selected_backend="opencv",
        timeout_ms=1000,
    )

    assert result.width == 3
    assert result.height == 2
    assert result.channels == 3
    assert result.requested_backend == "cpu"
    assert result.backend == "opencv"
    assert result.masked_url == "rtsp://***:***@camera.local/live?token=%2A%2A%2A"
    assert adapter.configs == [config]
    assert session.closed is True


def test_probe_first_frame_classifies_auth_and_masks_credentials() -> None:
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"
    config = _DecodeConfig(raw_url)
    adapter = _Adapter([RuntimeError(f"401 Unauthorized for {raw_url}")])

    with pytest.raises(RTSPProbeError) as error:
        probe_first_frame(raw_url, decoder=adapter, config=config)

    assert error.value.error_class == "auth"
    assert error.value.masked_url == (
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A"
    )
    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "plain" not in str(error.value)


def test_probe_first_frame_classifies_timeout_and_closes_session() -> None:
    config = _DecodeConfig("rtsp://camera.local/live")
    session = _Session([None, None])
    adapter = _Adapter([session])
    values = iter((0.0, 0.1, 0.2, 0.6))

    with pytest.raises(RTSPProbeError) as error:
        probe_first_frame(
            config.url,
            decoder=adapter,
            config=config,
            timeout_ms=500,
            monotonic=lambda: next(values),
        )

    assert error.value.error_class == "timeout"
    assert session.closed is True


def test_mask_rtsp_url_redacts_userinfo_query_values_and_fragment() -> None:
    assert (
        mask_rtsp_url(
            "rtsp://user:password@host/stream"
            "?profile=main&username=admin&secret=abc#fragment-secret"
        )
        == "rtsp://***:***@host/stream"
        "?profile=%2A%2A%2A&username=%2A%2A%2A&secret=%2A%2A%2A"
    )

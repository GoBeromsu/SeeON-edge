"""RTSP ingest/decode/probe characterization, migrated off the edge sources
package.

ARCHITECTURE CHANGE (confirmed by `rg` returning zero hits in worker/ for
FallbackRTSPBackend, ObservingRTSPBackend, create_backend, _create_backend,
and any RTSP probe CLI `main()`): edge selected a decode backend PER
CONNECTION at runtime (env var / "auto" probing with in-process fallback
across NVDEC->OpenCV, self-tracked via DecodeSelection telemetry inside the
backend wrapper classes). Worker selects a decode adapter ONCE at boot via
a static profile map (worker/runtime/profile/registry.py:PROFILE_REGISTRY,
proved by tests/test_worker_decode_cpu.py::test_profile_policy_selects_cpu_
decode_only_for_cpu_and_mps -> {"cuda": "nvdec", "mps": "opencv",
"cpu": "opencv"}) plus a DecodeProbe preflight check (worker/runtime/
profile/boot.py), and injects exactly one DecodeAdapter into RTSPSource at
construction (worker/pipeline/ingest/rtsp.py). There is no runtime
fallback-with-retry-across-backends and no CLI entrypoint for RTSP probing
in worker; this is an intentional ADR-0002 fail-loud simplification, not an
oversight (tests/test_worker_nvdec_adapter.py::test_first_frame_failure_is_
masked_camera_local_and_has_no_cpu_fallback is named for exactly this).

Full test-by-test disposition of the 37 tests in the edge original:

PORTED (this file, adapted to decoder+config injection):
- test_opencv_rtsp_backend_falls_back_when_parameterized_open_fails ->
  test_cpu_av_falls_back_through_parameterized_open_signatures_to_bare_url
  (CpuAvAdapter._open_capture's 3-arg->2-arg->1-arg TypeError cascade,
  worker/adapters/decode/cpu_av/adapter.py:131-138, exercised only on the
  happy 3-arg path by tests/test_worker_decode_cpu.py, never the full
  cascade)
- test_probe_first_frame_classifies_decode_failure ->
  test_probe_first_frame_classifies_decode_failure_and_closes_session
  (worker/pipeline/ingest/probe.py:107-121 _frame_dimensions still raises
  a "decode" RTSPProbeError for a malformed packet shape; untested
  worker-side)
- test_rtsp_source_stop_predicate_cancels_initial_open_backoff (backoff_wait
  / stop_requested params exist on worker's RTSPSource but are untested)
- test_rtsp_source_backoff_is_bounded (_backoff_delay exponential-with-cap,
  untested worker-side)
- test_rtsp_source_never_recovered_terminates_with_reconnect_budget
  (terminal budget-exhaustion-with-zero-recoveries path, untested)
- test_rtsp_source_stop_predicate_cancels_reconnect_backoff_promptly
  (backoff_wait/stop_requested during a RECONNECT, not just initial, open)
- test_rtsp_source_records_offline_before_reconnect_budget_exhaustion
  (liveness "degraded" callback fires even though budget then exhausts)
- test_rtsp_source_subtracts_consumer_time_from_pacing_delay (external,
  test-driven clock advance between yield and the remaining-delay
  computation; not exercised by the zero-consumer-time pacing test below)

SUPERSEDED (file:line):
- test_rtsp_source_uses_injected_rgb_backend_without_double_conversion ->
  tests/test_worker_ingest_rtsp.py:51-69 (exact-config open/close/packet
  sequence) + tests/test_worker_decode_cpu.py:148-193 (BGR->RGB conversion
  now lives entirely inside CpuAvAdapter; RTSPSource touches no cv2 call at
  all in the new architecture, so "without double conversion" is structural)
- test_probe_first_frame_reports_resolution_channels_and_releases_capture ->
  tests/test_worker_ingest_rtsp.py:158-180
- test_probe_first_frame_classifies_timeout_and_releases_capture ->
  tests/test_worker_ingest_rtsp.py:200-216
- test_probe_first_frame_classifies_auth_and_masks_credentials ->
  tests/test_worker_ingest_rtsp.py:183-197 (same fixture URL)
- test_mask_rtsp_url_redacts_userinfo_query_values_and_fragment ->
  tests/test_worker_ingest_rtsp.py:219-227 (byte-identical assertion)
- test_ffprobe_failure_never_exposes_rtsp_credentials (the flagged "L533
  landmine") -> tests/test_worker_nvdec_probe.py:88-108
  test_ffprobe_failure_masks_command_url_and_credentials (same fixture URL,
  same error-message format, same credential-absence assertions, now via
  probe_stream_metadata(config, runner=...) injection instead of
  monkeypatching subprocess.run)
- test_nvdec_capture_read_times_out_and_cleans_up_hanging_pipe ->
  tests/test_worker_nvdec_process.py:89-103
  test_decode_process_read_timeout_reaps_hanging_child_and_reader
- test_opencv_rtsp_backend_sets_timeout_and_buffer_properties ->
  tests/test_worker_decode_cpu.py:82-114
- test_opencv_rtsp_backend_defaults_rtsp_transport_to_tcp and
  test_opencv_rtsp_backend_preserves_operator_capture_options ->
  tests/test_worker_decode_cpu.py:117-145 (parametrized, both cases)
- test_opencv_rtsp_backend_read_returns_rgb_frame ->
  tests/test_worker_decode_cpu.py:148-193 (same BGR->RGB pixel assertion)
- test_opencv_rtsp_backend_releases_capture ->
  tests/test_worker_decode_cpu.py:227-241 (stronger: also proves
  idempotency) and :196-207 / :210-224 (failed-open / failed-read release)
- test_rtsp_source_reconnects_after_read_failures_and_resumes ->
  tests/test_worker_ingest_rtsp.py:72-99
- test_rtsp_source_retries_initial_open_failures_before_yielding_a_frame and
  test_rtsp_source_propagates_programming_errors_from_open ->
  tests/test_worker_ingest_rtsp.py:102-124
  test_rtsp_source_retries_external_open_failure_but_propagates_programming_error
  (the reconnect loop is generic in count; the OSError-retry-then-success
  and TypeError-propagates-immediately guarantees are both already proved
  there)
- test_rtsp_source_recovers_when_reconnect_open_raises -> composition of
  tests/test_worker_ingest_rtsp.py:72-99 (read-failure triggers reconnect)
  and :102-124 (open OSError is retried, success recovers); the reconnect
  loop code path (rtsp.py:76-89) is the same branch used for both the
  initial and the post-read-failure open, so a failure specifically at the
  reconnect-open call site is not a materially distinct branch
- test_rtsp_source_paces_processed_fps_with_injected_clock ->
  tests/test_worker_ingest_rtsp.py:127-155
  test_rtsp_source_paces_packets_with_injected_clock
- test_rtsp_source_window_fill_wall_clock_matches_configured_fps -> the
  underlying per-iteration pacing arithmetic (worker/pipeline/ingest/
  rtsp.py:111-115) recomputes `started_at` fresh every loop iteration with
  no persistent/cumulative deadline accumulator, so there is no drift
  mechanism a 30-frame run could expose that the 2-frame case above
  doesn't already exercise; also NOTE the wall-clock-matches-fps assertion
  itself (`frame.time_sec == n/fps`) no longer applies to RTSPSource at
  all -- time_sec is now decoder-owned (set by CpuAvAdapter/NvdecCuvidAdapter
  from their own stream-origin clock, not computed by RTSPSource)

IMPOSSIBLE-WITH-REASON (obsolete by the architecture change above, cite
worker/runtime/profile/registry.py PROFILE_REGISTRY as the boot-time
replacement mechanism unless noted otherwise):
- test_create_backend_defaults_to_auto_fail_loud_nvdec
- test_create_backend_selects_opencv_from_env
- test_create_backend_selects_nvdec
- test_create_backend_cpu_alias_maps_to_opencv (intent preserved by
  PROFILE_REGISTRY's "cpu"->"opencv" mapping, tests/test_worker_decode_cpu.py:260-268,
  but no create_backend()-equivalent call surface exists to port onto)
- test_create_backend_rejects_unknown_backend (no create_backend; unknown-
  profile validation, if any, lives at the profile/boot layer, out of RTSP
  module scope)
- test_fallback_backend_uses_safe_backend_when_preferred_open_fails
- test_fallback_backend_prefers_first_backend_that_yields_a_frame
- test_fallback_backend_records_bounded_reason_and_masks_credentials
- test_observing_backend_records_forced_cpu_selection
- test_fallback_backend_classifies_first_frame_failure
- test_fallback_backend_propagates_programming_errors
  (FallbackRTSPBackend/ObservingRTSPBackend do not exist worker-side; a
  single failing adapter now degrades that camera outright with NO
  cross-backend retry, proved negatively by
  tests/test_worker_nvdec_adapter.py:179-226
  test_first_frame_failure_is_masked_camera_local_and_has_no_cpu_fallback
  and tests/test_worker_decode_cpu.py:271-315
  test_rtsp_source_degrades_only_failed_camera_and_closes_its_capture)
- test_probe_first_frame_auto_fails_loud_when_nvdec_down (probe_first_frame
  no longer does backend_name="auto" resolution internally; the caller
  must inject the decoder. The underlying "no silent fallback" guarantee
  this test targeted is proved by test_first_frame_failure_is_masked_
  camera_local_and_has_no_cpu_fallback cited above)
- test_probe_first_frame_reports_forced_nvdec_selection and
  test_probe_first_frame_reports_cpu_alias_as_selected_opencv (no
  `_BACKEND_FACTORIES` registry exists worker-side; requested_backend/
  selected_backend are now plain caller-supplied strings, and the
  "requested may differ from selected" concern is already covered by
  tests/test_worker_ingest_rtsp.py:158-180 which passes
  requested_backend="cpu", selected_backend="opencv")
- test_probe_cli_accepts_backend_contract_vocabulary (GENUINE GAP, not a
  supersession: worker/pipeline/ingest/probe.py has no `main()` at all --
  confirmed via its `__all__` list, which has no "main" entry. No CLI
  entrypoint for RTSP probing exists anywhere in worker/. Flagged for
  team-lead; not silently dropped.)
- test_nvdec_capture_repeated_timeouts_do_not_accumulate_reader_threads
  (obsolete by construction, not merely untested: FFmpegDecodeProcess
  spawns exactly one reader thread in `__init__`
  (worker/adapters/decode/nvdec_cuvid/process.py:103-109), never per
  read_frame() call, so repeated read timeouts on the same process cannot
  accumulate threads -- there is no code path capable of the failure mode
  this test guarded against)

Fixture-privacy note: every rtsp:// URL used below (rtsp://camera/live,
rtsp://camera.local/live) already appears in tests/test_worker_ingest_rtsp.py
and/or tests/test_worker_decode_cpu.py; no new fixture strings were needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from contracts.frame import Frame
from worker.adapters.decode.cpu_av import CpuAvAdapter, CpuAvConfig
from worker.adapters.decode.cpu_av.adapter import CpuAvOpenError
from worker.pipeline.ingest.probe import RTSPProbeError, probe_first_frame
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class _DecodeConfig:
    url: str


class _Session:
    def __init__(self, packets: list[FramePacket | None]) -> None:
        self._packets = packets
        self.closed = False

    def read(self) -> FramePacket | None:
        return self._packets.pop(0) if self._packets else None

    def close(self) -> None:
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


class _FakeCapture:
    def isOpened(self) -> bool:
        return True

    def set(self, prop_id: int, value: float) -> bool:
        del prop_id, value
        return True

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        return False, None

    def release(self) -> None:
        return None


def test_cpu_av_refuses_to_degrade_when_the_open_signature_is_unsupported() -> None:
    """ADR-0003: an unsupported OpenCV build is reported, not worked around.

    This test previously asserted the opposite: that the adapter retried as
    ``(url, backend)`` and then ``(url)`` until one signature stuck. That
    cascade silently dropped the required open/read timeouts and the explicit
    FFmpeg backend, so a capture opened through it had neither the timeout
    contract nor the backend the adapter claims to require -- and the operator
    saw a working camera instead of a configuration error.
    """
    calls: list[tuple[str, int | None, list[int] | None]] = []
    capture = _FakeCapture()

    def factory(
        url: str,
        backend: int | None = None,
        params: list[int] | None = None,
    ) -> _FakeCapture:
        calls.append((url, backend, params))
        if params is not None or backend is not None:
            raise TypeError("VideoCapture signature not supported")
        return capture

    adapter = CpuAvAdapter(capture_factory=factory)
    config = CpuAvConfig(
        camera_id="camera-a",
        url="rtsp://camera/live",
        open_timeout_ms=1234,
        read_timeout_ms=5678,
    )

    with pytest.raises(CpuAvOpenError) as captured:
        _ = adapter.open(config)

    # Exactly one attempt, with the full explicit signature.
    assert calls == [
        (
            "rtsp://camera/live",
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                1234,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5678,
            ],
        ),
    ]
    assert "supported boundary" in str(captured.value)
    # The credentialed URL never leaks into the error text.
    assert "rtsp://" not in str(captured.value)


def test_probe_first_frame_classifies_decode_failure_and_closes_session() -> None:
    config = _DecodeConfig("rtsp://camera.local/live")
    malformed = _packet(0)
    # A decoder is an infrastructure boundary and can violate its return type at
    # runtime. Forge only the packet envelope after valid lease ownership exists;
    # the probe must classify the invalid geometry and release that ownership.
    object.__setattr__(malformed, "width", 0)
    session = _Session([malformed])
    adapter = _Adapter([session])

    with pytest.raises(RTSPProbeError) as error:
        probe_first_frame(config.url, decoder=adapter, config=config)

    assert error.value.error_class == "decode"
    assert "codec" in str(error.value)
    assert malformed.released is True
    assert session.closed is True


def test_rtsp_source_stop_predicate_cancels_initial_open_backoff() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    adapter = _Adapter([OSError("unavailable")])
    waits: list[float] = []
    stop = False

    def backoff_wait(delay_sec: float) -> bool:
        nonlocal stop
        waits.append(delay_sec)
        stop = True
        return True

    packets = list(
        RTSPSource(
            config,
            adapter,
            backoff_wait=backoff_wait,
            stop_requested=lambda: stop,
            max_total_reconnects=None,
        )
    )

    assert packets == []
    assert waits == [0.25]
    assert len(adapter.configs) == 1


def test_rtsp_source_backoff_is_bounded() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    source = RTSPSource(
        config,
        _Adapter([]),
        reconnect_initial_backoff_sec=0.5,
        reconnect_max_backoff_sec=1.0,
    )

    assert [source._backoff_delay(reconnect) for reconnect in (1, 2, 3, 4, 10_000)] == [
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_rtsp_source_never_recovered_terminates_with_reconnect_budget() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    sessions = [_Session([None]), _Session([None]), _Session([None])]
    adapter = _Adapter(list(sessions))
    sleeps: list[float] = []

    packets = list(
        RTSPSource(
            config,
            adapter,
            max_failures=1,
            reconnect_initial_backoff_sec=0.25,
            reconnect_max_backoff_sec=1.0,
            max_total_reconnects=2,
            sleep=sleeps.append,
            pace_wait=lambda _delay: False,
        )
    )

    assert packets == []
    assert len(adapter.configs) == 3
    assert all(session.closed for session in sessions)
    assert sleeps == [0.25, 0.5]


def test_rtsp_source_stop_predicate_cancels_reconnect_backoff_promptly() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    session = _Session([None])
    adapter = _Adapter([session])
    stop = False
    waits: list[float] = []

    def backoff_wait(delay_sec: float) -> bool:
        nonlocal stop
        waits.append(delay_sec)
        stop = True
        return stop

    packets = list(
        RTSPSource(
            config,
            adapter,
            max_failures=1,
            reconnect_initial_backoff_sec=30.0,
            reconnect_max_backoff_sec=30.0,
            max_total_reconnects=None,
            backoff_wait=backoff_wait,
            stop_requested=lambda: stop,
            pace_wait=lambda _delay: False,
        )
    )

    assert packets == []
    assert len(adapter.configs) == 1
    assert session.closed is True
    assert waits == [30.0]


def test_rtsp_source_records_offline_before_reconnect_budget_exhaustion() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    adapter = _Adapter([_Session([None]), _Session([None])])
    liveness: list[tuple[str, str]] = []
    source = RTSPSource(
        config,
        adapter,
        max_failures=1,
        reconnect_initial_backoff_sec=0.25,
        reconnect_max_backoff_sec=1.0,
        max_total_reconnects=1,
        sleep=lambda _delay: None,
        pace_wait=lambda _delay: False,
    )
    source.set_liveness_callbacks(
        on_reconnecting=lambda reason: liveness.append(("degraded", reason)),
        on_recovered=lambda reason: liveness.append(("ready", reason)),
    )

    assert list(source) == []
    assert len(adapter.configs) == 2
    assert liveness == [("degraded", "read_failure")]


def test_rtsp_source_subtracts_consumer_time_from_pacing_delay() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    session = _Session([_packet(0), _packet(1), None])
    adapter = _Adapter([session])
    now = 10.0
    waits: list[float] = []

    def clock() -> float:
        return now

    def pace_wait(delay_sec: float) -> bool:
        nonlocal now
        waits.append(delay_sec)
        now += delay_sec
        return False

    packets = iter(
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

    assert next(packets).seq == 0
    now += 0.2  # simulate consumer-side processing time between frames
    assert next(packets).seq == 1
    now += 0.6  # consumer time exceeds the frame interval: no wait needed
    assert list(packets) == []
    assert waits == pytest.approx([0.3])


def test_rtsp_reconnect_assigns_a_new_epoch_after_each_sessions_first_frame() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    first = _Session([_packet(0), None])
    second = _Session([_packet(0), None])
    source = RTSPSource(
        config,
        _Adapter([first, second]),
        worker_boot_id="worker-boot-1",
        max_failures=1,
        max_total_reconnects=1,
        sleep=lambda _delay: None,
        pace_wait=lambda _delay: False,
    )

    packets = list(source)

    assert [packet.seq for packet in packets] == [0, 0]
    assert [packet.pts for packet in packets] == [0.0, 0.0]
    assert packets[0].worker_boot_id == packets[1].worker_boot_id == "worker-boot-1"
    assert packets[0].stream_epoch != packets[1].stream_epoch
    assert packets[0].frame_key != packets[1].frame_key


def test_failed_open_does_not_consume_a_stream_epoch() -> None:
    config = _DecodeConfig("rtsp://camera/live")
    recovered = _Session([_packet(0), None])
    source = RTSPSource(
        config,
        _Adapter([OSError("unavailable"), recovered]),
        worker_boot_id="worker-boot-1",
        max_failures=1,
        max_total_reconnects=1,
        sleep=lambda _delay: None,
        pace_wait=lambda _delay: False,
    )

    packet = next(iter(source))

    assert packet.stream_epoch == 1

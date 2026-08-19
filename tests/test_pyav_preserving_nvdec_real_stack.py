from __future__ import annotations

import base64
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import av
import pytest
from fanout_benchmark_harness import build_recorded_clip, recorded_stream_fanout

from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import PacketRingLimits

pytestmark = pytest.mark.real_stack


def _worker_main(urls: tuple[str, ...]) -> int:
    camera_ids = tuple(f"camera-{index + 1}" for index in range(len(urls)))
    repository = PacketRingRepository(
        camera_ids,
        per_camera_limits=PacketRingLimits(2_000, 32 * 1024 * 1024, 30.0),
        global_max_bytes=64 * 1024 * 1024,
    )
    sessions: list[Any] = []
    try:
        for camera_id, url in zip(camera_ids, urls, strict=True):
            session = PyAvPreservingAdapter(
                repository,
                decode_backend="nvdec",
            ).open(
                NvdecCuvidConfig(
                    camera_id,
                    url,
                    open_timeout_ms=5_000,
                    read_timeout_ms=2_000,
                )
            )
            session.set_stream_identity("real-stack-boot", 1)
            sessions.append(session)
        _emit({"event": "ready", "worker_pid": os.getpid()})
        command = sys.stdin.readline().strip()
        if command.startswith("SLOW "):
            duration_sec = float(command.split()[1])
            samples: list[dict[str, float | int]] = []
            deadline = time.monotonic() + duration_sec
            while True:
                packets = tuple(
                    packet
                    for packet in repository.ring(camera_ids[0]).snapshot()
                    if packet.stream.media_type == "video"
                )
                samples.append(
                    {
                        "packet_count": len(packets),
                        "max_presentation_time": max(
                            (float(packet.presentation_time) for packet in packets),
                            default=-1.0,
                        ),
                    }
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                threading.Event().wait(min(3.0, remaining))
                frame = sessions[0].read()
                if frame is not None:
                    frame.release()
            _emit(
                {
                    "event": "slow_consumer",
                    "samples": samples,
                    "overflow_count": sessions[0].decoder_input_overflow_count,
                }
            )
            return 0
        if command != "READ":
            raise RuntimeError("real-stack worker expected READ or SLOW command")

        frames: list[dict[str, object]] = []
        for camera_id, session in zip(camera_ids, sessions, strict=True):
            frame = session.read()
            if frame is None:
                raise RuntimeError(f"{camera_id} produced no decoded frame")
            frames.append(
                {
                    "camera_id": frame.camera_id,
                    "width": frame.width,
                    "height": frame.height,
                    "source_pts": frame.source_pts,
                }
            )
            frame.release()
        packets = tuple(
            packet
            for packet in repository.ring(camera_ids[0]).snapshot()
            if packet.stream.media_type == "video"
        )[-32:]
        _emit(
            {
                "event": "frames",
                "frames": frames,
                "payloads": [base64.b64encode(packet.payload).decode() for packet in packets],
            }
        )

        command = sys.stdin.readline().strip().split()
        if command[:1] != ["EXPECT_FAILURE"] or len(command) != 2:
            raise RuntimeError("real-stack worker expected EXPECT_FAILURE <index>")
        failed_index = int(command[1])
        failure: dict[str, object] | None = None
        for _ in range(8):
            try:
                frame = sessions[failed_index].read()
            except RuntimeError as error:
                failure = {
                    "error_type": type(error).__name__,
                    "is_runtime_error": isinstance(error, RuntimeError),
                    "detail": str(error),
                }
                break
            if frame is not None:
                frame.release()
        if failure is None:
            raise RuntimeError("killed decoder did not fail loudly within bounded reads")
        _emit({"event": "failure", "camera_index": failed_index, **failure})
        return 0
    finally:
        for session in sessions:
            session.close()
        repository.close()


def _emit(document: dict[str, object]) -> None:
    print(json.dumps(document, separators=(",", ":")), flush=True)


def _read_document(process: subprocess.Popen[str], *, timeout: float) -> dict[str, Any]:
    stdout = process.stdout
    if stdout is None:
        raise RuntimeError("worker helper stdout is unavailable")
    readable, _, _ = select.select((stdout,), (), (), timeout)
    if not readable:
        raise TimeoutError(f"worker helper emitted no state within {timeout}s")
    line = stdout.readline()
    if not line:
        stderr = "" if process.stderr is None else process.stderr.read()
        raise RuntimeError(
            f"worker helper exited early with code {process.poll()}: {stderr[-1_000:]}"
        )
    return json.loads(line)


def _independent_demux(
    url: str,
    *,
    opened: threading.Event,
    stop: threading.Event,
    payloads: list[bytes],
    errors: list[Exception],
) -> None:
    container = None
    try:
        container = av.open(
            url,
            mode="r",
            options={"rtsp_transport": "tcp"},
            timeout=(5.0, 5.0),
        )
        opened.set()
        video = container.streams.video[0]
        for packet in container.demux(video):
            payload = bytes(packet)
            if packet.dts is not None and payload:
                payloads.append(payload)
            if stop.is_set():
                return
    except Exception as error:  # noqa: BLE001 - test-thread boundary
        errors.append(error)
        opened.set()
    finally:
        if container is not None:
            container.close()


def _longest_identical_run(left: list[bytes], right: list[bytes]) -> tuple[int, int]:
    longest = 0
    byte_count = 0
    for left_index, left_payload in enumerate(left):
        for right_index, right_payload in enumerate(right):
            if left_payload != right_payload:
                continue
            length = 0
            matched_bytes = 0
            while (
                left_index + length < len(left)
                and right_index + length < len(right)
                and left[left_index + length] == right[right_index + length]
            ):
                matched_bytes += len(left[left_index + length])
                length += 1
            if length > longest:
                longest = length
                byte_count = matched_bytes
    return longest, byte_count


def test_slow_frame_consumer_does_not_freeze_evidence_ring(tmp_path: Path) -> None:
    """A 1-read/3s consumer cannot backpressure the source-packet ring."""
    clip = build_recorded_clip(tmp_path / "slow-consumer.mp4")
    with recorded_stream_fanout(stream_count=1, clip=clip, tmp_path=tmp_path) as (_, urls):
        repository_root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen(
            (sys.executable, str(Path(__file__).resolve()), "--worker", urls[0]),
            cwd=repository_root,
            env={**os.environ, "PYTHONPATH": str(repository_root)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            assert _read_document(process, timeout=20.0)["event"] == "ready"
            assert process.stdin is not None
            process.stdin.write("SLOW 21\n")
            process.stdin.flush()
            result = _read_document(process, timeout=35.0)
            assert result["event"] == "slow_consumer"
            samples = result["samples"]
            counts = [sample["packet_count"] for sample in samples]
            presentation_times = [sample["max_presentation_time"] for sample in samples]
            assert counts[-1] > counts[0]
            assert presentation_times[-1] > presentation_times[0]
            assert sum(
                later > earlier
                for earlier, later in zip(
                    presentation_times, presentation_times[1:], strict=False
                )
            ) >= 4
            assert result["overflow_count"] > 0
            print(
                "SLOW_CONSUMER_RING_GROWTH "
                f"packet_count={counts[0]}->{counts[-1]} "
                f"max_presentation_time={presentation_times[0]:.3f}"
                f"->{presentation_times[-1]:.3f} "
                f"overflow_count={result['overflow_count']}",
                flush=True,
            )
            assert process.wait(timeout=10.0) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def test_nvdec_packet_tee_real_stack(tmp_path: Path) -> None:
    """Two cameras decode in child ffmpeg processes and preserve source bytes."""
    clip = build_recorded_clip(tmp_path / "recorded.mp4")
    with recorded_stream_fanout(stream_count=2, clip=clip, tmp_path=tmp_path) as (_, urls):
        opened = threading.Event()
        stop_demux = threading.Event()
        independent_payloads: list[bytes] = []
        demux_errors: list[Exception] = []
        demux_thread = threading.Thread(
            target=_independent_demux,
            kwargs={
                "url": urls[0],
                "opened": opened,
                "stop": stop_demux,
                "payloads": independent_payloads,
                "errors": demux_errors,
            },
            daemon=True,
            name="independent-rtsp-demux",
        )
        demux_thread.start()
        assert opened.wait(10.0), "independent RTSP demux did not open"
        assert not demux_errors

        repository_root = Path(__file__).resolve().parents[1]
        inherited_pythonpath = os.environ.get("PYTHONPATH", "")
        process = subprocess.Popen(
            (sys.executable, str(Path(__file__).resolve()), "--worker", *urls),
            cwd=repository_root,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    value for value in (str(repository_root), inherited_pythonpath) if value
                ),
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            ready = _read_document(process, timeout=20.0)
            assert ready == {"event": "ready", "worker_pid": process.pid}
            ps_output = subprocess.run(
                ("ps", "-ww", "-o", "pid,ppid,comm,args", "--ppid", str(process.pid)),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=True,
            ).stdout
            print(f"ps --ppid {process.pid}\n{ps_output}", flush=True)
            decoder_rows = [
                row
                for row in ps_output.splitlines()
                if "ffmpeg" in row and "-i pipe:0" in row and "pipe:1" in row
            ]
            assert len(decoder_rows) == 2

            assert process.stdin is not None
            process.stdin.write("READ\n")
            process.stdin.flush()
            frames = _read_document(process, timeout=20.0)
            assert frames["event"] == "frames"
            assert [frame["camera_id"] for frame in frames["frames"]] == [
                "camera-1",
                "camera-2",
            ]
            assert all(frame["source_pts"] is not None for frame in frames["frames"])

            worker_payloads = [base64.b64decode(value) for value in frames["payloads"]]
            stop_demux.set()
            demux_thread.join(timeout=10.0)
            assert not demux_thread.is_alive()
            assert not demux_errors
            matched_packets, matched_bytes = _longest_identical_run(
                worker_payloads,
                independent_payloads,
            )
            assert matched_packets >= 8
            print(
                "SINK_BYTE_EQUALITY "
                f"matched_packets={matched_packets} matched_payload_bytes={matched_bytes}",
                flush=True,
            )

            first_decoder_pid = min(int(row.split()[0]) for row in decoder_rows)
            os.kill(first_decoder_pid, signal.SIGKILL)
            process.stdin.write("EXPECT_FAILURE 0\n")
            process.stdin.flush()
            failure = _read_document(process, timeout=20.0)
            assert failure["event"] == "failure"
            assert failure["camera_index"] == 0
            # Both the demux RuntimeError and the dead-child NvdecUnavailableError
            # are loud bounded failures; the contract is "no silent fallback",
            # not which member of the RuntimeError family won the race.
            assert failure["is_runtime_error"] is True
            assert failure["error_type"] in {"RuntimeError", "NvdecUnavailableError"}
            assert failure["detail"]
            print(
                f"DEAD_DECODER_FAILURE error_type={failure['error_type']} "
                f"detail={failure['detail']}",
                flush=True,
            )
            assert process.wait(timeout=10.0) == 0
        finally:
            stop_demux.set()
            demux_thread.join(timeout=10.0)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_pyav_preserving_nvdec_real_stack.py --worker URL [URL]")
    raise SystemExit(_worker_main(tuple(sys.argv[2:])))

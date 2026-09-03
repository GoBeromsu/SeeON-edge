"""Pin the final ``decoder_for`` packet-sink selection matrix.

``nvdec`` + a packet sink deliberately selects ``PyAvPreservingAdapter``: its
NVDEC branch opens the demux-only ``NvdecPacketTeeSession`` added in todo 4.
CPU and VAAPI retain the same preserving adapter, whose sessions decode in
PyAV. Without a sink, each token keeps its direct decoder selection.

Assertions are on real constructed adapter objects, not on mock call counts.
CI-safe: no camera is opened, no subprocess is spawned.
"""

from __future__ import annotations

from pathlib import Path
from typing import final

import av
import numpy as np
import pytest

from worker.adapters.decode.cpu_av import CpuAvAdapter
from worker.adapters.decode.nvdec_cuvid import NvdecCuvidAdapter
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.adapters.decode.pyav_nvdec import NvdecPacketTeeSession
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.adapters.decode.vaapi import VaapiAdapter
from worker.pipeline.output.evidence.packet_ring import PacketRingLimits, SourcePacketRing
from worker.runtime.ingest_composition import decoder_for, resolve_decode_backend
from worker.types.source_packet import SourcePacket


@final
class _Sink:
    def __init__(self) -> None:
        self.packets: list[SourcePacket] = []

    def append(self, packet: SourcePacket) -> bool:
        self.packets.append(packet)
        return True


@final
class _FakeDecoderProcess:
    def write_packet(self, payload: bytes) -> None:
        del payload

    def close_input(self) -> None:
        return None

    def read_frame(self, timeout_sec: float) -> bytes | None:
        del timeout_sec
        return None

    def reap(self, timeout_sec: float) -> int | None:
        del timeout_sec
        return 0


def _fake_process_spawner(_args: tuple[str, ...], _frame_size: int) -> _FakeDecoderProcess:
    return _FakeDecoderProcess()


def _encode_source(path: Path) -> None:
    output = av.open(str(path), mode="w", format="mp4")
    stream = output.add_stream("libx264", rate=1)
    stream.width = 2
    stream.height = 2
    stream.pix_fmt = "yuv420p"
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
        output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()


@pytest.mark.parametrize("decode", ["opencv", "cpu", "vaapi"])
def test_cpu_and_vaapi_with_a_packet_sink_keep_the_preserving_adapter(
    decode: str,
) -> None:
    """SEAM D1: non-NVDEC preserving behavior remains unchanged."""
    adapter = decoder_for(decode, packet_sink=_Sink())  # type: ignore[arg-type]

    assert type(adapter) is PyAvPreservingAdapter


def test_nvdec_with_a_packet_sink_selects_the_tee_based_preserving_adapter() -> None:
    """SEAM D1: the preserving adapter's NVDEC branch opens
    ``NvdecPacketTeeSession`` rather than restoring in-process PyAV decode."""
    adapter = decoder_for("nvdec", packet_sink=_Sink())  # type: ignore[arg-type]

    assert type(adapter) is PyAvPreservingAdapter
    assert adapter._decode_backend == "nvdec"  # noqa: SLF001 - selection seam


def test_nvdec_packet_sink_opens_a_tee_session(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _encode_source(source)
    sink = SourcePacketRing("camera-a", PacketRingLimits(1, 1_024, 1.0))
    adapter = decoder_for("nvdec", packet_sink=sink)
    adapter._process_spawner = _fake_process_spawner  # noqa: SLF001 - test injection

    session = adapter.open(NvdecCuvidConfig(camera_id="camera-a", url=str(source)))
    try:
        assert isinstance(session, NvdecPacketTeeSession)
    finally:
        session.close()


@pytest.mark.parametrize(
    ("decode", "expected"),
    [
        ("opencv", CpuAvAdapter),
        ("cpu", CpuAvAdapter),
        ("nvdec", NvdecCuvidAdapter),
        ("vaapi", VaapiAdapter),
    ],
)
def test_without_a_packet_sink_each_token_selects_its_own_adapter(
    decode: str, expected: type
) -> None:
    """SEAM D2: the sink-free selection matrix. Wave 2 must not disturb it --
    the ``nvdec`` cell here stays ``NvdecCuvidAdapter``."""
    adapter = decoder_for(decode)  # type: ignore[arg-type]
    assert isinstance(adapter, expected)


def test_unknown_token_fails_closed_with_and_without_a_packet_sink() -> None:
    """SEAM D3: ADR-0002 fail-closed. An unrecognized token must raise before
    adapter construction, whether or not packet preservation is requested."""
    sink = _Sink()

    with pytest.raises(RuntimeError, match="unsupported decode policy"):
        _ = decoder_for("mystery")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="unsupported decode policy"):
        _ = decoder_for("mystery", packet_sink=sink)  # type: ignore[arg-type]


def test_nvdec_override_on_a_non_nvdec_profile_still_raises_with_a_sink() -> None:
    """SEAM D4: the per-camera override conflict rule is independent of the
    packet sink and must survive the refactor."""
    # Given
    sink = _Sink()

    # When / Then
    assert resolve_decode_backend("nvdec", None) == "nvdec"
    assert resolve_decode_backend("nvdec", "auto") == "nvdec"
    assert resolve_decode_backend("nvdec", "cpu") == "cpu"
    with pytest.raises(RuntimeError, match="requires the nvdec boot profile"):
        _ = decoder_for("opencv", "nvdec", packet_sink=sink)  # type: ignore[arg-type]

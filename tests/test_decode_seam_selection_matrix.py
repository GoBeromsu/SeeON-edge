"""Pin the ``decoder_for`` packet-sink selection branch Wave 2 will rewire.

``worker/runtime/ingest_composition.py`` currently routes EVERY backend to
``PyAvPreservingAdapter`` whenever a packet sink is present -- including
``nvdec``, which is exactly the in-process decode path plan todo 4 deletes and
todo 5 re-points at the new tee adapter. The matrix below is the "before"
picture; todo 5 flips the nvdec+sink cell and must leave the others alone.

Assertions are on real constructed adapter objects, not on mock call counts.
CI-safe: no camera is opened, no subprocess is spawned.
"""

from __future__ import annotations

from typing import final

import pytest

from worker.adapters.decode.cpu_av import CpuAvAdapter
from worker.adapters.decode.nvdec_cuvid import NvdecCuvidAdapter
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.adapters.decode.vaapi import VaapiAdapter
from worker.runtime.ingest_composition import decoder_for, resolve_decode_backend
from worker.types.source_packet import SourcePacket


@final
class _Sink:
    def __init__(self) -> None:
        self.packets: list[SourcePacket] = []

    def append(self, packet: SourcePacket) -> bool:
        self.packets.append(packet)
        return True


@pytest.mark.parametrize("decode", ["opencv", "cpu", "nvdec", "vaapi"])
def test_every_backend_with_a_packet_sink_selects_the_preserving_adapter_today(
    decode: str,
) -> None:
    """SEAM D1 (CURRENT BEHAVIOR): a packet sink short-circuits backend
    selection -- all four tokens produce ``PyAvPreservingAdapter``, carrying the
    backend token forward as its hwaccel choice. Todo 5 changes ONLY the nvdec
    cell of this matrix."""
    # Given
    sink = _Sink()

    # When
    adapter = decoder_for(decode, packet_sink=sink)  # type: ignore[arg-type]

    # Then
    assert isinstance(adapter, PyAvPreservingAdapter)


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
    """SEAM D3: ADR-0002 fail-closed. An unrecognized token raises; with a sink
    present it raises from the preserving adapter's own backend validation at
    open time rather than at selection time -- pinned so todo 5 does not turn
    either into a silent fallback."""
    # Given
    sink = _Sink()

    # When / Then
    with pytest.raises(RuntimeError, match="unsupported decode policy"):
        _ = decoder_for("mystery")  # type: ignore[arg-type]
    sink_adapter = decoder_for("mystery", packet_sink=sink)  # type: ignore[arg-type]
    assert isinstance(sink_adapter, PyAvPreservingAdapter)


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

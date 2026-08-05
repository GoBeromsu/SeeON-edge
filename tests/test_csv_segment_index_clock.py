"""Clock-unification regression tests for #165.

``CsvSegmentIndex`` tracks segment start/end times on FFmpeg's own
generation-local clock (seconds since that segment muxer process started).
``ClipRecordingCoordinator.finalize`` windows by *event* time, a
decode-session-relative clock that resets on its own schedule (RTSP
reconnects, independent of when a clip finalize closes and reopens the
encoder generation). #165 found these two clocks compared directly, so a
clip whose event fired long after its encoder generation started (or whose
generation predates an RTSP reconnect) failed to find any overlapping
segments and was reported ``NO_FRAMES``/``NO_SEGMENTS`` even though the
frames were on disk the whole time.

These tests exercise ``CsvSegmentIndex.observe_frame``/``select`` directly
-- no ffmpeg process, no encoder session -- to pin the axis-conversion fix
in isolation from the rest of the encode stack (already covered by
``tests/test_worker_segment_encoder.py``).
"""

from __future__ import annotations

from pathlib import Path

from worker.adapters.encode.csv_segment_index import CsvSegmentIndex

_FPS = 5.0


def _write_csv(list_path: Path, rows: tuple[tuple[str, float, float], ...]) -> None:
    list_path.write_text(
        "".join(f"{name},{start:.6f},{end:.6f}\n" for name, start, end in rows),
        encoding="utf-8",
    )


def test_select_maps_a_late_starting_generation_back_to_its_local_axis(
    tmp_path: Path,
) -> None:
    """(a) A generation that only starts after many prior clip finalizes.

    The real-world repro from #165: 25 clip finalizes in one day each closed
    the encoder session, so generation 20's first frame arrived when the
    event clock already read far into the run (~12500s), even though this
    generation's own segments start counting from 0 again.
    """
    list_path = tmp_path / "segments.csv"
    _write_csv(list_path, (("seg-00000.mp4", 0.0, 90.0),))
    index = CsvSegmentIndex(list_path, generation=20)
    index.refresh()

    # This generation's first observed frame is already at event-clock 12500s.
    index.observe_frame(12500.0, _FPS)

    # An event at 12500s +-30s must resolve against local 0..90s, not be
    # compared against the raw (~12500-scale) event-clock numbers.
    selected = index.select(start_time_sec=12470.0, end_time_sec=12530.0)

    assert [segment.path.name for segment in selected] == ["seg-00000.mp4"]


def test_select_finds_nothing_without_axis_conversion_as_a_sanity_check(
    tmp_path: Path,
) -> None:
    """Same setup as above, proving the *raw* (unconverted) numbers never
    overlap -- i.e. this is a real axis mismatch, not a coincidence."""
    list_path = tmp_path / "segments.csv"
    _write_csv(list_path, (("seg-00000.mp4", 0.0, 90.0),))
    index = CsvSegmentIndex(list_path, generation=20)
    index.refresh()

    # No frame observed yet: origin is None, so select() falls back to
    # comparing the query directly against the local segment times -- this
    # is exactly the pre-fix (broken) behaviour, kept here as a control.
    selected = index.select(start_time_sec=12470.0, end_time_sec=12530.0)

    assert selected == ()


def test_select_survives_an_rtsp_reconnect_that_resets_the_event_clock(
    tmp_path: Path,
) -> None:
    """(b) A decode-session (RTSP) reconnect resets the event clock to ~0
    mid-generation, while this encoder generation is untouched by the
    reconnect and keeps accumulating frames. The origin must re-anchor to
    the new (small) event-clock readings without any explicit reconnect
    signal, purely from continuing to observe frames.
    """
    list_path = tmp_path / "segments.csv"
    index = CsvSegmentIndex(list_path, generation=5)

    # Pre-reconnect: 50 frames (10s of local time) at a steady event clock
    # starting at 1000s.
    for i in range(50):
        index.observe_frame(1000.0 + i / _FPS, _FPS)

    # RTSP reconnect: the decode session's clock restarts near 0. This
    # encoder generation was not closed, so frame_count keeps growing.
    for i in range(15):
        index.observe_frame(i / _FPS, _FPS)

    # Segments produced around local time 10..13s (i.e. the 50th-65th
    # frames at 5fps) land on disk as usual.
    _write_csv(list_path, (("seg-00000.mp4", 10.0, 13.0),))
    index.refresh()

    # A new event fires shortly after the reconnect, at the *new* small
    # event-clock reading (~2.0s post-reconnect), windowed +-1s.
    selected = index.select(start_time_sec=1.0, end_time_sec=3.0)

    assert [segment.path.name for segment in selected] == ["seg-00000.mp4"]


def test_select_trims_a_partial_overlap_after_axis_conversion(tmp_path: Path) -> None:
    """(c) Once the window is converted into the local axis, a segment that
    only partially overlaps it must still be selected (the coordinator, not
    this index, is responsible for clamping the *reported duration* -- see
    ``test_worker_clip_recording.py::
    test_finalize_converts_the_duration_window_into_the_sessions_local_axis``
    -- but selection itself must not drop a partially-overlapping segment).
    """
    list_path = tmp_path / "segments.csv"
    _write_csv(
        list_path,
        (
            ("seg-00000.mp4", 0.0, 40.0),
            ("seg-00001.mp4", 40.0, 80.0),
        ),
    )
    index = CsvSegmentIndex(list_path, generation=1)
    index.refresh()
    index.observe_frame(12470.0, _FPS)

    # Window local axis: [0, 60] once converted (event 12470..12530 minus
    # origin 12470). Segment 0 (local 0..40) overlaps only the first 40s;
    # segment 1 (local 40..80) overlaps only the last 20s of the window.
    selected = index.select(start_time_sec=12470.0, end_time_sec=12530.0)

    assert {segment.path.name for segment in selected} == {
        "seg-00000.mp4",
        "seg-00001.mp4",
    }


def test_origin_is_none_until_a_frame_is_observed(tmp_path: Path) -> None:
    index = CsvSegmentIndex(tmp_path / "segments.csv", generation=1)

    assert index.origin_time_sec is None

    index.observe_frame(5.0, _FPS)

    assert index.origin_time_sec == 5.0

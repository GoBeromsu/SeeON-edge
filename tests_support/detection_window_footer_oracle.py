"""Subprocess-only CPython reference for Rust TZif temporal comparisons."""

from __future__ import annotations

import _zoneinfo
import json
import sys
import zoneinfo
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import _common

from worker.domains.detection_window import DetectionWindow

EPOCH = datetime(1970, 1, 1)
SECOND = 1_000_000
DAY = 86_400 * SECOND


def microseconds(delta: timedelta) -> int:
    return (delta.days * 86_400 + delta.seconds) * SECOND + delta.microseconds


def civil(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    assert value.tzinfo is None, "oracle clock fields must be naive civil fields"
    return value


def utc_clock(instant: int) -> list:
    value = EPOCH + timedelta(microseconds=instant)
    return [value.isoformat(timespec="microseconds"), 0, "E", 0]


def instant(text: str) -> int:
    return microseconds(civil(text) - EPOCH)


def wall_delta(instant_us: int, target: zoneinfo.ZoneInfo) -> int:
    utc = EPOCH + timedelta(microseconds=instant_us)
    # Do not call the resulting target datetime's utcoffset(). CPython contains
    # can use its civil time even when that separate operation would fail.
    local = utc.replace(tzinfo=UTC).astimezone(target).replace(tzinfo=None)
    return microseconds(local - utc)


def window_boundaries(window: dict, target: zoneinfo.ZoneInfo, anchor: str) -> list:
    anchor_us = instant(anchor)
    delta = wall_delta(anchor_us, target)
    local = EPOCH + timedelta(microseconds=anchor_us + delta)
    clocks = []
    for text in (window["start"], window["end"]):
        hour, minute = map(int, text.split(":"))
        boundary = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        boundary_us = microseconds(boundary - EPOCH) - delta
        clocks.extend(utc_clock(boundary_us + step) for step in (-1, 0, 1))
    return clocks


def transition_boundaries(target: zoneinfo.ZoneInfo, start: str, end: str) -> list:
    left = instant(start)
    stop = instant(end)
    assert 0 < stop - left <= 400 * DAY, "bounded one-year transition scan"
    before = wall_delta(left, target)
    transitions = []
    while left < stop:
        # An hourly bracket also finds short Jan-1 spill intervals that a
        # day-sized bracket could miss when its two endpoints have equal deltas.
        right = min(left + 3_600 * SECOND, stop)
        after = wall_delta(right, target)
        if before != after:
            lo, hi = left, right
            while hi - lo > 1:
                middle = (lo + hi) // 2
                if wall_delta(middle, target) == before:
                    lo = middle
                else:
                    hi = middle
            transitions.append(hi)
        left, before = right, after
    assert transitions, "transition fixture must exercise an actual civil discontinuity"
    return [
        utc_clock(boundary + step)
        for boundary in transitions
        for step in (-SECOND, -1, 0, 1, SECOND - 1, SECOND)
    ]


def generated_clocks(root: Path, case: dict, target: zoneinfo.ZoneInfo) -> list:
    sampling = case.get("sampling")
    if sampling is None:
        return []
    kind = sampling["kind"]
    if kind == "edge":
        delta = wall_delta(instant(sampling["anchor"]), target)
        if sampling["side"] == "lower":
            bound = datetime.min
        else:
            assert sampling["side"] == "upper"
            bound = datetime.max
        boundary = microseconds(bound - EPOCH) - delta
        return [utc_clock(boundary + step) for step in (-1, 0, 1)]
    clocks = window_boundaries(case["window"], target, sampling["anchor"])
    if kind == "transitions":
        clocks.extend(transition_boundaries(target, sampling["start"], sampling["end"]))
    elif kind == "history":
        with (root / case["window"]["tz"]).open("rb") as stream:
            transitions = _common.load_data(stream)[1]
        assert transitions, "historical fixture needs real TZif transitions"
        last = transitions[-1] * SECOND
        # CPython selects the body/footer using a whole-second UTC cutoff.
        clocks.extend(utc_clock(last + step) for step in (-SECOND, 0, 1, SECOND - 1, SECOND))
    else:
        assert kind == "window", f"unexpected sampling kind: {kind}"
    return clocks


def outcomes(root: Path, case: dict) -> list:
    definition = case["window"]
    target = zoneinfo.ZoneInfo(definition["tz"])
    window = DetectionWindow(**definition)
    rows = []
    seen = set()
    clocks = case.get("clocks", []) + generated_clocks(root, case, target)
    for text, requested_offset, relation, fold in clocks:
        fields = civil(text)
        if relation == "S":
            # This is a SOURCE-clock offset check, including valid gap/fold
            # choices. Never use an out-of-range target offset as a source.
            now = fields.replace(tzinfo=target, fold=fold)
            assert now.tzinfo is zoneinfo.ZoneInfo(definition["tz"])
        else:
            assert relation == "E" and isinstance(requested_offset, int)
            now = fields.replace(tzinfo=timezone(timedelta(seconds=requested_offset)))
        source_delta = now.utcoffset()
        assert source_delta is not None
        source_us = microseconds(source_delta)
        assert source_us % SECOND == 0 and -DAY < source_us < DAY
        offset = source_us // SECOND
        assert requested_offset is None or requested_offset == offset
        key = (text, offset, relation)
        if key in seen:
            continue
        seen.add(key)
        try:
            result = window.contains(now)
        except OverflowError as exc:
            if not case.get("edge", False):
                raise
            # Preserve the actual exception receipt. Only explicit civil-edge
            # probes may compare an observed OverflowError to ClockOutOfRange;
            # ValueError and every other unexpected exception fail loudly.
            result = {"exception": type(exc).__name__, "message": str(exc)}
        rows.append([text, offset, relation, result])
    assert rows and len(rows) <= 48, "bounded informative temporal probes"
    if case.get("sampling", {}).get("kind") in ("window", "transitions", "history"):
        if definition["start"] != definition["end"]:
            booleans = {row[3] for row in rows if isinstance(row[3], bool)}
            assert booleans == {False, True}, "boundary samples must exercise both window outcomes"
    return rows


def main() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:3] != (3, 12, 3):
        raise AssertionError("temporal reference requires CPython 3.12.3")
    if not __debug__:
        raise AssertionError("temporal reference requires enabled assertions")
    if zoneinfo.ZoneInfo is not _zoneinfo.ZoneInfo:
        raise AssertionError("temporal reference requires the CPython C implementation")
    root = Path(sys.argv[1])
    zoneinfo.reset_tzpath([str(root)])
    zoneinfo.ZoneInfo.clear_cache()
    results = [outcomes(root, case) for case in json.loads(sys.argv[2])]
    receipt = json.dumps(results, separators=(",", ":"))
    assert len(receipt.encode()) + 1 < 4096, "compact oracle receipt"
    print(receipt)


if __name__ == "__main__":
    main()

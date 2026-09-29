"""Real DetectionWindow/ZoneInfo differential using one explicit immutable TZif tree.

SEEON_TEST_BED_POLICY_PROBE and SEEON_TEST_ZONEINFO_DIR are separate opt-ins.
Configured invalid inputs fail; neither runtime nor Python consults a fallback
zone database. Python's canonical module is injected with REAL ZoneInfo.from_file
objects made from exactly the supplied files. Hashes are recorded and rechecked;
this harness does not infer an IANA release from filenames (parent pins 2026a).
Microsecond civil datetime, its actual utcoffset and target-relative tzinfo
object identity are the Rust aware-clock API. Identity is input metadata only.

Admission is narrower than Python: ASCII one/two-digit HH:MM, integral-second
UTC offsets, bounded path/key/TZif and, for external clocks only, Jiff's checked
timestamp range. Rejections outside that domain are explicitly not parity.
Source-local nonexistent DST
clocks ARE accepted by both APIs: the differential deliberately preserves Python
same-ZoneInfo-object behavior rather than normalizing the reference to help Rust.
No host local timezone, fixed UTC/KST offsets, alternate database or naive-clock
fallback is used. This is window/scored bed CPU evidence, not geometry/GPU proof.
"""

from __future__ import annotations

import hashlib
import io
import os
import zoneinfo
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest
import test_rust_bed_policy_parity as bed

from worker.domains import detection_window as canonical

bed_probe = bed.bed_probe
_KEYS = ("UTC", "Asia/Seoul", "America/New_York")
_HEADER = "WINDOWPROBE\t1"


@pytest.fixture
def frozen_zones(monkeypatch, record_property):
    configured = os.environ.get("SEEON_TEST_ZONEINFO_DIR")
    if configured is None:
        pytest.skip("explicit zoneinfo tree absent; skip is not timezone qualification")
    if not configured:
        pytest.fail("configured zoneinfo directory is empty", pytrace=False)
    root = Path(configured)
    if not root.is_absolute() or not root.is_dir():
        pytest.fail(
            "zoneinfo directory must explicitly name an existing absolute directory", pytrace=False
        )
    blobs, zones = {}, {}
    for key in _KEYS:
        path = root / key
        if len(os.fsencode(path)) > 4096:
            pytest.fail("configured TZif path exceeds admitted bound", pytrace=False)
        try:
            with path.open("rb") as stream:
                data = stream.read(1_048_577)
        except OSError:
            pytest.fail("configured TZif data missing", pytrace=False)
        if len(data) > 1_048_576:
            pytest.fail("configured TZif data exceeds bound", pytrace=False)
        try:
            zones[key] = ZoneInfo.from_file(io.BytesIO(data), key=key)
        except (OSError, ValueError, EOFError):
            pytest.fail("configured TZif data invalid", pytrace=False)
        blobs[key] = data
        record_property(f"tzif_sha256_{key}", hashlib.sha256(data).hexdigest())

    def supplied_zone(key):
        try:
            return zones[key]
        except KeyError as exc:
            raise ZoneInfoNotFoundError(key) from exc

    # Injection changes data lookup only, not contains/parse/conversion policy.
    monkeypatch.setattr(canonical, "ZoneInfo", supplied_zone)
    yield zones
    for key, before in blobs.items():
        with (root / key).open("rb") as stream:
            assert stream.read(1_048_577) == before, "TZif changed during differential"


@pytest.fixture(params=["from_file", "no_cache"])
def distinct_zones(frozen_zones, request, monkeypatch):
    """Distinct real objects with equal keys and the same verified supplied bytes."""
    root = Path(os.environ["SEEON_TEST_ZONEINFO_DIR"])
    if request.param == "from_file":
        zones = {}
        for key in frozen_zones:
            with (root / key).open("rb") as stream:
                zones[key] = ZoneInfo.from_file(stream, key=key)
        yield zones
        return

    def reject_package_lookup(key):
        raise ZoneInfoNotFoundError(key)

    previous_path = zoneinfo.TZPATH
    with monkeypatch.context() as restricted:
        # no_cache is the actual constructor, not a from_file stand-in. Only the
        # verified tree is searchable, and a missing file cannot use tzdata.
        restricted.setattr("zoneinfo._common.load_tzdata", reject_package_lookup)
        try:
            zoneinfo.reset_tzpath((str(root),))
            yield {key: ZoneInfo.no_cache(key) for key in frozen_zones}
        finally:
            zoneinfo.reset_tzpath(previous_path)


def _at(zones, text, source="UTC", fold=0):
    return datetime.fromisoformat(text).replace(tzinfo=zones[source], fold=fold)


def _raw_row(start, end, zone, wall):
    # Negative transport/config requests need not resolve a valid target.
    return bed._line("W", *map(bed._hex, (start, end, zone)), wall)


def _row(start, end, zone, now):
    window = canonical.DetectionWindow(start, end, zone)
    return _raw_row(
        start,
        end,
        zone,
        bed._wall(now, actual_target_tzinfo=bed._target_tzinfo(window)),
    )


def _compare(probe, cases):
    requests, expected = [_HEADER], [_HEADER]
    membership = []
    for index, (start, end, zone, now) in enumerate(cases):
        requests.append(_row(start, end, zone, now))
        reference = canonical.DetectionWindow(start, end, zone).contains(now)
        membership.append(reference)
        expected.append(bed._line("W", index, int(reference)))
    expected.append(bed._line("END", len(cases)))
    result = bed._invoke(probe, bed._payload(requests))
    assert result.returncode == 0 and result.stderr == b"", result.stderr
    assert result.stdout == bed._payload(expected)
    return membership


@pytest.mark.parametrize("zone", _KEYS)
@pytest.mark.parametrize("start,end", [("9:5", "17:00"), ("09:05", "09:05"), ("21:00", "05:00")])
def test_half_open_equal_overnight_microsecond_edges(bed_probe, frozen_zones, zone, start, end):
    cases = []
    for boundary in (start, end):
        hour, minute = map(int, boundary.split(":"))
        center = _at(frozen_zones, "2026-07-31T00:00:00", zone).replace(hour=hour, minute=minute)
        for delta in (-1, 0, 1):
            now = center + timedelta(microseconds=delta)
            cases.append((start, end, zone, now))
            cases.append((start, end, zone, now.astimezone(frozen_zones["UTC"])))
    _compare(bed_probe, cases)


@pytest.mark.parametrize("target", _KEYS)
def test_cross_offset_conversion_uses_injected_source_offset(bed_probe, frozen_zones, target):
    cases = []
    for source in _KEYS:
        for text in ("1960-01-01T00:00:00", "2024-11-03T01:30:00", "2050-07-01T12:00:00"):
            for fold in (0, 1):
                cases.append(("08:30", "09:00", target, _at(frozen_zones, text, source, fold)))
    _compare(bed_probe, cases)


def test_seoul_historical_half_hour_and_future_posix_rules(bed_probe, frozen_zones):
    cases = [
        ("08:30", "08:31", "Asia/Seoul", _at(frozen_zones, "1960-01-01T00:00:00")),
        ("08:30", "08:31", "Asia/Seoul", _at(frozen_zones, "2026-01-01T00:00:00")),
    ]
    for year in (2050, 2100, 2400, 9998):
        for month in (1, 7):
            now = _at(frozen_zones, f"{year:04}-{month:02}-01T12:00:00")
            cases.extend(
                (start, end, "America/New_York", now)
                for start, end in (("07:00", "08:00"), ("08:00", "09:00"))
            )
    _compare(bed_probe, cases)


def test_new_york_gap_from_real_instants_and_fold_both_occurrences(bed_probe, frozen_zones):
    cases = []
    for text in (
        "2024-03-10T06:59:59.999999",
        "2024-03-10T07:00:00",
        "2024-03-10T07:00:00.000001",
        "2024-11-03T05:59:59.999999",
        "2024-11-03T06:00:00",
        "2024-11-03T06:30:00",
    ):
        now = _at(frozen_zones, text)
        cases.extend(
            (start, end, "America/New_York", now)
            for start, end in (("02:00", "03:00"), ("01:30", "02:00"))
        )
    for fold in (0, 1):
        for text in (
            "2024-11-03T01:29:59.999999",
            "2024-11-03T01:30:00",
            "2024-11-03T01:59:59.999999",
        ):
            now = _at(frozen_zones, text, "America/New_York", fold)
            cases.append(("01:30", "02:00", "America/New_York", now))
    _compare(bed_probe, cases)


@pytest.mark.parametrize("fold", [0, 1])
def test_source_local_gap_preserves_canonical_same_zone_semantics(bed_probe, frozen_zones, fold):
    # Not xfail: Python astimezone(same object) preserves 02:30 in the gap.
    # A Rust implementation that first converts it to an instant differs.
    now = _at(frozen_zones, "2024-03-10T02:30:00", "America/New_York", fold)
    assert _compare(bed_probe, [("02:00", "03:00", "America/New_York", now)]) == [True]


@pytest.mark.parametrize("fold", [0, 1])
def test_equal_key_distinct_gap_clocks_differ_only_by_identity_tag(
    bed_probe, frozen_zones, distinct_zones, fold
):
    zone = "America/New_York"
    same = _at(frozen_zones, "2024-03-10T02:30:00", zone, fold)
    external = _at(distinct_zones, "2024-03-10T02:30:00", zone, fold)
    target = bed._target_tzinfo(canonical.DetectionWindow("02:00", "03:00", zone))
    assert same.tzinfo is target and external.tzinfo is not target
    assert same.tzinfo.key == external.tzinfo.key
    same_wall = bed._wall(same, actual_target_tzinfo=target).split("\t")
    external_wall = bed._wall(external, actual_target_tzinfo=target).split("\t")
    assert same_wall[0] == "S" and external_wall[0] == "E"
    assert same_wall[1:] == external_wall[1:]
    cases = [
        (start, end, zone, now)
        for start, end in (("02:00", "03:00"), ("03:00", "04:00"), ("01:00", "02:00"))
        for now in (same, external)
    ]
    assert _compare(bed_probe, cases) == [True, False, False, fold == 0, False, fold == 1]
    assert external.astimezone(target).replace(tzinfo=None) == datetime(
        2024, 3, 10, 3 if fold == 0 else 1, 30
    )


@pytest.mark.parametrize("fold", [0, 1])
def test_fall_fold_identity_and_actual_offset_when_target_changes_to_utc(
    bed_probe, frozen_zones, distinct_zones, fold
):
    zone = "America/New_York"
    same = _at(frozen_zones, "2024-11-03T01:30:00", zone, fold)
    external = _at(distinct_zones, "2024-11-03T01:30:00", zone, fold)
    target = bed._target_tzinfo(canonical.DetectionWindow("01:00", "02:00", zone))
    walls = [bed._wall(now, actual_target_tzinfo=target).split("\t") for now in (same, external)]
    assert [wall[0] for wall in walls] == ["S", "E"]
    assert walls[0][1:] == walls[1][1:]
    assert int(walls[0][-1]) == (-4 if fold == 0 else -5) * 3600
    cases = [
        (start, end, target_zone, now)
        for now in (same, external)
        for start, end, target_zone in (
            ("01:00", "02:00", zone),
            ("05:00", "06:00", "UTC"),
            ("06:00", "07:00", "UTC"),
        )
    ]
    assert _compare(bed_probe, cases) == [True, fold == 0, fold == 1] * 2


@pytest.mark.parametrize("zone", _KEYS)
def test_supported_python_calendar_edges(bed_probe, frozen_zones, zone):
    _compare(
        bed_probe,
        [
            ("00:00", "23:59", zone, _at(frozen_zones, text))
            for text in (
                "0001-01-01T12:00:00",
                "0001-01-02T00:00:00.000001",
                "9999-12-29T12:00:00.999999",
            )
        ],
    )


@pytest.mark.parametrize("zone", _KEYS)
@pytest.mark.parametrize("text", ["0001-01-01T00:00:00", "9999-12-31T23:59:59.999999"])
def test_same_target_python_calendar_extremes_are_civil(bed_probe, frozen_zones, zone, text):
    now = _at(frozen_zones, text, zone)
    assert _compare(bed_probe, [("23:59", "00:01", zone, now)]) == [True]


@pytest.mark.parametrize(
    "text,zone",
    [
        ("0001-01-01T00:00:00", "America/New_York"),
        ("9999-12-31T23:59:59.999999", "Asia/Seoul"),
    ],
)
def test_external_conversion_outside_python_calendar_is_rejected(
    bed_probe, frozen_zones, text, zone
):
    now = _at(frozen_zones, text)
    request = bed._payload([_HEADER, _row("00:00", "23:59", zone, now)])
    with pytest.raises(OverflowError):
        canonical.DetectionWindow("00:00", "23:59", zone).contains(now)
    result = bed._invoke(bed_probe, request)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


def test_external_jiff_extreme_timestamp_rejection_is_not_python_equivalence(
    bed_probe, frozen_zones, distinct_zones
):
    # Same-target datetime.max succeeds above. This DISTINCT source requires an
    # instant; Jiff reserves upper-bound headroom that Python does not require.
    now = _at(distinct_zones, "9999-12-31T23:59:59.999999")
    assert now.tzinfo is not frozen_zones["UTC"]
    request = bed._payload([_HEADER, _row("23:59", "00:00", "UTC", now)])
    assert canonical.DetectionWindow("23:59", "00:00", "UTC").contains(now)
    result = bed._invoke(bed_probe, request)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
def test_distinct_equal_zone_minimum_with_in_range_intermediate_utc(
    bed_probe, frozen_zones, distinct_zones, zone
):
    now = _at(distinct_zones, "0001-01-01T00:00:00", zone)
    assert now.tzinfo is not frozen_zones[zone]
    assert _compare(bed_probe, [("23:59", "00:01", zone, now)]) == [True]


@pytest.mark.parametrize(
    "text,zone",
    [
        ("0001-01-01T00:00:00", "Asia/Seoul"),
        ("9999-12-31T23:59:59.999999", "America/New_York"),
    ],
)
def test_distinct_equal_zone_can_overflow_before_returning_to_in_range_civil_time(
    bed_probe, frozen_zones, distinct_zones, text, zone
):
    now = _at(distinct_zones, text, zone)
    assert now.tzinfo is not frozen_zones[zone]
    request = bed._payload([_HEADER, _row("23:59", "00:01", zone, now)])
    # Do not use a mathematical round trip as the oracle: real astimezone
    # overflows while subtracting this source offset, before target conversion.
    with pytest.raises(OverflowError):
        canonical.DetectionWindow("23:59", "00:01", zone).contains(now)
    result = bed._invoke(bed_probe, request)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


@pytest.mark.parametrize(
    "time", ["", "24:00", "23:60", "001:00", "1:", "1:2:3", " 1:00", "1:00 ", "-1:00"]
)
def test_invalid_window_times_fail_real_reference_and_probe(bed_probe, frozen_zones, time):
    now = _at(frozen_zones, "2026-01-01T00:00:00")
    request = bed._payload([_HEADER, _row(time, "05:00", "UTC", now)])
    with pytest.raises(canonical.DetectionWindowError):
        canonical.DetectionWindow(time, "05:00", "UTC").contains(now)
    result = bed._invoke(bed_probe, request)
    assert result.returncode == 2 and result.stdout == b""


class _MissingOffset(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize("source_tzinfo", [None, _MissingOffset()], ids=["none", "offset-none"])
def test_naive_clock_rejected_without_machine_local_fallback(
    bed_probe, frozen_zones, source_tzinfo
):
    now = datetime(2026, 1, 1, 12, tzinfo=source_tzinfo)
    without_target = bed._wall(now, actual_target_tzinfo=None).split("\t")
    assert without_target[0] == "E" and without_target[-1] == "-"
    request = bed._payload([_HEADER, _row("09:00", "17:00", "UTC", now)])
    with pytest.raises(canonical.DetectionWindowError, match="timezone-aware"):
        canonical.DetectionWindow("09:00", "17:00", "UTC").contains(now)
    result = bed._invoke(bed_probe, request)
    assert result.returncode == 2 and result.stdout == b""


@pytest.mark.parametrize("relation", ["S", "E"])
@pytest.mark.parametrize(
    "civil,offset",
    [
        ("0000-01-01T00:00:00", "0"),
        ("10000-01-01T00:00:00", "0"),
        ("2026-02-30T00:00:00", "0"),
        ("2026-01-01T00:00:00.000000001", "0"),
        ("2026-01-01T00:00:00", "86400"),
        ("2026-01-01T00:00:00", "-86400"),
        ("2026-01-01T00:00:00", "0.5"),
        ("2026-01-01T00:00:00", "-"),
    ],
)
def test_clock_transport_admission_not_wider_python_parity(
    bed_probe, frozen_zones, relation, civil, offset
):
    row = _raw_row("00:00", "01:00", "UTC", bed._line(relation, bed._hex(civil), offset))
    result = bed._invoke(bed_probe, bed._payload([_HEADER, row]))
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


@pytest.mark.parametrize("transport", ["window", "bed"])
@pytest.mark.parametrize(
    "defect",
    [
        "missing-tag",
        "unknown-tag",
        "lowercase-tag",
        "S-no-civil",
        "E-no-civil",
        "S-no-offset",
        "E-no-offset",
    ],
)
def test_present_wall_requires_explicit_relation_civil_and_offset(
    bed_probe, frozen_zones, transport, defect
):
    civil = bed._hex("2026-01-01T12:00:00.000000")
    wall = {
        "missing-tag": bed._line(civil, 0),
        "unknown-tag": bed._line("?", civil, 0),
        "lowercase-tag": bed._line("s", civil, 0),
        "S-no-civil": "S",
        "E-no-civil": "E",
        "S-no-offset": bed._line("S", civil),
        "E-no-offset": bed._line("E", civil),
    }[defect]
    if transport == "window":
        request = bed._payload([_HEADER, _raw_row("09:00", "17:00", "UTC", wall)])
    else:
        fields = bed._History().line(bed._u(0), actual_target_tzinfo=None).split("\t")
        assert fields[5] == "-"  # The wall field alone is replaced, not decision inputs.
        fields[5:6] = wall.split("\t")
        request = bed._payload([bed._HEADER, bed._new(bed._config()), bed._line(*fields)])
    result = bed._invoke(bed_probe, request)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


def test_absent_wall_is_not_an_aware_clock_for_window_evaluation(bed_probe, frozen_zones):
    request = bed._payload([_HEADER, _raw_row("09:00", "17:00", "UTC", "-")])
    result = bed._invoke(bed_probe, request)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


def test_fractional_offset_python_domain_is_explicitly_wider(bed_probe, frozen_zones):
    now = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(microseconds=500000)))
    row = _raw_row("11:59", "12:00", "UTC", bed._line("E", bed._hex("2026-01-01T12:00:00"), "0.5"))
    assert canonical.DetectionWindow("11:59", "12:00", "UTC").contains(now)
    result = bed._invoke(bed_probe, bed._payload([_HEADER, row]))
    assert result.returncode == 2 and result.stdout == b""


@pytest.mark.parametrize("relation", ["S", "E"])
@pytest.mark.parametrize(
    "defect", ["unset", "empty", "relative", "missing", "invalid", "large", "unknown-zone"]
)
def test_zone_data_failure_never_falls_back(bed_probe, frozen_zones, tmp_path, relation, defect):
    now = _at(frozen_zones, "2026-01-01T12:00:00")
    env = dict(os.environ)
    zone = "UTC"
    if defect == "unset":
        env.pop("SEEON_TEST_ZONEINFO_DIR", None)
    elif defect in ("empty", "relative"):
        env["SEEON_TEST_ZONEINFO_DIR"] = "" if defect == "empty" else "zoneinfo"
    elif defect == "unknown-zone":
        zone = "Unknown/NoSuchZone"
    else:
        env["SEEON_TEST_ZONEINFO_DIR"] = str(tmp_path)
        if defect != "missing":
            data = b"not a TZif" if defect == "invalid" else b"x" * 1_048_577
            (tmp_path / "UTC").write_bytes(data)
            with pytest.raises(ValueError):
                ZoneInfo.from_file(io.BytesIO(data), key="UTC")
    # Raw negative metadata is deliberate: even S cannot bypass construction of
    # a window whose target data is absent/corrupt/unknown. No lookup fallback.
    row = _raw_row(
        "09:00",
        "17:00",
        zone,
        bed._line(relation, bed._hex("2026-01-01T12:00:00.000000"), 0),
    )
    if defect == "unknown-zone":
        with pytest.raises(ZoneInfoNotFoundError):
            canonical.DetectionWindow("09:00", "17:00", zone).contains(now)
    result = bed._invoke(bed_probe, bed._payload([_HEADER, row]), env=env)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


def test_window_transport_requires_calls_and_enforces_call_bound(bed_probe, frozen_zones):
    now = _at(frozen_zones, "2026-01-01T12:00:00")
    for rows in ([_HEADER], [_HEADER, *[_row("09:00", "17:00", "UTC", now)] * 129]):
        result = bed._invoke(bed_probe, bed._payload(rows))
        assert result.returncode == 2 and result.stdout == b""


def test_bed_window_suppression_consumes_raw_onset_and_uses_wall_clock(bed_probe, frozen_zones):
    window = canonical.DetectionWindow("21:00", "05:00", "Asia/Seoul")
    outside = _at(frozen_zones, "2026-07-31T12:00:00", "Asia/Seoul")
    inside = _at(frozen_zones, "2026-07-31T22:00:00", "Asia/Seoul")
    rows = bed._exercise(
        bed_probe,
        [
            bed._u(0, wall=outside),
            bed._u(2, wall=outside),
            bed._u(4, ((7, bed._OUT),), wall=outside),
            bed._u(5, ((7, bed._OUT),), wall=inside),
            bed._u(6, wall=inside),
            bed._u(8, ((7, bed._OUT),), wall=inside),
            bed._u(10, wall=outside),
            bed._u(12, ((7, bed._OUT),), wall=inside),
            None,
            bed._u(14),
            bed._u(16, ((7, bed._OUT),)),
            window,
            bed._u(18, wall=inside),
            bed._u(20, ((7, bed._OUT),), wall=inside),
        ],
        bed._config(night_window=window),
    )
    assert rows[2][1][0].reason == "outside-detection-window"
    # The suppressed raw exit consumed its arm. The return at t=6 earned only
    # one second, so t=8 could not emit; t=10 supplies the full two-second rearm.
    assert not rows[5][0]
    assert rows[7][1][0].triggered is True and len(rows[7][0]) == 1
    assert [index for index, row in enumerate(rows) if row[0]] == [7, 10, 13]


def test_window_suppressed_recovery_preserves_open_episode(bed_probe, frozen_zones):
    window = canonical.DetectionWindow("21:00", "05:00", "Asia/Seoul")
    outside = _at(frozen_zones, "2026-07-31T12:00:00", "Asia/Seoul")
    inside = _at(frozen_zones, "2026-07-31T22:00:00", "Asia/Seoul")
    rows = bed._exercise(
        bed_probe,
        [
            bed._u(0, wall=inside),
            bed._u(2, wall=inside),
            bed._u(4, ((7, bed._OUT),), wall=inside),
            bed._u(6, wall=outside),
            bed._u(8, ((7, bed._OUT),), wall=inside),
            bed._u(10, wall=inside),
            bed._u(12, ((7, bed._OUT),), wall=inside),
        ],
        bed._config(night_window=window),
    )
    assert [index for index, row in enumerate(rows) if row[0]] == [2, 6]
    assert rows[4][1][0].triggered is False and not rows[4][0]
    assert rows[4][1][0].reason == "episode-already-open"


@pytest.mark.parametrize("fold", [0, 1])
def test_bed_reused_gap_clock_recomputes_identity_after_target_changes(
    bed_probe, frozen_zones, fold
):
    gap = canonical.DetectionWindow("02:00", "03:00", "America/New_York")
    utc_outside = canonical.DetectionWindow("02:00", "03:00", "UTC")
    utc_inside = canonical.DetectionWindow(
        "07:00" if fold == 0 else "06:00", "08:00" if fold == 0 else "07:00", "UTC"
    )
    now = _at(frozen_zones, "2024-03-10T02:30:00", "America/New_York", fold)
    # The same datetime is reused, but identity belongs to each input target,
    # including the disabled interval. Never read the monitor's initial config.
    walls = [
        bed._wall(now, actual_target_tzinfo=bed._target_tzinfo(window)).split("\t")
        for window in (gap, utc_outside, gap, None, gap, utc_inside, gap)
    ]
    assert [wall[0] for wall in walls] == ["S", "E", "S", "E", "S", "E", "S"]
    assert all(wall[1:] == walls[0][1:] for wall in walls)
    rows = bed._exercise(
        bed_probe,
        [
            bed._u(0, wall=now),
            bed._u(2, wall=now),
            utc_outside,
            bed._u(4, ((7, bed._OUT),), wall=now),
            gap,
            bed._u(6, wall=now),
            bed._u(8, ((7, bed._OUT),), wall=now),
            None,
            bed._u(10, wall=now),
            bed._u(12, ((7, bed._OUT),), wall=now),
            gap,
            bed._u(14, wall=now),
            bed._u(16, ((7, bed._OUT),), wall=now),
            utc_inside,
            bed._u(18, wall=now),
            bed._u(20, ((7, bed._OUT),), wall=now),
            gap,
            bed._u(22, wall=now),
            bed._u(24, ((7, bed._OUT),), wall=now),
        ],
        bed._config(night_window=gap),
    )
    assert rows[3][1][0].reason == "outside-detection-window"
    assert [index for index, row in enumerate(rows) if row[0]] == [6, 9, 12, 15, 18]


def test_fatal_onset_conserves_prefix_and_exact_release_after_poison(bed_probe, frozen_zones):
    # Rust capacity safety, NOT Python-domain parity: both real Python onsets
    # exist, Rust admits one then returns it on the fatal partial-state error.
    window = canonical.DetectionWindow("21:00", "05:00", "Asia/Seoul")
    outside = _at(frozen_zones, "2026-07-31T12:00:00", "Asia/Seoul")
    inside = _at(frozen_zones, "2026-07-31T22:00:00", "Asia/Seoul")
    people = ((9, bed._IN), (2, bed._IN))
    steps = [
        bed._u(0, people, wall=outside),
        bed._u(2, people, wall=outside),
        bed._u(4, ((9, bed._OUT), (2, bed._OUT)), wall=inside),
    ]
    observed = bed._exercise(bed_probe, steps, bed._config(night_window=window))
    first, second = observed[-1][0]
    history = bed._History()
    requests = [
        bed._HEADER,
        bed._new(bed._config(night_window=window), capacity=1),
        *(history.line(step, actual_target_tzinfo=bed._target_tzinfo(window)) for step in steps),
        bed._line("R", bed._event(first)),
        bed._line("C", "-", bed._num(5.0)),
    ]
    result = bed._invoke(bed_probe, bed._payload(requests))
    assert result.returncode == 2 and result.stderr == b"bed-policy-probe: rejected\n"
    lines = result.stdout.decode("ascii").splitlines()
    assert [line for line in lines if line.startswith("E\t")] == [bed._line("E", bed._event(first))]
    assert bed._hex(second.identity) not in result.stdout.decode("ascii")
    errors = [line.split("\t") for line in lines if line.startswith("X\t")]
    assert [(e[1], e[2], e[-1]) for e in errors] == [("3", "fatal", "1"), ("5", "poisoned", "0")]
    assert (
        bed._line("Z", 9, 0, bed._hex("normal"))
        in lines[lines.index(next(line for line in lines if line.startswith("O\t4\t"))) :]
    )
    assert result.stdout.endswith(b"END\t6\n")

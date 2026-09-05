import csv
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contracts.replay_trace import ReplayRow, ReplayTraceHeader, encode_jsonl
from scripts.qa import golden_from_worksheet, golden_labeller_html, golden_worksheet
from tests_support import episode_metric
from tests_support.golden_episodes import load_golden_episodes

_ROSTER = tuple(f"camera-{index:02d}" for index in range(13))


class _NoFallModel:
    artifact_digest = "synthetic-fall-model"

    def predict(self, features: object) -> object:
        del features
        from worker.interfaces.fall_model import FallV2Probabilities

        return FallV2Probabilities(background=1.0, fall_transition=0.0, fallen=0.0)


def _manifest_rows(per_event: int = 4) -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for camera_index, camera_id in enumerate(_ROSTER):
        for event_type, horizon in (("fall", 126), ("bed-exit", 66)):
            for index in range(per_event):
                detected_at = start + timedelta(
                    days=camera_index, seconds=(index + 1) * horizon
                )
                rows.append(
                    {
                        "camera_id": camera_id,
                        "event_type": event_type,
                        "detected_at": detected_at.isoformat(),
                        "edge_event_id": f"{camera_id}-{event_type}-{index}",
                        "clip_path": f"/synthetic/{camera_id}-{event_type}-{index}.mp4",
                        "clip_started_at": (detected_at - timedelta(seconds=10)).isoformat(),
                        "clip_duration_s": 300,
                    }
                )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _labelled_copy(source: Path, output: Path, labeller: str) -> None:
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    # The browser exporter preserves worksheet columns and appends labeller.
    fields.append("labeller")
    for index, row in enumerate(rows):
        row["label"] = "real" if index == 0 else "false"
        row["labeller"] = labeller
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_complete_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "incidents.jsonl"
    _write_manifest(manifest, _manifest_rows())
    worksheet = tmp_path / "worksheet.csv"
    assert golden_worksheet.build(manifest, worksheet, 100, _ROSTER) == 100
    labelled_one = tmp_path / "reviewer-one.csv"
    labelled_two = tmp_path / "reviewer-two.csv"
    _labelled_copy(worksheet, labelled_one, "reviewer-one")
    _labelled_copy(worksheet, labelled_two, "reviewer-two")
    fixture = tmp_path / "golden.json"
    assert golden_from_worksheet.convert(
        [labelled_one, labelled_two], fixture, manifest, None, 5.0
    ) == 100
    return manifest, worksheet, fixture


def test_golden_toolchain_builds_complete_fixture_and_metric_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, worksheet, fixture = _build_complete_fixture(tmp_path)

    rendered = tmp_path / "labeller.html"
    assert golden_labeller_html.render(worksheet, rendered, "reviewer-one") == 100
    page = rendered.read_text(encoding="utf-8")
    with worksheet.open(encoding="utf-8", newline="") as handle:
        worksheet_rows = list(csv.DictReader(handle))
    rows_match = re.search(r"const rows = (.+);", page)
    assert rows_match is not None
    rendered_rows = json.loads(rows_match.group(1))
    assert {row["episode_id"] for row in rendered_rows} == {
        row["episode_id"] for row in worksheet_rows
    }

    goldens = load_golden_episodes(fixture)
    assert len(goldens) == 100
    assert {episode.event_type for episode in goldens} == {"fall", "bed_exit"}

    traces = tmp_path / "traces"
    traces.mkdir()
    trace = ReplayRow(
        camera_id=_ROSTER[0],
        seq=0,
        pts_ns=0,
        epoch=0,
        source_event="open",
        source="nvdcf",
        tracks=(),
        bed_polygon_id=None,
        bed_polygon=None,
        bed_polygon_image_size=None,
        night_window_active=False,
        frame_width=100,
        frame_height=100,
    )
    (traces / "trace.jsonl").write_text(encode_jsonl(ReplayTraceHeader(), [trace]))
    report = tmp_path / "metric.json"
    monkeypatch.setattr(episode_metric, "_resolve_fall_model", _NoFallModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "episode_metric",
            "--traces",
            str(traces),
            "--golden",
            str(fixture),
            "--out",
            str(report),
        ],
    )

    assert episode_metric.main() == 1
    assert json.loads(report.read_text())["trace_source"] == "nvdcf"


def test_worksheet_requires_five_clip_bearing_candidates_per_camera(tmp_path: Path) -> None:
    manifest = tmp_path / "incidents.jsonl"
    rows = [row for row in _manifest_rows() if row["camera_id"] != _ROSTER[-1]]
    rows.extend(row for row in _manifest_rows() if row["camera_id"] == _ROSTER[-1])
    _write_manifest(manifest, rows[:-4])

    with pytest.raises(ValueError, match="fewer than five candidates: camera-12"):
        golden_worksheet.build(manifest, tmp_path / "worksheet.csv", 100, _ROSTER)


def test_worksheet_selection_is_deterministic_and_conversion_rejects_incomplete_labels(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "incidents.jsonl"
    _write_manifest(manifest, _manifest_rows())
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    assert golden_worksheet.build(manifest, first, 100, _ROSTER) == 100
    assert golden_worksheet.build(manifest, second, 100, _ROSTER) == 100
    assert first.read_text() == second.read_text()

    incomplete = tmp_path / "incomplete.csv"
    _labelled_copy(first, incomplete, "reviewer-one")
    with incomplete.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[-1]["label"] = ""
    with incomplete.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="must contain labelled rows"):
        golden_from_worksheet.convert(
            [incomplete], tmp_path / "golden.json", manifest, None, 5.0
        )

    malformed = tmp_path / "malformed.csv"
    _labelled_copy(first, malformed, "reviewer-one")
    with malformed.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[-1]["label"] = "not-a-label"
    with malformed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="must contain labelled rows"):
        golden_from_worksheet.convert(
            [malformed], tmp_path / "golden.json", manifest, None, 5.0
        )

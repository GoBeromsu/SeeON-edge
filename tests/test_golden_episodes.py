import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.qa.golden_from_worksheet import convert
from scripts.qa.golden_worksheet import build
from tests_support.golden_episodes import load_golden_episodes


def _episode(
    *, labels: dict[str, str], resolved: str, resolution: str, start: int = 1
) -> dict[str, object]:
    return {
        "episode_id": "episode-1",
        "camera_id": "camera-1",
        "event_type": "fall",
        "start_ns": start,
        "end_ns": start + 10,
        "labels": labels,
        "resolved": resolved,
        "resolution": resolution,
        "corroborating_overlap_s": 1,
    }


def _fixture(
    episodes: list[dict[str, object]], labellers: list[str], provisional: bool
) -> dict[str, object]:
    return {
        "schema": "golden-episodes-v1",
        "horizons": {"fall": 120, "bed_exit": 60},
        "corpus_sha256": "0" * 64,
        "labellers": labellers,
        "provisional": provisional,
        "episodes": episodes,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_committed_fixture_is_an_empty_provisional_placeholder() -> None:
    """Labels are owner-supplied; the repo never carries invented episodes."""
    payload = json.loads(Path("tests/fixtures/episodes/golden-v1.json").read_text(encoding="utf-8"))
    assert payload["provisional"] is True
    assert payload["episodes"] == []
    assert load_golden_episodes(Path("tests/fixtures/episodes/golden-v1.json")) == ()


def test_non_provisional_fixture_enforces_quota(tmp_path: Path) -> None:
    episodes = [
        {
            **_episode(
                labels={"a": "real", "b": "real"},
                resolved="real",
                resolution="agree",
                start=1 + index * 1_000_000_000_000,
            ),
            "episode_id": f"episode-{index}",
            "camera_id": f"camera-{index % 20}",
        }
        for index in range(100)
    ]
    path = tmp_path / "golden.json"
    _write(path, _fixture(episodes, ["a", "b"], provisional=False))
    loaded = load_golden_episodes(path)
    assert len(loaded) == 100
    per_camera = {
        camera: sum(item.camera_id == camera for item in loaded)
        for camera in {e.camera_id for e in loaded}
    }
    assert min(per_camera.values()) >= 5
    short = _fixture(episodes[:99], ["a", "b"], provisional=False)
    _write(path, short)
    with pytest.raises(ValueError):
        load_golden_episodes(path)


def test_loader_accepts_agreement_and_third_pass_and_rejects_unresolved_disagreement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.json"
    _write(
        path,
        _fixture(
            [_episode(labels={"a": "real", "b": "real"}, resolved="real", resolution="agree")],
            ["a", "b"],
            False,
        ),
    )
    with pytest.raises(ValueError, match="exactly 100"):
        load_golden_episodes(path)

    third = _fixture(
        [
            _episode(
                labels={"a": "real", "b": "false", "c": "real"},
                resolved="real",
                resolution="third-pass",
            )
        ],
        ["a", "b", "c"],
        True,
    )
    _write(path, third)
    assert load_golden_episodes(path)[0].resolution == "third-pass"
    third["episodes"][0]["resolution"] = "agree"  # type: ignore[index]
    _write(path, third)
    with pytest.raises(ValueError, match="invalid agreeing"):
        load_golden_episodes(path)


def test_loader_requires_provisional_for_one_labeller_and_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    _write(
        path,
        _fixture(
            [_episode(labels={"a": "real"}, resolved="real", resolution="single")], ["a"], False
        ),
    )
    with pytest.raises(ValueError, match="fewer than two"):
        load_golden_episodes(path)
    payload = _fixture(
        [
            _episode(labels={"a": "real"}, resolved="real", resolution="single"),
            _episode(labels={"a": "real"}, resolved="real", resolution="single", start=5),
        ],
        ["a"],
        True,
    )
    payload["episodes"][1]["episode_id"] = "episode-2"  # type: ignore[index]
    _write(path, payload)
    with pytest.raises(ValueError, match="overlapping"):
        load_golden_episodes(path)


def _worksheet(
    path: Path, labeller: str, label: str, *, detected_at: str = "2026-01-01T00:00:10+00:00"
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "camera_id",
                "event_type",
                "detected_at",
                "last_detected_at",
                "label",
                "labeller",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode_id": "e1",
                "camera_id": "cam",
                "event_type": "fall",
                "detected_at": detected_at,
                "last_detected_at": detected_at,
                "label": label,
                "labeller": labeller,
            }
        )


def test_converter_stamps_corpus_and_uses_third_pass_with_earliest_onset(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("corpus", encoding="utf-8")
    first, second, third, output = (
        tmp_path / name for name in ("first.csv", "second.csv", "third.csv", "golden.json")
    )
    _worksheet(first, "a", "real", detected_at="2026-01-01T00:00:20+00:00")
    _worksheet(second, "b", "false", detected_at="2026-01-01T00:00:10+00:00")
    _worksheet(third, "c", "real")
    assert convert([first, second], output, corpus, third, 5) == 1
    payload = json.loads(output.read_text())
    assert payload["corpus_sha256"] == hashlib.sha256(b"corpus").hexdigest()
    assert payload["episodes"][0]["resolution"] == "third-pass"
    assert payload["episodes"][0]["start_ns"] == 1_767_225_605_000_000_000


def test_worksheet_keeps_five_episodes_per_camera(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for camera in ("a", "b"):
        for index in range(6):
            rows.append(
                {
                    "camera_id": camera,
                    "event_type": "fall",
                    "detected_at": f"2026-01-01T00:{index * 3:02d}:00+00:00",
                    "edge_event_id": f"{camera}-{index}",
                    "clip_path": f"/{camera}-{index}.mp4",
                }
            )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "worksheet.csv"
    assert build(manifest, output, limit=20) >= 10
    with output.open(encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    assert all(sum(row["camera_id"] == camera for row in selected) >= 5 for camera in {"a", "b"})

"""Cross-language parity: the C++ perception port vs the C3/C4 Python references.

Given the exact tensor rows / box sequences, when the production C++ code in
``worker/native/deepstream/src/native_perception.cpp`` parses or associates
them, then every output must be bit/value identical to the Python references
(``parity/geometry.py``, ``parity/parse.py``,
``association/legacy_greedy_iou.py``).

All floats cross the process boundary as C99 hexfloats so the comparison is
exact, never a decimal round-trip. Prototype tensors are synthesized from the
same SplitMix64 stream on both sides.

Marked ``integration``: compiles the driver with the host C++ toolchain, so it
is excluded from the default CI selector but REQUIRED locally before touching
the native perception path.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Final

import numpy as np
import pytest

from worker.native.deepstream.association.legacy_greedy_iou import (
    LegacyGreedyBboxIouStrategy,
)
from worker.native.deepstream.parity.geometry import AffineMetadata, letterbox_affine
from worker.native.deepstream.parity.parse import (
    parse_bed_rows,
    parse_person_rows,
    parse_pose_rows,
)
from worker.types.perception_frame import (
    PerceptionFrameIdentity,
    PersonBox,
    PersonBoxChannel,
)

pytestmark = pytest.mark.integration

_SRC: Final = Path(__file__).resolve().parents[1] / "worker" / "native" / "deepstream" / "src"
_GEOMETRIES: Final = ((1080, 1920), (720, 1280), (360, 640), (102, 100), (640, 640), (576, 720))
_MASK: Final = (2**64) - 1


def _splitmix64_signed_units(seed: int, count: int) -> np.ndarray:
    """The driver's SplitMix64 -> [-1, 1) stream, reproduced exactly."""
    state = seed & _MASK
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        state = (state + 0x9E3779B97F4A7C15) & _MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        z = z ^ (z >> 31)
        values[index] = float(z >> 11) * (2.0**-53) * 2.0 - 1.0
    return values


@dataclass(slots=True)  # policy: MUTABLE_OK - owns a live subprocess conversation
class _Driver:
    stdin: IO[str]
    stdout: IO[str]
    lines: list[str] = field(default_factory=list)

    def send(self, line: str) -> None:
        _ = self.stdin.write(line + "\n")
        self.stdin.flush()

    def receive(self) -> str:
        line = self.stdout.readline()
        assert line != "", "driver closed stdout unexpectedly"
        return line.rstrip("\n")


@pytest.fixture(scope="module")
def driver() -> Iterator[_Driver]:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ toolchain unavailable")
    build_dir = Path("/tmp/seeon-native-perception-parity")
    build_dir.mkdir(exist_ok=True)
    binary = build_dir / "native_perception_driver"
    compile_command = (
        compiler,
        "-std=c++20",
        "-O2",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{_SRC}",
        str(_SRC / "native_perception_driver.cpp"),
        str(_SRC / "native_perception.cpp"),
        "-o",
        str(binary),
    )
    subprocess.run(compile_command, check=True)
    process = subprocess.Popen(  # noqa: S603 - repo-owned parity driver
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    driver_value = _Driver(process.stdin, process.stdout)
    yield driver_value
    process.stdin.close()
    _ = process.wait(timeout=10)


def _hex_row(values: np.ndarray) -> str:
    return " ".join(float(value).hex() for value in values)


def _expected_affine_line(affine: AffineMetadata) -> str:
    return (
        f"AFFINE {affine.source_height} {affine.source_width} {affine.tensor_height} "
        f"{affine.tensor_width} {affine.content_height} {affine.content_width} "
        f"{affine.gain.hex()} {affine.box_pad_x} {affine.box_pad_y} "
        f"{affine.keypoint_pad_x.hex()} {affine.keypoint_pad_y.hex()}"
    )


def _normalized_hexfloats(line: str) -> tuple[str, ...]:
    """Normalize hexfloat spellings (0x1.8p+0 vs 0x1.8000p0) via float()."""
    tokens: list[str] = []
    for token in line.split(" "):
        if token.startswith(("0x", "-0x")):
            tokens.append(float.fromhex(token).hex())
        else:
            tokens.append(token)
    return tuple(tokens)


def test_letterbox_affine_matches_reference_when_geometry_varies(driver: _Driver) -> None:
    for height, width in _GEOMETRIES:
        driver.send(f"AFFINE {height} {width}")
        received = _normalized_hexfloats(driver.receive())
        expected = _normalized_hexfloats(_expected_affine_line(letterbox_affine(height, width)))
        assert received == expected, f"affine mismatch for {height}x{width}"


def test_box_and_keypoint_inverses_match_reference_when_padding_is_odd(
    driver: _Driver,
) -> None:
    rng = np.random.default_rng(20260825)
    for height, width in _GEOMETRIES:
        affine = letterbox_affine(height, width)
        for _ in range(25):
            raw = rng.uniform(-8.0, 648.0, size=4)
            x1, _ = sorted((raw[0], raw[2]))
            y_pair = sorted((raw[1], raw[3]))
            driver.send(
                f"INVBOX {height} {width} {float(x1).hex()} {float(y_pair[0]).hex()} "
                f"{float(raw[2]).hex()} {float(y_pair[1]).hex()}"
            )
            expected_box = affine.invert_box((float(x1), y_pair[0], float(raw[2]), y_pair[1]))
            assert driver.receive() == "BOX " + " ".join(str(v) for v in expected_box)
            point = rng.uniform(-8.0, 648.0, size=2)
            driver.send(f"INVKP {height} {width} {float(point[0]).hex()} {float(point[1]).hex()}")
            expected_point = affine.invert_keypoint((float(point[0]), float(point[1])))
            assert driver.receive() == f"KP {expected_point[0]} {expected_point[1]}"


def _pose_rows(rng: np.random.Generator, count: int) -> np.ndarray:
    rows = rng.uniform(0.0, 640.0, size=(count, 57))
    # scores straddle the strict 0.05 threshold, including exactly 0.05
    rows[:, 4] = rng.uniform(0.0, 1.0, size=count)
    rows[0, 4] = 0.05  # strict >: must be dropped by both sides
    if count > 1:
        rows[1, 4] = np.nextafter(0.05, 1.0)  # barely above: must be kept
    rows[:, 5] = 0.0
    return rows


def test_pose_parser_matches_reference_when_scores_straddle_threshold(
    driver: _Driver,
) -> None:
    rng = np.random.default_rng(20260826)
    for height, width in _GEOMETRIES:
        affine = letterbox_affine(height, width)
        rows = _pose_rows(rng, 24)
        parsed = parse_pose_rows(rows, affine)
        driver.send(f"POSE {height} {width} {rows.shape[0]}")
        for row in rows:
            driver.send(_hex_row(row))
        assert driver.receive() == f"POSECOUNT {len(parsed.boxes)}"
        for box, pose in zip(parsed.boxes, parsed.poses, strict=True):
            expected_box = f"POSEBOX {box[0]} {box[1]} {box[2]} {box[3]} {float(box[4]).hex()}"
            assert _normalized_hexfloats(driver.receive()) == _normalized_hexfloats(expected_box)
            expected_points = "POSEKP " + " ".join(
                f"{point.x} {point.y} {float(point.score).hex()}" for point in pose
            )
            assert _normalized_hexfloats(driver.receive()) == _normalized_hexfloats(expected_points)


def test_person_parser_matches_reference_when_classes_and_scores_mix(
    driver: _Driver,
) -> None:
    rng = np.random.default_rng(20260827)
    confidence = 0.25
    for height, width in _GEOMETRIES:
        affine = letterbox_affine(height, width)
        rows = rng.uniform(0.0, 640.0, size=(30, 6))
        rows[:, 4] = rng.uniform(0.0, 1.0, size=30)
        rows[:, 5] = rng.integers(0, 3, size=30).astype(np.float64)
        rows[0, 5] = 0.0
        rows[0, 4] = confidence  # >= keeps (reference uses <, so equal passes)
        parsed = parse_person_rows(rows, affine, confidence=confidence)
        driver.send(f"PERSON {height} {width} {float(confidence).hex()} {rows.shape[0]}")
        for row in rows:
            driver.send(_hex_row(row))
        assert driver.receive() == f"PERSONCOUNT {len(parsed.boxes)}"
        for box in parsed.boxes:
            expected = f"PERSONBOX {box[0]} {box[1]} {box[2]} {box[3]} {float(box[4]).hex()}"
            assert _normalized_hexfloats(driver.receive()) == _normalized_hexfloats(expected)


def test_bed_parser_matches_reference_when_masks_and_polygons_resolve(
    driver: _Driver,
) -> None:
    rng = np.random.default_rng(20260828)
    seed = 424242
    height, width = 360, 640
    affine = letterbox_affine(height, width)
    prototypes = _splitmix64_signed_units(seed, 32 * 160 * 160).reshape(32, 160, 160)
    rows = rng.uniform(0.0, 640.0, size=(8, 38))
    rows[:, 4] = rng.uniform(0.3, 1.0, size=8)
    rows[:, 5] = 59.0
    rows[2, 5] = 0.0  # non-bed class: dropped
    rows[3, 4] = 0.1  # below confidence: dropped
    # make coefficients moderate so masks activate
    rows[:, 6:] = rng.uniform(-0.5, 0.5, size=(8, 32))
    confidence = 0.25
    parsed = parse_bed_rows(rows, prototypes, affine, confidence=confidence)
    driver.send(f"BED {height} {width} {float(confidence).hex()} {seed} 48 {rows.shape[0]}")
    for row in rows:
        driver.send(_hex_row(row))
    assert driver.receive() == f"BEDCOUNT {len(parsed.regions)}"
    for region in parsed.regions:
        expected_box = (
            f"BEDBOX {region.x1} {region.y1} {region.x2} {region.y2} "
            f"{float(region.confidence).hex()}"
        )
        assert _normalized_hexfloats(driver.receive()) == _normalized_hexfloats(expected_box)
        polygon = region.polygon or ()
        expected_poly = f"BEDPOLY {len(polygon)}" + "".join(f" {x} {y}" for x, y in polygon)
        assert driver.receive() == expected_poly


def _observe_reference(
    strategy: LegacyGreedyBboxIouStrategy,
    boxes: tuple[tuple[int, int, int, int, float], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    identity = PerceptionFrameIdentity("boot", "camera-a", 1, 0, None)
    channel = PersonBoxChannel(
        state=_channel_state(len(boxes)),
        boxes=tuple(PersonBox(*box) for box in boxes),
    )
    result = strategy.observe(identity, channel)
    return result.track_ids, result.selected_cue_indexes


def _channel_state(count: int):  # noqa: ANN202
    from worker.types.perception_frame import ChannelState

    return ChannelState.INFERRED if count else ChannelState.INFERRED_EMPTY


def test_association_matches_reference_when_ties_misses_and_resets_occur(
    driver: _Driver,
) -> None:
    rng = np.random.default_rng(20260829)
    strategy = LegacyGreedyBboxIouStrategy(min_iou=0.3, max_misses=3)
    driver.send(f"ASSOC NEW {(0.3).hex()} 3")
    assert driver.receive() == "ACK"

    def observe(boxes: tuple[tuple[int, int, int, int, float], ...]) -> None:
        expected_ids, expected_cues = _observe_reference(strategy, boxes)
        driver.send(f"ASSOC OBSERVE {len(boxes)}")
        for box in boxes:
            driver.send(f"{box[0]} {box[1]} {box[2]} {box[3]} {float(box[4]).hex()}")
        expected = f"ASSOC {len(boxes)}" + "".join(
            f" {track}:{cue}" for track, cue in zip(expected_ids, expected_cues, strict=True)
        )
        assert driver.receive() == expected

    # seed two tracks
    observe(((10, 10, 50, 50, 0.9), (100, 100, 160, 160, 0.8)))
    # exact equal-IoU tie: both existing tracks against one identical box
    observe(((10, 10, 50, 50, 0.9), (100, 100, 160, 160, 0.8), (300, 300, 340, 340, 0.7)))
    # drift + a new box overlapping both track regions equally
    observe(((12, 12, 52, 52, 0.9),))
    # empty observation counts a miss on both sides
    for _ in range(3):
        observe(())
    # eviction boundary crossed (misses > 3): fresh ids mint next
    observe(((10, 10, 50, 50, 0.9),))
    # coast never counts a miss
    strategy.coast()
    driver.send("ASSOC COAST")
    assert driver.receive() == "ACK"
    observe(((10, 10, 50, 50, 0.9),))
    # random fuzzing: 40 frames of jittering boxes
    boxes = np.array([[20, 20, 80, 80], [200, 60, 280, 180], [400, 200, 470, 300]])
    for _ in range(40):
        jitter = rng.integers(-12, 13, size=boxes.shape)
        frame = boxes + jitter
        drop = int(rng.integers(0, 4))
        rows = tuple(
            (int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(conf))
            for index, (row, conf) in enumerate(
                zip(frame, rng.uniform(0.4, 1.0, size=3), strict=True)
            )
            if index != drop
        )
        observe(rows)
    # epoch reset: durable ids restart from zero on both sides
    strategy.reset()
    driver.send("ASSOC RESET")
    assert driver.receive() == "ACK"
    observe(((10, 10, 50, 50, 0.9),))

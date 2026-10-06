"""The private fall bundle gate fails selected tests without the bundle; it never skips.

Each case runs this suite's own conftest.py in a throwaway pytest session whose
rootdir has no models/ unless the case provisions the sentinel, under
``--strict-markers`` with the marker list read from this repository's pyproject.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)

_REPO = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO / "pyproject.toml"
_CONFTEST = _REPO / "tests" / "conftest.py"
_SENTINEL = "models/fall/pose-bbox56-gru/model.onnx"


@pytest.fixture
def gated(pytester: pytest.Pytester) -> pytest.Pytester:
    with _PYPROJECT.open("rb") as handle:
        markers = tomllib.load(handle)["tool"]["pytest"]["ini_options"]["markers"]
    pytester.makeini("[pytest]\nmarkers =\n" + "".join(f"    {m}\n" for m in markers))
    pytester.makeconftest(_CONFTEST.read_text(encoding="utf-8"))
    # The listed basename sits in a package so it imports as
    # bundle_probe.test_fetch_models: this in-process session shares sys.modules
    # with a suite run that has already imported tests/test_fetch_models.py.
    listed = pytester.mkpydir("bundle_probe") / "test_fetch_models.py"
    listed.write_text("def test_listed_module():\n    pass\n", encoding="utf-8")
    pytester.makepyfile(
        test_declares_marker=(
            "import pytest\n\n"
            "pytestmark = pytest.mark.private_bundle\n\n\n"
            "def test_declared_marker():\n    pass\n"
        ),
        test_unrelated="def test_unmarked():\n    pass\n",
    )
    return pytester


def _run(pytester: pytest.Pytester, *args: str) -> pytest.HookRecorder:
    return pytester.inline_run("-p", "no:cacheprovider", "--strict-markers", *args)


def test_selected_private_bundle_test_fails_when_bundle_is_missing(
    gated: pytest.Pytester,
) -> None:
    result = _run(gated)

    passed, skipped, failed = result.listoutcomes()
    assert [report.head_line for report in passed] == ["test_unmarked"]
    assert skipped == []
    assert sorted(report.head_line for report in failed) == [
        "test_declared_marker",
        "test_listed_module",
    ]
    for report in failed:
        assert report.when == "setup"
        assert _SENTINEL in report.longreprtext


def test_selected_private_bundle_test_runs_when_bundle_is_present(
    gated: pytest.Pytester,
) -> None:
    sentinel = gated.path / _SENTINEL
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"onnx")

    _run(gated).assertoutcome(passed=3, skipped=0, failed=0)


def test_private_bundle_is_deselected_by_marker_expression(gated: pytest.Pytester) -> None:
    result = _run(gated, "-m", "not private_bundle")

    result.assertoutcome(passed=1, skipped=0, failed=0)
    deselected = [item.name for call in result.getcalls("pytest_deselected") for item in call.items]
    assert sorted(deselected) == ["test_declared_marker", "test_listed_module"]

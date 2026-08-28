"""The release path: tag -> carrier guard -> GitHub Release -> published images.

Three things must not drift silently.

* The carrier list in ``scripts/release_guard.py`` must stay exhaustive. A new
  workspace member that declares its own version would otherwise be released
  without ever being checked against the tag.
* The release must not be a prerelease. ``edge-images.yml`` is gated on
  ``prerelease == false``; a prerelease release would publish no images while
  looking like a successful release.
* ``release.yml`` must keep its write scope on the one job that creates the
  release, and pin every action to a commit.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

import pytest
import yaml

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW: Final = ".github/workflows/release.yml"
EDGE_IMAGES_WORKFLOW: Final = ".github/workflows/edge-images.yml"
TAG_PREFIX: Final = "seeon-edge-v"
_ACTION_PIN: Final = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "release_guard", REPO_ROOT / "scripts" / "release_guard.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workflow(relative: str) -> dict[str, object]:
    loaded = yaml.load(
        (REPO_ROOT / relative).read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert isinstance(loaded, dict), relative
    return loaded


def _jobs(workflow: dict[str, object]) -> dict[str, dict[str, object]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    return jobs  # type: ignore[return-value]


def _tracked(*patterns: str) -> list[str]:
    listed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *patterns],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert listed, patterns
    return listed


def test_the_guard_knows_every_file_in_the_tree_that_states_a_version() -> None:
    guard = _load_guard()

    declared_toml = set()
    for relative in _tracked("pyproject.toml", "*/pyproject.toml"):
        data = tomllib.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
        if "version" in data.get("project", {}):
            declared_toml.add(relative)

    declared_json = set()
    for relative in _tracked("package.json", "*/package.json"):
        if "version" in json.loads((REPO_ROOT / relative).read_text(encoding="utf-8")):
            declared_json.add(relative)

    # A workspace member that grows a `version` must be added to the guard, or
    # a release can ship with it saying something else.
    assert declared_toml == set(guard.TOML_CARRIERS)
    assert declared_json == set(guard.JSON_CARRIERS)


def test_every_carrier_is_currently_in_lockstep() -> None:
    guard = _load_guard()
    carriers = guard.read_carriers()
    assert len(set(carriers.values())) == 1, carriers
    assert guard.check(carriers, f"{TAG_PREFIX}{carriers['pyproject.toml']}") == []


@pytest.mark.parametrize(
    "tag",
    [
        "seeon-edge-v9.9.9",  # a tag ahead of the tree
        "v0.1.0",  # the prefix dropped
        "seeon-edge-0.1.0",  # the `v` dropped
    ],
)
def test_the_guard_refuses_a_tag_that_disagrees_with_the_tree(tag: str) -> None:
    guard = _load_guard()
    assert guard.check(guard.read_carriers(), tag) != []


def test_the_guard_refuses_carriers_that_disagree_with_each_other() -> None:
    guard = _load_guard()
    carriers = guard.read_carriers()
    carriers["front/package.json"] = "0.2.0"
    assert guard.check(carriers, f"{TAG_PREFIX}0.1.0") != []


def test_release_is_triggered_by_the_tag_and_rehearsed_by_dispatch() -> None:
    workflow = _workflow(RELEASE_WORKFLOW)
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    assert triggers["push"] == {"tags": [f"{TAG_PREFIX}*"]}
    rehearsal = triggers["workflow_dispatch"]["inputs"]["rehearsal"]  # type: ignore[index]
    # Default `true`: a dispatch someone fires without reading the form must
    # rehearse, never release.
    assert rehearsal["type"] == "boolean"
    assert rehearsal["default"] == "true"


def test_release_write_scope_lives_only_on_the_job_that_creates_the_release() -> None:
    workflow = _workflow(RELEASE_WORKFLOW)
    assert workflow["permissions"] == {"contents": "read"}
    jobs = _jobs(workflow)
    assert set(jobs) == {"guard", "release"}
    assert "permissions" not in jobs["guard"]
    assert jobs["release"]["permissions"] == {"actions": "write", "contents": "write"}
    # The release job runs only behind the guard, only on a tag, and never on a
    # rehearsal.
    assert jobs["release"]["needs"] == "guard"
    assert jobs["release"]["if"] == "github.ref_type == 'tag' && inputs.rehearsal != true"


def test_release_workflow_pins_every_action_to_a_commit() -> None:
    workflow = _workflow(RELEASE_WORKFLOW)
    pinned = 0
    for job_name, job in _jobs(workflow).items():
        for step in job["steps"]:  # type: ignore[index]
            if "uses" not in step:
                continue
            pinned += 1
            assert _ACTION_PIN.match(str(step["uses"])), (job_name, step["uses"])
    assert pinned >= 2, pinned


def test_the_release_is_not_a_prerelease_because_that_is_what_publishes_images() -> None:
    source = (REPO_ROOT / RELEASE_WORKFLOW).read_text(encoding="utf-8")
    # Comments explain why the flag is absent, so only executable lines count.
    executable = [
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    ]
    assert any("gh release create" in line for line in executable)
    # `--prerelease` here would leave `edge-images.yml` skipped: a release that
    # publishes nothing while reporting success.
    assert not [line for line in executable if "--prerelease" in line]

    images = _workflow(EDGE_IMAGES_WORKFLOW)
    triggers = images["on"]
    assert isinstance(triggers, dict)
    assert triggers["release"] == {"types": ["published"]}
    assert "github.event.release.prerelease == false" in str(
        _jobs(images)["publish"]["if"]
    )


def test_the_release_job_dispatches_the_image_build_itself() -> None:
    """The `release:` trigger cannot fire for a release this workflow creates.

    GitHub: "With the exception of `workflow_dispatch` and `repository_dispatch`,
    other `GITHUB_TOKEN`-triggered events do not create workflow runs at all."
    The release job therefore dispatches edge-images.yml explicitly, on the tag,
    so the images are built from the released commit. Without this step a
    release reports success and publishes nothing.
    """
    workflow = _workflow(RELEASE_WORKFLOW)
    steps = _jobs(workflow)["release"]["steps"]  # type: ignore[index]
    dispatch = [
        step
        for step in steps
        if "gh workflow run edge-images.yml" in str(step.get("run", ""))
    ]
    assert len(dispatch) == 1, steps
    run = str(dispatch[0]["run"])
    # Dispatched on the tag AND building the tag: the sealed commit, not main.
    assert '--ref "$TAG"' in run
    assert '-f ref="$TAG"' in run
    # Dispatching a workflow needs `actions: write`, and it is granted.
    assert _jobs(workflow)["release"]["permissions"]["actions"] == "write"  # type: ignore[index]


def test_database_format_identity_is_not_treated_as_the_product_version() -> None:
    guard = _load_guard()
    identity = (REPO_ROOT / "front/src/shared/releaseIdentity.ts").read_text(encoding="utf-8")
    # It spells like the tag prefix and is deliberately NOT a carrier: it names
    # the on-disk database format lineage, not the shipped product version.
    assert f"'{TAG_PREFIX}1'" in identity
    assert "front/src/shared/releaseIdentity.ts" not in guard.read_carriers()

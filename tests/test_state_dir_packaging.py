"""Static packaging contract for local runtime paths and central edge state.

The API and worker retain separate image-owned XDG state directories for
non-database runtime files, but all persistent SQLite owners use the one
``edge-state`` volume at ``/var/lib/seeon-state/edge.sqlite3``. The migrator
alone also mounts the two released legacy volumes during one-time import.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from backend.app.edge_db import EDGE_DATABASE_PATH, EDGE_STATE_DIRECTORY
from backend.app.features.cameras.store import CameraRegistryStore
from worker.runtime.config.lkg_store import WorkerConfigLkgStore
from worker.runtime.lease import GPU_LEASE_FILENAME

ROOT = Path(__file__).resolve().parents[1]


class _ComposeLoader(yaml.SafeLoader):
    """compose.edge.yaml uses Docker Compose's `!reset null` tag (clears an
    inherited `build:` key) which plain `yaml.safe_load` doesn't know how to
    construct; only the volume/environment structure matters here, so the
    tagged value itself is discarded."""


_ComposeLoader.add_constructor("!reset", lambda loader, node: None)

EXPECTED_WORKER_STATE_DIR = "/root/.local/state/ml-worker"
EXPECTED_API_STATE_DIR = "/root/.local/state/ml-api"
EXPECTED_EDGE_STATE_DIR = "/var/lib/seeon-state"
EXPECTED_EDGE_DATABASE = "/var/lib/seeon-state/edge.sqlite3"

WORKER_RESOLVER_PATH = ROOT / "worker" / "runtime" / "state_dir.py"
API_RESOLVER_PATH = ROOT / "backend" / "app" / "shared" / "state_dir.py"


def _dockerfile_mkdir_p_args(dockerfile_name: str) -> list[str]:
    """Every whitespace-separated path argument passed to a `RUN mkdir -p ...`
    instruction in the given Dockerfile, in file order."""
    text = (ROOT / dockerfile_name).read_text(encoding="utf-8")
    args: list[str] = []
    instructions: list[str] = []
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        current = f"{current} {stripped}".strip()
        if stripped.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        instructions.append(current)
        current = ""
    if current:
        instructions.append(current)

    for instruction in instructions:
        marker = "mkdir -p "
        idx = instruction.find(marker)
        if idx == -1:
            continue
        args.extend(instruction[idx + len(marker) :].split())
    return args


def _compose_named_volume_target(compose: dict, service: str, volume_name: str) -> str:
    """Return the target for an exact named-volume source on ``service``."""
    entries = compose["services"][service]["volumes"]
    for entry in entries:
        if isinstance(entry, str) and entry.startswith(f"{volume_name}:"):
            _, target = entry.split(":", 1)
            return target
    raise AssertionError(f"no {volume_name!r} volume mounted on service {service!r}")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.load(
        (ROOT / "compose.edge.yaml").read_text(encoding="utf-8"), Loader=_ComposeLoader
    )


# --- Dockerfiles own the path at build time -------------------------------


def test_dockerfile_edge_mkdirs_worker_state_dir() -> None:
    args = _dockerfile_mkdir_p_args("Dockerfile.edge")
    assert EXPECTED_WORKER_STATE_DIR in args, (
        f"Dockerfile.edge must `RUN mkdir -p {EXPECTED_WORKER_STATE_DIR}`; "
        f"found mkdir -p args: {args}"
    )


def test_dockerfile_backend_mkdirs_api_state_dir() -> None:
    args = _dockerfile_mkdir_p_args("Dockerfile.backend")
    assert EXPECTED_API_STATE_DIR in args, (
        f"Dockerfile.backend must `RUN mkdir -p {EXPECTED_API_STATE_DIR}`; "
        f"found mkdir -p args: {args}"
    )


def test_dockerfiles_declare_no_volume_for_state_dir() -> None:
    # Match only an actual `VOLUME` instruction line, not the word appearing
    # in an explanatory comment (e.g. "deliberately no VOLUME here").
    volume_instruction = re.compile(r"^\s*VOLUME\b", re.MULTILINE)
    for name in ("Dockerfile.edge", "Dockerfile.backend"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not volume_instruction.search(text), (
            f"{name} must not declare a VOLUME instruction for the image-owned "
            "state dir (prevents anonymous-volume sprawl and derived-image RUN "
            "neutralization)"
        )


# --- compose owns one central database volume -----------------------------


@pytest.mark.parametrize("service", ["edge-db-migrator", "ml-api"])
def test_compose_mounts_central_state_at_baked_path(compose: dict, service: str) -> None:
    target = _compose_named_volume_target(compose, service, "edge-state")
    assert target == EXPECTED_EDGE_STATE_DIR


def test_compose_no_longer_sets_ml_worker_state_dir_env() -> None:
    text = (ROOT / "compose.edge.yaml").read_text(encoding="utf-8")
    assert "ML_WORKER_STATE_DIR" not in text, (
        "ML_WORKER_STATE_DIR must be removed from compose.edge.yaml — production "
        "path ownership belongs to the Docker image, not an env override"
    )


# --- image-owned local state paths remain stable --------------------------


def test_worker_resolver_matches_dockerfile_and_compose(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    from worker.runtime.state_dir import resolve_state_dir as worker_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    resolved = worker_resolve_state_dir("ml-worker")

    assert str(resolved) == EXPECTED_WORKER_STATE_DIR
    assert str(resolved) in _dockerfile_mkdir_p_args("Dockerfile.edge")
    assert _compose_named_volume_target(compose, "ml-worker", "worker-local-state") == (
        EXPECTED_EDGE_STATE_DIR
    )


def test_api_resolver_matches_dockerfile_and_compose(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    from backend.app.shared.state_dir import resolve_state_dir as api_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    resolved = api_resolve_state_dir("ml-api")

    assert str(resolved) == EXPECTED_API_STATE_DIR
    assert str(resolved) in _dockerfile_mkdir_p_args("Dockerfile.backend")
    assert _compose_named_volume_target(compose, "ml-api", "edge-state") != str(resolved)


# --- resolvers must not read any override env var (env sprawl is the point) ---


def test_worker_resolver_reads_no_environment_override() -> None:
    source = WORKER_RESOLVER_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source and "getenv" not in source, (
        "worker/runtime/state_dir.py must not read any env var override — "
        "the single XDG-style rule has no override, by design"
    )


def test_api_resolver_reads_no_environment_override() -> None:
    source = API_RESOLVER_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source and "getenv" not in source, (
        "backend/app/shared/state_dir.py must not read any env var override — "
        "the single XDG-style rule has no override, by design"
    )


# --- persistent database owners converge on the central path --------------


def test_shared_edge_database_path_matches_compose_mount(compose: dict) -> None:
    assert str(EDGE_STATE_DIRECTORY) == EXPECTED_EDGE_STATE_DIR
    assert str(EDGE_DATABASE_PATH) == EXPECTED_EDGE_DATABASE
    assert _compose_named_volume_target(compose, "ml-api", "edge-state") == str(
        EDGE_DATABASE_PATH.parent
    )
    assert not any(
        str(volume).startswith("edge-state:")
        for volume in compose["services"]["ml-worker"]["volumes"]
    )


def test_worker_config_cache_is_local_and_api_database_is_central() -> None:
    worker_path = WorkerConfigLkgStore().database_path
    api_path = CameraRegistryStore.from_env().path

    assert worker_path.name == "config-lkg"
    assert worker_path.parent == Path.home() / ".local/state/ml-worker"
    assert api_path.name == EDGE_DATABASE_PATH.name


# --- the GPU lease remains in the worker-local state directory -------------


def test_gpu_lease_uses_worker_local_state_not_central_database_volume(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    from worker.runtime.state_dir import resolve_state_dir as worker_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    lease_path = worker_resolve_state_dir("ml-worker") / GPU_LEASE_FILENAME

    assert str(lease_path.parent) == EXPECTED_WORKER_STATE_DIR
    assert _compose_named_volume_target(compose, "ml-worker", "worker-local-state") != str(
        lease_path.parent
    )

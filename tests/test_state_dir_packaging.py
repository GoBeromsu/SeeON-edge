"""Static packaging contract: compose.edge.yaml volume mounts, Dockerfile.edge /
Dockerfile.backend ``RUN mkdir -p`` lines, and the ml-worker/ml-api
``resolve_state_dir`` resolvers must all agree on the same image-owned state
directory (Pre-mortem Scenario 1: a mount/resolver drift makes every redeploy
silently start from empty state).

No live Docker required — everything here is parsed as text/YAML, so this file
carries no ``real_stack`` marker and always runs in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class _ComposeLoader(yaml.SafeLoader):
    """compose.edge.yaml uses Docker Compose's `!reset null` tag (clears an
    inherited `build:` key) which plain `yaml.safe_load` doesn't know how to
    construct; only the volume/environment structure matters here, so the
    tagged value itself is discarded."""


_ComposeLoader.add_constructor("!reset", lambda loader, node: None)

EXPECTED_WORKER_STATE_DIR = "/root/.local/state/ml-worker"
EXPECTED_API_STATE_DIR = "/root/.local/state/ml-api"

WORKER_RESOLVER_PATH = ROOT / "worker" / "runtime" / "state_dir.py"
API_RESOLVER_PATH = ROOT / "backend" / "app" / "shared" / "state_dir.py"

# The GPU lease is worker boot's first state-dir write (Pre-mortem Scenario 1a);
# its filename constant already exists on `worker.runtime.lease` independently
# of the new resolver, so this import is safe regardless of PR A's progress.
from worker.runtime.lease import GPU_LEASE_FILENAME  # noqa: E402


def _dockerfile_mkdir_p_args(dockerfile_name: str) -> list[str]:
    """Every whitespace-separated path argument passed to a `RUN mkdir -p ...`
    line in the given Dockerfile, in file order."""
    text = (ROOT / dockerfile_name).read_text(encoding="utf-8")
    args: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        marker = "mkdir -p "
        idx = stripped.find(marker)
        if idx == -1:
            continue
        args.extend(stripped[idx + len(marker) :].split())
    return args


def _compose_named_volume_target(compose: dict, service: str, volume_name: str) -> str:
    """The container-path target of a named volume (e.g. ``ml-worker-state``)
    mounted on ``service``. Deliberately matches only entries that start with
    the exact named-volume source, since other volume entries in this compose
    file use `${VAR:?message with spaces}` sources that are unsafe to
    naively `.split(":")` in full."""
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


# --- compose attaches named volumes at the image-owned path ---------------


def test_compose_worker_volume_mounts_at_worker_state_dir(compose: dict) -> None:
    target = _compose_named_volume_target(compose, "ml-worker", "ml-worker-state")
    assert target == EXPECTED_WORKER_STATE_DIR


def test_compose_api_volume_mounts_at_api_state_dir(compose: dict) -> None:
    target = _compose_named_volume_target(compose, "ml-api", "ml-api-state")
    assert target == EXPECTED_API_STATE_DIR


def test_compose_no_longer_sets_ml_worker_state_dir_env() -> None:
    text = (ROOT / "compose.edge.yaml").read_text(encoding="utf-8")
    assert "ML_WORKER_STATE_DIR" not in text, (
        "ML_WORKER_STATE_DIR must be removed from compose.edge.yaml — production "
        "path ownership belongs to the Docker image, not an env override"
    )


# --- three-way drift: compose target == Dockerfile mkdir == resolver output ---


def test_worker_resolver_matches_dockerfile_and_compose(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    if not WORKER_RESOLVER_PATH.exists():
        pytest.skip(
            "worker/runtime/state_dir.py does not exist yet (concurrent lane in "
            "progress) — rerun this test once PR A's resolver lands"
        )
    from worker.runtime.state_dir import resolve_state_dir as worker_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    resolved = worker_resolve_state_dir("ml-worker")

    assert str(resolved) == EXPECTED_WORKER_STATE_DIR
    assert str(resolved) in _dockerfile_mkdir_p_args("Dockerfile.edge")
    assert _compose_named_volume_target(compose, "ml-worker", "ml-worker-state") == str(resolved)


def test_api_resolver_matches_dockerfile_and_compose(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    if not API_RESOLVER_PATH.exists():
        pytest.skip(
            "backend/app/shared/state_dir.py does not exist yet (concurrent lane "
            "in progress) — rerun this test once PR A's resolver lands"
        )
    from backend.app.shared.state_dir import resolve_state_dir as api_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    resolved = api_resolve_state_dir("ml-api")

    assert str(resolved) == EXPECTED_API_STATE_DIR
    assert str(resolved) in _dockerfile_mkdir_p_args("Dockerfile.backend")
    assert _compose_named_volume_target(compose, "ml-api", "ml-api-state") == str(resolved)


# --- resolvers must not read any override env var (env sprawl is the point) ---


def test_worker_resolver_reads_no_environment_override() -> None:
    if not WORKER_RESOLVER_PATH.exists():
        pytest.skip("worker/runtime/state_dir.py does not exist yet (concurrent lane in progress)")
    source = WORKER_RESOLVER_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source and "getenv" not in source, (
        "worker/runtime/state_dir.py must not read any env var override — "
        "the single XDG-style rule has no override, by design"
    )


def test_api_resolver_reads_no_environment_override() -> None:
    if not API_RESOLVER_PATH.exists():
        pytest.skip(
            "backend/app/shared/state_dir.py does not exist yet (concurrent lane in progress)"
        )
    source = API_RESOLVER_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source and "getenv" not in source, (
        "backend/app/shared/state_dir.py must not read any env var override — "
        "the single XDG-style rule has no override, by design"
    )


# --- the GPU lease is worker boot's first state-dir write (Scenario 1a) ---


def test_gpu_lease_write_dir_matches_mounted_worker_volume(
    monkeypatch: pytest.MonkeyPatch, compose: dict
) -> None:
    """`GpuLease.acquire()` runs before any other state-dir consumer in worker
    boot. If its write directory isn't the mounted volume path, every redeploy
    silently starts from empty state before any other fix in this plan even
    gets a chance to matter."""
    if not WORKER_RESOLVER_PATH.exists():
        pytest.skip(
            "worker/runtime/state_dir.py does not exist yet (concurrent lane in "
            "progress) — rerun this test once PR A's resolver lands"
        )
    from worker.runtime.state_dir import resolve_state_dir as worker_resolve_state_dir

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/root")))
    lease_path = worker_resolve_state_dir("ml-worker") / GPU_LEASE_FILENAME

    assert str(lease_path.parent) == EXPECTED_WORKER_STATE_DIR
    assert _compose_named_volume_target(compose, "ml-worker", "ml-worker-state") == str(
        lease_path.parent
    )

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final, TypeAlias

import yaml

from worker.domains import DOMAIN_REGISTRY
from worker.runtime.config import load_worker_config

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
EDGE_COMPOSE_FILE: Final = "compose.edge.yaml"
EDGE_IMAGES_WORKFLOW: Final = ".github/workflows/edge-images.yml"
EDGE_PREFLIGHT_SCRIPT: Final = "scripts/edge-preflight/check-nvidia-runtime.sh"
EDGE_SERVICES: Final = {
    "ml-api": "Dockerfile.backend",
    "ml-worker": "Dockerfile.edge",
}
ComposeValue: TypeAlias = (
    str | int | float | bool | None | list["ComposeValue"] | dict[str, "ComposeValue"]
)


class ComposeLoader(yaml.SafeLoader):
    pass


def _compose_tag(
    loader: ComposeLoader,
    tag_suffix: str,
    node: yaml.Node,
) -> ComposeValue:
    del tag_suffix
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return [item for item in loader.construct_sequence(node)]
    if isinstance(node, yaml.MappingNode):
        return {str(key): value for key, value in loader.construct_mapping(node).items()}
    return None


ComposeLoader.add_multi_constructor("!", _compose_tag)


def _compose_services(compose_file: str) -> dict[str, dict[str, ComposeValue]]:
    compose = yaml.load(
        (REPO_ROOT / compose_file).read_text(encoding="utf-8"),
        Loader=ComposeLoader,
    )
    if not isinstance(compose, dict):
        return {}
    services = compose.get("services", {})
    if not isinstance(services, dict):
        return {}
    return {
        str(name): {str(key): value for key, value in service.items()}
        for name, service in services.items()
        if isinstance(service, dict)
    }


def _workflow(path: str) -> dict[str, object]:
    workflow = yaml.load(
        (REPO_ROOT / path).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(workflow, dict)
    return workflow


def _mapping_field(service: dict[str, ComposeValue], field_name: str) -> dict[str, ComposeValue]:
    value = service.get(field_name, {})
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _list_field(service: dict[str, ComposeValue], field_name: str) -> list[ComposeValue]:
    value = service.get(field_name, [])
    if not isinstance(value, list):
        return []
    return list(value)


def test_edge_worker_runtime_status_environment_contract() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    worker_environment = _mapping_field(services["ml-worker"], "environment")

    assert {
        "RELAY_URL",
        "RELAY_TOKEN",
        "API_FACILITY_ID",
        "ML_WORKER_PROFILE",
    } <= set(worker_environment)


def test_edge_compose_contains_exactly_ml_edge_api_and_worker() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)

    assert set(services) == set(EDGE_SERVICES), sorted(services)


def test_edge_services_pin_release_images_with_dockerfiles_for_build() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    expected_image_env = {
        "ml-api": "ML_API_IMAGE",
        "ml-worker": "ML_WORKER_IMAGE",
    }

    failures: list[str] = []
    for service_name, expected_dockerfile in EDGE_SERVICES.items():
        service = services[service_name]
        image = str(service.get("image", ""))
        if expected_image_env[service_name] not in image:
            failures.append(
                f"{service_name} must pin {expected_image_env[service_name]}, image is {image!r}"
            )
        if service.get("pull_policy") != "always":
            failures.append(
                f"{service_name} must set pull_policy: always for pinned release images"
            )
        if not (REPO_ROOT / expected_dockerfile).exists():
            failures.append(f"{expected_dockerfile} must exist for the release image build")

    assert not failures, "\n".join(failures)


def test_edge_image_release_workflow_publishes_digest_env_artifact() -> None:
    workflow_path = REPO_ROOT / EDGE_IMAGES_WORKFLOW
    source = workflow_path.read_text(encoding="utf-8")
    workflow = _workflow(EDGE_IMAGES_WORKFLOW)

    triggers = workflow.get("on")
    assert isinstance(triggers, dict)
    assert "release" in triggers
    assert "workflow_dispatch" in triggers

    permissions = workflow.get("permissions")
    assert isinstance(permissions, dict)
    assert permissions["contents"] == "read"
    assert permissions["packages"] == "write"

    assert "file: Dockerfile.backend" in source
    assert "file: Dockerfile.edge" in source
    assert "docker/build-push-action@v6" in source
    assert "actions/upload-artifact@v4" in source
    assert "steps.build-api.outputs.digest" in source
    assert "steps.build-worker.outputs.digest" in source
    assert "ML_API_IMAGE=" in source
    assert "ML_WORKER_IMAGE=" in source
    assert "edge-ml-image-refs.env" in source


def test_legacy_multi_target_ml_dockerfile_is_removed() -> None:
    assert not (REPO_ROOT / "Dockerfile").exists()


def test_edge_api_host_port_is_loopback_only() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    ports = _list_field(services["ml-api"], "ports")

    assert ports == ["127.0.0.1:${ML_SERVING_PORT:-8000}:8000"]


def test_edge_api_persists_runtime_camera_registry_state() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_volumes = _list_field(services["ml-api"], "volumes")
    compose = yaml.load(
        (REPO_ROOT / EDGE_COMPOSE_FILE).read_text(encoding="utf-8"),
        Loader=ComposeLoader,
    )

    assert "ml-api-state:/root/.local/state/ml-api" in api_volumes
    assert "ml-api-state" in compose.get("volumes", {})


def test_edge_service_builds_do_not_depend_on_dockerfile_targets() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)

    failures: list[str] = []
    for service_name in EDGE_SERVICES:
        build = _mapping_field(services[service_name], "build")
        if "target" in build:
            failures.append(f"{service_name} build target is {build['target']!r}")

    assert not failures, "\n".join(failures)


def test_edge_compose_has_gpu_runtime_preflight_guard() -> None:
    script = REPO_ROOT / EDGE_PREFLIGHT_SCRIPT

    assert script.exists()
    source = script.read_text(encoding="utf-8")
    assert "nvidia-ctk runtime configure --runtime=docker" in source
    assert "docker info" in source
    assert "nvidia-container-runtime" in source
    assert "docker compose pull" not in source
    assert "docker compose up" not in source


def test_api_image_does_not_copy_worker_package() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.backend").read_text(encoding="utf-8")

    assert "COPY edge" not in dockerfile


def test_rtsp_script_surface_uses_reusable_worker_names() -> None:
    scripts_dir = REPO_ROOT / "scripts"
    smoke_script = scripts_dir / "ml-worker-rtsp-smoke.sh"

    assert (scripts_dir / "ml-worker-nursing-home-backend-e2e.sh").exists()
    assert smoke_script.exists()
    assert not (scripts_dir / "ml-edge-four-rtsp-smoke.sh").exists()
    assert not (scripts_dir / "ml-edge-four-mock-rtsp-e2e.sh").exists()
    assert not (scripts_dir / "ml-edge-four-mock-rtsp-ingest-e2e.sh").exists()

    smoke_source = smoke_script.read_text(encoding="utf-8")
    e2e_source = (scripts_dir / "ml-worker-nursing-home-backend-e2e.sh").read_text(encoding="utf-8")
    assert "load_worker_config" in smoke_source
    assert "expected exactly 4 cameras" not in smoke_source
    assert "ml-edge-four" not in smoke_source
    assert "NURSING_HOME_RTSP_URL" in e2e_source
    assert "rtsp-loop-video.sh" not in e2e_source


def test_real_rtsp_bedexit_script_generates_supported_production_config(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "models/fall/lstm-runtime"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.pt").write_bytes(b"placeholder")
    (artifact_dir / "arch.json").write_text(
        '{"hidden":4,"layers":1,"dropout":0.0}',
        encoding="utf-8",
    )
    (artifact_dir / "metadata.yaml").write_text("type: lstm\n", encoding="utf-8")
    config_path = tmp_path / "ml-worker.yaml"
    environment = {
        **os.environ,
        "RELAY_URL": "http://127.0.0.1:8000",
        "RELAY_TOKEN": "relay-token-1",
        "ML_EDGE_E2E_RUNTIME_LSTM_DIR": str(artifact_dir),
        "E2E_FACILITY_ID": "facility-1",
        "E2E_CAMERA_ID": "camera-1",
        "E2E_RESIDENT_ID": "resident-1",
        "BED_EXIT_RTSP_URL": "rtsp://camera-1.local/trackID=2",
    }

    # Execute the script directly, the way an operator does, so its own shebang
    # applies. Handing it to a bare `bash` would bypass the interpreter the
    # script pins for itself: Homebrew bash 5.3.15 deadlocks on any heredoc
    # body over PIPE_BUF (512 bytes), and this file has two -- it never execs
    # the command and hangs with no output. See issue #9.
    #
    # `timeout` and `stdin` are belt-and-braces: a future regression must fail
    # this test rather than hang the whole run, which is what happened before.
    subprocess.run(
        [
            str(REPO_ROOT / "scripts/ml-worker-real-rtsp-bedexit-e2e.sh"),
            "--render-config",
            str(config_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert set(payload["domains"]) <= set(DOMAIN_REGISTRY)

    config = load_worker_config(config_path)

    assert config.enabled_domains == ("bed_exit",)


def test_repo_does_not_own_rtsp_generation_surface() -> None:
    scripts_dir = REPO_ROOT / "scripts"
    assert not (scripts_dir / "rtsp-loop-video.sh").exists()

    active_surface = [
        *[script.read_text(encoding="utf-8") for script in sorted(scripts_dir.glob("*.sh"))],
    ]
    active_text = "\n".join(active_surface)

    forbidden_generation_terms = (
        "rtsp-loop-video",
        "mediamtx",
        "stream_loop",
        "-f rtsp",
        "NURSING_HOME_FALL_VIDEO",
        "RTSP_FIXTURE_IMAGE",
        "RTSP_FIXTURE_WAIT_SECONDS",
        "E2E_RTSP_STREAM_NAME",
        "RTSP_DOCKER_NETWORK",
        "RTSP_NETWORK_ALIAS",
        "RTSP_HOST_PORT",
        "RTSP_DETACH",
        "RTSP_READY_WAIT_SECONDS",
    )
    failures = [term for term in forbidden_generation_terms if term.lower() in active_text.lower()]

    assert not failures, f"RTSP generation terms remain in active surface: {failures}"


# ``test_worker_imports_no_api_or_serving_packages`` was removed with the legacy
# ``edge/`` tree: it read ``edge/runtime/edge_worker.py`` directly. Its intent --
# the worker must not reach into the API or serving packages -- is now enforced
# by the import-linter contract "backend and worker are independent
# (worker→backend relay boundary only)" in ``pyproject.toml`` and by
# ``tests/test_serving_boundary_contract.py``.


def test_edge_compose_keeps_backend_url_on_api_only() -> None:
    """The external-backend URL env belongs to ml-api only; the worker never gets one.

    ml-api derives its Event API / ml-config URLs from a single packaging-time
    base var (API_BACKEND_BASE_URL). The old field-specific names
    (API_BACKEND_EVENTS_URL / API_BACKEND_CONFIG_URL) remain valid overrides
    on ml-api but must stay excluded from the worker too, so a future revert
    to the old scheme can't accidentally leak a backend URL onto the worker.
    """
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_env = _mapping_field(services["ml-api"], "environment")
    worker_env = _mapping_field(services["ml-worker"], "environment")

    assert "API_BACKEND_BASE_URL" in api_env
    assert "API_BACKEND_" + "ALERT_URL" not in api_env
    assert "API_BACKEND_" + "HEARTBEAT_URL" not in api_env
    assert "API_" + "INGEST_" + "KEY_ID" not in api_env
    assert "API_" + "INGEST_" + "SECRET" not in api_env
    assert "API_EDGE_RELAY_TOKEN" in api_env
    assert "API_DASHBOARD_USERNAME" in api_env
    assert "API_DASHBOARD_PASSWORD" in api_env
    assert "API_ALLOW_LEGACY_DASHBOARD_AUTH" not in api_env
    assert "API_DASHBOARD_USERNAME" not in worker_env
    assert "API_DASHBOARD_PASSWORD" not in worker_env
    assert worker_env["RELAY_URL"] == "http://ml-api:8000"
    assert "RELAY_TOKEN" in worker_env
    assert "API_BACKEND_EVENTS_URL" not in worker_env
    assert "API_BACKEND_BASE_URL" not in worker_env
    assert "API_BACKEND_CONFIG_URL" not in worker_env
    assert "API_" + "INGEST_" + "KEY_ID" not in worker_env
    assert "API_" + "INGEST_" + "SECRET" not in worker_env


def test_edge_compose_wires_worker_overlay_stream_to_ml_api_only() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_env = _mapping_field(services["ml-api"], "environment")
    worker_env = _mapping_field(services["ml-worker"], "environment")
    worker_ports = _list_field(services["ml-worker"], "ports")

    assert api_env["ML_API_WORKER_STREAM_ORIGIN"] == (
        "http://ml-worker:${ML_WORKER_DEV_MJPEG_PORT:-8090}"
    )
    assert worker_env["ML_WORKER_DEV_MJPEG"] == "${ML_WORKER_DEV_MJPEG:-true}"
    assert worker_env["ML_WORKER_DEV_MJPEG_HOST"] == "0.0.0.0"
    assert worker_env["ML_WORKER_DEV_MJPEG_PORT"] == "${ML_WORKER_DEV_MJPEG_PORT:-8090}"
    assert worker_ports == []


def test_edge_compose_wires_worker_probe_origin_to_ml_api_only() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_env = _mapping_field(services["ml-api"], "environment")

    assert api_env["ML_API_WORKER_PROBE_ORIGIN"] == (
        "http://ml-worker:${ML_WORKER_DEV_MJPEG_PORT:-8090}"
    )


def test_event_clip_export_gates_are_explicit_off_by_default_allowlist() -> None:
    # Given: the tracked production Compose and its operator env example.
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_env = _mapping_field(services["ml-api"], "environment")
    worker_env = _mapping_field(services["ml-worker"], "environment")
    env_example = (REPO_ROOT / ".env.edge.prod.example").read_text(encoding="utf-8")

    # When: event-clip export environment entries are selected from each service.
    configured = {
        key: value
        for environment in (api_env, worker_env)
        for key, value in environment.items()
        if "EVENT_CLIP_EXPORT_ENABLED" in key
    }

    # Then: only the two owned gates are allowed and both require explicit opt-in.
    assert configured == {
        "ML_API_EVENT_CLIP_EXPORT_ENABLED": "${ML_API_EVENT_CLIP_EXPORT_ENABLED:-0}",
        "ML_WORKER_EVENT_CLIP_EXPORT_ENABLED": ("${ML_WORKER_EVENT_CLIP_EXPORT_ENABLED:-0}"),
    }
    assert "\nML_API_EVENT_CLIP_EXPORT_ENABLED=0\n" in env_example
    assert "\nML_WORKER_EVENT_CLIP_EXPORT_ENABLED=0\n" in env_example

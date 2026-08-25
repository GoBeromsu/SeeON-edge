from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Final, TypeAlias

import yaml

from backend.app.core.config import Settings
from worker.runtime.config.pull_models import BackendWorkerConfigPayload

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
EDGE_COMPOSE_FILE: Final = "compose.edge.yaml"
EDGE_IMAGES_WORKFLOW: Final = ".github/workflows/edge-images.yml"
EDGE_PREFLIGHT_SCRIPT: Final = "scripts/edge-preflight/check-nvidia-runtime.sh"
EDGE_RUNTIME_SERVICES: Final = {
    "ml-api": "Dockerfile.backend",
    "ml-worker": "Dockerfile.edge",
}
#: One-shot operator tool behind the `ops` profile. It never starts with the
#: stack, but it is the only place the documented requeue command can run: the
#: worker image carries no `scripts/ops` and the backend image has no writable
#: worker-state mount.
EDGE_OPS_SERVICES: Final = {"edge-refused-evidence"}

EDGE_SERVICES: Final = {
    "edge-filesystem-inventory",
    "edge-db-migrator",
    *EDGE_OPS_SERVICES,
    *EDGE_RUNTIME_SERVICES,
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

    # Explicit allowlist only: relay secret, profile, and RTSP destination policy.
    # Live clip export is a dashboard runtime setting and must never appear here.
    assert set(worker_environment) == {
        "RELAY_TOKEN",
        "ML_WORKER_PROFILE",
        "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS",
        "ML_RTSP_ALLOW_LOCAL_DESTINATIONS",
    }
    assert worker_environment["ML_RTSP_ALLOW_PRIVATE_DESTINATIONS"] == (
        "${ML_RTSP_ALLOW_PRIVATE_DESTINATIONS:-0}"
    )
    assert worker_environment["ML_RTSP_ALLOW_LOCAL_DESTINATIONS"] == (
        "${ML_RTSP_ALLOW_LOCAL_DESTINATIONS:-0}"
    )
    assert not any("EVENT_CLIP_EXPORT" in key for key in worker_environment)
    assert "API_FACILITY_ID" not in worker_environment


def test_edge_compose_contains_inventory_migrator_api_and_worker() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)

    assert set(services) == EDGE_SERVICES, sorted(services)


def test_edge_db_migrator_owns_schema_lifecycle_before_runtime_start() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    inventory = services["edge-filesystem-inventory"]
    migrator = services["edge-db-migrator"]
    api_depends_on = _mapping_field(services["ml-api"], "depends_on")
    worker_depends_on = _mapping_field(services["ml-worker"], "depends_on")

    assert inventory["restart"] == "no"
    assert inventory["command"] == ["python", "-m", "backend.app.edge_db.inventory"]
    assert "edge-state:/var/lib/seeon-state:ro" in _list_field(inventory, "volumes")
    assert "worker-local-state:/var/lib/seeon-worker-state:ro" in _list_field(
        inventory, "volumes"
    )
    assert any(
        str(volume).endswith(":/var/lib/clip-store:ro")
        for volume in _list_field(inventory, "volumes")
    )
    assert migrator["depends_on"] == {
        "edge-filesystem-inventory": {"condition": "service_completed_successfully"}
    }
    assert migrator["restart"] == "no"
    assert migrator["command"] == [
        "python",
        "-m",
        "backend.app.edge_db.compact_cutover",
        "--source",
        "/var/lib/seeon-state/edge.sqlite3",
        "--live",
        "/var/lib/seeon-state/edge.sqlite3",
        "--archive",
        "/var/lib/seeon-state/edge-v17-archive.sqlite3",
        "--candidate",
        "/var/lib/seeon-state/edge-v18-candidate.sqlite3",
        "--receipt",
        "/var/lib/seeon-state/schema18-cutover-receipts.jsonl",
        "--clip-store",
        "/var/lib/clip-store",
        "--worker-state",
        "/var/lib/seeon-worker-state",
    ]
    assert api_depends_on == {"edge-db-migrator": {"condition": "service_completed_successfully"}}
    assert worker_depends_on == {"ml-api": {"condition": "service_healthy"}}


def test_edge_services_pin_release_images_with_dockerfiles_for_build() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    expected_image_env = {
        "ml-api": "ML_API_IMAGE",
        "ml-worker": "ML_WORKER_IMAGE",
    }

    failures: list[str] = []
    for service_name, expected_dockerfile in EDGE_RUNTIME_SERVICES.items():
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
    migrator = services["edge-db-migrator"]
    assert "ML_API_IMAGE" in str(migrator["image"])
    assert migrator["pull_policy"] == "always"


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

    assert ports == ["127.0.0.1:8000:8000"]


def test_edge_runtime_state_volumes_follow_backend_ownership() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    compose = yaml.load(
        (REPO_ROOT / EDGE_COMPOSE_FILE).read_text(encoding="utf-8"),
        Loader=ComposeLoader,
    )

    for service_name in ("edge-db-migrator", "ml-api"):
        assert "edge-state:/var/lib/seeon-state" in _list_field(services[service_name], "volumes")
    assert "edge-state:/var/lib/seeon-state" not in _list_field(services["ml-worker"], "volumes")
    assert "worker-local-state:/var/lib/seeon-state" in _list_field(
        services["ml-worker"], "volumes"
    )
    assert set(compose.get("volumes", {})) == {
        "edge-state",
        "ml-api-state",
        "ml-worker-state",
        "worker-engine-cache",
        "worker-local-state",
    }
    for runtime_name in EDGE_RUNTIME_SERVICES:
        runtime_volumes = _list_field(services[runtime_name], "volumes")
        assert not any(
            str(volume).startswith(("ml-api-state:", "ml-worker-state:"))
            for volume in runtime_volumes
        )


def test_edge_service_builds_do_not_depend_on_dockerfile_targets() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)

    failures: list[str] = []
    for service_name in EDGE_RUNTIME_SERVICES:
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


def test_real_rtsp_bedexit_script_uses_runtime_authorities_without_static_yaml(
    tmp_path: Path,
) -> None:
    script = REPO_ROOT / "scripts/ml-worker-real-rtsp-bedexit-e2e.sh"
    source = script.read_text(encoding="utf-8")
    environment = {
        **os.environ,
        "RELAY_URL": "http://127.0.0.1:8000",
        "RELAY_TOKEN": "relay-secret-1",
        "E2E_DASHBOARD_USERNAME": "operator-secret-name",
        "E2E_DASHBOARD_PASSWORD": "dashboard-secret-1",
        "E2E_FACILITY_ID": "facility-1",
        "E2E_CAMERA_ID": "camera-1",
        "E2E_RESIDENT_ID": "resident-1",
        "BED_EXIT_RTSP_URL": "rtsp://camera-user:camera-secret@camera-1.local/trackID=2",
        "EVIDENCE_DIR": str(tmp_path / "evidence"),
        "ML_EDGE_E2E_TMP_ROOT": str(tmp_path / "runtime"),
    }

    before = tuple(tmp_path.iterdir())
    completed = subprocess.run(
        [str(script), "--dry-run"],
        check=True,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )

    assert tuple(tmp_path.iterdir()) == before
    assert "/api/v1/cameras" in source
    assert "/api/v1/detection-settings" in source
    assert "/api/v1/relay/config" in source
    assert "--config" not in source
    assert "--render-config" not in source
    assert not re.search(r"(?m)^\s*(?:cameras|models|domains|clip):\s*$", source)
    assert "python -m worker" in source

    assert json.loads(completed.stdout) == {
        "mode": "dry-run",
        "authority": {
            "camera_registry": "http://127.0.0.1:8000/api/v1/cameras",
            "detection_settings": "http://127.0.0.1:8000/api/v1/detection-settings",
            "worker_config": "http://127.0.0.1:8000/api/v1/relay/config",
        },
        "worker_yaml": False,
        "facility_id": "facility-1",
        "camera_id": "camera-1",
        "resident_id": "resident-1",
        "rtsp_url": "rtsp://<redacted>",
        "frames_per_pass": 3200,
        "expected_detection_timezone": "Asia/Seoul",
    }

    output = completed.stdout + completed.stderr
    for secret in (
        "relay-secret-1",
        "operator-secret-name",
        "dashboard-secret-1",
        "camera-user",
        "camera-secret",
        "camera-1.local",
    ):
        assert secret not in output


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
    assert "RELAY_URL" not in worker_env
    assert "RELAY_TOKEN" in worker_env
    assert "API_BACKEND_EVENTS_URL" not in worker_env
    assert "API_BACKEND_BASE_URL" not in worker_env
    assert "API_BACKEND_CONFIG_URL" not in worker_env
    assert "API_" + "INGEST_" + "KEY_ID" not in worker_env
    assert "API_" + "INGEST_" + "SECRET" not in worker_env


def test_internal_origins_and_ports_are_baked_runtime_topology() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_env = _mapping_field(services["ml-api"], "environment")
    worker_env = _mapping_field(services["ml-worker"], "environment")
    worker_ports = _list_field(services["ml-worker"], "ports")

    assert "ML_API_WORKER_STREAM_ORIGIN" not in api_env
    assert "ML_API_WORKER_PROBE_ORIGIN" not in api_env
    assert Settings.model_fields["worker_stream_origin"].default == "http://ml-worker:8090"
    assert Settings.model_fields["worker_probe_origin"].default == "http://ml-worker:8090"
    assert "RELAY_URL" not in worker_env
    assert not {
        "ML_WORKER_DEV_MJPEG",
        "ML_WORKER_DEV_MJPEG_HOST",
        "ML_WORKER_DEV_MJPEG_PORT",
    }.intersection(worker_env)
    assert worker_ports == []

    pulled = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": []}
    ).to_worker_config("http://ml-api:8000", "relay-token")
    assert pulled.relay.url == "http://ml-api:8000"
    assert pulled.dev_mjpeg.enabled is True
    assert pulled.dev_mjpeg.host == "0.0.0.0"
    assert pulled.dev_mjpeg.port == 8090


def test_cpu_intel_and_nvidia_overlays_keep_hardware_opt_in() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    worker = services["ml-worker"]
    assert "deploy" not in worker
    assert "devices" not in worker
    assert "NVIDIA_DRIVER_CAPABILITIES" not in _mapping_field(worker, "environment")

    cpu_worker = _compose_services("compose.edge.cpu.yaml")["ml-worker"]
    assert cpu_worker["deploy"] == "null"

    intel_worker = _compose_services("compose.edge.igpu.yaml")["ml-worker"]
    assert intel_worker["devices"] == ["/dev/dri:/dev/dri"]
    assert _mapping_field(intel_worker, "environment") == {"LIBVA_DRIVER_NAME": "iHD"}

    nvidia_worker = _compose_services("compose.edge.nvidia.yaml")["ml-worker"]
    nvidia_deploy = _mapping_field(nvidia_worker, "deploy")
    assert nvidia_deploy == {
        "resources": {
            "reservations": {
                "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
            }
        }
    }
    assert _mapping_field(nvidia_worker, "environment") == {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video"
    }


def test_edge_compose_exposes_no_static_roster_or_mutable_policy_authority() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    runtime_environment = {
        key
        for service_name in EDGE_RUNTIME_SERVICES
        for key in _mapping_field(services[service_name], "environment")
    }
    worker_command = _list_field(services["ml-worker"], "command")
    example = (REPO_ROOT / ".env.edge.prod.example").read_text(encoding="utf-8")

    assert "--config" not in worker_command
    assert not any(
        marker in key
        for key in runtime_environment
        for marker in ("CAMERA_INVENTORY", "EVENT_CLIP_EXPORT", "MODEL_", "POLICY")
    )
    assert "API_FACILITY_ID=" not in example
    assert "EDGE_CAMERA_CONFIG=" not in example
    assert "EVENT_CLIP_EXPORT_ENABLED=" not in example


def test_clip_export_is_not_managed_by_topology_environment() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_env = _mapping_field(services["ml-api"], "environment")
    worker_env = _mapping_field(services["ml-worker"], "environment")
    env_examples = "\n".join(
        (REPO_ROOT / name).read_text(encoding="utf-8")
        for name in (".env.example", ".env.edge.prod.example")
    )

    assert not any("EVENT_CLIP_EXPORT_ENABLED" in key for key in api_env)
    assert not any("EVENT_CLIP_EXPORT_ENABLED" in key for key in worker_env)
    assert "EVENT_CLIP_EXPORT_ENABLED" not in env_examples

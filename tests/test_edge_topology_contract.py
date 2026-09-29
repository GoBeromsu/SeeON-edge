from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final, TypeAlias

import pytest
import yaml
from pydantic import ValidationError

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

#: One-shot model provisioner on the worker image. Models are a pinned external
#: artifact, never baked into an image or bind-mounted from the checkout: this
#: service fills the `worker-models` volume from the committed manifest and
#: gates ml-worker through depends_on.
EDGE_MODEL_FETCH_SERVICE: Final = "edge-model-fetch"
EDGE_ENGINE_BUILD_SERVICE: Final = "edge-engine-build"
MODELS_VOLUME: Final = "worker-models"

#: Edge product state. edge-db-migrator provisions it before ml-api starts; the
#: SQLite cutover runs from the ops-profile edge-db-cutover one-shot.
EDGE_POSTGRES_SERVICE: Final = "postgres"
EDGE_DB_CUTOVER_SERVICE: Final = "edge-db-cutover"
CI_WORKFLOW: Final = ".github/workflows/ci.yml"
EDGE_DEV_COMPOSE_FILE: Final = "compose.edge.dev.yaml"
EDGE_ENV_EXAMPLE: Final = ".env.edge.prod.example"
#: Compose secret name -> the required env var naming its host file.
POSTGRES_SECRETS: Final = {
    "pg_superuser_password": "PG_SUPERUSER_PASSWORD_HOST_FILE",
    "pg_owner_dsn": "PG_OWNER_DSN_HOST_FILE",
    "pg_runtime_dsn": "PG_RUNTIME_DSN_HOST_FILE",
}

EDGE_SERVICES: Final = {
    "edge-db-migrator",
    EDGE_POSTGRES_SERVICE,
    EDGE_DB_CUTOVER_SERVICE,
    EDGE_MODEL_FETCH_SERVICE,
    EDGE_ENGINE_BUILD_SERVICE,
    *EDGE_OPS_SERVICES,
    *EDGE_RUNTIME_SERVICES,
}
ComposeValue: TypeAlias = (
    str | int | float | bool | list["ComposeValue"] | dict[str, "ComposeValue"] | None
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
        return list(loader.construct_sequence(node))
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


def _compose_top_level(compose_file: str, field_name: str) -> dict[str, ComposeValue]:
    compose = yaml.load(
        (REPO_ROOT / compose_file).read_text(encoding="utf-8"),
        Loader=ComposeLoader,
    )
    assert isinstance(compose, dict)
    value = compose.get(field_name, {})
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


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


def _scalars(value: ComposeValue) -> list[str]:
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in [str(key), *_scalars(item)]]
    if isinstance(value, list):
        return [text for item in value for text in _scalars(item)]
    return [] if value is None else [str(value)]


def test_edge_worker_runtime_status_environment_contract() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    worker_environment = _mapping_field(services["ml-worker"], "environment")

    # Explicit allowlist only: Flow topology, relay secret, and RTSP destination policy.
    # Live clip export is a dashboard runtime setting and must never appear here.
    assert set(worker_environment) == {
        "RELAY_TOKEN",
        "ML_WORKER_PROFILE",
        "ML_WORKER_IMAGE",
        "ML_WORKER_FLOW_INFER_CONFIG",
        "ML_WORKER_FLOW_TRACKER_CONFIG",
        "ML_WORKER_FLOW_TRACKER_LIBRARY",
        "ML_WORKER_FLOW_RECORD_DIR",
        "ML_WORKER_FLOW_RECORD_CACHE_SECONDS",
        "ML_WORKER_FLOW_FRAME_WIDTH",
        "ML_WORKER_FLOW_FRAME_HEIGHT",
        "ML_WORKER_FLOW_BATCH_SIZE",
        "ML_WORKER_CLIP_ANALYSIS_CPU",
        "ML_WORKER_FLOW_ENGINE_PATH",
        "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH",
        "ML_WORKER_FLOW_ONNX_PATH",
        "ML_WORKER_FLOW_PARSER_LIBRARY",
        "NVIDIA_DRIVER_CAPABILITIES",
        "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS",
        "ML_WORKER_EXECUTION_RECORDS_ENABLED",
        "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY",
        "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX",
        "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS",
        "ML_RTSP_ALLOW_LOCAL_DESTINATIONS",
        "WORKER_REPLAY_TRACE_DIR",
    }
    # Replay-trace capture is opt-in: empty by default (disabled).
    assert worker_environment["WORKER_REPLAY_TRACE_DIR"] == "${WORKER_REPLAY_TRACE_DIR:-}"
    assert worker_environment["ML_RTSP_ALLOW_PRIVATE_DESTINATIONS"] == (
        "${ML_RTSP_ALLOW_PRIVATE_DESTINATIONS:-0}"
    )
    # Execution-record export is a seam: OFF by default, and the sizes have NO
    # compose default so an enabled-but-unsized deployment refuses to start.
    assert worker_environment["ML_WORKER_EXECUTION_RECORDS_ENABLED"] == (
        "${ML_WORKER_EXECUTION_RECORDS_ENABLED:-0}"
    )
    for key in (
        "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY",
        "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX",
        "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS",
    ):
        assert worker_environment[key] == "${" + key + ":-}"
    assert worker_environment["ML_RTSP_ALLOW_LOCAL_DESTINATIONS"] == (
        "${ML_RTSP_ALLOW_LOCAL_DESTINATIONS:-0}"
    )
    assert worker_environment["ML_WORKER_PROFILE"] == "flow"
    assert worker_environment["ML_WORKER_CLIP_ANALYSIS_CPU"] == ("${ML_WORKER_CLIP_ANALYSIS_CPU:-}")
    assert not any("EVENT_CLIP_EXPORT" in key for key in worker_environment)
    assert "API_FACILITY_ID" not in worker_environment


def test_edge_compose_contains_migrator_api_and_worker() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)

    assert set(services) == EDGE_SERVICES, sorted(services)


def test_edge_db_migrator_provisions_postgres_before_ml_api() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    migrator = services["edge-db-migrator"]
    command = _list_field(migrator, "command")
    healthcheck = _mapping_field(services[EDGE_POSTGRES_SERVICE], "healthcheck")
    api_environment = _mapping_field(services["ml-api"], "environment")
    api_depends_on = _mapping_field(services["ml-api"], "depends_on")
    worker_depends_on = _mapping_field(services["ml-worker"], "depends_on")

    # The job waits for a server that accepts connections, then gates ml-api.
    assert _mapping_field(migrator, "depends_on") == {
        EDGE_POSTGRES_SERVICE: {"condition": "service_healthy"}
    }
    assert _list_field(healthcheck, "test")[:2] == ["CMD", "pg_isready"]
    assert migrator["restart"] == "no"
    assert "profiles" not in migrator, "must run on every `up`, not behind an opt-in profile"
    # Idempotent: a rerun verifies the schema, runtime role, runtime password and
    # authority file and changes none of them.
    assert command == [
        "python",
        "-m",
        "backend.app.edge_db.migration",
        "provision",
        "--owner-dsn-file",
        "/run/secrets/pg_owner_dsn",
        "--schema",
        "seeon_edge",
        "--runtime-dsn-file",
        "/run/secrets/pg_runtime_dsn",
        "--authority-file",
        "/run/seeon-authority/authority.json",
    ]
    # Provisioning is PostgreSQL-only; it never opens the SQLite state.
    assert _list_field(migrator, "volumes") == ["edge-pg-authority:/run/seeon-authority"]
    assert api_depends_on == {"edge-db-migrator": {"condition": "service_completed_successfully"}}
    assert api_environment["API_POSTGRES_SCHEMA"] == command[command.index("--schema") + 1]
    # The worker waits on both the healthy API and a verified models volume.
    assert worker_depends_on == {
        "ml-api": {"condition": "service_healthy"},
        EDGE_MODEL_FETCH_SERVICE: {"condition": "service_completed_successfully"},
        EDGE_ENGINE_BUILD_SERVICE: {"condition": "service_completed_successfully"},
    }


def test_edge_postgres_publishes_no_port() -> None:
    """The superuser password is the database's only barrier, so the server is
    reachable only from the private Compose network."""
    services = _compose_services(EDGE_COMPOSE_FILE)

    assert not {"ports", "expose", "network_mode"}.intersection(services[EDGE_POSTGRES_SERVICE])
    for service_name, service in services.items():
        targets = {
            str(port["target"])
            if isinstance(port, dict)
            else str(port).split("/")[0].rsplit(":", 1)[-1]
            for port in _list_field(service, "ports")
        }
        assert "5432" not in targets, f"{service_name} publishes the PostgreSQL port"


def test_edge_postgres_image_is_the_ci_digest() -> None:
    """The edge server is the exact image the PostgreSQL tests ran against."""
    image = _compose_services(EDGE_COMPOSE_FILE)[EDGE_POSTGRES_SERVICE]["image"]
    jobs = _workflow(CI_WORKFLOW)["jobs"]
    assert isinstance(jobs, dict)

    ci_images: set[str] = set()
    for job in jobs.values():
        assert isinstance(job, dict)
        job_services = job.get("services", {})
        if "postgres" in job_services:
            ci_images.add(job_services["postgres"]["image"])

    assert re.fullmatch(r"postgres@sha256:[0-9a-f]{64}", str(image))
    assert ci_images == {image}


def test_edge_ml_api_holds_only_the_runtime_dsn() -> None:
    """A compromised API process must not get DDL or superuser rights: only the
    one-shots hold the owner DSN and only postgres reads the superuser password."""
    services = _compose_services(EDGE_COMPOSE_FILE)
    api = services["ml-api"]
    holders = {
        secret_name: {
            service_name
            for service_name, service in services.items()
            if secret_name in _list_field(service, "secrets")
        }
        for secret_name in POSTGRES_SECRETS
    }

    assert _list_field(api, "secrets") == ["pg_runtime_dsn"]
    assert _mapping_field(api, "environment")["API_POSTGRES_DSN_FILE"] == (
        "/run/secrets/pg_runtime_dsn"
    )
    assert holders == {
        "pg_superuser_password": {EDGE_POSTGRES_SERVICE},
        "pg_owner_dsn": {"edge-db-migrator", EDGE_DB_CUTOVER_SERVICE},
        "pg_runtime_dsn": {"edge-db-migrator", "ml-api"},
    }


def test_edge_owner_dsn_reaches_only_the_postgres_one_shots() -> None:
    """The owner DSN can reach a service as a secret, a bind mount of its host
    file or an environment value. Only the two one-shots may name it, and the
    only one a plain `docker compose up` starts is the migrator, which exits."""
    references = ("pg_owner_dsn", POSTGRES_SECRETS["pg_owner_dsn"])
    holders: set[str] = set()
    for compose_file in (EDGE_COMPOSE_FILE, EDGE_DEV_COMPOSE_FILE):
        for service_name, service in _compose_services(compose_file).items():
            if any(ref in text for text in _scalars(service) for ref in references):
                holders.add(service_name)
    services = _compose_services(EDGE_COMPOSE_FILE)
    default_holders = {name for name in holders if not services[name].get("profiles")}

    assert holders == {"edge-db-migrator", EDGE_DB_CUTOVER_SERVICE}
    assert default_holders == {"edge-db-migrator"}
    assert services["edge-db-migrator"]["restart"] == "no"


def test_edge_postgres_credentials_are_required_secret_files() -> None:
    """No credential has a default path or an inline value: a missing host file
    stops the render instead of mounting a stale one, and nothing secret shows in
    `docker inspect` or `docker compose config`."""
    services = _compose_services(EDGE_COMPOSE_FILE)
    secrets = _compose_top_level(EDGE_COMPOSE_FILE, "secrets")

    assert set(secrets) == set(POSTGRES_SECRETS)
    for secret_name, path_variable in POSTGRES_SECRETS.items():
        assert secrets[secret_name] == {"file": secrets[secret_name]["file"]}, secret_name
        assert re.fullmatch(rf"\$\{{{path_variable}:\?[^}}]+\}}", str(secrets[secret_name]["file"]))
    assert _mapping_field(services[EDGE_POSTGRES_SERVICE], "environment") == {
        "POSTGRES_PASSWORD_FILE": "/run/secrets/pg_superuser_password"
    }
    for service_name, service in services.items():
        environment = _mapping_field(service, "environment")
        assert not {"POSTGRES_PASSWORD", "PGPASSWORD"}.intersection(environment), service_name
        for key, value in environment.items():
            assert not re.search(r"postgres(ql)?://|password=", str(value), re.IGNORECASE), (
                f"{service_name} {key} carries an inline credential"
            )


def test_edge_postgres_state_lives_in_named_volumes() -> None:
    """The database and its authority file survive `down` and container
    replacement, and ml-api cannot rewrite its own fence token."""
    services = _compose_services(EDGE_COMPOSE_FILE)
    volumes = _compose_top_level(EDGE_COMPOSE_FILE, "volumes")
    command = _list_field(services["edge-db-migrator"], "command")
    authority_file = str(command[command.index("--authority-file") + 1])

    def mounted_by(volume: str) -> set[str]:
        return {
            service_name
            for service_name, service in services.items()
            if volume in _list_field(service, "volumes")
        }

    assert _list_field(services[EDGE_POSTGRES_SERVICE], "volumes") == [
        "edge-pgdata:/var/lib/postgresql"
    ]
    assert mounted_by("edge-pg-authority:/run/seeon-authority") == {
        "edge-db-migrator",
        EDGE_DB_CUTOVER_SERVICE,
    }
    assert mounted_by("edge-pg-authority:/run/seeon-authority:ro") == {"ml-api"}
    assert authority_file.startswith("/run/seeon-authority/")
    assert (
        _mapping_field(services["ml-api"], "environment")["API_POSTGRES_AUTHORITY_FILE"]
        == authority_file
    )
    # Project-scoped named volumes: not external, no fixed name, default driver.
    for volume_name in ("edge-pgdata", "edge-pg-authority", "edge-migration"):
        assert volumes[volume_name] == {}, volume_name


def test_edge_db_cutover_is_an_ops_one_shot() -> None:
    """Each cutover step is `docker compose run --rm edge-db-cutover <step>`. It
    never starts with the stack, so it never holds the SQLite deployment lock or
    repeats a transfer on restart."""
    cutover = _compose_services(EDGE_COMPOSE_FILE)[EDGE_DB_CUTOVER_SERVICE]

    assert cutover["profiles"] == ["ops"]
    assert cutover["restart"] == "no"
    assert not {"ports", "expose", "healthcheck", "command"}.intersection(cutover)
    assert cutover["entrypoint"] == ["python", "-m", "backend.app.edge_db.migration"]
    assert _list_field(cutover, "secrets") == ["pg_owner_dsn"]
    # edge-state is read-write: export takes deployment.lock beside the database.
    assert _list_field(cutover, "volumes") == [
        "edge-state:/var/lib/seeon-state",
        "worker-local-state:/var/lib/seeon-worker-state:ro",
        "edge-migration:/var/lib/seeon-migration",
        "edge-pg-authority:/run/seeon-authority",
    ]
    assert _mapping_field(cutover, "depends_on") == {
        EDGE_POSTGRES_SERVICE: {"condition": "service_healthy"}
    }


def test_edge_dev_overlay_never_pulls_the_api_image() -> None:
    """The dev overlay's ML_API_IMAGE is a local build tag; pulling it fails."""
    base = _compose_services(EDGE_COMPOSE_FILE)
    overlay = _compose_services(EDGE_DEV_COMPOSE_FILE)
    api_image_services = {
        name for name, service in base.items() if "${ML_API_IMAGE" in str(service.get("image"))
    }

    assert api_image_services >= {
        "ml-api",
        "edge-db-migrator",
        EDGE_DB_CUTOVER_SERVICE,
        *EDGE_OPS_SERVICES,
    }
    for service_name in sorted(api_image_services):
        assert overlay.get(service_name, {}).get("pull_policy") == "never", service_name


def test_edge_env_example_declares_every_postgres_secret_path() -> None:
    """CI renders compose.edge.yaml with this example; a missing path fails it."""
    lines = (REPO_ROOT / EDGE_ENV_EXAMPLE).read_text(encoding="utf-8").splitlines()
    entries = {
        key: value
        for key, _, value in (line.partition("=") for line in lines)
        if key and not key.startswith("#")
    }

    for path_variable in POSTGRES_SECRETS.values():
        # An absolute path and nothing else: no DSN, host or password fits.
        assert re.fullmatch(r"/[\w./-]+", entries.get(path_variable, "")), path_variable


def test_edge_model_fetch_owns_the_models_volume_before_worker_start() -> None:
    """Owner decision 2026-08-28: models stay out of the images and are fetched
    at a pinned revision by a worker-side one-shot, mirroring edge-db-migrator.
    The backend never reads /app/models (Lane D), so only the worker mounts it."""
    services = _compose_services(EDGE_COMPOSE_FILE)
    fetch = services[EDGE_MODEL_FETCH_SERVICE]

    assert "ML_WORKER_IMAGE" in str(fetch["image"]), "same image as the runtime it prepares"
    assert fetch["pull_policy"] == "always"
    assert fetch["restart"] == "no"
    assert "profiles" not in fetch, "must run on every `up`, not behind an opt-in profile"
    assert fetch["command"] == [
        "python",
        "-m",
        "worker.tools.fetch_models",
        "--dest",
        "/models",
    ]
    assert _list_field(fetch, "volumes") == [f"{MODELS_VOLUME}:/models:rw"]
    assert set(_mapping_field(fetch, "environment")) == {"HF_TOKEN"}, (
        "only the optional HF token crosses into the fetcher; no relay secret, no profile"
    )
    assert _mapping_field(fetch, "environment")["HF_TOKEN"] == "${HF_TOKEN:-}"
    assert "depends_on" not in fetch, "model provisioning is independent of the database cutover"

    worker_volumes = _list_field(services["ml-worker"], "volumes")
    assert f"{MODELS_VOLUME}:/models:ro" in worker_volumes
    # The selection document is an opt-in overlay: binding it unconditionally
    # makes Docker create a directory on a host without one and the worker
    # refuses to boot (#498).
    assert not any("model-selection.json" in str(volume) for volume in worker_volumes)
    overlay = yaml.safe_load(Path("compose.edge.model-selection.yaml").read_text(encoding="utf-8"))
    for service_name in ("edge-model-fetch", "ml-worker"):
        assert _list_field(overlay["services"][service_name], "volumes") == [
            "/deployment/model-selection.json:/app/model-selection.json:ro"
        ]
    assert not any(str(volume).startswith("./models") for volume in worker_volumes)
    engine_build = services[EDGE_ENGINE_BUILD_SERVICE]
    assert _list_field(engine_build, "volumes") == [
        f"{MODELS_VOLUME}:/app/models:ro",
        "worker-engine-cache:/var/cache/seeon/tensorrt:rw",
    ]
    # The engine build reads provisioned models but never writes them. Every
    # other service stays off the models volume.
    excluded_services = {"edge-model-fetch", "ml-worker", EDGE_ENGINE_BUILD_SERVICE}
    for service_name in sorted(set(services) - excluded_services):
        volumes = _list_field(services[service_name], "volumes")
        assert not any("/models" in str(volume) for volume in volumes), (
            f"{service_name} must not mount the models volume"
        )
        assert "HF_TOKEN" not in _mapping_field(services[service_name], "environment")
    assert "HF_TOKEN" not in _mapping_field(services["ml-worker"], "environment")


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
    for service_name in ("edge-db-migrator", EDGE_DB_CUTOVER_SERVICE):
        one_shot = services[service_name]
        assert "ML_API_IMAGE" in str(one_shot["image"]), service_name
        assert one_shot["pull_policy"] == "always", service_name


def test_edge_image_release_workflow_publishes_digest_env_artifact() -> None:
    workflow_path = REPO_ROOT / EDGE_IMAGES_WORKFLOW
    source = workflow_path.read_text(encoding="utf-8")
    workflow = _workflow(EDGE_IMAGES_WORKFLOW)

    triggers = workflow.get("on")
    assert isinstance(triggers, dict)
    assert "release" in triggers
    assert "workflow_dispatch" in triggers
    # One build path: PRs and main pushes build both images in this workflow
    # too (the separate edge-worker-image.yml gate is folded in).
    assert "pull_request" in triggers
    assert triggers["push"] == {"branches": ["main"]}

    # `packages: write` is granted on the publishing job, not workflow-wide, so
    # a job added to this file later starts read-only. The workflow runs on
    # `pull_request` and `permissions:` takes no expression, so the grant cannot
    # be event-scoped; tests/test_public_repository_privacy.py is what asserts
    # every step able to spend the token stays gated on PUSH_IMAGES.
    permissions = workflow.get("permissions")
    assert isinstance(permissions, dict)
    assert permissions == {"contents": "read"}
    assert workflow["jobs"]["publish"]["permissions"] == {
        "contents": "read",
        "packages": "write",
    }

    assert "file: Dockerfile.backend" in source
    assert "file: Dockerfile.edge" in source
    # Actions are pinned to immutable commits (a tag is a pointer the upstream
    # owner can move), with the released version kept in a trailing comment.
    assert "docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8 # v6.19.2" in source
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2" in source
    assert "steps.build-api.outputs.digest" in source
    assert "steps.build-worker.outputs.digest" in source
    assert "ML_API_IMAGE=" in source
    assert "ML_WORKER_IMAGE=" in source
    assert "edge-ml-image-refs.env" in source
    # Digests are job outputs so a downstream job can pin `@sha256:` without
    # parsing the artifact. They come from the resolve step rather than straight
    # off a build, because a release may REUSE an already-published digest
    # instead of building (docs/runbooks/edge-image-publish.md, "Per-image
    # isolation at release time"). The build digests are still what that step
    # consumes on the build path -- asserted above.
    outputs = workflow["jobs"]["publish"]["outputs"]
    assert outputs["ml-api-digest"] == "${{ steps.digests.outputs.ml-api }}"
    assert outputs["ml-worker-digest"] == "${{ steps.digests.outputs.ml-worker }}"
    # Whether each image was built or reused travels with the digests, so a
    # reviewer never has to infer it from the digest alone.
    assert outputs["ml-api-origin"] == "${{ steps.digests.outputs.ml-api-origin }}"
    assert outputs["ml-worker-origin"] == "${{ steps.digests.outputs.ml-worker-origin }}"


def test_edge_image_workflow_never_pushes_from_pull_requests() -> None:
    # Publish policy is unchanged: release/dispatch (and now main) push;
    # pull requests never log in, push, or upload a pinnable artifact.
    source = (REPO_ROOT / EDGE_IMAGES_WORKFLOW).read_text(encoding="utf-8")
    workflow = _workflow(EDGE_IMAGES_WORKFLOW)
    job = workflow["jobs"]["publish"]

    assert job["env"]["PUSH_IMAGES"] == "${{ github.event_name != 'pull_request' }}"
    steps = {step.get("name"): step for step in job["steps"]}
    assert steps["Login to GitHub Container Registry"]["if"] == "env.PUSH_IMAGES == 'true'"
    assert steps["Upload edge image refs"]["if"] == "env.PUSH_IMAGES == 'true'"
    for name in ("Build and push ml-api image", "Build and push ml-worker image"):
        assert steps[name]["with"]["push"] == "${{ env.PUSH_IMAGES == 'true' }}"
    # Staging tag on main pushes only; the full-SHA tag is always present.
    assert 'if [ "${GITHUB_EVENT_NAME}" = "push" ]' in source
    assert "main-$SHORT_SHA" in source
    assert "$IMAGE_NAMESPACE/$image:$DEPLOY_SHA" in source


def test_edge_worker_boot_smoke_runs_on_the_single_build() -> None:
    """The shipped image is Dockerfile.edge's final stage and boots via carrier."""
    source = (REPO_ROOT / EDGE_IMAGES_WORKFLOW).read_text(encoding="utf-8")
    workflow = _workflow(EDGE_IMAGES_WORKFLOW)
    steps = workflow["jobs"]["publish"]["steps"]
    worker_step = next(s for s in steps if s.get("name") == "Build and push ml-worker image")
    carrier_smoke = next(
        s
        for s in steps
        if "docker load --input /tmp/ml-worker-runtime.tar" in str(s.get("run", ""))
    )
    pull_smoke = next(s for s in steps if "docker pull" in str(s.get("run", "")))

    assert not (REPO_ROOT / ".github/workflows/edge-worker-image.yml").exists()
    assert "load" not in worker_step["with"]
    assert worker_step["with"]["outputs"] == (
        "${{ env.RELEASE_BUILD != 'true' && 'type=docker,dest=/tmp/ml-worker-runtime.tar' || '' }}"
    )
    # A release still pushes an OCI index, because a release's digest is the one
    # a later release may reuse and re-tagging only preserves a digest when the
    # manifest is an index. See docs/runbooks/edge-image-publish.md.
    assert worker_step["with"]["provenance"] == "${{ env.RELEASE_BUILD == 'true' }}"
    assert worker_step["with"]["cache-from"] == "type=gha,scope=edge-ml-worker"
    assert worker_step["with"]["cache-to"] == (
        "${{ env.PUSH_IMAGES == 'true' && 'type=gha,scope=edge-ml-worker,mode=max' || '' }}"
    )

    assert source.count("file: Dockerfile.edge") == 1
    assert carrier_smoke["if"] == "env.BUILD_ML_WORKER == 'true' && env.RELEASE_BUILD != 'true'"
    assert "docker run --rm --network none" in str(carrier_smoke["run"])
    assert "python -m worker --check-config" in str(carrier_smoke["run"])
    assert 'test "$status" -eq 0' in str(carrier_smoke["run"])

    # The reuse/release path still pulls the published bytes and boots them, so
    # a seal never pins a worker digest that was not booted in this run.
    assert "$IMAGE_NAMESPACE/ml-worker@$ML_WORKER_DIGEST" in str(pull_smoke["run"])
    assert "docker run --rm" in str(pull_smoke["run"])
    assert "python -m worker --check-config" in str(pull_smoke["run"])

    dockerfile = (REPO_ROOT / "Dockerfile.edge").read_text(encoding="utf-8")
    stages = re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", dockerfile, re.MULTILINE)
    assert stages[-1] == "runtime", stages
    assert "bootsmoke" not in dockerfile
    for retired in (
        (Path("worker") / "native").as_posix(),
        (Path("worker") / "runtime" / "deepstream").as_posix(),
        (Path("worker") / "adapters" / "decode").as_posix(),
        (Path("worker") / "adapters" / "encode").as_posix(),
        (Path("worker") / "pipeline" / "ingest").as_posix(),
        (Path("worker") / "pipeline" / "bus").as_posix(),
        (Path("worker") / "tools" / "deepstream_canary").as_posix(),
        "deepstream-native-build",
        "preflight",
    ):
        assert retired not in dockerfile, retired
    dev_compose = (REPO_ROOT / "compose.edge.dev.yaml").read_text(encoding="utf-8")
    assert "target:" not in dev_compose
    for doc in ("AGENTS.md", "docs/runbooks/edge-image-publish.md"):
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "docker build" in line and "Dockerfile.edge" in line:
                assert "--target" not in line, (doc, line)


def test_a_publishing_run_never_records_an_empty_digest() -> None:
    """The seal must never carry `ml-api@` with nothing after it.

    A pull request builds ml-api with no exporter (no push, no load), so buildx
    reports no digest -- which the artifact used to interpolate straight into
    `ML_API_IMAGE=.../ml-api@`, advertising an unpullable ref. That is harmless
    only because a PR uploads no artifact. On a run that DOES publish, an empty
    digest means the build did not export what the release is about to pin, and
    it has to fail loudly instead.
    """
    source = (REPO_ROOT / EDGE_IMAGES_WORKFLOW).read_text(encoding="utf-8")
    assert 'raise SystemExit(f"{image} was built but exported no digest")' in source
    assert 'if os.environ.get("PUSH_IMAGES") == "true":' in source
    # ...and the reused branch has the same shape: no digest, no release.
    assert 'raise SystemExit(f"{image} was reused but no published digest was recorded")' in source


def test_release_isolation_keys_on_the_dispatch_not_only_the_release_event() -> None:
    """A release's images arrive by dispatch, so reuse must recognise that.

    release.yml cannot rely on the `release` event to publish the images: a
    release it creates with the default GITHUB_TOKEN raises no event that starts
    a workflow run, so it dispatches this workflow on the tag instead (#460).
    Keying reuse on `github.event_name == 'release'` alone would therefore be
    dead code -- the isolation would never once engage on a real release, and
    nothing would fail to say so. RELEASE_BUILD has to count both shapes.
    """
    workflow = _workflow(EDGE_IMAGES_WORKFLOW)
    release_build = workflow["jobs"]["publish"]["env"]["RELEASE_BUILD"]
    assert "github.event_name == 'release'" in release_build, release_build
    assert "workflow_dispatch" in release_build, release_build
    assert "startsWith(inputs.ref, 'seeon-edge-v')" in release_build, release_build

    # ...and the tag the seal is cut for comes from whichever shape it was.
    release_tag = workflow["jobs"]["publish"]["env"]["RELEASE_TAG"]
    assert "github.event.release.tag_name" in release_tag, release_tag
    assert "inputs.ref" in release_tag, release_tag

    # The release workflow really does dispatch this workflow on the tag; if
    # that step were dropped, the reuse path above would go dormant again.
    release_source = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "gh workflow run edge-images.yml" in release_source


def test_legacy_multi_target_ml_dockerfile_is_removed() -> None:
    assert not (REPO_ROOT / "Dockerfile").exists()


def test_edge_api_host_port_is_loopback_only() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    ports = _list_field(services["ml-api"], "ports")

    assert ports == ["127.0.0.1:8000:8000"]


def test_edge_runtime_state_volumes_follow_backend_ownership() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    volumes = _compose_top_level(EDGE_COMPOSE_FILE, "volumes")

    for service_name in (EDGE_DB_CUTOVER_SERVICE, "ml-api"):
        assert "edge-state:/var/lib/seeon-state" in _list_field(services[service_name], "volumes")
    for service_name in ("edge-db-migrator", "ml-worker"):
        assert not any(
            str(volume).startswith("edge-state:")
            for volume in _list_field(services[service_name], "volumes")
        ), service_name
    assert "worker-local-state:/var/lib/seeon-state" in _list_field(
        services["ml-worker"], "volumes"
    )
    assert set(volumes) == {
        "edge-migration",
        "edge-pg-authority",
        "edge-pgdata",
        "edge-state",
        "worker-engine-cache",
        "worker-local-state",
        MODELS_VOLUME,
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


def test_flow_compose_reserves_the_required_nvidia_hardware() -> None:
    services = _compose_services(EDGE_COMPOSE_FILE)
    worker = services["ml-worker"]
    assert _mapping_field(worker, "deploy") == {
        "resources": {
            "reservations": {
                "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
            }
        }
    }
    assert "devices" not in worker
    assert (
        _mapping_field(worker, "environment")["NVIDIA_DRIVER_CAPABILITIES"]
        == "compute,utility,video"
    )


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


def test_edge_api_execution_records_seam_environment_contract() -> None:
    """ml-api's execution-record seam: OFF by default, budget has no compose default."""
    services = _compose_services(EDGE_COMPOSE_FILE)
    api_environment = _mapping_field(services["ml-api"], "environment")
    assert api_environment["ML_API_EXECUTION_RECORDS_ENABLED"] == (
        "${ML_API_EXECUTION_RECORDS_ENABLED:-0}"
    )
    assert api_environment["ML_API_EXECUTION_RECORDS_BUDGET_BYTES"] == (
        "${ML_API_EXECUTION_RECORDS_BUDGET_BYTES:-}"
    )


#: The only interpolation form compose.edge.yaml uses for ml-api's ML_API_* keys.
_COMPOSE_DEFAULT: Final = re.compile(r"\$\{(?P<name>\w+):-(?P<default>[^}]*)\}")


def _rendered_api_settings_environment() -> dict[str, str]:
    """ml-api's ML_API_* environment as `docker compose config` renders it from the example.

    `${NAME:-default}` takes the default when NAME is unset or empty.
    """
    lines = (REPO_ROOT / EDGE_ENV_EXAMPLE).read_text(encoding="utf-8").splitlines()
    example = {
        key: value
        for key, _, value in (line.partition("=") for line in lines)
        if key and not key.startswith("#")
    }
    api_environment = _mapping_field(_compose_services(EDGE_COMPOSE_FILE)["ml-api"], "environment")
    rendered = {}
    for key, value in api_environment.items():
        if not key.startswith("ML_API_"):
            continue
        match = _COMPOSE_DEFAULT.fullmatch(str(value))
        assert match is not None, key
        rendered[key] = example.get(match["name"]) or match["default"]
    return rendered


def _api_settings_from(monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]) -> Settings:
    for key in list(os.environ):
        if key.upper().startswith("ML_API_"):
            monkeypatch.delenv(key)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return Settings()


def test_ml_api_settings_accept_the_rendered_edge_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose renders the unset budget as ''; records stay off and the API boots."""
    environment = _rendered_api_settings_environment()

    settings = _api_settings_from(monkeypatch, environment)

    assert environment["ML_API_EXECUTION_RECORDS_BUDGET_BYTES"] == ""
    assert settings.execution_records_enabled is False
    assert settings.execution_records_budget_bytes is None


@pytest.mark.parametrize("budget", ["", None], ids=["empty", "missing"])
def test_ml_api_settings_still_require_a_budget_when_records_are_enabled(
    monkeypatch: pytest.MonkeyPatch, budget: str | None
) -> None:
    environment = _rendered_api_settings_environment() | {"ML_API_EXECUTION_RECORDS_ENABLED": "1"}
    if budget is None:
        del environment["ML_API_EXECUTION_RECORDS_BUDGET_BYTES"]
    else:
        environment["ML_API_EXECUTION_RECORDS_BUDGET_BYTES"] = budget

    with pytest.raises(ValidationError) as refused:
        _api_settings_from(monkeypatch, environment)

    # The budget rule refused it, not an integer parse of the empty string.
    assert [(error["type"], error["loc"]) for error in refused.value.errors()] == [
        ("value_error", ())
    ]

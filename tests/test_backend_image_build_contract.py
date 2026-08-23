from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_release_image_build_keeps_dashboard_credentials_out_of_static_assets() -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/edge-images.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert workflow["jobs"]["publish"]["env"]["DEPLOY_REF"] == (
        "${{ github.event.release.tag_name || inputs.ref || github.sha }}"
    )
    steps = workflow["jobs"]["publish"]["steps"]

    assert all(step.get("name") != "Require configured frontend relay token" for step in steps)

    build_step = next(step for step in steps if step.get("name") == "Build and push ml-api image")
    build_args = build_step["with"]["build-args"].splitlines()
    assert "VITE_ML_API_BASE_URL=/api/v1" in build_args
    assert all("RELAY_TOKEN" not in value and "secrets." not in value for value in build_args)


def test_backend_image_declares_frontend_build_arguments() -> None:
    dockerfile = (ROOT / "Dockerfile.backend").read_text(encoding="utf-8")
    assert "ARG VITE_ML_API_BASE_URL=/api/v1" in dockerfile
    assert "VITE_ML_API_RELAY_TOKEN" not in dockerfile


def test_backend_image_includes_the_legacy_evidence_drain_script() -> None:
    dockerfile = (ROOT / "Dockerfile.backend").read_text(encoding="utf-8")

    assert "COPY scripts/ops ./scripts/ops" in dockerfile

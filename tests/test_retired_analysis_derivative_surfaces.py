"""Task 12: retired analysis/derivative/label surfaces are absent."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan

ROOT = Path(__file__).resolve().parents[1]
RETIRED_ROUTE_PATHS = frozenset(
    {
        "/api/v1/clips/{clip_id}/analysis",
        "/api/v1/clips/{clip_id}/derivatives/{kind}",
        "/api/v1/clips/{clip_id}/label",
        "/api/v1/relay/analysis-traces",
    }
)
RETIRED_PRODUCTION_MODULES = (
    "backend.app.features.clips.derivative_control",
    "worker.runtime.derivative_runtime",
    "worker.runtime.telemetry.analysis_trace_sender",
    "worker.pipeline.output.annotated_derivative",
    "worker.pipeline.output.evidence.derivative_artifact_store",
    "worker.pipeline.output.evidence.derivative_store",
)
RETIRED_PRODUCTION_NAMES = frozenset(
    {
        "control_derivative",
        "ClipAnalysis",
        "ClipAnalysisResponse",
        "ClipAnalysisValueResponse",
        "ClipDerivativeResponse",
        "DerivativeKind",
        "DerivativeProjection",
        "LabelClipRequest",
        "LabelClipResponse",
        "LabelRecord",
        "LabelStore",
        "open_verified_annotated",
        "RelayAnalysisTraceRequest",
        "RelayAnalysisTraceResponse",
        "require_relay_analysis_trace",
        "AnalysisTraceSender",
        "DerivativeCommand",
        "DerivativeCommandExecutor",
        "DerivativeControl",
        "DerivativeControlService",
        "AnnotatedDerivativeJob",
        "DerivativeArtifactStore",
        "derivative_state",
    }
)
PRODUCTION_ROOTS = (ROOT / "backend" / "app", ROOT / "worker")


def _registered_paths() -> set[str]:
    app = create_app(lifespan=no_lifespan)
    return {route.path for route in app.routes if isinstance(route, APIRoute)}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def test_retired_routes_are_absent_from_the_app_table() -> None:
    registered = _registered_paths()
    assert registered.isdisjoint(RETIRED_ROUTE_PATHS)


def test_retired_http_surfaces_are_404_by_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "clip-store" / "clips" / "clip-a"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"clean-video")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "camera_id": "camera-a",
                "event_ref": "event-a",
                "event_type": "fall",
                "started_at": "2026-08-25T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/clip-a/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        probes = (
            client.get("/api/v1/clips/clip-a/analysis"),
            client.post("/api/v1/clips/clip-a/derivatives/still"),
            client.get("/api/v1/clips/clip-a/derivatives/video"),
            client.delete("/api/v1/clips/clip-a/derivatives/still"),
            client.put("/api/v1/clips/clip-a/label", json={"label": "TRUE_POSITIVE"}),
            client.post(
                "/api/v1/relay/analysis-traces",
                json={"camera_id": "camera-a", "frames": [], "truncation": {}},
            ),
        )
    assert {response.status_code for response in probes} == {404}
    for response in probes:
        assert response.json() == {"detail": "Not Found"}


def test_retired_production_modules_are_absent() -> None:
    for module_name in RETIRED_PRODUCTION_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def _imported_or_defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            if node.module is not None:
                names.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_app_and_worker_import_graphs_have_no_retired_production_symbol() -> None:
    offenders: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in sorted(root.rglob("*.py")):
            found = _imported_or_defined_names(path) & RETIRED_PRODUCTION_NAMES
            if found:
                offenders.append(f"{path.relative_to(ROOT)}:{sorted(found)}")
    assert offenders == []


def test_clip_artifacts_schema_is_only_clean_and_optional_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "clip-store" / "clips" / "clip-a"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"clean-video")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "camera_id": "camera-a",
                "event_ref": "event-a",
                "event_type": "fall",
                "started_at": "2026-08-25T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/clip-a/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips/clip-a/artifacts")
        video = client.get("/api/v1/clips/clip-a/video?view=annotated")
    assert response.status_code == 200
    body = response.json()
    assert set(body) <= {"clip_id", "clean", "snapshot"}
    assert body["clip_id"] == "clip-a"
    assert body["clean"] == "AVAILABLE"
    assert "analysis" not in body
    assert "annotated" not in body
    assert "playback_view" not in body
    assert "annotated_fallback_to_clean" not in body
    assert video.status_code == 200
    assert video.content == b"clean-video"
    assert video.headers.get("x-clip-view", "clean") == "clean"
    assert "x-clip-view-fallback" not in {key.lower() for key in video.headers}

"""Render the standalone canary Compose project and run-local worker config."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from shared.release_identity import (
    EDGE_DATABASE_FORMAT_IDENTITY,
    EDGE_DATABASE_SCHEMA_VERSION,
)
from worker.tools.deepstream_canary.report import JsonValue, canonical_json, write_receipt_manifest

PROJECT_NAME: Final = "seeon-ds-canary"
SUPPORT_DIR: Final = Path("scripts/qa/deepstream-canary")
BASE_COMPOSE: Final = SUPPORT_DIR / "compose.canary.yaml"


@dataclass(frozen=True, slots=True)
class RenderRequest:
    evidence_dir: Path
    worker_image: str
    relay_token: str
    camera_count: int
    model_dir: Path
    engine_cache_source: Path | None = None


def _publisher_block(camera_count: int, worker_image: str, corpus_dir: Path) -> str:
    blocks: list[str] = []
    for index in range(1, camera_count + 1):
        camera = f"loop-{index:02d}"
        offset_ms = (index - 1) * 67
        blocks.append(
            f"  publisher-{index:02d}:\n"
            f"    image: {worker_image}\n"
            "    pull_policy: never\n"
            "    restart: \"no\"\n"
            "    depends_on:\n      mediamtx:\n        condition: service_healthy\n"
            "    networks: [canary]\n"
            f"    volumes:\n      - {corpus_dir}:/corpus:ro\n"
            "    command: [ffmpeg, -re, -stream_loop, \"-1\", "
            f"-itsoffset, \"0.{offset_ms:03d}\", -i, /corpus/loopback.mp4, "
            "-map, \"0:v:0\", -c:v, copy, -f, rtsp, -rtsp_transport, tcp, "
            f"rtsp://mediamtx:8554/{camera}]\n"
            "    cap_drop: [ALL]\n"
            "    security_opt: [no-new-privileges:true]\n"
        )
    return "".join(blocks)


def _worker_config(camera_count: int) -> bytes:
    cameras: list[JsonValue] = [
        {
            "camera_id": f"loop-{index:02d}",
            "facility_id": "canary-facility",
            "space_id": f"canary-space-{index:02d}",
            "label": f"Canary {index:02d}",
            "rtsp_url": f"rtsp://mediamtx:8554/loop-{index:02d}",
            "online": True,
            "fps": 15.0,
            "frame_stride": 1,
            "decode_backend": "nvdec",
            "domains": ["fall", "bed_exit"],
        }
        for index in range(1, camera_count + 1)
    ]
    payload: dict[str, JsonValue] = {
        "registry_version": 1,
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": cameras,
        "clip_export_enabled": False,
        "clip_export_version": 0,
    }
    return canonical_json(payload)


def render_compose(request: RenderRequest) -> tuple[Path, str]:
    """Create one self-contained Compose file with one publisher per camera."""
    root = request.evidence_dir.resolve()
    run_dir = root / "run"
    paths = {
        "CANARY_ASSET_DIR": SUPPORT_DIR.resolve(),
        "CANARY_RECEIPT_DIR": root / "raw",
        "CANARY_MODEL_DIR": run_dir / "models",
        "CANARY_STATE_DIR": run_dir / "state",
        "CANARY_SOCKET_DIR": run_dir / "sockets",
        "CANARY_ENGINE_DIR": run_dir / "engine-cache",
        "CANARY_SCRATCH_DIR": run_dir / "scratch",
        "CANARY_CLIP_DIR": run_dir / "clips",
        "CANARY_CONFIG_PATH": run_dir / "configs",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for key in (
        "CANARY_RECEIPT_DIR",
        "CANARY_STATE_DIR",
        "CANARY_SOCKET_DIR",
        "CANARY_ENGINE_DIR",
        "CANARY_SCRATCH_DIR",
        "CANARY_CLIP_DIR",
    ):
        paths[key].mkdir(parents=True, exist_ok=True, mode=0o700)
    if paths["CANARY_MODEL_DIR"].exists():
        raise FileExistsError(paths["CANARY_MODEL_DIR"])
    shutil.copytree(request.model_dir.resolve(), paths["CANARY_MODEL_DIR"])
    paths["CANARY_CONFIG_PATH"].mkdir(mode=0o755)
    zero_config = paths["CANARY_CONFIG_PATH"] / "worker-zero.json"
    workload_config = paths["CANARY_CONFIG_PATH"] / "worker-workload.json"
    zero_config.write_bytes(_worker_config(0))
    workload_config.write_bytes(_worker_config(request.camera_count))
    _ = zero_config.chmod(0o444)
    _ = workload_config.chmod(0o444)
    _ = paths["CANARY_RECEIPT_DIR"].chmod(0o777)
    (paths["CANARY_RECEIPT_DIR"] / "active-config").write_text(
        "zero\n", encoding="utf-8"
    )
    template = BASE_COMPOSE.read_text(encoding="utf-8")
    replacements = {
        "CANARY_WORKER_IMAGE": request.worker_image,
        "CANARY_EDGE_DATABASE_SCHEMA_VERSION": str(EDGE_DATABASE_SCHEMA_VERSION),
        "CANARY_EDGE_DATABASE_FORMAT_IDENTITY": EDGE_DATABASE_FORMAT_IDENTITY,
        **{name: str(path) for name, path in paths.items()},
    }
    for name, value in replacements.items():
        template = template.replace(f"${{{name}}}", value)
    source_mount = (
        ""
        if request.engine_cache_source is None
        else f"      - {request.engine_cache_source.resolve()}:/prepared-cache:ro\n"
    )
    template = template.replace("      # CANARY_ENGINE_SOURCE_MOUNT\n", source_mount)
    publisher = _publisher_block(
        request.camera_count, request.worker_image, paths["CANARY_SCRATCH_DIR"]
    )
    rendered = template.replace("\nnetworks:\n", f"{publisher}\nnetworks:\n")
    compose_path = root / "compose.rendered.yaml"
    compose_path.write_text(rendered, encoding="utf-8")
    compose_digest = hashlib.sha256(compose_path.read_bytes()).hexdigest()
    _ = write_receipt_manifest(root, (compose_path, zero_config, workload_config))
    return compose_path, compose_digest

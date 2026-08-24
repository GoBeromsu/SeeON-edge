"""Explicit opt-in configuration for the production-unwired C5 child."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, override

_DEFAULT_EXECUTABLE: Final = Path("/usr/local/bin/seeon-deepstream-child")
_DEFAULT_SOCKET_DIR: Final = Path("/run/seeon/deepstream")


@dataclass(frozen=True, slots=True)
class ChildConfig:
    executable: Path
    worker_boot_id: uuid.UUID
    socket_dir: Path
    first_fault_path: Path
    lease_state_dir: Path | None = None
    child_instance_id: uuid.UUID = field(default_factory=uuid.uuid4)
    startup_timeout_sec: float = 10.0
    stop_timeout_sec: float = 5.0
    qa_mode: bool = False


@dataclass(frozen=True, slots=True)
class DarkChildConfigError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def configured_dark_supervisors(env: Mapping[str, str]) -> tuple[ChildConfig, ...]:
    """Return explicit GPU-0 dark child only; unchanged environments return none."""
    if env.get("SEEON_DEEPSTREAM_DARK_CHILD") != "1":
        return ()
    visible = env.get("NVIDIA_VISIBLE_DEVICES", "0").strip()
    visible_devices = (
        ("0",)
        if visible in ("", "all")
        else tuple(part.strip() for part in visible.split(","))
    )
    if visible_devices != ("0",):
        raise DarkChildConfigError("unsupported_gpu", visible)
    boot = uuid.UUID(env["SEEON_WORKER_BOOT_ID"]) if "SEEON_WORKER_BOOT_ID" in env else uuid.uuid4()
    socket_dir = Path(env.get("SEEON_DEEPSTREAM_SOCKET_DIR", str(_DEFAULT_SOCKET_DIR)))
    executable = Path(env.get("SEEON_DEEPSTREAM_CHILD", str(_DEFAULT_EXECUTABLE)))
    fault_root = Path(
        env.get("SEEON_DEEPSTREAM_FIRST_FAULT_DIR", "/var/lib/seeon-state/deepstream")
    )
    return (
        ChildConfig(
            executable=executable,
            worker_boot_id=boot,
            socket_dir=socket_dir / "gpu-0",
            first_fault_path=fault_root / "deepstream-gpu-0.fault",
        ),
    )


__all__ = ["ChildConfig", "DarkChildConfigError", "configured_dark_supervisors"]

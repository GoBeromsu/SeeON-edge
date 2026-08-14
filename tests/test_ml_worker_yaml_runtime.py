"""Runtime-loader coverage for retired mutable YAML authority.

Model artifacts are validated after versioned config is pulled. A local YAML
model policy must therefore be rejected by the loader itself, even when its
artifact is missing, rather than becoming a fallback configuration source.
The complete camera/model/domain/clip rejection matrix lives in the related
YAML authority tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.loader import load_worker_config


def test_worker_yaml_loader_rejects_static_model_before_artifact_validation(
    tmp_path: Path,
) -> None:
    config_path = _write_static_model_config(
        tmp_path / "ml-worker.yaml",
        artifact_dir=tmp_path / "missing",
    )

    with pytest.raises(WorkerConfigError, match="static models policy is retired"):
        load_worker_config(config_path)


def _write_static_model_config(path: Path, *, artifact_dir: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "relay": {
                    "url": "http://127.0.0.1:8000",
                    "token": "relay-token-1",
                },
                "models": {
                    "fall": {
                        "type": "lstm",
                        "framework": "pytorch",
                        "mode": "sequence",
                        "artifact_dir": str(artifact_dir),
                        "weights": "model.pt",
                        "architecture": "arch.json",
                        "metadata": "metadata.yaml",
                        "window": 3,
                        "stride": 1,
                        "input_shape": [3, 51],
                        "operating_threshold": 0.5,
                    }
                },
                "cameras": [],
            }
        ),
        encoding="utf-8",
    )
    return path

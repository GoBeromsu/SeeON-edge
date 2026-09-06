from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from worker.runtime.flow.cold_start import (
    EngineIdentityError,
    FlowColdStart,
    verify_engine_identity,
    verify_flow_boot_inputs,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cold_start_verifies_before_warmup(tmp_path: Path) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    files = {}
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {"engine_sha256": _digest(engine), "image_digest": "image", "batch_size": "2", **files}
        )
    )
    warmed = []
    FlowColdStart(engine, identity, files, lambda: warmed.append(True)).run()
    assert warmed == [True]


def test_missing_engine_names_operator_tool(tmp_path: Path) -> None:
    with pytest.raises(EngineIdentityError, match="edge-engine-build"):
        verify_engine_identity(tmp_path / "missing.engine", tmp_path / "identity.json", {})


def test_engine_identity_refuses_a_roster_larger_than_the_engine(tmp_path: Path) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps({"engine_sha256": _digest(engine), "image_digest": "image", "batch_size": "2"})
    )

    with pytest.raises(
        EngineIdentityError, match="engine batch 2 does not cover deployed roster batch 3"
    ):
        verify_engine_identity(engine, identity, {}, deployed_batch=3)

    verify_engine_identity(engine, identity, {}, deployed_batch=2)


def _write_onnx(path: Path, batch: int | str) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "pose",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [batch, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [batch, 3, 640, 640])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def _flow_env(tmp_path: Path, *, batch: int, onnx_batch: int | str) -> dict[str, str]:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    onnx_path = tmp_path / "model.onnx"
    _write_onnx(onnx_path, onnx_batch)
    files = {
        "infer_config_sha256": tmp_path / "infer.txt",
        "tracker_config_sha256": tmp_path / "tracker.txt",
        "tracker_library_sha256": tmp_path / "tracker.so",
        "onnx_sha256": onnx_path,
        "parser_lib_sha256": tmp_path / "parser.so",
    }
    for path in files.values():
        if path != onnx_path:
            path.write_bytes(path.name.encode())
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "engine_sha256": _digest(engine),
                "image_digest": "image",
                "batch_size": str(batch),
                **{key: _digest(path) for key, path in files.items()},
            }
        )
    )
    return {
        "ML_WORKER_FLOW_ENGINE_PATH": str(engine),
        "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH": str(identity),
        "ML_WORKER_FLOW_INFER_CONFIG": str(files["infer_config_sha256"]),
        "ML_WORKER_FLOW_TRACKER_CONFIG": str(files["tracker_config_sha256"]),
        "ML_WORKER_FLOW_TRACKER_LIBRARY": str(files["tracker_library_sha256"]),
        "ML_WORKER_FLOW_ONNX_PATH": str(onnx_path),
        "ML_WORKER_FLOW_PARSER_LIBRARY": str(files["parser_lib_sha256"]),
        "ML_WORKER_FLOW_RECORD_DIR": str(tmp_path / "records"),
        "ML_WORKER_FLOW_RECORD_CACHE_SECONDS": "10",
        "ML_WORKER_FLOW_FRAME_WIDTH": "640",
        "ML_WORKER_FLOW_FRAME_HEIGHT": "640",
        "ML_WORKER_FLOW_BATCH_SIZE": str(batch),
    }


def test_flow_boot_refuses_fixed_onnx_batch_above_one(tmp_path: Path) -> None:
    with pytest.raises(EngineIdentityError, match=r"edge-engine-build.*export_pose_onnx"):
        verify_flow_boot_inputs(_flow_env(tmp_path, batch=13, onnx_batch=1))


def test_flow_boot_admits_symbolic_onnx_batch_above_one(tmp_path: Path) -> None:
    assert (
        verify_flow_boot_inputs(_flow_env(tmp_path, batch=13, onnx_batch="batch"))["batch_size"]
        == "13"
    )


def test_flow_boot_admits_fixed_onnx_at_batch_one(tmp_path: Path) -> None:
    assert verify_flow_boot_inputs(_flow_env(tmp_path, batch=1, onnx_batch=1))["batch_size"] == "1"


def test_flow_boot_reports_digest_mismatch_before_onnx_shape(tmp_path: Path) -> None:
    env = _flow_env(tmp_path, batch=13, onnx_batch=1)
    Path(env["ML_WORKER_FLOW_ONNX_PATH"]).write_bytes(b"changed")
    with pytest.raises(EngineIdentityError, match="digest mismatch"):
        verify_flow_boot_inputs(env)

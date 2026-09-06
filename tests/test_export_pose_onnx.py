from __future__ import annotations

import hashlib
import os
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from worker.adapters.model.errors import ModelLoadError
from worker.tools.export_pose_onnx import export_pose_onnx


def _write_export(
    path: Path, *, batch: int | str = "batch", date: str | None = None, value_info: int = 0
) -> None:
    shape = helper.make_tensor("shape", TensorProto.INT64, [3], [2, 300, 57])
    graph = helper.make_graph(
        [
            helper.make_node(
                "ConstantOfShape",
                ["shape"],
                ["output0"],
                value=helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0]),
            )
        ],
        "pose",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [batch, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [2, 300, 57])],
        initializer=[shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    if date is not None:
        item = model.metadata_props.add()
        item.key, item.value = "date", date
    for index in range(value_info):
        model.graph.value_info.append(
            helper.make_tensor_value_info(f"inferred_{index}", TensorProto.FLOAT, [1])
        )
    onnx.save(model, path)


def _exporter(*, batch: int | str = "batch", date: str | None = "volatile", value_info: int = 0):
    def export(weights: Path) -> Path:
        result = weights.with_suffix(".onnx")
        _write_export(result, batch=batch, date=date, value_info=value_info)
        return result

    return export


def test_export_writes_matching_sidecar_and_does_not_modify_source_dir(tmp_path: Path) -> None:
    source = tmp_path / "yolo26n-pose.pt"
    source.write_bytes(b"weights")
    before = {entry.name for entry in tmp_path.iterdir()}

    digest = export_pose_onnx(source, exporter=_exporter())

    destination = source.with_suffix(".onnx")
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    assert destination.with_suffix(".onnx.sha256").read_text(encoding="ascii") == f"{digest}\n"
    assert source.read_bytes() == b"weights"
    assert {entry.name for entry in tmp_path.iterdir()} == before | {
        "yolo26n-pose.onnx",
        "yolo26n-pose.onnx.sha256",
    }


def test_export_requires_force_to_replace_differing_artifact(tmp_path: Path) -> None:
    source = tmp_path / "yolo26n-pose.pt"
    source.write_bytes(b"weights")
    destination = source.with_suffix(".onnx")
    destination.write_bytes(b"different")

    with pytest.raises(ModelLoadError, match="pass --force"):
        export_pose_onnx(source, exporter=_exporter())
    export_pose_onnx(source, force=True, exporter=_exporter())


def test_export_rejects_fixed_batch_output(tmp_path: Path) -> None:
    source = tmp_path / "yolo26n-pose.pt"
    source.write_bytes(b"weights")

    with pytest.raises(ModelLoadError, match="fixed batch dimension 1"):
        export_pose_onnx(source, exporter=_exporter(batch=1))


def test_export_strips_date_and_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "yolo26n-pose.pt"
    source.write_bytes(b"weights")
    first = export_pose_onnx(source, exporter=_exporter(date="first", value_info=3))
    second = export_pose_onnx(source, exporter=_exporter(date="second", value_info=7))

    assert first == second
    written = onnx.load(source.with_suffix(".onnx"))
    assert all(item.key != "date" for item in written.metadata_props)
    assert len(written.graph.value_info) == 0


def test_real_export_is_opt_in() -> None:
    if os.environ.get("SEEON_REAL_EXPORT") != "1":
        pytest.skip("set SEEON_REAL_EXPORT=1 to run the real pose export")
    source = Path("models/pose/yolo26n-pose.pt")
    if not source.is_file():
        pytest.skip("pose weights are absent")
    assert export_pose_onnx(source)

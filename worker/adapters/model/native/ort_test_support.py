"""Test-only bindings for real ORT execution; never imported by production."""

import ctypes as ct
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


class Tensor(ct.Structure):
    _fields_ = [
        ("name", ct.c_char_p),
        ("data", ct.POINTER(ct.c_float)),
        ("elements", ct.c_size_t),
        ("shape", ct.c_int64 * 8),
        ("rank", ct.c_size_t),
    ]


class Info(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("input_count", ct.c_uint32),
        ("output_count", ct.c_uint32),
        ("threads", ct.c_uint32),
        ("runtime_version", ct.c_char * 64),
    ]


def tensor(name, values, shape=None):
    dimensions = values.shape if shape is None else shape
    return Tensor(
        name.encode(),
        values.ctypes.data_as(ct.POINTER(ct.c_float)),
        values.size,
        (ct.c_int64 * 8)(*dimensions),
        len(dimensions),
    )


def model_bytes(*, gather=False, input_type=TensorProto.FLOAT):
    if gather:
        inputs = [helper.make_tensor_value_info("input", input_type, [2])]
        outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [2])]
        nodes = [
            helper.make_node("Cast", ["input"], ["indices"], to=TensorProto.INT64),
            helper.make_node("Gather", ["data", "indices"], ["output0"], axis=0),
        ]
        initializers = [helper.make_tensor("data", TensorProto.FLOAT, [2], [10.0, 20.0])]
    else:
        inputs = [helper.make_tensor_value_info("input", input_type, [2, 3])]
        outputs = [
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("output1", TensorProto.FLOAT, [2, 1]),
        ]
        nodes = [
            helper.make_node("Cast", ["input"], ["floats"], to=TensorProto.FLOAT),
            helper.make_node("Sin", ["floats"], ["output0"]),
            helper.make_node("ReduceSum", ["output0"], ["output1"], axes=[1], keepdims=1),
        ]
        initializers = []
    model = helper.make_model(
        helper.make_graph(nodes, "native-cpu-test", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 12)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def oracle(blob, values):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(blob, options, providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    return session.run(None, {"input": values})


class Native:
    def __init__(self):
        assert ort.__version__ == "1.29.0", "tests require the existing pinned ORT runtime"
        self.runtime = Path(ort.__file__).parent / "capi" / f"libonnxruntime.so.{ort.__version__}"
        assert self.runtime.is_file(), "ORT C API runtime library is required"
        library = Path(os.environ["SEEON_TEST_ORT_ADAPTER"]).resolve(strict=True)
        self.api = ct.CDLL(str(library))
        error_args = [ct.POINTER(ct.c_char), ct.c_size_t]
        self.api.seeon_ort_open.argtypes = [
            ct.c_char_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_uint32,
            ct.POINTER(ct.c_void_p),
            *error_args,
        ]
        self.api.seeon_ort_open.restype = ct.c_int
        self.api.seeon_ort_info.argtypes = [ct.c_void_p, ct.POINTER(Info), *error_args]
        self.api.seeon_ort_info.restype = ct.c_int
        self.api.seeon_ort_run.argtypes = [
            ct.c_void_p,
            ct.POINTER(Tensor),
            ct.POINTER(Tensor),
            ct.c_size_t,
            *error_args,
        ]
        self.api.seeon_ort_run.restype = ct.c_int
        self.api.seeon_ort_close.argtypes = [ct.c_void_p]
        self.api.seeon_ort_close.restype = None

    def open(self, blob, threads=1, runtime=None):
        handle = ct.c_void_p()
        error = ct.create_string_buffer(256)
        data = ct.create_string_buffer(blob)
        code = self.api.seeon_ort_open(
            os.fsencode(self.runtime if runtime is None else runtime),
            data,
            len(blob),
            threads,
            ct.byref(handle),
            error,
            len(error),
        )
        return code, handle, error.value.decode(errors="replace")

    def info(self, handle):
        value = Info()
        error = ct.create_string_buffer(256)
        code = self.api.seeon_ort_info(handle, ct.byref(value), error, len(error))
        return code, value, error.value

    def run(self, handle, input_tensor, outputs):
        error = ct.create_string_buffer(256)
        code = self.api.seeon_ort_run(
            handle, ct.byref(input_tensor), outputs, len(outputs), error, len(error)
        )
        return code, error.value.decode(errors="replace")

    def close(self, handle):
        self.api.seeon_ort_close(handle)


def buffers():
    first = np.full((2, 3), -1234.5, dtype=np.float32)
    second = np.full((2, 1), -1234.5, dtype=np.float32)
    descriptors = (Tensor * 2)(tensor("output0", first), tensor("output1", second))
    return first, second, descriptors

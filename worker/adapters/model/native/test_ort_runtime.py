"""Opt-in native CPU tests: make test-ort (no GPU or model-download dependency)."""

import ctypes as ct
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager

import numpy as np
from onnx import TensorProto
from ort_test_support import Native, Tensor, buffers, model_bytes, oracle, tensor


class OrtRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native()
        cls.blob = model_bytes()

    @contextmanager
    def opened(self, blob=None, threads=1):
        code, handle, error = self.native.open(self.blob if blob is None else blob, threads)
        self.assertEqual(code, 0, error)
        self.assertIsNotNone(handle.value)
        try:
            yield handle
        finally:
            self.native.close(handle)

    def test_actual_outputs_match_independent_python_cpu_session_repeatedly(self):
        values = np.array([[-3.25, -0.125, 0.0], [0.5, 1.75, 12.0]], dtype=np.float32)
        expected = oracle(self.blob, values)
        with self.opened() as handle:
            for _ in range(5):
                first, second, outputs = buffers()
                self.assertEqual(self.native.run(handle, tensor("input", values), outputs), (0, ""))
                for descriptor, actual, reference in zip(
                    outputs, (first, second), expected, strict=True
                ):
                    self.assertEqual(descriptor.elements, reference.size)
                    self.assertEqual(
                        list(descriptor.shape)[: descriptor.rank], list(reference.shape)
                    )
                    np.testing.assert_array_equal(actual, reference)

    def test_metadata_reports_loaded_runtime_and_both_original_thread_modes(self):
        for threads in (0, 1):
            with self.subTest(threads=threads), self.opened(threads=threads) as handle:
                code, info, error = self.native.info(handle)
                self.assertEqual(code, 0, error)
                self.assertEqual((info.abi_version, info.input_count, info.output_count), (1, 1, 2))
                self.assertEqual(info.threads, threads)
                self.assertEqual(info.runtime_version, b"1.29.0")

    def test_input_refusal_preserves_output_and_keeps_session_usable(self):
        values = np.ones((2, 3), dtype=np.float32)
        with self.opened() as handle:
            for case in ("nan", "name", "count", "rank", "zero", "overflow"):
                with self.subTest(case=case):
                    changed = values.copy()
                    descriptor = tensor("input", changed)
                    if case == "nan":
                        changed[0, 0] = np.nan
                    elif case == "name":
                        descriptor.name = b"unknown"
                    elif case == "count":
                        descriptor.elements -= 1
                    elif case == "rank":
                        descriptor.rank = 9
                    elif case == "zero":
                        descriptor.shape[0] = 0
                    else:
                        descriptor.shape[0] = (1 << 63) - 1
                    first, second, outputs = buffers()
                    before = (first.tobytes(), second.tobytes(), bytes(outputs))
                    code, diagnostic = self.native.run(handle, descriptor, outputs)
                    self.assertEqual(code, 1, diagnostic)
                    self.assertTrue(diagnostic)
                    self.assertEqual((first.tobytes(), second.tobytes(), bytes(outputs)), before)
                    self.assertEqual(
                        self.native.run(handle, tensor("input", values), outputs)[0], 0
                    )

    def test_output_capacity_failure_is_atomic_and_poison_is_sticky(self):
        values = np.ones((2, 3), dtype=np.float32)
        with self.opened() as handle:
            first, second, outputs = buffers()
            outputs[1].elements = 1
            before = (first.tobytes(), second.tobytes(), bytes(outputs))
            code, diagnostic = self.native.run(handle, tensor("input", values), outputs)
            self.assertEqual(code, 5, diagnostic)
            self.assertEqual((first.tobytes(), second.tobytes(), bytes(outputs)), before)
            outputs[1].elements = second.size
            code, diagnostic = self.native.run(handle, tensor("input", values), outputs)
            self.assertEqual(code, 6, diagnostic)
            self.assertEqual((first.tobytes(), second.tobytes()), before[:2])

    def test_actual_operator_failure_poison_prevents_later_valid_execution(self):
        blob = model_bytes(gather=True)
        with self.opened(blob) as handle:
            result = np.full(2, -1234.5, dtype=np.float32)
            outputs = (Tensor * 1)(tensor("output0", result))
            before = result.tobytes(), bytes(outputs)
            invalid = np.array([1000.0, -1000.0], dtype=np.float32)
            code, diagnostic = self.native.run(handle, tensor("input", invalid), outputs)
            self.assertEqual(code, 4, diagnostic)
            self.assertTrue(diagnostic)
            self.assertEqual((result.tobytes(), bytes(outputs)), before)
            valid = np.array([0.0, 1.0], dtype=np.float32)
            self.assertEqual(self.native.run(handle, tensor("input", valid), outputs)[0], 6)
            self.assertEqual((result.tobytes(), bytes(outputs)), before)

    def test_unknown_output_name_is_refused_without_poisoning(self):
        values = np.ones((2, 3), dtype=np.float32)
        with self.opened() as handle:
            first, second, outputs = buffers()
            outputs[1].name = b"unknown"
            before = first.tobytes(), second.tobytes(), bytes(outputs)
            self.assertEqual(self.native.run(handle, tensor("input", values), outputs)[0], 1)
            self.assertEqual((first.tobytes(), second.tobytes(), bytes(outputs)), before)
            outputs[1].name = b"output1"
            self.assertEqual(self.native.run(handle, tensor("input", values), outputs)[0], 0)

    def test_model_and_runtime_open_failures_never_publish_handles(self):
        cases = [
            (b"", 1, None, 1),
            (b"not an ONNX graph", 1, None, 3),
            (self.blob, 2, None, 1),
            (self.blob, 1, "/no-such-ort-library.so", 2),
            (model_bytes(input_type=TensorProto.INT64), 1, None, 3),
        ]
        for blob, threads, runtime, expected in cases:
            with self.subTest(expected=expected, threads=threads, runtime=runtime):
                code, handle, diagnostic = self.native.open(blob, threads, runtime)
                self.assertEqual(code, expected, diagnostic)
                self.assertIsNone(handle.value)
                self.assertTrue(diagnostic)
        self.native.close(None)

    def test_null_arguments_and_bounded_error_buffer(self):
        error = (ct.c_char * 4)(b"x", b"x", b"z", b"z")
        code = self.native.api.seeon_ort_run(None, None, None, 0, error, 2)
        self.assertEqual(code, 1)
        self.assertEqual(bytes(error)[1:], b"\0zz")
        code = self.native.api.seeon_ort_run(None, None, None, 0, None, 0)
        self.assertEqual(code, 1)

    def test_process_opt_out_is_required_before_loading_the_vendor(self):
        script = """
import ctypes as c
import os
api = c.CDLL(os.environ["SEEON_TEST_ORT_ADAPTER"])
api.seeon_ort_open.argtypes = [
    c.c_char_p, c.c_void_p, c.c_size_t, c.c_uint32,
    c.POINTER(c.c_void_p), c.POINTER(c.c_char), c.c_size_t,
]
api.seeon_ort_open.restype = c.c_int
handle = c.c_void_p()
error = c.create_string_buffer(256)
data = c.create_string_buffer(b"x")
status = api.seeon_ort_open(
    b"/must-not-load-vendor-library", data, 1, 1,
    c.byref(handle), error, len(error),
)
assert status == 2, (status, error.value)
assert b"ORT_DISABLE_TELEMETRY=1" in error.value, error.value
assert handle.value is None
"""
        for value in (None, "0"):
            with self.subTest(value=value):
                environment = dict(os.environ)
                if value is None:
                    environment.pop("ORT_DISABLE_TELEMETRY", None)
                else:
                    environment["ORT_DISABLE_TELEMETRY"] = value
                result = subprocess.run(
                    [sys.executable, "-P", "-c", script],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_repeated_creation_and_release_remains_callable(self):
        values = np.ones((2, 3), dtype=np.float32)
        for _ in range(10):
            with self.opened() as handle:
                _first, _second, outputs = buffers()
                self.assertEqual(self.native.run(handle, tensor("input", values), outputs)[0], 0)


if __name__ == "__main__":
    unittest.main()

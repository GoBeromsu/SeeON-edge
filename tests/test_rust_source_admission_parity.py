"""Original G002 finite SourceAdmission differential, NOT runtime qualification.

SEEON_TEST_SOURCE_ADMISSION_PROBE must name the parent-built executable. Only an
ABSENT opt-in skips; blank/unusable paths and mismatched imported oracle sources
fail. No builds are performed. The three hashes were verified by the parent
against installed sources in frozen worker image
fffd430545508b7473644975c25e077367d8b42cf032085e96d8f5dcca49171f.
This binds source semantics, not the running image or executable's provenance.

The oracle is the actual LatestMetadataSlot.register_source/publish/peek/counters/
expected_binding, using actual image-free MetadataFrame/PerceptionFrame classes.
Only Rust's three strict-watermark reasons map to Python's observable `late`.
Success/refusal, binding, readiness, high water and retained accepted scalar
header are compared after EVERY operation, in order, without truncating strings
or narrowing integers. UUID normalization is real uuid.UUID/str at fixture/wire
construction, never a decision-dependent repair. Fixed/seeded inputs are built
before observing either implementation. Raw Rust reasons remain separately
asserted by branch cases, including precedence hidden by Python's `late` bucket.

Mapped domain: one camera, unchanged camera id on explicit re-registration,
normalized registration UUIDs, u64 generation/epoch/ordinals and optional i64 PTS.
Negative PTS and empty boot/transform are intentionally passed directly to these
real dataclasses and publish(), NOT assemble_perception_frame validation.
Python's multi-camera mailbox and permissive malformed binding registration are
not equated to the Rust constructor. No take/remove, native transport admission,
rotation/floor, or Arc policy-work lifetime parity is claimed. With no take(),
peek non-emptiness is comparable to historical admission readiness. Transport
malformation and Rust-only registration validation controls are labeled below.
Protocol/API and byte budgets: worker/policy/examples/source_admission_probe.rs.
"""

from __future__ import annotations

import hashlib
import os
import random
import selectors
import struct
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from worker.runtime.flow import metadata_slot as canonical
from worker.types import metadata as metadata_types
from worker.types import perception_frame as perception
from worker.types.metadata import MetadataFrame, SourceBinding

_ENV = "SEEON_TEST_SOURCE_ADMISSION_PROBE"
_FROZEN_SOURCES = (
    (canonical, "a5d704c502ba5c04e78a09863482a13d7fdf55e4fc4d43f825014c43298ae5c0"),
    (metadata_types, "94edd4502b2e2a5d224485d2da5e35995a10cceca444914626bb787a1b52e64e"),
    (perception, "fe70153a22348b4cd044372afb888b7dd104322f86d3356fde351a539536edf6"),
)
_MAGIC = b"SRCADM01"
_MAX_INPUT = 128 * 1024
_MAX_OUTPUT = 1024 * 1024
_TIMEOUT = 10.0
_I64_MIN, _I64_MAX, _U64_MAX = -(1 << 63), (1 << 63) - 1, (1 << 64) - 1
_TRANSPORT = "fixed-native-handle\x00\t\n운송"
_CHILD = "abcdef01-2345-6789-abcd-ef0123456789"
_OTHER_CHILD = "01234567-89ab-cdef-0123-456789abcdef"
_BASE = SourceBinding("worker-boot", _CHILD, "camera-a", 7, 11, "transform-a")
_LONG = "시작é\x00\n\t끝" * 256 + "é"
_STATUS = (
    "registered",
    "accepted",
    "malformed",
    "invalid_child_instance_id",
    "unknown_source",
    "boot_mismatch",
    "child_mismatch",
    "generation_mismatch",
    "epoch_mismatch",
    "transform_mismatch",
    "transport_mismatch",
    "pts_missing",
    "discarded_publication",
    "non_increasing_pts",
    "non_increasing_canonical_sequence",
    "non_increasing_native_publication_sequence",
    "regressing_fence",
    "generation_exhausted",
    "epoch_exhausted",
)
_PYTHON_OBSERVABLE = {
    "non_increasing_pts": "late",
    "non_increasing_canonical_sequence": "late",
    "non_increasing_native_publication_sequence": "late",
}
_Operation = SourceBinding | MetadataFrame
_Water = tuple[int, int, int]


def _probe_path() -> str:
    configured = os.environ.get(_ENV)
    if configured is None:
        pytest.skip("source-admission opt-in is absent; skip is not qualification")
    if not configured.strip():
        pytest.fail("configured source-admission probe must name an executable", pytrace=False)
    try:
        executable = Path(configured).expanduser().resolve()
        usable = executable.is_file() and os.access(executable, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        pytest.fail("configured source-admission probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured source-admission probe is missing or unexecutable", pytrace=False)
    return str(executable)


def _configured_probe() -> str:
    executable = _probe_path()
    for module, expected in _FROZEN_SOURCES:
        try:
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        except (OSError, TypeError, ValueError):
            pytest.fail(f"cannot read imported source-admission oracle {module.__name__}")
        if digest != expected:
            pytest.fail(
                f"source-admission oracle {module.__name__} differs from frozen SHA: {digest}",
                pytrace=False,
            )
    return executable


@pytest.fixture
def source_admission_probe() -> str:
    return _configured_probe()


def _invoke(probe: str, payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # One extra byte is permitted solely for the probe's input-budget control.
    assert len(payload) <= _MAX_INPUT + 1
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + _TIMEOUT
    try:
        with subprocess.Popen(
            [probe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            try:
                assert process.stdin is not None
                assert process.stdout is not None
                assert process.stderr is not None
                with selectors.DefaultSelector() as selector:
                    for name in ("stdin", "stdout", "stderr"):
                        stream = getattr(process, name)
                        os.set_blocking(stream.fileno(), False)
                        event = selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ
                        selector.register(stream, event, name)
                    sent = 0
                    view = memoryview(payload)
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired([probe], _TIMEOUT)
                        events = selector.select(remaining)
                        if not events:
                            raise subprocess.TimeoutExpired([probe], _TIMEOUT)
                        for key, _ in events:
                            stream = key.fileobj
                            if key.data == "stdin":
                                try:
                                    written = os.write(stream.fileno(), view[sent : sent + 65536])
                                except BlockingIOError:
                                    continue
                                except BrokenPipeError:
                                    written = 0
                                sent += written
                                if sent == len(payload) or written == 0:
                                    selector.unregister(stream)
                                    stream.close()
                            else:
                                target = buffers[key.data]
                                cap = _MAX_OUTPUT if key.data == "stdout" else 1024
                                try:
                                    chunk = os.read(
                                        stream.fileno(), min(65536, cap - len(target) + 1)
                                    )
                                except BlockingIOError:
                                    continue
                                if not chunk:
                                    selector.unregister(stream)
                                    stream.close()
                                else:
                                    target.extend(chunk)
                                    if len(target) > cap:
                                        pytest.fail(
                                            "source-admission probe exceeded output bounds",
                                            pytrace=False,
                                        )
                    returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pytest.fail("source-admission probe failed to execute within its bound", pytrace=False)
    return subprocess.CompletedProcess(
        [probe], returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    )


def _frame(
    binding: SourceBinding,
    pts: int | None,
    seq: int,
    native: int,
    *,
    child_spelling: str | None = None,
) -> MetadataFrame:
    identity = perception.PerceptionFrameIdentity(
        binding.worker_boot_id, binding.camera_id, binding.stream_epoch, seq, pts
    )
    frame = perception.PerceptionFrameV1(
        identity,
        perception.PersonBoxChannel(perception.ChannelState.SKIPPED),
        perception.HumanPoseChannel(perception.ChannelState.SKIPPED),
        perception.BedRegionChannel(perception.ChannelState.SKIPPED),
    )
    child = uuid.UUID(binding.child_instance_id if child_spelling is None else child_spelling)
    return MetadataFrame(frame, binding.source_generation, child, native, binding.transform_id)


def _metadata_binding(frame: MetadataFrame) -> SourceBinding:
    return SourceBinding(
        frame.identity.worker_boot_id,
        str(frame.child_instance_id),
        frame.identity.camera_id,
        frame.source_generation,
        frame.identity.stream_epoch,
        frame.transform_id,
    )


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _wire_binding(binding: SourceBinding) -> bytes:
    return b"".join(
        (
            _text(binding.worker_boot_id),
            _text(binding.child_instance_id),
            _text(binding.camera_id),
            struct.pack("<QQ", binding.source_generation, binding.stream_epoch),
            _text(binding.transform_id),
        )
    )


def _wire_frame(frame: MetadataFrame, transport: str = _TRANSPORT) -> bytes:
    pts = frame.identity.source_pts
    return b"".join(
        (
            b"\x02",
            _text(transport),
            _wire_binding(_metadata_binding(frame)),
            b"\0" if pts is None else b"\x01" + struct.pack("<q", pts),
            struct.pack("<QQ", frame.identity.seq, frame.native_publish_sequence),
        )
    )


def _payload(
    initial: SourceBinding, operations: Sequence[_Operation], transport: str = _TRANSPORT
) -> bytes:
    records = [
        b"\x01" + _wire_binding(op) if isinstance(op, SourceBinding) else _wire_frame(op, transport)
        for op in operations
    ]
    return (
        _MAGIC
        + struct.pack("<I", 1 + len(records))
        + _text(transport)
        + _wire_binding(initial)
        + b"".join(records)
    )


@dataclass(frozen=True)
class _Snapshot:
    result: str
    transport: str | None
    binding: SourceBinding | None
    ready: bool
    water: _Water | None
    retained: tuple[SourceBinding, _Water] | None


class _Cursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def take(self, count: int) -> bytes:
        end = self.position + count
        assert count >= 0 and end <= len(self.data), "truncated probe response"
        result = self.data[self.position : end]
        self.position = end
        return result

    def integer(self, fmt: str) -> int:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def flag(self) -> bool:
        value = self.integer("<B")
        assert value in (0, 1), "invalid probe response flag"
        return bool(value)

    def text(self) -> str:
        return self.take(self.integer("<I")).decode("utf-8")

    def binding(self) -> SourceBinding:
        return SourceBinding(
            self.text(),
            self.text(),
            self.text(),
            self.integer("<Q"),
            self.integer("<Q"),
            self.text(),
        )

    def water(self) -> _Water:
        return struct.unpack("<qQQ", self.take(24))


def _observations(
    probe: str, initial: SourceBinding, operations: Sequence[_Operation]
) -> list[_Snapshot]:
    payload = _payload(initial, operations)
    assert len(payload) <= _MAX_INPUT
    result = _invoke(probe, payload)
    assert result.returncode == 0 and result.stderr == b"", result
    cursor = _Cursor(result.stdout)
    assert cursor.take(12) == payload[:12]
    observations = []
    for _ in range(1 + len(operations)):
        code = cursor.integer("<B")
        assert code < len(_STATUS)
        transport, binding, ready, water = None, None, False, None
        if cursor.flag():
            transport, binding = cursor.text(), cursor.binding()
            ready = cursor.flag()
            water = cursor.water() if cursor.flag() else None
        retained = (cursor.binding(), cursor.water()) if cursor.flag() else None
        observations.append(_Snapshot(_STATUS[code], transport, binding, ready, water, retained))
    assert cursor.position == len(result.stdout), "trailing probe response"
    return observations


def _pair(
    probe: str,
    initial: SourceBinding,
    operations: Sequence[_Operation],
    expected: Sequence[str] | None = None,
) -> list[_Snapshot]:
    observations = _observations(probe, initial, operations)
    if expected is not None:
        assert len(expected) == len(observations)
    slot = canonical.LatestMetadataSlot()
    for index, (op, observed) in enumerate(zip([initial, *operations], observations, strict=True)):
        before = asdict(slot.counters())
        if isinstance(op, SourceBinding):
            # Restrict only the paired domain; malformed controls never enter here.
            assert op.camera_id == initial.camera_id and op.camera_id != ""
            assert str(uuid.UUID(op.child_instance_id)) == op.child_instance_id
            slot.register_source(op)
            assert asdict(slot.counters()) == before
            reason = "registered"
        else:
            previous = slot.peek(initial.camera_id)
            admitted = slot.publish(op)
            after = asdict(slot.counters())
            delta = {
                name: value - before[name] for name, value in after.items() if value != before[name]
            }
            if admitted:
                assert delta == {"accepted": 1, **({"overwritten": 1} if previous else {})}
                reason = "accepted"
            else:
                assert len(delta) == 1
                reason, increment = next(iter(delta.items()))
                assert increment == 1 and reason not in ("accepted", "overwritten", "pull_failures")
        latest = slot.peek(initial.camera_id)
        retained = None
        if latest is not None:
            assert latest.identity.source_pts is not None
            retained = (
                _metadata_binding(latest),
                (latest.identity.source_pts, latest.identity.seq, latest.native_publish_sequence),
            )
        reference = _Snapshot(
            reason,
            _TRANSPORT,
            slot.expected_binding(initial.camera_id),
            latest is not None,
            None if retained is None else retained[1],
            retained,
        )
        # This is the sole documented observable classification, not an oracle.
        comparable = replace(
            observed, result=_PYTHON_OBSERVABLE.get(observed.result, observed.result)
        )
        assert comparable == reference, f"operation {index}: {op!r}; raw Rust={observed!r}"
        if expected is not None:
            assert observed.result == expected[index], f"raw Rust reason at operation {index}"
    return observations


@pytest.mark.parametrize(
    "generation,epoch", [(0, 0), (_U64_MAX, 0), (0, _U64_MAX), (_U64_MAX, _U64_MAX)]
)
def test_first_missing_negative_pts_and_exact_integer_limits(
    source_admission_probe: str, generation: int, epoch: int
) -> None:
    binding = replace(_BASE, source_generation=generation, stream_epoch=epoch)
    _pair(
        source_admission_probe,
        binding,
        [
            _frame(binding, None, _U64_MAX, _U64_MAX),
            _frame(binding, _I64_MIN, 0, 0),
            _frame(binding, -1, 1, 3),
            _frame(binding, 0, 2, 4),
            _frame(binding, _I64_MAX - 1, _U64_MAX - 1, _U64_MAX - 1),
            _frame(binding, _I64_MAX, _U64_MAX, _U64_MAX),
            _frame(binding, _I64_MIN, 0, 0),
        ],
        ["registered", "pts_missing", *("accepted" for _ in range(5)), "non_increasing_pts"],
    )


@pytest.mark.parametrize(
    "pts,seq,native,reason",
    [
        (None, _U64_MAX, _U64_MAX, "pts_missing"),
        (10, _U64_MAX, _U64_MAX, "non_increasing_pts"),
        (9, _U64_MAX, _U64_MAX, "non_increasing_pts"),
        (_I64_MAX, 20, _U64_MAX, "non_increasing_canonical_sequence"),
        (_I64_MAX, 19, _U64_MAX, "non_increasing_canonical_sequence"),
        (_I64_MAX, _U64_MAX, 30, "non_increasing_native_publication_sequence"),
        (_I64_MAX, _U64_MAX, 29, "non_increasing_native_publication_sequence"),
        (10, 20, 30, "non_increasing_pts"),
        (11, 20, 30, "non_increasing_canonical_sequence"),
    ],
)
def test_independent_highwaters_and_combined_faults_are_atomic(
    source_admission_probe: str, pts: int | None, seq: int, native: int, reason: str
) -> None:
    _pair(
        source_admission_probe,
        _BASE,
        [
            _frame(_BASE, 10, 20, 30),
            _frame(_BASE, pts, seq, native),
            # Lower than the rejected candidate's other dimensions: detects poisoning.
            _frame(_BASE, 11, 21, 31),
            _frame(_BASE, 12, 22, 100),
            _frame(_BASE, 13, 200, 101),
        ],
        ["registered", "accepted", reason, "accepted", "accepted", "accepted"],
    )


_IDENTITY_FAULTS = (
    ({"camera_id": ""}, "malformed"),
    ({"camera_id": "unknown-camera"}, "unknown_source"),
    ({"worker_boot_id": "other-boot"}, "boot_mismatch"),
    ({"child_instance_id": _OTHER_CHILD}, "child_mismatch"),
    ({"source_generation": 0}, "generation_mismatch"),
    ({"stream_epoch": 0}, "epoch_mismatch"),
    ({"transform_id": "other-transform"}, "transform_mismatch"),
)


@pytest.mark.parametrize("prime", [False, True])
@pytest.mark.parametrize("changes,reason", _IDENTITY_FAULTS)
def test_each_identity_refusal_preserves_unready_and_ready_state(
    source_admission_probe: str, prime: bool, changes: dict, reason: str
) -> None:
    operations = [_frame(_BASE, 10, 20, 30)] if prime else []
    operations.extend(
        [
            _frame(replace(_BASE, **changes), _I64_MAX, _U64_MAX, _U64_MAX),
            _frame(_BASE, 11, 21, 31),
        ]
    )
    _pair(
        source_admission_probe,
        _BASE,
        operations,
        ["registered", *(["accepted"] if prime else []), reason, "accepted"],
    )


@pytest.mark.parametrize("prime", [False, True])
def test_combined_identity_fault_precedence_before_missing_pts_and_lateness(
    source_admission_probe: str, prime: bool
) -> None:
    wrong = SourceBinding("other-boot", _OTHER_CHILD, "", 0, 0, "other-transform")
    candidates = [wrong, replace(wrong, camera_id="unknown-camera")]
    # Fixed repair order is input generation, not dependent on either decision.
    for field in (
        "camera_id",
        "worker_boot_id",
        "child_instance_id",
        "source_generation",
        "stream_epoch",
        "transform_id",
    ):
        wrong = replace(wrong, **{field: getattr(_BASE, field)})
        candidates.append(wrong)
    operations = [_frame(_BASE, 10, 20, 30)] if prime else []
    operations.extend(_frame(candidate, None, 0, 0) for candidate in candidates)
    operations.append(_frame(_BASE, 11, 21, 31))
    _pair(
        source_admission_probe,
        _BASE,
        operations,
        [
            "registered",
            *(["accepted"] if prime else []),
            "malformed",
            "unknown_source",
            "boot_mismatch",
            "child_mismatch",
            "generation_mismatch",
            "epoch_mismatch",
            "transform_mismatch",
            "pts_missing",
            "accepted",
        ],
    )


@pytest.mark.parametrize("value", ["", _LONG], ids=["empty", "long-unicode"])
@pytest.mark.parametrize(
    "field,reason", [("worker_boot_id", "boot_mismatch"), ("transform_id", "transform_mismatch")]
)
def test_identity_strings_are_exact_utf8_not_trimmed_normalized_or_truncated(
    source_admission_probe: str, value: str, field: str, reason: str
) -> None:
    binding = replace(_BASE, worker_boot_id=value, transform_id=value, camera_id="카메라\x00\n\t ")
    different = (value + "suffix", "" if value else _LONG, value[:-1] + "é" if value else " ")
    operations = [_frame(binding, -10, 0, 0)]
    expected = ["registered", "accepted"]
    for index, mismatch in enumerate(different, start=1):
        operations.extend(
            [
                _frame(replace(binding, **{field: mismatch}), _I64_MAX, _U64_MAX, _U64_MAX),
                _frame(binding, -10 + index, index, index),
            ]
        )
        expected.extend((reason, "accepted"))
    _pair(source_admission_probe, binding, operations, expected)


def test_frame_uuid_spellings_use_actual_uuid_normalization(source_admission_probe: str) -> None:
    spellings = [
        _CHILD,
        _CHILD.upper(),
        _CHILD.replace("-", ""),
        "{" + _CHILD.upper() + "}",
        "urn:uuid:" + _CHILD,
    ]
    operations = [
        _frame(_BASE, index, index, index, child_spelling=spelling)
        for index, spelling in enumerate(spellings)
    ]
    _pair(
        source_admission_probe, _BASE, operations, ["registered", *(["accepted"] * len(spellings))]
    )


def test_equal_binding_reregistration_resets_readiness_and_all_ordinals(
    source_admission_probe: str,
) -> None:
    _pair(
        source_admission_probe,
        _BASE,
        [
            _frame(_BASE, _I64_MAX, _U64_MAX, _U64_MAX),
            _BASE,
            _frame(_BASE, None, _U64_MAX, _U64_MAX),
            _frame(_BASE, _I64_MIN, 0, 0),
            _BASE,
            _frame(_BASE, _I64_MIN, 0, 0),
        ],
        [
            "registered",
            "accepted",
            "registered",
            "pts_missing",
            "accepted",
            "registered",
            "accepted",
        ],
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        *_IDENTITY_FAULTS[2:],
        (
            {
                "worker_boot_id": "replacement-boot",
                "child_instance_id": "00000000-0000-0000-0000-000000000000",
                "source_generation": 0,
                "stream_epoch": 0,
                "transform_id": "replacement-transform",
            },
            "boot_mismatch",
        ),
    ],
)
def test_changed_binding_invalidates_old_headers_but_not_transport_identity(
    source_admission_probe: str, changes: dict, reason: str
) -> None:
    replacement = replace(_BASE, **changes)
    _pair(
        source_admission_probe,
        _BASE,
        [
            _frame(_BASE, _I64_MAX - 1, _U64_MAX - 1, _U64_MAX - 1),
            replacement,
            _frame(_BASE, _I64_MAX, _U64_MAX, _U64_MAX),
            _frame(replacement, _I64_MIN, 0, 0),
            _frame(_BASE, _I64_MAX, _U64_MAX, _U64_MAX),
            _frame(replacement, -1, 1, 1),
        ],
        ["registered", "accepted", "registered", reason, "accepted", reason, "accepted"],
    )


def test_seeded_trace_is_generated_without_observed_decisions(source_admission_probe: str) -> None:
    rng = random.Random(0x6002)
    operations: list[_Operation] = []
    for cycle in range(3):
        binding = replace(_BASE, source_generation=cycle, worker_boot_id=f"boot-{cycle}")
        operations.extend((binding, _frame(binding, _I64_MIN, 0, 0)))
        for index in range(1, 23):
            candidate = binding
            fault = rng.randrange(9)
            if fault < len(_IDENTITY_FAULTS):
                changes, _ = _IDENTITY_FAULTS[fault]
                candidate = replace(binding, **changes)
            operations.append(
                _frame(
                    candidate,
                    rng.choice((None, -1, index - 1, index)),
                    rng.randrange(24),
                    rng.randrange(24),
                )
            )
        operations.append(_frame(binding, 1000, 1000, 1000))
    _pair(source_admission_probe, _BASE, operations)


def test_transport_accepts_exact_input_and_operation_bounds(source_admission_probe: str) -> None:
    extra = _MAX_INPUT - len(_payload(_BASE, []))
    binding = replace(_BASE, worker_boot_id=_BASE.worker_boot_id + "x" * extra)
    assert len(_payload(binding, [])) == _MAX_INPUT
    _pair(source_admission_probe, binding, [], ["registered"])
    operations = [_frame(_BASE, index, index * 2, index * 3) for index in range(127)]
    _pair(source_admission_probe, _BASE, operations, ["registered", *(["accepted"] * 127)])


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"camera_id": ""}, "malformed"),
        ({"camera_id": "", "child_instance_id": ""}, "malformed"),
        *[
            ({"child_instance_id": child}, "invalid_child_instance_id")
            for child in (
                "",
                _CHILD.upper(),
                _CHILD.replace("-", ""),
                "{" + _CHILD + "}",
                "g" + _CHILD[1:],
                _CHILD + "0",
                _CHILD[:-1] + "가",
            )
        ],
    ],
)
def test_rust_only_registration_validation_is_not_python_parity(
    source_admission_probe: str, changes: dict, reason: str
) -> None:
    invalid = replace(_BASE, **changes)
    (unregistered,) = _observations(source_admission_probe, invalid, [])
    assert unregistered == _Snapshot(reason, None, None, False, None, None)
    observations = _observations(
        source_admission_probe,
        _BASE,
        [_frame(_BASE, 10, 20, 30), invalid, _frame(_BASE, 11, 21, 31)],
    )
    assert [row.result for row in observations] == ["registered", "accepted", reason, "accepted"]
    assert replace(observations[2], result="accepted") == observations[1]
    assert observations[3].water == (11, 21, 31)
    assert observations[3].retained == (_BASE, (11, 21, 31))
    assert observations[3].binding == _BASE and observations[3].ready
    assert all(row.transport == _TRANSPORT for row in observations)
    # Explicitly expose, rather than hide, the canonical constructor distinction.
    slot = canonical.LatestMetadataSlot()
    slot.register_source(invalid)
    assert slot.expected_binding(invalid.camera_id) == invalid


@pytest.mark.parametrize(
    "defect",
    [
        "magic",
        "zero-operations",
        "operation-bound",
        "short-header",
        "utf8",
        "text-length",
        "truncated",
        "trailing",
        "opcode",
        "pts-tag",
        "late-truncation",
        "failed-initial-with-operation",
        "input-bound",
        "output-bound",
    ],
)
def test_probe_malformed_transport_refuses_atomically_not_domain_parity(
    source_admission_probe: str, defect: str
) -> None:
    payload = _payload(_BASE, [])
    if defect == "magic":
        payload = b"BADMAGIC" + payload[8:]
    elif defect == "zero-operations":
        payload = _MAGIC + struct.pack("<I", 0)
    elif defect == "operation-bound":
        payload = _MAGIC + struct.pack("<I", 129)
    elif defect == "short-header":
        payload = payload[:10]
    elif defect == "utf8":
        payload = _MAGIC + struct.pack("<II", 1, 1) + b"\xff"
    elif defect == "text-length":
        payload = _MAGIC + struct.pack("<II", 1, 0xFFFFFFFF)
    elif defect == "truncated":
        payload = payload[:-1]
    elif defect == "trailing":
        payload += b"\0"
    elif defect == "opcode":
        payload = _MAGIC + struct.pack("<I", 2) + payload[12:] + b"\x03"
    elif defect == "pts-tag":
        record = _wire_frame(_frame(_BASE, None, 0, 0))
        record = record[:-17] + b"\x02" + record[-16:]
        payload = _MAGIC + struct.pack("<I", 2) + payload[12:] + record
    elif defect == "late-truncation":
        payload = _payload(_BASE, [_frame(_BASE, 1, 1, 1)])[:-1]
    elif defect == "failed-initial-with-operation":
        payload = _payload(replace(_BASE, camera_id=""), [_frame(_BASE, 1, 1, 1)])
    elif defect == "input-bound":
        payload = payload.ljust(_MAX_INPUT + 1, b"x")
    else:
        assert defect == "output-bound"
        binding = replace(_BASE, worker_boot_id="x" * 9000)
        payload = _payload(
            binding, [_frame(binding, 1, 1, 1), *[_frame(_BASE, 2, 2, 2) for _ in range(126)]]
        )
        assert len(payload) < _MAX_INPUT
    result = _invoke(source_admission_probe, payload)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"source-admission-probe: rejected\n"


def test_absent_opt_in_skips_without_qualifying(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="not qualification"):
        _configured_probe()


@pytest.mark.parametrize(
    "setting", ["empty", "whitespace", "missing", "directory", "unexecutable", "nul"]
)
def test_invalid_explicit_opt_in_fails_not_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, setting: str
) -> None:
    unexecutable = tmp_path / "unexecutable"
    unexecutable.write_bytes(b"not executable")
    unexecutable.chmod(0o600)
    value = {
        "empty": "",
        "whitespace": " \t\n",
        "missing": str(tmp_path / "absent"),
        "directory": str(tmp_path),
        "unexecutable": str(unexecutable),
        "nul": "bad\0path",
    }[setting]
    # The OS rejects NUL before the configuration validator can observe it.
    monkeypatch.setattr(os, "environ", {_ENV: value})
    with pytest.raises(pytest.fail.Exception, match="configured source-admission probe"):
        _configured_probe()


@pytest.mark.parametrize("source_index", range(3))
@pytest.mark.parametrize("unreadable", [False, True])
def test_each_imported_oracle_hash_is_required_on_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source_index: int, unreadable: bool
) -> None:
    executable = tmp_path / "path-check-only"
    executable.write_bytes(b"not invoked by this configuration test")
    executable.chmod(0o700)
    monkeypatch.setenv(_ENV, str(executable))
    replacement = tmp_path / "changed-oracle.py"
    if not unreadable:
        replacement.write_bytes(b"different oracle source\n")
    module, _ = _FROZEN_SOURCES[source_index]
    # Isolate this negative control from unrelated source drift in opt-out CI.
    # The real probe fixture still checks all three unmodified source pins.
    monkeypatch.setitem(globals(), "_FROZEN_SOURCES", (_FROZEN_SOURCES[source_index],))
    monkeypatch.setattr(module, "__file__", str(replacement))
    message = (
        "cannot read imported source-admission oracle" if unreadable else "differs from frozen SHA"
    )
    with pytest.raises(pytest.fail.Exception, match=message) as failure:
        _configured_probe()
    assert module.__name__ in str(failure.value)

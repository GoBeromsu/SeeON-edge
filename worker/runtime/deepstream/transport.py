"""Leak-safe inherited socketpair and identity-pipe child spawn."""

from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

from worker.runtime.deepstream.config import ChildConfig


@dataclass(frozen=True, slots=True)
class SpawnRequest:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    pass_fds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ChildTransport:
    process: subprocess.Popen[bytes]
    control: socket.socket
    wake: socket.socket
    access_units: socket.socket
    previews: socket.socket
    failures: socket.socket
    ready_fd: int


def spawn_process(request: SpawnRequest) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - image-owned executable
        request.command,
        env=request.environment,
        pass_fds=request.pass_fds,
        stdin=subprocess.DEVNULL,
        # The child owns decode, inference, parsing and association, so it is
        # the only component that can report why any of them failed. Its
        # diagnostics - GStreamer, TensorRT, CUDA and our own - are written to
        # stderr, and discarding that stream leaves the entire media plane
        # unobservable to an operator. Inherit the supervisor's stderr so the
        # child reaches the container log. stdout stays discarded: nothing in
        # the child writes structured output there, and the IPC contract runs
        # over inherited sockets rather than a pipe.
        stdout=subprocess.DEVNULL,
        stderr=None,
    )


def _close_sockets(sockets: list[socket.socket]) -> None:
    for endpoint in sockets:
        endpoint.close()


def _close_fds(descriptors: list[int]) -> None:
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


# The kernel default (~200 KiB) holds well under a second of fleet video, so
# any stall on the worker's drain thread backed up into the child's sender
# and shed units (#429). Ask for the maximum; the kernel clamps to
# net.core.{rmem,wmem}_max.
AU_SOCKET_BUFFER_BYTES = 4 * 1024 * 1024


def widen_access_unit_buffers(parent: socket.socket, child: socket.socket) -> None:
    parent.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, AU_SOCKET_BUFFER_BYTES)
    child.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, AU_SOCKET_BUFFER_BYTES)


def spawn_child(config: ChildConfig) -> ChildTransport:
    sockets: list[socket.socket] = []
    descriptors: list[int] = []
    try:
        control_parent, control_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sockets.extend((control_parent, control_child))
        wake_parent, wake_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        sockets.extend((wake_parent, wake_child))
        au_parent, au_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        sockets.extend((au_parent, au_child))
        widen_access_unit_buffers(au_parent, au_child)
        preview_parent, preview_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        sockets.extend((preview_parent, preview_child))
        failure_parent, failure_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sockets.extend((failure_parent, failure_child))
        identity_read, identity_write = os.pipe()
        descriptors.extend((identity_read, identity_write))
        ready_read, ready_write = os.pipe()
        descriptors.extend((ready_read, ready_write))
        identity = (
            config.worker_boot_id.bytes
            + config.child_instance_id.bytes
            + os.getpid().to_bytes(4, "little")
        )
        _ = os.write(identity_write, identity)
        os.close(identity_write)
        descriptors.remove(identity_write)
        command = (
            str(config.executable),
            "--control-fd",
            str(control_child.fileno()),
            "--wake-fd",
            str(wake_child.fileno()),
            "--au-fd",
            str(au_child.fileno()),
            "--preview-fd",
            str(preview_child.fileno()),
            "--failure-fd",
            str(failure_child.fileno()),
            "--identity-fd",
            str(identity_read),
            "--ready-fd",
            str(ready_write),
            "--qa-mode",
            "1" if config.qa_mode else "0",
            "--box-source",
            config.box_source,
            "--engine-cache",
            str(config.engine_cache),
            "--target-fps",
            str(config.target_fps),
        )
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = "0"
        process = spawn_process(
            SpawnRequest(
                command,
                child_env,
                (
                    control_child.fileno(),
                    wake_child.fileno(),
                    au_child.fileno(),
                    preview_child.fileno(),
                    failure_child.fileno(),
                    identity_read,
                    ready_write,
                ),
            )
        )
    except BaseException:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - resource boundary
        _close_sockets(sockets)
        _close_fds(descriptors)
        raise
    control_child.close()
    wake_child.close()
    au_child.close()
    preview_child.close()
    failure_child.close()
    os.close(identity_read)
    os.close(ready_write)
    return ChildTransport(
        process,
        control_parent,
        wake_parent,
        au_parent,
        preview_parent,
        failure_parent,
        ready_read,
    )


__all__ = ["ChildTransport", "SpawnRequest", "spawn_child", "spawn_process"]

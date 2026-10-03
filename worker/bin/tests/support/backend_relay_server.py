"""Real no-lifespan relay + PostgreSQL sandbox; test-only lost-response TCP proxy."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from contextlib import suppress
from pathlib import Path

import psycopg
import uvicorn
from psycopg.rows import dict_row

from tests_support.postgres_sandbox import open_product_sandbox, product_audit_runtime
from tests_support.relay_postgres_runtime import relay_postgres_app

HEAD_LIMIT = 16 * 1024
BODY_LIMIT = 64 * 1024
WIRE_TIMEOUT = 10
CONTROL_TIMEOUT = 90


async def emit(message):
    data = (json.dumps(message, default=str, allow_nan=False) + "\n").encode()
    if len(data) > BODY_LIMIT:
        raise ValueError("control response exceeds cap")
    deadline = asyncio.get_running_loop().time() + 2
    while data:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("control output deadline")
        try:
            count = os.write(sys.stdout.fileno(), data)
            if count == 0:
                raise BrokenPipeError("control output closed")
            data = data[count:]
        except BlockingIOError:
            await asyncio.sleep(0.01)


async def close_writer(writer):
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=1)
    except (TimeoutError, OSError):
        writer.transport.abort()


async def message(reader):
    head = await reader.readuntil(b"\r\n\r\n")
    if len(head) > HEAD_LIMIT:
        raise ValueError("HTTP head exceeds cap")
    headers = {}
    for line in head.split(b"\r\n")[1:-2]:
        name, value = line.split(b":", 1)
        name = name.strip().lower()
        if name in headers:
            raise ValueError("duplicate HTTP header")
        headers[name] = value.strip()
    if b"transfer-encoding" in headers:
        raise ValueError("this bounded fixture requires Content-Length framing")
    length = int(headers[b"content-length"])
    if not 0 <= length <= BODY_LIMIT:
        raise ValueError("HTTP body exceeds cap")
    return head, await reader.readexactly(length)


class LostResponseProxy:
    def __init__(self, backend, edge_event_id, entry_path):
        self.backend = backend
        self.edge_event_id = edge_event_id
        self.entry_path = entry_path
        self.original_entry = entry_path.read_bytes()
        self.connections = 0
        self.observed = []
        self.errors = []
        self.tasks = set()
        self.server = None

    async def start(self):
        self.server = await asyncio.start_server(
            self.accept, "127.0.0.1", 0, limit=HEAD_LIMIT, backlog=2
        )
        return self.server.sockets[0].getsockname()

    def accept(self, reader, writer):
        task = asyncio.create_task(self.handle(reader, writer))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def handle(self, reader, writer):
        self.connections += 1
        ordinal = self.connections
        try:
            await asyncio.wait_for(self.forward(reader, writer, ordinal), WIRE_TIMEOUT)
        except (OSError, ValueError, LookupError, asyncio.IncompleteReadError) as error:
            self.errors.append(type(error).__name__)
        finally:
            await close_writer(writer)

    async def forward(self, reader, writer, ordinal):
        if ordinal > 2:
            raise ValueError("unexpected implicit retry or additional HTTP request")
        head, body = await message(reader)
        request_line = head.split(b"\r\n", 1)[0]
        if request_line != b"POST /api/v1/relay/alerts HTTP/1.1":
            raise ValueError("only the real relay alert route is in this test")
        upstream_reader, upstream_writer = await asyncio.open_connection(
            *self.backend, limit=HEAD_LIMIT
        )
        try:
            # Forward the entire worker request byte-for-byte, including Host.
            upstream_writer.write(head + body)
            await upstream_writer.drain()
            response_head, response_body = await message(upstream_reader)
            status = int(response_head.split(b"\r\n", 1)[0].split()[1])
            receipt = json.loads(response_body)
            try:
                queued = self.entry_path.read_bytes()
            except FileNotFoundError:
                queued = None
            observation = {
                "request_line": request_line.decode("ascii"),
                "request_body": json.loads(body),
                "request_bytes": body.decode("utf-8"),
                "backend_status": status,
                "backend_body": receipt,
                "dropped": False,
                "queue_unchanged_before_response": queued == self.original_entry,
            }
            self.observed.append(observation)
            if not 200 <= status < 300 or receipt != {
                "status": "accepted_local",
                "edge_event_id": self.edge_event_id,
            }:
                raise ValueError("real backend did not return local acceptance")
            # Only a completely observed, genuine backend success is lost.
            # The first client receives zero response bytes; no fake ACK exists.
            if ordinal == 1:
                observation["dropped"] = True
            else:
                writer.write(response_head + response_body)
                await writer.drain()
        finally:
            await close_writer(upstream_writer)

    async def idle(self):
        if self.tasks:
            await asyncio.wait_for(asyncio.gather(*tuple(self.tasks)), WIRE_TIMEOUT + 3)

    async def close(self):
        if self.server is not None:
            self.server.close()
            await asyncio.wait_for(self.server.wait_closed(), timeout=1)
        await self.idle()


def snapshot(sandbox, proxy):
    with sandbox.admin.cursor(row_factory=dict_row) as cursor:

        def rows(query):
            cursor.execute(query)
            return cursor.fetchall()

        return {
            "incidents": rows("SELECT * FROM incidents ORDER BY edge_event_id"),
            "outbox": rows("SELECT * FROM event_outbox ORDER BY edge_event_id"),
            "relay_alert_audit": rows(
                "SELECT * FROM audit_events WHERE action='relay.alert' ORDER BY audit_id"
            ),
            "connections": proxy.connections,
            "requests": proxy.observed,
            "proxy_errors": proxy.errors,
        }


async def serve(sandbox, alert, entry_path):
    audit = product_audit_runtime(sandbox)
    app = relay_postgres_app(
        sandbox, audit, camera_id=alert["camera_id"], backend_camera_id=alert["camera_id"]
    )
    # No central client and no lifespan/background sender: accepted_local is real.
    proxy = None
    transport = None
    server_task = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(8)
            listener.setblocking(False)
            backend = listener.getsockname()
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    lifespan="off",
                    access_log=False,
                    log_config=None,
                    timeout_keep_alive=1,
                    timeout_graceful_shutdown=2,
                )
            )
            server_task = asyncio.create_task(server.serve(sockets=[listener]))
            deadline = asyncio.get_running_loop().time() + 5
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError("backend exited before readiness")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("uvicorn readiness deadline")
                await asyncio.sleep(0.01)
            proxy = LostResponseProxy(backend, alert["edge_event_id"], entry_path)
            address = await proxy.start()
            reader = asyncio.StreamReader(limit=4096)
            transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
            )
            await emit({"kind": "ready", "proxy_address": f"{address[0]}:{address[1]}"})
            for _ in range(4):  # empty/before-retry/after-retry snapshots, then stop
                line = await asyncio.wait_for(reader.readuntil(b"\n"), CONTROL_TIMEOUT)
                command = json.loads(line)
                if command["op"] == "stop":
                    return command["id"]
                if command["op"] != "snapshot":
                    raise ValueError("unknown test-only control command")
                await proxy.idle()
                await emit({"id": command["id"], "snapshot": snapshot(sandbox, proxy)})
            raise ValueError("control command budget exhausted")
        finally:
            if transport is not None:
                transport.close()
            try:
                if proxy is not None:
                    await proxy.close()
            finally:
                try:
                    if server_task is not None:
                        server.should_exit = True
                        await asyncio.wait_for(server_task, timeout=5)
                finally:
                    audit.stop()
                    if not audit.close_session_once():
                        raise RuntimeError("owned audit session did not close after drain")


async def main():
    if sys.flags.optimize:
        raise RuntimeError("optimized Python removes required canonical audit admission checks")
    dsn = os.environ["SEEON_TEST_POSTGRES_DSN"]
    if not dsn.strip() or "\x00" in dsn:
        raise ValueError("explicit isolated test DSN required")
    alert = json.loads(Path(sys.argv[1]).read_bytes())
    with (
        psycopg.connect(dsn, autocommit=True, connect_timeout=5) as admin,
        open_product_sandbox(admin, dsn) as sandbox,
    ):
        stop_id = await serve(sandbox, alert, Path(sys.argv[2]))
    # The parent sees this only after pool shutdown and owned schema deletion.
    await emit({"id": stop_id, "stopped": True})


if __name__ == "__main__":
    os.set_blocking(sys.stdout.fileno(), False)
    try:
        asyncio.run(main())
    except (OSError, RuntimeError, ValueError, LookupError, psycopg.Error) as error:
        # Do not print a libpq exception/DSN, even on failed preconditions.
        with suppress(OSError, TimeoutError):
            asyncio.run(emit({"error": type(error).__name__}))
        sys.exit(1)

# /// script
# requires-python = ">=3.11"
# ///
# --- How to run ---
# CANARY_RELAY_TOKEN=<run-token> CANARY_RECEIPT_DIR=<dir> python relay_stub.py

"""Egress-free canary relay sink that stores exact request bytes."""

from __future__ import annotations

import hashlib
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

TOKEN: Final = os.environ["CANARY_RELAY_TOKEN"]
RECEIPTS: Final = Path(os.environ["CANARY_RECEIPT_DIR"])
WORKER_CONFIG_DIR: Final = Path(os.environ["CANARY_WORKER_CONFIG_DIR"])
ACTIVE_CONFIG: Final = Path(os.environ["CANARY_ACTIVE_CONFIG"])
MAX_BODY: Final = 2 * 1024 * 1024


class RelayHandler(BaseHTTPRequestHandler):
    server_version = "SeeONCanaryRelay/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health/live":
            body = b'{"status":"ok"}\n'
        elif self.path == "/api/v1/cameras/worker-config":
            if self.headers.get("X-Edge-Relay-Token", "") != TOKEN:
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            selected = ACTIVE_CONFIG.read_text(encoding="utf-8").strip()
            body = (WORKER_CONFIG_DIR / f"worker-{selected}.json").read_bytes()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        authorization = self.headers.get("Authorization", "")
        relay_token = self.headers.get("X-Edge-Relay-Token", "")
        if authorization not in {f"Bearer {TOKEN}", TOKEN} and relay_token != TOKEN:
            self.send_error(HTTPStatus.UNAUTHORIZED)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        body = self.rfile.read(length)
        digest = hashlib.sha256(body).hexdigest()
        stamp = time.time_ns()
        destination = RECEIPTS / f"relay-{stamp}-{digest}.json"
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as target:
            target.write(body)
        self.send_response(HTTPStatus.ACCEPTED)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"accepted"}\n')

    def log_message(self, format: str, *args: str) -> None:
        print(f"relay_stub {self.address_string()} {format % args}", flush=True)


if __name__ == "__main__":
    RECEIPTS.mkdir(parents=True, exist_ok=True, mode=0o700)
    ThreadingHTTPServer(("0.0.0.0", 8000), RelayHandler).serve_forever()

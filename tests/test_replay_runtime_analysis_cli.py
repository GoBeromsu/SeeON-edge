from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from shared.events.replay_wire import MAX_REPLAY_BODY_BYTES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ops" / "replay-runtime-analysis.py"


class _ReplayHandler(BaseHTTPRequestHandler):
    body = b"{}"
    status = 200

    def do_POST(self) -> None:
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _serve(body: bytes, *, status: int = 200) -> ThreadingHTTPServer:
    handler = type(
        "BoundReplayHandler",
        (_ReplayHandler,),
        {"body": body, "status": status},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, name="replay-cli-fake", daemon=True)
    thread.start()
    return server


def _run_cli(tmp_path: Path, worker_url: str) -> subprocess.CompletedProcess[str]:
    trace = tmp_path / "trace.json"
    _ = trace.write_text(json.dumps({"camera_id": "camera-replay-http"}), encoding="utf-8")
    database = tmp_path / "never-opened.sqlite3"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--database",
            str(database),
            "--camera-id",
            "camera-replay-http",
            "--worker-url",
            worker_url,
            "--relay-token",
            "relay-token",
            "--module-id",
            "bed_exit",
            "--policy-json",
            json.dumps({"ok": True}),
            "--requested-by",
            "test-operator",
            "--trace-json",
            str(trace),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_typed_refusal(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert completed.returncode == 2
    assert "KeyError" not in completed.stderr
    assert "ValueError" not in completed.stderr
    assert "Traceback" not in completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "refused"
    assert isinstance(payload["detail"], str)
    assert payload["detail"]
    return payload


@pytest.mark.parametrize(
    ("body", "detail_fragment"),
    [
        (b'{"reproducible": true}', "event_count"),
        (b'{"reproducible": true, "event_count": "1"}', "event_count"),
        (b'{"reproducible": true, "event_count": 1.5}', "event_count"),
        (b'{"reproducible": true, "event_count": -1}', "event_count"),
        (b'{"reproducible": true, "event_count": true}', "event_count"),
        (b'{"event_count": 0}', "reproducible"),
        (b'{"reproducible": "true", "event_count": 0}', "reproducible"),
        (b'{"reproducible": true, "event_count":', "JSON"),
        (b"{not-json", "JSON"),
    ],
)
def test_packaged_replay_cli_refuses_malformed_worker_payloads(
    tmp_path: Path, body: bytes, detail_fragment: str
) -> None:
    server = _serve(body)
    try:
        host, port = server.server_address
        completed = _run_cli(tmp_path, f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
    payload = _assert_typed_refusal(completed)
    assert detail_fragment.lower() in payload["detail"].lower()
    assert not (tmp_path / "never-opened.sqlite3").exists()


def test_packaged_replay_cli_accepts_payload_without_reasons_field(tmp_path: Path) -> None:
    body = json.dumps({"reproducible": True, "event_count": 0}).encode()
    server = _serve(body)
    try:
        host, port = server.server_address
        completed = _run_cli(tmp_path, f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["event_count"] == 0


def test_packaged_replay_cli_refuses_truncated_content_length(tmp_path: Path) -> None:
    full = b'{"reproducible": true, "event_count": 0}'

    class _Truncating(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(full)))
            self.end_headers()
            self.wfile.write(full[:12])

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Truncating)
    thread = Thread(target=server.serve_forever, name="replay-cli-trunc", daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        completed = _run_cli(tmp_path, f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
    payload = _assert_typed_refusal(completed)
    assert "truncat" in payload["detail"].lower()


def test_packaged_replay_cli_refuses_over_limit_content_length(tmp_path: Path) -> None:
    class _Oversize(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(MAX_REPLAY_BODY_BYTES + 1))
            self.end_headers()
            self.wfile.write(b"x")

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Oversize)
    thread = Thread(target=server.serve_forever, name="replay-cli-oversize", daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        completed = _run_cli(tmp_path, f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
    payload = _assert_typed_refusal(completed)
    assert str(MAX_REPLAY_BODY_BYTES) in payload["detail"]
    assert completed.returncode == 2


def test_packaged_replay_cli_accepts_5000_digit_event_count(tmp_path: Path) -> None:
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        digits = "1" * 5_000
        body = (
            '{"reproducible": true, "event_count": ' + digits + "}"
        ).encode()
        assert len(body) < 6_144
        server = _serve(body)
        try:
            host, port = server.server_address
            completed = _run_cli(tmp_path, f"http://{host}:{port}")
        finally:
            server.shutdown()
            server.server_close()
        assert completed.returncode == 0
        assert "Traceback" not in completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["event_count"] == int(digits)
        assert payload["reproducible"] is True
        assert not (tmp_path / "never-opened.sqlite3").exists()
    finally:
        sys.set_int_max_str_digits(previous)


def test_packaged_replay_cli_refuses_incomplete_chunked_transfer(tmp_path: Path) -> None:
    class _IncompleteChunked(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"20\r\n{\"reproducible\": true,\r\n")

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _IncompleteChunked)
    thread = Thread(target=server.serve_forever, name="replay-cli-chunked", daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        completed = _run_cli(tmp_path, f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
    payload = _assert_typed_refusal(completed)
    assert "truncat" in payload["detail"].lower()
    assert "Traceback" not in completed.stderr
    assert "IncompleteRead" not in completed.stderr
    assert "relay-token" not in completed.stderr
    assert "relay-token" not in completed.stdout
    assert not (tmp_path / "never-opened.sqlite3").exists()


def test_packaged_replay_cli_accepts_valid_reproducible_payload(tmp_path: Path) -> None:
    body = json.dumps(
        {
            "reproducible": True,
            "event_count": 3,
            "module_qualified_id": "bed_exit.v1",
        }
    ).encode()
    server = _serve(body)
    try:
        host, port = server.server_address
        completed = _run_cli(tmp_path, f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload == {
        "event_count": 3,
        "module_qualified_id": "bed_exit.v1",
        "reproducible": True,
    }
    assert not (tmp_path / "never-opened.sqlite3").exists()


def test_packaged_replay_cli_refuses_valid_non_reproducible_payload(tmp_path: Path) -> None:
    body = json.dumps(
        {
            "reproducible": False,
            "event_count": 0,
            "non_reproducible_reason": "truncated-or-incomplete-initial-state",
        }
    ).encode()
    server = _serve(body)
    try:
        host, port = server.server_address
        completed = _run_cli(tmp_path, f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
    payload = _assert_typed_refusal(completed)
    assert "incomplete" in payload["detail"]

"""Digest-auth 를 대신 붙여주는 카메라 웹 UI 리버스 프록시.

Chrome 은 URL 내장 자격증명을 막고, 자체서명 인증서 경고와 Digest 팝업이
자동화를 방해한다. 이 프록시는 평문 HTTP 로 받아서 카메라의 HTTPS + Digest
로 중계한다. 카메라 한 대당 로컬 포트 하나.

성능이 곧 성패다. setup.html 은 동기 XHR 을 수백 번 던지는데, 경로가
Tailscale -> SSH 터널 -> 카메라라 왕복이 비싸다. 매 요청마다 TLS 핸드셰이크와
401 재시도를 새로 하면 페이지가 몇 분씩 메인 스레드를 물고 늘어져 브라우저
자동화가 통째로 죽는다. 그래서:

  1. requests.Session 으로 TLS 커넥션을 재사용한다.
  2. 정적 자산(js/css/이미지/폰트)은 메모리에 캐시해 두 번째부터는 즉답한다.
  3. 카메라는 SessionID 를 응답 헤더로 돌려주고 웹앱이 그걸 다시 요청 헤더로
     보내므로, 헤더를 양방향으로 그대로 통과시킨다.
"""

import os
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings()

# 카메라 펌웨어의 setup.html 은 sha256 을 crypto-js.googlecode.com 에서 받는다.
# 구글코드는 2016 년에 폐쇄돼 그 요청이 영영 끝나지 않는다. 같은 버전(3.1.2)을
# 로컬에서 대신 먹여준다.
_SHA256_REMOTE = "https://crypto-js.googlecode.com/svn/tags/3.1.2/build/rollups/sha256.js"
_SHA256_LOCAL = "/__local/sha256.js"
_SHA256_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha256.js")

# 자격증명과 카메라 IP 는 이 저장소에 두지 않는다. 둘 다 운영 비밀이고
# 저장소는 공유 대상이다. 환경변수로 주입한다.
#
#   CAM_USER      기본 admin
#   CAM_PASSWORD  필수
#   CAM_CHANNELS  "채널:IP" 를 쉼표로. 예 "1:10.0.0.11,2:10.0.0.12"
#
# 채널 N 은 로컬 프록시 포트 194NN, SSH 터널 포트 184NN 로 매핑된다
# (edge.sh 가 여는 포워드와 같은 규칙).
USER = os.environ.get("CAM_USER", "admin")
PASSWORD = os.environ.get("CAM_PASSWORD")
if not PASSWORD:
    sys.exit("CAM_PASSWORD 가 필요하다. 자격증명 노트에서 읽어 환경변수로 넘겨라.")


def _parse_channels(spec):
    """"1:10.0.0.11,2:10.0.0.12" -> [(19401, 18401, "ch1 10.0.0.11"), ...]"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        ch, _, ip = item.partition(":")
        n = int(ch)
        out.append((19400 + n, 18400 + n, f"ch{n} {ip}"))
    return out


# (로컬 포트, 카메라로 가는 SSH 터널 포트, 라벨)
CAMERAS = _parse_channels(os.environ.get("CAM_CHANNELS", ""))
if not CAMERAS:
    sys.exit("CAM_CHANNELS 가 비어 있다. 예: CAM_CHANNELS='1:10.0.0.11,2:10.0.0.12'")

_STATIC_EXT = (
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif",
    ".ico", ".woff", ".woff2", ".ttf", ".svg",
)

_SKIP = {"host", "connection", "content-length", "accept-encoding", "authorization"}


class _LegacyTLSAdapter(HTTPAdapter):
    """카메라는 자체서명 인증서에 구형 TLS/암호군을 쓴다. 기본 컨텍스트는
    TLS1.2 미만과 SECLEVEL 2 미만을 거부해 핸드셰이크가 깨진다. 경로가 이미
    SSH 터널 안이므로 검증을 끄고 하한을 내린다."""

    def init_poolmanager(self, *a, **kw):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.TLSv1
        ctx.set_ciphers("DEFAULT@SECLEVEL=0")
        kw["ssl_context"] = ctx
        return super().init_poolmanager(*a, **kw)


def make_handler(target_port, label):
    base = f"https://127.0.0.1:{target_port}"
    session = requests.Session()
    session.verify = False
    session.auth = HTTPDigestAuth(USER, PASSWORD)
    session.mount("https://", _LegacyTLSAdapter(pool_connections=8, pool_maxsize=16))
    cache = {}
    cache_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 소음을 줄인다
            pass

        def _send(self, status, ctype, payload, extra=()):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            for k, v in extra:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _relay(self, body=None):
            path = self.path
            bare = path.split("?")[0]

            if bare == _SHA256_LOCAL:
                with open(_SHA256_PATH, "rb") as fh:
                    self._send(200, "application/javascript", fh.read())
                return

            cacheable = self.command == "GET" and bare.lower().endswith(_STATIC_EXT)
            if cacheable:
                with cache_lock:
                    hit = cache.get(path)
                if hit is not None:
                    self._send(hit[0], hit[1], hit[2])
                    return

            headers = {k: v for k, v in self.headers.items() if k.lower() not in _SKIP}
            t0 = time.monotonic()
            try:
                r = session.request(
                    self.command, base + path, data=body, headers=headers,
                    timeout=(10, 60), allow_redirects=False,
                )
            except requests.RequestException as e:
                # 터널이 잠깐 끊기거나 카메라가 늦게 답하는 일은 흔하다. 브라우저에
                # 502 를 돌려주고 이 요청만 버려야 나머지 채널이 계속 산다.
                self._send(502, "text/plain", f"proxy error ({label}): {e}".encode())
                return

            payload = r.content
            # 죽은 googlecode 스크립트 참조를 로컬 사본으로 갈아끼운다.
            if _SHA256_REMOTE.encode() in payload:
                payload = payload.replace(_SHA256_REMOTE.encode(), _SHA256_LOCAL.encode())

            ctype = r.headers.get("Content-Type", "application/octet-stream")
            # SessionID / Location 처럼 웹앱이 읽어야 하는 헤더는 살려서 넘긴다.
            extra = [(k, v) for k, v in r.headers.items()
                     if k.lower() in ("sessionid", "location", "set-cookie")]

            if cacheable and r.status_code == 200:
                with cache_lock:
                    cache[path] = (r.status_code, ctype, payload)

            self._send(r.status_code, ctype, payload, extra)
            dt = time.monotonic() - t0
            if dt > 1.0 or r.status_code >= 400:
                print(
                    f"[{label}] {r.status_code} {len(payload)}B {dt:.1f}s {path[:80]}",
                    flush=True,
                )

        def do_GET(self):
            self._relay()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self._relay(self.rfile.read(n) if n else b"")

        do_PUT = do_POST

    return Handler


def main():
    for local_port, target_port, label in CAMERAS:
        srv = ThreadingHTTPServer(("127.0.0.1", local_port), make_handler(target_port, label))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"http://localhost:{local_port}  ->  {label}", flush=True)
    print("ready", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    sys.exit(main())

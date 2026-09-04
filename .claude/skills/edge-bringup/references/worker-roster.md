# 워커 로스터에 카메라 물리기

RTSP 가 살아난 뒤의 마지막 단계다. 카메라 스트림이 다 정상이어도 ML 워커가
그 카메라를 모르면 아무 일도 일어나지 않는다. 이 문서는 카메라를 로스터에
등록하는 API, 그 인증, 그리고 등록만 하고 끝냈다가 놓치는 함정들을 다룬다.

## 목차

1. [왜 이 단계가 따로 필요한가](#1-왜-이-단계가-따로-필요한가)
2. [인증은 대시보드 세션 쿠키다](#2-인증은-대시보드-세션-쿠키다-릴레이-토큰이-아니다)
3. [카메라 등록 API](#3-카메라-등록-api)
4. [등록 후 워커를 재시작해라](#4-등록-후-워커를-재시작해라)
5. [API 로는 못 하는 것](#5-api-로는-못-하는-것)
6. [검증 — 세 단계로, 화면 밖에서](#6-검증--세-단계로-화면-밖에서)
7. [등록 스크립트](#7-등록-스크립트)

## 1. 왜 이 단계가 따로 필요한가

`GET /api/v1/cameras` 가 `registry_version=0`, `cameras: []` 를 돌려주는 상태는
정상 부팅과 구분이 안 된다. compose 가 뜨고, `ml-api` 헬스체크가 통과하고,
`ml-worker` 도 죽지 않고 떠 있다 — 그런데 카메라가 로스터에 하나도 없으면
워커는 그냥 아무것도 처리할 게 없는 채로 대기한다. 에러가 안 나기 때문에
"브링업이 끝났다"는 착각이 여기서 생긴다. RTSP 가 다 살아 있어도 이 단계를
건너뛰면 결과는 똑같이 "아무 일도 안 일어남"이다.

## 2. 인증은 대시보드 세션 쿠키다 (릴레이 토큰이 아니다)

카메라 CRUD (`POST/GET/PATCH/DELETE /api/v1/cameras`) 는 **대시보드 세션**으로
인증한다. `X-Edge-Relay-Token` 은 여기서 안 통한다 — 이걸 혼동해서 401 을
보는 일이 흔하다.

```bash
curl -s -i -c cookies.txt -X POST http://<edge>:8000/api/v1/auth/session \
  -H 'Content-Type: application/json' \
  -d '{"username":"<username>","password":"<password>"}'
```

성공은 **204** 다 (`backend/app/features/auth/router.py:47`). 응답에 바디는
없고, `Set-Cookie` 로 HttpOnly 쿠키 `ml_dashboard_session` 이 온다. 유효기간은
12 시간(`DASHBOARD_SESSION_TTL_SECONDS`, `backend/app/shared/dashboard_auth.py:25`).
이후 카메라 API 호출은 전부 이 쿠키를 실어 보내면 된다.

`X-Edge-Relay-Token` 이 실제로 쓰이는 곳은 딱 하나, 워커 자신이 주기 폴링으로
부르는 `GET /api/v1/cameras/worker-config` 뿐이다
(`backend/app/features/cameras/router.py:443`, `_authorize_worker`). 사람이
카메라를 등록하는 경로와는 아예 다른 인증 체계라고 생각하는 게 맞다. 코드를
더 보면 카메라 CRUD 쪽에도 릴레이 토큰을 "레거시" 경로로 받아주는 분기가 남아
있긴 하다(`authorize_dashboard` 의 `legacy_token` 인자, `dashboard_auth.py:241`)
— 그런데 `dashboard_sessions()` 가 항상 세션 스토어를 돌려주기 때문에(파일에
"unreachable in practice today" 라고 스스로 적어뒀다, `dashboard_auth.py:254`)
이 분기는 오늘 이 배포에서는 절대 안 탄다. 즉 릴레이 토큰으로 카메라를 등록해
보려는 시도는 코드상 원천적으로 막혀 있다.

### 자격증명 우선순위

높은 쪽이 이긴다 (`dashboard_auth.py:154`, `_resolve_credentials`):

1. **대시보드에서 바꿔 저장한 값** — `catalog.sqlite3` 의 단일 행 `credentials`
   테이블. scrypt 로 해시돼 저장된다(`dashboard_credentials.py`). 한 번
   저장되면 이후 부팅에서 아래 두 단계보다 항상 이긴다.
2. **`API_DASHBOARD_USERNAME` / `API_DASHBOARD_PASSWORD` env 쌍** — **둘 다**
   설정돼야 인정된다. 한쪽만 있으면 503
   `dashboard credentials are incompletely configured` 가 뜬다
   (`dashboard_auth.py:168-173`).
3. **내장 기본값** `admin` / `admin`.

### 함정: env 쌍이 있으면 내장 기본값은 절대 안 먹는다

`compose.edge.yaml` 은 `API_DASHBOARD_USERNAME`/`API_DASHBOARD_PASSWORD` 를
`${...:?...}` 로 필수 걸어 둔다(`compose.edge.yaml:31-32`) — **빈 값으로는
compose 렌더 자체가 깨진다.** 그래서 "내장 기본값으로 되돌리고 싶다"고
env 값을 비우는 선택지가 없다. 되돌리려면 env 에 `admin`/`admin` 을 그대로
명시해야 한다.

그리고 env 쌍이 한 번이라도 채워져 있으면 내장 `admin`/`admin` 은 먹지 않는다
(우선순위 2번이 3번을 가린다). `admin`/`admin` 으로 401 이 나면 카메라 계정을
의심하기 전에 `.env.edge.*` 부터 봐라. 영속 자격증명이 저장돼 있는지는
`credentials` 테이블을 직접 조회해 확인할 수 있다:

```bash
sqlite3 <ml-api 상태 볼륨>/catalog.sqlite3 \
  "SELECT username, algorithm, updated_at FROM credentials WHERE id = 1;"
```

한 행이라도 나오면 1번 규칙이 적용 중이라는 뜻이다 — env 값을 아무리 고쳐도
로그인 값은 안 바뀐다.

## 3. 카메라 등록 API

```
POST /api/v1/cameras
```

`CreateCameraRequest` 스키마이고 `extra="forbid"` 다
(`backend/app/features/cameras/router.py:123-137`). 모르는 필드를 하나라도
보내면 즉시 422 다.

| 필드 | 필수 | 비고 |
|---|---|---|
| `label` | 필수 | 빈 문자열 불가 |
| `rtsp_url` | 필수 | 빈 문자열 불가. 3절 하단 참고 |
| `space_id` | 선택 | 백엔드 룸 매핑용 |
| `decode_backend` | 선택 | `auto`\|`nvdec`\|`opencv`\|`cpu` 중 하나 (대소문자 무관). 그 외 값은 400 |
| `fps` | 선택 | 0 보다 큰 수. 아니면 400 |
| `floor` | 선택 | 고정 카탈로그 정수(B1=-1 … 10층=10). 범위 밖이면 400 |
| `force_register` | 있지만 죽은 필드 | 아래 참고 |

`force_register` 는 예전에 probe 실패 시 등록을 막던 게이트를 우회하는
탈출구였다. 지금은 등록이 항상 저장하므로(probe 는 상태 표시에만 쓰인다)
값과 무관하게 결과가 같다. `extra="forbid"` 때문에 필드를 지우면 옛 클라이언트
호출이 422 로 깨져서 그대로 남아 있을 뿐이다(`router.py:132-137`의 주석
참고).

등록은 성공하면 201 과 함께 카메라 레코드를 돌려준다. probe 가 실패해도
거절하지 않는다 — offline/`never_connected` 상태로 그대로 목록에 들어가고,
실제 연결 여부는 워커의 첫 heartbeat 가 확정한다(같은 파일 251-261 줄의
주석에 그 이유가 적혀 있다: probe 를 등록 게이트로 쓰면 워커가 카메라 1대
이상일 때만 부팅하는 구조와 맞물려 최초 등록 자체가 불가능해진다).

### rtsp_url 은 서브스트림을 직접 박아야 한다

ML 입력은 서브스트림을 쓴다. IDIS 계열은 `trackID=2` 가 서브스트림,
`trackID=1` 이 메인이다. **API 스키마 어디에도 서브스트림을 고르는 필드가
없다** — `rtsp_url` 문자열 자체에 원하는 트랙을 직접 넣어야 한다.

```
rtsp://<사용자>:<비밀번호>@<카메라 IP>:554/trackID=2   # 서브스트림 (이걸 써라)
rtsp://<사용자>:<비밀번호>@<카메라 IP>:554/trackID=1   # 메인 스트림 (CPU 디코드에는 넣지 마라)
```

이게 왜 중요한지: 실증 엣지 노드는 GPU 가 없어 `ML_WORKER_PROFILE=cpu` 로
돈다. 메인 스트림(고해상도 H265)을 CPU 로 디코드하면 카메라 대수가 늘수록
디코드 실패가 누적된다 — 실제로 13 대를 메인 스트림으로 물렸을 때 최근 로그
8000 줄 중 6700 줄이 HEVC 디코드 실패였다. 폴백 모델은 30 프레임 윈도우가
필요해서, 여기서 드롭된 프레임이 곧 낙상 감지 누락으로 이어진다(issue #154).

대시보드 등록/수정 화면(`front/src/features/settings/CameraRegisterModal.tsx`,
`CameraEditModal.tsx`)은 `rtsp_url` 에 `trackID` 가 없거나 `trackID=1` 이면
서브스트림을 쓰라는 경고 문구를 보여준다(`rtspSubstreamGuidance.ts`). **이건
안내일 뿐 등록을 막지 않는다** — 벤더별로 서브스트림 경로 형식이 달라
`rtsp_url` 문자열만으로는 IDIS 가 아닌 카메라를 안전하게 구분할 수 없고,
하드 거부는 이미 등록된(메인 스트림을 넣은) 기존 카메라 행까지 깨뜨린다.
기존 행을 서브스트림으로 옮기는 것은 이 스킬의 범위가 아닌 별도 운영
판단이다. curl/스크립트로 직접 `POST /api/v1/cameras` 를 호출하는 경로(7절)
는 이 경고를 거치지 않으므로, 그쪽에서 등록할 때는 `trackID=2` 를 직접
챙겨야 한다.

**등록 후 실제로 서브스트림에 붙었는지 확인해라.** 두 가지 방법이 있다.

1. **엣지에서 `ffprobe`** (가장 확실하다, `references/troubleshooting.md`
   A절 참고):

   ```bash
   ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
     -show_entries stream=codec_name,width,height,r_frame_rate \
     -of csv=p=0 "rtsp://<사용자>:<비밀번호>@<카메라 IP>:554/trackID=2"
   ```

   IDIS 서브스트림 정상 출력은 `hevc,640,360,30/1` 이다. 해상도가 이보다 훨씬
   크게 나오면 실제로는 메인 스트림에 붙은 것이다 — URL을 다시 확인해라.

2. **등록/연결 확인 API 응답의 `width`/`height`**: `POST /api/v1/cameras` 와
   `POST /api/v1/cameras/{id}/test` 는 probe 가 성공하면 해상도를 함께
   돌려준다(`TestCameraResponse.width/height`, `router.py:167-179`). 카메라
   목록 화면에서 "연결 확인"을 눌러도 같은 값을 볼 수 있다. `640x360` 근처가
   아니면 서브스트림이 아니다.

## 4. 등록 후 워커를 재시작해라

가장 잘 놓치는 지점이다. 워커의 재시작 판정은 로컬 레지스트리가 아니라
백엔드가 내려주는 두 숫자, `restart_epoch` 와 `config_version` 만 본다
(`worker/runtime/config/restart.py:14-23`, `RestartDirective`). 워커는 60초
간격으로(`make_restart_check` 의 `poll_interval_sec` 기본값,
`restart.py:49`) `GET /api/v1/cameras/worker-config` 를 다시 불러 이 두
숫자가 올라갔는지만 비교한다(`restart.py:36-41`, `RestartDirectiveTracker
.observe`). 올라가 있으면 워커가 스스로 멈추고
(`Flow media plane/lifecycle.py:299-306`, `_watch_restart`), compose 의
`restart: unless-stopped` 가 다시 띄우면서 새 로스터로 재부팅한다.

문제는 카메라 등록/수정/삭제가 로컬 레지스트리의 `registry_version` 만
올린다는 것이다. `config_version` 은 백엔드 config-pull 이 성공했을 때만
갱신되고(`backend/app/lifespan.py:396`), `restart_epoch` 는 그 값을 바꾸는
API 호출이 따로 있어야만 오른다(아래 참고). **로컬 등록만으로는 이 둘 중
어느 것도 안 움직인다.** 그래서 카메라를 등록해 놓고 60초, 아니 몇 분을
기다려도 워커는 옛 로스터 그대로다.

가장 확실한 방법은 그냥 컨테이너를 재시작하는 것이다:

```bash
scripts/edge.sh run "cd '<repo>' && \
  docker compose --env-file .env.edge.prod \
    -f compose.edge.yaml -f compose.edge.cpu.yaml \
    restart ml-worker"
```

(GPU 노드라면 `compose.edge.cpu.yaml` 오버레이는 빼라 — 4단계 참고.)

한 가지 더: `POST /api/v1/relay/restart` 라는 엔드포인트가 이미 존재한다
(`backend/app/features/relay/router.py:244-251`). `X-Edge-Relay-Token` 으로
인증하고, 호출할 때마다 `restart_epoch` 를 1 증가시킨다. 즉 SSH 로 들어가지
않고도 이 엔드포인트만 호출하면 다음 60초 폴링에서 워커가 스스로 재시작을
건다:

```bash
curl -s -X POST http://<edge>:8000/api/v1/relay/restart \
  -H "X-Edge-Relay-Token: <자격증명 노트에서>"
```

다만 이건 프론트엔드나 다른 백엔드 코드 어디에서도 호출하는 곳이 없는,
API로만 존재하는 경로다 — 즉시 반영되는 `docker compose restart` 와 달리
최대 60초의 폴링 지연이 붙는다. 급하면 `docker compose restart` 를 써라.

## 5. API 로는 못 하는 것

### 카메라별 `frame_stride`

`CreateCameraRequest` 에 `frame_stride` 필드가 없다. 있는 건 시설 전체에
적용되는 `ML_DEFAULT_FRAME_STRIDE` 뿐이다(`router.py:954-971`,
`_default_frame_stride`). 카메라마다 다른 값을 주는 API 경로는 없다.

### 카메라별 세밀 제어가 필요하면 `EDGE_CAMERA_CONFIG`

`worker/ml-worker.local.yaml` 같은 YAML 로 워커를 직접 부팅시키는 경로가
있다(`EDGE_CAMERA_CONFIG` 환경변수, `compose.edge.yaml:97-103`). 여기서는
카메라별 `frame_stride` 와 `streams.sub`/`streams.main` 을 개별 지정할 수
있다(`worker/ml-worker.example.yaml:58` 부근 참고). 다만
`.env.edge.prod.example` 은 이 값을 빈 채로 배포한다
(`.env.edge.prod.example:39`, `EDGE_CAMERA_CONFIG=`) — 즉 기본 배포는
**API pull 경로**로 돈다. 이 절차로 등록한 카메라는 YAML 이 아니라
`ml-api` 의 로컬 레지스트리를 거쳐 워커에 전달된다. 현장에서 헷갈리지
않으려면 그 edge 노드가 실제로 어느 경로로 부팅했는지(`EDGE_CAMERA_CONFIG`
가 비어 있는지) 먼저 확인해라 — 두 경로가 섞이면 등록해도 반영이 안 되는
것처럼 보인다.

## 6. 검증 — 세 단계로, 화면 밖에서

등록했다고 끝이 아니다. 아래 순서대로 확인해야 "보인다"고 말할 수 있다.

### 6.1 레지스트리

```bash
curl -s -b cookies.txt http://<edge>:8000/api/v1/cameras | python3 -m json.tool
```

`registry_version` 이 등록한 수만큼 올라가 있어야 한다(생성마다 정확히
+1 씩, `backend/app/features/cameras/store.py` 의 `create`/`update`/`delete`
가 각각 증가시킨다). 각 카메라의 `status` 가 `online` 인지도 본다.

### 6.2 런타임

```bash
curl -s -b cookies.txt http://<edge>:8000/api/v1/status | python3 -m json.tool
```

`runtime.worker.alive == true`(`worker/runtime/telemetry/wire.py:70` 의
`alive` 필드), 그리고 `runtime.cameras` 의 개수가 등록한 카메라 수와
같은지 본다. 여기가 비어 있으면 relay 쪽이 막힌 것이다 —
`references/troubleshooting.md` 의 relay 403 항목을 봐라.

### 6.3 실제 영상

```bash
curl -s -b cookies.txt -o snap.jpg -D - \
  http://<edge>:8000/api/v1/streams/<camera_id>/snapshot
```

카메라마다 `image/jpeg` 로 수십 KB 가 돌아오는지 확인해라. 이 라우트도
대시보드 세션으로 인증한다(2절과 같은 체계, `streams_router.py:132`).

주의: 워커가 스트림에 막 붙는 중이면 503 `worker stream unavailable`
(`streams_router.py:358-364`)이 잠깐 나올 수 있다. **한 번의 503 으로
실패 판정하지 마라** — 실제로 마지막 한 대가 첫 시도에 503, 몇 초 뒤 재시도
에서 정상이었다. 간격을 두고 두세 번 다시 찔러서 구분해라.

## 7. 등록 스크립트

`urllib` + `http.cookiejar` 로 세션을 유지하며 순차 등록하는 예시다.
비밀번호는 환경변수로 받는다 — 스크립트에 실제 값을 적지 마라.

```python
#!/usr/bin/env python3
"""카메라 목록을 순차 등록한다.

사용법:
  API_EDGE_BASE_URL=http://<edge>:8000 \
  API_DASHBOARD_USERNAME=<username> API_DASHBOARD_PASSWORD=<password> \
  CAM_USER=<카메라 계정> CAM_PASSWORD=<카메라 비밀번호> \
  python3 register_cameras.py

대시보드 자격증명과 카메라 자격증명은 서로 다른 것이다. 대시보드 쪽은 이 API 에
로그인하기 위한 것이고, RTSP URL 에 들어가는 것은 카메라 쪽이다. 하나로 뭉뚱그리면
등록은 201 로 성공했는데 워커가 붙을 때 401 이 나는, 원인 찾기 고약한 상태가 된다.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ["API_EDGE_BASE_URL"].rstrip("/")
USERNAME = os.environ["API_DASHBOARD_USERNAME"]
PASSWORD = os.environ["API_DASHBOARD_PASSWORD"]
CAM_USER = os.environ["CAM_USER"]
CAM_PASSWORD = os.environ["CAM_PASSWORD"]

# (라벨, 카메라 IP) 더미 예시. 실제 목록으로 교체해라.
CAMERAS = [
    ("101호", "10.0.0.11"),
    ("102호", "10.0.0.12"),
]

cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))


def _request(method: str, path: str, body: dict | None = None) -> dict | None:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with opener.open(req, timeout=10) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def login() -> None:
    _request("POST", "/api/v1/auth/session", {"username": USERNAME, "password": PASSWORD})


def register(label: str, camera_ip: str) -> None:
    # trackID=2 가 IDIS 서브스트림이다. 메인(trackID=1)을 넣지 마라.
    # 비밀번호는 퍼센트 인코딩하지 않는다 — IDIS 는 userinfo 를 디코딩하지 않아
    # `%21` 같은 값을 그대로 비밀번호로 읽고 401 을 낸다.
    rtsp_url = f"rtsp://{CAM_USER}:{CAM_PASSWORD}@{camera_ip}:554/trackID=2"
    try:
        result = _request("POST", "/api/v1/cameras", {"label": label, "rtsp_url": rtsp_url})
    except urllib.error.HTTPError as exc:
        print(f"[{label}] 등록 실패: {exc.code} {exc.read().decode('utf-8', 'replace')}")
        return
    print(f"[{label}] 등록됨: id={result.get('id')} status={result.get('status')}")


def main() -> int:
    login()
    for label, camera_ip in CAMERAS:
        register(label, camera_ip)
    print("등록 끝. ml-worker 재시작을 잊지 마라 (4절 참고).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`rtsp_url` 조립부는 실제 URL 인코딩/포트/경로 규칙에 맞춰 고쳐 써야 한다 —
카메라 벤더·펌웨어별로 경로 형식이 다르다. 이 스크립트는 뼈대일 뿐이다.

## 참고 문서

- `SKILL.md` — RTSP 를 살리는 절차 전체.
- `references/camera-webguard.md` — IDIS WebGuard 조작.
- `references/troubleshooting.md` — 이 환경에서 실제로 겪은 실패와 원인.

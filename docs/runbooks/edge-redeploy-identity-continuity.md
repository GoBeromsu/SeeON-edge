# 엣지 재배포 시 카메라 신원 연속성

이미지를 교체하거나 스택을 다시 세울 때 **카메라 신원(Hub 매핑)을 잃지 않기 위한 절차**다.
GPU 프로파일 전환, Dockerfile 변경, 호스트 이전 모두 이 절차를 따른다.

`worker-migration-rollback.md` 는 스키마와 DB 마이그레이션을 다룬다. 이 문서는 그
위에 있는 문제 — **DB 는 멀쩡한데 Hub 가 카메라를 못 알아보는 상태** — 를 다룬다.

## 왜 필요한가

카메라의 Hub 정식 식별자는 `backend_camera_id` 에 들어 있고, 이 값은 **topology sync
만 채운다**. `POST/PATCH /cameras` 는 `extra="forbid"` 로 이 필드를 거부한다.

그리고 매핑이 없으면 워커에게 내려가는 설정이 로컬 UUID 로 폴백한다.

```python
# backend/app/features/cameras/router.py
canonical_id = str(record.get("backend_camera_id") or record.get("id", ""))
```

Hub 는 자기가 발급하지 않은 ID 를 `FACILITY_BINDING_MISMATCH` 로 거부하고, 엣지에는
relay 502 로만 보인다. 즉 **매핑을 잃으면 알림이 Hub 에 도달하지 않는데, 증상은
인증 오류처럼 보인다.**

현재 클린 상태에서 sync 로 매핑을 새로 만드는 경로는 막혀 있다(#308). 그러므로
재배포 시에는 **기존 매핑을 보존하는 것이 유일한 안전한 방법**이다.

## 하지 말 것

- **카메라를 지우고 다시 등록하지 않는다.** ID 가 새로 발급되고 Hub 매핑이 끊긴다.
  복구 경로가 없다.
- **`edge-state` 볼륨을 새로 만들지 않는다.** compose 프로젝트 이름(= 작업 디렉터리
  이름)이 바뀌면 볼륨도 새로 생기고, 결과는 카메라 전멸과 같다.
- **`docker run` 으로 손수 조립하지 않는다.** compose 가 서비스 별칭(`ml-api`,
  `ml-worker`)을 만들어 준다. 별칭이 없으면 `worker_stream_origin`
  (`http://ml-worker:8090`) 이 해석되지 않아 영상이 오프라인으로 표시된다.

## 절차

### 1. 재배포 전 상태 보존

```bash
docker run --rm -v <project>_edge-state:/src:ro -v "$PWD":/dst alpine:3.22 \
  tar czf /dst/edge-state.tar.gz -C /src .
```

`edge.sqlite3` 에 카메라 레지스트리(`camera_registry.cameras_json`, 매핑 포함)와
enrollment 가 들어 있다.

### 2. 같은 볼륨을 유지한 채 이미지만 교체

compose 프로젝트 이름을 바꾸지 않는다. 디렉터리를 옮겨야 한다면 `-p <기존
프로젝트명>` 을 명시한다.

```bash
docker compose -f compose.edge.yaml -f compose.edge.<profile>.yaml up -d --pull never
```

### 3. 순서를 지킨다

`edge-filesystem-inventory` → `edge-db-migrator` → `ml-api`(healthy) →
`ml-worker`. compose 의존성이 이를 강제한다. inventory 는 schema 17 이전에만
worker delivery queue 와 clip staging 을 비운 상태인지 확인하며, schema 17 이후에는
대기 envelope 를 backend 가 drain 할 수 있도록 통과한다.

워커가 떠 있는 상태로 migrator 를 돌리면 다음으로 실패한다.

```
EDGE_DB_IMPORT_FAILED: edge deployment lock is held by a running runtime
```

이는 정상 동작이다. 워커를 먼저 정지한 뒤 다시 올린다.

### 4. 워커 설정 캐시가 낡았을 때만 다시 가져온다

레지스트리를 복원했는데 워커가 예전 카메라 ID 를 계속 쓰면 last-known-good 이
남아 있는 것이다. 워커는 데이터베이스를 갖지 않는다. `ml-worker` 를 재시작해서
백엔드의 `worker-config` 를 다시 가져오게 한다. 정상 pull 이 성공하면 검증된
bounded cache가 새 payload로 교체된다.

```bash
docker compose ... restart ml-worker
```

재시작 뒤에는 아래 검증 절차의 heartbeat와 `worker-config` camera ID를 확인한다.
캐시 파일을 직접 수정하거나 SQLite를 열어 복구하지 않는다.

## 검증

순서대로 확인한다. **토큰부터 의심하지 않는다.**

```bash
# 1) 릴레이가 붙는가
docker compose ... logs --since 3m ml-api | grep -oE '"POST /api/v1/relay/heartbeat HTTP/1.1" [0-9]+' | sort | uniq -c
#    202 = 정상.  403/502 면 아래로.

# 2) 워커에게 내려간 camera_id 모양
docker compose ... exec -T ml-worker python -c "
import json,os,urllib.request
r=urllib.request.Request('http://ml-api:8000/api/v1/cameras/worker-config',
    headers={'X-Edge-Relay-Token':os.environ['RELAY_TOKEN']})
import sys
with urllib.request.urlopen(r,timeout=20) as x: d=json.loads(x.read().decode())
print([c['camera_id'] for c in d['cameras']][:3])"
```

`camera_id` 가 **로컬 UUID 형식이면 매핑이 깨진 것**이다. Hub 정식 식별자는 cuid
(`cmsnvr…`) 또는 손으로 심은 값(`cam_sp_205` 형식)이다.

`worker-config` 는 릴레이 자격증명을 쓴다. 대시보드 세션으로 호출하면 401 이며 이는
인증 설계상 정상이다.

## 환경변수 계약

배포 경계를 넘는 값만 넘긴다. 나머지는 baked topology 이거나 대시보드 소유다.

| 서비스 | 키 |
| --- | --- |
| `ml-api` | `API_BACKEND_BASE_URL`, `API_BACKEND_INGEST_TIMEOUT_SEC`, `API_DASHBOARD_USERNAME`, `API_DASHBOARD_PASSWORD`, `ML_RTSP_ALLOW_PRIVATE_DESTINATIONS`, `ML_RTSP_ALLOW_LOCAL_DESTINATIONS` |
| `ml-worker` | `ML_WORKER_PROFILE`, `ML_RTSP_ALLOW_PRIVATE_DESTINATIONS`, `ML_RTSP_ALLOW_LOCAL_DESTINATIONS` |

정확한 목록은 `edge-env-inventory.json` 을 기준으로 한다.

구 배포본에서 env 를 복사하면 **은퇴한 키 때문에 워커가 부팅을 거부**한다
(`worker configuration refused: retired edge environment key(s)`). 대표적으로
`RELAY_URL`, `ML_WORKER_DEV_MJPEG*`, `CLIP_STORE_DIR`, `ML_WORKER_FALL_MODEL_*`.

`API_BACKEND_BASE_URL` 은 **origin 까지만** 넣는다. 제품이 `/api/v1/...` 를 붙인다.

## 알려진 제약

- 클린 상태에서 sync 로 카메라를 새로 연결하는 경로는 동작하지 않는다 — **#308**.
  해결 전까지 재배포는 기존 매핑 보존에 의존한다.
- 같은 시설에 두 번째 엣지를 enroll 하면 Hub 가 소유권 이전을 수행할 수 있어
  운영 중인 엣지가 밀려날 수 있다. 클린 설치 실험은 라이브 시설에서 하지 않는다.
  `tests_support/local_backend_fixture.py` 로 오프라인 재현한다.

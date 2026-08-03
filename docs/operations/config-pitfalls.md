# 운영 런북: 설정 함정(Config Pitfalls)

이 문서는 backend/worker 설정값 중 "어느 프로세스의 env에 설정해야 하는지",
"YAML과 env 중 무엇이 우선하는지"가 직관과 다르게 동작해서 실제로는 조용히
무효화되는 사례를 모은다. 코드 주석에는 남아 있지만 운영자가 배포/트러블슈팅
중에 바로 찾아볼 수 있는 곳이 없었다 (#130).

각 항목은 오늘(2026-08-03) 시점 HEAD 기준으로 실제 코드를 읽고 검증한
내용이며, 1번 항목은 QA 중 무효화가 실증됐다.

## 요약 표

| 설정값 | 설정 위치(프로세스) | 우선순위 | 함정 |
| --- | --- | --- | --- |
| `ML_DEFAULT_CAMERA_FPS` / `ML_DEFAULT_FRAME_STRIDE` | **backend** 프로세스 env | env만 존재 (worker에는 이 이름의 env가 없음) | worker 프로세스 env에 설정하면 조용히 무효 (QA에서 실증) |
| `dev_mjpeg.enabled` (YAML) | worker YAML 또는 `ML_WORKER_DEV_MJPEG*` env | YAML 명시 시 무조건 승리, YAML 침묵 시 env가 결정 | 과거(#113) pull-config 병합에서 이 우선순위가 깨져 YAML 값이 매 pull마다 리셋된 적 있음 (수정됨) |
| `clip.enabled` / `models.fall.type` (YAML) | worker YAML 또는 env | 동일하게 YAML 명시 시 승리, 침묵 시 env — 단 **부팅 시 1회만** 계산 | 재-pull/재시작 없이는 값이 바뀌지 않음 |
| 상태 저장 위치 | 고정: `~/.local/state/<runtime>/` | env override 없음 (의도적) | 운영자가 다른 경로를 기대하면 함정 — 변경은 #38에서 추적 중 |
| relay token | worker 설정의 relay token 하나 | 동일 값이 MJPEG probe token으로도 사용됨 | 별도 토큰으로 오해하기 쉬움 |

---

## 1. `ML_DEFAULT_CAMERA_FPS` / `ML_DEFAULT_FRAME_STRIDE`는 backend 프로세스 env에 설정해야 한다

- `backend/app/features/cameras/router.py:884` — `_default_camera_fps()`가
  `ML_DEFAULT_CAMERA_FPS`를 읽는다.
- `backend/app/features/cameras/router.py:904` — `_default_frame_stride()`가
  `ML_DEFAULT_FRAME_STRIDE`를 읽는다.
- 두 값 모두 `backend/app/features/cameras/router.py:411-414`에서 relay
  pull-config 응답(`camera["fps"]`, `camera["frame_stride"]`)에 실려 worker로
  전달된다.

**함정**: 이 env var들은 이름만 보면 worker 설정처럼 보이지만, 실제로는
**backend 프로세스**의 환경변수를 읽는다. worker 컨테이너/프로세스의 env에
설정해도 backend가 읽지 않으므로 조용히 무효화된다. 이 무효화는 2026-08-03
QA에서 실제로 재현됐다. 두 값 모두 비어 있거나 파싱 실패 시 `None`을
반환하고, 그 경우 worker는 자신의 기본값(fps 5.0, stride 1)을 그대로 쓴다.

## 2. `dev_mjpeg`: YAML 명시값이 env보다 무조건 승리

- `worker/runtime/worker.py:605-632` (`_resolve_mjpeg_config`).
- YAML `dev_mjpeg.enabled: true`가 명시되어 있으면 그 값이 그대로 쓰인다
  (worker.py:619-622: `configured.enabled`가 참이면 `configured` 사용).
- YAML이 침묵(기본값 disabled)하면 `ML_WORKER_DEV_MJPEG*` env가 결정한다
  (worker.py:623: `dev_mjpeg_config(self._env)`).

**함정**: "YAML이 우선, 없으면 env"라는 방향은 직관적이지만, 과거(#113)에는
`BackendWorkerConfigPayload.to_worker_config`가 `dev_mjpeg`를 아예 전달하지
않아서 명시적으로 켠 YAML 값이 매 pull마다 pydantic 기본값(disabled)으로
조용히 리셋되던 버그가 있었다 (`worker/runtime/config/local_env.py:206-215`
주석 참고). 지금은 고쳐졌지만, 이 우선순위 규칙 자체를 기억해 두지 않으면
"YAML에 켜뒀는데 왜 꺼져 있지"라는 질문에 답하기 어렵다.

## 3. `clip.enabled` / `models.fall.type`도 동일한 "명시 시 승리" 패턴 — 단 부팅 시 1회만 고정

- `worker/runtime/config/local_env.py:187-233` (`resolve_local_overrides`).
- `models`: YAML에 `models.fall`이 명시돼 있으면 그 값이 쓰이고, 없으면
  env(`worker_models_config_from_environment`)가 결정한다 (local_env.py:218-222).
- `clip`: YAML `clip.enabled`가 참이면 그 값이 쓰이고, 아니면
  env(`clip_recording_config_from_environment`)가 결정한다 (local_env.py:223-227).
- 이 함수는 `WorkerRuntime` 부팅 경로에서 **한 번** 호출되어 결과가 그
  프로세스 수명 내내 고정된다.

**함정**: 2번 항목과 같은 "명시 시 승리, 침묵 시 env" 규칙이지만, 이 값들은
런타임 재-pull 시 재평가되지 않는다. 운영 중에 env를 바꾸거나 backend에서
`clip.enabled`를 토글해도 worker 프로세스를 재시작하기 전까지는 반영되지
않는다. 재-pull 시점의 재평가 여부는 별도 이슈(#127)에서 추적한다.

## 4. 상태 저장 위치는 홈 디렉토리 아래 고정 — env override 없음

- `worker/runtime/state_dir.py` — `resolve_state_dir()`이
  `~/.local/state/<runtime>/`를 반환하며, docstring이 "deliberately no
  environment override here"라고 명시한다.
- `backend/app/shared/state_dir.py` — backend 쪽도 동일한 규칙
  (`~/.local/state/ml-api/`), 동일하게 env override 없음.

**함정**: 이전 `/var/lib/ml-worker` + `ML_WORKER_STATE_DIR` env override
방식에서 의도적으로 바뀐 설계다 (root 소유 `/var/lib`가 macOS 개발 머신에는
없어서 GPU lease 획득 단계에서 바로 죽던 문제 때문). 운영자가 이전 방식을
기억해 env로 경로를 바꾸려 하면 아무 효과가 없다. 런타임에 선택 가능한 저장
위치는 issue #38(OPEN)에서 별도로 추적 중이며, 그 전까지는 이 경로가
고정값이다.

## 5. relay token이 MJPEG probe token을 겸한다

- `worker/runtime/worker.py:615-617` — `_resolve_mjpeg_config`의 docstring이
  "The relay token doubles as the probe token, matching edge, so the
  backend's probe origin authenticates against the same secret it already
  holds"라고 명시한다.
- `worker/runtime/worker.py:631` — 실제로
  `probe_token=self.config.relay.token.get_secret_value()`로 relay token을
  그대로 재사용한다.

**함정**: MJPEG 진단 포트를 위한 별도 토큰이 존재한다고 오해하기 쉽지만,
실제로는 worker 설정의 relay token 하나만 존재하며 이 값이 두 용도(backend
relay 인증, MJPEG probe 인증)에 모두 쓰인다. relay token을 로테이션하면
MJPEG probe 인증도 함께 바뀐다는 점을 배포 절차에서 고려해야 한다.

---

## 문서 갱신 절차

이 문서는 완결된 목록이 아니다. 새로운 설정 함정이 QA나 운영 중에 실증되면:

1. 위 요약 표에 행을 추가한다 (설정값 / 설정 위치 / 우선순위 / 함정).
2. 본문에 해당 항목의 상세 절(번호, file:line 근거, 함정 설명)을 추가한다.
3. 근거는 반드시 실제로 읽은 코드의 `file:line`을 인용한다 — 추정이나 기억에
   의존한 서술은 넣지 않는다.
4. 관련 PR의 체크리스트에 "새 설정 함정을 발견했다면 `docs/operations/config-pitfalls.md`를
   갱신했는지" 항목을 포함시킬 것을 권장한다.

## 관련 문서

- [`docs/runbooks/worker-migration-rollback.md`](../runbooks/worker-migration-rollback.md) —
  이미지 롤백 시 유지해야 하는 env/볼륨 규칙.
- [`docs/operations/clip-retention-policy.md`](clip-retention-policy.md) —
  clip 저장/보존 정책.

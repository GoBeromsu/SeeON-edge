# 클립 보관·감사 로그 운영 정책

증거 클립(evidence clip)과 클립 감사 로그(audit log)의 보관·회전·접근 기록에 대한
운영/컴플라이언스 참조 문서다. `#131`에서 지적된 세 가지 라이브 격차(감사 로그
커버리지 누락, 감사 파일 무제한 증가, 레거시 액터 처리 경로에 대한 문서 부재)를
다룬다.

## 1. 클립 보관 정책 (retention)

클립 자체(영상 파일 + `manifest.json`)의 보관·회전은 워커 쪽
`worker/pipeline/output/evidence/evidence_retention.py`와
`worker/pipeline/output/evidence/clip_config.py`에 이미 구현되어 있다. 이 문서는
그 동작을 요약하고, API 쪽 감사 로그 정책과의 관계를 명시한다.

### 1-1. 보관 일수 하한 (floor)

- `worker/pipeline/output/evidence/clip_config.py:17` — `MIN_RETENTION_DAYS = 60`.
- `configured_retention_days()` (`clip_config.py:32-39`)는 다음 환경변수를
  순서대로 읽는다: `CLIP_STORE_RETENTION_DAYS` → (없으면) `CLIP_RETENTION_DAYS`.
  둘 다 비어 있으면 기본값 `DEFAULT_RETENTION_DAYS = 60`을 쓴다.
- 값이 설정되어 있어도 **60일 미만으로는 절대 내려가지 않는다**:
  `max(MIN_RETENTION_DAYS, int(raw))`. 즉 운영자가 실수로 `CLIP_RETENTION_DAYS=7`을
  넣어도 실제 보관 기간은 60일로 강제된다.

| 환경변수 | 우선순위 | 기본값 | 하한 |
| --- | --- | --- | --- |
| `CLIP_STORE_RETENTION_DAYS` | 1순위 | - | 60일 |
| `CLIP_RETENTION_DAYS` | 2순위 (fallback) | - | 60일 |
| (미설정) | - | 60일 | 60일 |

### 1-2. 회전/삭제(purge) 메커니즘

`EvidenceRetention.rotate()` (`evidence_retention.py:102-145`)가 후보 클립
목록을 `finalized_at` 오름차순으로 정렬한 뒤, 다음 순서로 처리한다:

1. **보류(hold) 확인** — `is_held(clip_id)`가 참이면 삭제하지 않고 `HELD`로 기록한다
   (예: 사건 조사/법적 보류 중인 클립).
2. **보관 기한 확인** — `candidate.finalized_at > retention_cutoff`이면(즉 아직
   보관 기한 내이면) 건너뛴다. `retention_cutoff`는 호출자가
   `configured_retention_days()` 기반으로 계산해 넘긴다.
3. **삭제 전 검증** — `_verify_candidate()` (`evidence_retention.py:151-200`)가
   클립 디렉터리 경로, 심볼릭 링크 여부, `manifest.json`의 무결성/finalized 상태,
   미디어 파일 경로 일치 여부를 확인한다. 검증에 실패하면 삭제하지 않고
   `UNVERIFIABLE` 등으로 기록한다 — **불확실하면 지우지 않는다**가 원칙이다.
4. **삭제 실행** — 검증을 통과해야만 `shutil.rmtree()`로 클립 디렉터리를 삭제하고,
   삭제 후 경로가 실제로 사라졌는지 다시 확인한다(`PurgeResult.VERIFIED` 아니면
   `VERIFICATION_FAILED`).

디스크 사용량 상한(`disk_high_watermark`, 기본 `DEFAULT_DISK_HIGH_WATERMARK = 0.80`,
`CLIP_STORE_MAX_USAGE`/`CLIP_DISK_HIGH_WATERMARK` 환경변수로 조정 가능,
`clip_config.py:42-52`)을 넘으면 `RotationReport.pressure_blocked = True`로
보고되어 운영자가 디스크 압박 상황을 인지할 수 있다 — 다만 이 자체가 삭제
로직을 우회하지는 않는다(60일 하한과 보류 상태는 여전히 지켜진다).

## 2. 감사 로그 커버리지 (audit log coverage)

감사 로그(`AuditLogStore`, `backend/app/features/clips/audit_log.py`)는 클립
관련 API 접근을 JSONL로 append-only 기록한다. `#131` 이전에는 재생(play)과
라벨링(label)만 기록되고, 클립 목록 조회와 감사 로그 열람 자체는 기록되지 않는
격차가 있었다. 아래 표가 현재(이 변경 이후) 커버리지다.

| 엔드포인트 | 액션(action) | clip_id | 액터 해석 |
| --- | --- | --- | --- |
| `GET /api/v1/clips` | `list` | `-` (특정 클립에 국한되지 않음, `AUDIT_NO_CLIP_ID` 상수) | `_authorize()`의 반환값 |
| `GET /api/v1/clips/{clip_id}/video` | `play` | 실제 `clip_id` | 인증된 대시보드 세션 사용자명 |
| `PUT /api/v1/clips/{clip_id}/label` | `label` | 실제 `clip_id` | 요청의 `reviewer` (없으면 인증된 액터로 대체) |
| `GET /api/v1/audit` | `audit-view` | `-` (`AUDIT_NO_CLIP_ID`) | `_authorize()`의 반환값 |

`GET /api/v1/clips`와 `GET /api/v1/audit`는 하나의 클립에 국한되지 않는
액션이므로, `AuditLogStore.append()`가 요구하는 `clip_id` 필드에는 센티널 값
`AUDIT_NO_CLIP_ID = "-"`를 쓴다(`is_valid_clip_id()` 정규식
`^[A-Za-z0-9_-]{1,128}$`를 그대로 통과하므로 기존 검증 경로를 바꾸지 않는다).

`GET /api/v1/audit`는 자기 자신의 열람 기록이 응답에 섞이지 않도록, 먼저
`list_entries()`로 기존 항목을 스냅샷한 뒤 그 스냅샷을 응답으로 반환하고, 그
**다음에** `audit-view` 항목을 append한다(`router.py`의 `list_audit()`). 즉
연속으로 두 번 `GET /audit`를 호출하면 두 번째 응답에서만 첫 번째 호출의
`audit-view` 기록을 볼 수 있다.

모든 `append()` 호출은 기존과 동일하게 `post_backend_backup("clip_audit", entry)`를
통해 백엔드로도 best-effort 전송된다(`API_BACKEND_CLIP_EVENTS_URL` 설정 시).
즉 클립 목록 조회·감사 로그 열람도 다른 감사 액션과 동일하게 백엔드로 미러링된다.

## 3. 감사 파일 회전(rotation) 정책

`#131` 이전에는 `audit.jsonl`이 무한정 누적되는 격차가 있었다. 이제
`AuditLogStore.append()`가 매 기록 전에 현재 파일 크기를 확인하고, 임계값을
넘으면 타임스탬프가 붙은 아카이브 파일로 회전(rotate)한다.

### 3-1. 회전 임계값

- 기본값 `DEFAULT_AUDIT_LOG_MAX_BYTES = 10 MiB` (`10 * 1024 * 1024` 바이트).
- 환경변수 `API_AUDIT_LOG_MAX_BYTES`로 재정의 가능 (바이트 단위 정수). 값이
  없거나 0 이하/파싱 불가면 기본값으로 폴백한다.
- 근거: 감사 로그 한 줄은 `ts`/`actor`/`action`/`clip_id` 네 필드로 약
  120~160바이트다. 10 MiB면 회전 사이에 약 7만~9만 건을 담을 수 있어, 단일
  시설의 재생/라벨링/목록조회/감사열람 트래픽 대비 충분히 여유롭다.
- 라이브 파일이 존재하지 않는 상태(최초 기동)에서는 회전 검사가 그냥
  스킵된다 — 회전은 "이미 쓰여진 파일을 다시 쓸 때"만 의미가 있다.

### 3-2. 회전 방식

- `os.replace(self.path, archive_path)`로 라이브 파일(`audit.jsonl`)을
  아카이브 파일로 **원자적으로 rename**한다. 같은 파일시스템 내 rename은
  POSIX에서 원자적이므로, 동시에 읽는 프로세스는 회전 전 전체 파일(구 경로)
  또는 이미 rename된 아카이브(신 경로) 둘 중 하나만 보게 되며 절반만 쓰인
  파일을 보는 경우는 없다.
- 아카이브 파일명은 `audit-<UTC 타임스탬프, 마이크로초까지>.jsonl` 형식이다
  (예: `audit-20260803T120000123456Z.jsonl`). 같은 마이크로초에 충돌이 나면
  `-1`, `-2` 같은 접미사를 붙여 유일한 이름을 만든다.
- rename 직후 다음 `append()` 호출이 같은 경로(`audit.jsonl`)에 새 파일을
  만들어 이어서 기록한다 — 즉 회전은 기록 흐름을 끊지 않는다.
- rename이 `OSError`로 실패하면(예: 권한 문제) 회전을 건너뛰고 stderr에
  로그만 남긴다 — 회전 실패가 감사 기록 자체를 막지는 않는다(best-effort).

### 3-3. 아카이브 보관(prune) 정책

- 회전이 일어날 때마다, 같은 디렉터리의 `audit-*.jsonl` 아카이브들을 훑어
  파일의 mtime 기준으로 보관 기한이 지난 것을 삭제한다.
- 기본 보관 기간 `DEFAULT_AUDIT_ARCHIVE_RETENTION_DAYS = 60일`, 하한
  `MIN_AUDIT_ARCHIVE_RETENTION_DAYS = 60일`.
- 환경변수 `API_AUDIT_ARCHIVE_RETENTION_DAYS`로 재정의 가능하지만, **60일
  미만으로는 내려가지 않는다** (`max(60, value)`) — 클립 자체의 보관 하한
  (`worker/pipeline/output/evidence/clip_config.py:17`의
  `MIN_RETENTION_DAYS = 60`)과 정확히 동일한 하한을 감사 아카이브에도 걸어서,
  "클립은 사라졌는데 그 클립에 대한 감사 기록(누가 언제 재생/라벨링했는지)도
  같이 사라지는" 상황이 나지 않게 한다. 감사 아카이브는 항상 클립 자체와
  같거나 더 오래 남는다.
- 아카이브 삭제가 `OSError`로 실패하면(권한 등) 해당 파일은 건너뛰고
  stderr에 로그만 남긴다 — 다음 회전 때 다시 시도된다.

| 환경변수 | 기본값 | 하한 | 비고 |
| --- | --- | --- | --- |
| `API_AUDIT_LOG_MAX_BYTES` | 10 MiB (`10485760`) | 없음(0 이하는 무시하고 기본값 사용) | 라이브 파일 회전 임계값 |
| `API_AUDIT_ARCHIVE_RETENTION_DAYS` | 60일 | 60일 | 아카이브 보관 기간, 클립 보관 하한과 동일 |

## 4. 감사 액터 인증 경계

클립 API는 서버가 발급한 HttpOnly 대시보드 세션 쿠키만 운영자 권한으로
인정한다. 워커 relay 토큰은 `Authorization` 헤더, `X-Edge-Relay-Token`
헤더, `token` 쿼리 중 어느 형태로 보내도 클립 목록·재생·라벨링·감사 로그
열람 권한이 되지 않는다.

`dashboard_sessions()`는 영구 저장된 자격증명을 우선 사용하고, 아직 회전하지
않은 설치에서는 배포 시 명시한 `API_DASHBOARD_USERNAME` /
`API_DASHBOARD_PASSWORD` 부트스트랩 쌍을 사용한다. 두 값이 없거나 불완전하거나
저장소를 읽을 수 없으면 503으로 실패하며 내장 `admin`/`admin` 폴백은 없다.

따라서 감사 로그의 `actor` 필드는 실제 대시보드 세션 사용자명(또는 라벨링 시
명시적으로 지정된 `reviewer`)이다. 과거의 relay-token 호환 분기와
`"operator"`/`"bearer"`/`"legacy-dashboard"` 제네릭 actor는 제거되었다.

## 관련 문서

- [`docs/architecture.md`](../architecture.md) — 전체 아키텍처, 워커/API 레이어 구성.
- `worker/pipeline/output/evidence/evidence_retention.py`,
  `worker/pipeline/output/evidence/clip_config.py` — 클립 보관/회전 구현.
- `backend/app/features/clips/audit_log.py`,
  `backend/app/features/clips/router.py` — 감사 로그 구현/엔드포인트 배선.
- `backend/app/shared/dashboard_auth.py` — 대시보드 세션 인증, 레거시 경로 주석.

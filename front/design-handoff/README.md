# Handoff: Senior AI Lab Edge Console (관제 콘솔 리디자인)

## Overview
`SeniorAILab/eldercare-fall-ml-v2`의 `front/` SPA를 대체하는 엣지 관제 콘솔 리디자인.
요양원 1곳에 설치되는 단일 시설 엣지 장비의 콘솔로, 사용자는 **CCTV 설치 기사와 운영팀**(고객 아님).
핵심 결정: 다중 시설 개념 제거, 백엔드 리본 제거, CUDA 전용 진단 제거(실행 디바이스에 맞는 항목만),
영상은 object-cover로 꽉 채움, 페이지는 관제 / 이벤트 / 설정 3개 + 로그인.

## About the Design Files
이 번들의 `Eldercare Prototype.dc.html`은 **HTML로 만든 디자인 레퍼런스**(인터랙티브 프로토타입)이며 프로덕션 코드가 아니다.
할 일은 이 디자인을 **대상 코드베이스의 기존 환경(React + Vite + TypeScript, `front/src`)에서 그 컨벤션대로 재구현**하는 것.
프로토타입의 마크업/로직을 복사하지 말 것. 기존 repo의 feature 폴더 구조, API 클라이언트, 상태 관리 패턴을 따를 것.
`Eldercare Dashboard.dc.html`은 현재 UI 재현 + 검토 문서(참고용).

## Fidelity
**High-fidelity.** 색·타이포·간격·인터랙션 모두 확정값. 픽셀 단위로 재현하되, 스타일 값은 아래 디자인 토큰의
CSS 변수로 구현할 것 (기존 `front/src/styles/tokens-base.css`를 이 토큰들로 교체).

## Design Tokens
프로토타입은 OssHub 디자인 시스템 기반. 셰이프: 카드 `border-radius: 12px`(rounded-xl), 컨트롤 `8px`(rounded-lg), 배지 pill.
카드는 `1px solid var(--border)` + 흰 배경, 그림자 없음(모달만 `0 12px 36px rgba(0,0,0,.2)`).

색 (light):
- `--background: #ffffff`, `--foreground: #1a1a1a`
- `--card: #ffffff`, `--border: #e4e4e7`, `--input: #e4e4e7`
- `--muted: #f4f4f5`, `--muted-foreground: #71717a`
- `--primary: #2f6fb0` (브랜드 블루), on-primary 흰색
- `--destructive: #dc2626`
- 상태 배지: approved(초록 bg `#e7f7ee` / fg `#166e3d`), rejected(빨강 bg `#fdeceb` / fg `#b8261b`), pending(호박 bg `#fdf2df` / fg `#884e07`), closed(회색)
- 오버레이 색: 침대 영역/세그멘테이션 `#2bb6a3`(teal), 스켈레톤/박스 흰색 90% 불투명
- 영상 위 라벨: `background: rgba(0,0,0,.72); color: #fff`

타이포: Pretendard(폴백 Apple SD Gothic Neo, Noto Sans KR, system-ui).
- 페이지 제목 20px/600, 카드 제목 16px/600, 본문 14px, 메타 12px, 시간·수치는 `tabular-nums`, 경로·RTSP·파일명은 `font-mono` 12px

간격: 페이지 패딩 20-24px, 카드 패딩 20px, 카드 간 16px, 카드 내 행 간 8-12px, 최대 폭 1280px 중앙 정렬.

## Screens / Views

### 0. 로그인
- 중앙 카드 380px: 제목 "Senior AI Lab Edge", 아이디 / 비밀번호 입력(h-36px), primary 로그인 버튼 풀폭
- 성공 시 관제로 이동

### 1. 공통 셸 (NavBar)
- 상단 고정 바 56px, 하단 보더. 좌측 브랜드 텍스트 "Senior AI Lab Edge"
- 중앙/좌측 네비 3개: 관제(그리드 아이콘) · 이벤트(시계) · 설정(톱니). 활성 = `bg-muted` 라운드 칩 + semibold. 호실 상세도 "관제" 활성
- 우측: 계정 아이콘(사람 실루엣, ghost icon 버튼) → 계정 설정 모달, "로그아웃" outline sm 버튼
- **백엔드 연결 배지·시설명은 네비에 없음** (연결 상태는 설정 페이지 서버 연결 카드에만)

### 2. 관제 (카메라 월) — 시작 화면
- 헤더 줄: "관제" 제목 + 온라인 n / 오프라인 n 상태 배지(개수는 논브레이킹 스페이스로 붙임), 우측에 층 필터 select(전체/1층/2층…)
- 4열 그리드 gap 12px. 타일 = rounded-lg 보더 카드, 16:9 영상(**object-cover, 여백 금지**)
- 타일 하단 바: 온라인 = 영상 위 `linear-gradient(transparent, rgba(0,0,0,.72))` 오버레이에 흰 이름 + 초록 점(우측), 오프라인 = 회색 `bg-muted` 16:9 자리에 "오프라인" 텍스트, 하단 바는 흰 배경 + 보더 + 빨간 점
- 타일 전체 클릭 → 호실 상세. 층 필터는 즉시 필터링
- 낙상 알림 등 이벤트 배지는 여기 표시하지 않음 — 이 화면의 관심사는 카메라 생존 여부

### 3. 호실 상세
- 브레드크럼: "관제 / {호실명}" + 온라인/오프라인 배지. 관제 링크로 복귀(네비의 관제도 동일)
- 좌 2/3: 라이브 뷰 16:9
  - 좌하단 라벨: "라이브 · 온라인" (오버레이가 낙상이면 "라이브 · 낙상 없음")
  - **우상단 진단 오버레이**(font-mono 12px): "디코드 14.8 FPS · 추론 7.2 FPS · 프레임 42,318" — 실시간 갱신
  - 탐지 오버레이는 **단일 선택**: 없음 / 침대 이탈(teal 점선 침대 영역 + 인물 박스·스켈레톤 + "인물 1 · 침대 안 · 누움") / 낙상(인물 박스·스켈레톤 + "인물 1 · 서있음 · 낙상 없음")
  - 오프라인이면: 회색 패널 "카메라에 연결할 수 없습니다 — 탐지가 중단된 상태입니다" + [재연결 시도] [연결 관리] 버튼, 라벨·오버레이 숨김
- 우 1/3 스택:
  - 카메라 정보 카드: 제목 + [연결 관리] outline xs → 연결 관리 모달. dl: 층, RTSP(경로 마스킹 `rtsp://…/•••`)
  - 탐지 이벤트 카드: 헤더 우측 톱니 아이콘 링크 → 설정 페이지. 행 = 이벤트명 / 시간대(우측 정렬 tabular) / 상태 배지("탐지 중" approved, "꺼짐" closed, 카메라 오프라인 시 "중단됨" pending)
  - 카드 하단 보더 위: "오버레이" 라벨 + 칩 3개(없음/침대 이탈/낙상), 선택 = primary 틴트 칩 + " ✓"
- 하단: "이벤트 히스토리" + 정렬 select(최신순/종류별)
  - 최신순 = 4열 클립 카드 그리드. 종류별 = "침대 이탈 · n건" / "낙상 · n건" 그룹 헤딩(14px muted)으로 분리
  - 클립 카드 = 16:9 썸네일 + 하단(종류명 + 시간), 클릭 → 클립 재생 모달

### 4. 이벤트
- 제목 "이벤트" + 필터 한 줄: 종류 칩(전체 n / 침대 이탈 n / 낙상 n) 좌측, 카메라 select 우측(margin-left:auto)
- 4열 클립 카드(썸네일 좌상단에 카메라명 라벨) → 동일한 클립 재생 모달
- 빈 상태: "조건에 맞는 이벤트가 없습니다."

### 5. 설정 (7:5 2컬럼)
좌측:
- **카메라 카드**: 헤더 "카메라" + [카메라 등록] 버튼(카메라+플러스 아이콘 + 라벨, primary sm) → 등록 모달
  - 테이블: 카메라 / 층 / 상태(온라인·오프라인 배지) / 작업(수정 = primary 텍스트 → 연결 관리 모달, 삭제 = destructive 텍스트, 즉시 삭제 + 토스트)
  - 오프라인 행은 `bg-status-rejected-bg` 틴트
- 하단 버전 라인(font-mono 12px muted): "v2026.07.28 · ml-api@8c4e21 · ml-worker@1fa905" — 기술 정보 카드는 없음, 이 한 줄이 전부

우측 스택:
- **탐지 설정 카드** (제목 옆 "모든 카메라에 적용" 12px muted, 우측 연필 아이콘)
  - 읽기 모드(기본): 3컬럼 그리드(이벤트명 1fr / 시간대 auto 우측 정렬 / 배지 auto) — "침대 이탈 21:00–06:00 [탐지 중]" / "낙상 항상 [탐지 중]"
  - 연필 → 편집 모드: 행마다 체크박스(on/off) + select(항상/시간 지정) + 시간 지정 시 HH:MM 인풋 2개. **변경 즉시 저장**(저장 버튼 없음), "변경 즉시 모든 카메라에 적용됩니다." 안내 + [완료] ghost로 닫기
- **서버 연결 카드** (제목 + 우측 [정상 배지 + 연필])
  - 읽기: dl — 시설 ID(font-mono), 시설 토큰(`••••2f9a`), 마지막 동기화
  - 연필 → 보더 아래 편집 폼: 시설 ID / 토큰 인풋 + [연결] outline(성공 시 "연결 성공 · …" 초록 12px 인라인 메시지) + [저장] primary(닫힘 + 토스트)
- **처리 상태 카드** (제목 + 정상 배지): dl 4행 — 실행 디바이스 "Apple M2 · MPS"(CUDA 장비면 GPU명·CUDA), 디코드 "opencv (CPU)", 인코드 "libx264", 전송 지연 "최대 1.84초". FPS 평균·NVML 등 CUDA 전용 진단은 표시하지 않음. 값은 repo의 runtime status API(`decode_diagnostics`, clip encoder 필드)에서
- **클립 저장 공간 카드**: 제목 + "2.0 / 10.0 GB"(tabular), 프로그레스 바(h-8px pill, primary), 하단 "저장 위치 /mnt/data/clips"(font-mono) + [변경] ghost xs → **폴더 탐색기 모달**

### 모달 (공통: fixed 오버레이 rgba(0,0,0,.42), 카드 rounded-xl p-20px, 밖 클릭으로 닫힘)
1. **카메라 등록 (2단계, 440px→640px)**: 스텝 인디케이터 "1 연결 정보 → 2 침대 영역"(현재 스텝 semibold, 완료 스텝 링크로 복귀 가능)
   - 1단계: 이름, 층 칩 선택(+ 층 추가 = dashed 칩, 층 목록에 "n층" 추가), RTSP 주소. [취소][다음] — 이름 비면 진행 차단 토스트
   - 2단계: 라이브 미리보기 16:9. [인식 시작]→ teal 점선 폴리곤 펄스 애니메이션(1.2s ease-in-out 무한) + "인식 중…" 라벨, 1.5초 후 확정 폴리곤 + "침대 · 자동 인식됨" + "침대 1개 인식됨" 라벨. [정지] 가능. [이전][저장하고 완료] — 인식 완료 전 저장 차단 토스트. YOLO 세그멘테이션이 자동으로 침대를 따는 것이며 사용자가 그리지 않음
2. **연결 관리 (수정, 440px)**: 제목 + 온라인/오프라인 배지. 이름 / 층 칩 / RTSP. "침대 영역" 행 = [인식 완료 approved | 인식 필요 rejected] 배지 + 새로고침 아이콘 → **다시 인식 모달**(2단계와 동일 UI, ▶/⏹ 아이콘 버튼, [취소][완료]). 푸터: [삭제 destructive] 좌측, [연결][저장] 우측 — [연결] 성공 시 해당 카메라 online 전환 + "연결 성공" 토스트
3. **클립 재생 (720px)**: 헤더 = "종류 · 카메라명" + 시간 + × 닫기 아이콘. 플레이어 16:9(중앙 재생 버튼, 하단 시크바 + 0:04/0:12). dl: 파일(전체 경로+파일명 font-mono, 예 `/mnt/data/clips/cam-101a/2026-08-02_0312_bedexit.mp4`), 길이 0:12, 해상도 1920×1080, 크기 8.4 MB — 각각 개별 행. 푸터 우측 [다운로드] 하나
4. **폴더 탐색기 (420px)**: 현재 경로 줄(↑ 상위 이동 + font-mono 경로), 폴더 리스트(폴더 아이콘 + 이름, 클릭 드릴다운, 빈 폴더 = "하위 폴더 없음"), [취소][이 위치 사용]
5. **계정 설정 (380px)**: 아이디 / 새 비밀번호 / 새 비밀번호 확인 (현재 비밀번호 없음 — 단일 관리자 계정, 이미 인증됨). [취소][변경하기]

## Interactions & Behavior
- 토스트: 우하단 고정 카드, 2.2초 자동 소멸. 파괴적/완료 액션에 사용 (삭제, 등록 완료, 저장 위치 변경, 연결 성공, 차단 안내)
- 탐지 설정 편집은 저장 버튼 없이 즉시 반영. 서버 연결·계정은 명시적 저장
- 낙상/침대 이탈 스케줄: 이벤트별 독립, "항상" 또는 시간 범위(야간 21:00–06:00 등)
- 관제 타일 hover: cursor pointer (별도 효과 없음)
- 세그멘테이션 인식 애니메이션: `@keyframes pulseZone { 50% { opacity:.35 } }` 1.2s 무한

## State Management
- `page`: login | wall | detail | events | settings, `sel`(선택 카메라 id), `modal`(단일 슬롯: edit/add1/add2/reseg/clip/folder/account)
- `cams[]`: { id, name, floor, online, rtsp, seg(침대 영역 인식 여부) } — 수정·삭제·등록·연결 성공이 모두 이 배열에 반영
- `floors[]`: 동적 (층 추가 가능 — 요양원마다 다름)
- `events[]`: { id, cam, type: bedexit|fall, time } — 필터·정렬·그룹핑은 파생값
- 탐지 설정: 이벤트별 { on, mode(항상|시간 지정), start, end } — 전역(모든 카메라)
- 실데이터: 기존 repo의 카메라 CRUD·runtime status·clip API에 연결. 카메라 동기화(edge→백엔드 push)는 저장 시 자동 수행, 별도 버튼 없음

## Assets
외부 에셋 없음. 아이콘은 전부 인라인 SVG(24 viewBox, stroke 1.75–1.9, round cap) — 그리드/시계/톱니/사람/연필/새로고침/폴더/재생/정지/×/카메라+. 영상 자리는 프로토타입에선 어두운 그라디언트 placeholder.

## Files
- `Eldercare Prototype.dc.html` — 인터랙티브 프로토타입 (전 화면 + 전 모달 + 상태 로직, 이 문서의 기준)
- `Eldercare Dashboard.dc.html` — 현재 UI 재현(1a–1f) 및 리디자인 검토 문서 (배경 참고)
- 대상 코드베이스: `SeniorAILab/eldercare-fall-ml-v2` `front/src` (React + Vite + TS)

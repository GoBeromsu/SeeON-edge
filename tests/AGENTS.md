# TESTS KNOWLEDGE BASE

Own pytest coverage for the ML uv project, including dependency-boundary guards and fixtures.

## Local Ownership

- `test_contract_symbol_exports.py`: contract-symbol export checks (runner/tracker/worker_config). Import boundaries are enforced by import-linter (`uv run --group lint lint-imports`), not a pytest walker.
- `edge_worker_fixtures.py`, `demo_app_control_helpers.py`, `e2e_worker_relay_fixtures.py`: shared test helpers.
- `test_*`: package-specific and cross-boundary tests.
- `test_e2e_night_bed_exit_relay.py`: real-stack E2E (marked `real_stack`) -- synthetic RTSP via `mediamtx` + `ffmpeg` into the real worker/backend composition. Deselected in CI (`-m "not real_stack and not heavy and not integration"`) and skipped locally when `mediamtx` is not on PATH; see "Real-stack E2E" below.

## Imports

Allowed: any production package needed by the test under coverage, pytest helpers, and local fixtures.

Forbidden: importing private generated data/model artifacts as required test inputs, relying on camera hardware, network services, or uncommitted local files for default tests.

## Commands

```bash
uv run --group lint lint-imports
```

## Real-stack E2E

`test_e2e_night_bed_exit_relay.py` is marked `real_stack` and requires the `mediamtx`
RTSP server binary (plus `ffmpeg`, already a default dev dependency) on PATH. CI
deselects it (`uv run pytest -q -m "not real_stack and not heavy and not integration"`) rather than fetching an
external binary, per `tests/test_public_repository_privacy.py`'s untrusted-CI
contract -- it runs locally only. To run it:

```bash
# install mediamtx and put it on PATH, e.g.:
#   brew install mediamtx           # macOS
#   see https://github.com/bluenviron/mediamtx for other platforms
uv run pytest -m real_stack
```

Without `mediamtx` on PATH, the tests are skipped (not errored) with an explicit reason.

## Live-stack integration (`integration`)

`test_cloud_edge_provisioning_integration.py` is marked `integration` and drives a
**running, already-enrolled ml-api** — it is not a mock-backed test. CI deselects it
(the `not integration` above); the marker always said so in `pyproject.toml`, but the
CI argument only started honouring it once the test began failing on `main` with
`RuntimeError: CLOUD_EDGE_ML_URL is required`. Unlike `real_stack` it needs no RTSP
tooling, so `real_stack` would be the wrong marker for it.

It fails loudly rather than skipping, because every one of its inputs is a deliberate
pointer at live state that must not be guessed. Run it against a real Edge:

```bash
CLOUD_EDGE_ML_URL=http://<edge-host>:8000 \
CLOUD_EDGE_RELAY_TOKEN=... \
CLOUD_EDGE_ML_CATALOG_PATH=/var/lib/ml-api/catalog.sqlite3 \
CLOUD_EDGE_PRE_V1_BACKUP_PATH=... \
CLOUD_EDGE_SECRET_HANDOFF_PATH=... \
uv run pytest -m integration
```

It writes to the catalog sqlite file it is pointed at, so never aim it at a production
volume.

## 호스트 의존 테스트 (Local Hero) — 금지

테스트 결과가 **코드가 아니라 실행한 머신**으로 갈리면 그건 결함이다. 표준 용어로
hermetic 하지 않다고 하고, 안티패턴 이름은 **Local Hero**(= Hidden Dependency,
Operating System Evangelist, Environmental Vandal)다. 정의는 "작성된 개발 환경에
특정한 무언가에 의존하는 테스트 — 개발 머신에선 통과하고 다른 데선 실패한다".

목적 하나만 기억하면 된다: **테스트 실패는 코드의 버그를 뜻해야지 외부 요인을 뜻하면
안 된다.** 이 원칙이 깨지면 실패가 신호가 아니라 잡음이 되고, 진짜 회귀를 판정할 때마다
대조군을 따로 돌려야 한다.

참고:
- https://dzone.com/articles/unit-testing-anti-patterns-full-list (Local Hero 정의)
- https://testrigor.com/blog/hermetic-testing/ (hermetic testing 목적)
- https://thecodinggopher.substack.com/p/hermetic-software-explained (호스트 비의존 = 결정성)
- https://conf.researchr.org/details/icst-2023/cciw-2023-papers/3/ (Google, hermetic 환경으로 flakiness 감소)

### 이 저장소에서 실제로 터진 사례

**1. umask 상속 (2026-08-17, 84건)**

`clip_consistency` 계열 84건이 실패했는데 코드 변경과 무관했다. 같은 커밋에서:

```
umask 0002 → 52 failed, 3 passed
umask 0022 → 55 passed
```

원인: 픽스처가 clip store 디렉터리를 모드 없이 `mkdir()` 로 만들어 호스트 umask 를
상속했다. 프로덕션 `validate_directory` 는 그룹 쓰기(`0o022`)를 정당하게 거부하므로
**프로덕션이 옳고 테스트가 틀렸다**.

함정 두 개를 같이 밟았다:
- `mkdir(parents=True, mode=0o755)` 는 **leaf 에만** 모드를 적용한다. 부모는 여전히
  umask 를 탄다. 루트부터 한 단계씩 명시해야 한다.
- 검증기가 `clips_root.iterdir()` 로 **모든** 하위 디렉터리를 훑으므로 개별 호출부를
  쫓는 방식은 브리틀하다.

처방: `tests/conftest.py` 의 autouse 픽스처가 umask 를 `0o022` 로 고정한다. 환경 입력을
통제하는 것이 hermetic 처방이다. 의도적으로 안전하지 않은 권한을 시험하는 테스트는
`chmod` 로 직접 지정하므로 영향받지 않는다.

**2. 하드웨어 부재를 단언 (진행 중, 6건)**

`test_probe_*_is_honest_on_this_dev_machine`, `test_probe_*_unavailable_on_this_non_nvidia_host`
류는 "이 머신에는 GPU 가 없다"를 단언한다. docstring 에 전제가 적혀 있다 — "이 레포의
macOS CI/dev 머신에는 NVML 이 없다". 호스트가 nvidia 프로파일로 전환되어 RTX 5070 Ti
가 붙자 전부 뒤집혔다.

이런 테스트의 진짜 의도는 "프로브가 **정직한가**"이지 "이 머신에 GPU 가 없는가"가
아니다. 올바른 형태는 호스트 상태를 단언하지 말고 **불변식**을 단언하는 것이다:
`available=True` 면 `driver_version` 과 `device_name` 이 채워져 있어야 한다 —
이건 어느 머신에서든 성립한다.

### 새 테스트를 쓸 때 확인할 것

1. 이 테스트가 **다른 머신에서도** 같은 결과를 내는가? (umask, GPU, PATH, 로케일,
   타임존, CPU 코어 수, 파일시스템)
2. 호스트의 **상태**를 단언하고 있지 않은가? 단언해야 하는 건 코드의 **불변식**이다.
3. 환경이 실제로 필요하면 단언 대신 **가드**로 건너뛴다. 이 저장소는 이미 그 패턴을
   쓴다 — `test_e2e_night_bed_exit_relay.py` 는 `mediamtx` 가 PATH 에 없으면
   `pytest.skip` 한다. 없다고 단언하지 않는다.
4. 디렉터리를 만들면 모드를 명시한다. 부모까지.

## Gotchas

Keep boundary tests small and explicit. When import policy changes, update the `[tool.importlinter]` contracts in `pyproject.toml` and the relevant AGENTS files in the same change.
## Test Boundary

- Keep default tests deterministic and hardware-free.
- Exercise both allowed and forbidden imports when changing the dependency ladder.
- Use package-focused tests before the full suite.

## `heavy` — CI에서 돌리지 않는 테스트

`heavy`로 표시한 테스트는 **실제 인터프리터 서브프로세스를 띄우고, 그
프로세스가 벽시계 감시(워치독 deadline, hard-exit 경로)로 끝나기를
기다린다.** 한가한 호스트에서는 정확하지만 부하가 걸린 CI 러너에서는
**프로세스 기동만으로 deadline을 넘겨** 깨진다.

실제로 겪었다 — 전체 스위트가 112초에서 318초로 늘어난 실행에서
`test_watchdog_subprocess_hard_exits_with_fatal_accelerator_code`만
`TimeoutExpired`로 두 번 연속 실패했다. 로컬에서는 2.1초에 통과한다.

그래서 CI는 `-m "not real_stack and not heavy and not integration"`로 제외한다. 타임아웃 여유를
늘리는 것으로 덮지 않는다 — 그러면 CI 시간만 늘고 같은 종류의 불안정이
남는다.

`worker/runtime/`(워치독, 폴트 핸들러, 가속기 경로)을 건드리면 **로컬에서
직접 돌린다.**

```bash
uv run pytest -q -m heavy
```

# IDIS WebGuard 조작 — RTP 암호화 해제

`SKILL.md` 3 단계(454 의 원인)에서 여기로 온다. 브라우저로 카메라를 만지기
**전에 반드시 읽어라.** 특히 6 절(직접 POST 금지)과 7 절(건드리면 안 되는 것)을
건너뛰면 admin 비밀번호가 깨지거나 접속 경로 자체가 끊길 수 있다.

## 목차

1. [왜 IDIS 전용인가](#1-왜-idis-전용인가)
2. [왜 프록시가 필요한가](#2-왜-프록시가-필요한가-scriptscamproxypy)
3. [페이지를 여는 순서](#3-페이지를-여는-순서)
4. [레지스트리 모델](#4-레지스트리-모델)
5. [검증된 수정 절차](#5-검증된-수정-절차)
6. [레지스트리를 스크립트로 직접 POST 하면 안 되는 이유](#6-레지스트리를-스크립트로-직접-post-하면-안-되는-이유-가장-중요)
7. [건드리면 안 되는 것](#7-건드리면-안-되는-것)
8. [저장 후 검증은 브라우저 밖에서 두 번](#8-저장-후-검증은-브라우저-밖에서-두-번)
9. [브라우저 자동화 운영 주의](#9-브라우저-자동화-운영-주의)

## 1. 왜 IDIS 전용인가

이 문서에 나오는 `Page305`, `#use-rtp-encryption`, `REG.*` 레지스트리 트리,
`saveXmlRegistry`, DirectIP 8016 포트는 전부 IDIS 펌웨어 고유의 것이다.
Hanwha, Hikvision, Axis 카메라에는 하나도 통하지 않는다 — 페이지 구조도,
레지스트리 개념도, 저장 흐름도 벤더마다 다르게 짜여 있기 때문이다. 대상이
IDIS 계열(OpenIP/DirectIP, WebGuard 웹 설정 UI)이 아니라면 이 문서는 버리고
새로 조사해라.

## 2. 왜 프록시가 필요한가 (`scripts/camproxy.py`)

브라우저로 카메라를 직접 열지 않고 로컬 프록시를 거치는 이유는 하나가 아니라
겹겹이 쌓인 네 가지 장애물 때문이다.

- **Chrome 이 자동화를 막는다.** URL 에 자격증명을 심는 방식(`https://user:pass@host/`)을
  Chrome 이 차단하고, 자체서명 인증서 경고와 Digest 인증 팝업은 브라우저
  자동화 도구가 다룰 수 없는 네이티브 UI 로 뜬다. 프록시가 Digest 를 대신
  붙여서 평문 HTTP 로 내주면 브라우저는 로그인 화면 없이 바로 setup 페이지를
  받는다.
- **카메라의 TLS 가 요즘 기본값과 안 맞는다.** 카메라는 구형 TLS/암호군만
  쓴다. 파이썬 기본 SSL 컨텍스트는 TLS1.2 미만과 SECLEVEL 2 미만을 거부하도록
  잠겨 있어서 그대로 붙으면 핸드셰이크가 깨진다. 그래서 `_LegacyTLSAdapter`
  가 `TLSv1` 하한과 `DEFAULT@SECLEVEL=0` 으로 낮춰서 붙는다. 인증서 검증까지
  끄는 게 꺼림칙해 보일 수 있지만, 이 경로는 이미 SSH 터널 안이라 중간에서
  가로챌 수 있는 제3자가 없다 — 그래서 이 지점에서 검증을 끄는 것이 정당하다.
- **왕복 비용이 비싸 캐시 없이는 페이지가 죽는다.** `setup.html` 은 동기 XHR
  을 수백 번 던진다. 경로가 Tailscale → SSH 터널 → 카메라라 요청 하나하나가
  가볍지 않은데, 세션(TLS 커넥션) 재사용과 정적 자산 메모리 캐시가 없으면
  같은 js/css 를 수백 번 새로 협상하고 새로 받게 된다. 그러면 페이지 로드가
  몇 분씩 메인 스레드를 물고 늘어져 브라우저 자동화가 타임아웃으로 통째로
  죽는다.
- **카메라 펌웨어가 죽은 외부 스크립트를 참조한다.** `setup.html` 은 sha256
  구현을 `crypto-js.googlecode.com` 에서 받아오게 짜여 있다. 구글코드는 2016
  년에 폐쇄됐고, 그 도메인으로 가는 요청은 응답도 타임아웃도 없이 **영영
  끝나지 않는다.** 브라우저는 그 스크립트 로드가 끝나길 기다리다 멈춘다.
  프록시가 응답 본문에서 그 URL 을 로컬 사본(`scripts/sha256.js`, crypto-js
  3.1.2, 카메라가 기대하는 것과 같은 버전)으로 치환해서 이 요청 자체를
  없앤다. 이 결함은 저장소 이슈 #181 로도 등록돼 있다.
- **SessionID 가 양방향 헤더로 오간다.** 카메라는 로그인 성공 시 SessionID 를
  응답 헤더로 돌려주고, 웹앱은 이후 요청마다 그걸 다시 요청 헤더에 실어
  보낸다. 프록시가 이 헤더를 양쪽으로 그대로 통과시키지 않으면 두 번째
  요청부터 세션이 끊긴 것처럼 보인다.

채널 번호 규칙은 `edge.sh` 와 공유한다. 채널 N 은 로컬 프록시 포트 `194NN`,
그 프록시가 붙는 SSH 터널 포트는 `184NN` 이다 (예: 채널 3 → 프록시
`19403`, 터널 `18403`).

## 3. 페이지를 여는 순서

채널마다 **캐시를 먼저 데운 뒤** 브라우저로 연다. 캐시가 비어 있는 상태로
브라우저가 첫 요청을 던지면 위에서 설명한 왕복 비용을 그대로 맞고, 첫 로드가
타임아웃으로 죽는다. `curl` 로 미리 한 번 받아 두면 그 응답이 프록시 메모리에
캐시되고, 브라우저는 그 캐시를 즉답으로 받는다.

```bash
curl -s -o /dev/null --max-time 90 "http://127.0.0.1:194NN/setup/setup.html"
curl -s -o /dev/null --max-time 90 "http://127.0.0.1:194NN/setup/page.js"
# 그 다음 브라우저로 http://localhost:194NN/setup/setup.html
```

## 4. 레지스트리 모델

WebGuard 는 화면에 뿌리는 설정값을 `REG` 라는 인메모리 트리로 들고 있다.
트리의 각 노드는 다음 필드를 갖는다.

| 필드 | 뜻 |
|---|---|
| `$` | 현재 값 |
| `$type` | 값 타입 (예: `uint8`) |
| `$key` | 저장 시 쓰는 레지스트리 키 번호 |
| `$mod` | 저장 안 된 변경사항이 있는지 (더티 플래그) |

이 작업의 대상은 `REG.NetworkConfig.Rtsp.RtpEncryptionType` (`RtpEncryptionType`,
`uint8`, `$key=7`) 이다. 값 정의는 `RTP_ENCRYPT = {NONE: 0, S1: 1}` 이고,
**`1` 이면 RTSP `DESCRIBE` 가 454 로 죽는다. `0` 이어야 스트림이 나온다.**

## 5. 검증된 수정 절차

아래 스크립트는 실제 현장에서 검증됐다. 방어적으로 보이는 각 검사는 실제로
그 상황을 맞아서 넣은 것이니 지우지 마라.

```js
const log = {};
for (let i=0; i<40 && (typeof REG==='undefined' || !REG.NetworkConfig); i++) await new Promise(r=>setTimeout(r,1000));
if (typeof REG==='undefined' || !REG.NetworkConfig) { log.abort='REG not ready'; }
else {
  $('.ui-dialog-content').each(function(){ try { $(this).xdialog('close') } catch(e) {} });
  movePage('Page305');
  await new Promise(r=>setTimeout(r,3500));
  log.before = { checked: $('#use-rtp-encryption').length ? $('#use-rtp-encryption')[0].checked : null,
                 reg: REG.NetworkConfig.Rtsp.RtpEncryptionType.$, apply: $('#setup-apply').length };
  if (log.before.reg === 0) { log.result = 'ALREADY-OK'; }
  else if (log.before.checked !== true || log.before.reg !== 1 || log.before.apply !== 1) { log.result = 'ABORT unexpected pre-state'; }
  else {
    $('#use-rtp-encryption').click();
    await new Promise(r=>setTimeout(r,600));
    log.after = { checked: $('#use-rtp-encryption')[0].checked, reg: REG.NetworkConfig.Rtsp.RtpEncryptionType.$, mod: REG.NetworkConfig.Rtsp.RtpEncryptionType.$mod };
    if (log.after.checked === false && log.after.reg === 0 && log.after.mod === true) {
      $('#setup-apply').click();
      await new Promise(r=>setTimeout(r,9000));
      log.saved = { reg: REG.NetworkConfig.Rtsp.RtpEncryptionType.$, mod: REG.NetworkConfig.Rtsp.RtpEncryptionType.$mod };
      log.result = (log.saved.reg===0 && log.saved.mod===false) ? 'SAVED' : 'SAVE-UNCONFIRMED';
    } else { log.result = 'ABORT bad post-click state'; }
  }
}
JSON.stringify(log)
```

단계마다 왜 그렇게 짰는지 짚는다.

- **`REG` 준비를 최대 40 초 기다린다.** 이 페이지는 SPA 라 레지스트리를 로드
  직후가 아니라 비동기로 채운다. 성급하게 읽으면 `undefined` 를 보고 "설정이
  없다"고 잘못 판단하게 된다. 40 초는 느린 터널 경로에서도 채워지는 걸
  현장에서 확인한 여유값이다.
- **다이얼로그부터 닫는다.** 다른 모달(예: 로그인 알림, 다른 설정 저장 확인창)
  이 떠 있으면 그 아래 깔린 체크박스와 버튼은 클릭이 먹지 않는다. 시작하기
  전에 열려 있는 걸 전부 닫아야 이후 클릭이 의도한 요소에 맞는다.
- **사전 상태를 검사하고 어긋나면 중단한다.** `reg === 0` 이면 이미 해제돼
  있는 것이니 건드릴 필요가 없다 — 여기서 클릭하면 오히려 다시 켜는 꼴이
  된다. `checked`/`reg`/`apply` 조합이 예상과 다르면 이 카메라의 펌웨어
  버전이나 화면 구조가 지금 절차가 검증된 버전과 다르다는 신호다. 그 상태로
  계속 진행하면 엉뚱한 요소를 건드릴 위험이 있으니 멈추고 사람에게 보고해라.
- **`.checked = false` 를 대입하지 않고 `.click()` 을 쓴다.** 체크박스
  프로퍼티를 직접 바꾸면 화면 모양만 바뀔 뿐 레지스트리에는 아무 일도
  일어나지 않는다. 실제로 `REG` 를 쓰는 것은 체크박스에 걸린 change
  핸들러이고, 그 핸들러는 진짜 클릭(또는 클릭이 발생시키는 이벤트)에만
  반응한다. `.click()` 을 써야 그 핸들러가 타서 값이 실제로 바뀐다.
- **클릭 직후 `$mod === true` 를 확인한다.** 화면상 체크 표시가 없어졌다고
  해서 저장 대상이 됐다는 보장은 없다. 레지스트리가 스스로 "이 값이
  변경됐다"고 더티 마킹해야 `#setup-apply` 클릭 때 그게 저장 페이로드에
  실린다. `$mod` 가 여전히 `false` 면 클릭이 핸들러를 안 탄 것이니 저장해도
  아무 효과가 없다.
- **저장 후 `$mod === false` 를 확인한다.** 저장이 서버에 실제로 커밋되면
  펌웨어가 응답으로 더티 플래그를 내린다. 9 초를 기다린 뒤에도 `$mod` 가
  `true` 로 남아 있으면 저장 요청이 서버까지 못 갔거나 서버가 거부한
  것이다 — `SAVED` 라고 부르면 안 되고 `SAVE-UNCONFIRMED` 로 남겨서 다음
  단계(8 절)에서 반드시 다시 확인하게 만든다.

## 6. 레지스트리를 스크립트로 직접 POST 하면 안 되는 이유 (가장 중요)

체크박스를 클릭하는 대신 `REG.NetworkConfig.Rtsp.RtpEncryptionType.$ = 0` 을
대입하고 저장 API 를 직접 호출하고 싶어질 수 있다. 하지 마라. 코드를 보면
`saveXmlRegistry(_path, _mod)` 는 어떤 호출 경로를 타든 `_mod` 인자가 항상
`undefined` 인 채로 실행된다. 그 결과 이 함수는 "바뀐 값만" 이 아니라
**레지스트리 전체를 다시 쓴다.** 그리고 그 전체 쓰기 경로 안에서
`encryptByLocalRSA()` 가 저장되는 비밀번호 필드들을 **현재 세션의 RSA 키로
재암호화한다.**

문제는 이 세션 키가 항상 최초 로그인 시 발급된 키와 정확히 일치한다는
보장이 없다는 데 있다. 스크립트로 우회 경로를 타면서 세션 상태가 화면이
기대하는 것과 어긋나면, 저장되는 비밀번호가 **잘못된 키로 암호화돼 카메라
admin 계정 자체가 깨질 수 있다.** 계정이 깨지면 원격으로는 복구가 안 되고
현장 방문이 필요해진다 — RTP 값 하나 바꾸자고 감수할 위험이 아니다.

반면 카메라 자신의 체크박스 + 저장 버튼(5 절 절차)은 카메라가 원래 의도한
저장 흐름 그대로를 타므로 이 위험이 없다. 느려 보여도 이것이 유일하게
안전한 경로다.

## 7. 건드리면 안 되는 것

- **사용자 변경(비밀번호) 다이얼로그.** 이 절차 도중에 뜨더라도 입력하지도,
  제출하지도 마라. 목적과 무관한 값을 잘못 채우면 그 자체로 계정 상태를
  바꾼다.
- **SSL / `SslState`.** 절대 바꾸지 마라. 지금 브라우저가 카메라에 붙어 있는
  경로 자체가 이 설정에 의존한다. 여기를 건드리면 접속 경로가 끊기고,
  그러면 REG readback 도, 재시도도 못 하게 된다.
- **비밀번호 재시도.** IDIS 계정은 오인증이 쌓이면 잠긴다. 401 이 보이면
  다른 값을 연달아 넣지 말고, 먼저 자격증명 노트의 기록이 맞는지부터
  의심해라. (`SKILL.md` 2 단계 참고.)

## 8. 저장 후 검증은 브라우저 밖에서 두 번

페이지 안에서 본 `SAVED` 상태만 믿고 "고쳤다"고 보고하지 마라. 실제로 하위
에이전트가 이 절차를 실행하고 "고쳤다"고 보고했는데, 막상 readback 해 보니
전 채널이 그대로 `1` 이었던 전례가 있다. 브라우저 컨텍스트가 실제 저장을
반영하지 않고도 화면상으로는 성공한 것처럼 보일 수 있다는 뜻이다. 그래서
검증은 페이지 밖에서, 서로 다른 두 지점에서 한다.

1. **레지스트리 readback 이 `0` 인지** — 페이지를 새로고침한 뒤 다시
   `REG.NetworkConfig.Rtsp.RtpEncryptionType.$` 을 읽어서 브라우저 세션의
   기억이 아니라 서버가 실제로 들고 있는 값을 확인한다.
2. **엣지에서 `scripts/rtsp_sweep.sh` 가 실제 스트림을 잡는지** — 이게
   최종 증거다. RTSP `DESCRIBE` 가 454 없이 통과하고 실제 코덱/해상도가
   찍혀야 끝난 것이다.

## 9. 브라우저 자동화 운영 주의

탭 하나를 여러 액터(에이전트, 사람)가 동시에 조작하면 서로의 페이지 상태를
깨뜨린다. `movePage` 로 이동한 화면을 다른 액터가 바꾸거나, 한쪽이 다이얼로그를
띄운 채로 다른 쪽이 5 절 스크립트를 실행하면 사전 상태 검사가 틀어져
`ABORT` 로 빠지거나, 최악의 경우 검사를 통과했는데 실제로는 엉뚱한 카메라
화면을 저장하게 된다. 한 번에 한 액터만 그 탭을 만지게 조율해라.

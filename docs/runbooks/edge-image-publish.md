# 엣지 이미지 발행과 digest 확인

파일럿 런칭 당일 절차의 일부다. 제품 저장소(`eldercare-fall-ai`)의
`docs/rules/pilot-launch-runbook.md` 2.5단계가 이 문서를 가리킨다.

**이 저장소가 GHCR 네임스페이스와 워커 이미지를 소유하므로 구체적인
이미지 참조는 여기에만 둔다.** 제품 저장소는 ML 이미지 네임스페이스를
문자열로 갖지 않는다(`eldercare-fall-ai/scripts/repo-residue-check.mjs`가
검사한다).

## `main` 병합 뒤에 생기는 것

`.env.edge.prod`의 `ML_API_IMAGE` / `ML_WORKER_IMAGE`는 GHCR에 **이미 올라가
있는** 이미지를 가리킨다. 그 이미지를 만드는 워크플로
(`.github/workflows/edge-images.yml`)는 `release: published`, `main` push,
`workflow_dispatch`에서 실행된다.

즉 PR이 `main`에 병합되면 워크플로가 자동으로 `ml-api`와 `ml-worker`를
빌드하고 GHCR에 올린다. `ml-api` 이미지에는 프런트엔드도 함께 들어간다.
배포에 쓸 값은 태그가 아니라 그 실행이 남긴 digest-pinned artifact에서
가져온다.

클라우드(`eldercare-fall-ai`)는 사정이 다르다. 릴리스 발행이 Jenkins를
깨우고 Jenkins가 `Jenkinsfile`에서 backend/front 이미지를 직접 빌드한다.
`workflow_dispatch`는 특정 tag, branch, SHA를 다시 빌드해야 할 때만 쓴다.

## 절차

병합 뒤에는 해당 `main` push 실행이 성공할 때까지 기다린다. 아래 명령은
현재 `main`의 SHA로 실행을 고르므로, 뒤이어 다른 병합이 있어도 엉뚱한
artifact를 받지 않는다.

```bash
set -eu
REPO=SeniorAILab/eldercare-fall-ml-v2
MAIN_SHA=$(gh api "repos/$REPO/commits/main" --jq .sha)
RUN=$(gh run list --repo "$REPO" --workflow edge-images.yml \
      --commit "$MAIN_SHA" --limit 1 --json databaseId --jq '.[0].databaseId')
test -n "$RUN"
echo "run=$RUN"
gh run watch "$RUN" --repo "$REPO"
```

GitHub가 실행을 목록에 반영하기 전에는 `RUN`이 비어 있을 수 있다. 그때는
몇 초 기다린 뒤 같은 조회 명령부터 다시 실행한다. 실행이 실패하면 artifact를
사용하지 말고 실행 로그의 실패 단계를 해결한다.

**값을 손으로 만들지 않는다.** 워크플로 마지막 단계
(`Write edge image env artifact`)가 `ML_API_IMAGE` / `ML_WORKER_IMAGE`
두 줄을 **digest로 고정해** 그대로 찍어 준다. 실행 요약 화면에 `dotenv`
블록으로 보이므로 복사해서 `.env.edge.prod`에 붙인다.

```bash
# 실행 요약(웹)에서 바로 복사하거나, 현재 main SHA에 정확히 대응하는
# 이름의 artifact 하나만 받는다.
set -eu
REPO=SeniorAILab/eldercare-fall-ml-v2
MAIN_SHA=$(gh api "repos/$REPO/commits/main" --jq .sha)
RUN=${RUN:-$(gh run list --repo "$REPO" --workflow edge-images.yml \
       --commit "$MAIN_SHA" --limit 1 --json databaseId --jq '.[0].databaseId')}
test -n "$RUN"

# artifact name is derived by edge-images.yml from the resolved deploy SHA.
ARTIFACT="edge-ml-image-refs-$MAIN_SHA"
rm -rf /tmp/edge-refs
gh run download "$RUN" --repo "$REPO" --name "$ARTIFACT" --dir /tmp/edge-refs
cat /tmp/edge-refs/edge-ml-image-refs.env
```

받은 두 줄은 **태그가 아니라 `@sha256:` digest**다.

> **digest로 붙이는 편이 낫다.** 태그는 나중에 같은 이름으로 다른 이미지를
> 가리킬 수 있지만 digest는 고정이다. 엣지가 어떤 바이너리를 돌렸는지
> 나중에 다투지 않아도 된다. `.env.edge.prod.example`도 digest 형태를
> 예시로 든다.
>
> `compose.edge.yaml`이 digest 형식을 그대로 받는 것을 확인했다 —
> 가짜 digest로 `docker compose config`를 돌려 해석되는 것을 봤다.

## GHCR 권한 오류

푸시 대상 네임스페이스는 GitHub상 **구 저장소**(`eldercare-fall-ml`)에
연결돼 있는데, 워크플로는 이 저장소(`eldercare-fall-ml-v2`)에서 돈다.
GHCR은 다른 저장소에 연결된 패키지로의 푸시를 기본적으로 막으므로
`denied` 계열 오류가 날 수 있다.

막히면 둘 중 하나다.

- 패키지 설정에서 `eldercare-fall-ml-v2`에 write 권한을 준다
  (GitHub → Packages → 해당 패키지 → Manage Actions access → 저장소 추가)
- 또는 `edge-images.yml`의 `IMAGE_NAMESPACE`를 이 저장소가 소유하는
  이름으로 바꿔 다시 돌린다

**막혔다고 예전 이미지로 그냥 진행하지 않는다.** GHCR의 기존 이미지에는
`_normalize_api_base`(heartbeat URL의 `/api` prefix 보정)가 없다. 그대로
띄우면 엣지가 `{base}/v1/events/...`로 쏘고 클라우드는 모든 라우트를
`/api` 아래 두므로 404가 나는데, 엣지는 그것을 조용한 실패로 넘겨
**카메라가 계속 online으로 보인다.** 증상만 보고는 원인을 못 찾는다.

## pull 권한

엣지 기기에서 한 번도 pull한 적이 없으면 `read:packages` 권한 토큰으로
로그인한다(`.env.edge.prod.example`이 같은 안내를 한다).

```bash
docker login ghcr.io
```

기동 시 `manifest unknown`이나 `denied`가 나면 십중팔구 이 로그인 또는
위 이미지 참조 문제다.

## 진행 조건

워크플로 성공, 그리고 `.env.edge.prod`의 두 줄이 그 실행이 찍어 준
digest와 **글자 그대로 같음**. 옮겨 적지 말고 복사한다. 배포 순서나
one-off 데이터 작업이 필요한 변경은
[`event-thumbnail-rollout.md`](event-thumbnail-rollout.md)를 따른다.

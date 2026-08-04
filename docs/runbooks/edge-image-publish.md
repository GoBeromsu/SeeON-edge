# 엣지 이미지 발행 — `main` 병합만으로는 이미지가 생기지 않는다

파일럿 런칭 당일 절차의 일부다. 제품 저장소(`eldercare-fall-ai`)의
`docs/rules/pilot-launch-runbook.md` 2.5단계가 이 문서를 가리킨다.

**이 저장소가 GHCR 네임스페이스와 워커 이미지를 소유하므로 구체적인
이미지 참조는 여기에만 둔다.** 제품 저장소는 ML 이미지 네임스페이스를
문자열로 갖지 않는다(`eldercare-fall-ai/scripts/repo-residue-check.mjs`가
검사한다).

## 왜 별도 단계인가

`.env.edge.prod`의 `ML_API_IMAGE` / `ML_WORKER_IMAGE`는 GHCR에 **이미 올라가
있는** 이미지를 가리킨다. 그 이미지를 만드는 워크플로
(`.github/workflows/edge-images.yml`)는 **`main` push로 돌지 않는다.**
트리거는 `release: published`와 `workflow_dispatch` 둘뿐이다.

즉 PR을 병합해도 이미지는 그대로다. 새 코드가 담긴 이미지를 원하면
사람이 한 번 돌려야 한다.

클라우드(`eldercare-fall-ai`)는 사정이 다르다. 릴리스 발행이 Jenkins를
깨우고 Jenkins가 `Jenkinsfile`에서 backend/front 이미지를 직접 빌드한다.
**사람이 이미지 발행을 따로 해야 하는 것은 엣지뿐이다.**

## 절차

병합 직후 `main`으로 굽는다.

```bash
gh workflow run edge-images.yml --repo SeniorAILab/eldercare-fall-ml-v2 -f ref=main

# 실행 id를 얻어서 지켜본다. `gh run watch`는 인자 없이 쓰면 대화형으로
# 목록을 띄우므로 id를 넘긴다.
sleep 5   # 목록에 뜨기까지 잠깐 걸린다
RUN=$(gh run list --repo SeniorAILab/eldercare-fall-ml-v2 \
      --workflow edge-images.yml --limit 1 --json databaseId -q '.[0].databaseId')
echo "run=$RUN"
gh run watch "$RUN" --repo SeniorAILab/eldercare-fall-ml-v2
```

> 이 워크플로는 아직 한 번도 돈 적이 없어 `gh run list`가 **빈 목록**을
> 낸다(확인함). 위 `RUN`이 비어 있으면 워크플로가 아직 시작되지 않은
> 것이므로 몇 초 뒤 다시 조회한다.

**값을 손으로 만들지 않는다.** 워크플로 마지막 단계
(`Write edge image env artifact`)가 `ML_API_IMAGE` / `ML_WORKER_IMAGE`
두 줄을 **digest로 고정해** 그대로 찍어 준다. 실행 요약 화면에 `dotenv`
블록으로 보이므로 복사해서 `.env.edge.prod`에 붙인다.

```bash
# 실행 요약(웹)에서 바로 복사하거나, 아티팩트로 받는다.
# 아티팩트 이름에 SHA가 붙으므로 이름을 지정하지 말고 run id로 받는다 —
# 이 워크플로는 아티팩트를 하나만 올린다.
gh run download "$RUN" --repo SeniorAILab/eldercare-fall-ml-v2 --dir ./edge-refs
cat ./edge-refs/*/edge-ml-image-refs.env 2>/dev/null || \
  cat ./edge-refs/edge-ml-image-refs.env
```

받은 두 줄은 **태그가 아니라 `@sha256:` digest**다.

> **digest로 붙이는 편이 낫다.** 태그는 나중에 같은 이름으로 다른 이미지를
> 가리킬 수 있지만 digest는 고정이다. 엣지가 어떤 바이너리를 돌렸는지
> 나중에 다투지 않아도 된다. `.env.edge.prod.example`도 digest 형태를
> 예시로 든다.
>
> `compose.edge.yaml`이 digest 형식을 그대로 받는 것을 확인했다 —
> 가짜 digest로 `docker compose config`를 돌려 해석되는 것을 봤다.

## 첫 실행이 막힐 수 있다

**이 워크플로는 아직 한 번도 돈 적이 없다(`Total runs 0`).** 시간을
넉넉히 잡는다.

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
digest와 **글자 그대로 같음**. 옮겨 적지 말고 복사한다.

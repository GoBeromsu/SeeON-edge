import type { Camera } from '@/shared/api/client';

/** 파싱 못 하는 값은 시각을 지어내지 않고 그대로 모른다고 말한다. */
function formatConnectedAt(value: string): string {
  const date = new Date(value);
  return Number.isFinite(date.getTime())
    ? date.toLocaleString('ko-KR')
    : '시각 불명';
}

type CameraInfoCardProps = {
  camera: Camera;
  onManageConnection: () => void;
};

export function CameraInfoCard({ camera, onManageConnection }: CameraInfoCardProps): JSX.Element {
  return (
    <article className="rounded-card border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="min-w-0 truncate text-base font-semibold text-foreground">{camera.label}</h2>
        <button
          type="button"
          onClick={onManageConnection}
          className="inline-flex h-7 shrink-0 items-center rounded-control border border-border px-2.5 text-xs font-semibold text-foreground"
        >
          연결 관리
        </button>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-sm">
        <dt className="text-muted-foreground">층</dt>
        <dd className="text-foreground">{camera.floor_name ?? '미지정'}</dd>
        <dt className="text-muted-foreground">RTSP 주소</dt>
        <dd className="truncate font-mono text-foreground">{camera.rtsp_url_masked}</dd>
        <dt className="text-muted-foreground">클라우드 연동</dt>
        <dd data-testid="cloud-mapping" className="text-foreground">
          {camera.mapping_pending === false && camera.backend_camera_id ? (
            <span className="font-semibold text-emerald-600">연동 완료</span>
          ) : (
            <span className="font-semibold text-amber-600">
              연동 대기 — 방을 지정해야 클라우드로 전송됩니다
            </span>
          )}
        </dd>
        {/* 한 번도 붙은 적 없는 카메라와 붙었다가 끊긴 카메라는 기사에게
            완전히 다른 상황이다. 둘 다 '오프라인'으로만 보이면 설정 오류를
            일시적 장애로 착각해 그대로 현장을 떠난다. */}
        <dt className="text-muted-foreground">연결 이력</dt>
        <dd data-testid="connection-history" className="text-foreground">
          {camera.never_connected ? (
            <span className="font-semibold text-amber-600">
              한 번도 연결된 적 없음 — 주소와 계정을 확인하세요
            </span>
          ) : camera.last_ok_at ? (
            `마지막 연결 ${formatConnectedAt(camera.last_ok_at)}`
          ) : (
            '연결 기록 없음'
          )}
        </dd>
      </dl>
    </article>
  );
}

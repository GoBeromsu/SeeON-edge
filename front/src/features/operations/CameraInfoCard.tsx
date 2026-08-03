import type { Camera } from '@/shared/api/client';

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
      </dl>
    </article>
  );
}

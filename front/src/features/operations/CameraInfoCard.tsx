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
      </dl>
    </article>
  );
}

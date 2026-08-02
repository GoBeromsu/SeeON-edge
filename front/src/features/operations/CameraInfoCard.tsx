import { navigateToPage } from '@/features/operations/crossPageNavigation';
import type { Camera } from '@/shared/api/client';

type CameraInfoCardProps = {
  camera: Camera;
};

const STATUS_LABELS: Record<Camera['status'], string> = {
  online: '온라인',
  offline: '오프라인',
  starting: '시작 중',
  unknown: '알 수 없음',
};

/**
 * "연결 관리" (RTSP editing, bed-segmentation wizard) is Settings-page territory and isn't being
 * built this wave, so this deliberately navigates to Settings rather than duplicating that flow.
 */
export function CameraInfoCard({ camera }: CameraInfoCardProps): JSX.Element {
  return (
    <article className="rounded-card border border-border bg-card p-4">
      <h2 className="mb-3 text-base font-semibold text-foreground">카메라 정보</h2>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-sm">
        <dt className="text-muted-foreground">층</dt>
        <dd className="text-foreground">{camera.floor_name ?? '미지정'}</dd>
        <dt className="text-muted-foreground">RTSP</dt>
        <dd className="truncate font-mono text-foreground">{camera.rtsp_url_masked}</dd>
        <dt className="text-muted-foreground">상태</dt>
        <dd className="text-foreground">{STATUS_LABELS[camera.status] ?? camera.status}</dd>
      </dl>
      <button
        type="button"
        onClick={() => navigateToPage('settings')}
        className="mt-4 min-h-11 w-full rounded-control border border-border px-3 text-sm font-semibold text-foreground"
      >
        연결 관리
      </button>
    </article>
  );
}

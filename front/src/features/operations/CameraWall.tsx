import { useMemo } from 'react';
import { countCamerasByLiveness, filterCamerasByFloor, isCameraOnline, listFloors } from '@/features/operations/operationsModel';
import { CameraWallTile } from '@/features/operations/CameraWallTile';
import { useSnapshotQueue } from '@/features/operations/useSnapshotQueue';
import { detectionStateOf, DetectionStateIcon } from '@/features/operations/detectionStatus';
import { getPageLabel } from '@/shared/ui/NavBar';
import { statusBadgeClassName } from '@/shared/ui/StatusBadge';
import { useStatusResource } from '@/shared/api/usePollingResource';
import type { Camera } from '@/shared/api/client';
import type { PollingResourceStatus } from '@/shared/api/usePollingResource';

type CameraWallProps = {
  status: PollingResourceStatus;
  cameras: Camera[] | undefined;
  floor: string | undefined;
  onFloorChange: (floor: string | undefined) => void;
  onSelectCamera: (cameraId: string) => void;
  onRetry: () => void;
};

/** Non-breaking space keeps "온라인 3" from wrapping mid-badge (front/design-handoff/README.md §2). */
const NBSP = ' ';

export function CameraWall({ status, cameras, floor, onFloorChange, onSelectCamera, onRetry }: CameraWallProps): JSX.Element {
  const allCameras = cameras ?? [];
  const floors = useMemo(() => listFloors(allCameras), [allCameras]);
  const visible = useMemo(() => filterCamerasByFloor(allCameras, floor), [allCameras, floor]);
  const onlineVisible = useMemo(() => visible.filter(isCameraOnline), [visible]);
  const { online, offline } = useMemo(() => countCamerasByLiveness(allCameras), [allCameras]);
  const { entries, queue } = useSnapshotQueue(onlineVisible, status === 'success');
  // Same cadence as everywhere else (`useStatusResource` owns the interval); the wall subscribes
  // once and reads each camera's diagnostics by its own local id, exactly as RoomDetail does.
  const { data: runtimeStatus } = useStatusResource(true);
  const diagnosticsFor = (cameraId: string) => runtimeStatus?.runtime.cameras[cameraId];
  // 감지 중단은 타일 하나씩 뜻어보면 놓친다. 지금 보이는 카메라 기준으로 한 줄로 알린다 —
  // 토스트가 아니라 페이지에 붙어 있어야 근무 교대 뒤에도 사라지지 않는다.
  const blindVisible = useMemo(
    () => visible.filter((camera) => detectionStateOf(runtimeStatus?.runtime.cameras[camera.id]?.detection) === 'blind'),
    [visible, runtimeStatus],
  );

  return (
    <section aria-label="카메라 월">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
            {getPageLabel('operations')}
          </h1>
          <span className={statusBadgeClassName('approved')}>
            <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
            {`온라인${NBSP}${online}`}
          </span>
          <span className={statusBadgeClassName('rejected')}>
            <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
            {`오프라인${NBSP}${offline}`}
          </span>
        </div>
        <select
          aria-label="층 필터"
          value={floor ?? ''}
          onChange={(event) => onFloorChange(event.target.value || undefined)}
          className="h-9 rounded-control border border-input bg-card px-3 text-sm font-medium text-foreground"
        >
          <option value="">전체</option>
          {floors.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      {blindVisible.length > 0 ? (
        <div
          role="alert"
          data-testid="wall-detection-summary"
          data-blind-count={blindVisible.length}
          className="mb-4 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-card border border-border bg-status-rejectedBg px-3 py-2 text-sm font-semibold text-status-rejectedFg"
        >
          <DetectionStateIcon blind />
          <span className="break-keep">{`감지 중단 ${blindVisible.length}대`}</span>
          <span className="break-keep font-normal">
            {blindVisible.map((camera) => camera.label).join(', ')}
          </span>
          <span className="break-keep font-normal">— 타일을 열어 조치를 확인하세요.</span>
        </div>
      ) : null}

      {status === 'loading' ? (
        <p role="status" className="py-10 text-center text-sm text-muted-foreground">카메라 목록을 불러오는 중입니다…</p>
      ) : null}

      {status === 'error' ? (
        <div role="alert" className="flex flex-col items-center gap-3 py-10 text-center text-sm text-destructive">
          <p>카메라 목록을 불러오지 못했습니다.</p>
          <button type="button" onClick={onRetry} className="brand-action min-h-11 rounded-control px-4">다시 시도</button>
        </div>
      ) : null}

      {status !== 'loading' && status !== 'error' && visible.length === 0 ? (
        <p role="status" className="py-10 text-center text-sm text-muted-foreground">
          {floor ? `${floor}에 등록된 카메라가 없습니다.` : '등록된 카메라가 없습니다.'}
        </p>
      ) : null}

      {visible.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {visible.map((camera) => (
            <CameraWallTile
              key={camera.id}
              camera={camera}
              snapshot={entries.get(camera.id)}
              diagnostics={diagnosticsFor(camera.id)}
              queue={queue}
              onSelect={() => onSelectCamera(camera.id)}
            />
          ))}
        </div>
      ) : null}
    </section>
  );
}

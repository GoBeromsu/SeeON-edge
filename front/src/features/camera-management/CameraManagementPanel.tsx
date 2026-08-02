import type { Camera, CameraRegistry } from '@/shared/api/client';
import { CameraCard } from '@/features/cameras/CameraCard';

// Fault-triage default order: offline first, then unresolved/never-connected states, then online.
function statusPriority(camera: Camera): number {
  if (camera.status === 'offline') return 0;
  if (camera.never_connected || camera.status === 'unknown' || camera.status === 'starting') return 1;
  return 2;
}

function sortCamerasByFaultPriority(cameras: Camera[]): Camera[] {
  return cameras
    .map((camera, index) => ({ camera, index }))
    .sort((a, b) => statusPriority(a.camera) - statusPriority(b.camera) || a.index - b.index)
    .map((entry) => entry.camera);
}

export function CameraManagementPanel({
  registry,
  cameraError,
  lastSuccessAt,
  refreshing,
  onAddCamera,
  onSyncCameras,
  syncBusy = false,
  syncMessage = null,
  onUpdateStarted,
  onUpdated,
  onDelete,
  onViewClips,
  loading = false,
  hasLoadedSuccessfully = true,
  onRetry,
}: {
  registry: CameraRegistry;
  cameraError: string | null;
  lastSuccessAt: number | null;
  refreshing: boolean;
  onAddCamera: () => void;
  onSyncCameras?: () => void;
  syncBusy?: boolean;
  syncMessage?: string | null;
  onUpdateStarted: (camera: Camera) => number;
  onUpdated: (camera: Camera, previousCameraId?: string, startedAtGeneration?: number) => void;
  onDelete: (camera: Camera) => void;
  onViewClips?: (camera: Camera) => void;
  loading?: boolean;
  hasLoadedSuccessfully?: boolean;
  onRetry?: () => void;
}): JSX.Element {
  const sortedCameras = sortCamerasByFaultPriority(registry.cameras);
  return (
    <section className="rounded-2xl border border-border bg-surface p-5 shadow-sm">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-black text-brand">카메라 연결 설정</p>
          <h2 className="text-2xl font-black text-ink">카메라 관리</h2>
        </div>
        <div className="flex flex-wrap gap-2">
          {onSyncCameras ? (
            <button
              type="button"
              onClick={onSyncCameras}
              disabled={syncBusy}
              className="rounded-lg border border-border bg-surface px-5 py-3 text-sm font-black text-ink-soft disabled:opacity-60"
            >
              {syncBusy ? '동기화 중...' : '카메라 동기화'}
            </button>
          ) : null}
          <button
            type="button"
            onClick={onAddCamera}
            className="brand-action rounded-lg px-5 py-3 text-sm font-black"
          >
            + 카메라 추가
          </button>
        </div>
      </div>
      {syncMessage ? <p className="mb-4 text-sm font-bold text-brand" role="status">{syncMessage}</p> : null}
      <p className="mb-4 text-xs font-bold text-ink-faint">등록 버전 {registry.registry_version}</p>
      {cameraError ? (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-status-dangerBg px-4 py-3 text-sm font-bold text-status-danger" role="alert">
          <div>
            <p>{cameraError}</p>
            {lastSuccessAt !== null ? <p className="mt-1 text-xs">마지막 확인 {new Date(lastSuccessAt).toLocaleString('ko-KR')}</p> : null}
          </div>
          {onRetry ? <button type="button" onClick={onRetry} className="rounded-full bg-surface px-4 py-2 text-xs font-black">다시 시도</button> : null}
        </div>
      ) : null}
      {refreshing && !loading && cameraError === null ? (
        <p className="mb-4 text-sm font-bold text-ink-soft" role="status" aria-live="polite">카메라 목록을 새로 확인하고 있습니다.</p>
      ) : null}
      {loading ? (
        <p className="rounded-2xl bg-surface px-5 py-8 text-center font-bold text-ink-soft" role="status">카메라 목록을 불러오는 중...</p>
      ) : sortedCameras.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2">
          {sortedCameras.map((camera) => (
            <CameraCard
              key={camera.id}
              camera={camera}
              onUpdateStarted={onUpdateStarted}
              onUpdated={onUpdated}
              onDelete={onDelete}
              onViewClips={onViewClips}
              onRetried={onRetry}
            />
          ))}
        </div>
      ) : hasLoadedSuccessfully ? (
        <div className="rounded-xl border border-dashed border-brand bg-surface p-8 text-center text-ink-soft">
          등록된 카메라가 없습니다. 첫 카메라를 추가하세요.
        </div>
      ) : null}
    </section>
  );
}

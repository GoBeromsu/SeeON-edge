import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import type { DashboardLocation } from '@/app/dashboardLocation';
import { cameraMediaId, fetchCameraPose, getCameraSnapshotUrl, getCameraStreamUrl, setCameraPose, type Camera, type HeartbeatStatus, type StatusSnapshot } from '@/shared/api/client';
import { SnapshotQueue, type SnapshotEntry } from '@/features/operations/SnapshotQueue';
import { filterAndGroupCameras, getCameraHeartbeatAgeSec, getCameraLiveness, groupCameras, paginateCameras, sortCameras, UNCLASSIFIED } from '@/features/operations/operationsModel';
import { AccountSettingsModal } from '@/features/account-settings/AccountSettingsModal';

type ResourceState = 'idle' | 'loading' | 'success' | 'error';
type LocationUpdate = Partial<Record<keyof DashboardLocation, string | null>>;

export type OperationsViewProps = {
  active: boolean;
  cameras: Camera[];
  camerasState: ResourceState;
  camerasLastSuccessAt?: number | null;
  camerasRefreshing?: boolean;
  status: StatusSnapshot | null;
  statusState: ResourceState;
  location: DashboardLocation;
  onNavigate: (update: LocationUpdate) => void;
  onReplace: (update: LocationUpdate) => void;
  onRetryCameras?: () => void;
  onRetryStatus?: () => void;
};

const livenessLabels: Record<HeartbeatStatus | 'unknown', string> = {
  online: '연결됨',
  stale: '연결 지연',
  never_seen: '연결 이력 없음',
  unknown: '연결 정보 없음',
};

const snapshotLabels: Record<SnapshotEntry['state'], string> = {
  loading: '영상 불러오는 중',
  loaded: '영상 확인됨',
  stale: '이전 영상 · 새로 고치는 중',
  error: '영상을 불러올 수 없음',
};

function useSnapshotQueue(cameras: Camera[], enabled: boolean): { entries: Map<string, SnapshotEntry>; queue: SnapshotQueue } {
  const [queue] = useState(() => new SnapshotQueue(getCameraSnapshotUrl));
  const snapshot = useSyncExternalStore(queue.subscribe, queue.snapshot, queue.snapshot);
  useEffect(() => {
    queue.update(cameras.map((camera) => ({ id: camera.id, mediaId: cameraMediaId(camera) })), enabled);
  }, [cameras, enabled, queue]);
  useEffect(() => () => queue.dispose(), [queue]);
  return { entries: useMemo(() => new Map(snapshot.map((entry) => [entry.id, entry])), [snapshot]), queue };
}

function CameraCard({ camera, liveness, snapshot, queue, onSelect }: { camera: Camera; liveness: HeartbeatStatus | 'unknown'; snapshot?: SnapshotEntry; queue: SnapshotQueue; onSelect: () => void }): JSX.Element {
  const currentSnapshot = snapshot?.mediaId === cameraMediaId(camera) ? snapshot : undefined;
  return (
    <button type="button" aria-label={`${camera.label} 열기`} data-camera-id={camera.id} onClick={onSelect} className="min-h-11 w-full min-w-0 overflow-hidden rounded-xl border border-border bg-surface text-left focus:outline-none focus:ring-2 focus:ring-brand">
      <span className="relative block aspect-video bg-[var(--c-media)] text-[var(--c-media-ink)]">
        {[
          currentSnapshot?.lastLoadedUrl ? <img key={currentSnapshot.lastLoadedUrl} src={currentSnapshot.lastLoadedUrl} alt={`${camera.label} 최근 영상`} className="h-full w-full object-contain" /> : null,
          currentSnapshot?.requestUrl ? <img key={currentSnapshot.requestUrl} data-snapshot-preload src={currentSnapshot.requestUrl} alt={currentSnapshot.lastLoadedUrl ? '' : `${camera.label} 최근 영상`} className={currentSnapshot.lastLoadedUrl ? 'hidden' : 'h-full w-full object-contain'} onLoad={(event) => queue.resolve(camera.id, currentSnapshot.mediaId, 'loaded', event.currentTarget.getAttribute('src') ?? undefined)} onError={(event) => queue.resolve(camera.id, currentSnapshot.mediaId, 'error', event.currentTarget.getAttribute('src') ?? undefined)} /> : null,
        ]}
        <span className="media-status-overlay absolute bottom-2 left-2 rounded px-2 py-1 text-xs">{snapshotLabels[currentSnapshot?.state ?? 'loading']}</span>
      </span>
      <span className="block space-y-1 p-3">
        <span className="block font-bold">{camera.label}</span>
        <span className="block text-sm text-ink-soft">{livenessLabels[liveness]}</span>
        {currentSnapshot?.lastLoadedAt ? <span className="block text-xs tabular-nums text-ink-faint">마지막 영상 {new Intl.DateTimeFormat('ko-KR', { timeStyle: 'short' }).format(currentSnapshot.lastLoadedAt)}</span> : null}
      </span>
    </button>
  );
}

function formatHeartbeatAge(ageSec: number | null): string | null {
  if (ageSec === null) return null;
  const seconds = Math.max(0, Math.round(ageSec));
  if (seconds < 60) return `마지막 확인 ${seconds}초 전`;
  return `마지막 확인 ${Math.round(seconds / 60)}분 전`;
}

function focusedSnapshotLabel(snapshot: SnapshotEntry | undefined): string {
  return snapshot ? snapshotLabels[snapshot.state] : '스냅샷 정보 없음';
}

function usePoseToggle(cameraId: string): { showPose: boolean | null; busy: boolean; toggle: () => void } {
  const [showPose, setShowPose] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    setShowPose(null);
    fetchCameraPose(cameraId)
      .then((value) => {
        if (!cancelled) setShowPose(value);
      })
      .catch(() => {
        if (!cancelled) setShowPose(null);
      });
    return () => {
      cancelled = true;
    };
  }, [cameraId]);

  function toggle(): void {
    if (busyRef.current || showPose === null) return;
    const next = !showPose;
    busyRef.current = true;
    setBusy(true);
    setCameraPose(cameraId, next)
      .then((value) => setShowPose(value))
      .catch(() => setShowPose(showPose))
      .finally(() => {
        busyRef.current = false;
        setBusy(false);
      });
  }

  return { showPose, busy, toggle };
}

function FocusedCamera({ camera, liveness, ageSec, snapshot, onBack, onViewClips, onOpenSettings, onOpenAccountSettings }: {
  camera: Camera;
  liveness: HeartbeatStatus | 'unknown';
  ageSec: number | null;
  snapshot?: SnapshotEntry;
  onBack: () => void;
  onViewClips: () => void;
  onOpenSettings: () => void;
  onOpenAccountSettings: () => void;
}): JSX.Element {
  const streamUrl = getCameraStreamUrl(cameraMediaId(camera));
  const statusStripText = [livenessLabels[liveness], formatHeartbeatAge(ageSec), focusedSnapshotLabel(snapshot)]
    .filter((part): part is string => part !== null)
    .join(' · ');
  const { showPose, busy: poseBusy, toggle: togglePose } = usePoseToggle(cameraMediaId(camera));
  return (
    <section aria-labelledby="focused-camera-title" className="space-y-4">
      <button type="button" className="min-h-11 rounded-lg border border-border px-4 font-bold" onClick={onBack}>← 관제</button>
      <h1 id="focused-camera-title" data-dialog-focus-fallback tabIndex={-1} className="text-xl font-bold">{camera.label} · {camera.floor_name ?? UNCLASSIFIED}</h1>
      <FocusedMedia key={streamUrl} cameraLabel={camera.label} streamUrl={streamUrl} />
      <p role="status" aria-live="polite" aria-atomic="true" className="text-sm text-ink-soft">{statusStripText}</p>
      <div className="flex flex-wrap gap-3">
        <button type="button" className="brand-action min-h-11 rounded-lg px-4 font-bold" onClick={onViewClips}>이 카메라 클립 보기</button>
        <button type="button" className="min-h-11 rounded-lg border border-border px-4 font-bold" onClick={onOpenSettings}>카메라 설정</button>
        <button type="button" className="min-h-11 rounded-lg border border-border px-4 font-bold" onClick={onOpenAccountSettings}>계정 설정</button>
        <button
          type="button"
          aria-pressed={showPose === true}
          disabled={showPose === null || poseBusy}
          className="min-h-11 rounded-lg border border-border px-4 font-bold disabled:opacity-50"
          onClick={togglePose}
        >
          {showPose === true ? '자세 표시 끄기' : '자세 표시 켜기'}
        </button>
      </div>
    </section>
  );
}

function FocusedMedia({ cameraLabel, streamUrl }: { cameraLabel: string; streamUrl: string }): JSX.Element {
  const [state, setState] = useState<'loading' | 'loaded' | 'error'>('loading');
  const label = state === 'loading' ? '라이브 영상 불러오는 중' : state === 'loaded' ? '라이브 영상 연결됨' : '라이브 영상을 불러올 수 없음';
  return (
    <div className="relative aspect-video overflow-hidden rounded-xl bg-[var(--c-media)] text-[var(--c-media-ink)]">
      {state !== 'error' ? <img data-stream src={streamUrl} alt={`${cameraLabel} 라이브 영상`} className="h-full w-full object-contain" onLoad={() => setState('loaded')} onError={() => setState('error')} /> : null}
      <p role="status" aria-live="polite" aria-atomic="true" className={state === 'error' ? 'grid h-full place-items-center' : 'media-status-overlay absolute bottom-3 left-3 rounded px-3 py-2 text-sm'}>{label}</p>
    </div>
  );
}

export function OperationsView({ active, cameras, camerasState, camerasLastSuccessAt = null, camerasRefreshing = false, status, statusState, location, onNavigate, onReplace, onRetryCameras, onRetryStatus }: OperationsViewProps): JSX.Element | null {
  const [announcement, setAnnouncement] = useState('');
  const [accountSettingsOpen, setAccountSettingsOpen] = useState(false);
  const previousSelectedIdRef = useRef<string | null>(null);
  const scrollRestoreRef = useRef<number | null>(null);
  const filtered = useMemo(() => filterAndGroupCameras(cameras, { floor: location.floor, room: location.room }).cameras, [cameras, location.floor, location.room]);
  const requestedPage = Number(location.wallPage ?? '1');
  const pagination = useMemo(() => paginateCameras(filtered, requestedPage), [filtered, requestedPage]);
  const pageCameras = pagination.cameras;
  const selected = cameras.find((camera) => camera.id === location.camera && filtered.some((entry) => entry.id === camera.id));
  const { entries: snapshots, queue } = useSnapshotQueue(pageCameras, active && location.page === 'operations' && !selected);

  useEffect(() => {
    if (camerasState !== 'success') return;
    const update: LocationUpdate = {};
    if (location.camera && !selected) {
      update.camera = null;
      setAnnouncement('선택한 카메라가 현재 목록에 없어 선택을 해제했습니다.');
    }
    if (pagination.page !== requestedPage) update.wallPage = String(pagination.page);
    if (Object.keys(update).length) onReplace(update);
  }, [camerasState, location.camera, onReplace, pagination.page, requestedPage, selected]);

  // Screen-transition focus management (AC-11/AC-12) and wall scroll restore (AC-9),
  // kept local to this component -- no global state, cleared once consumed.
  useEffect(() => {
    const previousSelectedId = previousSelectedIdRef.current;
    if (selected && selected.id !== previousSelectedId) {
      document.getElementById('focused-camera-title')?.focus();
    } else if (!selected && previousSelectedId) {
      const invokerCard = Array.from(document.querySelectorAll<HTMLElement>('[data-camera-id]'))
        .find((element) => element.dataset.cameraId === previousSelectedId);
      (invokerCard ?? document.getElementById('operations-title'))?.focus();
      if (scrollRestoreRef.current !== null) {
        window.scrollTo(0, scrollRestoreRef.current);
        scrollRestoreRef.current = null;
      }
    }
    previousSelectedIdRef.current = selected ? selected.id : null;
  }, [selected]);

  if (!active || location.page !== 'operations') return null;

  const openCameraManagement = (): void => onNavigate({
    page: 'cameras', floor: null, room: null, camera: null,
    event: null, clip: null, wallPage: null,
  });

  if (selected) {
    const selectedSnapshot = snapshots.get(selected.id);
    const focusedSnapshot = selectedSnapshot?.mediaId === cameraMediaId(selected) ? selectedSnapshot : undefined;
    return (
      <>
        <FocusedCamera
          key={selected.id}
          camera={selected}
          liveness={getCameraLiveness(selected, cameras, status)}
          ageSec={getCameraHeartbeatAgeSec(selected, cameras, status)}
          snapshot={focusedSnapshot}
          onBack={() => onNavigate({ camera: null })}
          onViewClips={() => onNavigate({
            page: 'events', floor: null, room: null, camera: selected.id,
            event: null, clip: null, wallPage: null,
          })}
          onOpenSettings={openCameraManagement}
          onOpenAccountSettings={() => setAccountSettingsOpen(true)}
        />
        <AccountSettingsModal open={accountSettingsOpen} onClose={() => setAccountSettingsOpen(false)} />
      </>
    );
  }

  const sortedCameras = sortCameras(cameras);
  const floors = [...new Set(sortedCameras.map((camera) => camera.floor_name).filter((value): value is string => value !== null))];
  const rooms = [...new Map(sortedCameras.filter((camera) => !location.floor || camera.floor_name === location.floor).filter((camera) => camera.space_id).map((camera) => [camera.space_id!, camera.space_name ?? camera.space_id!])).entries()];
  const pageGroups = groupCameras(pageCameras);
  const updateFilter = (update: LocationUpdate): void => {
    onNavigate({ ...update, camera: null, wallPage: '1' });
    if (location.camera) setAnnouncement('필터가 변경되어 카메라 선택을 해제했습니다.');
  };
  const addCameraAction = <button type="button" className="brand-action min-h-11 rounded-lg px-4 font-bold" onClick={openCameraManagement}>카메라 등록하기</button>;
  const lastCameraSuccessLabel = camerasLastSuccessAt === null ? null : new Date(camerasLastSuccessAt).toLocaleString('ko-KR');

  return (
    <section aria-labelledby="operations-title" className="space-y-5">
      <div>
        <h1 id="operations-title" tabIndex={-1} className="text-xl font-bold">관제</h1>
        <p className="text-sm text-ink-soft">카메라 연결 상태와 최근 영상을 함께 확인합니다.</p>
      </div>
      <div aria-live="polite" className="sr-only">{announcement}</div>
      <div className="flex flex-wrap gap-3" aria-label="카메라 필터">
        <label className="grid gap-1 text-sm font-bold">층
          <select value={location.floor ?? ''} onChange={(event) => updateFilter({ floor: event.target.value || null, room: null })} className="min-h-11 rounded-lg border border-border bg-surface px-3">
            <option value="">전체 층</option>{floors.map((floor) => <option key={floor} value={floor}>{floor}</option>)}
          </select>
        </label>
        <label className="grid gap-1 text-sm font-bold">공간
          <select value={location.room ?? ''} onChange={(event) => updateFilter({ room: event.target.value || null })} className="min-h-11 rounded-lg border border-border bg-surface px-3">
            <option value="">전체 공간</option>{rooms.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          </select>
        </label>
      </div>

      {camerasState === 'loading' || camerasState === 'idle' ? <p>카메라 불러오는 중</p> : null}
      {camerasState === 'success' && camerasRefreshing && lastCameraSuccessLabel !== null ? <p data-camera-refresh-status role="status" aria-live="polite" aria-atomic="true">카메라 목록을 새로 확인하고 있습니다.</p> : null}
      {camerasState === 'error' && lastCameraSuccessLabel === null ? <div role="alert"><p>카메라 목록을 불러올 수 없음</p><button type="button" onClick={onRetryCameras}>다시 시도</button></div> : null}
      {camerasState === 'error' && lastCameraSuccessLabel !== null ? <div role="status" className="space-y-3"><p>{cameras.length === 0 ? '마지막 확인 당시 등록된 카메라가 없습니다.' : '마지막 카메라 목록을 표시합니다.'}</p><p>마지막 확인 {lastCameraSuccessLabel}</p><button type="button" onClick={onRetryCameras}>다시 시도</button>{cameras.length === 0 ? addCameraAction : null}</div> : null}
      {statusState === 'error' ? <p role="status">연결 상태를 새로 확인하지 못했습니다. <button type="button" onClick={onRetryStatus}>다시 시도</button></p> : null}
      {camerasState === 'success' && cameras.length === 0 ? <div className="space-y-3"><p>등록된 카메라가 없습니다.</p>{addCameraAction}</div> : null}
      {camerasState === 'success' && cameras.length > 0 && filtered.length === 0 ? <p>선택한 조건에 맞는 카메라가 없습니다.</p> : null}

      <div className="space-y-6">
        {pageGroups.map((floor) => <section key={floor.id} aria-labelledby={`floor-${floor.id}`} className="space-y-4">
          <h2 id={`floor-${floor.id}`} className="text-lg font-bold">{floor.label}</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {floor.rooms.flatMap((room) => room.cameras.map((camera) => <article key={camera.id} className="space-y-3">
              <h3 className="font-bold text-ink-soft">{room.label}</h3>
              <CameraCard camera={camera} liveness={getCameraLiveness(camera, cameras, status)} snapshot={snapshots.get(camera.id)} queue={queue} onSelect={() => { scrollRestoreRef.current = window.scrollY; onNavigate({ camera: camera.id }); }} />
            </article>))}
          </div>
        </section>)}
        {pagination.pageCount > 1 ? <nav aria-label="카메라 월 페이지" className="flex items-center justify-center gap-3">
          <button type="button" className="min-h-11" disabled={pagination.page === 1} onClick={() => onNavigate({ wallPage: String(pagination.page - 1) })}>이전</button>
          <span>{pagination.page} / {pagination.pageCount}</span>
          <button type="button" className="min-h-11" disabled={pagination.page === pagination.pageCount} onClick={() => onNavigate({ wallPage: String(pagination.page + 1) })}>다음</button>
        </nav> : null}
      </div>
    </section>
  );
}

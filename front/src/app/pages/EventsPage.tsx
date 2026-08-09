import { useCallback, useEffect, useMemo, useState } from 'react';
import { getPageLabel } from '@/shared/ui/NavBar';
import { useCamerasResource } from '@/shared/api/usePollingResource';
import { CameraFilterSelect } from '@/features/events/CameraFilterSelect';
import { ClipGrid } from '@/features/events/ClipGrid';
import { ClipPlaybackModal } from '@/features/events/ClipPlaybackModal';
import { EventTypeFilterChips } from '@/features/events/EventTypeFilterChips';
import { EventsPager } from '@/features/events/EventsPager';
import { toEventFacet } from '@/shared/api/clipEventFacet';
import { resolveCameraLabel } from '@/features/events/resolveCameraLabel';
import { useClipMetadata } from '@/features/events/useClipMetadata';
import { useEventsPage } from '@/features/events/useEventsPage';
import { useEventsLocation } from '@/features/events/useEventsLocation';
import type { Clip } from '@/shared/api/types';

const EMPTY_CLIPS: readonly Clip[] = [];

export function EventsPage(): JSX.Element {
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const camerasResource = useCamerasResource(true);
  const location = useEventsLocation();
  const clipsResource = useEventsPage({
    ...(selectedCameraId ? { cameraId: selectedCameraId } : {}),
    ...(location.eventType ? { eventType: location.eventType } : {}),
  });
  const clips = clipsResource.data?.clips ?? EMPTY_CLIPS;
  const typeCounts = useMemo(
    () => new Map(Object.entries(clipsResource.data?.event_type_counts ?? {})),
    [clipsResource.data?.event_type_counts],
  );
  const totalCount = useMemo(
    () => [...typeCounts.values()].reduce((total, count) => total + count, 0),
    [typeCounts],
  );
  const metadata = useClipMetadata({
    clipId: location.clipId,
    pageClips: clips,
    completeClips: clipsResource.data?.complete_clips ?? null,
    pageReady: clipsResource.data !== null,
  });
  const cameras = camerasResource.data?.cameras ?? [];
  const metadataMatchesFilters = metadata.clip !== null
    && (!selectedCameraId || metadata.clip.camera_id === selectedCameraId)
    && (!location.eventType || toEventFacet(metadata.clip.event_type) === location.eventType);
  const activeClip = location.clipId
    ? clips.find((clip) => clip.id === location.clipId) ?? (metadataMatchesFilters ? metadata.clip : null)
    : null;

  useEffect(() => {
    location.validate(clipsResource.status, clips, [...typeCounts.keys()]);
  }, [clips, clipsResource.status, location.validate, typeCounts]);

  useEffect(() => {
    if (metadata.status === 'invalid'
      || (metadata.status === 'success' && metadata.clip !== null && !metadataMatchesFilters)) {
      location.discardClip();
    }
  }, [location.discardClip, metadata.clip, metadata.status, metadataMatchesFilters]);

  const resolveLabel = useCallback((clip: Clip) => resolveCameraLabel(cameras, clip), [cameras]);
  const lastSuccessDate = clipsResource.lastSuccessAt === null ? null : new Date(clipsResource.lastSuccessAt);
  const handleCameraChange = useCallback((cameraId: string): void => {
    setSelectedCameraId(cameraId);
    if (cameraId && activeClip && activeClip.camera_id !== cameraId) location.discardClip();
  }, [activeClip, location.discardClip]);

  return (
    <section>
      <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
        {getPageLabel('events')}
      </h1>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <EventTypeFilterChips totalCount={totalCount} counts={typeCounts} selected={location.eventType} onSelect={location.setEventType} />
        <CameraFilterSelect cameras={cameras} value={selectedCameraId} onChange={handleCameraChange} className="ml-auto" />
      </div>

      <div className="mt-5">
        {clipsResource.data !== null ? (
          <div className="mb-3 flex flex-wrap items-center gap-3 text-sm text-muted-foreground" role="status">
            {clipsResource.refreshing ? (
              <span>이벤트 목록을 새로 고치는 중입니다.</span>
            ) : clipsResource.status === 'error' ? (
              <span className="text-status-pendingFg">이벤트 목록을 새로 고치지 못했습니다. 현재 페이지를 유지합니다.</span>
            ) : null}
            {lastSuccessDate ? (
              <span>
                마지막 확인{' '}
                <time data-testid="events-last-success" dateTime={lastSuccessDate.toISOString()}>
                  {lastSuccessDate.toLocaleTimeString('ko-KR')}
                </time>
              </span>
            ) : null}
            {clipsResource.status === 'error' && !clipsResource.refreshing ? (
              <button
                type="button"
                className="inline-flex min-h-11 items-center justify-center rounded-control border border-border bg-card px-4 font-semibold text-foreground hover:bg-muted"
                onClick={clipsResource.refresh}
              >
                다시 시도
              </button>
            ) : null}
          </div>
        ) : null}
        <ClipGrid
          status={clipsResource.status}
          hasData={clipsResource.data !== null}
          clips={clips}
          resolveCameraLabel={resolveLabel}
          onRetry={clipsResource.refresh}
          onSelect={location.openClip}
        />
        {clipsResource.data ? (
          <EventsPager
            pageIndex={clipsResource.pageIndex}
            total={clipsResource.data.pagination.total}
            visibleCount={clips.length}
            pendingPageIndex={clipsResource.pendingPageIndex}
            onNavigate={clipsResource.navigate}
            onRefresh={clipsResource.refresh}
          />
        ) : null}
      </div>

      <ClipPlaybackModal
        clip={activeClip}
        cameraLabel={activeClip ? resolveLabel(activeClip) : ''}
        open={location.clipId !== undefined}
        onClose={location.closeClip}
        lookupStatus={metadata.status}
        onRetry={metadata.retry}
      />
    </section>
  );
}

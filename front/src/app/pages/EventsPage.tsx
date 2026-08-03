import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getPageLabel } from '@/shared/ui/NavBar';
import { useCamerasResource, useClipsResource } from '@/shared/api/usePollingResource';
import { CameraFilterSelect } from '@/features/events/CameraFilterSelect';
import { ClipGrid } from '@/features/events/ClipGrid';
import { ClipPlaybackModal } from '@/features/events/ClipPlaybackModal';
import { EventTypeFilterChips } from '@/features/events/EventTypeFilterChips';
import { resolveCameraLabel } from '@/features/events/resolveCameraLabel';
import { useEventsLocation } from '@/features/events/useEventsLocation';
import type { Clip } from '@/shared/api/types';

export function EventsPage(): JSX.Element {
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const camerasResource = useCamerasResource(true);
  const clipsResource = useClipsResource(true, selectedCameraId || undefined);
  const { eventType, clipId, setEventType, openClip, closeClip } = useEventsLocation(clipsResource.status, clipsResource.data);

  // useClipsResource's polling loop doesn't restart on a cameraId change alone (see
  // usePollingResource: it only reruns its effect on `enabled`), so the camera filter needs an
  // explicit refetch or it would keep showing the previous camera's clips for up to 8s.
  const mountedRef = useRef(false);
  useEffect(() => {
    if (!mountedRef.current) {
      mountedRef.current = true;
      return;
    }
    clipsResource.retry();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCameraId]);

  const clips = clipsResource.data ?? [];
  const hasData = clipsResource.data !== null;

  const typeCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const clip of clips) counts.set(clip.event_type, (counts.get(clip.event_type) ?? 0) + 1);
    return counts;
  }, [clips]);

  const filteredClips = useMemo(
    () => (eventType === undefined ? clips : clips.filter((clip) => clip.event_type === eventType)),
    [clips, eventType],
  );

  const cameras = camerasResource.data?.cameras ?? [];
  const activeClip = clipId ? clips.find((clip) => clip.id === clipId) ?? null : null;

  const resolveLabel = useCallback((clip: Clip) => resolveCameraLabel(cameras, clip), [cameras]);

  return (
    <section>
      <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
        {getPageLabel('events')}
      </h1>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <EventTypeFilterChips totalCount={clips.length} counts={typeCounts} selected={eventType} onSelect={setEventType} />
        <CameraFilterSelect cameras={cameras} value={selectedCameraId} onChange={setSelectedCameraId} className="ml-auto" />
      </div>

      <div className="mt-5">
        <ClipGrid
          status={clipsResource.status}
          hasData={hasData}
          clips={filteredClips}
          resolveCameraLabel={resolveLabel}
          onRetry={clipsResource.retry}
          onSelect={openClip}
        />
      </div>

      <ClipPlaybackModal
        clip={activeClip}
        cameraLabel={activeClip ? resolveLabel(activeClip) : ''}
        open={clipId !== undefined}
        onClose={closeClip}
      />
    </section>
  );
}

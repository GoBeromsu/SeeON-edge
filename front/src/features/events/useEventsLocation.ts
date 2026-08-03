import { useCallback, useEffect, useState } from 'react';
import {
  canonicalizeDashboardLocation,
  serializeDashboardLocation,
  type DashboardLocation,
  type DashboardLocationData,
} from '@/app/dashboardLocation';
import type { PollingResourceStatus } from '@/shared/api/usePollingResource';
import type { Clip } from '@/shared/api/types';

export type EventsLocationState = {
  eventType: string | undefined;
  clipId: string | undefined;
  setEventType: (eventType: string | undefined) => void;
  openClip: (clipId: string) => void;
  closeClip: () => void;
};

function validationData(status: PollingResourceStatus, data: readonly Clip[] | null): DashboardLocationData | undefined {
  if (status !== 'success') return undefined;
  return {
    clips: {
      status: 'success',
      data: (data ?? []).map((clip) => ({ id: clip.id, event_type: clip.event_type })),
    },
  };
}

function currentSearch(): string {
  return window.location.search;
}

function writeSearch(search: string, mode: 'replace' | 'push'): void {
  const url = `${window.location.pathname}${search}${window.location.hash}`;
  if (mode === 'replace') window.history.replaceState(null, '', url);
  else window.history.pushState(null, '', url);
}

/**
 * URL-driven `event` (type filter) and `clip` (playback modal) state for the events page, scoped to
 * this feature only. `camera` intentionally is NOT persisted here: front/src/app/dashboardLocation.ts
 * only recognizes `event`/`clip` on the events page (it strips `camera` there — see
 * dashboardLocation.test.ts "strips floor/camera when the page is events") and that file is owned by
 * another lane this wave, so the camera filter is local component state instead (see EventsPage).
 */
export function useEventsLocation(clipsStatus: PollingResourceStatus, clipsData: readonly Clip[] | null): EventsLocationState {
  const [location, setLocation] = useState<DashboardLocation>(
    () => canonicalizeDashboardLocation(currentSearch()).location,
  );

  useEffect(() => {
    const result = canonicalizeDashboardLocation(currentSearch(), validationData(clipsStatus, clipsData));
    if (result.search !== currentSearch()) writeSearch(result.search, 'replace');
    setLocation(result.location);
  }, [clipsStatus, clipsData]);

  useEffect(() => {
    function onPopState(): void {
      setLocation(canonicalizeDashboardLocation(currentSearch(), validationData(clipsStatus, clipsData)).location);
    }
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, [clipsStatus, clipsData]);

  const navigate = useCallback((update: Partial<Record<'event' | 'clip', string | null>>): void => {
    const data = validationData(clipsStatus, clipsData);
    const current = canonicalizeDashboardLocation(currentSearch(), data).location;
    const next: Record<string, string | undefined> = { ...current };
    for (const [key, value] of Object.entries(update)) {
      if (value === null || value === undefined) delete next[key];
      else next[key] = value;
    }
    const result = canonicalizeDashboardLocation(serializeDashboardLocation(next as DashboardLocation), data);
    if (result.search !== currentSearch()) writeSearch(result.search, 'push');
    setLocation(result.location);
  }, [clipsStatus, clipsData]);

  return {
    eventType: location.event,
    clipId: location.clip,
    setEventType: (eventType) => navigate({ event: eventType ?? null }),
    openClip: (clipId) => navigate({ clip: clipId }),
    closeClip: () => navigate({ clip: null }),
  };
}

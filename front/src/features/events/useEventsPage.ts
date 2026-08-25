import { useCallback, useEffect, useRef, useState } from 'react';
import { EVENTS_PAGE_SIZE, EVENTS_POLL_INTERVAL_MS } from '@/features/events/paging';
import { fetchClipPage } from '@/shared/api/clipPagination';
import { HttpError } from '@/shared/api/http';
import type { ClipPage, ClipPageQuery } from '@/shared/api/clipPaginationTypes';
import type { PollingResourceStatus } from '@/shared/api/usePollingResource';

export type EventsPageFilters = {
  readonly cameraId?: string;
  readonly eventType?: string;
};

/** Cursor trail: index 0 is always the newest page (no cursor); each later entry is that page's cursor. */
type CursorTrail = readonly (string | null)[];

const FIRST_TRAIL: CursorTrail = [null];

type PageView = {
  readonly filterKey: string;
  readonly trail: CursorTrail;
  readonly page: ClipPage;
};

type PageState = {
  readonly status: PollingResourceStatus;
  readonly view: PageView | null;
  readonly error: unknown | null;
  readonly lastSuccessAt: number | null;
  readonly refreshing: boolean;
  readonly pendingPageIndex: number | null;
};

export type EventsPageResource = {
  readonly status: PollingResourceStatus;
  readonly data: ClipPage | null;
  readonly error: unknown | null;
  readonly lastSuccessAt: number | null;
  readonly refreshing: boolean;
  readonly pageIndex: number;
  readonly pendingPageIndex: number | null;
  readonly hasNextPage: boolean;
  readonly openNextPage: () => void;
  readonly openPreviousPage: () => void;
  readonly openFirstPage: () => void;
  readonly refresh: () => void;
};

function filterKey(filters: EventsPageFilters): string {
  return JSON.stringify([filters.cameraId ?? '', filters.eventType ?? '']);
}

function pageQuery(filters: EventsPageFilters, cursor: string | null): ClipPageQuery {
  return {
    ...(filters.cameraId ? { cameraId: filters.cameraId } : {}),
    ...(filters.eventType ? { eventType: filters.eventType } : {}),
    limit: EVENTS_PAGE_SIZE,
    ...(cursor ? { cursor } : {}),
  };
}

function isRetiredCursor(error: unknown): boolean {
  return error instanceof HttpError && (error.status === 400 || error.status === 404);
}

export function useEventsPage(filters: EventsPageFilters): EventsPageResource {
  const activeKey = filterKey(filters);
  const filtersRef = useRef(filters);
  filtersRef.current = filters;
  const activeKeyRef = useRef(activeKey);
  activeKeyRef.current = activeKey;
  const viewRef = useRef<PageView | null>(null);
  const requestRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<number | null>(null);
  const aliveRef = useRef(false);
  const loadRef = useRef<(trail: CursorTrail) => void>(() => undefined);
  const [state, setState] = useState<PageState>({
    status: 'loading', view: null, error: null, lastSuccessAt: null, refreshing: false, pendingPageIndex: null,
  });

  const load = useCallback((requestedTrail: CursorTrail): void => {
    if (!aliveRef.current) return;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    abortRef.current?.abort();
    const requestId = ++requestRef.current;
    const controller = new AbortController();
    abortRef.current = controller;
    const requestedFilters = filtersRef.current;
    const requestedKey = activeKeyRef.current;
    setState((current) => ({
      ...current,
      error: null,
      refreshing: current.view !== null,
      pendingPageIndex: requestedTrail.length - 1,
    }));

    void (async () => {
      let trail = requestedTrail;
      let page: ClipPage;
      try {
        page = await fetchClipPage(pageQuery(requestedFilters, trail.at(-1) ?? null), controller.signal);
      } catch (error) {
        // A cursor the backend no longer accepts (retention purged its boundary row) restores the
        // newest page exactly once instead of stranding the operator on a dead keyset position.
        if (trail.length === 1 || !isRetiredCursor(error)) throw error;
        trail = FIRST_TRAIL;
        page = await fetchClipPage(pageQuery(requestedFilters, null), controller.signal);
      }
      if (trail.length > 1 && page.clips.length === 0) {
        trail = FIRST_TRAIL;
        page = await fetchClipPage(pageQuery(requestedFilters, null), controller.signal);
      }
      if (!aliveRef.current || requestId !== requestRef.current || requestedKey !== activeKeyRef.current) return;
      const view = { filterKey: requestedKey, trail, page };
      viewRef.current = view;
      setState({
        status: 'success',
        view,
        error: null,
        lastSuccessAt: Date.now(),
        refreshing: false,
        pendingPageIndex: null,
      });
    })().catch((error: unknown) => {
      if (!aliveRef.current || requestId !== requestRef.current || controller.signal.aborted) return;
      setState((current) => ({ ...current, status: 'error', error, refreshing: false, pendingPageIndex: null }));
    }).finally(() => {
      if (!aliveRef.current || requestId !== requestRef.current) return;
      abortRef.current = null;
      if (viewRef.current !== null && viewRef.current.trail.length === 1) {
        timerRef.current = window.setTimeout(() => loadRef.current(FIRST_TRAIL), EVENTS_POLL_INTERVAL_MS);
      }
    });
  }, []);
  loadRef.current = load;

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      requestRef.current += 1;
      abortRef.current?.abort();
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, []);

  useEffect(() => {
    viewRef.current = null;
    setState({
      status: 'loading',
      view: null,
      error: null,
      lastSuccessAt: null,
      refreshing: false,
      pendingPageIndex: 0,
    });
    load(FIRST_TRAIL);
  }, [activeKey, load]);

  const currentView = state.view?.filterKey === activeKey ? state.view : null;
  const currentTrail = currentView?.trail ?? FIRST_TRAIL;
  const nextCursor = currentView?.page.pagination.next_cursor ?? null;
  return {
    status: currentView === null && state.status !== 'error' ? 'loading' : state.status,
    data: currentView?.page ?? null,
    error: state.error,
    lastSuccessAt: state.lastSuccessAt,
    refreshing: state.refreshing,
    pageIndex: currentTrail.length - 1,
    pendingPageIndex: state.pendingPageIndex,
    hasNextPage: nextCursor !== null,
    openNextPage: () => { if (nextCursor !== null) load([...currentTrail, nextCursor]); },
    openPreviousPage: () => { if (currentTrail.length > 1) load(currentTrail.slice(0, -1)); },
    openFirstPage: () => load(FIRST_TRAIL),
    refresh: () => load(currentTrail),
  };
}

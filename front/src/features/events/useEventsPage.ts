import { useCallback, useEffect, useRef, useState } from 'react';
import { EVENTS_PAGE_SIZE, EVENTS_POLL_INTERVAL_MS, getPageCount } from '@/features/events/paging';
import { fetchClipPage } from '@/shared/api/clipPagination';
import type { ClipPage, ClipPageQuery } from '@/shared/api/clipPaginationTypes';
import type { PollingResourceStatus } from '@/shared/api/usePollingResource';

export type EventsPageFilters = {
  readonly cameraId?: string;
  readonly eventType?: string;
};

type PageView = {
  readonly filterKey: string;
  readonly pageIndex: number;
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
  readonly navigate: (pageIndex: number) => void;
  readonly refresh: () => void;
};

function filterKey(filters: EventsPageFilters): string {
  return JSON.stringify([filters.cameraId ?? '', filters.eventType ?? '']);
}

function pageQuery(filters: EventsPageFilters, pageIndex: number): ClipPageQuery {
  return {
    ...(filters.cameraId ? { cameraId: filters.cameraId } : {}),
    ...(filters.eventType ? { eventType: filters.eventType } : {}),
    limit: EVENTS_PAGE_SIZE,
    offset: pageIndex * EVENTS_PAGE_SIZE,
  };
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
  const loadRef = useRef<(pageIndex: number) => void>(() => undefined);
  const [state, setState] = useState<PageState>({
    status: 'loading', view: null, error: null, lastSuccessAt: null, refreshing: false, pendingPageIndex: null,
  });

  const load = useCallback((requestedPageIndex: number): void => {
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
      pendingPageIndex: requestedPageIndex,
    }));

    void (async () => {
      let pageIndex = requestedPageIndex;
      let page = await fetchClipPage(pageQuery(requestedFilters, pageIndex), controller.signal);
      const lastPageIndex = getPageCount(page.pagination.total) - 1;
      if (pageIndex > lastPageIndex) {
        pageIndex = lastPageIndex;
        page = await fetchClipPage(pageQuery(requestedFilters, pageIndex), controller.signal);
        const correctedLastPageIndex = getPageCount(page.pagination.total) - 1;
        if (pageIndex > correctedLastPageIndex) {
          pageIndex = 0;
          page = await fetchClipPage(pageQuery(requestedFilters, pageIndex), controller.signal);
        }
      }
      if (!aliveRef.current || requestId !== requestRef.current || requestedKey !== activeKeyRef.current) return;
      const view = { filterKey: requestedKey, pageIndex, page };
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
      if (viewRef.current?.pageIndex === 0) {
        timerRef.current = window.setTimeout(() => loadRef.current(0), EVENTS_POLL_INTERVAL_MS);
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
    load(0);
  }, [activeKey, load]);

  const currentView = state.view?.filterKey === activeKey ? state.view : null;
  return {
    status: currentView === null && state.status !== 'error' ? 'loading' : state.status,
    data: currentView?.page ?? null,
    error: state.error,
    lastSuccessAt: state.lastSuccessAt,
    refreshing: state.refreshing,
    pageIndex: currentView?.pageIndex ?? 0,
    pendingPageIndex: state.pendingPageIndex,
    navigate: load,
    refresh: () => load(currentView?.pageIndex ?? 0),
  };
}

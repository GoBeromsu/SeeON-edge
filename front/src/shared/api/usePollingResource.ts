import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchCameras,
  fetchClips,
  fetchClipStorage,
  fetchConnection,
  fetchDetectionSettings,
  fetchRuntimeSettings,
  fetchStatus,
  fetchSystem,
} from '@/shared/api/client';
import type {
  CameraRegistry,
  Clip,
  ClipStorageInfo,
  ConnectionView,
  DetectionSettings,
  RuntimeSettings,
  StatusSnapshot,
  SystemSnapshot,
} from '@/shared/api/types';

export type PollingResourceStatus = 'idle' | 'loading' | 'success' | 'error';

export type PollingResource<T> = {
  status: PollingResourceStatus;
  data: T | null;
  error: unknown | null;
  lastSuccessAt: number | null;
  refreshing: boolean;
  retry: () => void;
  /** Commit an authoritative mutation response and invalidate older reads. */
  replace: (data: T) => void;
};

type PollingOptions<T> = {
  enabled: boolean;
  intervalMs: number;
  accept?: (incoming: T, current: T) => boolean;
};

export function usePollingResource<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  { enabled, intervalMs, accept }: PollingOptions<T>,
): PollingResource<T> {
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const requestRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<number | null>(null);
  const acceptRef = useRef(accept);
  acceptRef.current = accept;
  const aliveRef = useRef(false);
  const [resource, setResource] = useState<Omit<PollingResource<T>, 'retry' | 'replace'>>({
    status: enabled ? 'loading' : 'idle',
    data: null,
    error: null,
    lastSuccessAt: null,
    refreshing: false,
  });

  const run = useCallback(() => {
    if (!aliveRef.current) return;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    abortRef.current?.abort();
    const requestId = ++requestRef.current;
    const controller = new AbortController();
    abortRef.current = controller;
    setResource((current) => ({
      ...current,
      status: current.data === null ? 'loading' : current.status,
      error: null,
      refreshing: current.data !== null,
    }));

    void loaderRef.current(controller.signal).then(
      (data) => {
        if (!aliveRef.current || requestId !== requestRef.current) return;
        setResource((current) => {
          if (current.data !== null && acceptRef.current && !acceptRef.current(data, current.data)) {
            return { ...current, status: 'success', error: null, refreshing: false };
          }
          return { status: 'success', data, error: null, lastSuccessAt: Date.now(), refreshing: false };
        });
      },
      (error: unknown) => {
        if (!aliveRef.current || requestId !== requestRef.current || controller.signal.aborted) return;
        setResource((current) => ({ ...current, status: 'error', error, refreshing: false }));
      },
    ).finally(() => {
      if (!aliveRef.current || requestId !== requestRef.current) return;
      abortRef.current = null;
      timerRef.current = window.setTimeout(run, intervalMs);
    });
  }, [intervalMs]);

  const replace = useCallback((data: T) => {
    requestRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    setResource({
      status: 'success',
      data,
      error: null,
      lastSuccessAt: Date.now(),
      refreshing: false,
    });
  }, []);

  useEffect(() => {
    aliveRef.current = enabled;
    if (enabled) {
      run();
    } else {
      abortRef.current?.abort();
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      setResource((current) => ({ ...current, status: 'idle', refreshing: false }));
    }
    return () => {
      aliveRef.current = false;
      requestRef.current += 1;
      abortRef.current?.abort();
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [enabled, run]);

  return { ...resource, retry: run, replace };
}

export function useCamerasResource(enabled: boolean): PollingResource<CameraRegistry> {
  return usePollingResource(fetchCameras, { enabled, intervalMs: 5_000 });
}

export function useStatusResource(enabled: boolean): PollingResource<StatusSnapshot> {
  return usePollingResource(fetchStatus, { enabled, intervalMs: 3_000 });
}

export function useClipsResource(enabled: boolean, cameraId?: string): PollingResource<Clip[]> {
  return usePollingResource((signal) => fetchClips(cameraId, signal), { enabled, intervalMs: 8_000 });
}

export function useSystemResource(enabled: boolean): PollingResource<SystemSnapshot> {
  return usePollingResource(fetchSystem, { enabled, intervalMs: 10_000 });
}

export function useConnectionResource(enabled: boolean): PollingResource<ConnectionView> {
  return usePollingResource(fetchConnection, { enabled, intervalMs: 8_000 });
}

/** Applies to every camera; the backend polls this into worker config on its own multi-second cycle (no UI immediacy promise). */
export function useDetectionSettingsResource(enabled: boolean): PollingResource<DetectionSettings> {
  return usePollingResource(fetchDetectionSettings, { enabled, intervalMs: 15_000 });
}

export function useRuntimeSettingsResource(enabled: boolean): PollingResource<RuntimeSettings> {
  return usePollingResource(fetchRuntimeSettings, {
    enabled,
    intervalMs: 15_000,
    accept: (incoming, current) => incoming.version >= current.version,
  });
}

export function useClipStorageResource(enabled: boolean): PollingResource<ClipStorageInfo> {
  return usePollingResource(fetchClipStorage, { enabled, intervalMs: 15_000 });
}

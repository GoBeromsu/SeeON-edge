import { useEffect, useRef, useState } from 'react';

export type MjpegStreamStatus = 'connecting' | 'connected' | 'reconnecting';

export type MjpegStream = {
  /** `null` when there is nothing to stream (camera offline / target unset) -- render no `<img>`. */
  src: string | null;
  status: MjpegStreamStatus;
  onLoad: () => void;
  onError: () => void;
};

/** Forces a fresh `<img>` request even when no `onError` ever fires, to recover from a stream that
 * silently stalled (worker restart, network blip) while the browser keeps showing the last frame. */
const FORCED_REMOUNT_MS = 20_000;
const BASE_BACKOFF_MS = 1_000;
const MAX_BACKOFF_MS = 16_000;

function withCacheBust(url: string, key: number): string {
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}_r=${key}`;
}

/**
 * Drives an MJPEG `<img src=...>` with silent-stall recovery: a periodic forced re-mount (cache-
 * busting query param) on a timer, plus exponential backoff re-mounts after consecutive `onError`.
 * `status` is `'reconnecting'` while a backoff re-mount is pending -- callers show a "연결 끊김"
 * overlay for that state. Pass `null` for `baseUrl` to stop streaming (e.g. offline camera, tile
 * scrolled offscreen); the hook then reports `src: null` and callers fall back to a snapshot.
 */
export function useMjpegStream(baseUrl: string | null): MjpegStream {
  const [cacheKey, setCacheKey] = useState(0);
  const [status, setStatus] = useState<MjpegStreamStatus>('connecting');
  const errorCountRef = useRef(0);
  const backoffTimerRef = useRef<number | null>(null);

  const clearBackoffTimer = (): void => {
    if (backoffTimerRef.current !== null) {
      window.clearTimeout(backoffTimerRef.current);
      backoffTimerRef.current = null;
    }
  };

  // New target stream: reset error/backoff state and force a fresh mount.
  useEffect(() => {
    errorCountRef.current = 0;
    clearBackoffTimer();
    setStatus('connecting');
    setCacheKey((key) => key + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

  // Periodic forced re-mount, independent of error state.
  useEffect(() => {
    if (!baseUrl) return undefined;
    const interval = window.setInterval(() => {
      setCacheKey((key) => key + 1);
    }, FORCED_REMOUNT_MS);
    return () => window.clearInterval(interval);
  }, [baseUrl]);

  useEffect(() => clearBackoffTimer, []);

  const onLoad = (): void => {
    errorCountRef.current = 0;
    clearBackoffTimer();
    setStatus('connected');
  };

  const onError = (): void => {
    errorCountRef.current += 1;
    setStatus('reconnecting');
    const backoff = Math.min(BASE_BACKOFF_MS * 2 ** (errorCountRef.current - 1), MAX_BACKOFF_MS);
    clearBackoffTimer();
    backoffTimerRef.current = window.setTimeout(() => {
      backoffTimerRef.current = null;
      setCacheKey((key) => key + 1);
    }, backoff);
  };

  return {
    src: baseUrl ? withCacheBust(baseUrl, cacheKey) : null,
    status,
    onLoad,
    onError,
  };
}

import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelClipAnalysis, fetchClipAnalysis, triggerClipAnalysis } from '@/shared/api/client';
import type { ClipAnalysisStatus } from '@/shared/api/types';

type State = { status: ClipAnalysisStatus | null; pending: boolean };

const initial: ClipAnalysisStatus = { state: 'idle', served_media_sha256: '' };

export function useClipAnalysis(clipId: string | undefined, enabled: boolean): {
  status: ClipAnalysisStatus;
  received: boolean;
  settled: boolean;
  trigger: () => void;
  cancel: () => void;
} {
  const [state, setState] = useState<State>({ status: null, pending: false });
  const controller = useRef<AbortController | null>(null);
  const request = useCallback((operation: (id: string, signal: AbortSignal) => Promise<ClipAnalysisStatus>) => {
    if (!clipId) return;
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    setState((previous) => ({ ...previous, pending: true }));
    void operation(clipId, next.signal).then(
      (status) => { if (!next.signal.aborted) setState({ status, pending: false }); },
      () => { if (!next.signal.aborted) setState({ status: { state: 'unavailable', served_media_sha256: '' }, pending: false }); },
    );
  }, [clipId]);

  useEffect(() => {
    controller.current?.abort();
    if (!clipId || !enabled) {
      setState({ status: null, pending: false });
      return undefined;
    }
    request(fetchClipAnalysis);
    return () => controller.current?.abort();
  }, [clipId, enabled, request]);

  useEffect(() => {
    if (!enabled || (state.status?.state !== 'queued' && state.status?.state !== 'running')) return undefined;
    const timer = window.setInterval(() => request(fetchClipAnalysis), 2_000);
    return () => window.clearInterval(timer);
  }, [enabled, request, state.status?.state]);

  return {
    status: state.pending ? { state: 'running', served_media_sha256: state.status?.served_media_sha256 ?? '' } : state.status ?? initial,
    received: Boolean(state.status?.served_media_sha256),
    settled: state.status !== null,
    trigger: () => request(triggerClipAnalysis),
    cancel: () => {
      if (!clipId) return;
      controller.current?.abort();
      const next = new AbortController();
      controller.current = next;
      setState((previous) => ({ ...previous, pending: true }));
      void cancelClipAnalysis(clipId, next.signal).then(
        () => {
          if (!next.signal.aborted) {
            controller.current = null;
            request(fetchClipAnalysis);
          }
        },
        () => {
          if (!next.signal.aborted) setState({ status: { state: 'unavailable', served_media_sha256: '' }, pending: false });
        },
      );
    },
  };
}

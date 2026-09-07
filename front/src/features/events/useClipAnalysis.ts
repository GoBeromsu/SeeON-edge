import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelClipAnalysis, fetchClipAnalysis, triggerClipAnalysis } from '@/shared/api/client';
import type { ClipAnalysisStatus } from '@/shared/api/types';

type State = { status: ClipAnalysisStatus; pending: boolean; received: boolean; settled: boolean };

const initial: ClipAnalysisStatus = { state: 'idle', served_media_sha256: '' };

export function useClipAnalysis(clipId: string | undefined, enabled: boolean): {
  status: ClipAnalysisStatus;
  received: boolean;
  settled: boolean;
  trigger: () => void;
  cancel: () => void;
} {
  const [state, setState] = useState<State>({ status: initial, pending: false, received: false, settled: false });
  const controller = useRef<AbortController | null>(null);
  const request = useCallback((operation: (id: string, signal: AbortSignal) => Promise<ClipAnalysisStatus>) => {
    if (!clipId) return;
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    setState((previous) => ({ ...previous, pending: true }));
    void operation(clipId, next.signal).then(
      (status) => { if (!next.signal.aborted) setState({ status, pending: false, received: true, settled: true }); },
      () => { if (!next.signal.aborted) setState({ status: { state: 'unavailable', served_media_sha256: '' }, pending: false, received: false, settled: true }); },
    );
  }, [clipId]);

  useEffect(() => {
    controller.current?.abort();
    if (!clipId || !enabled) {
      setState({ status: initial, pending: false, received: false, settled: false });
      return undefined;
    }
    request(fetchClipAnalysis);
    return () => controller.current?.abort();
  }, [clipId, enabled, request]);

  useEffect(() => {
    if (!enabled || state.status.state !== 'running') return undefined;
    const timer = window.setInterval(() => request(fetchClipAnalysis), 2_000);
    return () => window.clearInterval(timer);
  }, [enabled, request, state.status.state]);

  return {
    status: state.pending ? { state: 'running', served_media_sha256: state.status.served_media_sha256 } : state.status,
    received: state.received,
    settled: state.settled,
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
          if (!next.signal.aborted) setState({ status: { state: 'unavailable', served_media_sha256: '' }, pending: false, received: false, settled: true });
        },
      );
    },
  };
}

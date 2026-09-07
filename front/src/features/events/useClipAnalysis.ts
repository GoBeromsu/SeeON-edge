import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelClipAnalysis, fetchClipAnalysis, triggerClipAnalysis } from '@/shared/api/client';
import type { ClipAnalysisStatus } from '@/shared/api/types';

type State = { status: ClipAnalysisStatus; pending: boolean };

const initial: ClipAnalysisStatus = { state: 'idle' };

export function useClipAnalysis(clipId: string | undefined, enabled: boolean): {
  status: ClipAnalysisStatus;
  trigger: () => void;
  cancel: () => void;
} {
  const [state, setState] = useState<State>({ status: initial, pending: false });
  const controller = useRef<AbortController | null>(null);
  const request = useCallback((operation: (id: string, signal: AbortSignal) => Promise<ClipAnalysisStatus>) => {
    if (!clipId) return;
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    setState((previous) => ({ ...previous, pending: true }));
    void operation(clipId, next.signal).then(
      (status) => { if (!next.signal.aborted) setState({ status, pending: false }); },
      () => { if (!next.signal.aborted) setState({ status: { state: 'unavailable' }, pending: false }); },
    );
  }, [clipId]);

  useEffect(() => {
    controller.current?.abort();
    if (!clipId || !enabled) {
      setState({ status: initial, pending: false });
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
    status: state.pending ? { state: 'running' } : state.status,
    trigger: () => request(triggerClipAnalysis),
    cancel: () => request(cancelClipAnalysis),
  };
}

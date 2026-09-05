import { useEffect, useRef, useState } from 'react';
import { fetchCameraOverlay, setCameraOverlay, type OverlaySelection } from '@/shared/api/client';
import { toast } from '@/shared/ui/Toast';

type OverlaySelectionState = {
  status: 'loading' | 'success' | 'error';
  selection: OverlaySelection | null;
  pending: boolean;
};

const OVERLAY_TARGETS: { key: keyof OverlaySelection; label: string }[] = [
  { key: 'person', label: '사람' },
  { key: 'bed', label: '침대' },
];

export function OverlayTargetIcon({ target }: { target: keyof OverlaySelection }): JSX.Element {
  return (
    <svg aria-hidden="true" data-overlay-target={target} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      {target === 'person' ? (
        <>
          <circle cx="12" cy="5" r="2.5" />
          <path d="M8 21v-5l1-6h6l1 6v5M9 12l-3 4M15 12l3 4" />
        </>
      ) : (
        <>
          <path d="M3 6v12M3 14h18v4M7 10h4a3 3 0 0 1 3 3v1M7 10V7h4a3 3 0 0 1 3 3" />
        </>
      )}
    </svg>
  );
}

export function useCameraOverlaySelection(cameraId: string): {
  state: OverlaySelectionState;
  toggle: (key: keyof OverlaySelection) => void;
  retry: () => void;
} {
  const [state, setState] = useState<OverlaySelectionState>({ status: 'loading', selection: null, pending: false });
  const [fetchAttempt, setFetchAttempt] = useState(0);
  const cameraIdRef = useRef(cameraId);
  cameraIdRef.current = cameraId;

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading', selection: null, pending: false });
    fetchCameraOverlay(cameraId)
      .then((selection) => {
        if (!cancelled) setState({ status: 'success', selection, pending: false });
      })
      .catch(() => {
        if (!cancelled) setState({ status: 'error', selection: null, pending: false });
      });
    return () => {
      cancelled = true;
    };
  }, [cameraId, fetchAttempt]);

  const retry = (): void => setFetchAttempt((attempt) => attempt + 1);

  const toggle = (key: keyof OverlaySelection): void => {
    if (!state.selection || state.pending) return;
    const requestedFor = cameraIdRef.current;
    const requested = { ...state.selection, [key]: !state.selection[key] };
    setState((previous) => ({ ...previous, pending: true }));
    setCameraOverlay(requestedFor, requested)
      .then((confirmed) => {
        if (cameraIdRef.current !== requestedFor) return;
        setState({ status: 'success', selection: confirmed, pending: false });
      })
      .catch(() => {
        if (cameraIdRef.current !== requestedFor) return;
        setState((previous) => ({ ...previous, pending: false }));
        toast.error('오버레이 대상을 변경하지 못했습니다.');
      });
  };

  return { state, toggle, retry };
}

type OverlaySelectionControlProps = {
  cameraId: string;
  /**
   * Notified with the confirmed overlay selection whenever it changes (initial load, retry, or a
   * successful selection) — `null` while loading/erroring. LiveStreamPanel's live badge label
   * (issue #102) needs the current selection from this same fetch/selection state, not an
   * independent copy that could fall out of sync with what the operator picked here, so the
   * shared RoomDetail ancestor lifts this via the callback rather than re-fetching.
   */
  onSelectionChange?: (selection: OverlaySelection | null) => void;
};

export function OverlaySelectionControl({ cameraId, onSelectionChange }: OverlaySelectionControlProps): JSX.Element {
  const { state, toggle, retry } = useCameraOverlaySelection(cameraId);

  useEffect(() => {
    onSelectionChange?.(state.selection);
  }, [state.selection, onSelectionChange]);

  return (
    <div className="border-t border-border pt-3">
      <p className="mb-2 text-sm font-semibold text-foreground">오버레이</p>
      {state.status === 'loading' ? (
        <p role="status" className="text-sm text-muted-foreground">오버레이 설정을 불러오는 중입니다…</p>
      ) : null}
      {state.status === 'error' ? (
        <div className="flex items-center gap-2">
          <p role="alert" className="text-sm text-destructive">오버레이 설정을 불러오지 못했습니다.</p>
          <button
            type="button"
            onClick={retry}
            className="min-h-11 rounded-control border border-border bg-card px-3 text-sm font-semibold text-foreground"
          >
            다시 시도
          </button>
        </div>
      ) : null}
      {state.status === 'success' ? (
        <div role="group" aria-label="오버레이 대상" className="flex flex-wrap gap-2">
          {OVERLAY_TARGETS.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              disabled={state.pending}
              onClick={() => toggle(key)}
              aria-pressed={state.selection?.[key] === true}
              className={`inline-flex min-h-11 items-center gap-2 rounded-control border px-3 text-sm font-semibold disabled:opacity-60 ${
                state.selection?.[key] ? 'border-primary bg-primary text-primary-foreground' : 'border-border bg-card text-foreground'
              }`}
            >
              <OverlayTargetIcon target={key} />
              {label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

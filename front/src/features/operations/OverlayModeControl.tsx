import { useEffect, useRef, useState } from 'react';
import { fetchCameraOverlay, setCameraOverlay, type OverlayMode } from '@/shared/api/client';
import { toast } from '@/shared/ui/Toast';

type OverlayModeState = {
  status: 'loading' | 'success' | 'error';
  mode: OverlayMode | null;
  pending: boolean;
};

const OVERLAY_CHIPS: { mode: OverlayMode; label: string }[] = [
  { mode: 'none', label: '오버레이 없음' },
  { mode: 'bedexit', label: '침대 이탈' },
  { mode: 'fall', label: '낙상' },
];

export function useCameraOverlayMode(cameraId: string): {
  state: OverlayModeState;
  select: (mode: OverlayMode) => void;
} {
  const [state, setState] = useState<OverlayModeState>({ status: 'loading', mode: null, pending: false });
  const cameraIdRef = useRef(cameraId);
  cameraIdRef.current = cameraId;

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading', mode: null, pending: false });
    fetchCameraOverlay(cameraId)
      .then((mode) => {
        if (!cancelled) setState({ status: 'success', mode, pending: false });
      })
      .catch(() => {
        if (!cancelled) setState({ status: 'error', mode: null, pending: false });
      });
    return () => {
      cancelled = true;
    };
  }, [cameraId]);

  const select = (mode: OverlayMode) => {
    const requestedFor = cameraIdRef.current;
    setState((previous) => ({ ...previous, pending: true }));
    setCameraOverlay(requestedFor, mode)
      .then((confirmed) => {
        if (cameraIdRef.current !== requestedFor) return;
        setState({ status: 'success', mode: confirmed, pending: false });
      })
      .catch(() => {
        if (cameraIdRef.current !== requestedFor) return;
        setState((previous) => ({ ...previous, pending: false }));
        toast.error('오버레이 모드를 변경하지 못했습니다.');
      });
  };

  return { state, select };
}

type OverlayModeControlProps = {
  cameraId: string;
};

export function OverlayModeControl({ cameraId }: OverlayModeControlProps): JSX.Element {
  const { state, select } = useCameraOverlayMode(cameraId);

  return (
    <div className="border-t border-border pt-3">
      <p className="mb-2 text-sm font-semibold text-foreground">오버레이</p>
      {state.status === 'loading' ? (
        <p role="status" className="text-sm text-muted-foreground">오버레이 모드를 불러오는 중입니다…</p>
      ) : null}
      {state.status === 'error' ? (
        <p role="alert" className="text-sm text-destructive">오버레이 모드를 불러오지 못했습니다.</p>
      ) : null}
      {state.status === 'success' ? (
        <div role="group" aria-label="오버레이 모드 선택" className="flex flex-wrap gap-2">
          {OVERLAY_CHIPS.map(({ mode, label }) => (
            <button
              key={mode}
              type="button"
              disabled={state.pending}
              onClick={() => select(mode)}
              aria-pressed={state.mode === mode}
              className={`min-h-11 rounded-control border px-3 text-sm font-semibold disabled:opacity-60 ${
                state.mode === mode ? 'border-primary bg-primary text-primary-foreground' : 'border-border bg-card text-foreground'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

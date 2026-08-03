import { useState } from 'react';
import { getCameraStreamUrl, type Camera, type RuntimeCameraDiagnostics } from '@/shared/api/client';

type LiveStreamPanelProps = {
  camera: Camera;
  diagnostics: RuntimeCameraDiagnostics | undefined;
  onRetryConnection: () => void;
  onManageConnection: () => void;
};

/**
 * Live MJPEG-over-`<img>` view (no WebRTC/HLS per the codebase convention). The diagnostics chip
 * only renders fields that actually exist on RuntimeCameraDiagnostics (decode backend + measured
 * FPS) — the design mock implies separate decode/inference FPS and a frame counter, but the API
 * only ever reports one `measured_fps`, so this deliberately omits the fabricated fields.
 */
export function LiveStreamPanel({ camera, diagnostics, onRetryConnection, onManageConnection }: LiveStreamPanelProps): JSX.Element {
  const [streamError, setStreamError] = useState(false);
  const online = camera.status === 'online';
  const showStream = online && !streamError;

  const fps = diagnostics?.measured_fps;
  const backend = diagnostics?.decode.selected ?? diagnostics?.decode.requested ?? null;

  if (!online) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-4 rounded-card border border-border bg-muted p-6 text-center">
        <div>
          <p className="text-base font-semibold text-foreground">카메라에 연결할 수 없습니다</p>
          <p className="mt-1 text-sm text-muted-foreground">탐지가 중단된 상태입니다</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              setStreamError(false);
              onRetryConnection();
            }}
            className="min-h-11 rounded-control border border-border bg-card px-4 text-sm font-semibold text-foreground"
          >
            재연결 시도
          </button>
          <button
            type="button"
            onClick={onManageConnection}
            className="min-h-11 rounded-control border border-border bg-card px-4 text-sm font-semibold text-foreground"
          >
            연결 관리
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-card border border-border bg-card event-media-frame">
      {showStream ? (
        <img
          key={camera.id}
          src={getCameraStreamUrl(camera.id)}
          alt={`${camera.label} 실시간 영상`}
          className="h-full w-full object-cover"
          onError={() => setStreamError(true)}
        />
      ) : (
        <span className="event-media-unavailable" role="status">실시간 영상을 불러올 수 없습니다</span>
      )}

      {showStream ? (
        <span
          className="media-status-overlay tabular-nums absolute right-3 top-3 rounded-control px-2 py-1 text-xs font-semibold"
          aria-label="스트림 진단 정보"
        >
          {fps !== null && fps !== undefined ? `${fps.toFixed(1)} FPS` : 'FPS 측정 중'}
          {backend ? ` · ${backend}` : ''}
        </span>
      ) : null}
    </div>
  );
}

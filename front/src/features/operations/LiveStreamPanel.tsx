import { getCameraStreamUrl, type Camera, type OverlayMode, type RuntimeCameraDiagnostics } from '@/shared/api/client';
import { useMjpegStream } from '@/features/operations/useMjpegStream';

type LiveStreamPanelProps = {
  camera: Camera;
  diagnostics: RuntimeCameraDiagnostics | undefined;
  overlayMode: OverlayMode | null;
  onRetryConnection: () => void;
  onManageConnection: () => void;
};

/**
 * Live MJPEG-over-`<img>` view (no WebRTC/HLS per the codebase convention). The diagnostics chip
 * only renders fields that actually exist on RuntimeCameraDiagnostics (decode backend + measured
 * FPS) — the design mock implies separate decode/inference FPS and a frame counter, but the API
 * only ever reports one `measured_fps`, so this deliberately omits the fabricated fields.
 *
 * Stall recovery (issue #83) is delegated to useMjpegStream: periodic forced re-mount + backoff
 * re-mount after consecutive onError, surfaced here as a "연결 끊김" overlay while reconnecting.
 *
 * The bottom-left badge label mirrors design-handoff/Eldercare Prototype.dc.html:645's `liveLabel`
 * data model: '라이브 · 낙상 없음' when the camera's overlay mode is 'fall', else '라이브 · 온라인'.
 * `overlayMode` is the OverlayModeControl selection lifted to the shared RoomDetail ancestor
 * (issue #102) rather than fetched again here, so the badge always agrees with what the operator
 * actually selected.
 */
export function LiveStreamPanel({ camera, diagnostics, overlayMode, onRetryConnection, onManageConnection }: LiveStreamPanelProps): JSX.Element {
  const online = camera.status === 'online';
  const stream = useMjpegStream(online ? getCameraStreamUrl(camera.id) : null);
  const showStream = online && stream.src !== null;

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
            onClick={onRetryConnection}
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
          key={stream.src}
          src={stream.src ?? undefined}
          alt={`${camera.label} 실시간 영상`}
          className="h-full w-full object-cover"
          onLoad={stream.onLoad}
          onError={stream.onError}
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

      {showStream ? (
        <span className="media-status-overlay absolute bottom-2 left-2 rounded-control px-2.5 py-1 text-xs font-semibold">
          {overlayMode === 'fall' ? '라이브 · 낙상 없음' : '라이브 · 온라인'}
        </span>
      ) : null}

      {stream.status === 'reconnecting' ? (
        <span
          role="status"
          className="absolute inset-0 flex items-center justify-center bg-black/60 text-sm font-semibold text-white"
        >
          연결 끊김 · 재연결 중…
        </span>
      ) : null}
    </div>
  );
}

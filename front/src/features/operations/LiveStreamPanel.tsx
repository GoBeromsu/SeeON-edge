import { getCameraStreamUrl, type Camera, type OverlayMode, type RuntimeCameraDiagnostics } from '@/shared/api/client';
import { useMjpegStream } from '@/shared/api/useMjpegStream';

type LiveStreamPanelProps = {
  camera: Camera;
  diagnostics: RuntimeCameraDiagnostics | undefined;
  overlayMode: OverlayMode | null;
  onRetryConnection: () => void;
  onManageConnection: () => void;
};

/**
 * Live MJPEG-over-`fetch`+`<canvas>` view (no WebRTC/HLS per the codebase convention). The
 * diagnostics chip only renders fields that actually exist on RuntimeCameraDiagnostics (decode
 * backend + measured FPS) — the design mock implies separate decode/inference FPS and a frame
 * counter, but the API only ever reports one `measured_fps`, so this deliberately omits the
 * fabricated fields.
 *
 * The chip has three distinct states (issue #160): no value yet ("FPS 측정 중"), a live value
 * ("N.N FPS · backend"), and a stale value — the reporting worker hasn't published within
 * `stale_after_sec` but `measured_fps` still holds its last-known reading. That last case is
 * shown as "⚠ 측정 중단 · 마지막 N.N" in a warning color instead of silently keeping the old
 * value on screen as if it were current.
 *
 * Stall recovery (issue #83) is delegated to useMjpegStream: it draws each frame onto the canvas
 * directly and only reconnects once frames have actually stopped arriving for a few seconds, so
 * the last frame stays on screen the whole time. This panel surfaces that as a small corner badge
 * (not a full-frame overlay) while `stream.status === 'stalled'`, so the frozen last frame stays
 * visible underneath instead of being hidden.
 *
 * The bottom-left badge reports only the selected display mode. `overlayMode` is the
 * OverlayModeControl selection lifted to the shared RoomDetail ancestor (issue #102), not
 * detection state, so the label must not imply that a fall or bed presence was evaluated.
 */
export function LiveStreamPanel({ camera, diagnostics, overlayMode, onRetryConnection, onManageConnection }: LiveStreamPanelProps): JSX.Element {
  const online = camera.status === 'online';
  const stream = useMjpegStream(online ? getCameraStreamUrl(camera.id) : null);

  const fps = diagnostics?.measured_fps;
  const backend = diagnostics?.decode.selected ?? diagnostics?.decode.requested ?? null;
  // stale는 워커가 죽어도 마지막 measured_fps를 지우지 않으므로(#160), 값이 없어
  // 아직 측정 전인 경우와 값이 있지만 멈춰버린 경우를 구분해서 보여준다.
  const isStale = diagnostics?.stale === true;
  const fpsKnown = fps !== null && fps !== undefined;
  const liveLabel = overlayMode === 'fall'
    ? '라이브 · 사람 표시'
    : overlayMode === 'bedexit'
      ? '라이브 · 침대 영역·사람 표시'
      : overlayMode === 'none'
        ? '라이브 · 원본'
        : '라이브';

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
      <canvas
        ref={stream.canvasRef}
        role="img"
        aria-label={`${camera.label} 실시간 영상`}
        className="h-full w-full object-cover"
      />

      <span
        className={`media-status-overlay tabular-nums absolute right-3 top-3 rounded-control px-2 py-1 text-xs font-semibold ${isStale && fpsKnown ? 'text-amber-400' : ''}`}
        aria-label="스트림 진단 정보"
      >
        {isStale && fpsKnown
          ? `⚠ 측정 중단 · 마지막 ${(fps as number).toFixed(1)}`
          : fpsKnown
            ? `${(fps as number).toFixed(1)} FPS${backend ? ` · ${backend}` : ''}`
            : 'FPS 측정 중'}
      </span>

      <span className="media-status-overlay absolute bottom-2 left-2 rounded-control px-2.5 py-1 text-xs font-semibold">
        {liveLabel}
      </span>

      {stream.status === 'stalled' ? (
        <span
          role="status"
          className="media-status-overlay absolute left-3 top-3 rounded-control px-2 py-1 text-xs font-semibold"
        >
          연결 끊김 · 재연결 중…
        </span>
      ) : null}
    </div>
  );
}

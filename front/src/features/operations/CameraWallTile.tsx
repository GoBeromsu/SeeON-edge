import { useEffect, useState } from 'react';
import { getCameraStreamUrl, type Camera } from '@/shared/api/client';
import type { SnapshotEntry, SnapshotQueue } from '@/features/operations/SnapshotQueue';
import { useMjpegStream } from '@/features/operations/useMjpegStream';

type CameraWallTileProps = {
  camera: Camera;
  snapshot: SnapshotEntry | undefined;
  queue: SnapshotQueue;
  onSelect: () => void;
};

/** Tracks whether `node` is at least partially in the viewport; `false` until IntersectionObserver
 * confirms visibility (jsdom/older browsers without IntersectionObserver stream unconditionally). */
function useIsTileVisible(node: HTMLElement | null): boolean {
  const [visible, setVisible] = useState(typeof IntersectionObserver === 'undefined');

  useEffect(() => {
    if (!node || typeof IntersectionObserver === 'undefined') return undefined;
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry?.isIntersecting ?? false),
      { threshold: 0.1 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);

  return visible;
}

/**
 * 16:9 object-cover tile. Online + visible tiles stream live MJPEG via `fetch`+`<canvas>` (issue
 * #82 — the wall must look like playing video, not a 5s snapshot poll); offscreen tiles fall back
 * to the SnapshotQueue snapshot to save bandwidth, and any tile also falls back to the snapshot
 * while the stream is stalled — no frame for a few seconds — and reconnecting (issue #83). Offline
 * tiles are unchanged (gray placeholder).
 */
export function CameraWallTile({ camera, snapshot, queue, onSelect }: CameraWallTileProps): JSX.Element {
  const online = camera.status === 'online';
  const [tileNode, setTileNode] = useState<HTMLButtonElement | null>(null);
  const isVisible = useIsTileVisible(tileNode);
  const streamActive = online && isVisible;
  const stream = useMjpegStream(streamActive ? getCameraStreamUrl(camera.id) : null);
  const showingStream = streamActive && stream.status === 'live';

  const snapshotStale = !showingStream && (snapshot?.state === 'error' || snapshot?.state === 'stale') && !!snapshot?.lastLoadedUrl;
  const showDisconnectedBadge = (streamActive && stream.status === 'stalled') || snapshotStale;

  return (
    <button
      type="button"
      ref={setTileNode}
      onClick={onSelect}
      aria-label={`${camera.label} 열기`}
      data-camera-id={camera.id}
      className="min-h-11 w-full overflow-hidden rounded-card border border-border bg-card text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
    >
      {online ? (
        <span className="relative block event-media-frame">
          {snapshot?.lastLoadedUrl ? (
            <img
              key={snapshot.lastLoadedUrl}
              src={snapshot.lastLoadedUrl}
              alt={`${camera.label} 최근 영상`}
              className="h-full w-full object-cover"
            />
          ) : (
            <span className="event-media-unavailable">
              {snapshot?.state === 'error' ? '영상을 불러올 수 없습니다' : '불러오는 중…'}
            </span>
          )}
          {streamActive ? (
            <canvas
              ref={stream.canvasRef}
              role="img"
              aria-label={`${camera.label} 실시간 영상`}
              className={`absolute inset-0 h-full w-full object-cover ${showingStream ? '' : 'opacity-0'}`}
            />
          ) : null}
          {snapshot?.requestUrl ? (
            <img
              src={snapshot.requestUrl}
              alt=""
              aria-hidden="true"
              className="hidden"
              onLoad={(event) => queue.resolve(camera.id, camera.id, 'loaded', event.currentTarget.getAttribute('src') ?? undefined)}
              onError={(event) => queue.resolve(camera.id, camera.id, 'error', event.currentTarget.getAttribute('src') ?? undefined)}
            />
          ) : null}
          {showDisconnectedBadge ? (
            <span
              role="status"
              className="media-status-overlay absolute left-2 top-2 rounded-control px-2 py-1 text-xs font-semibold"
            >
              연결 끊김
            </span>
          ) : null}
          <span
            className="media-status-overlay absolute inset-x-0 bottom-0 flex items-center justify-between px-3 py-2 text-sm font-semibold"
            style={{ background: 'linear-gradient(transparent, rgba(0,0,0,.72))' }}
          >
            <span className="truncate">{camera.label}</span>
            <span aria-hidden="true" className="ml-2 h-2 w-2 shrink-0 rounded-full bg-status-approvedFg" />
          </span>
        </span>
      ) : (
        <span className="flex aspect-video items-center justify-center rounded-t-card bg-muted text-sm text-muted-foreground">
          오프라인
        </span>
      )}
      {!online ? (
        <span className="flex items-center justify-between border-t border-border bg-card px-3 py-2 text-sm font-semibold text-foreground">
          <span className="truncate">{camera.label}</span>
          <span aria-hidden="true" className="ml-2 h-2 w-2 shrink-0 rounded-full bg-status-rejectedFg" />
        </span>
      ) : null}
    </button>
  );
}

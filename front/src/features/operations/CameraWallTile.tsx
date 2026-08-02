import type { Camera } from '@/shared/api/client';
import type { SnapshotEntry, SnapshotQueue } from '@/features/operations/SnapshotQueue';

type CameraWallTileProps = {
  camera: Camera;
  snapshot: SnapshotEntry | undefined;
  queue: SnapshotQueue;
  onSelect: () => void;
};

/** 16:9 object-cover tile: online = live snapshot + gradient name/status bar over the video; offline = gray placeholder + separate bottom bar. */
export function CameraWallTile({ camera, snapshot, queue, onSelect }: CameraWallTileProps): JSX.Element {
  const online = camera.status === 'online';

  return (
    <button
      type="button"
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
          <span
            className="media-status-overlay absolute inset-x-0 bottom-0 flex items-center justify-between px-3 py-2 text-sm font-semibold"
            style={{ background: 'linear-gradient(transparent, rgba(0,0,0,.72))' }}
          >
            <span className="truncate">{camera.label}</span>
            <span aria-hidden="true" className="ml-2 h-2 w-2 shrink-0 rounded-full bg-status-approvedFg" />
          </span>
        </span>
      ) : (
        <span className="block event-media-frame">
          <span className="event-media-unavailable">오프라인</span>
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

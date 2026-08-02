import { useMemo, useState } from 'react';
import { useClipsResource } from '@/shared/api/usePollingResource';
import { eventTypeLabel, formatClipDateTime } from '@/features/operations/operationsModel';
import type { Clip } from '@/shared/api/client';

type SortMode = 'recent' | 'type';

type EventHistoryListProps = {
  cameraId: string;
  onSelectClip: (clip: Clip) => void;
};

function sortByRecency(clips: readonly Clip[]): Clip[] {
  return [...clips].sort((left, right) => {
    const leftTime = left.created_at ? Date.parse(left.created_at) : 0;
    const rightTime = right.created_at ? Date.parse(right.created_at) : 0;
    return rightTime - leftTime;
  });
}

function groupByType(clips: readonly Clip[]): [string, Clip[]][] {
  const groups = new Map<string, Clip[]>();
  for (const clip of sortByRecency(clips)) {
    const list = groups.get(clip.event_type) ?? [];
    list.push(clip);
    groups.set(clip.event_type, list);
  }
  return [...groups.entries()].sort((left, right) => right[1].length - left[1].length);
}

function ClipTile({ clip, onSelect }: { clip: Clip; onSelect: () => void }): JSX.Element {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="flex flex-col overflow-hidden rounded-card border border-border bg-card text-left"
    >
      <span className="event-media-frame relative block">
        {clip.video_available ? (
          <video
            className="h-full w-full object-cover"
            src={clip.video_path}
            muted
            playsInline
            preload="metadata"
            tabIndex={-1}
            aria-hidden="true"
          />
        ) : (
          <span className="event-media-unavailable px-3 text-center text-xs">
            {clip.video_error ?? '영상을 사용할 수 없습니다.'}
          </span>
        )}
      </span>
      <span className="flex items-center justify-between gap-2 p-3">
        <span className="text-sm font-semibold text-foreground">{eventTypeLabel(clip.event_type)}</span>
        <span className="tabular-nums text-xs text-muted-foreground">{formatClipDateTime(clip.created_at)}</span>
      </span>
    </button>
  );
}

export function EventHistoryList({ cameraId, onSelectClip }: EventHistoryListProps): JSX.Element {
  const { status, data, retry } = useClipsResource(true, cameraId);
  const [sortMode, setSortMode] = useState<SortMode>('recent');
  const clips = data ?? [];
  const grouped = useMemo(() => groupByType(clips), [clips]);
  const hasData = clips.length > 0;

  return (
    <section aria-label="이벤트 히스토리">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">이벤트 히스토리</h2>
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          정렬
          <select
            value={sortMode}
            onChange={(event) => setSortMode(event.target.value as SortMode)}
            className="min-h-11 rounded-control border border-border bg-card px-2 text-sm text-foreground"
          >
            <option value="recent">최신순</option>
            <option value="type">종류별</option>
          </select>
        </label>
      </div>

      {status === 'loading' && !hasData ? (
        <p role="status" className="py-8 text-center text-sm text-muted-foreground">이벤트를 불러오는 중입니다…</p>
      ) : null}

      {status === 'error' && !hasData ? (
        <div role="alert" className="flex flex-col items-center gap-3 py-8 text-center text-sm text-destructive">
          <p>이벤트를 불러오지 못했습니다.</p>
          <button type="button" onClick={retry} className="min-h-11 rounded-control border border-border px-4 text-sm font-semibold text-foreground">
            다시 시도
          </button>
        </div>
      ) : null}

      {status !== 'loading' && status !== 'error' && !hasData ? (
        <p role="status" className="py-8 text-center text-sm text-muted-foreground">기록된 이벤트가 없습니다.</p>
      ) : null}

      {hasData && sortMode === 'recent' ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {sortByRecency(clips).map((clip) => (
            <ClipTile key={clip.id} clip={clip} onSelect={() => onSelectClip(clip)} />
          ))}
        </div>
      ) : null}

      {hasData && sortMode === 'type' ? (
        <div className="space-y-6">
          {grouped.map(([type, typeClips]) => (
            <div key={type}>
              <h3 className="mb-3 text-sm font-medium text-muted-foreground">
                {eventTypeLabel(type)} · {typeClips.length}건
              </h3>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {typeClips.map((clip) => (
                  <ClipTile key={clip.id} clip={clip} onSelect={() => onSelectClip(clip)} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

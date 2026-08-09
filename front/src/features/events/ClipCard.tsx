import { getEventTypeChipClassName, getEventTypeLabel } from '@/features/events/eventTypes';
import { formatClipTimestamp } from '@/features/events/formatters';
import type { Clip } from '@/shared/api/types';

type ClipCardProps = {
  clip: Clip;
  cameraLabel: string;
  onSelect: (clipId: string) => void;
};

/** 4-column grid card: 16:9 thumbnail (camera-label overlay, top-left) + event-type chip and timestamp below. */
export function ClipCard({ clip, cameraLabel, onSelect }: ClipCardProps): JSX.Element {
  return (
    <button
      type="button"
      className="flex flex-col overflow-hidden rounded-card border border-border bg-card text-left"
      onClick={() => onSelect(clip.id)}
    >
      <div className="event-media-frame relative">
        <span className="media-status-overlay absolute left-2 top-2 rounded-full px-2 py-0.5 text-xs font-semibold">
          {cameraLabel}
        </span>
        {clip.video_available ? (
          <div className="flex h-full items-center justify-center" data-testid="clip-thumbnail-available" aria-hidden="true">
            <svg className="h-8 w-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <path d="M8.5 6.75v10.5L17 12 8.5 6.75Z" strokeLinejoin="round" />
            </svg>
          </div>
        ) : (
          <div className="event-media-unavailable h-full px-3 text-center text-xs" data-testid="clip-thumbnail-unavailable">
            {clip.video_error ?? '영상을 사용할 수 없습니다.'}
          </div>
        )}
      </div>
      <div className="flex items-center justify-between gap-2 p-3">
        <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${getEventTypeChipClassName(clip.event_type)}`}>
          {getEventTypeLabel(clip.event_type)}
        </span>
        <span className="tabular-nums text-xs text-muted-foreground">{formatClipTimestamp(clip.created_at)}</span>
      </div>
    </button>
  );
}

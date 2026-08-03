import { useEffect, useState } from 'react';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { getEventTypeChipClassName, getEventTypeLabel } from '@/features/events/eventTypes';
import { formatClipTimestamp, formatDuration, formatResolution } from '@/features/events/formatters';
import { formatBytes } from '@/shared/format/bytes';
import type { Clip } from '@/shared/api/types';

type ClipPlaybackModalProps = {
  clip: Clip | null;
  cameraLabel: string;
  open: boolean;
  onClose: () => void;
};

type VideoMetadata = { duration: number; width: number; height: number };

/**
 * TP/FP labeling is removed per the redesign — this shows the event-type chip instead. `resolution`
 * is read off the native <video> element's loaded metadata (the API's Clip type has no dimensions
 * field); `길이` prefers the manifest's `duration_s` when the backend reports it, falling back to the
 * loaded metadata otherwise. The prototype's "파일" (full clip-store path) row is intentionally
 * omitted — AGENTS.md: "never read the clip-store path directly or expose camera credentials"; only
 * video_path (an API URL) is safe to show. `크기` is shown only when the manifest carries
 * `size_bytes` — never fabricated.
 */
export function ClipPlaybackModal({ clip, cameraLabel, open, onClose }: ClipPlaybackModalProps): JSX.Element | null {
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const clipId = clip?.id;

  useEffect(() => {
    setMetadata(null);
  }, [clipId]);

  if (!clip) return null;

  const title = `${getEventTypeLabel(clip.event_type)} · ${cameraLabel}`;

  const durationSeconds = clip.duration_s ?? (clip.video_available ? metadata?.duration ?? null : null);
  const sizeBytes = clip.size_bytes ?? null;

  return (
    <AccessibleDialog open={open} title={title} onClose={onClose} size="xl" initialFocus="heading">
      <span className={`mb-4 inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${getEventTypeChipClassName(clip.event_type)}`}>
        {getEventTypeLabel(clip.event_type)}
      </span>

      <div className="event-media-frame">
        {clip.video_available ? (
          <video
            className="h-full w-full"
            src={clip.video_path}
            controls
            preload="metadata"
            onLoadedMetadata={(event) => {
              const target = event.currentTarget;
              setMetadata({ duration: target.duration, width: target.videoWidth, height: target.videoHeight });
            }}
          />
        ) : (
          <div className="event-media-unavailable h-full px-4 text-center text-sm" data-testid="clip-modal-unavailable">
            {clip.video_error ?? '저장된 영상을 사용할 수 없습니다.'}
          </div>
        )}
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <dt className="text-muted-foreground">카메라</dt>
        <dd className="text-right">{cameraLabel}</dd>
        <dt className="text-muted-foreground">시간</dt>
        <dd className="text-right tabular-nums">{formatClipTimestamp(clip.created_at)}</dd>
        <dt className="text-muted-foreground">길이</dt>
        <dd className="text-right tabular-nums">{durationSeconds !== null ? formatDuration(durationSeconds) : '-'}</dd>
        <dt className="text-muted-foreground">해상도</dt>
        <dd className="text-right tabular-nums">{clip.video_available ? formatResolution(metadata?.width ?? null, metadata?.height ?? null) : '-'}</dd>
        {sizeBytes !== null ? (
          <>
            <dt className="text-muted-foreground">크기</dt>
            <dd className="text-right tabular-nums">{formatBytes(sizeBytes)}</dd>
          </>
        ) : null}
      </dl>

      <div className="dialog-actions mt-5">
        {clip.video_available ? (
          <a className="dialog-secondary-action" href={clip.video_path} download>다운로드</a>
        ) : null}
      </div>
    </AccessibleDialog>
  );
}

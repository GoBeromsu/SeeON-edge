import { useEffect, useState } from 'react';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { eventTypeLabel, formatClipDateTime } from '@/features/operations/operationsModel';
import type { Clip } from '@/shared/api/client';

type ClipPlayerModalProps = {
  clip: Clip | null;
  cameraLabel: string;
  onClose: () => void;
};

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${remainder.toString().padStart(2, '0')}`;
}

/**
 * The design mock's "파일" row shows a raw server filesystem path (e.g. /mnt/data/clips/...);
 * AGENTS.md forbids exposing the clip-store path directly, so that row is dropped. "길이"/"해상도"
 * are read from the loaded <video> element's real metadata instead of being fabricated, and "크기"
 * is omitted entirely since no API field carries file size.
 */
export function ClipPlayerModal({ clip, cameraLabel, onClose }: ClipPlayerModalProps): JSX.Element {
  const [metadata, setMetadata] = useState<{ duration: number; width: number; height: number } | null>(null);

  useEffect(() => {
    setMetadata(null);
  }, [clip?.id]);

  return (
    <AccessibleDialog
      open={clip !== null}
      title={clip ? `${eventTypeLabel(clip.event_type)} · ${cameraLabel}` : '클립 재생'}
      onClose={onClose}
    >
      {clip ? (
        <div className="flex flex-col gap-4">
          <p className="tabular-nums text-sm text-muted-foreground">{formatClipDateTime(clip.created_at)}</p>

          <div className="event-media-frame">
            {clip.video_available ? (
              <video
                key={clip.id}
                className="h-full w-full object-cover"
                src={clip.video_path}
                controls
                playsInline
                preload="metadata"
                onLoadedMetadata={(event) => {
                  const video = event.currentTarget;
                  setMetadata({ duration: video.duration, width: video.videoWidth, height: video.videoHeight });
                }}
              />
            ) : (
              <span className="event-media-unavailable">{clip.video_error ?? '영상을 사용할 수 없습니다.'}</span>
            )}
          </div>

          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-sm">
            <dt className="text-muted-foreground">길이</dt>
            <dd className="tabular-nums text-foreground">
              {metadata && Number.isFinite(metadata.duration) ? formatDuration(metadata.duration) : '측정 중'}
            </dd>
            <dt className="text-muted-foreground">해상도</dt>
            <dd className="tabular-nums text-foreground">
              {metadata && metadata.width > 0 ? `${metadata.width}×${metadata.height}` : '측정 중'}
            </dd>
          </dl>

          <div className="dialog-actions">
            <a
              href={clip.video_path}
              download
              className="brand-action rounded-lg px-6 py-2 text-sm font-bold"
              aria-disabled={!clip.video_available}
              onClick={(event) => {
                if (!clip.video_available) event.preventDefault();
              }}
            >
              다운로드
            </a>
          </div>
        </div>
      ) : null}
    </AccessibleDialog>
  );
}

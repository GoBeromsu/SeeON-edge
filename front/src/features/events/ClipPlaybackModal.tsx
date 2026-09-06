import { useEffect, useState } from 'react';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { getEventTypeChipClassName, getEventTypeLabel } from '@/features/events/eventTypes';
import { formatClipTimestamp, formatDuration, formatResolution } from '@/features/events/formatters';
import { formatBytes } from '@/shared/format/bytes';
import { AutoplayVideo } from '@/shared/ui/AutoplayVideo';
import { fetchClipArtifacts } from '@/shared/api/client';
import type { ClipArtifacts, Clip } from '@/shared/api/types';
import type { ClipMetadataStatus } from '@/features/events/useClipMetadata';

type Props = {
  clip: Clip | null;
  cameraLabel: string;
  open: boolean;
  onClose: () => void;
  lookupStatus: ClipMetadataStatus;
  onRetry: () => void;
};
type VideoMetadata = { duration: number; width: number; height: number };

const artifactCopy = (state: string | null | undefined): string => ({ PENDING: '준비 중', AVAILABLE: '사용 가능', UNAVAILABLE: '사용 불가', CORRUPT: '손상됨', PURGED: '삭제됨' }[state ?? ''] ?? '확인 중');

/** Privacy-bounded evidence detail: only API-projected identities/states are shown, never paths or credentials. */
export function ClipPlaybackModal({
  clip, cameraLabel, open, onClose, lookupStatus, onRetry,
}: Props): JSX.Element | null {
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [artifacts, setArtifacts] = useState<ClipArtifacts | null>(null);
  const clipId = clip?.id;

  useEffect(() => {
    setMetadata(null);
    setArtifacts(null);
    if (!clipId || !open) return;
    let active = true;
    void fetchClipArtifacts(clipId).then((next) => {
      if (active) setArtifacts(next);
    }).catch(() => {
      if (active) setArtifacts(null);
    });
    return () => { active = false; };
  }, [clipId, open]);

  if (!clip) {
    if (!open || (lookupStatus !== 'loading' && lookupStatus !== 'error')) return null;
    return <AccessibleDialog open title="이벤트 확인" onClose={onClose} size="xl" initialFocus="heading">{lookupStatus === 'loading' ? <p role="status">이벤트 정보를 불러오는 중입니다.</p> : <div role="alert"><p>이벤트 정보를 불러오지 못했습니다.</p><button type="button" className="dialog-secondary-action" onClick={onRetry}>다시 시도</button></div>}</AccessibleDialog>;
  }
  const title = `${getEventTypeLabel(clip.event_type)} · ${cameraLabel}`;
  const durationSeconds = clip.duration_s ?? (clip.video_available ? metadata?.duration ?? null : null);
  return <AccessibleDialog open={open} title={title} onClose={onClose} size="xl" initialFocus="heading">
    <span className={`mb-4 inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${getEventTypeChipClassName(clip.event_type)}`}>{getEventTypeLabel(clip.event_type)}</span>
    {artifacts ? (
      <p className="mt-3 text-sm text-muted-foreground" role="status" data-testid="clip-artifact-status">원본 {artifactCopy(artifacts.clean)}{artifacts.snapshot ? ` · 스냅샷 ${artifactCopy(artifacts.snapshot)}` : ''}</p>
    ) : <p className="mt-3 text-sm text-muted-foreground" data-testid="clip-artifact-status">증거 상태를 확인하지 못했습니다.</p>}
    <div className="event-media-frame relative mt-4">{clip.video_available ? <AutoplayVideo key={clip.video_path} src={clip.video_path} className="h-full w-full" onLoadedMetadata={(nextVideo) => { setMetadata({ duration: nextVideo.duration, width: nextVideo.videoWidth, height: nextVideo.videoHeight }); }} /> : <div className="event-media-unavailable h-full px-4 text-center text-sm" data-testid="clip-modal-unavailable">{clip.video_error ?? '저장된 영상을 사용할 수 없습니다.'}</div>}</div>
    <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm"><dt className="text-muted-foreground">카메라</dt><dd className="text-right">{cameraLabel}</dd><dt className="text-muted-foreground">시간</dt><dd className="text-right tabular-nums">{formatClipTimestamp(clip.detected_at ?? clip.created_at)}</dd><dt className="text-muted-foreground">길이</dt><dd className="text-right tabular-nums">{durationSeconds !== null ? formatDuration(durationSeconds) : '-'}</dd><dt className="text-muted-foreground">해상도</dt><dd className="text-right tabular-nums">{clip.video_available ? formatResolution(metadata?.width ?? null, metadata?.height ?? null) : '-'}</dd>{clip.size_bytes !== null && clip.size_bytes !== undefined ? <><dt className="text-muted-foreground">크기</dt><dd className="text-right tabular-nums">{formatBytes(clip.size_bytes)}</dd></> : null}</dl>
  </AccessibleDialog>;
}

import { useEffect, useRef, useState } from 'react';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { getEventTypeChipClassName, getEventTypeLabel } from '@/features/events/eventTypes';
import { formatClipTimestamp, formatDuration, formatResolution } from '@/features/events/formatters';
import { formatBytes } from '@/shared/format/bytes';
import { AutoplayVideo } from '@/shared/ui/AutoplayVideo';
import { controlClipDerivative, deleteClip, fetchClipAnalysis, fetchClipArtifacts } from '@/shared/api/client';
import type { ClipAnalysis, ClipArtifacts, ClipDeleteStatus, ClipDerivative, Clip } from '@/shared/api/types';
import type { ClipMetadataStatus } from '@/features/events/useClipMetadata';

type Props = {
  clip: Clip | null;
  cameraLabel: string;
  open: boolean;
  onClose: () => void;
  lookupStatus: ClipMetadataStatus;
  onRetry: () => void;
  /** Fires when the operator submits an exact-match delete so the parent can stop metadata validation immediately. */
  onDeleteStarted?: (clip: Clip) => void;
  /** Fires only for accepted PURGED deletion. */
  onDeleted: (clip: Clip) => void;
  /** Fires when a started delete ends without PURGED (HELD/terminal/transport) so suppression can clear. */
  onDeleteRejected?: (clip: Clip) => void;
};
type VideoMetadata = { duration: number; width: number; height: number };

const artifactCopy = (state: string | null | undefined): string => ({ AVAILABLE: '사용 가능', MISSING: '누락', UNAVAILABLE: '사용 불가', CORRUPT: '손상됨', TRUNCATED: '잘림', QUEUED: '대기 중', RUNNING: '생성 중', CANCELLED: '취소됨', NOT_REQUESTED: '미요청' }[state ?? ''] ?? '확인 중');

const DELETE_ACCEPTED_STATES = new Set<ClipDeleteStatus>(['PURGED']);

const deleteStatusCopy = (deleteStatus: ClipDeleteStatus): string => ({
  PURGED: '클립을 삭제했습니다.',
  HELD: '다른 처리(파생 증거 생성 또는 인시던트 처리)가 끝난 뒤 다시 시도하세요.',
  MISSING: '클립을 찾을 수 없습니다.',
  UNVERIFIABLE: '클립 상태를 확인할 수 없어 삭제하지 못했습니다.',
  DELETE_FAILED: '클립 삭제에 실패했습니다. 잠시 후 다시 시도하세요.',
  VERIFICATION_FAILED: '삭제 확인에 실패했습니다. 잠시 후 다시 시도하세요.',
})[deleteStatus];

/** Privacy-bounded evidence detail: only API-projected identities/states are shown, never paths or credentials. */
export function ClipPlaybackModal({
  clip, cameraLabel, open, onClose, lookupStatus, onRetry, onDeleteStarted, onDeleted, onDeleteRejected,
}: Props): JSX.Element | null {
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [artifacts, setArtifacts] = useState<ClipArtifacts | null>(null);
  const [analysis, setAnalysis] = useState<ClipAnalysis | null>(null);
  const [derivatives, setDerivatives] = useState<Partial<Record<'still' | 'video', ClipDerivative>>>({});
  const [view, setView] = useState<'clean' | 'annotated'>('clean');
  const [busy, setBusy] = useState<'still' | 'video' | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [confirmInput, setConfirmInput] = useState('');
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteStatus, setDeleteStatus] = useState<ClipDeleteStatus | null>(null);
  const [deleteRequestFailed, setDeleteRequestFailed] = useState(false);
  const clipId = clip?.id;
  // Guards stale fetchClipArtifacts/fetchClipAnalysis responses that resolve after an accepted
  // deletion from silently re-enabling playback (an old response for the same clip_id arriving
  // late must never resurrect a clip the operator just deleted).
  const deletedRef = useRef(false);
  const generationRef = useRef(0);
  const currentClipIdRef = useRef<string | undefined>(clipId);
  currentClipIdRef.current = clipId;
  const isDeleted = deleteStatus !== null && DELETE_ACCEPTED_STATES.has(deleteStatus);

  useEffect(() => {
    const generation = ++generationRef.current;
    setMetadata(null); setArtifacts(null); setAnalysis(null); setDerivatives({}); setView('clean');
    setDeleteDialogOpen(false); setConfirmInput(''); setDeleteBusy(false); setDeleteStatus(null); setDeleteRequestFailed(false);
    deletedRef.current = false;
    if (!clipId || !open) return;
    void fetchClipArtifacts(clipId).then((next) => {
      if (generationRef.current !== generation || deletedRef.current) return;
      if (next.playback_view === 'clean' || next.playback_view === 'annotated') { setArtifacts(next); setView(next.playback_view); }
      if (next.analysis !== 'AVAILABLE') return;
      void fetchClipAnalysis(clipId).then((analysis) => {
        if (generationRef.current === generation && !deletedRef.current) setAnalysis(analysis);
      }).catch(() => { if (generationRef.current === generation && !deletedRef.current) setAnalysis(null); });
    }).catch(() => { if (generationRef.current === generation && !deletedRef.current) setArtifacts(null); });
  }, [clipId, open]);

  const confirmDelete = async (): Promise<void> => {
    if (!clip || confirmInput !== clip.id || deleteBusy) return;
    const requestedClip = clip;
    const generation = generationRef.current;
    setDeleteBusy(true);
    setDeleteRequestFailed(false);
    // Stop parent metadata validation before the worker responds; list convergence can drop the
    // card while DELETE is in flight and must not issue a stale /metadata 404 for this clip.
    onDeleteStarted?.(requestedClip);
    try {
      const result = await deleteClip(requestedClip.id, confirmInput);
      if (generationRef.current !== generation || currentClipIdRef.current !== requestedClip.id) return;
      setDeleteStatus(result.status);
      setDeleteDialogOpen(false);
      if (DELETE_ACCEPTED_STATES.has(result.status)) {
        deletedRef.current = true;
        onDeleted(requestedClip);
      } else {
        onDeleteRejected?.(requestedClip);
      }
    } catch {
      if (generationRef.current === generation && currentClipIdRef.current === requestedClip.id) {
        setDeleteRequestFailed(true);
        onDeleteRejected?.(requestedClip);
      }
    } finally {
      if (generationRef.current === generation && currentClipIdRef.current === requestedClip.id) setDeleteBusy(false);
    }
  };

  if (!clip) {
    if (!open || (lookupStatus !== 'loading' && lookupStatus !== 'error')) return null;
    return <AccessibleDialog open title="이벤트 확인" onClose={onClose} size="xl" initialFocus="heading">{lookupStatus === 'loading' ? <p role="status">이벤트 정보를 불러오는 중입니다.</p> : <div role="alert"><p>이벤트 정보를 불러오지 못했습니다.</p><button type="button" className="dialog-secondary-action" onClick={onRetry}>다시 시도</button></div>}</AccessibleDialog>;
  }
  const selectDerivative = async (kind: 'still' | 'video', action: 'request' | 'cancel'): Promise<void> => {
    setBusy(kind);
    try { const next = await controlClipDerivative(clip.id, kind, action); setDerivatives((current) => ({ ...current, [kind]: next })); }
    finally { setBusy(null); }
  };
  const title = `${getEventTypeLabel(clip.event_type)} · ${cameraLabel}`;
  const durationSeconds = clip.duration_s ?? (clip.video_available ? metadata?.duration ?? null : null);
  const videoUrl = `${clip.video_path}${view === 'annotated' ? '?view=annotated' : ''}`;
  const annotatedReady = artifacts?.annotated === 'AVAILABLE';
  return <AccessibleDialog open={open} title={title} onClose={() => { if (!deleteDialogOpen && !deleteBusy) onClose(); }} size="xl" initialFocus="heading">
    <span className={`mb-4 inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${getEventTypeChipClassName(clip.event_type)}`}>{getEventTypeLabel(clip.event_type)}</span>
    <div className="flex flex-wrap items-center justify-between gap-2">
      {!isDeleted ? <div className="flex flex-wrap gap-2" role="group" aria-label="증거 보기 선택">
        <button type="button" className="dialog-secondary-action" aria-pressed={view === 'clean'} onClick={() => setView('clean')}>원본 영상</button>
        <button type="button" className="dialog-secondary-action" aria-pressed={view === 'annotated'} disabled={!annotatedReady} onClick={() => setView('annotated')}>주석 영상</button>
      </div> : null}
      {!isDeleted ? (
        <button
          type="button"
          className="inline-flex h-9 items-center justify-center rounded-control border border-destructive px-3 text-sm font-semibold text-destructive disabled:opacity-60"
          disabled={deleteBusy}
          onClick={() => { setDeleteRequestFailed(false); setDeleteDialogOpen(true); }}
        >
          클립 삭제
        </button>
      ) : null}
    </div>
    {deleteStatus !== null ? (
      <p className={`mt-3 text-sm ${isDeleted ? 'text-status-pendingFg' : 'text-destructive'}`} role={isDeleted ? 'status' : 'alert'} data-testid="clip-delete-status">
        {deleteStatusCopy(deleteStatus)}
      </p>
    ) : null}
    {deleteRequestFailed ? <p className="mt-3 text-sm text-destructive" role="alert">클립 삭제 요청을 보내지 못했습니다. 연결을 확인한 뒤 다시 시도하세요.</p> : null}
    {isDeleted ? (
      <p className="mt-3 text-sm text-muted-foreground" role="status">원본 {artifactCopy(artifacts?.clean)} · 분석 {artifactCopy(artifacts?.analysis)} · 주석 {artifactCopy(artifacts?.annotated)}</p>
    ) : artifacts ? <p className="mt-3 text-sm text-muted-foreground" role="status">원본 {artifactCopy(artifacts.clean)} · 분석 {artifactCopy(artifacts.analysis)} · 주석 {artifactCopy(artifacts.annotated)}{artifacts.annotated_fallback_to_clean ? ' · 주석 요청은 원본으로 대체됩니다.' : ''}</p> : <p className="mt-3 text-sm text-muted-foreground">증거 상태를 확인하지 못했습니다.</p>}
    <div className="event-media-frame relative mt-4">{!isDeleted && clip.video_available && (view === 'clean' || annotatedReady) ? <AutoplayVideo key={videoUrl} src={videoUrl} className="h-full w-full" onLoadedMetadata={(video) => setMetadata({ duration: video.duration, width: video.videoWidth, height: video.videoHeight })} /> : <div className="event-media-unavailable h-full px-4 text-center text-sm" data-testid="clip-modal-unavailable">{isDeleted ? '이 클립은 삭제되어 재생할 수 없습니다.' : (clip.video_error ?? `선택한 ${view === 'annotated' ? '주석' : '원본'} 증거를 사용할 수 없습니다.`)}</div>}</div>
    <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm"><dt className="text-muted-foreground">카메라</dt><dd className="text-right">{cameraLabel}</dd><dt className="text-muted-foreground">시간</dt><dd className="text-right tabular-nums">{formatClipTimestamp(clip.created_at)}</dd><dt className="text-muted-foreground">길이</dt><dd className="text-right tabular-nums">{durationSeconds !== null ? formatDuration(durationSeconds) : '-'}</dd><dt className="text-muted-foreground">해상도</dt><dd className="text-right tabular-nums">{clip.video_available ? formatResolution(metadata?.width ?? null, metadata?.height ?? null) : '-'}</dd>{clip.size_bytes !== null && clip.size_bytes !== undefined ? <><dt className="text-muted-foreground">크기</dt><dd className="text-right tabular-nums">{formatBytes(clip.size_bytes)}</dd></> : null}</dl>
    {analysis ? <section className="mt-4 border-t border-border pt-4" aria-label="적용 실행 증명"><h3 className="text-sm font-semibold">적용 실행 증명</h3><p className="mt-2 break-all text-xs text-muted-foreground">{analysis.module_qualified_id} · {analysis.policy_qualified_id} · {analysis.effective_policy_id}</p><p className="mt-1 text-sm">트리거: {analysis.triggered ? '예' : '아니오'} · {analysis.previous_state} → {analysis.current_state} · {analysis.reason}</p><p className="mt-1 break-all text-xs text-muted-foreground">런타임 {analysis.runtime_manifest_sha256} · 분석 {analysis.decision_trace_id}</p></section> : null}
    <AccessibleDialog
      open={deleteDialogOpen}
      title="클립 삭제 확인"
      onClose={() => { if (!deleteBusy) setDeleteDialogOpen(false); }}
    >
      <p className="text-sm text-foreground">이 작업은 되돌릴 수 없습니다. 삭제를 확인하려면 아래 클립 ID를 정확히 입력하세요.</p>
      <p className="mt-2 break-all rounded-control bg-muted px-3 py-2 text-sm font-mono" data-testid="delete-confirm-clip-id">{clip.id}</p>
      <label className="mt-3 block text-sm" htmlFor="delete-confirm-input">클립 ID 확인</label>
      <input
        id="delete-confirm-input"
        type="text"
        className="mt-1 w-full rounded-control border border-border bg-background px-3 py-2 text-sm"
        value={confirmInput}
        disabled={deleteBusy}
        autoComplete="off"
        spellCheck={false}
        onChange={(event) => setConfirmInput(event.target.value)}
      />
      {deleteRequestFailed ? <p className="mt-2 text-sm text-destructive" role="alert">클립 삭제 요청을 보내지 못했습니다. 다시 시도하세요.</p> : null}
      <div className="dialog-actions mt-4">
        <button type="button" className="dialog-secondary-action" disabled={deleteBusy} onClick={() => setDeleteDialogOpen(false)}>취소</button>
        <button
          type="button"
          className="inline-flex h-9 items-center justify-center rounded-control bg-destructive px-4 text-sm font-semibold text-destructive-foreground disabled:opacity-60"
          disabled={deleteBusy || confirmInput !== clip.id}
          onClick={() => void confirmDelete()}
        >
          {deleteBusy ? '삭제 중...' : '삭제'}
        </button>
      </div>
    </AccessibleDialog>
    {!isDeleted ? <section className="mt-4 border-t border-border pt-4" aria-label="파생 증거 제어"><h3 className="text-sm font-semibold">파생 증거</h3>{(['still', 'video'] as const).map((kind) => { const value = derivatives[kind]; const cancellable = value?.state === 'QUEUED' || value?.state === 'RUNNING'; return <div className="mt-2 flex flex-wrap items-center gap-2 text-sm" key={kind}><span>{kind === 'still' ? '주석 스틸' : '주석 영상'}: {artifactCopy(value?.state ?? (kind === 'video' ? artifacts?.annotated : 'NOT_REQUESTED'))}</span><button type="button" className="dialog-secondary-action" disabled={busy !== null} onClick={() => void selectDerivative(kind, 'request')}>{value?.state === 'CANCELLED' || value?.state === 'UNAVAILABLE' || value?.state === 'CORRUPT' ? '다시 요청' : '요청'}</button>{cancellable ? <button type="button" className="dialog-secondary-action" disabled={busy !== null} onClick={() => void selectDerivative(kind, 'cancel')}>취소</button> : null}{value?.reason ? <span className="text-destructive">{value.reason}</span> : null}</div>; })}</section> : null}
  </AccessibleDialog>;
}

import { useRef, useState } from 'react';
import { confirmTopologyPreview, fetchTopologyPreview } from '@/shared/api/topologyClient';
import type { TopologyPreview } from '@/shared/api/topologyClient';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { HttpError } from '@/shared/api/http';

type Props = {
  readonly open: boolean;
  readonly preview: TopologyPreview | null;
  readonly onClose: () => void;
  readonly onConfirmed: (serverRevision: number) => void;
  readonly onChanged: (preview: TopologyPreview | null) => void;
};

function isSamePreview(left: TopologyPreview, right: TopologyPreview): boolean {
  return left.confirmation_id === right.confirmation_id
    && left.digest === right.digest
    && left.snapshot_id === right.snapshot_id
    && left.client_revision === right.client_revision
    && left.server_revision === right.server_revision
    && left.expires_at === right.expires_at;
}

function isExpired(preview: TopologyPreview): boolean {
  return Date.parse(preview.expires_at) <= Date.now();
}

/** Mirrors ConnectionSettingsPanel's saveFailure() discipline: distinguish the failure reasons the
 * backend can actually return from `POST /connection/topology-preview/confirm` (see
 * backend/app/features/connection/topology_confirmation_router.py) instead of one generic message. */
function confirmFailure(error: unknown): string {
  if (!(error instanceof HttpError)) return '제외 확정에 실패했습니다. 미리보기를 새로 확인한 뒤 다시 시도하세요.';
  if (error.status === 401) return '등록 인증이 만료되었습니다. 1단계 장치 연결을 다시 확인하세요.';
  if (error.status === 403) return '이 시설에 반영할 권한이 없습니다. 관리자에게 문의하세요.';
  if (error.status === 409) return '서버 상태가 그 사이 바뀌었습니다. 미리보기를 새로 확인한 뒤 다시 시도하세요.';
  if (error.status === 410) return '확인 기한이 지났습니다. 다시 동기화해 새 미리보기를 받으세요.';
  if (error.status === 503) return '서버에 연결할 수 없습니다. 네트워크를 확인한 뒤 다시 시도하세요.';
  return '제외 확정에 실패했습니다. 미리보기를 새로 확인한 뒤 다시 시도하세요.';
}

export function TopologyConfirmationDialog({ open, preview, onClose, onConfirmed, onChanged }: Props): JSX.Element | null {
  const busyRef = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function confirm(): Promise<void> {
    if (!preview || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const current = (await fetchTopologyPreview()).preview;
      if (!current || !isSamePreview(preview, current)) {
        onChanged(current);
        setError('제외 미리보기가 변경되었습니다. 내용을 다시 확인하세요.');
        return;
      }
      if (isExpired(current)) {
        setError('확인 기한이 지났습니다. 다시 동기화해 새 미리보기를 받으세요.');
        return;
      }
      const result = await confirmTopologyPreview({
        confirmation_id: current.confirmation_id,
        digest: current.digest,
        client_revision: current.client_revision,
        server_revision: current.server_revision,
      });
      onConfirmed(result.server_revision);
      onClose();
    } catch (error) {
      setError(confirmFailure(error));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  const expired = preview ? isExpired(preview) : true;
  return (
    <AccessibleDialog open={open} title="변경 제외 확정" size="sm" initialFocus="heading" onClose={() => { if (!busyRef.current) onClose(); }}>
      {preview ? (
        <div className="space-y-4 text-sm">
          <p className="text-muted-foreground">아래 항목은 삭제되지 않고 비활성화됩니다. 이 작업은 기사님의 명시적 확인 후에만 실행됩니다.</p>
          <dl className="grid grid-cols-3 gap-3 rounded-control bg-muted p-4">
            <div><dt className="text-muted-foreground">카메라</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{preview.cameras}대</dd></div>
            <div><dt className="text-muted-foreground">방</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{preview.rooms}개</dd></div>
            <div><dt className="text-muted-foreground">층</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{preview.floors}개</dd></div>
          </dl>
          {/* server_revision은 지원팀 진단용 값이라 현장 화면에서는 기본으로 숨긴다
              (브리프: raw internal label을 그대로 노출하지 말 것 -- "서버 리비전" 포함). */}
          <details className="text-sm">
            <summary className="cursor-pointer text-muted-foreground">자세히</summary>
            <dl className="mt-2 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2">
              <dt className="text-muted-foreground">서버 리비전</dt><dd className="tabular-nums">{preview.server_revision}</dd>
            </dl>
          </details>
          {expired ? <p role="alert" className="auth-error">확인 기한이 지났습니다. 다시 동기화하세요.</p> : null}
          {error ? <p role="alert" className="auth-error">{error}</p> : null}
          <div className="dialog-actions">
            <button type="button" className="dialog-secondary-action !h-11 sm:!h-9" disabled={busy} onClick={onClose}>취소</button>
            <button type="button" className="inline-flex h-11 items-center justify-center rounded-control bg-destructive px-4 text-sm font-semibold text-destructive-foreground disabled:opacity-50 sm:h-9" disabled={busy || expired} onClick={() => void confirm()}>{busy ? '확정 중...' : '제외 확정'}</button>
          </div>
        </div>
      ) : <p role="alert" className="auth-error">확인할 제외 미리보기가 없습니다.</p>}
    </AccessibleDialog>
  );
}

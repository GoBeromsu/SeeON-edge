import type { CameraStatus, ConnectionView, SystemSnapshot } from '@/shared/api/client';

/** The four semantic status tones defined in front/src/styles/tokens-base.css (approved/rejected/pending/closed). */
export type StatusVariant = 'approved' | 'rejected' | 'pending' | 'closed';

const variantClassName: Record<StatusVariant, string> = {
  approved: 'bg-status-approvedBg text-status-approvedFg',
  rejected: 'bg-status-rejectedBg text-status-rejectedFg',
  pending: 'bg-status-pendingBg text-status-pendingFg',
  closed: 'bg-status-closedBg text-status-closedFg',
};

export function statusBadgeClassName(variant: StatusVariant): string {
  return `inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${variantClassName[variant]}`;
}

const cameraStatusMeta: Record<CameraStatus, { label: string; variant: StatusVariant }> = {
  online: { label: '온라인', variant: 'approved' },
  offline: { label: '오프라인', variant: 'rejected' },
  starting: { label: '시작 중', variant: 'pending' },
  unknown: { label: '확인 중', variant: 'closed' },
};

export function getCameraStatusMeta(status: CameraStatus): { label: string; className: string; variant: StatusVariant } {
  const meta = cameraStatusMeta[status];
  return { ...meta, className: statusBadgeClassName(meta.variant) };
}

type StatusBadgeProps = {
  status: CameraStatus;
};

export function StatusBadge({ status }: StatusBadgeProps): JSX.Element {
  const meta = getCameraStatusMeta(status);

  return (
    <span className={meta.className}>
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
      {meta.label}
    </span>
  );
}

function backendStatusMeta(configured: boolean, reachable: boolean | null): { label: string; variant: StatusVariant } {
  if (!configured) {
    return { label: '백엔드 미설정', variant: 'pending' };
  }
  if (reachable === true) {
    return { label: '백엔드 연결됨', variant: 'approved' };
  }
  if (reachable === false) {
    return { label: '백엔드 연결 실패', variant: 'rejected' };
  }
  return { label: '백엔드 대기 중', variant: 'closed' };
}

export function getBackendStatus(system: SystemSnapshot | null): { label: string; className: string } {
  if (!system) {
    return { label: '백엔드 확인 중', className: statusBadgeClassName('closed') };
  }
  const meta = backendStatusMeta(system.backend.configured, system.backend.reachable);
  return { label: meta.label, className: statusBadgeClassName(meta.variant) };
}

/** Same {configured, reachable} contract as getBackendStatus, but with the 설정 페이지 서버 연결 카드's own wording ("정상"/"연결 실패"/"미설정"). */
export function getConnectionStatus(connection: ConnectionView | null): { label: string; className: string } {
  if (!connection) {
    return { label: '확인 중', className: statusBadgeClassName('closed') };
  }
  if (!connection.configured) {
    return { label: '미설정', className: statusBadgeClassName('pending') };
  }
  if (connection.reachable === true) {
    return { label: '정상', className: statusBadgeClassName('approved') };
  }
  if (connection.reachable === false) {
    return { label: '연결 실패', className: statusBadgeClassName('rejected') };
  }
  return { label: '확인 중', className: statusBadgeClassName('closed') };
}

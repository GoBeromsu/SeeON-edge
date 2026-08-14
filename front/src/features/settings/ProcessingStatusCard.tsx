import { statusBadgeClassName } from '@/shared/ui/StatusBadge';
import type { StatusSnapshot } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

type ProcessingStatusCardProps = {
  resource: PollingResource<StatusSnapshot>;
};

const UNKNOWN = '확인 중';

function deviceLabel(status: StatusSnapshot): string {
  const device = status.runtime.device;
  if (!device?.device_name && !device?.backend) return UNKNOWN;
  return [device.device_name, device.backend].filter(Boolean).join(' · ') || UNKNOWN;
}

/** No single global "decode backend" field exists yet — this takes the first camera's runtime decode diagnostics as a representative sample. */
function decodeLabel(status: StatusSnapshot): string {
  const first = Object.values(status.runtime.cameras)[0];
  return first?.decode.selected ?? first?.decode.requested ?? UNKNOWN;
}

function encodeLabel(status: StatusSnapshot): string {
  return status.runtime.clip_recorder?.encoder ?? UNKNOWN;
}

function latencyLabel(status: StatusSnapshot): string {
  const maxSec = Object.values(status.runtime.cameras)
    .map((camera) => camera.latency?.max_sec)
    .filter((value): value is number => typeof value === 'number');
  if (maxSec.length === 0) return UNKNOWN;
  return `최대 ${Math.max(...maxSec).toFixed(2)}초`;
}

function clipExportAppliedLabel(status: StatusSnapshot): string {
  const applied = status.runtime.clip_export_applied;
  if (applied.enabled === null || applied.version === null || applied.freshness === 'unknown') {
    return `워커 적용: ${UNKNOWN}`;
  }
  const freshness = applied.freshness === 'stale'
    ? ' · 상태 지연'
    : applied.freshness === 'offline'
      ? ' · 워커 오프라인'
      : '';
  return `워커 적용: ${applied.enabled ? 'ON' : 'OFF'} · 버전 ${applied.version}${freshness}`;
}

function workerBadge(status: StatusSnapshot | null): { label: string; className: string } {
  const alive = status?.runtime.worker?.alive;
  if (alive === true) return { label: '정상', className: statusBadgeClassName('approved') };
  if (alive === false) return { label: '중단됨', className: statusBadgeClassName('rejected') };
  return { label: UNKNOWN, className: statusBadgeClassName('closed') };
}

/**
 * 설정 페이지 "처리 상태" 카드 — 디바이스에 상관없이 backend가 보고하는 값을 그대로 보여준다
 * (CUDA 전용 가정 없음, front/design-handoff/README.md §5).
 */
export function ProcessingStatusCard({ resource }: ProcessingStatusCardProps): JSX.Element {
  if (resource.status === 'loading' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <p className="text-sm text-muted-foreground">처리 상태를 불러오는 중입니다...</p>
      </article>
    );
  }

  if (resource.status === 'error' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <h2 className="text-base font-semibold text-foreground">처리 상태</h2>
        <p className="mt-2 text-sm text-destructive">처리 상태를 불러오지 못했습니다.</p>
        <button type="button" className="dialog-secondary-action mt-2" onClick={() => resource.retry()}>
          다시 시도
        </button>
      </article>
    );
  }

  const status = resource.data;
  const badge = workerBadge(status);

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">처리 상태</h2>
        <span className={badge.className}>
          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
          {badge.label}
        </span>
      </div>

      <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        <dt className="text-muted-foreground">실행 디바이스</dt>
        <dd className="text-right text-foreground">{status ? deviceLabel(status) : UNKNOWN}</dd>
        <dt className="text-muted-foreground">디코드</dt>
        <dd className="text-right text-foreground">{status ? decodeLabel(status) : UNKNOWN}</dd>
        <dt className="text-muted-foreground">인코드</dt>
        <dd className="text-right text-foreground">{status ? encodeLabel(status) : UNKNOWN}</dd>
        <dt className="text-muted-foreground">클립 내보내기</dt>
        <dd className="text-right text-foreground">{status ? clipExportAppliedLabel(status) : `워커 적용: ${UNKNOWN}`}</dd>
        <dt className="text-muted-foreground">전송 지연</dt>
        <dd className="text-right tabular-nums text-foreground">{status ? latencyLabel(status) : UNKNOWN}</dd>
      </dl>
    </article>
  );
}

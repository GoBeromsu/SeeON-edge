import { statusBadgeClassName } from '@/shared/ui/StatusBadge';
import type { StatusSnapshot } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

type ProcessingStatusCardProps = {
  resource: PollingResource<StatusSnapshot>;
};

const UNKNOWN = '확인 중';

/** Hub가 이보다 오래 EVENT를 못 가져가면 "전송 지연" 대신 Hub 미전달로 경고한다 (issue: 9일간 5xx 무경고 적체). */
const HUB_STALL_THRESHOLD_MS = 5 * 60 * 1000;

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

/** Live(미확정) EVENT 대기 수. by_kind는 delivery_queue가 있으면 항상 온다 -- accepted_count는 EVENT/CLIP 등 전체 kind를 합친 값이라 대체 지표로 쓰면 과다 집계된다. */
function alertQueueCountLabel(status: StatusSnapshot): string {
  const queue = status.runtime.delivery_queue;
  if (!queue) return UNKNOWN;
  const eventCount = queue.by_kind?.EVENT ?? 0;
  const deadLettered = queue.dead_lettered_count;
  const suffix = deadLettered ? ` (실패 ${deadLettered})` : '';
  return `${eventCount}건 대기${suffix}`;
}

/** Hub가 HUB_STALL_THRESHOLD_MS 넘게 가장 오래된 EVENT를 못 가져가면 경고 문구를 낸다. */
function alertQueueWarning(status: StatusSnapshot): string | null {
  const acceptedAt = status.runtime.delivery_queue?.oldest_event_accepted_at;
  if (!acceptedAt) return null;
  const ageMs = Date.now() - new Date(acceptedAt).getTime();
  if (!Number.isFinite(ageMs) || ageMs < HUB_STALL_THRESHOLD_MS) return null;
  const ageMinutes = Math.floor(ageMs / 60_000);
  const age = ageMinutes < 60 ? `${ageMinutes}분` : `${Math.floor(ageMinutes / 60)}시간`;
  return `Hub 미전달 ${age}`;
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
        <dt className="text-muted-foreground">알림 전송</dt>
        <dd className="text-right text-foreground">
          {status ? alertQueueCountLabel(status) : UNKNOWN}
          {status && alertQueueWarning(status) ? (
            <span className="ml-2 text-destructive">{alertQueueWarning(status)}</span>
          ) : null}
        </dd>
      </dl>
    </article>
  );
}

import { useDetectionSettingsResource } from '@/shared/api/usePollingResource';
import type { DetectionDomainKey } from '@/shared/api/client';
import { OverlayModeControl } from '@/features/operations/OverlayModeControl';

type DetectionSettingsCardProps = {
  cameraId: string;
};

const DOMAIN_LABELS: Record<DetectionDomainKey, string> = {
  fall: '낙상 탐지',
  bed_exit: '침대 이탈 탐지',
};

/**
 * GET /detection-settings is a single global config shared by every camera (not per-camera), so
 * this card displays it read-only with a note to that effect rather than implying a per-room
 * toggle; editing the schedule is Settings-page territory (not built this wave).
 */
export function DetectionSettingsCard({ cameraId }: DetectionSettingsCardProps): JSX.Element {
  const { status, data, retry } = useDetectionSettingsResource(true);

  return (
    <article className="rounded-card border border-border bg-card p-4">
      <h2 className="mb-1 text-base font-semibold text-foreground">탐지 이벤트</h2>
      <p className="mb-3 text-xs text-muted-foreground">모든 카메라에 공통으로 적용되는 설정입니다.</p>

      {status === 'loading' ? (
        <p role="status" className="text-sm text-muted-foreground">탐지 설정을 불러오는 중입니다…</p>
      ) : null}
      {status === 'error' ? (
        <div role="alert" className="flex items-center justify-between gap-2 text-sm text-destructive">
          <span>탐지 설정을 불러오지 못했습니다.</span>
          <button type="button" onClick={retry} className="min-h-11 shrink-0 rounded-control border border-border px-3 text-xs font-semibold text-foreground">
            다시 시도
          </button>
        </div>
      ) : null}
      {status === 'success' && data ? (
        <ul className="space-y-2 text-sm">
          {(Object.keys(DOMAIN_LABELS) as DetectionDomainKey[]).map((key) => {
            const domain = data.domains[key];
            return (
              <li key={key} className="flex items-center justify-between gap-2">
                <span className="text-foreground">{DOMAIN_LABELS[key]}</span>
                <span className="text-muted-foreground">
                  {domain?.on
                    ? domain.mode === 'window' && domain.start && domain.end
                      ? `사용 중 · ${domain.start}–${domain.end}`
                      : '사용 중 · 상시'
                    : '사용 안 함'}
                </span>
              </li>
            );
          })}
        </ul>
      ) : null}

      <OverlayModeControl cameraId={cameraId} />
    </article>
  );
}

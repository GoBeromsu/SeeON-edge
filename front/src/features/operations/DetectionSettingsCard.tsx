import { useDetectionSettingsResource } from '@/shared/api/usePollingResource';
import { DOMAIN_LABELS, DOMAIN_ORDER, formatDomainSchedule } from '@/features/settings/detectionSettingsForm';
import { navigateToPage } from '@/features/operations/crossPageNavigation';
import { statusBadgeClassName } from '@/shared/ui/StatusBadge';
import { OverlayModeControl } from '@/features/operations/OverlayModeControl';
import type { Camera, DetectionDomainKey, DetectionDomainSetting, OverlayMode } from '@/shared/api/client';

type DetectionSettingsCardProps = {
  camera: Camera;
  onOverlayModeChange?: (mode: OverlayMode | null) => void;
  onRecognizeBedZone?: () => void;
};

function GearIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <circle cx="12" cy="12" r="2.75" />
      <path d="M12 3.75v2.1M12 18.15v2.1M20.25 12h-2.1M5.85 12h-2.1M17.66 6.34l-1.49 1.49M7.83 16.17l-1.49 1.49M17.66 17.66l-1.49-1.49M7.83 7.83 6.34 6.34" />
    </svg>
  );
}

/**
 * Camera reachability wins over the domain's own on/off flag: a camera that's offline can't be
 * detecting anything regardless of what the global schedule says (front/design-handoff/README.md
 * §3 탐지 이벤트: "탐지 중" approved / "꺼짐" closed / camera offline 시 "중단됨" pending).
 */
function statusPill(
  setting: DetectionDomainSetting,
  online: boolean,
  missingBedZone: boolean,
): { label: string; className: string } {
  if (!online) return { label: '중단됨', className: statusBadgeClassName('pending') };
  if (!setting.on) return { label: '꺼짐', className: statusBadgeClassName('closed') };
  if (missingBedZone) return { label: '침대 영역 미설정', className: statusBadgeClassName('rejected') };
  return { label: '탐지 중', className: statusBadgeClassName('approved') };
}

/**
 * GET /detection-settings is a single global config shared by every camera (not per-camera), so
 * editing the schedule itself stays Settings-page territory — the gear icon here navigates there
 * rather than duplicating that editor.
 */
export function DetectionSettingsCard({
  camera,
  onOverlayModeChange,
  onRecognizeBedZone,
}: DetectionSettingsCardProps): JSX.Element {
  const { status, data, retry } = useDetectionSettingsResource(true);
  const online = camera.status === 'online';

  return (
    <article className="rounded-card border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">탐지 이벤트</h2>
        <button
          type="button"
          className="icon-button"
          aria-label="탐지 설정으로 이동"
          onClick={() => navigateToPage('settings')}
        >
          <GearIcon />
        </button>
      </div>

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
        <div className="grid grid-cols-[1fr_auto_auto] items-center gap-x-3 gap-y-2 text-sm">
          {DOMAIN_ORDER.map((domain: DetectionDomainKey) => {
            const setting = data.domains[domain];
            const missingBedZone = domain === 'bed_exit' && camera.bed_zone == null;
            const pill = statusPill(setting, online, missingBedZone);
            return (
              <div className="contents" key={domain}>
                <span className="text-foreground">{DOMAIN_LABELS[domain]}</span>
                <span className="text-right tabular-nums text-muted-foreground">{formatDomainSchedule(setting)}</span>
                <span className={pill.className}>{pill.label}</span>
              </div>
            );
          })}
        </div>
      ) : null}

      {status === 'success' && data?.domains.bed_exit.on && camera.bed_zone == null ? (
        <div className="mt-3 rounded-control border border-destructive/30 bg-destructive/5 p-3 text-sm">
          <p className="text-foreground">침대 이탈 탐지를 시작하려면 실제 카메라 영상에서 침대 영역을 인식해야 합니다.</p>
          <button
            type="button"
            className="dialog-secondary-action mt-2"
            onClick={onRecognizeBedZone ?? (() => navigateToPage('settings'))}
          >
            침대 영역 인식
          </button>
        </div>
      ) : null}

      <div className="mt-3">
        <OverlayModeControl cameraId={camera.id} onModeChange={onOverlayModeChange} />
        <p className="mt-2 text-xs text-muted-foreground">
          침대 이탈은 저장된 인식 영역과 사람을, 낙상은 사람을 표시합니다. 없음은 오버레이 없이 깨끗한 영상을 표시합니다.
        </p>
      </div>
    </article>
  );
}

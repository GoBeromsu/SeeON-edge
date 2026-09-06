import { useDetectionSettingsResource } from '@/shared/api/usePollingResource';
import { DOMAIN_LABELS, DOMAIN_ORDER, formatDomainSchedule } from '@/features/settings/detectionSettingsForm';
import { navigateToPage } from '@/features/operations/crossPageNavigation';
import { OverlaySelectionControl } from '@/features/operations/OverlayModeControl';
import type { Camera, DetectionDomainKey, DetectionDomainSetting, OverlaySelection } from '@/shared/api/client';

type DetectionSettingsCardProps = {
  camera: Camera;
  onOverlaySelectionChange?: (selection: OverlaySelection | null) => void;
  onEditBedZones: () => void;
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
type DetectionStatusIcon = 'active' | 'off' | 'paused' | 'bed-missing';

function detectionStatus(
  setting: DetectionDomainSetting,
  online: boolean,
  missingBedZone: boolean,
): { icon: DetectionStatusIcon; label: string; className: string } {
  if (!online) return { icon: 'paused', label: '중단됨', className: 'text-status-pendingFg' };
  if (!setting.on) return { icon: 'off', label: '꺼짐', className: 'text-status-closedFg' };
  if (missingBedZone) return { icon: 'bed-missing', label: '침대 영역 미설정', className: 'text-status-rejectedFg' };
  return { icon: 'active', label: '탐지 중', className: 'text-status-approvedFg' };
}

function DetectionStatus({ icon, label, className }: ReturnType<typeof detectionStatus>): JSX.Element {
  return (
    <svg
      role="img"
      aria-label={label}
      className={`h-5 w-5 ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <title>{label}</title>
      {icon === 'active' ? (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="m8 12 2.5 2.5L16 9" />
        </>
      ) : null}
      {icon === 'off' ? (
        <>
          <path d="M12 2v10" />
          <path d="M6.3 5.7a9 9 0 1 0 11.4 0" />
        </>
      ) : null}
      {icon === 'paused' ? (
        <>
          <path d="M7 15a4 4 0 0 1 .6-7.95A6 6 0 0 1 19 9a3.5 3.5 0 0 1-.5 6.96" />
          <path d="M9.5 13v6M14.5 13v6" />
        </>
      ) : null}
      {icon === 'bed-missing' ? (
        <>
          <path d="M3 6v12M3 14h18v4M7 10h4a3 3 0 0 1 3 3v1M7 10V7h4a3 3 0 0 1 3 3" />
          <path d="m4 4 16 16" />
        </>
      ) : null}
    </svg>
  );
}

function BedEditIcon(): JSX.Element {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <path d="M3 6v12M3 14h18v4M7 10h4a3 3 0 0 1 3 3v1M7 10V7h4a3 3 0 0 1 3 3" />
      <path d="m17 5 1 2 2 1-2 1-1 2-1-2-2-1 2-1Z" />
    </svg>
  );
}

/**
 * GET /detection-settings is a single global config shared by every camera (not per-camera), so
 * editing the schedule itself stays Settings-page territory — the gear icon here navigates there
 * rather than duplicating that editor.
 */
export function DetectionSettingsCard({
  camera,
  onOverlaySelectionChange,
  onEditBedZones,
}: DetectionSettingsCardProps): JSX.Element {
  const { status, data, retry } = useDetectionSettingsResource(true);
  const online = camera.status === 'online';

  return (
    <article className="rounded-card border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">탐지 이벤트</h2>
        <div className="flex items-center gap-1">
          <button
            type="button"
            className="icon-button"
            aria-label="침대 영역 편집"
            title="침대 영역 편집"
            onClick={onEditBedZones}
          >
            <BedEditIcon />
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="탐지 설정으로 이동"
            onClick={() => navigateToPage('settings')}
          >
            <GearIcon />
          </button>
        </div>
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
            const statusIcon = detectionStatus(setting, online, missingBedZone);
            return (
              <div className="contents" key={domain}>
                <span className="text-foreground">{DOMAIN_LABELS[domain]}</span>
                <span className="text-right tabular-nums text-muted-foreground">{formatDomainSchedule(setting)}</span>
                <DetectionStatus {...statusIcon} />
              </div>
            );
          })}
        </div>
      ) : null}

      <div className="mt-3">
        <OverlaySelectionControl cameraId={camera.id} onSelectionChange={onOverlaySelectionChange} />
      </div>
    </article>
  );
}
